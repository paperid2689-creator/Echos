from __future__ import annotations
import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoProcessor
import dataloader
import testing
import training

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    Qwen2_5_VLForConditionalGeneration = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen2.5-VL classification")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--fraction-per-class", type=float, default=1.0)
    parser.add_argument("--cache-samples", type=int, default=0)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--test-fraction", type=float, default=1.0)
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-full-final", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_ddp() -> bool:
    return bool(int(os.environ.get("WORLD_SIZE", "1")) > 1)


def load_model_and_processor(model_path: Path, dtype: torch.dtype):
    if Qwen2_5_VLForConditionalGeneration is None:
        raise ImportError("Qwen2_5_VLForConditionalGeneration is unavailable in this environment.")

    device_map = None
    if torch.cuda.is_available() and not is_ddp():
        device_map = "auto"

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
    )
    return model, processor


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    samples = dataloader.collect_samples(args.dataset_root, args.fraction_per_class)
    train_samples, test_samples = dataloader.split_train_test(samples, args.train_ratio, args.seed)
    train_samples, test_samples = dataloader.clamp_train_test(
        train_samples,
        test_samples,
        args.max_train_samples,
        args.test_fraction,
        args.seed,
    )

    run_name = f"qwen25_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    split_info = {
        "total": len(samples),
        "train": len(train_samples),
        "test": len(test_samples),
        "train_ratio": args.train_ratio,
        "fraction_per_class": args.fraction_per_class,
        "max_train_samples": args.max_train_samples,
        "test_fraction": args.test_fraction,
    }
    (run_dir / "split.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model, processor = load_model_and_processor(args.model_path, dtype)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, str(args.adapter_path))
    model.config.use_cache = False

    use_gradient_checkpointing = not is_ddp()
    tokenizer_fn = dataloader.build_tokenizer_fn(processor, dataloader.PROMPT, args.image_size)
    train_dataset = dataloader.VLMDataset(train_samples, tokenizer_fn)

    dataloader.cache_samples(run_dir, train_samples, test_samples, tokenizer_fn, args.cache_samples)
    if args.cache_only:
        print(json.dumps({"cached": args.cache_samples}, indent=2))
        return
    if args.eval_only:
        _, cached_test = dataloader.load_cached_samples(run_dir)
        eval_samples = cached_test if cached_test else test_samples
        if args.eval_max_samples > 0 and len(eval_samples) > args.eval_max_samples:
            rng = random.Random(args.seed)
            eval_samples = rng.sample(list(eval_samples), args.eval_max_samples)
        metrics = testing.evaluate_classification(
            model=model,
            processor=processor,
            test_samples=eval_samples,
            prompt=dataloader.PROMPT,
            max_new_tokens=args.max_new_tokens,
            image_size=args.image_size,
            eval_batch_size=args.eval_batch_size,
        )
        metrics.update(split_info)
        metrics["cached_eval"] = len(eval_samples)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return

    training.run_training(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        test_samples=test_samples,
        prompt=dataloader.PROMPT,
        run_dir=run_dir,
        split_info=split_info,
        use_gradient_checkpointing=use_gradient_checkpointing,
        args=args,
    )


if __name__ == "__main__":
    main()
