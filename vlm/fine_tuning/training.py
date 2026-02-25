from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Dict, Sequence
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments, TrainerCallback

import dataloader
from testing import evaluate_classification


def apply_lora(model, lora_r: int, lora_alpha: int, lora_dropout: float, use_gradient_checkpointing: bool):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def build_training_args(
    run_dir: Path,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    use_gradient_checkpointing: bool,
):
    return TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=use_gradient_checkpointing,
        ddp_find_unused_parameters=True,
    )


class BestCheckpointCallback(TrainerCallback):
    def __init__(
        self,
        trainer: Trainer,
        processor,
        test_samples: Sequence[dataloader.Sample],
        prompt: str,
        max_new_tokens: int,
        image_size: int,
        run_dir: Path,
        split_info: Dict[str, float],
        eval_max_samples: int,
        eval_batch_size: int,
        seed: int,
    ):
        self.trainer = trainer
        self.processor = processor
        self.test_samples = test_samples
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.image_size = image_size
        self.best_accuracy = -1.0
        self.run_dir = run_dir
        self.split_info = split_info
        self.eval_batch_size = eval_batch_size
        self.eval_subset = None
        if eval_max_samples > 0 and len(test_samples) > eval_max_samples:
            rng = random.Random(seed)
            self.eval_subset = rng.sample(list(test_samples), eval_max_samples)

    def evaluate_and_save(self, samples: Sequence[dataloader.Sample] | None = None) -> Dict[str, float]:
        eval_samples = samples if samples is not None else self.test_samples
        metrics = evaluate_classification(
            model=self.trainer.model,
            processor=self.processor,
            test_samples=eval_samples,
            prompt=self.prompt,
            max_new_tokens=self.max_new_tokens,
            image_size=self.image_size,
            eval_batch_size=self.eval_batch_size,
        )
        accuracy = metrics.get("accuracy", -1.0)
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            best_dir = self.run_dir / "best_adapter"
            best_dir.mkdir(parents=True, exist_ok=True)
            self.trainer.model.save_pretrained(best_dir)
            self.processor.save_pretrained(best_dir)
        return metrics

    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.trainer.is_world_process_zero():
            return control

        eval_samples = self.eval_subset if self.eval_subset is not None else self.test_samples
        metrics = self.evaluate_and_save(samples=eval_samples)
        metrics.update(self.split_info)
        metrics["epoch"] = int(state.epoch) if state.epoch is not None else None
        with (self.run_dir / "metrics_epoch.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        return control


def run_training(
    model,
    processor,
    train_dataset,
    test_samples: Sequence[dataloader.Sample],
    prompt: str,
    run_dir: Path,
    split_info: Dict[str, float],
    use_gradient_checkpointing: bool,
    args,
) -> None:
    model = apply_lora(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    training_args = build_training_args(
        run_dir=run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=dataloader.DataCollator(),
    )

    callback = BestCheckpointCallback(
        trainer=trainer,
        processor=processor,
        test_samples=test_samples,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        image_size=args.image_size,
        run_dir=run_dir,
        split_info=split_info,
        eval_max_samples=args.eval_max_samples,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
    )
    trainer.add_callback(callback)

    trainer.train()

    if trainer.is_world_process_zero():
        final_samples = test_samples if args.eval_full_final else (callback.eval_subset or test_samples)
        final_metrics = callback.evaluate_and_save(samples=final_samples)
        final_metrics.update(split_info)
        (run_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
        print(json.dumps(final_metrics, indent=2))
        print(f"Saved run artifacts to: {run_dir}")
