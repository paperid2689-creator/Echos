from __future__ import annotations
import json
import math
import time
from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional as F
from dataloader import CLASS_NAMES, Sample


def _batch_score_label_sequences(
    model,
    processor,
    images,
    prompt: str,
    labels: Sequence[str],
) -> Tuple[List[List[float]], List[List[float]]]:
    user_messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for image in images
    ]
    prompt_texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in user_messages
    ]
    prompt_inputs = processor(text=prompt_texts, images=images, return_tensors="pt", padding=True)
    prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()

    batch_scores: List[List[float]] = [[] for _ in images]
    batch_nlls: List[List[float]] = [[] for _ in images]

    for label in labels:
        full_texts = [
            processor.apply_chat_template(
                messages + [{"role": "assistant", "content": [{"type": "text", "text": label}]}],
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in user_messages
        ]
        full_inputs = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
        full_inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in full_inputs.items()}

        with torch.inference_mode():
            outputs = model(**full_inputs)
            logits = outputs.logits
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

        input_ids = full_inputs["input_ids"]
        attention = full_inputs["attention_mask"]
        seq_lens = attention.sum(dim=1).tolist()

        for idx in range(len(images)):
            prompt_len = int(prompt_lens[idx])
            seq_len = int(seq_lens[idx])
            label_len = max(0, seq_len - prompt_len)
            if label_len == 0:
                batch_scores[idx].append(float("-inf"))
                batch_nlls[idx].append(float("inf"))
                continue

            start = prompt_len - 1
            end = start + label_len
            label_ids = input_ids[idx, prompt_len:seq_len]
            token_log_probs = log_probs[idx, start:end].gather(1, label_ids.unsqueeze(1)).squeeze(1)
            seq_log_prob = float(token_log_probs.sum().item())
            avg_nll = float((-token_log_probs.mean()).item())
            batch_scores[idx].append(seq_log_prob)
            batch_nlls[idx].append(avg_nll)

    return batch_scores, batch_nlls


def evaluate_classification(
    model,
    processor,
    test_samples: Sequence[Sample],
    prompt: str,
    max_new_tokens: int,
    image_size: int,
    eval_batch_size: int = 1,
) -> Dict[str, float]:
    model.eval()
    y_true: List[str] = []
    y_pred: List[str] = []
    ce_losses: List[float] = []
    total = len(test_samples)
    start_time = time.time()
    _ = max_new_tokens

    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be >= 1")

    _ = max_new_tokens

    print(f"eval start: {total} samples, batch size {eval_batch_size}", flush=True)

    for batch_start in range(0, total, eval_batch_size):
        batch_samples = test_samples[batch_start : batch_start + eval_batch_size]
        images = [Image.open(sample.image_path).convert("RGB") for sample in batch_samples]
        if image_size > 0:
            images = [image.resize((image_size, image_size), Image.BICUBIC) for image in images]

        batch_scores, batch_nlls = _batch_score_label_sequences(model, processor, images, prompt, CLASS_NAMES)
        for idx, sample in enumerate(batch_samples):
            scores = batch_scores[idx]
            nlls = batch_nlls[idx]
            pred_index = int(np.argmax(scores))
            pred = CLASS_NAMES[pred_index]
            true_index = CLASS_NAMES.index(sample.label)
            ce_losses.append(nlls[true_index])
            y_true.append(sample.label)
            y_pred.append(pred)

        done = min(batch_start + len(batch_samples), total)
        if done == total or done % 10 == 0:
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"eval progress: {done}/{total} ({rate:.2f} samples/s)", flush=True)
        if done == 100:
            interim = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, labels=list(CLASS_NAMES), average="macro", zero_division=0)),
                "cross_entropy": float(np.mean(ce_losses)) if ce_losses else math.nan,
                "samples": done,
            }
            print(json.dumps(interim), flush=True)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CLASS_NAMES), average="macro", zero_division=0)),
        "cross_entropy": float(np.mean(ce_losses)) if ce_losses else math.nan,
    }
