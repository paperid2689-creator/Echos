from __future__ import annotations
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES: Tuple[str, ...] = ("A4C", "SC", "PL", "PSAV", "PSMV", "Random")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PROMPT = (
    "The image folder contains the dataset collected by scanning the CAE Blue Phantom "
    "using the GE Healthcare Vivid-Q US machine and the GE M4S Matrix Probe. "
    "The dataset is intended for training, validating, testing and fine-tuning the "
    "proposed AI framework designed to classify and grade ultrasound cardiac images. "
    "The folder is organized as follows: Images Dataset contains the images used for "
    "training, validation, and testing the AI framework. It is organized into six subfolders: "
    "five representing different cardiac views and one containing random images. "
    "Each image is named starting with a number, which indicates the grade of the image. "
    "The included cardiac views are: Apical Four Chamber (A4C), Subcostal Four Chamber (SC), "
    "Parasternal Long Axis (PL), Parasternal Short Axis - Aortic Valve (PSAV), "
    "Parasternal Short Axis - Mitral Valve (PSMV). "
    "Respond with exactly one label token only: A4C, SC, PL, PSAV, PSMV, Random."
)


@dataclass
class Sample:
    image_path: Path
    label: str


def collect_samples(dataset_root: Path, fraction_per_class: float) -> List[Sample]:
    if not (0 < fraction_per_class <= 1.0):
        raise ValueError("--fraction-per-class must be in (0, 1].")

    samples: List[Sample] = []
    for class_name in CLASS_NAMES:
        class_dir = dataset_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        images = [
            path
            for path in sorted(class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if fraction_per_class < 1.0 and images:
            keep = max(1, int(len(images) * fraction_per_class))
            images = images[:keep]

        samples.extend(Sample(image_path=path, label=class_name) for path in images)

    if not samples:
        raise RuntimeError("No samples found.")

    return samples


def split_train_test(samples: Sequence[Sample], train_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0, 1).")

    rng = random.Random(seed)
    train_samples: List[Sample] = []
    test_samples: List[Sample] = []

    by_class: Dict[str, List[Sample]] = {class_name: [] for class_name in CLASS_NAMES}
    for sample in samples:
        by_class[sample.label].append(sample)

    for class_name, class_samples in by_class.items():
        rng.shuffle(class_samples)
        split_idx = int(len(class_samples) * train_ratio)
        split_idx = min(max(split_idx, 1), len(class_samples) - 1) if len(class_samples) > 1 else len(class_samples)
        train_samples.extend(class_samples[:split_idx])
        test_samples.extend(class_samples[split_idx:])

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)
    return train_samples, test_samples


def clamp_train_test(
    train_samples: List[Sample],
    test_samples: List[Sample],
    max_train_samples: int,
    test_fraction: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample]]:
    rng = random.Random(seed)

    if max_train_samples > 0 and len(train_samples) > max_train_samples:
        rng.shuffle(train_samples)
        train_samples = train_samples[:max_train_samples]

    if not (0.0 < test_fraction <= 1.0):
        raise ValueError("--test-fraction must be in (0, 1].")
    if test_fraction < 1.0 and test_samples:
        keep = max(1, int(len(test_samples) * test_fraction))
        rng.shuffle(test_samples)
        test_samples = test_samples[:keep]

    return train_samples, test_samples


def cache_samples(
    run_dir: Path,
    train_samples: Sequence[Sample],
    test_samples: Sequence[Sample],
    tokenizer_fn,
    cache_count: int,
) -> None:
    if cache_count <= 0:
        return

    cache_train = list(train_samples[:cache_count])
    cache_test = list(test_samples[:cache_count])
    cache_payload = {
        "train": [{"image_path": str(s.image_path), "label": s.label} for s in cache_train],
        "test": [{"image_path": str(s.image_path), "label": s.label} for s in cache_test],
    }
    (run_dir / "cache_samples.json").write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")

    for sample in cache_train + cache_test:
        _ = tokenizer_fn(sample)


def load_cached_samples(run_dir: Path) -> Tuple[List[Sample], List[Sample]]:
    cache_path = run_dir / "cache_samples.json"
    if not cache_path.exists():
        return [], []

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    cached_train = [Sample(image_path=Path(item["image_path"]), label=item["label"]) for item in payload.get("train", [])]
    cached_test = [Sample(image_path=Path(item["image_path"]), label=item["label"]) for item in payload.get("test", [])]
    return cached_train, cached_test


def build_tokenizer_fn(processor, prompt: str, image_size: int):
    def tokenize(sample: Sample):
        image = Image.open(sample.image_path).convert("RGB")
        if image_size > 0:
            image = image.resize((image_size, image_size), Image.BICUBIC)
        answer = sample.label

        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        full_messages = user_messages + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]

        prompt_text = processor.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
        full_inputs = processor(text=[full_text], images=[image], return_tensors="pt")

        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100

        output = {
            "input_ids": full_inputs["input_ids"].squeeze(0),
            "attention_mask": full_inputs["attention_mask"].squeeze(0),
            "pixel_values": full_inputs["pixel_values"].squeeze(0),
            "labels": labels.squeeze(0),
            "label_text": sample.label,
            "image_path": str(sample.image_path),
        }
        for key, value in full_inputs.items():
            if key in output or not isinstance(value, torch.Tensor):
                continue
            output[key] = value.squeeze(0)
        return output

    return tokenize


class VLMDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], tokenizer_fn):
        self.samples = list(samples)
        self.tokenizer_fn = tokenizer_fn

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return self.tokenizer_fn(sample)


class DataCollator:
    @staticmethod
    def _pad_stack_tensors(tensors: List[torch.Tensor], pad_value: int | float = 0) -> torch.Tensor:
        rank = tensors[0].ndim
        max_shape = [max(t.shape[d] for t in tensors) for d in range(rank)]
        padded = []
        for tensor in tensors:
            canvas = torch.full(max_shape, pad_value, dtype=tensor.dtype)
            slices = tuple(slice(0, dim) for dim in tensor.shape)
            canvas[slices] = tensor
            padded.append(canvas)
        return torch.stack(padded, dim=0)

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in features]
        attention_masks = [item["attention_mask"] for item in features]
        labels = [item["labels"] for item in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
        }

        for key in features[0].keys():
            if key in {"input_ids", "attention_mask", "labels", "label_text", "image_path"}:
                continue

            values = [item[key] for item in features]
            if not all(isinstance(v, torch.Tensor) for v in values):
                batch[key] = values
                continue

            same_shape = all(v.shape == values[0].shape for v in values)
            if same_shape:
                batch[key] = torch.stack(values, dim=0)
            else:
                batch[key] = self._pad_stack_tensors(values, pad_value=0)
        return batch
