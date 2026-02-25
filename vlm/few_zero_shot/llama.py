import os
import random
from collections import Counter
from datetime import datetime
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import logging

# -----------------------------------------------------------------------------
# CUDA / performance settings
# -----------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

if torch.cuda.is_available():
    # Allow TF32 on Ampere+ GPUs for speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
LOG_DIR = "./VideoLLaMA3_HUMAN"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(
    LOG_DIR, f"videollama3_human_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logger = logging.getLogger("videollama3_human_eval")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_FILE)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def log(msg: str):
    logger.info(msg)



DATA_ROOT = "/Images Dataset"
MODEL_PATH_LLAMA3 = "/llms/VideoLLaMA3-7B/" 
LABEL_TO_SUBFOLDER = {
    "A4C": "A4C",
    "PSMV": "PSMV",
    "PSAV": "PSAV",
    "PLAX": "PLAX",
    "Random": "Random",
}
LABELS = list(LABEL_TO_SUBFOLDER.keys())

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

LABEL_DESCRIPTIONS = {
    "A4C": "apical four-chamber view",
    "PSMV": "parasternal short-axis view of the mitral valve",
    "PSAV": "parasternal short-axis view of the aortic valve",
    "PLAX": "parasternal long-axis view",
    "Random": "random non-cardiac image",
}


CHOICES = ["A", "B", "C", "D", "E"]
CHOICE_TO_LABEL = {
    "A": "A4C",
    "B": "PSMV",
    "C": "PSAV",
    "D": "PLAX",
    "E": "Random",
}
LABEL_TO_CHOICE = {v: k for k, v in CHOICE_TO_LABEL.items()}

# Few-shot ratios (fractions of TEST set used as support pool)
FEW_SHOT_SAMPLE_RATIOS = [0.0, 0.01, 0.03, 0.05, 0.07]

RANDOM_SEED = 42
DEBUG_NUM_IMAGES = 10

# Dataset split 70/30 (train/test)
TEST_RATIO = 0.30

# 1-shot in-context: max 1 support image per prompt (fast + safe)
MAX_SUPPORT_EXAMPLES_PER_PROMPT = 1

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
def collect_image_label_pairs():
    pairs = []
    for label, subfolder in LABEL_TO_SUBFOLDER.items():
        folder = os.path.join(DATA_ROOT, subfolder)
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                pairs.append((os.path.join(folder, f), label))
    return pairs


def train_test_split(pairs, test_ratio=0.3, seed=42):
    """
    Split full dataset into train and test:
    - test_ratio fraction for test
    - remaining for train
    """
    if not pairs:
        return [], []
    random.seed(seed)
    indices = list(range(len(pairs)))
    random.shuffle(indices)

    n_test = max(1, int(test_ratio * len(pairs)))
    if len(pairs) > 1:
        n_test = min(n_test, len(pairs) - 1)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    train_pairs = [pairs[i] for i in train_indices]
    test_pairs = [pairs[i] for i in test_indices]
    return train_pairs, test_pairs


def create_few_shot_split(test_pairs, sample_ratio, seed=42):
    """
    From the TEST set only, create:
    - support_pairs: sample_ratio * len(test_pairs)
    - eval_pairs   : remaining test samples
    """
    if not test_pairs:
        return [], []

    random.seed(seed)
    indices = list(range(len(test_pairs)))
    random.shuffle(indices)

    n_support = max(1, int(sample_ratio * len(test_pairs)))
    if len(test_pairs) > 1:
        n_support = min(n_support, len(test_pairs) - 1)

    support_indices = indices[:n_support]
    eval_indices = indices[n_support:]

    support_pairs = [test_pairs[i] for i in support_indices]
    eval_pairs = [test_pairs[i] for i in eval_indices]

    return support_pairs, eval_pairs


# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
def load_model_and_processor():
    log(f"Loading VideoLLaMA3 from {MODEL_PATH_LLAMA3}")
    if DEVICE.startswith("cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH_LLAMA3,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            device_map={"": DEVICE},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH_LLAMA3,
            trust_remote_code=True,
            torch_dtype=DTYPE,
        )
        model.to(DEVICE)

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False  # less memory for long prompts

    processor = AutoProcessor.from_pretrained(MODEL_PATH_LLAMA3, trust_remote_code=True)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"

    log("Model and processor loaded successfully.")
    return model, processor, tokenizer


# -----------------------------------------------------------------------------
# PROMPT / PREDICTION
# -----------------------------------------------------------------------------
def build_prompt_and_images(
    support_pairs,
    query_img_path: str,
    max_support: int,
    seed: int,
):
    """
    Build text prompt + list of PIL images (support + query).
    max_support controls how many support examples we include.
    With MAX_SUPPORT_EXAMPLES_PER_PROMPT = 1 this is 1-shot.
    """
    random.seed(seed)

    # Compact mapping text for 5-way classification
    mapping_text = (
        "Label mapping:\n"
        "A: A4C (apical four-chamber)\n"
        "B: Fish (non-cardiac fish-like image)\n"
        "C: Mercedes (non-cardiac Mercedes-like pattern)\n"
        "D: PLAX (parasternal long-axis)\n"
        "E: Random (other random image)\n"
    )
    prompt_lines = [mapping_text]
    images = []

    # ------- SUPPORT EXAMPLES (CAPPED, SHORT TEXT) -------
    if support_pairs and max_support > 0:
        k = min(max_support, len(support_pairs))
        active_support_pairs = random.sample(support_pairs, k)

        prompt_lines.append("Here is a labeled example:\n")
        for ex_idx, (ex_img_path, ex_label) in enumerate(active_support_pairs, 1):
            ex_img = Image.open(ex_img_path).convert("RGB")
            images.append(ex_img)
            ch = LABEL_TO_CHOICE[ex_label]

            prompt_lines.append(
                f"Example {ex_idx}:\n"
                f"USER: <image>\n"
                f"ASSISTANT: {ch}\n"
            )

    # ------- QUERY EXAMPLE -------
    query_img = Image.open(query_img_path).convert("RGB")
    images.append(query_img)

    prompt_lines.append(
        "Now classify a new image.\n"
        "USER: <image>\n"
        "Answer with exactly one letter (A, B, C, D, or E).\n"
        "ASSISTANT:"
    )

    prompt = "\n".join(prompt_lines)
    return prompt, images


def predict_label_for_image(
    model,
    processor,
    tokenizer,
    image_path: str,
    seed: int,
    support_pairs=None,
):
    """
    1-shot (or 0-shot if no support_pairs) prediction.
    One forward pass per sample for speed.
    """
    # zero-shot if no support
    max_k = 0 if not support_pairs else min(
        MAX_SUPPORT_EXAMPLES_PER_PROMPT, len(support_pairs)
    )

    prompt, images = build_prompt_and_images(
        support_pairs=support_pairs,
        query_img_path=image_path,
        max_support=max_k,
        seed=seed,
    )

    inputs = processor(
        text=prompt,
        images=images,
        return_tensors="pt",
    )

    if hasattr(inputs, "to"):
        inputs = inputs.to(DEVICE)
    else:
        inputs = {
            k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)

    with torch.inference_mode():
        outputs = model(**inputs)
        # logits for the first token after 'ASSISTANT:'
        last_logits = outputs.logits[0, -1, :]

    log_probs = F.log_softmax(last_logits, dim=-1)

    choice_scores = {}
    for ch in CHOICES:
        token_ids = tokenizer(ch, add_special_tokens=False).input_ids
        score = sum(log_probs[i] for i in token_ids).item()
        choice_scores[ch] = score

    best_choice = max(choice_scores, key=choice_scores.get)
    pred_label = CHOICE_TO_LABEL[best_choice]

    return pred_label, choice_scores, max_k  # max_k = # support examples actually used


# -----------------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------------
def evaluate(model, processor, tokenizer, eval_pairs, support_pairs, sample_ratio: float):
    log("\n===== VideoLLaMA3 HUMAN | FEW-SHOT IN-CONTEXT (1-shot max) =====")
    log(f"Test-set few-shot ratio      : {sample_ratio:.2f}")
    log(f"Support pool size (from test): {len(support_pairs)}")
    log(f"Max support per prompt       : {MAX_SUPPORT_EXAMPLES_PER_PROMPT}")
    log(f"Evaluation (held-out) size   : {len(eval_pairs)}")

    if not eval_pairs:
        log("No evaluation pairs for this ratio; skipping.")
        return

    correct = 0
    predictions = []
    used_support_counts = []

    for idx, (img_path, true_label) in enumerate(eval_pairs, 1):
        pred_label, scores, used_support = predict_label_for_image(
            model,
            processor,
            tokenizer,
            img_path,
            seed=RANDOM_SEED + idx,
            support_pairs=support_pairs,
        )

        predictions.append((img_path, true_label, pred_label))
        used_support_counts.append(used_support)
        if pred_label == true_label:
            correct += 1

        log(f"[{idx}/{len(eval_pairs)}] {img_path}")
        log(f" GT   : {true_label}")
        log(f" Pred : {pred_label} (used {used_support} support examples)")
        if idx <= DEBUG_NUM_IMAGES:
            log(
                f" Scores ? A:{scores['A']:.4f}  "
                f"B:{scores['B']:.4f}  "
                f"C:{scores['C']:.4f}  "
                f"D:{scores['D']:.4f}  "
                f"E:{scores['E']:.4f}"
            )

    acc = correct / len(eval_pairs) if eval_pairs else 0.0
    avg_support_used = sum(used_support_counts) / len(used_support_counts)

    log("\n" + "-" * 60)
    log(
        f"FEW-SHOT ACCURACY (ratio={sample_ratio:.2f}, "
        f"support_pool={len(support_pairs)}, eval={len(eval_pairs)}): "
        f"{acc:.4f} ({correct}/{len(eval_pairs)})"
    )
    log(f"Average support examples actually used per query: {avg_support_used:.2f}")
    log("-" * 60)

    gt_c = Counter([gt for _, gt, _ in predictions])
    pred_c = Counter([pred for _, _, pred in predictions])
    log("Ground-truth: " + " | ".join(f"{lbl}:{gt_c.get(lbl, 0)}" for lbl in LABELS))
    log("Predictions : " + " | ".join(f"{lbl}:{pred_c.get(lbl, 0)}" for lbl in LABELS))


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    log(f"Log file: {LOG_FILE}")
    all_pairs = collect_image_label_pairs()
    log(f"Total images in HUMAN dataset: {len(all_pairs)}")
    if not all_pairs:
        log("No images found!")
        return

    # 70/30 train/test split
    train_pairs, test_pairs = train_test_split(
        all_pairs, test_ratio=TEST_RATIO, seed=RANDOM_SEED
    )
    log(f"Train size: {len(train_pairs)} (70%)")
    log(f"Test size : {len(test_pairs)} (30%)")

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    model, processor, tokenizer = load_model_and_processor()

    # On TEST set only: few-shot support pool + held-out evaluation
    for ratio in FEW_SHOT_SAMPLE_RATIOS:
        log("\n" + "=" * 70)
        log(f"Preparing few-shot split on HUMAN TEST for ratio={ratio:.2f}")

        support_pairs, eval_pairs = create_few_shot_split(
            test_pairs, sample_ratio=ratio, seed=RANDOM_SEED
        )

        log(
            f"Few-shot ratio={ratio:.2f} on test: "
            f"support_pool={len(support_pairs)}, eval={len(eval_pairs)}"
        )

        if len(eval_pairs) == 0:
            log("No evaluation samples left after splitting; skipping this ratio.")
            continue

        evaluate(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            eval_pairs=eval_pairs,
            support_pairs=support_pairs,
            sample_ratio=ratio,
        )


if __name__ == "__main__":
    main()