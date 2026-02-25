import os
import random
from collections import Counter
import csv
import torch
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)

# ============================
# CONFIG
# ============================

# Root folder that contains the 6 subfolders: A4C, SC, PL, PSAV, PSMV, Random
DATA_ROOT = "/Images Dataset"
MODEL_PATH = "/llms/Qwen2.5-VL-7B/"

# Where to save results
RESULT_DIR = "Res-17"

# Labels and their corresponding folders
LABEL_TO_SUBFOLDER = {
    "A4C": "A4C",
    "SC": "SC",
    "PL": "PL",
    "PSAV": "PSAV",
    "PSMV": "PSMV",
    "Random": "Random",
}
LABELS = list(LABEL_TO_SUBFOLDER.keys())

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

# Natural-language descriptions for each view (for prompt text)
LABEL_DESCRIPTIONS = {
    "A4C": "Apical Four Chamber (A4C)",
    "SC": "Subcostal Four Chamber (SC)",
    "PL": "Parasternal Long Axis (PL)",
    "PSAV": "Parasternal Short Axis - Aortic Valve (PSAV)",
    "PSMV": "Parasternal Short Axis - Mitral Valve (PSMV)",
    "Random": "a random or non-standard view (Random)",
}

# Few-shot: fractions of each class used as few-shot exemplars
# 0.00 = 0% (true zero-shot), 0.01 = 1%, etc.
SHOT_RATIOS = [0.005, 0.007]

# Fraction of remaining images to evaluate on (after removing few-shot examples)
SAMPLE_RATIO = 0.3
RANDOM_SEED = 42
DEBUG_SHOW_SCORES = True
DEBUG_NUM_IMAGES = 5
TARGET_IMAGE_SIZE = (224, 224)  # (width, height)

# ============================
# UTIL: LOAD & RESIZE IMAGE
# ============================

def load_image_224(path, size=TARGET_IMAGE_SIZE):
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    return img

# ============================
# DATA COLLECTION
# ============================

def collect_image_label_pairs():
    
    image_label_pairs = []

    for label, subfolder in LABEL_TO_SUBFOLDER.items():
        folder_path = os.path.join(DATA_ROOT, subfolder)

        if not os.path.isdir(folder_path):
            print(f"Warning: folder not found: {folder_path}")
            continue

        for fname in os.listdir(folder_path):
            if fname.lower().endswith(IMAGE_EXTENSIONS):
                img_path = os.path.join(folder_path, fname)
                image_label_pairs.append((img_path, label))

    return image_label_pairs


def select_label_examples_by_ratio(image_label_pairs, shot_ratio_per_class: float):

    label_to_paths = {lbl: [] for lbl in LABELS}
    for img_path, label in image_label_pairs:
        if label in label_to_paths:
            label_to_paths[label].append(img_path)

    label_to_examples = {lbl: [] for lbl in LABELS}
    used_paths = set()

    print(f"\nSelecting few-shot examples with ratio {shot_ratio_per_class:.4f} per class")

    for label in LABELS:
        paths = label_to_paths[label]
        n_c = len(paths)
        if n_c == 0:
            continue

        random.shuffle(paths)

        if shot_ratio_per_class <= 0.0:
            n_shots = 0
        else:
            n_shots = int(shot_ratio_per_class * n_c)
            if n_shots <= 0:
                n_shots = 1  # ensure at least 1 example if ratio > 0 and class non-empty

        chosen = paths[:n_shots]
        label_to_examples[label] = chosen
        used_paths.update(chosen)

    remaining_pairs = [(p, l) for (p, l) in image_label_pairs if p not in used_paths]

    print("Few-shot examples selected per class:")
    for lbl in LABELS:
        print(f"  {lbl}: {len(label_to_examples[lbl])}")

    return label_to_examples, remaining_pairs


# ============================
# MODEL LOADING
# ============================

def load_model_and_processors():
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",          # you can try torch.float16 to save more memory
        device_map="cuda:1",
        trust_remote_code=True,      # needed for many Qwen VL models
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    return model, processor, tokenizer


# ============================
# BUILD LABEL-SPECIFIC FEW-SHOT CONTEXT
# ============================

def build_label_context_inputs(
    model,
    processor,
    test_image_path: str,
    label: str,
    label_examples,
):
    """
    Build the chat context specific to a label:
      - few-shot: example images of that label
      - query:   test image asking "Is this [label]?"
    This context is used to score "Yes" vs "No" for that label.

    If label_examples is empty, this becomes a pure zero-shot label query.
    """
    messages = []

    # Few-shot exemplars for this label (may be empty if 0% few-shot)
    desc = LABEL_DESCRIPTIONS[label]
    for ex_img_path in label_examples:
        ex_img = load_image_224(ex_img_path)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ex_img},
                    {
                        "type": "text",
                        "text": (
                            f"This is an example cardiac ultrasound image. "
                            f"It shows the {desc} view."
                        ),
                    },
                ],
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Yes"},
                ],
            }
        )

    # Query for this label
    query_img = load_image_224(test_image_path)
    query_text = (
        f"Now look at this new cardiac ultrasound image. "
        f"Does it also show the {desc} view? "
        f"Answer with exactly one token: Yes or No."
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": query_img},
                {"type": "text", "text": query_text},
            ],
        }
    )

    # Build inputs stopping right before the assistant answer
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # expect assistant reply next
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)
    return inputs


# ============================
# SCORING "Yes" vs "No" ANSWERS
# ============================

def score_answer_for_context(model, tokenizer, base_inputs, answer_text: str) -> float:
    """
    Given a label-specific context (ending with the user query),
    append the candidate answer_text ("Yes" or "No") and compute
    the negative log-likelihood (NLL) of that answer.

    Only the appended answer tokens are supervised in the loss.
    """
    input_ids_ctx = base_inputs["input_ids"]   # [1, L_ctx]
    attn_mask_ctx = base_inputs.get(
        "attention_mask",
        torch.ones_like(input_ids_ctx),
    )

    # Tokenize the answer text alone, without special tokens
    with torch.no_grad():
        answer_ids = tokenizer(
            answer_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(model.device)          # [1, L_ans]

    ans_len = answer_ids.shape[-1]

    # Concatenate context + answer tokens
    input_ids_full = torch.cat([input_ids_ctx, answer_ids], dim=-1)
    attn_mask_full = torch.cat(
        [attn_mask_ctx, torch.ones_like(answer_ids)],
        dim=-1,
    )

    # Labels: supervise only the appended answer tokens
    labels = input_ids_full.clone()
    labels[:, :-ans_len] = -100

    # Build full input dict (preserve vision fields)
    inputs_full = {}
    for k, v in base_inputs.items():
        if k in ["input_ids", "attention_mask"]:
            continue
        inputs_full[k] = v
    inputs_full["input_ids"] = input_ids_full
    inputs_full["attention_mask"] = attn_mask_full

    with torch.no_grad():
        outputs = model(**inputs_full, labels=labels)
        loss = outputs.loss.item()

    score = -loss * max(ans_len, 1)  # higher = better
    return score


def score_label_for_image(
    model,
    processor,
    tokenizer,
    image_path: str,
    label: str,
    label_examples,
):
    """
    For a given label:
      - Build label-specific context with few-shot examples + query.
      - Score "Yes" and "No".
      - Return a single scalar score, e.g., s_yes - s_no.
    """
    base_inputs = build_label_context_inputs(
        model,
        processor,
        image_path,
        label,
        label_examples,
    )

    s_yes = score_answer_for_context(model, tokenizer, base_inputs, "Yes")
    s_no = score_answer_for_context(model, tokenizer, base_inputs, "No")

    # Label score: margin between Yes and No
    label_score = s_yes - s_no
    return label_score, s_yes, s_no


def predict_label_for_image(
    model,
    processor,
    tokenizer,
    image_path: str,
    label_to_examples,
    debug_scores: bool = False,
):
    """
    Evaluate all labels {A4C, SC, PL, PSAV, PSMV, Random} using binary
    few-shot scoring, and return the label with highest score.
    Works for both few-shot and zero-shot (when some example lists are empty).
    """
    best_label = None
    best_score = float("-inf")

    scores = {}
    yes_scores = {}
    no_scores = {}

    for label in LABELS:
        examples = label_to_examples.get(label, [])
        # We allow examples to be empty (for 0% shot ratio)
        s_label, s_yes, s_no = score_label_for_image(
            model,
            processor,
            tokenizer,
            image_path,
            label,
            examples,
        )
        scores[label] = s_label
        yes_scores[label] = s_yes
        no_scores[label] = s_no

        if s_label > best_score:
            best_score = s_label
            best_label = label

        # Clear some cache per-label to be extra safe on memory
        torch.cuda.empty_cache()

    if debug_scores:
        return best_label, scores, yes_scores, no_scores
    return best_label


# ============================
# EVALUATION (WITH SAVING)
# ============================

def evaluate_for_ratio(
    model,
    processor,
    tokenizer,
    label_to_examples,
    remaining_pairs,
    shot_ratio: float,
    sample_ratio: float,
    debug_for_this_ratio: bool,
):
    """
    Run few-shot evaluation for a given (shot_ratio, sample_ratio) pair and:
      - print metrics
      - save summary + predictions under RESULT_DIR
    """
    os.makedirs(RESULT_DIR, exist_ok=True)

    n_remaining = len(remaining_pairs)
    n_eval = max(1, int(sample_ratio * n_remaining))
    eval_samples = remaining_pairs[:n_eval]

    print(f"\n===== SHOT_RATIO = {shot_ratio:.3f}, SAMPLE_RATIO = {sample_ratio:.3f} =====")
    print(f"Evaluating on {n_eval} images (~{sample_ratio*100:.3f}% of {n_remaining})")

    correct = 0
    predictions = []  # list of dicts for CSV

    for idx, (img_path, true_label) in enumerate(eval_samples, start=1):
        debug = debug_for_this_ratio and idx <= DEBUG_NUM_IMAGES

        if debug:
            pred_label, scores, yes_scores, no_scores = predict_label_for_image(
                model,
                processor,
                tokenizer,
                img_path,
                label_to_examples=label_to_examples,
                debug_scores=True,
            )
        else:
            pred_label = predict_label_for_image(
                model,
                processor,
                tokenizer,
                img_path,
                label_to_examples=label_to_examples,
                debug_scores=False,
            )

        is_correct = int(pred_label == true_label)
        if is_correct:
            correct += 1

        print(f"[{idx}/{n_eval}] {img_path}")
        print(f"   GT   : {true_label}")
        print(f"   Pred : {pred_label}")

        if debug:
            print("   Label scores (Yes - No):")
            for lbl in LABELS:
                print(
                    f"      {lbl}: score={scores[lbl]:.4f}, "
                    f"Yes={yes_scores[lbl]:.4f}, No={no_scores[lbl]:.4f}"
                )

        # store per-sample prediction
        predictions.append(
            {
                "index": idx,
                "image_path": img_path,
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": is_correct,
            }
        )

    # Overall accuracy
    accuracy = correct / n_eval if n_eval > 0 else 0.0
    print("\n----------------------")
    print(
        f"Few-shot (binary per-label) accuracy "
        f"@ SHOT_RATIO={shot_ratio:.3f}, SAMPLE_RATIO={sample_ratio:.3f}: {accuracy:.4f}"
    )
    print(f"Correct: {correct} / {n_eval}")
    print("----------------------")

    # Per-class stats
    gt_counts = Counter([s["true_label"] for s in predictions])
    pred_counts = Counter([s["pred_label"] for s in predictions])

    per_class_correct = Counter()
    per_class_total = Counter()
    for s in predictions:
        gt = s["true_label"]
        pred = s["pred_label"]
        per_class_total[gt] += 1
        if gt == pred:
            per_class_correct[gt] += 1

    # ---- Save results to files in RESULT_DIR ----
    shot_str = f"{shot_ratio:.3f}".replace(".", "_")
    sample_str = f"{sample_ratio:.3f}".replace(".", "_")
    summary_path = os.path.join(RESULT_DIR, f"summary_shot_{shot_str}_sample_{sample_str}.txt")
    predictions_path = os.path.join(RESULT_DIR, f"predictions_shot_{shot_str}_sample_{sample_str}.csv")

    # Summary text file
    with open(summary_path, "w") as f:
        f.write(f"MODEL: {MODEL_PATH}\n")
        f.write(f"DATA_ROOT: {DATA_ROOT}\n")
        f.write(f"SHOT_RATIO_PER_CLASS: {shot_ratio:.4f}\n")
        f.write(f"SAMPLE_RATIO: {sample_ratio:.4f}\n")
        f.write(f"NUM_EVAL: {n_eval}\n")
        f.write(f"ACCURACY: {accuracy:.6f}\n")
        f.write(f"CORRECT: {correct}\n")
        f.write("\nGround-truth distribution in eval set:\n")
        for lbl in LABELS:
            f.write(f"  {lbl}: {gt_counts.get(lbl, 0)}\n")
        f.write("\nPrediction distribution:\n")
        for lbl in LABELS:
            f.write(f"  {lbl}: {pred_counts.get(lbl, 0)}\n")
        f.write("\nPer-class accuracy:\n")
        for lbl in LABELS:
            total = per_class_total.get(lbl, 0)
            if total == 0:
                acc_lbl = 0.0
            else:
                acc_lbl = per_class_correct.get(lbl, 0) / total
            f.write(
                f"  {lbl}: {acc_lbl:.6f} "
                f"({per_class_correct.get(lbl, 0)}/{total})\n"
            )

    # Predictions CSV
    with open(predictions_path, "w", newline="") as cf:
        writer = csv.DictWriter(
            cf,
            fieldnames=["index", "image_path", "true_label", "pred_label", "correct"],
        )
        writer.writeheader()
        for s in predictions:
            writer.writerow(s)

    print(f"Saved summary to: {summary_path}")
    print(f"Saved predictions to: {predictions_path}")


def main():
    # 1) Collect full dataset
    image_label_pairs = collect_image_label_pairs()
    n_total = len(image_label_pairs)
    print(f"Total images found: {n_total}")

    if n_total == 0:
        print("No images found. Check DATA_ROOT and folder names.")
        return

    random.seed(RANDOM_SEED)

    # 2) Load model, processor, tokenizer (once)
    model, processor, tokenizer = load_model_and_processors()

    # 3) For each shot ratio, select few-shot examples and evaluate
    for i, shot_ratio in enumerate(SHOT_RATIOS):
        # Select few-shot examples for this shot_ratio
        label_to_examples, remaining_pairs = select_label_examples_by_ratio(
            image_label_pairs,
            shot_ratio_per_class=shot_ratio,
        )

        n_remaining = len(remaining_pairs)
        print(f"Remaining images after few-shot selection (shot_ratio={shot_ratio:.3f}): {n_remaining}")

        if n_remaining == 0:
            print("No remaining images for evaluation; skipping this shot ratio.")
            continue

        # Shuffle remaining once for this ratio
        random.shuffle(remaining_pairs)

        debug_for_this_ratio = DEBUG_SHOW_SCORES and (i == 0)

        # Evaluate at fixed SAMPLE_RATIO
        evaluate_for_ratio(
            model,
            processor,
            tokenizer,
            label_to_examples,
            remaining_pairs,
            shot_ratio=shot_ratio,
            sample_ratio=SAMPLE_RATIO,
            debug_for_this_ratio=debug_for_this_ratio,
        )


if __name__ == "__main__":
    main()
