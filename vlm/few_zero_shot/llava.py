import os
import random
from collections import Counter
import csv
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor,
    AutoTokenizer,
)

# ============================
# CONFIG
# ============================

DATA_ROOT = "/Images Dataset/"
VIDEOLLAVA_MODEL_ID = "/llms/Video-LLaVA-7B-hf/"
OUTPUT_DIR = "LLAVA_Few"
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

# Natural language descriptions for prompt text
LABEL_DESCRIPTIONS = {
    "A4C": "Apical Four Chamber (A4C)",
    "SC": "Subcostal Four Chamber (SC)",
    "PL": "Parasternal Long Axis (PL)",
    "PSAV": "Parasternal Short Axis - Aortic Valve (PSAV)",
    "PSMV": "Parasternal Short Axis - Mitral Valve (PSMV)",
    "Random": "a random or non-standard view (Random)",
}

# Map labels -> single-letter answers for the model
ANSWER_TOKENS = {
    "A4C": "A",
    "SC": "B",
    "PL": "C",
    "PSAV": "D",
    "PSMV": "E",
    "Random": "F",
}
INV_ANSWER_TOKENS = {v: k for k, v in ANSWER_TOKENS.items()}

# Sweep of few-shot ratios per class (used for support selection + eval split)
SHOT_RATIOS = [0.0, 0.01, 0.03, 0.05, 0.07]

# Use only 30% of all images as working subset (support + eval)
DATA_USAGE_RATIO = 0.30

# Among the remaining images (after taking shots), use this fraction for eval
SAMPLE_RATIO = 0.50

# Number of images to use to estimate priors for A/B/C/D/E/F
CALIBRATION_NUM_IMAGES = 200

RANDOM_SEED = 42

DEBUG_SHOW_SCORES = True
DEBUG_NUM_IMAGES = 5

TARGET_IMAGE_SIZE = (224, 224)

# Max number of few-shot examples per label inserted into the prompt
MAX_EXAMPLES_PER_LABEL_IN_PROMPT = 4

# Simple in-memory cache for resized images
_image_cache = {}

# Accumulate log lines for saving global text log
LOG_LINES = []


# ============================
# LOGGING HELPER
# ============================

def log(msg=""):
    """Print to console and store in global LOG_LINES for later saving."""
    print(msg)
    LOG_LINES.append(str(msg))


# ============================
# UTIL: LOAD & RESIZE IMAGE
# ============================

def load_image_224(path, size=TARGET_IMAGE_SIZE):
    """Load image from disk once and cache it."""
    key = (path, size)
    if key in _image_cache:
        return _image_cache[key]
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    _image_cache[key] = img
    return img


# ============================
# DATA COLLECTION
# ============================

def collect_image_label_pairs():
    """
    Walk subfolders and collect (image_path, label) pairs.
    """
    image_label_pairs = []

    for label, subfolder in LABEL_TO_SUBFOLDER.items():
        folder_path = os.path.join(DATA_ROOT, subfolder)

        if not os.path.isdir(folder_path):
            log("Warning: folder not found: " + str(folder_path))
            continue

        for fname in os.listdir(folder_path):
            if fname.lower().endswith(IMAGE_EXTENSIONS):
                img_path = os.path.join(folder_path, fname)
                image_label_pairs.append((img_path, label))

    return image_label_pairs


def select_label_examples_by_ratio(image_label_pairs, shot_ratio_per_class):
    """
    From the given list of (img_path, label), select a few-shot subset per class.
    Returns:
      label_to_examples: dict[label] -> list[example_img_path]
      remaining_pairs:   list[(img_path, label)] not used as examples
    """
    label_to_paths = {lbl: [] for lbl in LABELS}
    for img_path, label in image_label_pairs:
        if label in label_to_paths:
            label_to_paths[label].append(img_path)

    label_to_examples = {lbl: [] for lbl in LABELS}
    used_paths = set()

    log("")
    log("Selecting few-shot examples with ratio %.4f per class (within working set)" % shot_ratio_per_class)

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
                n_shots = 1  # at least 1 if ratio>0

        chosen = paths[:n_shots]
        label_to_examples[label] = chosen
        used_paths.update(chosen)

    remaining_pairs = [(p, l) for (p, l) in image_label_pairs if p not in used_paths]

    log("Few-shot examples selected per class (within working set):")
    for lbl in LABELS:
        log("  %s: %d" % (lbl, len(label_to_examples[lbl])))

    return label_to_examples, remaining_pairs


def split_support_and_eval_from_working_set(working_pairs, shot_ratio_per_class, sample_ratio, seed=42):
    """
    On the 30% working subset:
      1) choose support examples (few-shot) per label
      2) from the remaining images, sample a subset for evaluation
    """
    label_to_examples, remaining_pairs = select_label_examples_by_ratio(
        working_pairs,
        shot_ratio_per_class=shot_ratio_per_class,
    )

    random.seed(seed)
    random.shuffle(remaining_pairs)

    n_remaining = len(remaining_pairs)
    n_eval = max(1, int(sample_ratio * n_remaining))
    eval_pairs = remaining_pairs[:n_eval]

    return label_to_examples, eval_pairs, n_remaining, n_eval


# ============================
# MODEL LOADING (LLAVA ONLY)
# ============================

def load_model_and_processors():
    """
    Load Video-LLaVA (via VideoLlavaForConditionalGeneration) + processor + tokenizer.
    """
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        VIDEOLLAVA_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    processor = VideoLlavaProcessor.from_pretrained(VIDEOLLAVA_MODEL_ID)

    # Ensure tokenizer exists
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        tokenizer = processor.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(VIDEOLLAVA_MODEL_ID, use_fast=True)

    model.eval()
    tokenizer.padding_side = "left"

    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer

    return model, processor, tokenizer


# ============================
# CONTEXT CONSTRUCTION: JOINT MULTI-CLASS + FEW-SHOT
# ============================

def build_joint_context_inputs(
    model,
    processor,
    image_path,
    label_to_examples=None,
    max_examples_per_label=MAX_EXAMPLES_PER_LABEL_IN_PROMPT,
):
    """
    Build a joint multi-class few-shot context:

      - If label_to_examples is provided:
          For each label L, include up to K example images with:
            USER: <image>
            "This is an example of [desc]. The correct label is 'X'."
            ASSISTANT: X
      - Then include the query image and ask for one letter A-F.

    For zero-shot calibration, call with label_to_examples=None.
    """
    images = []
    text_chunks = []

    # Instruction header
    text_chunks.append(
        "You are an expert in classifying cardiac ultrasound (echo) views."
    )

    # Few-shot exemplars (if provided)
    if label_to_examples is not None:
        text_chunks.append(
            "Here are some labeled example images for each possible view."
        )
        example_idx = 1
        for lbl in LABELS:
            examples = label_to_examples.get(lbl, [])
            if not examples:
                continue
            desc = LABEL_DESCRIPTIONS[lbl]
            token = ANSWER_TOKENS[lbl]

            for ex_path in examples[:max_examples_per_label]:
                ex_img = load_image_224(ex_path)
                images.append(ex_img)
                text_chunks.append(
                    f"\nExample {example_idx}:\n"
                    f"<image>\n"
                    f"This is a cardiac ultrasound image. It shows the {desc} view.\n"
                    f"The correct answer letter is '{token}'."
                )
                example_idx += 1

    # Now add the query image
    query_img = load_image_224(image_path)
    images.append(query_img)

    # Options string: (A) descA (B) descB ...
    option_pieces = []
    for lbl in LABELS:
        letter = ANSWER_TOKENS[lbl]
        desc = LABEL_DESCRIPTIONS[lbl]
        option_pieces.append(f"({letter}) {desc}")
    options_str = " ".join(option_pieces)

    text_chunks.append(
        "\nNow here is a new cardiac ultrasound image:\n"
        "<image>\n"
        f"The possible views are: {options_str}.\n"
        "Question: Which view does this image show?\n"
        "Answer with exactly one letter from {A,B,C,D,E,F}."
    )

    prompt = "\n".join(text_chunks)

    inputs = processor(
        text=prompt,
        images=images,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    return inputs


# ============================
# SCORING SINGLE-LETTER ANSWERS
# ============================

def score_answer_for_context(model, tokenizer, base_inputs, answer_text):
    """
    Append an answer token (e.g. 'A') to the text tokens and compute
    a score proportional to log p(answer | context).
    """
    input_ids_ctx = base_inputs["input_ids"]
    attn_mask_ctx = base_inputs.get(
        "attention_mask",
        torch.ones_like(input_ids_ctx),
    )

    with torch.no_grad():
        answer_ids = tokenizer(
            answer_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(model.device)

    ans_len = answer_ids.shape[-1]

    # concat context + answer
    input_ids_full = torch.cat([input_ids_ctx, answer_ids], dim=-1)
    attn_mask_full = torch.cat(
        [attn_mask_ctx, torch.ones_like(answer_ids)],
        dim=-1,
    )

    # supervise only the appended answer tokens
    labels = input_ids_full.clone()
    labels[:, :-ans_len] = -100

    # preserve vision-related fields
    inputs_full = {}
    for k, v in base_inputs.items():
        if k in ["input_ids", "attention_mask"]:
            continue
        inputs_full[k] = v
    inputs_full["input_ids"] = input_ids_full
    inputs_full["attention_mask"] = attn_mask_full
    inputs_full["labels"] = labels

    with torch.no_grad():
        outputs = model(**inputs_full)
        loss = outputs.loss.item()

    # Negative loss is proportional to log p(answer | context).
    # Multiply by ans_len to get (approx) sum of log-probs over tokens.
    score = -loss * max(ans_len, 1)
    return score


# ============================
# PRIOR ESTIMATION (CALIBRATION, ZERO-SHOT)
# ============================

def estimate_label_priors_joint(model, processor, tokenizer, image_label_pairs, max_calib=200):
    """
    Estimate the *average* score for each label's letter (A-F) over a calibration set,
    in ZERO-SHOT mode (no few-shot examples). We will subtract these priors at
    prediction time to cancel global biases.
    """
    if len(image_label_pairs) == 0:
        return {lbl: 0.0 for lbl in LABELS}

    pairs = list(image_label_pairs)
    random.seed(RANDOM_SEED)
    random.shuffle(pairs)
    calib_pairs = pairs[:min(max_calib, len(pairs))]

    sums = {lbl: 0.0 for lbl in LABELS}
    n_calib = len(calib_pairs)

    log("")
    log("===== Estimating label priors over %d calibration images (zero-shot) =====" % n_calib)

    for idx, (img_path, _) in enumerate(calib_pairs, start=1):
        # zero-shot: no label_to_examples
        base_inputs = build_joint_context_inputs(
            model,
            processor,
            img_path,
            label_to_examples=None,
            max_examples_per_label=0,
        )
        for lbl in LABELS:
            ans = ANSWER_TOKENS[lbl]
            s_lbl = score_answer_for_context(model, tokenizer, base_inputs, ans)
            sums[lbl] += s_lbl

        if idx % 10 == 0 or idx == n_calib:
            log("  Calib [%d/%d] %s" % (idx, n_calib, img_path))

    priors = {lbl: (sums[lbl] / max(n_calib, 1)) for lbl in LABELS}

    log("")
    log("Estimated average raw scores (priors) per label:")
    for lbl in LABELS:
        log("  %s (token %s): %.4f" % (lbl, ANSWER_TOKENS[lbl], priors[lbl]))

    return priors


# ============================
# PREDICTION WITH CALIBRATION + FEW-SHOT
# ============================

def predict_label_for_image_joint_calibrated(
    model,
    processor,
    tokenizer,
    image_path,
    label_to_examples,
    label_log_priors,
    debug_scores=False,
):
    """
    Joint multi-class prediction with calibration and few-shot:

      - Build ONE context with few-shot examples (label_to_examples) and the query.
      - For each label L, score its letter (A-F).
      - Subtract the pre-estimated average score (prior) for that label.
      - Return label with highest calibrated score.
    """
    base_inputs = build_joint_context_inputs(
        model,
        processor,
        image_path,
        label_to_examples=label_to_examples,
        max_examples_per_label=MAX_EXAMPLES_PER_LABEL_IN_PROMPT,
    )

    raw_scores = {}
    calib_scores = {}
    for lbl in LABELS:
        ans_token = ANSWER_TOKENS[lbl]
        raw = score_answer_for_context(model, tokenizer, base_inputs, ans_token)
        raw_scores[lbl] = raw
        prior = label_log_priors.get(lbl, 0.0)
        calib_scores[lbl] = raw - prior

    best_label = max(calib_scores, key=calib_scores.get)

    if debug_scores:
        return best_label, calib_scores, raw_scores
    return best_label


# ============================
# EVALUATION (PER SHOT RATIO)
# ============================

def evaluate_for_ratio(
    model,
    processor,
    tokenizer,
    label_to_examples,
    eval_pairs,
    shot_ratio,
    total_remaining,
    label_log_priors,
    debug_for_this_ratio,
    csv_rows,
    ratio_log_lines,
):
    """
    Evaluate for a single shot_ratio.
    csv_rows: list to fill with per-image results for this ratio.
    ratio_log_lines: log strings for this ratio; will be saved to file.
    """

    # Local logger that writes to global and ratio-specific logs
    def log_eval(msg=""):
        print(msg)
        LOG_LINES.append(str(msg))
        ratio_log_lines.append(str(msg))

    n_eval = len(eval_pairs)

    log_eval("")
    log_eval("===== SHOT_RATIO = %.3f =====" % shot_ratio)
    log_eval(
        "Evaluating on %d images (from %d remaining images in the working set after selecting few-shot examples)"
        % (n_eval, total_remaining)
    )

    correct = 0
    predictions = []

    for idx, (img_path, true_label) in enumerate(eval_pairs, start=1):
        debug = debug_for_this_ratio and idx <= DEBUG_NUM_IMAGES

        if debug:
            pred_label, calib_scores, raw_scores = predict_label_for_image_joint_calibrated(
                model,
                processor,
                tokenizer,
                img_path,
                label_to_examples=label_to_examples,
                label_log_priors=label_log_priors,
                debug_scores=True,
            )
        else:
            pred_label = predict_label_for_image_joint_calibrated(
                model,
                processor,
                tokenizer,
                img_path,
                label_to_examples=label_to_examples,
                label_log_priors=label_log_priors,
                debug_scores=False,
            )

        predictions.append((img_path, true_label, pred_label))
        is_correct = int(pred_label == true_label)
        if is_correct:
            correct += 1

        # Log per-image info
        log_eval("[%d/%d] %s" % (idx, n_eval, img_path))
        log_eval("   GT   : %s" % true_label)
        log_eval("   Pred : %s" % pred_label)

        if debug:
            log_eval("   Raw scores (higher = more likely before calibration):")
            for lbl in LABELS:
                log_eval("      %s: raw=%.4f" % (lbl, raw_scores[lbl]))
            log_eval("   Calibrated scores (raw - prior):")
            for lbl in LABELS:
                log_eval("      %s: calib=%.4f" % (lbl, calib_scores[lbl]))

        # Append a row for CSV output (per ratio)
        csv_rows.append({
            "shot_ratio": shot_ratio,
            "img_path": img_path,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": is_correct,
        })

    accuracy = correct / float(n_eval) if n_eval > 0 else 0.0
    log_eval("")
    log_eval("----------------------")
    log_eval("Joint multi-class (few-shot, calibrated) accuracy @ SHOT_RATIO=%.3f: %.4f" % (shot_ratio, accuracy))
    log_eval("Correct: %d / %d" % (correct, n_eval))
    log_eval("----------------------")

    gt_counts = Counter([gt for _, gt, _ in predictions])
    pred_counts = Counter([pred for _, _, pred in predictions])

    log_eval("Ground-truth distribution in eval set:")
    for lbl in LABELS:
        log_eval("  %s: %d" % (lbl, gt_counts.get(lbl, 0)))

    log_eval("Prediction distribution:")
    for lbl in LABELS:
        log_eval("  %s: %d" % (lbl, pred_counts.get(lbl, 0)))

    log_eval("Per-class accuracy:")
    per_class_correct = Counter()
    per_class_total = Counter()
    for _, gt, pred in predictions:
        per_class_total[gt] += 1
        if gt == pred:
            per_class_correct[gt] += 1

    for lbl in LABELS:
        total = per_class_total.get(lbl, 0)
        if total == 0:
            acc_lbl = 0.0
        else:
            acc_lbl = per_class_correct.get(lbl, 0) / float(total)
        log_eval("  %s: %.4f (%d/%d)" % (lbl, acc_lbl, per_class_correct.get(lbl, 0), total))


# ============================
# MAIN
# ============================

def main():
    # Ensure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_pairs = collect_image_label_pairs()
    n_total = len(all_pairs)
    log("Total images found: %d" % n_total)

    if n_total == 0:
        log("No images found. Check DATA_ROOT and folder names.")
        return

    random.seed(RANDOM_SEED)
    random.shuffle(all_pairs)

    n_working = max(1, int(DATA_USAGE_RATIO * n_total))
    working_pairs = all_pairs[:n_working]
    log("Using %d images (~%.1f%% of total) as working set." % (n_working, DATA_USAGE_RATIO * 100.0))
    log("All few-shot examples and evaluation samples will come ONLY from this working set.")

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    model, processor, tokenizer = load_model_and_processors()
    log("Loaded Video-LLaVA model from: %s" % str(VIDEOLLAVA_MODEL_ID))

    # ===== 1) Estimate priors over A/B/C/D/E/F once, using zero-shot prompts =====
    label_log_priors = estimate_label_priors_joint(
        model,
        processor,
        tokenizer,
        image_label_pairs=all_pairs,
        max_calib=CALIBRATION_NUM_IMAGES,
    )

    # Global CSV aggregator across all shot ratios
    csv_rows_all = []

    # ===== 2) Run evaluation for different shot ratios =====
    for i, shot_ratio in enumerate(SHOT_RATIOS):
        label_to_examples, eval_pairs, n_remaining, n_eval = split_support_and_eval_from_working_set(
            working_pairs,
            shot_ratio_per_class=shot_ratio,
            sample_ratio=SAMPLE_RATIO,
            seed=RANDOM_SEED + i,
        )

        log("")
        log(
            "Remaining images in working set after few-shot selection (shot_ratio=%.3f): %d"
            % (shot_ratio, n_remaining)
        )
        log("Eval subset size (sampled from remaining working images): %d" % n_eval)

        debug_for_this_ratio = DEBUG_SHOW_SCORES and (i == 0)

        # Per-ratio structures
        ratio_log_lines = []
        csv_rows_ratio = []

        # Evaluate for this ratio (now truly few-shot)
        evaluate_for_ratio(
            model,
            processor,
            tokenizer,
            label_to_examples,
            eval_pairs,
            shot_ratio=shot_ratio,
            total_remaining=n_remaining,
            label_log_priors=label_log_priors,
            debug_for_this_ratio=debug_for_this_ratio,
            csv_rows=csv_rows_ratio,
            ratio_log_lines=ratio_log_lines,
        )

        # Save per-ratio text log & CSV under LLAVA_Few
        ratio_id = f"{shot_ratio:.3f}".replace(".", "_")

        txt_path = os.path.join(OUTPUT_DIR, f"log_shot_{ratio_id}.txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(ratio_log_lines))
        log("Saved per-ratio text log to: %s" % txt_path)

        csv_path = os.path.join(OUTPUT_DIR, f"predictions_shot_{ratio_id}.csv")
        fieldnames = ["shot_ratio", "img_path", "true_label", "pred_label", "correct"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows_ratio:
                writer.writerow(row)
        log("Saved per-ratio CSV predictions to: %s" % csv_path)

        # Extend global aggregator
        csv_rows_all.extend(csv_rows_ratio)

    # ===== 3) Save global logs to text and CSV files =====
    log_txt_path = "videollava_eval_log.txt"
    with open(log_txt_path, "w") as f:
        f.write("\n".join(LOG_LINES))

    log("")
    log("Saved global text log to: %s" % log_txt_path)

    csv_global_path = "videollava_eval_predictions.csv"
    fieldnames = ["shot_ratio", "img_path", "true_label", "pred_label", "correct"]
    with open(csv_global_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows_all:
            writer.writerow(row)

    log("Saved global CSV predictions to: %s" % csv_global_path)


if __name__ == "__main__":
    main()