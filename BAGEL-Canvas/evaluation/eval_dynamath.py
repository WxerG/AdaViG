#!/usr/bin/env python3
"""Evaluate BAGEL-Canvas results on DynaMath (reasoning subset).

DynaMath has 501 seed questions × 10 variants each = 5010 samples.
Metrics (mirrored from lmms-eval `dynamath_reasoning`):
  - dynamath_average : per-seed mean accuracy across 10 variants, then mean over seeds
  - dynamath_worst   : per-seed min accuracy across 10 variants, then mean over seeds

Rule-based scoring (no LLM judge):
  1. Extract answer from <answer>...</answer> → \\boxed{...} → last-number regex.
  2. relax_exact_match (string inclusion within relax_portion).
  3. math_verify.parse() + verify() fallback for symbolic/numeric equivalence.

Usage:
    python evaluation/eval_dynamath.py \\
        --result_dir /path/to/results/dynamath/run_name \\
        --gt_path data_handlers/dynamath/test.jsonl
"""

import argparse
import json
import os
import re
from collections import defaultdict

from math_verify import parse as mv_parse, verify as mv_verify


# ---------------------------------------------------------------------------
# Answer extraction (ported from lmms_eval.tasks._task_utils.reasoning_utils)
# ---------------------------------------------------------------------------


def extract_boxed_answer(predict_str: str) -> str:
    boxed_start = "\\boxed{"
    start_indices = []
    pos = 0
    while True:
        pos = predict_str.find(boxed_start, pos)
        if pos == -1:
            break
        start_indices.append(pos)
        pos += 1
    if not start_indices:
        return ""
    results = []
    for start_pos in start_indices:
        brace_count = 0
        pos = start_pos + len(boxed_start) - 1
        while pos < len(predict_str):
            ch = predict_str[pos]
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    content_start = start_pos + len(boxed_start)
                    results.append(predict_str[content_start:pos])
                    break
            pos += 1
    return results[-1] if results else ""


def extract_answer_tag(predict_str: str) -> str:
    if not predict_str:
        return ""

    # 1. <answer>...</answer> (case-insensitive, BAGEL sometimes emits <Answer>)
    m = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2. Standard \boxed{...}
    boxed = extract_boxed_answer(predict_str)
    if boxed:
        return boxed

    # 3. Malformed </boxed>X</boxed> (BAGEL-Canvas sometimes emits this)
    m = re.search(r"</boxed>\s*(.+?)\s*</boxed>", predict_str)
    if m:
        return m.group(1).strip()

    # 4. "The correct choice is X" / "the answer is X" — prose patterns
    for line in reversed(predict_str.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        # MCQ letter in angle brackets: <B>, <C> etc.
        m = re.search(r"<([A-Ha-h])>\.?\s*$", line)
        if m:
            return m.group(1).upper()
        # "choice is X" / "answer is X"
        m = re.search(
            r"(?:correct\s+)?(?:choice|answer|option)\s+is\s+\(?([A-Ha-h])\)?\b",
            line,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).upper()

    # 5. Last number anywhere in text (relaxed: any line, not just trailing)
    all_nums = re.findall(r"-?\d+(?:\.\d+)?", predict_str)
    if all_nums:
        return all_nums[-1]

    # 6. LaTeX fraction outside boxed: \frac{a}{b} or \tfrac{a}{b}
    fracs = re.findall(r"\\t?frac\{([^}]+)\}\{([^}]+)\}", predict_str)
    if fracs:
        num, den = fracs[-1]
        return f"{num}/{den}"

    # 7. MCQ letter anywhere near end
    m = re.search(
        r"\b([A-H])\b", predict_str[-100:] if len(predict_str) > 100 else predict_str
    )
    if m:
        return m.group(1)

    return ""


MCQ_LETTERS = {"A", "B", "C", "D", "E", "F", "G", "H"}


def parse_mcq(predict_str: str) -> str:
    if not predict_str or not predict_str.strip():
        return ""
    response = predict_str.strip()
    for ch in [",", ".", "!", "?", ";", ":", "'", '"']:
        response = response.strip(ch)
    response = " " + response + " "
    for letter in MCQ_LETTERS:
        if f" {letter} " in response:
            return letter
    return ""


def simple_parse(predict_str: str) -> str:
    if predict_str.endswith("."):
        predict_str = predict_str[:-1]
    return predict_str.strip()


def relax_exact_match(
    predict_str: str, ground_truth: str, relax_portion: float = 0.9
) -> float:
    if parse_mcq(ground_truth) in MCQ_LETTERS:
        p = parse_mcq(predict_str)
        return (
            1.0 if p.lower().strip() == parse_mcq(ground_truth).lower().strip() else 0.0
        )
    if predict_str in ground_truth and len(predict_str) >= relax_portion * len(
        ground_truth
    ):
        return 1.0
    if ground_truth in predict_str and len(ground_truth) >= relax_portion * len(
        predict_str
    ):
        return 1.0
    return 1.0 if predict_str.strip() == ground_truth.strip() else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    p = simple_parse(predict_str)
    g = simple_parse(ground_truth)
    score = relax_exact_match(p, g)
    if score == 0.0:
        try:
            gold = mv_parse(g)
            pred = mv_parse(p)
            score = 1.0 if mv_verify(gold, pred) else 0.0
        except Exception:
            score = 0.0
    return score


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------


def load_gt(gt_path: str):
    """Returns list of GT dicts indexed by row order (matches 5-digit row_idx prefix)."""
    entries = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def load_predictions(result_dir: str):
    """Walks result_dir/*/reasoning_result.json, returns dict row_idx -> extracted_answer."""
    preds = {}
    for d in sorted(os.listdir(result_dir)):
        rfile = os.path.join(result_dir, d, "reasoning_result.json")
        if not os.path.isfile(rfile):
            continue
        with open(rfile, "r", encoding="utf-8") as f:
            data = json.load(f)
        text_parts = [
            s["content"]
            for s in data.get("reasoning_steps", [])
            if s.get("type") == "text"
        ]
        full_text = "\n".join(text_parts)
        extracted = data.get("extracted_answer") or extract_answer_tag(full_text)
        parts = d.split("__", 1)
        row_idx = int(parts[0]) if len(parts) == 2 and parts[0].isdigit() else None
        preds[d] = {
            "row_index": row_idx,
            "dirname": d,
            "extracted": extracted or "",
            "full_text": full_text,
            "n_images": sum(
                1 for s in data.get("reasoning_steps", []) if s.get("type") == "image"
            ),
        }
    return preds


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


def evaluate(gt_entries, preds):
    by_row = {}
    for dirname, p in preds.items():
        if p["row_index"] is not None:
            by_row[p["row_index"]] = p

    # Per-sample accuracies
    per_seed = defaultdict(list)  # seed_id -> list of 0/1 accuracies
    records = []
    n_with_pred = 0
    n_image = 0

    for i, gt in enumerate(gt_entries):
        pred_info = by_row.get(i, {})
        raw_pred = pred_info.get("extracted", "")
        n_imgs = pred_info.get("n_images", 0)
        if raw_pred:
            n_with_pred += 1
        if n_imgs > 0:
            n_image += 1

        truth = str(gt["answer"]).strip()
        acc = acc_reward(raw_pred, truth) if raw_pred else 0.0
        seed_id = gt.get("seed_id", gt["id"])
        per_seed[seed_id].append(acc)

        records.append(
            {
                "row_index": i,
                "id": gt["id"],
                "seed_id": seed_id,
                "answer": truth,
                "predicted_raw": raw_pred,
                "correct": bool(acc),
                "n_images": n_imgs,
            }
        )

    # Aggregate: dynamath_average = mean over seeds of (per-seed mean accuracy)
    #            dynamath_worst   = mean over seeds of (per-seed min accuracy)
    seed_avgs = [sum(v) / len(v) for v in per_seed.values()]
    seed_worsts = [min(v) for v in per_seed.values()]
    dynamath_average = sum(seed_avgs) / len(seed_avgs) if seed_avgs else 0.0
    dynamath_worst = sum(seed_worsts) / len(seed_worsts) if seed_worsts else 0.0

    total = len(gt_entries)
    per_sample_acc = sum(r["correct"] for r in records) / total if total else 0.0

    return {
        "dynamath_average": dynamath_average,
        "dynamath_worst": dynamath_worst,
        "per_sample_acc": per_sample_acc,
        "total": total,
        "n_seeds": len(per_seed),
        "n_with_pred": n_with_pred,
        "n_image_steps": n_image,
        "records": records,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--gt_path", default="data_handlers/dynamath/test.jsonl")
    args = parser.parse_args()

    gt_entries = load_gt(args.gt_path)
    print(f"GT: {len(gt_entries)} samples")

    preds = load_predictions(args.result_dir)
    print(f"Predictions: {len(preds)} result dirs")

    results = evaluate(gt_entries, preds)

    print(f"\n{'=' * 60}")
    print(f"dynamath_average : {results['dynamath_average']:.4f}")
    print(f"dynamath_worst   : {results['dynamath_worst']:.4f}")
    print(f"per_sample_acc   : {results['per_sample_acc']:.4f}")
    print(f"seeds            : {results['n_seeds']}")
    print(f"total            : {results['total']}")
    print(f"with pred        : {results['n_with_pred']}/{results['total']}")
    print(f"with image steps : {results['n_image_steps']}/{results['total']}")
    print(f"{'=' * 60}")

    out_path = os.path.join(args.result_dir, "dynamath_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
