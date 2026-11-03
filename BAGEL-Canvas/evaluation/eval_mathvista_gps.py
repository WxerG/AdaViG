#!/usr/bin/env python3
"""Evaluate BAGEL-Canvas results on MathVista GPS (geometry problem solving) subset.

Usage:
    python evaluation/eval_mathvista_gps.py \
        --result_dir /path/to/results/mathvista/run_name \
        --gt_path data_handlers/mathvista/testmini_gps.jsonl
"""

import argparse
import json
import os
import re


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_boxed(text: str) -> str:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    depth, start = 0, idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return ""


def extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    # Last line fallback
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def normalize_mcq(ans: str, choices: list) -> str:
    """Map extracted answer to one of the choices (text), return empty if no match."""
    ans = ans.strip()
    options = [chr(ord("A") + i) for i in range(len(choices))]

    # Extract letter from "(A)" or "A." or bare "A"
    m = re.match(r"^\(?([A-Ea-e])\)?[.\s]?", ans)
    if m:
        letter = m.group(1).upper()
        if letter in options:
            return choices[options.index(letter)]

    # Direct choice match
    for choice in choices:
        if ans.strip().lower() == choice.strip().lower():
            return choice

    # Levenshtein fallback (simple)
    if choices:

        def lev(a, b):
            a, b = a.lower(), b.lower()
            dp = list(range(len(b) + 1))
            for i, ca in enumerate(a):
                ndp = [i + 1]
                for j, cb in enumerate(b):
                    ndp.append(
                        min(ndp[j] + 1, dp[j + 1] + 1, dp[j] + (0 if ca == cb else 1))
                    )
                dp = ndp
            return dp[-1]

        return min(choices, key=lambda c: lev(ans, c))

    return ans


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------


def load_gt(gt_path: str):
    entries = []
    with open(gt_path) as f:
        for line in f:
            item = json.loads(line)
            # Parse options string -> list
            opts = item.get("options", "[]")
            if isinstance(opts, str):
                try:
                    opts = json.loads(opts.replace("'", '"'))
                except Exception:
                    opts = []
            item["options"] = opts
            entries.append(item)
    return entries


def load_predictions(result_dir: str):
    """Returns dict: row_index -> extracted_answer."""
    preds = {}
    for d in os.listdir(result_dir):
        rfile = os.path.join(result_dir, d, "reasoning_result.json")
        if not os.path.isfile(rfile):
            continue
        with open(rfile) as f:
            data = json.load(f)

        text_parts = [
            s["content"]
            for s in data.get("reasoning_steps", [])
            if s.get("type") == "text"
        ]
        full_text = "\n".join(text_parts)
        extracted = data.get("extracted_answer", "") or extract_answer(full_text)

        parts = d.split("__", 1)
        if len(parts) == 2 and parts[0].isdigit():
            row_idx = int(parts[0])
        else:
            row_idx = None

        preds[d] = {
            "row_index": row_idx,
            "extracted": extracted,
            "n_images": sum(
                1 for s in data.get("reasoning_steps", []) if s.get("type") == "image"
            ),
        }
    return preds


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


def evaluate(gt_entries, preds):
    # Build row_index -> pred map
    by_row = {}
    for dirname, p in preds.items():
        if p["row_index"] is not None:
            by_row[p["row_index"]] = p

    correct = total = skipped = 0
    records = []

    for i, gt in enumerate(gt_entries):
        pred_info = by_row.get(i, {})
        raw_pred = pred_info.get("extracted", "")
        choices = gt.get("options", [])
        n_imgs = pred_info.get("n_images", 0)

        if n_imgs == 0:
            skipped += 1

        # Normalize: if MCQ, map to choice text
        if choices:
            pred_norm = normalize_mcq(raw_pred, choices)
            truth_norm = gt["answer"].strip()
        else:
            pred_norm = raw_pred.strip()
            truth_norm = gt["answer"].strip()

        acc = 1.0 if pred_norm.lower() == truth_norm.lower() else 0.0
        correct += acc
        total += 1
        records.append(
            {
                "id": gt["id"],
                "answer": truth_norm,
                "predicted_raw": raw_pred,
                "predicted_norm": pred_norm,
                "correct": bool(acc),
                "n_images": n_imgs,
            }
        )

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, skipped, total, records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument(
        "--gt_path", default="data_handlers/mathvista/testmini_gps.jsonl"
    )
    args = parser.parse_args()

    gt_entries = load_gt(args.gt_path)
    print(f"GT: {len(gt_entries)} GPS samples")

    preds = load_predictions(args.result_dir)
    print(f"Predictions: {len(preds)} result dirs")

    accuracy, skipped, total, records = evaluate(gt_entries, preds)
    n_with_pred = sum(1 for r in records if r["predicted_raw"])
    skip_rate = skipped / total if total > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"Accuracy:    {accuracy:.4f} ({int(accuracy * total)}/{total})")
    print(f"Skip rate:   {skip_rate:.1%} ({skipped}/{total})")
    print(f"With pred:   {n_with_pred}/{total}")
    print(f"{'=' * 50}")

    out_path = os.path.join(args.result_dir, "mvgps_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "skip_rate": skip_rate,
                "total": total,
                "correct": int(accuracy * total),
                "skipped_images": skipped,
                "records": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
