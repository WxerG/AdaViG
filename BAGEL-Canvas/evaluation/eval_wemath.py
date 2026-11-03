#!/usr/bin/env python3
"""Evaluate BAGEL-Canvas results on We-Math testmini.

Usage:
    python evaluation/eval_wemath.py \
        --result_dir /path/to/results/wemath/run_name \
        --gt_path data_handlers/wemath/testmini.jsonl

Produces the same joker/loose/strict metrics as lmms-eval's wemath task.
"""

import argparse
import json
import os
import re

import pandas as pd


# ---------------------------------------------------------------------------
# Answer extraction (same logic as lmms-eval reasoning_utils)
# ---------------------------------------------------------------------------


def extract_boxed_answer(text: str) -> str:
    """Extract content from \\boxed{...}, handling nested braces."""
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
    """Try <answer> tag first, then \\boxed{}, then last number."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    return ""


def normalize_answer(ans: str) -> str:
    """Normalize extracted answer for comparison."""
    ans = ans.strip().rstrip(".")
    # For multiple choice, keep just the letter
    m = re.match(r"^([A-Ea-e])\b", ans)
    if m:
        return m.group(1).upper()
    return ans.upper()


# ---------------------------------------------------------------------------
# We-Math scoring (copied from lmms-eval wemath_utils.py)
# ---------------------------------------------------------------------------


def process_steps_data(df, steps):
    steps_data = {
        f"{steps}steps_{i}": df[df["key"] == f"{steps}steps_{i}"]
        for i in range(1, steps + 1)
    }
    steps_data[f"{steps}steps_multi"] = df[df["key"] == f"{steps}steps_multi"]
    for key, data in steps_data.items():
        suffix = key.split("_")[-1]
        data = data.copy()
        data.columns = [col + f"_{suffix}" for col in data.columns]
        steps_data[key] = data
    merged_data = steps_data[f"{steps}steps_1"]
    for i in range(2, steps + 1):
        merged_data = pd.merge(
            merged_data,
            steps_data[f"{steps}steps_{i}"],
            left_on="ID_1",
            right_on=f"ID_{i}",
            how="left",
        )
    merged_data = pd.merge(
        merged_data,
        steps_data[f"{steps}steps_multi"],
        left_on="ID_1",
        right_on="ID_multi",
        how="left",
    )
    return merged_data


def calculate_metrics(merged_2steps, merged_3steps):
    metrics = {}
    metrics["steps2_filtered_rows_1_loose"] = merged_2steps[
        ((merged_2steps["joker_1"] == False) & (merged_2steps["joker_2"] == False))
        & (merged_2steps["joker_multi"] == True)
    ]
    metrics["steps2_filtered_rows_1_strict"] = merged_2steps[
        ((merged_2steps["joker_1"] == False) | (merged_2steps["joker_2"] == False))
        & (merged_2steps["joker_multi"] == True)
    ]
    metrics["steps2_filtered_rows_2"] = merged_2steps[
        ((merged_2steps["joker_1"] == True) & (merged_2steps["joker_2"] == True))
        & (merged_2steps["joker_multi"] == False)
    ]
    metrics["steps2_filtered_rows_3"] = merged_2steps[
        ((merged_2steps["joker_1"] == False) | (merged_2steps["joker_2"] == False))
        & (merged_2steps["joker_multi"] == False)
    ]
    metrics["steps2_filtered_rows_4_loose"] = merged_2steps[
        ((merged_2steps["joker_1"] == True) | (merged_2steps["joker_2"] == True))
        & (merged_2steps["joker_multi"] == True)
    ]
    metrics["steps2_filtered_rows_4_strict"] = merged_2steps[
        ((merged_2steps["joker_1"] == True) & (merged_2steps["joker_2"] == True))
        & (merged_2steps["joker_multi"] == True)
    ]
    metrics["steps3_filtered_rows_1_loose"] = merged_3steps[
        (
            (merged_3steps["joker_1"] == False)
            & (merged_3steps["joker_2"] == False)
            & (merged_3steps["joker_3"] == False)
        )
        & (merged_3steps["joker_multi"] == True)
    ]
    metrics["steps3_filtered_rows_1_strict"] = merged_3steps[
        (
            (merged_3steps["joker_1"] == False)
            | (merged_3steps["joker_2"] == False)
            | (merged_3steps["joker_3"] == False)
        )
        & (merged_3steps["joker_multi"] == True)
    ]
    metrics["steps3_filtered_rows_2"] = merged_3steps[
        (
            (merged_3steps["joker_1"] == True)
            & (merged_3steps["joker_2"] == True)
            & (merged_3steps["joker_3"] == True)
        )
        & (merged_3steps["joker_multi"] == False)
    ]
    metrics["steps3_filtered_rows_3"] = merged_3steps[
        (
            (merged_3steps["joker_1"] == False)
            | (merged_3steps["joker_2"] == False)
            | (merged_3steps["joker_3"] == False)
        )
        & (merged_3steps["joker_multi"] == False)
    ]
    metrics["steps3_filtered_rows_4_loose"] = merged_3steps[
        (
            (merged_3steps["joker_1"] == True)
            | (merged_3steps["joker_2"] == True)
            | (merged_3steps["joker_3"] == True)
        )
        & (merged_3steps["joker_multi"] == True)
    ]
    metrics["steps3_filtered_rows_4_strict"] = merged_3steps[
        (
            (merged_3steps["joker_1"] == True)
            & (merged_3steps["joker_2"] == True)
            & (merged_3steps["joker_3"] == True)
        )
        & (merged_3steps["joker_multi"] == True)
    ]
    return metrics


def compute_final_scores(metrics, total_count=525):
    total_counts = {
        "InadequateGeneralization": len(metrics["steps2_filtered_rows_2"])
        + len(metrics["steps3_filtered_rows_2"]),
        "InsufficientKnowledge": len(metrics["steps2_filtered_rows_3"])
        + len(metrics["steps3_filtered_rows_3"]),
        "CompleteMastery_loose": len(metrics["steps2_filtered_rows_4_loose"])
        + len(metrics["steps3_filtered_rows_4_loose"]),
        "CompleteMastery_strict": len(metrics["steps2_filtered_rows_4_strict"])
        + len(metrics["steps3_filtered_rows_4_strict"]),
        "RoteMemorization_loose": len(metrics["steps2_filtered_rows_1_loose"])
        + len(metrics["steps3_filtered_rows_1_loose"]),
        "RoteMemorization_strict": len(metrics["steps2_filtered_rows_1_strict"])
        + len(metrics["steps3_filtered_rows_1_strict"]),
    }
    rates = {
        "InadequateGeneralization_rate": total_counts["InadequateGeneralization"]
        / total_count,
        "InsufficientKnowledge_rate": total_counts["InsufficientKnowledge"]
        / total_count,
        "CompleteMastery_loose_rate": total_counts["CompleteMastery_loose"]
        / total_count,
        "CompleteMastery_strict_rate": total_counts["CompleteMastery_strict"]
        / total_count,
        "RoteMemorization_loose_rate": total_counts["RoteMemorization_loose"]
        / max(
            total_counts["CompleteMastery_loose"]
            + total_counts["RoteMemorization_loose"],
            1,
        ),
        "RoteMemorization_strict_rate": total_counts["RoteMemorization_strict"]
        / max(
            total_counts["CompleteMastery_strict"]
            + total_counts["RoteMemorization_strict"],
            1,
        ),
    }
    return total_counts, rates


def compute_wemath_score(total_counts, total=525):
    score_loose = (
        total
        - 0.5 * total_counts["InadequateGeneralization"]
        - total_counts["RoteMemorization_loose"]
        - total_counts["InsufficientKnowledge"]
    ) / total
    score_strict = (
        total
        - 0.5 * total_counts["InadequateGeneralization"]
        - total_counts["RoteMemorization_strict"]
        - total_counts["InsufficientKnowledge"]
    ) / total
    return score_loose, score_strict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_gt(gt_path: str):
    """Load ground truth from wemath testmini.jsonl, return list of dicts."""
    entries = []
    with open(gt_path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def load_predictions(result_dir: str):
    """Load predictions from BAGEL-Canvas result directory.

    Handles both old format (dirname = item_id, e.g. '3stpes_1')
    and new format (dirname = '{row_index:05d}__{item_id}', e.g. '00003__3stpes_1').
    Returns dict mapping row_index -> extracted_answer.
    """
    predictions = {}
    for dirname in os.listdir(result_dir):
        rfile = os.path.join(result_dir, dirname, "reasoning_result.json")
        if not os.path.isfile(rfile):
            continue
        with open(rfile) as f:
            data = json.load(f)

        # Extract answer from reasoning text
        extracted = data.get("extracted_answer", "")
        if not extracted:
            text_parts = [
                s["content"]
                for s in data.get("reasoning_steps", [])
                if s.get("type") == "text"
            ]
            full_text = "\n".join(text_parts)
            extracted = extract_answer(full_text)

        # Parse dirname to get row_index
        # New format: "00003__3stpes_1" -> row_index=3
        # Old format: "3stpes_1" -> need to match by ID
        parts = dirname.split("__", 1)
        if len(parts) == 2 and parts[0].isdigit():
            row_index = int(parts[0])
        else:
            row_index = None  # will match by ID later

        predictions[dirname] = {
            "row_index": row_index,
            "item_id": data.get("id", parts[-1] if len(parts) == 2 else dirname),
            "extracted_answer": extracted,
        }
    return predictions


def match_predictions_to_gt(gt_entries, predictions):
    """Match predictions to ground truth entries. Returns list of (gt, pred_answer) pairs."""
    matched = []

    # Build row_index -> prediction map
    by_row = {}
    by_id = {}
    for dirname, pred in predictions.items():
        if pred["row_index"] is not None:
            by_row[pred["row_index"]] = pred
        else:
            item_id = pred["item_id"]
            if item_id not in by_id:
                by_id[item_id] = []
            by_id[item_id].append(pred)

    # Match
    id_counters = {}
    for i, gt in enumerate(gt_entries):
        if i in by_row:
            matched.append((gt, by_row[i]["extracted_answer"]))
        else:
            # Old format: match by ID sequentially
            item_id = gt["id"]
            if item_id not in id_counters:
                id_counters[item_id] = 0
            idx = id_counters[item_id]
            preds_for_id = by_id.get(item_id, [])
            if idx < len(preds_for_id):
                matched.append((gt, preds_for_id[idx]["extracted_answer"]))
            else:
                matched.append((gt, ""))  # missing prediction
            id_counters[item_id] = idx + 1

    return matched


def evaluate(matched):
    """Score all matched pairs and compute wemath metrics."""
    records = []
    correct = 0
    total = 0
    for gt, pred_answer in matched:
        pred_norm = normalize_answer(pred_answer)
        gt_answer = normalize_answer(gt["answer"])
        acc = 1.0 if pred_norm == gt_answer else 0.0
        correct += acc
        total += 1
        records.append(
            {
                "ID": gt["id"],
                "key": gt.get("key", ""),
                "knowledge concept": gt.get("knowledge_concept", ""),
                "answer": gt["answer"],
                "predicted": pred_answer,
                "acc_score": acc,
            }
        )

    df = pd.DataFrame(records)
    df["joker"] = df["acc_score"] == 1.0

    # Simple accuracy
    simple_acc = correct / total if total > 0 else 0.0

    # We-Math metrics
    data_2steps = df[df["key"].str.contains("2steps")]
    data_3steps = df[df["key"].str.contains("3steps")]

    if len(data_2steps) == 0 or len(data_3steps) == 0:
        print(
            f"WARNING: 2steps={len(data_2steps)}, 3steps={len(data_3steps)}. Cannot compute wemath metrics."
        )
        return simple_acc, None, None, df

    merged_2steps = process_steps_data(data_2steps, 2)
    merged_3steps = process_steps_data(data_3steps, 3)
    metrics = calculate_metrics(merged_2steps, merged_3steps)
    total_counts, rates = compute_final_scores(metrics, total_count=525)
    score_loose, score_strict = compute_wemath_score(total_counts)

    return simple_acc, score_loose, score_strict, df


def main():
    parser = argparse.ArgumentParser(description="Evaluate BAGEL-Canvas on We-Math")
    parser.add_argument(
        "--result_dir",
        required=True,
        help="Path to result directory (contains per-sample subdirs)",
    )
    parser.add_argument(
        "--gt_path",
        default="data_handlers/wemath/testmini.jsonl",
        help="Path to ground truth JSONL",
    )
    args = parser.parse_args()

    gt_entries = load_gt(args.gt_path)
    print(
        f"Ground truth: {len(gt_entries)} entries, {len(set(e['id'] for e in gt_entries))} unique IDs"
    )

    predictions = load_predictions(args.result_dir)
    print(f"Predictions: {len(predictions)} result dirs")

    matched = match_predictions_to_gt(gt_entries, predictions)
    n_with_pred = sum(1 for _, p in matched if p)
    print(f"Matched: {n_with_pred}/{len(matched)} entries have predictions")

    simple_acc, score_loose, score_strict, df = evaluate(matched)

    print(f"\n{'=' * 50}")
    print(
        f"Simple Accuracy: {simple_acc:.4f} ({int(simple_acc * len(matched))}/{len(matched)})"
    )
    if score_loose is not None:
        print(f"We-Math Score (Loose):  {score_loose:.4f} ({score_loose:.2%})")
        print(f"We-Math Score (Strict): {score_strict:.4f} ({score_strict:.2%})")
    print(f"{'=' * 50}")

    # Save detailed results
    out_path = os.path.join(args.result_dir, "wemath_eval_results.json")
    results = {
        "simple_accuracy": simple_acc,
        "score_loose": score_loose,
        "score_strict": score_strict,
        "total_entries": len(matched),
        "entries_with_predictions": n_with_pred,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
