"""Rule-based MathVision evaluation for counterfactual calibration runs.

This replaces an unavailable private evaluator; it is not an LLM judge.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from answer_utils import extract_answer


def is_correct(prediction, item):
    if prediction is None:
        return False
    prediction = str(prediction).strip().rstrip(".")
    answer = str(item.get("answer", "")).strip()
    if prediction.casefold() == answer.casefold():
        return True
    if len(answer) == 1 and answer.upper() in "ABCDE" and item.get("options"):
        index = ord(answer.upper()) - ord("A")
        if index < len(item["options"]):
            answer = str(item["options"][index])
    from math_verify import parse, verify

    return bool(verify(parse(answer), parse(prediction)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", type=Path, required=True)
    args = parser.parse_args()
    details = []
    for path in sorted(args.result_dir.iterdir()):
        if not path.is_dir() or not (path / "ori.json").exists():
            continue
        item = json.loads((path / "ori.json").read_text())
        result_path = path / "reasoning_result.json"
        prediction = None
        if result_path.exists():
            result = json.loads(result_path.read_text())
            text = "\n".join(
                step.get("content", "")
                for step in result.get("reasoning_steps", [])
                if step.get("type") == "text"
            )
            prediction = result.get("extracted_answer") or extract_answer(text)
        details.append(
            {
                "id": path.name,
                "pred_normalized": prediction,
                "correctness": is_correct(prediction, item),
            }
        )
    if not details:
        raise ValueError("No sample directories with ori.json found")
    summary = {
        "evaluator": "rule-based-math-verify",
        "total": len(details),
        "accuracy": sum(row["correctness"] for row in details) / len(details),
        "details": details,
    }
    (args.result_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Accuracy: {summary['accuracy']:.4%} ({len(details)} samples)")


if __name__ == "__main__":
    main()
