"""Search rounded coarse thresholds, then refine the best regions locally."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

INTENT = "r_all__L27__s5"
FIDELITY = "orig_vit__L0_14_27__s5"


DEFAULT_INTENT = 0.60
DEFAULT_FIDELITY = 0.01
COARSE_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)


def snap(value, step):
    """Round to a bounded decimal grid without exposing float artifacts."""
    return round(min(1.0, max(0.0, np.floor(float(value) / step + 0.5) * step)), 6)


def coarse_candidates(values, step, default):
    return sorted(
        {default, *(snap(v, step) for v in np.quantile(values, COARSE_QUANTILES))}
    )


def rank_results(rows):
    # Final threshold keys make ties reproducible.
    return pd.DataFrame(rows).sort_values(
        [
            "accuracy",
            "counterfactual_cost_ms",
            "intent_threshold",
            "fidelity_threshold",
        ],
        ascending=[False, True, True, True],
    )


def search_thresholds(df, app):
    intent = df[INTENT].to_numpy()
    fidelity = df[FIDELITY].to_numpy()
    rows = []
    seen = set()

    def evaluate(intent_thr, fidelity_thr, stage):
        pair = (intent_thr, fidelity_thr)
        if pair in seen:
            return
        seen.add(pair)
        keep = ~app | (intent > intent_thr) | (fidelity > fidelity_thr)
        rows.append(
            {
                "intent_threshold": intent_thr,
                "fidelity_threshold": fidelity_thr,
                "accuracy": float(
                    np.where(keep, df["off_correct"], df["wo_correct"]).mean()
                ),
                "counterfactual_cost_ms": float(
                    np.where(keep, df["off_cost_ms"], df["wo_cost_ms"]).mean()
                ),
                "abort_count": int((~keep).sum()),
                "search_stage": stage,
            }
        )

    if not app.any():
        raise ValueError(
            "No gate-applicable samples available for threshold calibration"
        )
    intent_candidates = coarse_candidates(intent[app], 0.1, DEFAULT_INTENT)
    fidelity_candidates = coarse_candidates(fidelity[app], 0.01, DEFAULT_FIDELITY)
    for intent_thr in intent_candidates:
        for fidelity_thr in fidelity_candidates:
            evaluate(intent_thr, fidelity_thr, "coarse")

    coarse = rank_results(rows)
    # Freeze the three best coarse pairs before adding any fine candidates.
    for center in coarse.head(3).to_dict("records"):
        for intent_offset in (-0.05, 0.0, 0.05):
            for fidelity_offset in (-0.005, 0.0, 0.005):
                evaluate(
                    snap(center["intent_threshold"] + intent_offset, 0.05),
                    snap(center["fidelity_threshold"] + fidelity_offset, 0.005),
                    "fine",
                )
    default = next(
        row.copy()
        for row in rows
        if (
            row["intent_threshold"] == DEFAULT_INTENT
            and row["fidelity_threshold"] == DEFAULT_FIDELITY
        )
    )
    references = {
        "default_thresholds": default,
        "always_keep": {
            "accuracy": float(df["off_correct"].mean()),
            "counterfactual_cost_ms": float(df["off_cost_ms"].mean()),
        },
        "skip_all_applicable": {
            "accuracy": float(
                np.where(app, df["wo_correct"], df["off_correct"]).mean()
            ),
            "counterfactual_cost_ms": float(
                np.where(app, df["wo_cost_ms"], df["off_cost_ms"]).mean()
            ),
        },
    }
    return (
        rank_results(rows),
        references,
        {
            "coarse_quantiles": list(COARSE_QUANTILES),
            "coarse_intent_candidates": intent_candidates,
            "coarse_fidelity_candidates": fidelity_candidates,
            "fine_intent_step": 0.05,
            "fine_fidelity_step": 0.005,
            "fine_centers": 3,
            "coarse_configurations": len(coarse),
            "fine_configurations": len(rows) - len(coarse),
        },
    )


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "analysis/adavig_gate_dataset/aligned_dataset.csv",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=root / "analysis/adavig_pre_gate_search"
    )
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    if df.empty:
        raise ValueError("Calibration dataset is empty")
    required = [
        INTENT,
        FIDELITY,
        "off_correct",
        "wo_correct",
        "off_cost_ms",
        "wo_cost_ms",
        "gate_applicable",
    ]
    if df[required].isna().any().any():
        raise ValueError(
            "Calibration data has missing features or outcomes; check trace alignment"
        )
    app = df["gate_applicable"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    results, references, search = search_thresholds(df, app)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "results.csv", index=False)
    summary = {
        "samples": len(df),
        "configurations": len(results),
        "best": results.iloc[0].to_dict(),
        "references": references,
        "search": search,
        "note": "Counterfactual estimate; validate with an online gated run.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
