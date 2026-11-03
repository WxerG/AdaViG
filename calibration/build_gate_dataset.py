#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(
    os.environ.get("ADAVIG_ROOT", Path(__file__).resolve().parents[1])
).expanduser()


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


BASE_RESULTS = _env_path(
    "ADAVIG_OFF_RESULTS_DIR",
    ROOT / "results/mathvision/mathvision_baseline",
)
WO_RESULTS = _env_path(
    "ADAVIG_WO_RESULTS_DIR",
    ROOT / "results/mathvision/mathvision_text-only",
)
BASE_METRICS = _env_path(
    "ADAVIG_OFF_METRICS_DIR",
    ROOT / "analysis/runtime_metrics/results/mathvision/mathvision_baseline",
)
OUT_DIR = _env_path("ADAVIG_GATE_DATASET_DIR", ROOT / "analysis/adavig_gate_dataset")
EPS = 1e-9

LAYER_SETS = {
    "L0": [0],
    "L7": [7],
    "L14": [14],
    "L21": [21],
    "L27": [27],
    "L21_27": [21, 27],
    "L14_21_27": [14, 21, 27],
    "L0_14_27": [0, 14, 27],
    "L0_7_14_21_27": [0, 7, 14, 21, 27],
}

STEP_WINDOWS = {
    "s1": [1],
    "s3": [3],
    "s5": [5],
    "s7": [7],
    "s9": [9],
    "s1_3": [1, 3],
    "s1_3_5": [1, 3, 5],
    "s1_3_5_7": [1, 3, 5, 7],
    "s1_3_5_7_9": [1, 3, 5, 7, 9],
    "s41_45_49": [41, 45, 49],
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _safe(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _load_eval(root: Path, prefix: str) -> pd.DataFrame:
    data = _load_json(root / "eval_summary.json")
    rows = []
    for row in data["details"]:
        if row.get("correctness") is None:
            raise ValueError(f"Missing evaluated outcome for sample {row['id']}")
        rows.append(
            {
                "sample_id": str(row["id"]),
                f"{prefix}_correct": int(bool(row.get("correctness"))),
                f"{prefix}_pred": row.get("pred_normalized"),
            }
        )
    return pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)


def _reasoning_cost(sample_dir: Path, trial_dir: Path | None) -> tuple[float, int]:
    rr = _load_json(sample_dir / "reasoning_result.json")
    total_ms = 0.0
    for item in rr.get("profiling", []):
        total_ms += _safe(item.get("elapsed_ms"))
    image_iters = 0
    if trial_dir is not None and (trial_dir / "step_trace.jsonl").exists():
        image_iters = sum(
            1
            for r in _load_jsonl(trial_dir / "step_trace.jsonl")
            if r.get("stage") == "image_generation"
        )
    return total_ms, image_iters


def _step_trace_info(trial_dir: Path) -> dict[str, Any]:
    rows = _load_jsonl(trial_dir / "step_trace.jsonl")
    first_text = next(
        (
            r
            for r in rows
            if r.get("iteration") == 1 and r.get("stage") == "text_decode"
        ),
        None,
    )
    return {
        "gate_applicable": bool(
            first_text and first_text.get("next_action") == "image"
        ),
        "first_next_action": None
        if first_text is None
        else first_text.get("next_action"),
        "n_image_iters": sum(1 for r in rows if r.get("stage") == "image_generation"),
    }


def _build_pre_features(trial_dir: Path) -> dict[str, float]:
    p = trial_dir / "attn_exact_segments.jsonl"
    rows = _load_jsonl(p)
    rows = [
        r
        for r in rows
        if r.get("stage") == "image_generation" and r.get("query_kind") == "vae"
    ]
    grouped: dict[tuple[int, int], dict[str, float]] = {}
    for r in rows:
        key = (int(r["layer"]), int(r["diffusion_step_idx"]))
        cur = grouped.setdefault(
            key,
            {"gen_text": 0.0, "input_text": 0.0, "orig_vit": 0.0, "orig_vae": 0.0},
        )
        mass = _safe(r.get("mass_mean"))
        kind = str(r.get("kind") or "")
        source = str(r.get("source") or "")
        if kind == "text" and source == "generated_text":
            cur["gen_text"] += mass
        elif kind == "text":
            cur["input_text"] += mass
        elif kind == "vit" and source == "input_image":
            cur["orig_vit"] += mass
        elif kind == "vae" and source == "input_image":
            cur["orig_vae"] += mass

    out: dict[str, float] = {}
    for lname, layers in LAYER_SETS.items():
        for wname, steps in STEP_WINDOWS.items():
            sel = [grouped[(l, s)] for l in layers for s in steps if (l, s) in grouped]
            if not sel:
                continue
            gen_text = float(np.mean([x["gen_text"] for x in sel]))
            input_text = float(np.mean([x["input_text"] for x in sel]))
            all_text = gen_text + input_text
            orig_vit = float(np.mean([x["orig_vit"] for x in sel]))
            orig_vae = float(np.mean([x["orig_vae"] for x in sel]))
            orig_total = orig_vit + orig_vae

            prefix = f"{lname}__{wname}"
            out[f"r_gen__{prefix}"] = gen_text / (gen_text + orig_total + EPS)
            out[f"r_all__{prefix}"] = all_text / (all_text + orig_total + EPS)
            out[f"log_gen__{prefix}"] = math.log((gen_text + EPS) / (orig_total + EPS))
            out[f"log_all__{prefix}"] = math.log((all_text + EPS) / (orig_total + EPS))
            out[f"gap_gen__{prefix}"] = gen_text - orig_total
            out[f"gap_all__{prefix}"] = all_text - orig_total
            out[f"orig_total__{prefix}"] = orig_total
            positive_vit = [x["orig_vit"] for x in sel if x["orig_vit"] > 0]
            out[f"orig_vit__{prefix}"] = (
                (float(np.mean(positive_vit)) if positive_vit else 0.0)
                if prefix == "L0_14_27__s5"
                else orig_vit
            )
            out[f"orig_vae__{prefix}"] = orig_vae
            out[f"gen_text__{prefix}"] = gen_text
            out[f"all_text__{prefix}"] = all_text
    return out


def _collect_dataset() -> pd.DataFrame:
    off = _load_eval(BASE_RESULTS, "off")
    wo = _load_eval(WO_RESULTS, "wo")
    rows = []
    for sample_dir in sorted(BASE_RESULTS.iterdir(), key=lambda p: p.name):
        if not sample_dir.is_dir():
            continue
        sid = sample_dir.name
        trial_dir = BASE_METRICS / sid / "trial_0"
        if not trial_dir.exists():
            continue
        total_ms, img_count = _reasoning_cost(sample_dir, trial_dir)
        w_total_ms, _ = _reasoning_cost(WO_RESULTS / sid, None)
        row = {
            "sample_id": sid,
            "off_cost_ms": total_ms,
            "wo_cost_ms": w_total_ms,
            "off_image_count": img_count,
            **_step_trace_info(trial_dir),
            **_build_pre_features(trial_dir),
        }
        rows.append(row)
    if not rows:
        raise ValueError("No aligned metric traces found")
    df = pd.DataFrame(rows)
    aligned = df.merge(off, on="sample_id", how="left", validate="one_to_one").merge(
        wo, on="sample_id", how="left", validate="one_to_one"
    )
    if aligned[["off_correct", "wo_correct"]].isna().any().any():
        raise ValueError("Some metric traces have no matching evaluated outcome")
    return aligned


def main() -> None:
    missing = [
        str(path)
        for path in (BASE_RESULTS, WO_RESULTS, BASE_METRICS)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required AdaViG input directories. "
            "Override them with ADAVIG_OFF_RESULTS_DIR, ADAVIG_WO_RESULTS_DIR, and "
            "ADAVIG_OFF_METRICS_DIR if your run layout differs.\n" + "\n".join(missing)
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _collect_dataset().sort_values("sample_id").reset_index(drop=True)
    df["trace_label"] = np.select(
        [
            (df["off_correct"] == 1) & (df["wo_correct"] == 0),
            (df["off_correct"] == 0) & (df["wo_correct"] == 1),
            (df["off_correct"] == 1) & (df["wo_correct"] == 1),
            (df["off_correct"] == 0) & (df["wo_correct"] == 0),
        ],
        ["helpful", "harmful", "neutral_keep", "neutral_skip"],
        default="unknown",
    )
    df["accuracy_delta"] = df["off_correct"] - df["wo_correct"]
    df.to_csv(OUT_DIR / "aligned_dataset.csv", index=False)
    summary = {
        "adavig_root": str(ROOT),
        "off_results_dir": str(BASE_RESULTS),
        "wo_results_dir": str(WO_RESULTS),
        "off_metrics_dir": str(BASE_METRICS),
        "output_dir": str(OUT_DIR),
        "n_samples": int(len(df)),
        "gate_applicable": int(df["gate_applicable"].fillna(False).astype(bool).sum()),
        "baseline_accuracy": float(df["off_correct"].mean()),
        "without_image_accuracy": float(df["wo_correct"].mean()),
        "label_counts": {
            str(k): int(v)
            for k, v in df["trace_label"].value_counts(dropna=False).to_dict().items()
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    report = [
        "# AdaViG Gate Dataset",
        "",
        f"- n_samples: `{summary['n_samples']}`",
        f"- gate_applicable: `{summary['gate_applicable']}`",
        f"- baseline_accuracy: `{summary['baseline_accuracy']:.6f}`",
        f"- without_image_accuracy: `{summary['without_image_accuracy']:.6f}`",
        "",
        "## Trace Labels",
    ]
    for key, value in summary["label_counts"].items():
        report.append(f"- {key}: `{value}`")
    (OUT_DIR / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
