"""Early diffusion gate for AdaViG.

The serialized feature names are shared with the offline calibration tools.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

EPS = 1e-9
DECISION_STEP = 5
ATTENTION_LAYERS = (0, 14, 27)


def _threshold(name: str, default: float) -> float:
    value = float(os.environ.get(name) or default)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return value


class OnlineGateAbortImageGeneration(RuntimeError):
    """Stop denoising and resume reasoning without the generated image."""


class OnlineGateController:
    def __init__(self) -> None:
        self.policy = os.environ.get("BAGEL_ONLINE_GATE_POLICY", "").strip().lower()
        if self.policy not in {"", "adavig_pre"}:
            raise ValueError("BAGEL_ONLINE_GATE_POLICY must be empty or 'adavig_pre'")
        self.enabled = self.policy == "adavig_pre"
        self.metric_path = os.environ.get("BAGEL_ONLINE_GATE_METRIC_PATH", "").strip()
        self.adavig_pre_r_all_thr = _threshold("ADAVIG_PRE_R_ALL_THR", 0.6)
        self.adavig_pre_orig_vit_thr = _threshold("ADAVIG_PRE_ORIG_VIT_THR", 0.01)
        self._segment_rows: dict[tuple, list[dict[str, Any]]] = {}
        self._pre_decisions: dict[tuple, dict[str, Any]] = {}

    @staticmethod
    def _key(trace: dict[str, Any]) -> tuple | None:
        if trace.get("sample_id") is None or trace.get("trial_id") is None:
            return None
        try:
            return (
                str(trace["sample_id"]),
                int(trace["trial_id"]),
                trace.get("iteration"),
            )
        except (TypeError, ValueError):
            return None

    def exact_attention_needed(self, trace: dict | None, layer_idx: int) -> bool:
        if not self.enabled or not trace:
            return False
        if trace.get("context") in {"cfg_text", "cfg_img"}:
            return False
        return (
            trace.get("stage") == "image_generation"
            and trace.get("query_kind") == "vae"
            and trace.get("diffusion_step_idx") == DECISION_STEP
            and layer_idx in ATTENTION_LAYERS
        )

    def record_segment_records(
        self, trace: dict, layer_idx: int, segment_records: list[dict]
    ) -> None:
        if not self.exact_attention_needed(trace, layer_idx):
            return
        key = self._key(trace)
        if key is not None:
            self._segment_rows.setdefault(key, []).extend(
                dict(row, layer=int(layer_idx)) for row in segment_records
            )

    def maybe_decide_pre(self, trace: dict) -> dict | None:
        if not self.enabled or trace.get("diffusion_step_idx") != DECISION_STEP:
            return None
        key = self._key(trace)
        if key is None:
            return None
        if key in self._pre_decisions:
            return self._pre_decisions[key]
        rows = self._segment_rows.get(key)
        if not rows:
            return None

        def mass(layer: int, kind: str, source: str | None = None) -> float:
            return sum(
                float(row.get("mass_mean") or 0.0)
                for row in rows
                if row.get("layer") == layer
                and row.get("kind") == kind
                and (source is None or row.get("source") == source)
            )

        text = mass(27, "text")
        image = mass(27, "vit", "input_image") + mass(27, "vae", "input_image")
        intent = text / (text + image + EPS)
        # Preserve the calibration convention: average positive input-ViT masses.
        visual_masses = [
            mass(layer, "vit", "input_image") for layer in ATTENTION_LAYERS
        ]
        visual_masses = [value for value in visual_masses if value > 0]
        fidelity = sum(visual_masses) / max(len(visual_masses), 1)
        decision = {
            "sample_id": key[0],
            "trial_id": key[1],
            "iteration": key[2],
            "kind": "pre_decision",
            "policy": self.policy,
            "r_all__L27__s5": intent,
            "orig_vit__L0_14_27__s5": fidelity,
            "choose_off": intent > self.adavig_pre_r_all_thr
            or fidelity > self.adavig_pre_orig_vit_thr,
            "pre_gate_r_all_thr": self.adavig_pre_r_all_thr,
            "pre_gate_orig_vit_thr": self.adavig_pre_orig_vit_thr,
        }
        self._pre_decisions[key] = decision
        if self.metric_path:
            path = Path(self.metric_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(decision) + "\n")
        return decision

    def finalize_trial(self, sample_id: str | None, trial_id: int | None) -> None:
        if sample_id is None or trial_id is None:
            return
        for store in (self._segment_rows, self._pre_decisions):
            for key in list(store):
                if key[:2] == (str(sample_id), int(trial_id)):
                    del store[key]


_CONTROLLER = OnlineGateController()


def get_online_gate_controller() -> OnlineGateController:
    return _CONTROLLER
