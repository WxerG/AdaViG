"""Runtime metric dump helpers for controlled instrumentation.

Design goals:
- Keep the hot path simple.
- Gate expensive tensor dumps behind an env flag and selected-case list.
- Separate lightweight JSONL summaries from heavier `.pt` tensor dumps.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set

import torch

_SENTINEL = object()  # signals the I/O thread to stop


def _io_worker(q: queue.Queue) -> None:
    """Background thread: drain the queue and write lines to disk."""
    while True:
        item = q.get()
        if item is _SENTINEL:
            q.task_done()
            break
        path, line = item
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass  # never crash the background thread
        finally:
            q.task_done()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes"}


def _parse_int_set(raw: str) -> Set[int]:
    out: Set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.add(int(token))
        except ValueError:
            continue
    return out


def _parse_name_set(raw: str) -> Set[str]:
    out: Set[str] = set()
    for token in raw.split(","):
        token = token.strip()
        if token:
            out.add(token)
    return out


@dataclass(frozen=True)
class TraceKey:
    sample_id: str
    trial_id: int


class RuntimeMetricDumper:
    def __init__(self) -> None:
        self.enabled = _env_flag("BAGEL_METRIC_DUMP", "0")
        self.jsonl_all = _env_flag("BAGEL_METRIC_JSONL_ALL", "1")
        self.save_tensors = _env_flag("BAGEL_METRIC_SAVE_TENSORS", "1")
        self.jsonl_allowlist = _parse_name_set(
            os.environ.get("BAGEL_METRIC_JSONL_ALLOWLIST", "")
        )
        self.case_view = os.environ.get("BAGEL_METRIC_CASE_VIEW", "on").strip().lower()
        self.dump_dir = Path(
            os.environ.get(
                "BAGEL_METRIC_DUMP_DIR",
                str(Path(__file__).resolve().parents[1] / "analysis/runtime_metrics"),
            )
        )
        self.run_group = os.environ.get("BAGEL_METRIC_RUN_GROUP", "").strip()
        self.run_id = os.environ.get("BAGEL_METRIC_RUN_ID", "").strip()
        self.case_file = os.environ.get(
            "BAGEL_METRIC_CASES_JSON",
            str(Path(__file__).resolve().parents[1] / "analysis/selected_cases.json"),
        )
        self.full_layers = _parse_int_set(
            os.environ.get("BAGEL_METRIC_FULL_LAYERS", "0,7,14,21,27")
        )
        self._selected_trials: Set[TraceKey] = set()
        self._selected_samples: Set[str] = set()
        self._loaded = False

        # Async I/O: background thread drains the write queue so GPU isn't blocked.
        self._io_queue: queue.Queue = queue.Queue(maxsize=4096)
        self._io_thread = threading.Thread(
            target=_io_worker, args=(self._io_queue,), daemon=True, name="metric-io"
        )
        self._io_thread.start()

    def flush(self) -> None:
        """Block until all queued writes have been flushed to disk."""
        self._io_queue.join()

    def shutdown(self) -> None:
        """Flush and stop the background I/O thread."""
        self._io_queue.put(_SENTINEL)
        self._io_thread.join()

    def configure_run(
        self, *, run_group: Optional[str] = None, run_id: Optional[str] = None
    ) -> None:
        if run_group is not None:
            self.run_group = str(run_group).strip()
        if run_id is not None:
            self.run_id = str(run_id).strip()

    @staticmethod
    def _is_bucket_mapping(obj: Any) -> bool:
        return isinstance(obj, dict) and all(isinstance(v, list) for v in obj.values())

    def _extract_bucket_mapping(
        self, payload: Dict[str, Any], bucket_key: str
    ) -> Dict[str, list]:
        raw = payload.get(bucket_key)
        view = self.case_view

        # New format:
        # {
        #   "sample_buckets": {"on": {...}, "off": {...}},
        #   "trial_buckets": {"on": {...}, "off": {...}}
        # }
        if isinstance(raw, dict):
            if view in {"on", "off"} and self._is_bucket_mapping(raw.get(view)):
                return dict(raw.get(view))
            if view == "all":
                merged: Dict[str, list] = {}
                for view_key in ("on", "off"):
                    view_mapping = raw.get(view_key)
                    if not self._is_bucket_mapping(view_mapping):
                        continue
                    for bucket, items in view_mapping.items():
                        merged.setdefault(bucket, [])
                        merged[bucket].extend(items if isinstance(items, list) else [])
                if merged:
                    return merged

        # Backward-compatible format:
        # {
        #   "comparison_views": {"on": {...}, "off": {...}}
        # }
        views = payload.get("comparison_views")
        if isinstance(views, dict):
            if view in {"on", "off"}:
                view_payload = views.get(view)
                if isinstance(view_payload, dict) and self._is_bucket_mapping(
                    view_payload.get(bucket_key)
                ):
                    return dict(view_payload.get(bucket_key))
            if view == "all":
                merged = {}
                for view_key in ("on", "off"):
                    view_payload = views.get(view_key)
                    if not isinstance(view_payload, dict):
                        continue
                    view_mapping = view_payload.get(bucket_key)
                    if not self._is_bucket_mapping(view_mapping):
                        continue
                    for bucket, items in view_mapping.items():
                        merged.setdefault(bucket, [])
                        merged[bucket].extend(items if isinstance(items, list) else [])
                if merged:
                    return merged

        # Legacy flat format:
        # {
        #   "sample_buckets": {...},
        #   "trial_buckets": {...}
        # }
        if self._is_bucket_mapping(raw):
            return dict(raw)

        return {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.enabled:
            return
        path = Path(self.case_file)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        for items in self._extract_bucket_mapping(payload, "sample_buckets").values():
            if not isinstance(items, list):
                continue
            for item in items:
                sid = str(item.get("sample_id", "")).strip()
                if sid:
                    self._selected_samples.add(sid)
                for tid in item.get("trial_indices") or []:
                    try:
                        self._selected_trials.add(
                            TraceKey(sample_id=sid, trial_id=int(tid))
                        )
                    except Exception:
                        continue

        for items in self._extract_bucket_mapping(payload, "trial_buckets").values():
            if not isinstance(items, list):
                continue
            for item in items:
                sid = str(item.get("sample_id", "")).strip()
                if sid:
                    self._selected_samples.add(sid)
                for tid in item.get("trial_indices") or []:
                    try:
                        self._selected_trials.add(
                            TraceKey(sample_id=sid, trial_id=int(tid))
                        )
                    except Exception:
                        continue

    def sample_selected(self, sample_id: Optional[str]) -> bool:
        self._ensure_loaded()
        if not self.enabled or not sample_id:
            return False
        return str(sample_id) in self._selected_samples

    def trial_selected(self, sample_id: Optional[str], trial_id: Optional[int]) -> bool:
        self._ensure_loaded()
        if not self.enabled or sample_id is None or trial_id is None:
            return False
        return (
            TraceKey(sample_id=str(sample_id), trial_id=int(trial_id))
            in self._selected_trials
        )

    def should_dump_full(
        self,
        sample_id: Optional[str],
        trial_id: Optional[int],
        layer_idx: Optional[int],
    ) -> bool:
        if layer_idx is None or layer_idx not in self.full_layers:
            return False
        return self.trial_selected(sample_id, trial_id)

    def should_write_jsonl(self, sample_id: Optional[str]) -> bool:
        self._ensure_loaded()
        if not self.enabled or not sample_id:
            return False
        if self.jsonl_all:
            return True
        return str(sample_id) in self._selected_samples

    def jsonl_kind_enabled(self, kind: str) -> bool:
        if not self.jsonl_allowlist:
            return True
        return str(kind) in self.jsonl_allowlist

    def should_write_jsonl_kind(self, kind: str, sample_id: Optional[str]) -> bool:
        return self.should_write_jsonl(sample_id) and self.jsonl_kind_enabled(kind)

    def _trace_dir(self, trace: Dict[str, Any]) -> Optional[Path]:
        if not self.enabled:
            return None
        sample_id = trace.get("sample_id")
        trial_id = trace.get("trial_id")
        if sample_id is None:
            return None
        base = self.dump_dir
        if self.run_group:
            base = base / self.run_group
        if self.run_id:
            base = base / self.run_id
        base = base / str(sample_id)
        if trial_id is None:
            return base
        return base / f"trial_{int(trial_id)}"

    def write_jsonl(
        self, kind: str, trace: Dict[str, Any], payload: Dict[str, Any]
    ) -> Optional[Path]:
        if not self.should_write_jsonl_kind(kind, trace.get("sample_id")):
            return None
        target = self._trace_dir(trace)
        if target is None:
            return None
        path = target / f"{kind}.jsonl"
        record = {**trace, **payload}
        line = json.dumps(record, ensure_ascii=False) + "\n"
        self._io_queue.put((path, line))  # non-blocking; background thread writes
        return path

    def save_tensor(
        self,
        kind: str,
        trace: Dict[str, Any],
        payload: Dict[str, Any],
        *,
        layer_idx: Optional[int] = None,
        full_only: bool = False,
    ) -> Optional[Path]:
        if not self.enabled or not self.save_tensors:
            return None
        if full_only and not self.should_dump_full(
            trace.get("sample_id"), trace.get("trial_id"), layer_idx
        ):
            return None
        target = self._trace_dir(trace)
        if target is None:
            return None
        target.mkdir(parents=True, exist_ok=True)
        name = kind
        if layer_idx is not None:
            name = f"{name}_layer{int(layer_idx):02d}"
        iteration = trace.get("iteration")
        stage = trace.get("stage", "unknown")
        suffix = f"{stage}"
        if iteration is not None:
            suffix += f"_step{iteration}"
        diffusion_step_idx = trace.get("diffusion_step_idx")
        if stage == "image_generation" and diffusion_step_idx is not None:
            suffix += f"_diff{int(diffusion_step_idx):02d}"
        path = target / f"{name}_{suffix}.pt"
        torch.save(payload, path)
        return path

    def save_metric_tensor(
        self,
        kind: str,
        trace: Dict[str, Any],
        payload: Dict[str, Any],
        *,
        require_jsonl_gate: bool = True,
    ) -> Optional[Path]:
        if not self.enabled or not self.save_tensors:
            return None
        if require_jsonl_gate and not self.should_write_jsonl(trace.get("sample_id")):
            return None
        target = self._trace_dir(trace)
        if target is None:
            return None
        target.mkdir(parents=True, exist_ok=True)
        stage = trace.get("stage", "unknown")
        iteration = trace.get("iteration")
        suffix = f"{stage}"
        if iteration is not None:
            suffix += f"_step{iteration}"
        diffusion_step_idx = trace.get("diffusion_step_idx")
        if stage == "image_generation" and diffusion_step_idx is not None:
            suffix += f"_diff{int(diffusion_step_idx):02d}"
        path = target / f"{kind}_{suffix}.pt"
        torch.save(payload, path)
        return path


_DUMPER = RuntimeMetricDumper()


def get_metric_dumper() -> RuntimeMetricDumper:
    return _DUMPER
