"""Text decoding and trace metadata shared by both reasoning modes."""

from contextlib import contextmanager
import time

import torch

from .base import InterleaveReasoner
from runtime_metrics import get_metric_dumper


@contextmanager
def nvtx_range(message):
    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


class ReasonerBase(InterleaveReasoner):
    def __init__(self, inferencer):
        super().__init__(inferencer)
        self.metric_dumper = get_metric_dumper()

    def _set_trace(
        self,
        gen_context,
        *,
        sample_id,
        trial_id,
        iteration,
        stage,
        context_label=None,
        label=None,
        query_kind=None,
        extra_trace_fields=None,
    ):
        fields = dict(extra_trace_fields or {})
        fields.update(
            sample_id=str(sample_id) if sample_id is not None else None,
            trial_id=trial_id,
            iteration=iteration,
            stage=stage,
            context=context_label,
            label=label,
            query_kind=query_kind,
        )
        self.inferencer.update_trace(gen_context, **fields)

    def _generate_text_segment(self, gen_context, inference_kwargs):
        start = time.perf_counter()
        generation = dict(
            do_sample=inference_kwargs.get("do_sample", True),
            temperature=inference_kwargs.get("text_temperature", 0.7),
            return_ids=True,
        )
        ids = self.inferencer.gen_text(gen_context, **generation)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        text_ids = list(ids[1:-2])
        token_texts = [
            self.tokenizer.decode([token], skip_special_tokens=False)
            for token in text_ids
        ]
        text = "".join(token_texts).strip()
        stats = {
            "elapsed_ms": elapsed_ms,
            "text_tokens": len(text_ids),
            "total_tokens": len(ids),
        }
        return text, self.action_token_map.get(ids[-1], "undefined"), stats
