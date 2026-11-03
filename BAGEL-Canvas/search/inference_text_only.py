#!/usr/bin/env python3
"""Text-only counterfactual reasoning without generated images."""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from PIL import Image
from tqdm.auto import tqdm

from . import base
from .common import ReasonerBase, nvtx_range
from .runner import run_inference
from data.data_utils import pil_img2rgb


class TextOnlyReasoner(ReasonerBase):
    @torch.no_grad()
    def run(
        self,
        inputs: List[Union[str, Image.Image]],
        system_prompt: Optional[str],
        max_iterations: int,
        verbose_iter: bool,
        verbose_image: bool,
        inference_kwargs: Dict[str, Any],
        sample_id: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # Retain the attention-log schema for the single trajectory.
        trial_id = 0
        reasoning_steps: List[Dict[str, Any]] = []
        profiling_events: List[Dict[str, Any]] = []

        gen_context = self.inferencer.init_gen_context()

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if system_prompt:
                prompt_text = system_prompt.strip()
                if prompt_text:
                    token_count = self._count_text_tokens(prompt_text)
                    self._set_trace(
                        gen_context,
                        sample_id=sample_id,
                        trial_id=trial_id,
                        iteration="pre",
                        stage="text_prefill",
                        context_label="gen",
                        label="system_prompt",
                        query_kind="text",
                    )
                    with nvtx_range("text_prefill:system_prompt:gen"):
                        gen_context = self._profile_call(
                            profiling_events,
                            "text_prefill",
                            self.inferencer.update_context_text,
                            prompt_text,
                            gen_context,
                            segment_meta={
                                "context": "gen",
                                "label": "system_prompt",
                                "iteration": "pre",
                                "source": "system_prompt",
                            },
                            **self._text_prefill_meta(
                                "gen", "system_prompt", "pre", token_count
                            ),
                        )

            for input_item in inputs:
                if isinstance(input_item, str):
                    content = input_item.strip()
                    if content:
                        token_count = self._count_text_tokens(content)
                        self._set_trace(
                            gen_context,
                            sample_id=sample_id,
                            trial_id=trial_id,
                            iteration="pre",
                            stage="text_prefill",
                            context_label="gen",
                            label="input_text",
                            query_kind="text",
                        )
                        with nvtx_range("text_prefill:input_text:gen"):
                            gen_context = self._profile_call(
                                profiling_events,
                                "text_prefill",
                                self.inferencer.update_context_text,
                                content,
                                gen_context,
                                segment_meta={
                                    "context": "gen",
                                    "label": "input_text",
                                    "iteration": "pre",
                                    "source": "input_text",
                                },
                                **self._text_prefill_meta(
                                    "gen", "input_text", "pre", token_count
                                ),
                            )
                elif isinstance(input_item, Image.Image):
                    processed_image = self.inferencer.vae_transform.resize_transform(
                        pil_img2rgb(input_item)
                    )
                    prev_tokens_gen = self._kv_length(gen_context)
                    self._set_trace(
                        gen_context,
                        sample_id=sample_id,
                        trial_id=trial_id,
                        iteration="pre",
                        stage="visual_prefill",
                        context_label="gen",
                        label="input_image",
                        query_kind="vit",
                    )
                    with nvtx_range("visual_prefill:input_image:gen"):
                        gen_context = self._profile_call(
                            profiling_events,
                            "visual_prefill",
                            self.inferencer.update_context_image,
                            processed_image,
                            gen_context,
                            segment_meta={
                                "context": "gen",
                                "label": "input_image",
                                "iteration": "pre",
                                "source": "input_image",
                            },
                            **self._visual_prefill_meta(
                                "gen", "input_image", "pre", prev_tokens_gen
                            ),
                            vae=True,
                            vit=True,
                        )
                else:
                    raise ValueError(f"Unsupported input type: {type(input_item)}")

            current_mode = "text"
            with tqdm(
                total=max_iterations,
                desc="Reasoning Steps",
                leave=False,
                disable=(base.d_rank() != 0 or not verbose_iter),
            ) as pbar:
                iteration = 0
                completed_iterations = 0
                # Text and successfully generated images consume the budget.
                # Skipped or gate-aborted images only advance the trace index.
                while completed_iterations < max_iterations:
                    if current_mode == "text":
                        completed_iterations += 1
                        pbar.update(1)
                        pbar.set_description(
                            f"Step {completed_iterations}/{max_iterations}: Generating text"
                        )
                        self._set_trace(
                            gen_context,
                            sample_id=sample_id,
                            trial_id=trial_id,
                            iteration=iteration + 1,
                            stage="text_decode",
                            context_label="gen",
                            label="decode",
                            query_kind="text",
                        )
                        with nvtx_range("text_decode"):
                            decoded_text, next_action, decode_stats = (
                                self._generate_text_segment(
                                    gen_context, inference_kwargs
                                )
                            )
                        decode_event = {
                            "stage": "text_decode",
                            "iteration": iteration + 1,
                            "sample_id": str(sample_id)
                            if sample_id is not None
                            else None,
                            "trial_id": trial_id,
                            **decode_stats,
                        }
                        profiling_events.append(decode_event)
                        if self.metric_dumper.should_write_jsonl(sample_id):
                            self.metric_dumper.write_jsonl(
                                "step_trace",
                                {
                                    "sample_id": str(sample_id),
                                    "trial_id": trial_id,
                                    "iteration": iteration + 1,
                                    "stage": "text_decode",
                                },
                                {
                                    "next_action": next_action,
                                    "text_tokens": decode_stats.get("text_tokens"),
                                    "total_tokens": decode_stats.get("total_tokens"),
                                    "elapsed_ms": decode_stats.get("elapsed_ms"),
                                },
                            )

                        text_token_count = self._count_text_tokens(decoded_text)
                        self._set_trace(
                            gen_context,
                            sample_id=sample_id,
                            trial_id=trial_id,
                            iteration=iteration + 1,
                            stage="text_prefill",
                            context_label="gen",
                            label="generated_text",
                            query_kind="text",
                        )
                        with nvtx_range("text_prefill:generated_text:gen"):
                            gen_context = self._profile_call(
                                profiling_events,
                                "text_prefill",
                                self.inferencer.update_context_text,
                                decoded_text,
                                gen_context,
                                segment_meta={
                                    "context": "gen",
                                    "label": "generated_text",
                                    "iteration": iteration + 1,
                                    "source": "generated_text",
                                },
                                **self._text_prefill_meta(
                                    "gen",
                                    "generated_text",
                                    iteration + 1,
                                    text_token_count,
                                ),
                            )

                        reasoning_steps.append(
                            {
                                "type": "text",
                                "content": decoded_text,
                                "iteration": iteration + 1,
                            }
                        )

                        if next_action == "end":
                            break
                        else:
                            current_mode = "image" if next_action == "image" else "text"

                    elif current_mode == "image":
                        pbar.set_description(
                            f"Step {completed_iterations}/{max_iterations}: Skipping image"
                        )
                        image_event = {
                            "stage": "image_generation",
                            "iteration": iteration + 1,
                            "sample_id": str(sample_id)
                            if sample_id is not None
                            else None,
                            "trial_id": trial_id,
                            "elapsed_ms": 0.0,
                            "skipped": True,
                        }
                        profiling_events.append(image_event)
                        if self.metric_dumper.should_write_jsonl(sample_id):
                            self.metric_dumper.write_jsonl(
                                "step_trace",
                                {
                                    "sample_id": str(sample_id),
                                    "trial_id": trial_id,
                                    "iteration": iteration + 1,
                                    "stage": "image_generation",
                                },
                                {"elapsed_ms": 0.0, "skipped": True},
                            )
                        reasoning_steps.append(
                            {
                                "type": "image",
                                "content": None,
                                "iteration": iteration + 1,
                                "skipped": True,
                            }
                        )
                        current_mode = "text"

                    iteration += 1

        return reasoning_steps, profiling_events


def main(argv=None):
    run_inference(TextOnlyReasoner, argv)


if __name__ == "__main__":
    main()
