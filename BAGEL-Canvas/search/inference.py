#!/usr/bin/env python3
"""Visual reasoning with optional AdaViG early diffusion gating."""

import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from . import base
from .common import ReasonerBase, nvtx_range
from .runner import run_inference
from data.data_utils import pil_img2rgb
from online_gate import OnlineGateAbortImageGeneration, get_online_gate_controller


class VisualReasoner(ReasonerBase):
    def __init__(self, inferencer):
        super().__init__(inferencer)
        self._generated_image_swap_cache = None

    @staticmethod
    def _generated_image_intervention_mode() -> str:
        return (
            str(os.environ.get("BAGEL_GENERATED_IMAGE_INTERVENTION", "") or "")
            .strip()
            .lower()
        )

    @staticmethod
    def _generated_image_intervention_trial_filter() -> Optional[set[int]]:
        raw = str(
            os.environ.get("BAGEL_GENERATED_IMAGE_INTERVENTION_TRIALS", "") or ""
        ).strip()
        if not raw:
            return None
        out = set()
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.add(int(tok))
            except ValueError:
                continue
        return out or None

    def _load_generated_image_swap_map(self):
        if self._generated_image_swap_cache is not None:
            return self._generated_image_swap_cache
        path = str(os.environ.get("BAGEL_GENERATED_IMAGE_SWAP_JSON", "") or "").strip()
        if not path:
            self._generated_image_swap_cache = {}
            return self._generated_image_swap_cache
        try:
            with open(path) as f:
                self._generated_image_swap_cache = json.load(f)
        except Exception:
            self._generated_image_swap_cache = {}
        return self._generated_image_swap_cache

    def _apply_generated_image_intervention(
        self,
        image: Image.Image,
        *,
        sample_id: Optional[str],
        trial_id: Optional[int],
    ) -> tuple[Optional[Image.Image], str]:
        mode = self._generated_image_intervention_mode()
        if not mode:
            return image, "original"
        allowed_trials = self._generated_image_intervention_trial_filter()
        if allowed_trials is not None and trial_id not in allowed_trials:
            return image, "original"

        if mode == "skip":
            return None, "skip"
        if mode == "zero":
            return Image.new("RGB", image.size, (0, 0, 0)), "zero"
        if mode == "noise":
            seed = int(os.environ.get("BAGEL_GENERATED_IMAGE_NOISE_SEED", "1234"))
            rng = np.random.default_rng(seed + int(trial_id or 0))
            arr = rng.integers(
                0, 256, size=(image.size[1], image.size[0], 3), dtype=np.uint8
            )
            return Image.fromarray(arr, mode="RGB"), "noise"
        if mode == "swap":
            swap_map = self._load_generated_image_swap_map()
            sample_key = str(sample_id) if sample_id is not None else ""
            trial_key = str(trial_id) if trial_id is not None else ""
            src = (
                swap_map.get(sample_key, {}).get(trial_key)
                or swap_map.get(trial_key)
                or swap_map.get(f"{sample_key}:{trial_key}")
            )
            if src:
                try:
                    return pil_img2rgb(Image.open(src)), "swap"
                except Exception:
                    pass
        return image, "original"

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
        cfg_text_context = self.inferencer.init_gen_context()
        cfg_img_context = self.inferencer.init_gen_context()
        gate = get_online_gate_controller()

        image_shapes = (512, 512)
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
                    self._set_trace(
                        cfg_img_context,
                        sample_id=sample_id,
                        trial_id=trial_id,
                        iteration="pre",
                        stage="text_prefill",
                        context_label="cfg_img",
                        label="system_prompt",
                        query_kind="text",
                    )
                    with nvtx_range("text_prefill:system_prompt:cfg_img"):
                        cfg_img_context = self._profile_call(
                            profiling_events,
                            "text_prefill",
                            self.inferencer.update_context_text,
                            prompt_text,
                            cfg_img_context,
                            segment_meta={
                                "context": "cfg_img",
                                "label": "system_prompt",
                                "iteration": "pre",
                                "source": "system_prompt",
                            },
                            **self._text_prefill_meta(
                                "cfg_img", "system_prompt", "pre", token_count
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
                        self._set_trace(
                            cfg_img_context,
                            sample_id=sample_id,
                            trial_id=trial_id,
                            iteration="pre",
                            stage="text_prefill",
                            context_label="cfg_img",
                            label="input_text",
                            query_kind="text",
                        )
                        with nvtx_range("text_prefill:input_text:cfg_img"):
                            cfg_img_context = self._profile_call(
                                profiling_events,
                                "text_prefill",
                                self.inferencer.update_context_text,
                                content,
                                cfg_img_context,
                                segment_meta={
                                    "context": "cfg_img",
                                    "label": "input_text",
                                    "iteration": "pre",
                                    "source": "input_text",
                                },
                                **self._text_prefill_meta(
                                    "cfg_img", "input_text", "pre", token_count
                                ),
                            )
                elif isinstance(input_item, Image.Image):
                    processed_image = self.inferencer.vae_transform.resize_transform(
                        pil_img2rgb(input_item)
                    )
                    image_shapes = processed_image.size[::-1]
                    prev_tokens_gen = self._kv_length(gen_context)
                    self._set_trace(
                        gen_context,
                        sample_id=sample_id,
                        trial_id=trial_id,
                        iteration="pre",
                        stage="visual_prefill",
                        context_label="gen",
                        label="input_image",
                        query_kind="image",
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
                    prev_tokens_cfg = self._kv_length(cfg_text_context)
                    self._set_trace(
                        cfg_text_context,
                        sample_id=sample_id,
                        trial_id=trial_id,
                        iteration="pre",
                        stage="visual_prefill",
                        context_label="cfg_text",
                        label="input_image",
                        query_kind="image",
                    )
                    with nvtx_range("visual_prefill:input_image:cfg_text"):
                        cfg_text_context = self._profile_call(
                            profiling_events,
                            "visual_prefill",
                            self.inferencer.update_context_image,
                            processed_image,
                            cfg_text_context,
                            segment_meta={
                                "context": "cfg_text",
                                "label": "input_image",
                                "iteration": "pre",
                                "source": "input_image",
                            },
                            **self._visual_prefill_meta(
                                "cfg_text", "input_image", "pre", prev_tokens_cfg
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
                            extra_trace_fields={
                                "has_generated_image_context": any(
                                    step["type"] == "image" for step in reasoning_steps
                                )
                            },
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
                        self._set_trace(
                            cfg_img_context,
                            sample_id=sample_id,
                            trial_id=trial_id,
                            iteration=iteration + 1,
                            stage="text_prefill",
                            context_label="cfg_img",
                            label="generated_text",
                            query_kind="text",
                        )
                        with nvtx_range("text_prefill:generated_text:cfg_img"):
                            cfg_img_context = self._profile_call(
                                profiling_events,
                                "text_prefill",
                                self.inferencer.update_context_text,
                                decoded_text,
                                cfg_img_context,
                                segment_meta={
                                    "context": "cfg_img",
                                    "label": "generated_text",
                                    "iteration": iteration + 1,
                                    "source": "generated_text",
                                },
                                **self._text_prefill_meta(
                                    "cfg_img",
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
                            f"Step {completed_iterations}/{max_iterations}: Generating image"
                        )
                        try:
                            gen_image_kwargs = {
                                k: v
                                for k, v in inference_kwargs.items()
                                if k.startswith("cfg_")
                                or k
                                in [
                                    "timestep_shift",
                                    "num_timesteps",
                                    "enable_taylorseer",
                                ]
                            }
                            early_stop_raw = str(
                                os.environ.get(
                                    "BAGEL_IMAGEGEN_EARLY_STOP_SAMPLED_STEPS", ""
                                )
                                or ""
                            ).strip()
                            if early_stop_raw:
                                try:
                                    gen_image_kwargs["early_stop_sampled_steps"] = int(
                                        early_stop_raw
                                    )
                                except ValueError:
                                    pass
                            self._set_trace(
                                gen_context,
                                sample_id=sample_id,
                                trial_id=trial_id,
                                iteration=iteration + 1,
                                stage="image_generation",
                                context_label="gen",
                                label="generated_image",
                                query_kind="vae",
                            )
                            self._set_trace(
                                cfg_text_context,
                                sample_id=sample_id,
                                trial_id=trial_id,
                                iteration=iteration + 1,
                                stage="image_generation",
                                context_label="cfg_text",
                                label="generated_image",
                                query_kind="vae",
                            )
                            self._set_trace(
                                cfg_img_context,
                                sample_id=sample_id,
                                trial_id=trial_id,
                                iteration=iteration + 1,
                                stage="image_generation",
                                context_label="cfg_img",
                                label="generated_image",
                                query_kind="vae",
                            )
                            image_start = time.perf_counter()
                            with nvtx_range("image_generation"):
                                generated_image = self.inferencer.gen_image(
                                    image_shapes,
                                    gen_context,
                                    cfg_text_precontext=cfg_text_context,
                                    cfg_img_precontext=cfg_img_context,
                                    verbose=(verbose_image and base.d_rank() == 0),
                                    **gen_image_kwargs,
                                )
                            completed_iterations += 1
                            pbar.update(1)
                            image_elapsed_ms = (
                                time.perf_counter() - image_start
                            ) * 1000.0
                            image_event = {
                                "stage": "image_generation",
                                "iteration": iteration + 1,
                                "sample_id": str(sample_id)
                                if sample_id is not None
                                else None,
                                "trial_id": trial_id,
                                "elapsed_ms": image_elapsed_ms,
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
                                    {"elapsed_ms": image_elapsed_ms},
                                )
                            reasoning_steps.append(
                                {
                                    "type": "image",
                                    "content": generated_image,
                                    "iteration": iteration + 1,
                                }
                            )
                            context_image, intervention_label = (
                                self._apply_generated_image_intervention(
                                    pil_img2rgb(generated_image),
                                    sample_id=sample_id,
                                    trial_id=trial_id,
                                )
                            )
                            image_event["context_intervention"] = intervention_label
                            if context_image is not None:
                                processed_image = (
                                    self.inferencer.vae_transform.resize_transform(
                                        context_image
                                    )
                                )
                                prev_tokens_gen = self._kv_length(gen_context)
                                self._set_trace(
                                    gen_context,
                                    sample_id=sample_id,
                                    trial_id=trial_id,
                                    iteration=iteration + 1,
                                    stage="visual_prefill",
                                    context_label="gen",
                                    label="generated_image",
                                    query_kind="image",
                                )
                                with nvtx_range("visual_prefill:generated_image:gen"):
                                    gen_context = self._profile_call(
                                        profiling_events,
                                        "visual_prefill",
                                        self.inferencer.update_context_image,
                                        processed_image,
                                        gen_context,
                                        segment_meta={
                                            "context": "gen",
                                            "label": "generated_image",
                                            "iteration": iteration + 1,
                                            "source": "generated_image",
                                        },
                                        **self._visual_prefill_meta(
                                            "gen",
                                            "generated_image",
                                            iteration + 1,
                                            prev_tokens_gen,
                                        ),
                                        vae=True,
                                        vit=True,
                                    )
                                prev_tokens_cfg = self._kv_length(cfg_text_context)
                                self._set_trace(
                                    cfg_text_context,
                                    sample_id=sample_id,
                                    trial_id=trial_id,
                                    iteration=iteration + 1,
                                    stage="visual_prefill",
                                    context_label="cfg_text",
                                    label="generated_image",
                                    query_kind="image",
                                )
                                with nvtx_range(
                                    "visual_prefill:generated_image:cfg_text"
                                ):
                                    cfg_text_context = self._profile_call(
                                        profiling_events,
                                        "visual_prefill",
                                        self.inferencer.update_context_image,
                                        processed_image,
                                        cfg_text_context,
                                        segment_meta={
                                            "context": "cfg_text",
                                            "label": "generated_image",
                                            "iteration": iteration + 1,
                                            "source": "generated_image",
                                        },
                                        **self._visual_prefill_meta(
                                            "cfg_text",
                                            "generated_image",
                                            iteration + 1,
                                            prev_tokens_cfg,
                                        ),
                                        vae=True,
                                        vit=True,
                                    )
                            current_mode = "text"
                        except OnlineGateAbortImageGeneration as exc:
                            image_elapsed_ms = (
                                time.perf_counter() - image_start
                            ) * 1000.0
                            profiling_events.append(
                                {
                                    "stage": "image_generation",
                                    "iteration": iteration + 1,
                                    "sample_id": str(sample_id)
                                    if sample_id is not None
                                    else None,
                                    "trial_id": trial_id,
                                    "elapsed_ms": image_elapsed_ms,
                                    "gated_abort": True,
                                    "error": str(exc),
                                }
                            )
                            current_mode = "text"

                    iteration += 1

        gate.finalize_trial(sample_id, trial_id)
        return reasoning_steps, profiling_events


def main(argv=None):
    run_inference(VisualReasoner, argv)


if __name__ == "__main__":
    main()
