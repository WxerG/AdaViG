# Copyright (c) 2024 The Qwen Team and The HuggingFace Inc. team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/huggingface/transformers/blob/main/LICENSE.
#
# This modified file is released under the same license.


import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.functional import scaled_dot_product_attention
from transformers.utils import ModelOutput

from flash_attn import flash_attn_varlen_func
from modeling.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)

from modeling.qwen2.configuration_qwen2 import Qwen2Config as _Qwen2Config
from modeling.cache_utils.taylorseer import (
    cal_type,
    taylor_cache_init,
    derivative_approximation,
    taylor_formula,
)
from online_gate import get_online_gate_controller
from runtime_metrics import get_metric_dumper

torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096
# flex_attention = torch.compile(flex_attention) # , dynamic=True, mode='max-autotune'
flex_attention = torch.compile(flex_attention)


METRIC_DUMP_BLOCKS = os.environ.get("BAGEL_METRIC_DUMP_BLOCKS", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# BAGEL_METRIC_LITE=1  →  skip O(n²) exact attention, enable lightweight segment mass.
# BAGEL_METRIC_LITE=0  (default) →  original behaviour preserved.
# Individual env vars (BAGEL_METRIC_EXACT_ATTN / BAGEL_METRIC_SEGMENT_MASS) still
# override when explicitly set.
METRIC_LITE = os.environ.get("BAGEL_METRIC_LITE", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_exact_default = "0" if METRIC_LITE else "1"
_seg_mass_default = "1" if METRIC_LITE else "0"
METRIC_EXACT_ATTN = os.environ.get(
    "BAGEL_METRIC_EXACT_ATTN", _exact_default
).lower() in {"1", "true", "yes", "on"}
METRIC_SEGMENT_MASS = os.environ.get(
    "BAGEL_METRIC_SEGMENT_MASS", _seg_mass_default
).lower() in {"1", "true", "yes", "on"}
_SEGMENT_MASS_DECODE_STRIDE = int(
    os.environ.get("BAGEL_METRIC_SEGMENT_MASS_STRIDE", "100")
)
POST_IMAGE_TEXT_DECODE_MASK_ENV = "BAGEL_POST_IMAGE_TEXT_DECODE_MASK"
POST_IMAGE_TEXT_DECODE_INTERVENTION_MODE_ENV = (
    "BAGEL_POST_IMAGE_TEXT_DECODE_INTERVENTION_MODE"
)
POST_IMAGE_TEXT_DECODE_INTERVENTION_SEED_ENV = (
    "BAGEL_POST_IMAGE_TEXT_DECODE_INTERVENTION_SEED"
)
SAVE_IMAGEGEN_INTERNAL_STEPS = os.environ.get(
    "BAGEL_METRIC_SAVE_IMAGEGEN_INTERNAL_STEPS", "0"
).lower() in {"1", "true", "yes", "on"}


def _parse_positive_int_set(raw: str) -> set[int]:
    out: set[int] = set()
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0:
            out.add(value)
    return out


_IMAGEGEN_SAMPLE_STEPS = _parse_positive_int_set(
    os.environ.get("BAGEL_METRIC_IMAGEGEN_SAMPLE_STEPS", "")
)


def _should_sample_imagegen_step(trace: Optional[Dict[str, Any]]) -> bool:
    if not trace:
        return True
    step_idx = trace.get("diffusion_step_idx")
    if step_idx is None:
        return True
    try:
        step_idx = int(step_idx)
    except Exception:
        return True
    is_terminal = bool(trace.get("diffusion_is_terminal", False))
    if _IMAGEGEN_SAMPLE_STEPS:
        return step_idx in _IMAGEGEN_SAMPLE_STEPS or is_terminal
    return (step_idx - 1) % 10 == 0 or step_idx == 45 or is_terminal


def _is_cfg_context(trace: Optional[Dict[str, Any]]) -> bool:
    """Whether this trace belongs to CFG side branches."""
    if not trace:
        return False
    context = str(trace.get("context") or "").strip().lower()
    return context in {"cfg_text", "cfg_img"}


def _should_save_attention_pt(trace: Optional[Dict[str, Any]]) -> bool:
    """Persist attention tensors only for gen branch, and only terminal calls for hot stages."""
    if not trace:
        return True
    if _is_cfg_context(trace):
        return False
    stage = str(trace.get("stage") or "")
    if stage == "image_generation":
        if SAVE_IMAGEGEN_INTERNAL_STEPS:
            return _should_sample_imagegen_step(trace)
        return bool(trace.get("diffusion_is_terminal", False))
    if stage == "text_decode":
        return bool(trace.get("decode_is_terminal", False))
    return True


class Qwen2Config(_Qwen2Config):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2Model`]. It is used to instantiate a
    Qwen2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-7B-beta [Qwen/Qwen2-7B-beta](https://huggingface.co/Qwen/Qwen2-7B-beta).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Qwen2Model, Qwen2Config

    >>> # Initializing a Qwen2 style configuration
    >>> configuration = Qwen2Config()

    >>> # Initializing a model from the Qwen2-7B style configuration
    >>> model = Qwen2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        is_causal=True,
        _attn_implementation="flash_attention_2",
        qk_norm=True,
        layer_module="Qwen2DecoderLayer",
        freeze_und=False,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
            is_causal=is_causal,
            _attn_implementation=_attn_implementation,
            **kwargs,
        )
        self.qk_norm = qk_norm
        self.layer_module = layer_module
        self.freeze_und = freeze_und


class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_lens(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0


def _get_post_image_text_decode_mask_kinds() -> Tuple[str, ...]:
    raw = str(os.environ.get(POST_IMAGE_TEXT_DECODE_MASK_ENV, "") or "").strip().lower()
    if not raw:
        return ()
    kinds: List[str] = []
    for token in raw.split(","):
        token = token.strip()
        if token in {"vit", "vae"} and token not in kinds:
            kinds.append(token)
    return tuple(kinds)


def _get_post_image_text_decode_intervention_mode() -> str:
    raw = (
        str(os.environ.get(POST_IMAGE_TEXT_DECODE_INTERVENTION_MODE_ENV, "") or "")
        .strip()
        .lower()
    )
    if raw in {"", "none", "off"}:
        return ""
    if raw in {"mask", "zero", "noise"}:
        return raw
    return ""


def _iter_post_image_target_ranges(
    *,
    key_values_lens: torch.Tensor,
    past_key_values: Optional[NaiveCache],
    trace: Optional[Dict[str, Any]],
) -> List[Tuple[int, int, List[Tuple[int, int]]]]:
    mask_kinds = _get_post_image_text_decode_mask_kinds()
    if not mask_kinds or past_key_values is None or not trace:
        return []
    if _is_cfg_context(trace):
        return []
    if str(trace.get("stage") or "") != "text_decode":
        return []
    if str(trace.get("query_kind") or "") != "text":
        return []

    raw_segments = list(getattr(past_key_values, "segment_ledger", []) or [])
    if not raw_segments:
        return []

    seq_ranges: List[Tuple[int, int, List[Tuple[int, int]]]] = []
    seq_k_start = 0
    for kv_len_t in key_values_lens.tolist():
        kv_len = int(kv_len_t)
        if kv_len <= 0:
            seq_ranges.append((seq_k_start, seq_k_start, []))
            continue
        past_len = kv_len
        seq_k_end = seq_k_start + kv_len
        local_ranges: List[Tuple[int, int]] = []
        for seg in raw_segments:
            if str(seg.get("source") or "") != "generated_image":
                continue
            if str(seg.get("kind") or "") not in mask_kinds:
                continue
            try:
                seg_start = int(seg.get("start", 0))
                seg_end = int(seg.get("end", 0))
            except Exception:
                continue
            local_start = max(seq_k_start, seg_start) - seq_k_start
            local_end = min(seq_k_end, seg_end) - seq_k_start
            local_end = min(local_end, past_len)
            if local_end > local_start:
                local_ranges.append((local_start, local_end))
        seq_ranges.append((seq_k_start, seq_k_end, local_ranges))
        seq_k_start += kv_len
    return seq_ranges


def _build_post_image_text_decode_attention_masks(
    *,
    query_lens: torch.Tensor,
    key_values_lens: torch.Tensor,
    past_key_values: Optional[NaiveCache],
    trace: Optional[Dict[str, Any]],
    device: torch.device,
) -> Optional[List[torch.Tensor]]:
    if _get_post_image_text_decode_intervention_mode() != "mask":
        return None

    masks: List[torch.Tensor] = []
    seq_ranges = _iter_post_image_target_ranges(
        key_values_lens=key_values_lens,
        past_key_values=past_key_values,
        trace=trace,
    )
    if not seq_ranges:
        return None
    applied_any = False
    query_lens_list = [int(x) for x in query_lens.tolist()]
    for q_len, (seq_k_start, seq_k_end, local_ranges) in zip(
        query_lens_list, seq_ranges
    ):
        kv_len = seq_k_end - seq_k_start
        if q_len <= 0 or kv_len <= 0:
            masks.append(
                torch.zeros(
                    (max(q_len, 0), max(kv_len, 0)), dtype=torch.float32, device=device
                )
            )
            continue

        mask = torch.zeros((q_len, kv_len), dtype=torch.float32, device=device)
        past_len = max(kv_len - q_len, 0)

        for q_idx in range(q_len):
            max_local_k = past_len + q_idx
            if max_local_k + 1 < kv_len:
                mask[q_idx, max_local_k + 1 :] = float("-inf")

        for local_start, local_end in local_ranges:
            if local_end > local_start:
                mask[:, local_start:local_end] = float("-inf")
                applied_any = True

        masks.append(mask)

    if not applied_any:
        return None
    return masks


def _apply_post_image_text_decode_kv_intervention(
    *,
    merged_key_states: torch.Tensor,
    merged_value_states: torch.Tensor,
    key_values_lens: torch.Tensor,
    past_key_values: Optional[NaiveCache],
    trace: Optional[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    mode = _get_post_image_text_decode_intervention_mode()
    if mode not in {"zero", "noise"}:
        return merged_key_states, merged_value_states

    seq_ranges = _iter_post_image_target_ranges(
        key_values_lens=key_values_lens,
        past_key_values=past_key_values,
        trace=trace,
    )
    if not seq_ranges:
        return merged_key_states, merged_value_states

    k = merged_key_states.clone()
    v = merged_value_states.clone()
    seed = int(os.environ.get(POST_IMAGE_TEXT_DECODE_INTERVENTION_SEED_ENV, "1234"))
    gen = None
    if mode == "noise":
        gen = torch.Generator(device=k.device)
        gen.manual_seed(seed)

    for seq_k_start, _, local_ranges in seq_ranges:
        for local_start, local_end in local_ranges:
            g0 = seq_k_start + local_start
            g1 = seq_k_start + local_end
            if g1 <= g0:
                continue
            if mode == "zero":
                k[g0:g1] = 0
                v[g0:g1] = 0
            else:
                seg_k = k[g0:g1]
                seg_v = v[g0:g1]
                k_mean = seg_k.mean(dim=0, keepdim=True)
                v_mean = seg_v.mean(dim=0, keepdim=True)
                k_std = seg_k.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
                v_std = seg_v.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
                k[g0:g1] = (
                    torch.randn(
                        seg_k.shape,
                        generator=gen,
                        device=seg_k.device,
                        dtype=seg_k.dtype,
                    )
                    * k_std
                    + k_mean
                )
                v[g0:g1] = (
                    torch.randn(
                        seg_v.shape,
                        generator=gen,
                        device=seg_v.device,
                        dtype=seg_v.dtype,
                    )
                    * v_std
                    + v_mean
                )

    return k, v


def maybe_dump_attention_debug(
    *,
    layer_idx: int,
    query_lens: torch.Tensor,
    key_values_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    packed_query_states: torch.Tensor,
    merged_key_states: torch.Tensor,
    merged_value_states: torch.Tensor,
    is_causal: bool,
    head_dim: int,
    past_key_values: Optional[NaiveCache] = None,
    trace: Optional[Dict[str, Any]] = None,
):
    """Optionally run PyTorch reference attention and persist its map for inspection."""

    gate = get_online_gate_controller()
    gate_needed = gate.exact_attention_needed(trace, layer_idx)

    if not METRIC_EXACT_ATTN and not gate_needed:
        return

    dumper = get_metric_dumper()
    # Exact attention solve is expensive; skip CFG branches.
    if _is_cfg_context(trace):
        return
    # Fast path: only instrument configured layers.
    if not gate_needed and layer_idx not in dumper.full_layers:
        return
    # During image generation, only sample every 10th diffusion step plus the terminal step.
    if trace:
        step_idx = trace.get("diffusion_step_idx")
        if (
            step_idx is not None
            and not gate_needed
            and not _should_sample_imagegen_step(trace)
        ):
            return
    sample_id = trace.get("sample_id") if trace else None
    trial_id = trace.get("trial_id") if trace else None
    should_write_summary = dumper.should_write_jsonl(sample_id)
    should_dump_full = dumper.should_dump_full(
        sample_id,
        trial_id,
        layer_idx,
    )
    should_save_tensor = bool(should_dump_full and _should_save_attention_pt(trace))
    if not should_save_tensor and not should_write_summary and not gate_needed:
        return

    from flash_attn.flash_attn_triton_amd.fwd_ref import (
        attention_varlen_forward_pytorch_ref_impl,
    )

    # Keep the reference attention solve on the active device; only move tensors to CPU when saving.
    q = packed_query_states.detach().to(torch.float32)
    k = merged_key_states.detach().to(torch.float32)
    v = merged_value_states.detach().to(torch.float32)
    cu_q = cu_seqlens_q.detach().to(dtype=torch.int32, device=q.device)
    cu_k = cu_seqlens_k.detach().to(dtype=torch.int32, device=q.device)
    q_lens = query_lens.detach().to(torch.int32).cpu()
    kv_lens = key_values_lens.detach().to(torch.int32).cpu()

    max_seqlen_q = int(q_lens.max().item()) if q_lens.numel() > 0 else 0
    max_seqlen_k = int(kv_lens.max().item()) if kv_lens.numel() > 0 else 0

    if max_seqlen_q == 0 or max_seqlen_k == 0:
        return
    # sd_mask is the unnormalized exp(scores); we re-normalize it below into exact attn_probs.
    _, _, sd_mask = attention_varlen_forward_pytorch_ref_impl(
        q=q,
        k=k,
        v=v,
        sm_scale=1.0 / math.sqrt(head_dim),
        causal=is_causal,
        layout="thd",
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        philox_seed=None,
        philox_offset=None,
        alibi_slopes=None,
        use_exp2=False,
    )

    attn_probs = sd_mask.to(torch.float32)
    denom = attn_probs.sum(dim=-1, keepdim=True).clamp_min(
        1e-12
    )  # k_len这里求和平均，每行需要是求和为1
    attn_probs = attn_probs / denom

    exact_payload = build_exact_attention_payload(
        attn_probs=attn_probs,
        q_lens=q_lens,
        kv_lens=kv_lens,
        layer_idx=layer_idx,
        trace=trace,
        past_key_values=past_key_values,
    )
    if exact_payload is not None and trace:
        if should_save_tensor:
            dumper.save_tensor(
                "attn_exact", trace, exact_payload, layer_idx=layer_idx, full_only=True
            )
        exact_summary = exact_payload.get("summary")
        if exact_summary is not None:
            dumper.write_jsonl(
                "attn_exact_summary",
                {
                    k: trace.get(k)
                    for k in (
                        "sample_id",
                        "trial_id",
                        "iteration",
                        "stage",
                        "context",
                        "query_kind",
                        "label",
                        "diffusion_step_idx",
                        "diffusion_total_steps",
                        "diffusion_is_terminal",
                    )
                },
                exact_summary,
            )
        focus_summary = exact_payload.get("focus_summary")
        if focus_summary:
            dumper.write_jsonl(
                "attn_focus_summary",
                {
                    k: trace.get(k)
                    for k in (
                        "sample_id",
                        "trial_id",
                        "iteration",
                        "stage",
                        "context",
                        "query_kind",
                        "label",
                        "diffusion_step_idx",
                        "diffusion_total_steps",
                        "diffusion_is_terminal",
                    )
                },
                focus_summary,
            )
        segment_table = exact_payload.get("segment_table")
        if segment_table:
            dumper.write_jsonl(
                "attn_segment_table",
                {
                    k: trace.get(k)
                    for k in (
                        "sample_id",
                        "trial_id",
                        "iteration",
                        "stage",
                        "context",
                        "query_kind",
                        "label",
                        "diffusion_step_idx",
                        "diffusion_total_steps",
                        "diffusion_is_terminal",
                    )
                },
                segment_table,
            )
        segment_records = exact_payload.get("segment_records") or []
        if segment_records:
            gate.record_segment_records(trace, layer_idx, segment_records)
        for segment_record in segment_records:
            dumper.write_jsonl(
                "attn_exact_segments",
                {
                    k: trace.get(k)
                    for k in (
                        "sample_id",
                        "trial_id",
                        "iteration",
                        "stage",
                        "context",
                        "query_kind",
                        "label",
                        "diffusion_step_idx",
                        "diffusion_total_steps",
                        "diffusion_is_terminal",
                    )
                },
                segment_record,
            )

    payload = {
        "layer": layer_idx,
        "trace": dict(trace or {}),
        "attn_probs": attn_probs.cpu(),
        "query_lens": q_lens,
        "key_value_lens": kv_lens,
    }
    if should_save_tensor and trace:
        dumper.save_tensor(
            "full_attn", trace, payload, layer_idx=layer_idx, full_only=True
        )


def _normalize_attn_layout(
    attn_probs: torch.Tensor, total_q: int
) -> Optional[torch.Tensor]:
    if attn_probs.dim() == 4:
        attn_probs = attn_probs.squeeze(0)
    if attn_probs.shape[0] == total_q:
        return attn_probs.permute(1, 0, 2).contiguous()
    if attn_probs.shape[1] == total_q:
        return attn_probs.contiguous()
    return None


def _build_local_key_segments(
    *,
    seq_k_start: int,
    seq_k_end: int,
    q_len: int,
    raw_segments: List[Dict[str, Any]],
    trace: Dict[str, Any],
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for seg in raw_segments:
        start = max(seq_k_start, int(seg.get("start", 0)))
        end = min(seq_k_end, int(seg.get("end", 0)))
        if end <= start:
            continue
        local_seg = dict(seg)
        local_seg["global_start"] = start
        local_seg["global_end"] = end
        local_seg["local_start"] = start - seq_k_start
        local_seg["local_end"] = end - seq_k_start
        segments.append(local_seg)

    current_local_start = max((seq_k_end - seq_k_start) - q_len, 0)
    if current_local_start < (seq_k_end - seq_k_start):
        segments.append(
            {
                "kind": trace.get("query_kind", "current"),
                "context": trace.get("context"),
                "label": trace.get("label"),
                "source": "current_query",
                "global_start": seq_k_start + current_local_start,
                "global_end": seq_k_end,
                "local_start": current_local_start,
                "local_end": seq_k_end - seq_k_start,
            }
        )
    return segments


def _augment_decode_text_gap_segments(
    *,
    segments: List[Dict[str, Any]],
    seq_k_start: int,
    seq_k_end: int,
    trace: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Fill uncovered key ranges for text_decode as synthetic text segments.

    This only affects exact-attention metric accounting; model forward logic is unchanged.
    """
    if str(trace.get("stage") or "") != "text_decode":
        return segments
    if str(trace.get("query_kind") or "") != "text":
        return segments

    total_k = max(seq_k_end - seq_k_start, 0)
    if total_k <= 0:
        return segments

    intervals: List[Tuple[int, int]] = []
    for seg in segments:
        s = max(0, int(seg.get("local_start", 0)))
        e = min(total_k, int(seg.get("local_end", 0)))
        if e > s:
            intervals.append((s, e))
    if not intervals:
        return segments

    intervals.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))

    out = list(segments)
    cursor = 0
    for s, e in merged:
        if cursor < s:
            out.append(
                {
                    "kind": "text",
                    "context": trace.get("context"),
                    "label": trace.get("label"),
                    "source": "decode_history_gap",
                    "global_start": seq_k_start + cursor,
                    "global_end": seq_k_start + s,
                    "local_start": cursor,
                    "local_end": s,
                }
            )
        cursor = max(cursor, e)
    if cursor < total_k:
        out.append(
            {
                "kind": "text",
                "context": trace.get("context"),
                "label": trace.get("label"),
                "source": "decode_history_gap",
                "global_start": seq_k_start + cursor,
                "global_end": seq_k_end,
                "local_start": cursor,
                "local_end": total_k,
            }
        )

    out.sort(
        key=lambda seg: (int(seg.get("local_start", 0)), int(seg.get("local_end", 0)))
    )
    return out


def _safe_relation_metric_name(relation_label: str) -> str:
    return (
        relation_label.replace(":", "__")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _safe_segment_token(value: Any) -> str:
    raw = str(value if value is not None else "na")
    return raw.replace(":", "__").replace("/", "_").replace("-", "_").replace(" ", "_")


def _segment_focus_group(seg: Dict[str, Any]) -> str:
    kind = str(seg.get("kind") or "")
    source = str(seg.get("source") or "")
    iteration = seg.get("iteration")
    try:
        iteration_int = int(iteration)
    except Exception:
        iteration_int = None

    if kind == "vit" and source == "input_image":
        return "original_image_vit"
    if kind == "vae" and source == "input_image":
        return "original_image_vae"
    if kind == "vit" and source == "generated_image":
        return "generated_image_vit"
    if kind == "vae" and source == "generated_image":
        return "generated_image_vae"
    if kind == "vae" and source == "current_query":
        return "generated_image_current_vae"
    if kind == "text" and source == "current_query":
        return "current_text_self"
    if kind == "text" and source == "generated_text" and iteration_int == 1:
        return "step1_text"
    if kind == "text" and source == "generated_text":
        return "generated_text_prompt"
    return "other"


def _append_dynamic_text_segment_summary(
    *,
    summary: Dict[str, Any],
    prefix: str,
    flat_segments: List[Dict[str, Any]],
    block_mass_mean: torch.Tensor,
) -> None:
    signature_counts: Dict[Tuple[str, str, str], int] = {}

    for idx, seg in enumerate(flat_segments):
        if str(seg.get("kind") or "") != "text":
            continue

        source = str(seg.get("source") or "na")
        label = str(seg.get("label") or "na")
        source_iteration = seg.get("iteration")
        source_iteration_str = str(
            source_iteration if source_iteration is not None else "na"
        )
        signature = (source, label, source_iteration_str)
        occurrence_idx = signature_counts.get(signature, 0)
        signature_counts[signature] = occurrence_idx + 1
        mass_value = float(block_mass_mean[idx].item())

        detailed_key = (
            f"{prefix}_to_text__src_{_safe_segment_token(source)}"
            f"__label_{_safe_segment_token(label)}"
            f"__iter_{_safe_segment_token(source_iteration_str)}"
            f"__occ_{occurrence_idx:02d}"
        )
        summary[detailed_key] = mass_value

        if source == "generated_text":
            iter_key = f"{prefix}_to_generated_text_iter_{_safe_segment_token(source_iteration_str)}"
            if iter_key not in summary:
                summary[iter_key] = mass_value
            else:
                summary[f"{iter_key}__occ_{occurrence_idx:02d}"] = mass_value


def _build_focus_summary(
    *,
    trace: Dict[str, Any],
    flat_segments: List[Dict[str, Any]],
    block_mass_mean: torch.Tensor,
) -> Dict[str, Any]:
    def _sum_mass(pred) -> float:
        total = 0.0
        for idx, seg in enumerate(flat_segments):
            if pred(seg):
                total += float(block_mass_mean[idx].item())
        return total

    stage = str(trace.get("stage") or "")
    query_kind = str(trace.get("query_kind") or "")
    summary: Dict[str, Any] = {}

    if stage == "image_generation" and query_kind == "vae":
        summary["vae_to_generated_text_prompt"] = _sum_mass(
            lambda seg: (
                str(seg.get("kind") or "") == "text"
                and str(seg.get("source") or "") == "generated_text"
            )
        )
        summary["vae_to_original_image_vit"] = _sum_mass(
            lambda seg: (
                str(seg.get("kind") or "") == "vit"
                and str(seg.get("source") or "") == "input_image"
            )
        )
        summary["vae_to_original_image_vae"] = _sum_mass(
            lambda seg: (
                str(seg.get("kind") or "") == "vae"
                and str(seg.get("source") or "") == "input_image"
            )
        )
        summary["vae_to_generated_image_vae"] = _sum_mass(
            lambda seg: (
                str(seg.get("kind") or "") == "vae"
                and str(seg.get("source") or "") == "current_query"
            )
        )
        _append_dynamic_text_segment_summary(
            summary=summary,
            prefix="vae",
            flat_segments=flat_segments,
            block_mass_mean=block_mass_mean,
        )
        return summary

    if stage == "text_decode" and query_kind == "text":
        text_to_original_image_vit = _sum_mass(
            lambda seg: (
                str(seg.get("kind") or "") == "vit"
                and str(seg.get("source") or "") == "input_image"
            )
        )
        text_to_original_image_vae = _sum_mass(
            lambda seg: (
                str(seg.get("kind") or "") == "vae"
                and str(seg.get("source") or "") == "input_image"
            )
        )
        text_to_generated_image_vit = _sum_mass(
            lambda seg: (
                str(seg.get("kind") or "") == "vit"
                and str(seg.get("source") or "") == "generated_image"
            )
        )
        text_to_generated_image_vae = _sum_mass(
            lambda seg: (
                str(seg.get("kind") or "") == "vae"
                and str(seg.get("source") or "") == "generated_image"
            )
        )
        text_to_generated_text_prompt = _sum_mass(
            lambda seg: (
                str(seg.get("kind") or "") == "text"
                and str(seg.get("source") or "") == "generated_text"
            )
        )
        text_to_step1_text = _sum_mass(
            lambda seg: (
                str(seg.get("kind") or "") == "text"
                and str(seg.get("source") or "") == "generated_text"
                and str(seg.get("iteration") or "") == "1"
            )
        )
        text_to_current_text_self = _sum_mass(
            lambda seg: (
                str(seg.get("kind") or "") == "text"
                and str(seg.get("source") or "") == "current_query"
            )
        )
        text_to_original_image_total = (
            text_to_original_image_vit + text_to_original_image_vae
        )
        text_to_generated_image_total = (
            text_to_generated_image_vit + text_to_generated_image_vae
        )
        text_to_vit_total = text_to_original_image_vit + text_to_generated_image_vit
        text_to_vae_total = text_to_original_image_vae + text_to_generated_image_vae
        delta_text_to_generated_image_minus_original_image = (
            text_to_generated_image_total - text_to_original_image_total
        )
        delta_text_to_vae_minus_vit = text_to_vae_total - text_to_vit_total
        summary.update(
            {
                "has_generated_image_context": any(
                    str(seg.get("source") or "") == "generated_image"
                    for seg in flat_segments
                ),
                "text_to_original_image_vit": text_to_original_image_vit,
                "text_to_original_image_vae": text_to_original_image_vae,
                "text_to_generated_image_vit": text_to_generated_image_vit,
                "text_to_generated_image_vae": text_to_generated_image_vae,
                "text_to_generated_text_prompt": text_to_generated_text_prompt,
                "text_to_step1_text": text_to_step1_text,
                "text_to_current_text_self": text_to_current_text_self,
                "text_to_original_image_total": text_to_original_image_total,
                "text_to_generated_image_total": text_to_generated_image_total,
                "text_to_vit_total": text_to_vit_total,
                "text_to_vae_total": text_to_vae_total,
                "delta_text_to_generated_image_minus_original_image": (
                    delta_text_to_generated_image_minus_original_image
                ),
                "delta_text_to_generated_image_minus_original_image_gt_0": (
                    delta_text_to_generated_image_minus_original_image > 0.0
                ),
                "delta_text_to_vae_minus_vit": delta_text_to_vae_minus_vit,
            }
        )
        _append_dynamic_text_segment_summary(
            summary=summary,
            prefix="text",
            flat_segments=flat_segments,
            block_mass_mean=block_mass_mean,
        )
    return summary


def _build_segment_records(
    *,
    layer_idx: int,
    flat_segments: List[Dict[str, Any]],
    block_mass_mean: torch.Tensor,
    block_mass_max: torch.Tensor,
) -> List[Dict[str, Any]]:
    signature_counts: Dict[Tuple[str, str, str, str], int] = {}
    records: List[Dict[str, Any]] = []

    for seg_idx, seg in enumerate(flat_segments):
        sig = (
            str(seg.get("kind") or "unknown"),
            str(seg.get("source") or "na"),
            str(seg.get("label") or "na"),
            str(seg.get("iteration") or "na"),
        )
        occurrence_idx = signature_counts.get(sig, 0)
        signature_counts[sig] = occurrence_idx + 1

        relation_kind = str(seg.get("relation_kind") or "unknown")
        source = str(seg.get("source") or "na")
        label = str(seg.get("label") or "na")
        source_iteration = seg.get("iteration")
        source_iteration_str = str(
            source_iteration if source_iteration is not None else "na"
        )
        segment_key = (
            f"seg{seg_idx:02d}__{_safe_segment_token(relation_kind)}"
            f"__src_{_safe_segment_token(source)}"
            f"__label_{_safe_segment_token(label)}"
            f"__iter_{_safe_segment_token(source_iteration_str)}"
            f"__occ_{occurrence_idx:02d}"
        )

        records.append(
            {
                "layer": layer_idx,
                "segment_flat_index": seg_idx,
                "segment_key": segment_key,
                "segment_index": int(seg.get("segment_index", seg_idx)),
                "seq_index": int(seg.get("seq_index", 0)),
                "occurrence_index_with_same_signature": occurrence_idx,
                "kind": seg.get("kind"),
                "context": seg.get("context"),
                "label": seg.get("label"),
                "source": seg.get("source"),
                "source_iteration": source_iteration,
                "length": int(seg.get("length", 0)),
                "global_start": int(seg.get("global_start", 0)),
                "global_end": int(seg.get("global_end", 0)),
                "local_start": int(seg.get("local_start", 0)),
                "local_end": int(seg.get("local_end", 0)),
                "relation_kind": relation_kind,
                "relation_label": seg.get("relation_label"),
                "focus_group": _segment_focus_group(seg),
                "mass_mean": float(block_mass_mean[seg_idx].item()),
                "mass_max": float(block_mass_max[seg_idx].item()),
            }
        )

    return records


def build_exact_attention_payload(
    *,
    attn_probs: torch.Tensor,
    q_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    layer_idx: int,
    trace: Optional[Dict[str, Any]],
    past_key_values: Optional[NaiveCache],
) -> Optional[Dict[str, Any]]:
    if trace is None or past_key_values is None:
        return None

    total_q = int(q_lens.sum().item()) if q_lens.numel() > 0 else 0
    if total_q <= 0:
        return None

    attn_hqk = _normalize_attn_layout(attn_probs, total_q)
    if attn_hqk is None:
        return None

    num_heads = attn_hqk.shape[0]
    raw_segments = list(getattr(past_key_values, "segment_ledger", []) or [])

    q_cursor = 0
    k_cursor = 0
    flat_segments: List[Dict[str, Any]] = []
    head_block_masses: List[torch.Tensor] = []
    visual_indices_all: List[int] = []
    visual_scores_head_all: List[torch.Tensor] = []
    guided_visual_indices_all: List[int] = []
    guided_visual_scores_head_all: List[torch.Tensor] = []
    query_kind = trace.get("query_kind")
    # 造[head,Bk] 即跟不同k区段的比如t2t,t2v
    for seq_idx, (q_len_t, kv_len_t) in enumerate(
        zip(q_lens.tolist(), kv_lens.tolist())
    ):
        q_len = int(q_len_t)
        kv_len = int(kv_len_t)
        if q_len <= 0 or kv_len <= 0:
            q_cursor += q_len
            k_cursor += kv_len
            continue

        seq_attn = attn_hqk[:, q_cursor : q_cursor + q_len, :kv_len]
        local_segments = _build_local_key_segments(
            seq_k_start=k_cursor,
            seq_k_end=k_cursor + kv_len,
            q_len=q_len,
            raw_segments=raw_segments,
            trace=trace,
        )
        local_segments = _augment_decode_text_gap_segments(
            segments=local_segments,
            seq_k_start=k_cursor,
            seq_k_end=k_cursor + kv_len,
            trace=trace,
        )

        for seg_idx, seg in enumerate(local_segments):
            local_start = int(seg["local_start"])
            local_end = int(seg["local_end"])
            if local_end <= local_start:
                continue
            # For the current query block only: sum over the key block, then average over
            # the current query tokens. This yields one scalar per head for this key block.
            mass_h = seq_attn[:, :, local_start:local_end].sum(dim=-1).mean(dim=-1)
            head_block_masses.append(mass_h)
            relation_kind = f"{query_kind or 'unknown'}2{seg.get('kind') or 'unknown'}"
            relation_label = f"{relation_kind}:{seg.get('source') or 'na'}"
            flat_segments.append(
                {
                    "seq_index": seq_idx,
                    "segment_index": seg_idx,
                    "kind": seg.get("kind"),
                    "context": seg.get("context"),
                    "label": seg.get("label"),
                    "source": seg.get("source"),
                    "iteration": seg.get("iteration"),
                    "global_start": int(seg["global_start"]),
                    "global_end": int(seg["global_end"]),
                    "local_start": local_start,
                    "local_end": local_end,
                    "length": local_end - local_start,
                    "relation_kind": relation_kind,
                    "relation_label": relation_label,
                }
            )

        if query_kind == "text":
            text_locals: List[int] = []
            visual_locals: List[int] = []
            visual_globals: List[int] = []
            for seg in local_segments:
                local_range = list(
                    range(int(seg["local_start"]), int(seg["local_end"]))
                )
                global_range = list(
                    range(int(seg["global_start"]), int(seg["global_end"]))
                )
                if seg.get("kind") == "text":
                    text_locals.extend(local_range)
                elif seg.get("kind") in {"vae", "vit"}:
                    visual_locals.extend(local_range)
                    visual_globals.extend(global_range)
            if text_locals:
                text_index_tensor = torch.tensor(text_locals, dtype=torch.long)
            if visual_locals:
                visual_index_tensor = torch.tensor(visual_locals, dtype=torch.long)
                # Token score: average over current text queries only.
                visual_score_head = seq_attn[:, :, visual_index_tensor].sum(
                    dim=1
                ) / float(q_len)
                visual_indices_all.extend(visual_globals)
                visual_scores_head_all.append(visual_score_head.cpu())
                current_query_seg = next(
                    (
                        seg
                        for seg in local_segments
                        if seg.get("kind") == "text"
                        and seg.get("source") == "current_query"
                    ),
                    None,
                )
                if current_query_seg is not None:
                    q_local_start = int(current_query_seg["local_start"])
                    q_local_end = int(current_query_seg["local_end"])
                    if (
                        q_local_end > q_local_start
                        and (q_local_end - q_local_start) == q_len
                    ):
                        current_text_index_tensor = torch.arange(
                            q_local_start, q_local_end, dtype=torch.long
                        )
                        current_text_importance_head = seq_attn[
                            :, :, current_text_index_tensor
                        ].sum(dim=1)
                        current_text_importance_mean = (
                            current_text_importance_head.mean(dim=0)
                        )
                        weight_denom = current_text_importance_mean.sum().clamp_min(
                            1e-12
                        )
                        current_text_weights = (
                            current_text_importance_mean / weight_denom
                        )
                        guided_visual_score_head = (
                            current_text_weights.view(1, -1, 1)
                            * seq_attn[:, :, visual_index_tensor]
                        ).sum(dim=1)
                        guided_visual_indices_all.extend(visual_globals)
                        guided_visual_scores_head_all.append(
                            guided_visual_score_head.cpu()
                        )

        q_cursor += q_len
        k_cursor += kv_len

    if not head_block_masses:
        return None

    # Shape: [H, Bk]. We only score the current query block against all key blocks.
    head_block_mass = torch.stack(head_block_masses, dim=1).cpu()
    block_mass_mean = head_block_mass.mean(dim=0)
    block_mass_max = head_block_mass.max(dim=0).values

    payload: Dict[str, Any] = {
        "layer": layer_idx,
        "trace": dict(trace),
        "segments": flat_segments,
        "query_scope": "current_query_block",
        "query_token_count": total_q,
        "head_block_mass": head_block_mass,
        "block_mass_mean": block_mass_mean,
        "block_mass_max": block_mass_max,
    }

    summary: Dict[str, Any] = {
        "layer": layer_idx,
        "query_kind": query_kind,
        "query_scope": "current_query_block",
        "num_segments": len(flat_segments),
    }

    if flat_segments:
        segment_kind_labels = [
            str(seg.get("kind") or "unknown") for seg in flat_segments
        ]
        segment_relation_labels = [
            str(seg.get("relation_label") or "unknown") for seg in flat_segments
        ]
        payload["segment_kind_labels"] = segment_kind_labels
        payload["segment_relation_labels"] = segment_relation_labels
        for target_kind in ("text", "vit", "vae"):
            kind_mask = torch.tensor(
                [1.0 if kind == target_kind else 0.0 for kind in segment_kind_labels],
                dtype=head_block_mass.dtype,
            )
            nae_to_kind_head = (head_block_mass * kind_mask.unsqueeze(0)).sum(dim=1)
            payload[f"nae_to_{target_kind}_head"] = nae_to_kind_head
            summary[f"nae_to_{target_kind}_mean"] = float(
                nae_to_kind_head.mean().item()
            )
            summary[f"nae_to_{target_kind}_max"] = float(nae_to_kind_head.max().item())
        for relation_label in sorted(set(segment_relation_labels)):
            rel_mask = torch.tensor(
                [
                    1.0 if rel == relation_label else 0.0
                    for rel in segment_relation_labels
                ],
                dtype=head_block_mass.dtype,
            )
            rel_mass_head = (head_block_mass * rel_mask.unsqueeze(0)).sum(dim=1)
            safe_name = _safe_relation_metric_name(relation_label)
            summary[f"mass__{safe_name}"] = float(rel_mass_head.mean().item())

    segment_records = _build_segment_records(
        layer_idx=layer_idx,
        flat_segments=flat_segments,
        block_mass_mean=block_mass_mean,
        block_mass_max=block_mass_max,
    )

    if block_mass_mean.numel() > 0:
        top_mean_idx = int(torch.argmax(block_mass_mean).item())
        top_max_idx = int(torch.argmax(block_mass_max).item())
        summary.update(
            {
                "top_segment_label_mean": str(
                    flat_segments[top_mean_idx].get("relation_label") or "unknown"
                ),
                "top_segment_mass_mean": float(block_mass_mean[top_mean_idx].item()),
                "top_segment_label_max": str(
                    flat_segments[top_max_idx].get("relation_label") or "unknown"
                ),
                "top_segment_mass_max": float(block_mass_max[top_max_idx].item()),
            }
        )
    if visual_scores_head_all:
        visual_score_head = torch.cat(visual_scores_head_all, dim=1)
        payload["visual_key_indices_global"] = torch.tensor(
            visual_indices_all, dtype=torch.long
        )
        payload["visual_token_score_head"] = visual_score_head
        visual_score_mean = visual_score_head.mean(dim=0)
        payload["visual_token_score_mean"] = visual_score_mean
        summary["num_visual_keys"] = int(visual_score_head.shape[1])
        top_visual_idx = int(torch.argmax(visual_score_mean).item())
        summary["top_visual_global_idx"] = int(visual_indices_all[top_visual_idx])
        summary["top_visual_score_mean"] = float(
            visual_score_mean[top_visual_idx].item()
        )
        visual_score_denom = visual_score_mean.sum().clamp_min(1e-12)
        visual_score_norm = visual_score_mean / visual_score_denom
        visual_score_entropy = -(
            visual_score_norm * torch.log2(visual_score_norm.clamp_min(1e-12))
        ).sum()
        payload["visual_score_entropy"] = visual_score_entropy
        summary["visual_score_entropy"] = float(visual_score_entropy.item())
    if guided_visual_scores_head_all:
        guided_visual_score_head = torch.cat(guided_visual_scores_head_all, dim=1)
        payload["guided_visual_key_indices_global"] = torch.tensor(
            guided_visual_indices_all, dtype=torch.long
        )
        payload["guided_visual_score_head"] = guided_visual_score_head
        guided_visual_score_mean = guided_visual_score_head.mean(dim=0)
        payload["guided_visual_score_mean"] = guided_visual_score_mean
        summary["num_guided_visual_keys"] = int(guided_visual_score_head.shape[1])
        top_guided_idx = int(torch.argmax(guided_visual_score_mean).item())
        summary["top_guided_visual_global_idx"] = int(
            guided_visual_indices_all[top_guided_idx]
        )
        summary["top_guided_visual_score_mean"] = float(
            guided_visual_score_mean[top_guided_idx].item()
        )
        guided_visual_score_denom = guided_visual_score_mean.sum().clamp_min(1e-12)
        guided_visual_score_norm = guided_visual_score_mean / guided_visual_score_denom
        guided_visual_score_entropy = -(
            guided_visual_score_norm
            * torch.log2(guided_visual_score_norm.clamp_min(1e-12))
        ).sum()
        payload["guided_visual_score_entropy"] = guided_visual_score_entropy
        summary["guided_visual_score_entropy"] = float(
            guided_visual_score_entropy.item()
        )

    focus_summary = _build_focus_summary(
        trace=dict(trace),
        flat_segments=flat_segments,
        block_mass_mean=block_mass_mean,
    )
    if focus_summary:
        payload["focus_summary"] = focus_summary
    payload["segment_records"] = segment_records
    payload["segment_table"] = {
        "layer": layer_idx,
        "num_segments": len(segment_records),
        "segments": segment_records,
    }
    payload["summary"] = summary
    return payload


def maybe_dump_attention_blocks(
    *,
    layer_idx: int,
    packed_query_states: torch.Tensor,
    merged_key_states: torch.Tensor,
    head_dim: int,
    past_key_values: Optional[NaiveCache] = None,
):
    """Persist compressed q/k summaries for selected cases.

    This is the default analysis path: save small raw tensors that can be
    re-used offline, instead of full attention maps for every step/layer.
    NOTE: this is only a q/k proxy and should not be treated as exact
    attention-mass statistics in analysis.
    """

    if not METRIC_DUMP_BLOCKS or past_key_values is None:
        return

    trace = dict(getattr(past_key_values, "metric_trace", {}) or {})
    dumper = get_metric_dumper()
    # Fast path: only instrument configured layers.
    if layer_idx not in dumper.full_layers:
        return
    sample_id = trace.get("sample_id")
    should_write_summary = dumper.should_write_jsonl(sample_id)
    # Disable attn_blocks tensor dumps; keep lightweight JSONL summaries only.
    if not should_write_summary:
        return

    key_total = int(merged_key_states.shape[0])
    query_total = int(packed_query_states.shape[0])
    current_start = max(key_total - query_total, 0)

    raw_segments = list(getattr(past_key_values, "segment_ledger", []) or [])
    segments = []
    for seg in raw_segments:
        try:
            start = max(0, int(seg.get("start", 0)))
            end = min(key_total, int(seg.get("end", 0)))
        except Exception:
            continue
        if end <= start:
            continue
        segments.append(
            {
                "kind": seg.get("kind", "unknown"),
                "context": seg.get("context"),
                "label": seg.get("label"),
                "source": seg.get("source"),
                "start": start,
                "end": end,
            }
        )

    if current_start < key_total:
        segments.append(
            {
                "kind": trace.get("query_kind", "current"),
                "context": trace.get("context"),
                "label": trace.get("label"),
                "source": "current_query",
                "start": current_start,
                "end": key_total,
            }
        )

    if not segments:
        segments.append(
            {
                "kind": trace.get("query_kind", "unknown"),
                "context": trace.get("context"),
                "label": trace.get("label"),
                "source": "fallback_all",
                "start": 0,
                "end": key_total,
            }
        )

    q_cpu = packed_query_states.detach().to(torch.float32).cpu()
    k_cpu = merged_key_states.detach().to(torch.float32).cpu()

    query_head_mean = q_cpu.mean(dim=0)  # [num_heads, head_dim]
    query_global_mean = query_head_mean.mean(dim=0)  # [head_dim]

    key_segment_head_means = []
    key_segment_global_means = []
    key_segment_lengths = []
    key_segment_labels = []
    for seg in segments:
        seg_tensor = k_cpu[seg["start"] : seg["end"]]
        seg_head_mean = seg_tensor.mean(dim=0)  # [num_kv_heads, head_dim]
        seg_global_mean = seg_head_mean.mean(dim=0)  # [head_dim]
        key_segment_head_means.append(seg_head_mean)
        key_segment_global_means.append(seg_global_mean)
        key_segment_lengths.append(seg["end"] - seg["start"])
        key_segment_labels.append(f"{seg['kind']}:{seg.get('source') or 'na'}")

    key_segment_head_means_t = torch.stack(key_segment_head_means, dim=0)
    key_segment_global_means_t = torch.stack(key_segment_global_means, dim=0)
    segment_logits = (key_segment_global_means_t @ query_global_mean) / math.sqrt(
        head_dim
    )
    segment_probs = torch.softmax(segment_logits, dim=0)

    payload = {
        "layer": layer_idx,
        "trace": trace,
        "query_head_mean": query_head_mean,
        "query_global_mean": query_global_mean,
        "key_segment_head_means": key_segment_head_means_t,
        "key_segment_global_means": key_segment_global_means_t,
        "key_segment_lengths": torch.tensor(key_segment_lengths, dtype=torch.int32),
        "key_segment_labels": key_segment_labels,
        "segment_logits": segment_logits,
        "segment_probs": segment_probs,
        "segments": segments,
    }
    top_idx = (
        int(torch.argmax(segment_probs).item()) if segment_probs.numel() > 0 else -1
    )
    dumper.write_jsonl(
        "attn_blocks_summary",
        {
            k: trace.get(k)
            for k in (
                "sample_id",
                "trial_id",
                "iteration",
                "stage",
                "context",
                "query_kind",
                "label",
            )
        },
        {
            "layer": layer_idx,
            "num_segments": len(segments),
            "query_tokens": query_total,
            "key_tokens": key_total,
            "top_segment_idx": top_idx,
            "top_segment_label": key_segment_labels[top_idx] if top_idx >= 0 else None,
            "top_segment_prob": float(segment_probs[top_idx].item())
            if top_idx >= 0
            else None,
        },
    )


def maybe_dump_segment_attn_mass(
    *,
    layer_idx: int,
    packed_query_states: torch.Tensor,
    merged_key_states: torch.Tensor,
    head_dim: int,
    past_key_values: Optional[NaiveCache] = None,
):
    """Compute per-segment attention mass via lightweight Q·K matmul.

    For decode (seq_q=1) this is a single small matmul — ~1ms, no O(n²)
    attention map.  Gives exact text/vit/vae mass fractions.

    Sampling: prefill stages always dump; decode stages sample every
    ``_SEGMENT_MASS_DECODE_STRIDE`` steps (by seq_k position).
    """

    if not METRIC_SEGMENT_MASS or past_key_values is None:
        return

    trace = dict(getattr(past_key_values, "metric_trace", {}) or {})
    dumper = get_metric_dumper()

    if layer_idx not in dumper.full_layers:
        return
    if not dumper.should_write_jsonl(trace.get("sample_id")):
        return
    if _is_cfg_context(trace):
        return

    # During diffusion, reuse the configured exact-attention sampling cadence.
    step_idx = trace.get("diffusion_step_idx")
    if step_idx is not None and not _should_sample_imagegen_step(trace):
        return

    seq_k = merged_key_states.shape[0]
    seq_q = packed_query_states.shape[0]
    if seq_k == 0 or seq_q == 0:
        return

    # For decode, sample at stride intervals to keep I/O bounded.
    if (
        seq_q == 1
        and _SEGMENT_MASS_DECODE_STRIDE > 0
        and seq_k % _SEGMENT_MASS_DECODE_STRIDE != 0
    ):
        return

    raw_segments = list(getattr(past_key_values, "segment_ledger", []) or [])

    num_heads_q = packed_query_states.shape[1]
    num_kv_heads = merged_key_states.shape[1]
    num_groups = num_heads_q // num_kv_heads if num_kv_heads > 0 else 1

    with torch.no_grad():
        # Last query token is representative (for prefill it sees full context).
        q = packed_query_states[-1].detach().float()  # [num_heads_q, head_dim]
        k = merged_key_states.detach().float()  # [seq_k, num_kv_heads, head_dim]

        # GQA-aware score computation.
        # q → [num_kv_heads, groups, head_dim]
        q_grouped = q.view(num_kv_heads, num_groups, head_dim)
        # k → [num_kv_heads, head_dim, seq_k]
        scores = torch.bmm(q_grouped, k.permute(1, 2, 0)) / math.sqrt(head_dim)
        # → [num_heads_q, seq_k]
        scores = scores.reshape(num_heads_q, seq_k)
        attn = torch.softmax(scores, dim=-1)

        # Aggregate mass by segment kind.
        mass_by_kind: Dict[str, torch.Tensor] = {}
        for seg in raw_segments:
            start = max(0, int(seg.get("start", 0)))
            end = min(seq_k, int(seg.get("end", 0)))
            if end <= start:
                continue
            kind = seg.get("kind", "text")
            seg_mass = attn[:, start:end].sum(dim=-1)  # [num_heads_q]
            if kind in mass_by_kind:
                mass_by_kind[kind] = mass_by_kind[kind] + seg_mass
            else:
                mass_by_kind[kind] = seg_mass.clone()

        # Current query tokens (appended at tail of merged_key_states).
        current_start = max(seq_k - seq_q, 0)
        if current_start < seq_k:
            current_mass = attn[:, current_start:].sum(dim=-1)
            qk = trace.get("query_kind", "text")
            if qk in mass_by_kind:
                mass_by_kind[qk] = mass_by_kind[qk] + current_mass
            else:
                mass_by_kind[qk] = current_mass.clone()

    record: Dict[str, Any] = {
        "layer": layer_idx,
        "query_tokens": seq_q,
        "key_tokens": seq_k,
    }
    for kind in ("text", "vit", "vae"):
        if kind in mass_by_kind:
            m = mass_by_kind[kind]
            record[f"mass_{kind}"] = round(float(m.mean().item()), 6)
            record[f"mass_{kind}_max"] = round(float(m.max().item()), 6)
        else:
            record[f"mass_{kind}"] = 0.0
            record[f"mass_{kind}_max"] = 0.0

    trace_keys = (
        "sample_id",
        "trial_id",
        "iteration",
        "stage",
        "context",
        "query_kind",
        "label",
    )
    dumper.write_jsonl(
        "segment_attn_mass",
        {k: trace.get(k) for k in trace_keys},
        record,
    )


@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    packed_query_sequence: torch.FloatTensor = None
    past_key_values: Optional[NaiveCache] = None


def pad_sequence(tensor, pad_size):
    H, L, D = tensor.shape
    pad_tensor = tensor.new_zeros((H, pad_size, D))
    return torch.cat([tensor, pad_tensor], dim=1)


class PackedAttention(Qwen2Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        if self.config.qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask: List[torch.Tensor],
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ):
        packed_query_states = self.q_proj(packed_sequence).view(
            -1, self.num_heads, self.head_dim
        )
        packed_key_states = self.k_proj(packed_sequence).view(
            -1, self.num_key_value_heads, self.head_dim
        )
        packed_value_states = self.v_proj(packed_sequence).view(
            -1, self.num_key_value_heads, self.head_dim
        )

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states,
            packed_key_states,
            packed_cos,
            packed_sin,
            unsqueeze_dim=1,
        )

        if isinstance(attention_mask, List):
            packed_key_states = packed_key_states[:, :, None, :].repeat(
                1, 1, self.num_key_value_groups, 1
            )
            packed_key_states = packed_key_states.reshape(
                -1, self.num_heads, self.head_dim
            )
            packed_value_states = packed_value_states[:, :, None, :].repeat(
                1, 1, self.num_key_value_groups, 1
            )
            packed_value_states = packed_value_states.reshape(
                -1, self.num_heads, self.head_dim
            )

            unpacked_query_states = packed_query_states.transpose(0, 1).split(
                sample_lens, dim=1
            )
            unpacked_key_states = packed_key_states.transpose(0, 1).split(
                sample_lens, dim=1
            )
            unpacked_value_states = packed_value_states.transpose(0, 1).split(
                sample_lens, dim=1
            )
            upacked_attn_output = []
            for (
                query_states,
                key_states,
                value_states,
                attention_mask_per_sample,
            ) in zip(
                unpacked_query_states,
                unpacked_key_states,
                unpacked_value_states,
                attention_mask,
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0),
                        key_states.to(torch.bfloat16).unsqueeze(0),
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states = pad_sequence(
                packed_query_states.permute(1, 0, 2), pad_size
            )
            packed_key_states = pad_sequence(
                packed_key_states.permute(1, 0, 2), pad_size
            )
            packed_value_states = pad_sequence(
                packed_value_states.permute(1, 0, 2), pad_size
            )
            packed_attn_output = flex_attention(
                packed_query_states.unsqueeze(0),
                packed_key_states.unsqueeze(0),
                packed_value_states.unsqueeze(0),
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(
            -1, self.hidden_size
        )
        packed_attn_output = self.o_proj(packed_attn_output)

        return packed_attn_output

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
    ):
        packed_query_states = self.q_proj(packed_query_sequence).view(
            -1, self.num_heads, self.head_dim
        )
        packed_key_states = self.k_proj(packed_query_sequence).view(
            -1, self.num_key_value_heads, self.head_dim
        )
        packed_value_states = self.v_proj(packed_query_sequence).view(
            -1, self.num_key_value_heads, self.head_dim
        )

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states,
            packed_key_states,
            packed_cos,
            packed_sin,
            unsqueeze_dim=1,
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if (
            past_key_values is not None
            and past_key_values.key_cache[self.layer_idx] is not None
        ):
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros(
                (seqlens, self.num_key_value_heads, self.head_dim)
            )
            merged_value_states = past_key_states.new_zeros(
                (seqlens, self.num_key_value_heads, self.head_dim)
            )
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(key_values_lens, dim=0), (1, 0)
        )
        metric_trace = (
            dict(getattr(past_key_values, "metric_trace", {}) or {})
            if past_key_values is not None
            else {}
        )

        maybe_dump_attention_blocks(
            layer_idx=self.layer_idx if self.layer_idx is not None else -1,
            packed_query_states=packed_query_states,
            merged_key_states=merged_key_states,
            head_dim=self.head_dim,
            past_key_values=past_key_values,
        )
        maybe_dump_attention_debug(
            layer_idx=self.layer_idx if self.layer_idx is not None else -1,
            query_lens=query_lens,
            key_values_lens=key_values_lens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            packed_query_states=packed_query_states,
            merged_key_states=merged_key_states,
            merged_value_states=merged_value_states,
            is_causal=is_causal,
            head_dim=self.head_dim,
            past_key_values=past_key_values,
            trace=metric_trace,
        )
        maybe_dump_segment_attn_mass(
            layer_idx=self.layer_idx if self.layer_idx is not None else -1,
            packed_query_states=packed_query_states,
            merged_key_states=merged_key_states,
            head_dim=self.head_dim,
            past_key_values=past_key_values,
        )

        merged_key_states, merged_value_states = (
            _apply_post_image_text_decode_kv_intervention(
                merged_key_states=merged_key_states,
                merged_value_states=merged_value_states,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                trace=metric_trace,
            )
        )
        attn_masks = _build_post_image_text_decode_attention_masks(
            query_lens=query_lens,
            key_values_lens=key_values_lens,
            past_key_values=past_key_values,
            trace=metric_trace,
            device=packed_query_states.device,
        )
        if attn_masks is not None:
            expanded_key_states = merged_key_states[:, :, None, :].repeat(
                1, 1, self.num_key_value_groups, 1
            )
            expanded_key_states = expanded_key_states.reshape(
                -1, self.num_heads, self.head_dim
            )
            expanded_value_states = merged_value_states[:, :, None, :].repeat(
                1, 1, self.num_key_value_groups, 1
            )
            expanded_value_states = expanded_value_states.reshape(
                -1, self.num_heads, self.head_dim
            )
            unpacked_query_states = packed_query_states.transpose(0, 1).split(
                query_lens.tolist(), dim=1
            )
            unpacked_key_states = expanded_key_states.transpose(0, 1).split(
                key_values_lens.tolist(), dim=1
            )
            unpacked_value_states = expanded_value_states.transpose(0, 1).split(
                key_values_lens.tolist(), dim=1
            )
            unpacked_outputs: List[torch.Tensor] = []
            for (
                query_states,
                key_states,
                value_states,
                attention_mask_per_sample,
            ) in zip(
                unpacked_query_states,
                unpacked_key_states,
                unpacked_value_states,
                attn_masks,
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0),
                        key_states.to(torch.bfloat16).unsqueeze(0),
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attn_mask=attention_mask_per_sample.to(
                            torch.bfloat16
                        ).unsqueeze(0),
                        is_causal=False,
                    )
                unpacked_outputs.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(unpacked_outputs, dim=1)
        else:
            packed_attn_output = flash_attn_varlen_func(
                q=packed_query_states,
                k=merged_key_states,
                v=merged_value_states,
                cu_seqlens_q=cu_seqlens_q.to(torch.int32),
                cu_seqlens_k=cu_seqlens_k.to(torch.int32),
                max_seqlen_q=max(query_lens).item(),
                max_seqlen_k=max(key_values_lens).item(),
                causal=is_causal,
            )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output)

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values


class PackedAttentionMoT(Qwen2Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        if self.config.qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.q_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()

        self.q_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj_moe_gen = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ):
        packed_query_states = packed_sequence.new_zeros(
            (packed_sequence.shape[0], self.num_heads * self.head_dim)
        )
        packed_key_states = packed_sequence.new_zeros(
            (packed_sequence.shape[0], self.num_key_value_heads * self.head_dim)
        )
        packed_value_states = packed_sequence.new_zeros(
            (packed_sequence.shape[0], self.num_key_value_heads * self.head_dim)
        )

        packed_sequence_und = packed_sequence[packed_und_token_indexes]
        packed_sequence_gen = packed_sequence[packed_gen_token_indexes]

        packed_query_states[packed_und_token_indexes] = self.q_proj(packed_sequence_und)
        packed_query_states[packed_gen_token_indexes] = self.q_proj_moe_gen(
            packed_sequence_gen
        )

        packed_key_states[packed_und_token_indexes] = self.k_proj(packed_sequence_und)
        packed_key_states[packed_gen_token_indexes] = self.k_proj_moe_gen(
            packed_sequence_gen
        )

        packed_value_states[packed_und_token_indexes] = self.v_proj(packed_sequence_und)
        packed_value_states[packed_gen_token_indexes] = self.v_proj_moe_gen(
            packed_sequence_gen
        )

        packed_query_states = packed_query_states.view(
            -1, self.num_heads, self.head_dim
        )
        packed_key_states = packed_key_states.view(
            -1, self.num_key_value_heads, self.head_dim
        )
        packed_value_states = packed_value_states.view(
            -1, self.num_key_value_heads, self.head_dim
        )
        if self.config.freeze_und:
            packed_value_states[packed_und_token_indexes] = packed_value_states[
                packed_und_token_indexes
            ].detach()

        packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
        packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

        packed_query_states_[packed_und_token_indexes] = self.q_norm(
            packed_query_states[packed_und_token_indexes]
        )
        if self.config.freeze_und:
            packed_query_states_[packed_und_token_indexes] = packed_query_states_[
                packed_und_token_indexes
            ].detach()
        packed_query_states_[packed_gen_token_indexes] = self.q_norm_moe_gen(
            packed_query_states[packed_gen_token_indexes]
        )

        packed_key_states_[packed_und_token_indexes] = self.k_norm(
            packed_key_states[packed_und_token_indexes]
        )
        if self.config.freeze_und:
            packed_key_states_[packed_und_token_indexes] = packed_key_states_[
                packed_und_token_indexes
            ].detach()
        packed_key_states_[packed_gen_token_indexes] = self.k_norm_moe_gen(
            packed_key_states[packed_gen_token_indexes]
        )

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states_, packed_key_states_ = apply_rotary_pos_emb(
            packed_query_states_,
            packed_key_states_,
            packed_cos,
            packed_sin,
            unsqueeze_dim=1,
        )

        if isinstance(attention_mask, List):
            packed_key_states_ = packed_key_states_[:, :, None, :].repeat(
                1, 1, self.num_key_value_groups, 1
            )
            packed_key_states_ = packed_key_states_.reshape(
                -1, self.num_heads, self.head_dim
            )
            packed_value_states = packed_value_states[:, :, None, :].repeat(
                1, 1, self.num_key_value_groups, 1
            )
            packed_value_states = packed_value_states.reshape(
                -1, self.num_heads, self.head_dim
            )

            unpacked_query_states = packed_query_states_.transpose(0, 1).split(
                sample_lens, dim=1
            )
            unpacked_key_states = packed_key_states_.transpose(0, 1).split(
                sample_lens, dim=1
            )
            unpacked_value_states = packed_value_states.transpose(0, 1).split(
                sample_lens, dim=1
            )
            upacked_attn_output = []
            for (
                query_states,
                key_states,
                value_states,
                attention_mask_per_sample,
            ) in zip(
                unpacked_query_states,
                unpacked_key_states,
                unpacked_value_states,
                attention_mask,
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0),
                        key_states.to(torch.bfloat16).unsqueeze(0),
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states_ = pad_sequence(
                packed_query_states_.permute(1, 0, 2), pad_size
            )
            packed_key_states_ = pad_sequence(
                packed_key_states_.permute(1, 0, 2), pad_size
            )
            packed_value_states = pad_sequence(
                packed_value_states.permute(1, 0, 2), pad_size
            )
            packed_attn_output = flex_attention(
                packed_query_states_.unsqueeze(0),  # 1, num_head, L, head_dim
                packed_key_states_.unsqueeze(0),
                packed_value_states.unsqueeze(0),
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(
            -1, self.num_heads * self.head_dim
        )
        packed_attn_output_ = packed_attn_output.new_zeros(packed_attn_output.shape)
        packed_attn_output_[packed_und_token_indexes] = self.o_proj(
            packed_attn_output[packed_und_token_indexes]
        )
        packed_attn_output_[packed_gen_token_indexes] = self.o_proj_moe_gen(
            packed_attn_output[packed_gen_token_indexes]
        )

        return packed_attn_output_

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ):
        if mode == "und":
            packed_query_states = self.q_proj(packed_query_sequence).view(
                -1, self.num_heads, self.head_dim
            )
            packed_key_states = self.k_proj(packed_query_sequence).view(
                -1, self.num_key_value_heads, self.head_dim
            )
            packed_value_states = self.v_proj(packed_query_sequence).view(
                -1, self.num_key_value_heads, self.head_dim
            )
            packed_query_states = self.q_norm(packed_query_states)
            packed_key_states = self.k_norm(packed_key_states)
        elif mode == "gen":
            packed_query_sequence = packed_query_sequence.to(torch.bfloat16)
            packed_query_states = packed_query_sequence.new_zeros(
                (packed_query_sequence.shape[0], self.num_heads * self.head_dim)
            )
            packed_key_states = packed_query_sequence.new_zeros(
                (
                    packed_query_sequence.shape[0],
                    self.num_key_value_heads * self.head_dim,
                )
            )
            packed_value_states = packed_query_sequence.new_zeros(
                (
                    packed_query_sequence.shape[0],
                    self.num_key_value_heads * self.head_dim,
                )
            )

            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]

            packed_query_states[packed_text_indexes] = self.q_proj(
                packed_text_query_sequence
            )  # [2,3584]
            packed_query_states[packed_vae_token_indexes] = self.q_proj_moe_gen(
                packed_vae_query_sequence
            )  # [1024,3584]

            packed_key_states[packed_text_indexes] = self.k_proj(
                packed_text_query_sequence
            )
            packed_key_states[packed_vae_token_indexes] = self.k_proj_moe_gen(
                packed_vae_query_sequence
            )

            packed_value_states[packed_text_indexes] = self.v_proj(
                packed_text_query_sequence
            )
            packed_value_states[packed_vae_token_indexes] = self.v_proj_moe_gen(
                packed_vae_query_sequence
            )

            packed_query_states = packed_query_states.view(
                -1, self.num_heads, self.head_dim
            )
            packed_key_states = packed_key_states.view(
                -1, self.num_key_value_heads, self.head_dim
            )
            packed_value_states = packed_value_states.view(
                -1, self.num_key_value_heads, self.head_dim
            )

            packed_query_states = packed_query_states.to(torch.float32)
            packed_query_states[packed_text_indexes] = self.q_norm(
                packed_query_states[packed_text_indexes]
            )
            packed_query_states[packed_vae_token_indexes] = self.q_norm_moe_gen(
                packed_query_states[packed_vae_token_indexes]
            )

            packed_key_states = packed_key_states.to(torch.float32)
            packed_key_states[packed_text_indexes] = self.k_norm(
                packed_key_states[packed_text_indexes]
            )
            packed_key_states[packed_vae_token_indexes] = self.k_norm_moe_gen(
                packed_key_states[packed_vae_token_indexes]
            )

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states,
            packed_key_states,
            packed_cos,
            packed_sin,
            unsqueeze_dim=1,
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if (
            past_key_values is not None
            and past_key_values.key_cache[self.layer_idx] is not None
        ):
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros(
                size=[seqlens, self.num_key_value_heads, self.head_dim]
            )
            merged_value_states = past_key_states.new_zeros(
                size=[seqlens, self.num_key_value_heads, self.head_dim]
            )
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(key_values_lens, dim=0), (1, 0)
        )
        metric_trace = (
            dict(getattr(past_key_values, "metric_trace", {}) or {})
            if past_key_values is not None
            else {}
        )

        maybe_dump_attention_blocks(
            layer_idx=self.layer_idx if self.layer_idx is not None else -1,
            packed_query_states=packed_query_states,
            merged_key_states=merged_key_states,
            head_dim=self.head_dim,
            past_key_values=past_key_values,
        )
        maybe_dump_attention_debug(
            layer_idx=self.layer_idx if self.layer_idx is not None else -1,
            query_lens=query_lens,
            key_values_lens=key_values_lens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            packed_query_states=packed_query_states,
            merged_key_states=merged_key_states,
            merged_value_states=merged_value_states,
            is_causal=is_causal,
            head_dim=self.head_dim,
            past_key_values=past_key_values,
            trace=metric_trace,
        )
        maybe_dump_segment_attn_mass(
            layer_idx=self.layer_idx if self.layer_idx is not None else -1,
            packed_query_states=packed_query_states,
            merged_key_states=merged_key_states,
            head_dim=self.head_dim,
            past_key_values=past_key_values,
        )

        merged_key_states, merged_value_states = (
            _apply_post_image_text_decode_kv_intervention(
                merged_key_states=merged_key_states,
                merged_value_states=merged_value_states,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                trace=metric_trace,
            )
        )
        attn_masks = _build_post_image_text_decode_attention_masks(
            query_lens=query_lens,
            key_values_lens=key_values_lens,
            past_key_values=past_key_values,
            trace=metric_trace,
            device=packed_query_states.device,
        )
        if attn_masks is not None:
            expanded_key_states = merged_key_states[:, :, None, :].repeat(
                1, 1, self.num_key_value_groups, 1
            )
            expanded_key_states = expanded_key_states.reshape(
                -1, self.num_heads, self.head_dim
            )
            expanded_value_states = merged_value_states[:, :, None, :].repeat(
                1, 1, self.num_key_value_groups, 1
            )
            expanded_value_states = expanded_value_states.reshape(
                -1, self.num_heads, self.head_dim
            )
            unpacked_query_states = packed_query_states.transpose(0, 1).split(
                query_lens.tolist(), dim=1
            )
            unpacked_key_states = expanded_key_states.transpose(0, 1).split(
                key_values_lens.tolist(), dim=1
            )
            unpacked_value_states = expanded_value_states.transpose(0, 1).split(
                key_values_lens.tolist(), dim=1
            )
            unpacked_outputs: List[torch.Tensor] = []
            for (
                query_states,
                key_states,
                value_states,
                attention_mask_per_sample,
            ) in zip(
                unpacked_query_states,
                unpacked_key_states,
                unpacked_value_states,
                attn_masks,
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0),
                        key_states.to(torch.bfloat16).unsqueeze(0),
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attn_mask=attention_mask_per_sample.to(
                            torch.bfloat16
                        ).unsqueeze(0),
                        is_causal=False,
                    )
                unpacked_outputs.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(unpacked_outputs, dim=1)
        else:
            packed_attn_output = flash_attn_varlen_func(
                q=packed_query_states,  # [1026,28,128]
                k=merged_key_states,  # [1134,4,128]
                v=merged_value_states,
                cu_seqlens_q=cu_seqlens_q.to(torch.int32),
                cu_seqlens_k=cu_seqlens_k.to(torch.int32),
                max_seqlen_q=max(query_lens).item(),
                max_seqlen_k=max(key_values_lens).item(),
                causal=is_causal,
            )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        if mode == "und":
            packed_attn_output = self.o_proj(packed_attn_output)
        elif mode == "gen":
            packed_attn_output[packed_text_indexes] = self.o_proj(
                packed_attn_output[packed_text_indexes]
            )
            packed_attn_output[packed_vae_token_indexes] = self.o_proj_moe_gen(
                packed_attn_output[packed_vae_token_indexes]
            )

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = PackedAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:

        residual = packed_sequence
        packed_sequence = self.input_layernorm(packed_sequence)

        # Self Attention
        packed_sequence = self.self_attn(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
        )
        packed_sequence = residual + packed_sequence

        # Fully Connected
        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)
        packed_sequence = self.mlp(packed_sequence)
        packed_sequence = residual + packed_sequence

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
    ) -> BaseNavitOutputWithPast:

        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence)

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        packed_query_sequence = self.mlp(packed_query_sequence)
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


class Qwen2MoTDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: Optional[int] = None,
        attn_module: Optional[Qwen2Attention] = PackedAttentionMoT,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.freeze_und = config.freeze_und

        self.self_attn = attn_module(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm_moe_gen = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:

        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_und_token_indexes] = self.input_layernorm(
            packed_sequence[packed_und_token_indexes]
        )
        packed_sequence_[packed_gen_token_indexes] = self.input_layernorm_moe_gen(
            packed_sequence[packed_gen_token_indexes]
        )

        # Self Attention
        packed_sequence_ = self.self_attn(
            packed_sequence=packed_sequence_,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        if self.freeze_und:
            packed_sequence_[packed_und_token_indexes] = packed_sequence_[
                packed_und_token_indexes
            ].detach()
        packed_sequence = residual + packed_sequence_

        # Fully Connected
        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_und_token_indexes] = self.mlp(
            self.post_attention_layernorm(packed_sequence[packed_und_token_indexes])
        )
        if self.freeze_und:
            packed_sequence_[packed_und_token_indexes] = packed_sequence_[
                packed_und_token_indexes
            ].detach()

        packed_sequence_[packed_gen_token_indexes] = self.mlp_moe_gen(
            self.post_attention_layernorm_moe_gen(
                packed_sequence[packed_gen_token_indexes]
            )
        )
        packed_sequence = residual + packed_sequence_

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:

        enable_taylorseer = getattr(self, "enable_taylorseer", False)

        if enable_taylorseer and self.current["type"] == "full":
            self.current["module"] = "total"
            taylor_cache_init(cache_dic=self.cache_dic, current=self.current)

        if not enable_taylorseer or (
            enable_taylorseer and self.current["type"] == "full"
        ):
            residual = packed_query_sequence
            if mode == "und":
                packed_query_sequence = self.input_layernorm(packed_query_sequence)
            elif mode == "gen":
                packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
                packed_query_sequence_[packed_text_indexes] = self.input_layernorm(
                    packed_query_sequence[packed_text_indexes]
                )
                packed_query_sequence_[packed_vae_token_indexes] = (
                    self.input_layernorm_moe_gen(
                        packed_query_sequence[packed_vae_token_indexes]
                    )
                )
                packed_query_sequence = packed_query_sequence_

            # Self Attention
            packed_query_sequence, past_key_values = self.self_attn(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                mode=mode,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_text_indexes=packed_text_indexes,
            )
            packed_query_sequence = residual + packed_query_sequence

            # Fully Connected
            residual = packed_query_sequence
            if mode == "und":
                packed_query_sequence = self.post_attention_layernorm(
                    packed_query_sequence
                )
                packed_query_sequence = self.mlp(packed_query_sequence)
            elif mode == "gen":
                packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
                packed_vae_query_sequence = packed_query_sequence[
                    packed_vae_token_indexes
                ]
                packed_text_query_sequence = self.post_attention_layernorm(
                    packed_text_query_sequence
                ).to(torch.bfloat16)
                packed_vae_query_sequence = self.post_attention_layernorm_moe_gen(
                    packed_vae_query_sequence
                ).to(torch.bfloat16)

                packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(
                    torch.bfloat16
                )
                packed_query_sequence_[packed_text_indexes] = self.mlp(
                    packed_text_query_sequence
                )
                packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(
                    packed_vae_query_sequence
                )
                packed_query_sequence = packed_query_sequence_

            packed_query_sequence = residual + packed_query_sequence

        if enable_taylorseer:
            if self.current["type"] == "full":
                derivative_approximation(
                    cache_dic=self.cache_dic,
                    current=self.current,
                    feature=packed_query_sequence,
                )
            elif self.current["type"] == "Taylor":
                self.current["module"] = "total"
                packed_query_sequence = taylor_formula(
                    cache_dic=self.cache_dic, current=self.current
                )

        return packed_query_sequence, past_key_values


class Qwen2MoEDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = PackedAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:

        residual = packed_sequence
        packed_sequence = self.input_layernorm(packed_sequence)

        # Self Attention
        packed_sequence = self.self_attn(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
        )
        packed_sequence = residual + packed_sequence

        # Fully Connected
        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)

        packed_sequence_new = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_und = self.mlp(packed_sequence[packed_und_token_indexes])
        packed_sequence_gen = self.mlp_moe_gen(
            packed_sequence[packed_gen_token_indexes]
        )
        packed_sequence_new[packed_und_token_indexes] = packed_sequence_und
        packed_sequence_new[packed_gen_token_indexes] = packed_sequence_gen

        packed_sequence = residual + packed_sequence_new

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:

        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence)

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        if mode == "und":
            packed_query_sequence = self.mlp(packed_query_sequence)
        elif mode == "gen":
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(
                torch.bfloat16
            )
            packed_query_sequence_[packed_text_indexes] = self.mlp(
                packed_query_sequence[packed_text_indexes]
            )
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(
                packed_query_sequence[packed_vae_token_indexes]
            )
            packed_query_sequence = packed_query_sequence_
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


Decoder_layer_dict = {
    "Qwen2DecoderLayer": Qwen2DecoderLayer,
    "Qwen2MoEDecoderLayer": Qwen2MoEDecoderLayer,
    "Qwen2MoTDecoderLayer": partial(
        Qwen2MoTDecoderLayer, attn_module=PackedAttentionMoT
    ),
}


class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_moe = "Mo" in config.layer_module

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        layer_module = Decoder_layer_dict[config.layer_module]
        self.layers = nn.ModuleList(
            [
                layer_module(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_moe:
            self.norm_moe_gen = Qwen2RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        if self.config.freeze_und:
            packed_sequence[packed_und_token_indexes] = packed_sequence[
                packed_und_token_indexes
            ].detach()

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(packed_sequence, packed_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_moe:
            assert packed_und_token_indexes is not None
            if packed_gen_token_indexes is None:
                packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )

        for decoder_layer in self.layers:
            packed_sequence = decoder_layer(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                **extra_inputs,
            )

        if self.use_moe:
            packed_sequence_ = torch.zeros_like(packed_sequence)
            packed_sequence_[packed_und_token_indexes] = self.norm(
                packed_sequence[packed_und_token_indexes]
            )
            if self.config.freeze_und:
                packed_sequence_[packed_und_token_indexes] = packed_sequence_[
                    packed_und_token_indexes
                ].detach()
            packed_sequence_[packed_gen_token_indexes] = self.norm_moe_gen(
                packed_sequence[packed_gen_token_indexes]
            )
            return packed_sequence_
        else:
            return self.norm(packed_sequence)

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:

        enable_taylorseer = getattr(self, "enable_taylorseer", False)
        if enable_taylorseer:
            cal_type(self.cache_dic, self.current)
            self.current["stream"] = "layers_stream"

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(
            packed_query_sequence, packed_query_position_ids.unsqueeze(0)
        )
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_query_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs.update(mode=mode)
            if mode == "gen":
                assert packed_vae_token_indexes is not None
                assert packed_text_indexes is not None
                extra_inputs.update(
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_text_indexes=packed_text_indexes,
                )

        for layer_idx, decoder_layer in enumerate(self.layers):
            if enable_taylorseer:
                decoder_layer.current = self.current
                decoder_layer.cache_dic = self.cache_dic
                decoder_layer.enable_taylorseer = True
                self.current["layer"] = layer_idx
            packed_query_sequence, past_key_values = decoder_layer(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                **extra_inputs,
            )

        if self.use_moe:
            if mode == "und":
                packed_query_sequence = self.norm(packed_query_sequence)
            elif mode == "gen":
                packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
                packed_query_sequence_[packed_text_indexes] = self.norm(
                    packed_query_sequence[packed_text_indexes]
                )
                packed_query_sequence_[packed_vae_token_indexes] = self.norm_moe_gen(
                    packed_query_sequence[packed_vae_token_indexes]
                )
                packed_query_sequence = packed_query_sequence_
        else:
            packed_query_sequence = self.norm(packed_query_sequence)

        if enable_taylorseer:
            self.current["step"] += 1
        # torch.save(past_key_values.key_cache[27],"past_key_cache_after_decode.pt")
        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
        )


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def init_moe(self):
        for name, param in self.named_parameters():
            if "moe_gen" in name:
                original_name = name.replace("_moe_gen", "")
                param.data.copy_(self.state_dict()[original_name].data)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        outputs = self.model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        return outputs

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:
        # langugage model forward pass
        outputs = self.model(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )

        return outputs
