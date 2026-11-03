import os
import time
from datetime import datetime
import random
import numpy as np

import torch
import torch.distributed as dist
from accelerate import (
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
    init_empty_weights,
)

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer


# =========================
# Single-GPU friendly dist wrappers
# =========================
DIST_ENABLED = False


def d_rank():
    return dist.get_rank() if DIST_ENABLED and dist.is_initialized() else 0


def d_world_size():
    return dist.get_world_size() if DIST_ENABLED and dist.is_initialized() else 1


def d_barrier():
    if DIST_ENABLED and dist.is_initialized():
        dist.barrier()


def d_broadcast_object_list(obj_list, src=0):
    if DIST_ENABLED and dist.is_initialized():
        dist.broadcast_object_list(obj_list, src=src)


def timestamp() -> str:
    dt_string = datetime.now().strftime("%m-%d_%H-%M-%S")
    return dt_string


def setup_distributed():
    """
    Single-GPU: do NOT init torch.distributed (no port, no nccl).
    Multi-GPU: init nccl as before.
    """
    global DIST_ENABLED
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size <= 1:
        DIST_ENABLED = False
        torch.cuda.set_device(0)
        return 0  # rank/local_rank

    DIST_ENABLED = True
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return local_rank


def get_synchronized_timestamp():
    # Single-GPU: local timestamp
    if not DIST_ENABLED:
        ts = timestamp()
        print("Current Timestamp :", ts)
        return ts

    # Multi-GPU: broadcast from rank0
    if d_rank() == 0:
        timestamp_str = timestamp()
        print("Current Timestamp :", timestamp_str)
    else:
        timestamp_str = None

    timestamp_list = [timestamp_str]
    d_broadcast_object_list(timestamp_list, src=0)
    return timestamp_list[0]


def set_random_seed(seed: int = 42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_model(checkpoint_dir: str, rank: int, checkpoint_file: str, model_path: str):
    """Loads and dispatches the model across devices."""
    device = f"cuda:{rank}"
    if rank == 0:
        print(f"Loading model from {checkpoint_dir}")
        print(f"Available GPUs for parallel execution: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

    # --- Load Model Configurations ---
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(
        os.path.join(model_path, "vit_config.json")
    )
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # Load VAE model directly onto the target device for this process.
    vae_model, vae_config = load_ae(
        local_path=os.path.join(model_path, "ae.safetensors")
    )
    vae_model = vae_model.to(device).eval()

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config, meta=True
        )

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 512, 14)

    max_mem_per_gpu = "80GiB"
    if rank == 0:
        print("Setting up device mapping...")

    # NOTE: For single GPU, this is fine. Multi-GPU mapping logic unchanged.
    device_map = infer_auto_device_map(
        model,
        max_memory={rank: max_mem_per_gpu},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        dtype=torch.bfloat16,
    )
    if rank == 0:
        print(f"Device map for rank 0 (others will be similar): {device_map}")

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]

    first_device = device_map.get(same_device_modules[0], device)
    for k in same_device_modules:
        device_map[k] = first_device if k in device_map else device

    if rank == 0:
        print(f"Loading checkpoint: {checkpoint_path}")

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_path,
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder=f"/tmp/offload_rank_{rank}",
    )
    model = model.eval()

    d_barrier()
    if rank == 0:
        print("Model loaded successfully across all processes!")

    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


class InterleaveReasoner:
    """A class to handle the multi-step reasoning process."""

    def __init__(self, inferencer: InterleaveInferencer):
        self.inferencer = inferencer
        self.tokenizer = inferencer.tokenizer
        self.new_token_ids = inferencer.new_token_ids

        self.action_token_map = {
            self.new_token_ids["start_of_image"]: "image",
            self.new_token_ids["bos_token_id"]: "text",
            self.new_token_ids["end_of_text"]: "end",
        }
        if d_rank() == 0:
            print("InterleaveReasoner initialized on each process.")

    def _profile_call(
        self, profiling_events, stage, func, *args, record=None, **kwargs
    ):
        profile_meta = {}
        if record is not None:
            extra_keys = [k for k in kwargs.keys() if k.startswith("profile_")]
            for extra_key in extra_keys:
                profile_meta[extra_key[len("profile_") :]] = kwargs.pop(extra_key)

        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        event = {"stage": stage, "elapsed_ms": elapsed_ms}
        if record is not None:
            extra = self._build_profile_record(record, result, profile_meta)
            if extra:
                event.update(extra)
        profiling_events.append(event)
        return result

    def _count_text_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False)) + 2  # BOS/EOS

    @staticmethod
    def _kv_length(context) -> int:
        kv_lens = context.get("kv_lens", [])
        return kv_lens[-1] if kv_lens else 0

    @staticmethod
    def _text_prefill_meta(context_label, label, iteration, text_tokens):
        return {
            "record": "text_prefill",
            "profile_context": context_label,
            "profile_label": label,
            "profile_iteration": iteration,
            "profile_text_tokens": text_tokens,
        }

    @staticmethod
    def _visual_prefill_meta(context_label, label, iteration, prev_tokens):
        return {
            "record": "visual_prefill",
            "profile_context": context_label,
            "profile_label": label,
            "profile_iteration": iteration,
            "profile_prev_tokens": prev_tokens,
        }

    def _build_profile_record(self, record, result_context, meta):
        if record == "text_prefill":
            return {
                "context": meta.get("context"),
                "label": meta.get("label"),
                "iteration": meta.get("iteration"),
                "text_tokens": meta.get("text_tokens"),
            }
        if record == "visual_prefill":
            prev_tokens = meta.get("prev_tokens")
            visual_tokens = None
            if prev_tokens is not None:
                visual_tokens = max(self._kv_length(result_context) - prev_tokens, 0)
            return {
                "context": meta.get("context"),
                "label": meta.get("label"),
                "iteration": meta.get("iteration"),
                "visual_tokens": visual_tokens,
            }
        return None
