"""Shared command-line setup and one-pass dataset inference."""

import argparse
import json
import os
import traceback
from typing import Any, Dict, List


def _sample_output_dirname(item_id: Any, sample_row_index: int) -> str:
    return f"{int(sample_row_index):05d}__{str(item_id)}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a single reasoning trajectory per sample.",
        epilog="Save reasoning steps, generated images, and profiling per sample.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=["mathcanvas", "uni"],
        help="Dataset type; picks the appropriate handler.",
    )
    parser.add_argument(
        "--dataset_name", type=str, help="Dataset name (default: matches type)."
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Input data file path."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True, help="Checkpoint directory."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Where to store outputs."
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="model.safetensors",
        help="Checkpoint filename.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Base model config/tokenizer path.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Root directory for images referenced in the dataset.",
    )
    parser.add_argument(
        "--add_timestamp",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Append timestamp to output directory.",
    )
    parser.add_argument(
        "--run_description",
        type=str,
        default="",
        help="Optional suffix for the run directory.",
    )
    parser.add_argument(
        "--verbose_iter",
        action="store_true",
        help="Show reasoning progress bar per sample.",
    )
    parser.add_argument(
        "--verbose_image", action="store_true", help="Show image denoising progress."
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=10,
        help="Maximum text and successful image generation rounds per sample; skipped or gate-aborted images do not consume rounds.",
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip samples that already have outputs.",
    )
    parser.add_argument(
        "--do_sample",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Use sampling for text generation.",
    )
    parser.add_argument(
        "--text_temperature", type=float, default=0.7, help="Text sampling temperature."
    )
    parser.add_argument(
        "--cfg_text_scale", type=float, default=4.0, help="CFG scale for text prompts."
    )
    parser.add_argument(
        "--cfg_img_scale", type=float, default=2.0, help="CFG scale for image prompts."
    )
    parser.add_argument(
        "--cfg_interval",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        help="CFG application interval in [0,1].",
    )
    parser.add_argument(
        "--timestep_shift",
        type=float,
        default=3.0,
        help="Flow timestep shift parameter.",
    )
    parser.add_argument(
        "--num_timesteps", type=int, default=50, help="Number of denoising timesteps."
    )
    parser.add_argument(
        "--cfg_renorm_min", type=float, default=0.0, help="Minimum CFG renorm factor."
    )
    parser.add_argument(
        "--cfg_renorm_type",
        type=str,
        default="text_channel",
        choices=["global", "channel", "text_channel"],
        help="CFG renorm strategy.",
    )
    parser.add_argument(
        "--enable_taylorseer",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Enable TaylorSeer caching.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed set once at the start of the run.",
    )
    parser.add_argument(
        "--disable_tf32",
        action="store_true",
        help="Disable TF32 for debugging precision differences across GPUs.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic CUDA algorithms for debugging.",
    )
    return parser.parse_args(argv)


def run_sample(
    reasoner, handler, item, output_dir, *, dataset_name, image_root, **kwargs
):
    """Generate and save exactly one trajectory for a dataset record."""
    inputs = handler.prepare_input(
        item, dataset_name=dataset_name, image_root=image_root
    )
    steps, profiling = reasoner.run(
        inputs=inputs, system_prompt=None, sample_id=str(item.get("id")), **kwargs
    )
    steps = [
        step
        for step in steps
        if not (step.get("type") == "image" and step.get("content") is None)
    ]
    handler.save_results(
        steps,
        item,
        output_dir,
        image_root=image_root,
        profiling_log=profiling,
        dataset_name=dataset_name,
    )


def run_inference(reasoner_type, argv=None):
    from . import base
    import torch
    from tqdm.auto import tqdm
    from runtime_metrics import get_metric_dumper
    from data_handlers import MathCanvasHandler, UniHandler

    args = parse_args(argv)

    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.deterministic:
        torch.use_deterministic_algorithms(True)

    args.add_timestamp = args.add_timestamp.lower() == "true"
    args.do_sample = args.do_sample.lower() == "true"
    args.enable_taylorseer = args.enable_taylorseer.lower() == "true"

    rank = base.setup_distributed()
    world_size = base.d_world_size()
    base.set_random_seed(args.seed)

    if args.dataset_type == "mathcanvas":
        args.dataset_name = "mathcanvas"
        handler = MathCanvasHandler()
    elif args.dataset_type == "uni":
        handler = UniHandler()
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    INFERENCE_HYPERPARAMS = dict(
        do_sample=args.do_sample,
        text_temperature=args.text_temperature,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=args.cfg_interval,
        timestep_shift=args.timestep_shift,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=args.cfg_renorm_min,
        cfg_renorm_type=args.cfg_renorm_type,
        enable_taylorseer=args.enable_taylorseer,
    )
    if rank == 0:
        print("Inference hyperparameters:")
        for k, v in INFERENCE_HYPERPARAMS.items():
            print(f"  {k}: {v}")

    output_path = os.path.join(args.output_dir, args.dataset_name)
    if args.run_description:
        output_path = os.path.join(
            output_path, args.run_description
        )  # f"{output_path}_{args.run_description}"
    if args.add_timestamp:
        sync_timestamp = base.get_synchronized_timestamp()
        output_path = os.path.join(
            output_path, sync_timestamp
        )  # f"{output_path}_{sync_timestamp}"
    metric_dumper = get_metric_dumper()
    metric_run_group = os.path.basename(os.path.normpath(args.output_dir))
    metric_run_id = os.path.relpath(output_path, args.output_dir)
    metric_dumper.configure_run(run_group=metric_run_group, run_id=metric_run_id)
    if rank == 0 and metric_dumper.enabled:
        print(
            f"Metric dump dir: {metric_dumper.dump_dir / metric_run_group / metric_run_id}"
        )

    if rank == 0:
        os.makedirs(output_path, exist_ok=True)
        args_file_path = os.path.join(output_path, "inference_args.json")
        current_args_dict = vars(args)
        if os.path.exists(args_file_path):
            print(
                f"Found existing arguments file: {args_file_path}. Verifying consistency..."
            )
            try:
                with open(args_file_path, "r", encoding="utf-8") as f:
                    previous_args_dict = json.load(f)
                comp_prev = previous_args_dict.copy()
                comp_curr = current_args_dict.copy()
                for key in ["verbose_iter", "verbose_image", "input_path"]:
                    comp_prev.pop(key, None)
                    comp_curr.pop(key, None)
                if comp_prev != comp_curr:
                    print("\nERROR: Argument mismatch with previous run.", flush=True)
                    base.d_barrier()
                    return
                else:
                    print("Arguments are consistent. Resuming inference run.")
            except (json.JSONDecodeError, OSError) as exc:
                print(
                    f"Warning: Could not parse existing arguments file: {exc}. Overwriting."
                )
                with open(args_file_path, "w", encoding="utf-8") as f:
                    json.dump(current_args_dict, f, indent=4)
        else:
            print("No existing arguments file found. Saving current arguments.")
            with open(args_file_path, "w", encoding="utf-8") as f:
                json.dump(current_args_dict, f, indent=4)
    base.d_barrier()

    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = (
        base.setup_model(
            args.checkpoint_dir, rank, args.checkpoint_file, args.model_path
        )
    )
    inferencer = base.InterleaveInferencer(
        model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids
    )
    reasoner = reasoner_type(inferencer)

    all_eval_data: List[Dict[str, Any]] = []
    if rank == 0:
        print(
            f"Using '{args.dataset_type}' handler to load data from {args.input_path}..."
        )
        all_eval_data = handler.load_data(args.input_path)
        for row_index, item in enumerate(all_eval_data):
            item["_sample_row_index"] = row_index
            item["_sample_output_dirname"] = _sample_output_dirname(
                item.get("id"), row_index
            )
        print(f"Loaded {len(all_eval_data)} total items.")
        if not args.skip_completed:
            print("Filtering out already completed items...")
            initial_count = len(all_eval_data)
            all_eval_data = [
                item
                for item in all_eval_data
                if not os.path.exists(
                    os.path.join(
                        output_path,
                        str(item.get("_sample_output_dirname")),
                        "reasoning_result.json",
                    )
                )
            ]
            print(
                f"Skipped {initial_count - len(all_eval_data)} completed items. {len(all_eval_data)} items remaining."
            )
        print(f"Processing all {len(all_eval_data)} items without sampling.")
    if not base.DIST_ENABLED:
        eval_data_to_process = all_eval_data
    else:
        data_to_distribute = [all_eval_data] if rank == 0 else [None]
        base.d_broadcast_object_list(data_to_distribute, src=0)
        eval_data_to_process = data_to_distribute[0]

    if not base.DIST_ENABLED:
        local_eval_data = eval_data_to_process
        start_index, end_index = 0, len(local_eval_data)
    else:
        total_items = len(eval_data_to_process)
        items_per_gpu = (total_items + world_size - 1) // world_size
        start_index = rank * items_per_gpu
        end_index = min(start_index + items_per_gpu, total_items)
        local_eval_data = eval_data_to_process[start_index:end_index]

    print(
        f"Rank {rank}/{world_size}: Processing {len(local_eval_data)} items (from index {start_index} to {end_index - 1})."
    )
    base.d_barrier()

    progress_bar = (
        tqdm(local_eval_data, desc=f"Rank {rank} Inference", position=rank)
        if not args.verbose_iter
        else tqdm(local_eval_data, desc=f"Rank {rank} Inference", disable=(rank != 0))
    )

    for local_idx, item in enumerate(progress_bar):
        item_id = item.get("id", f"item_{base.timestamp()}")
        sample_index = int(item.get("_sample_row_index", start_index + local_idx))
        item_output_dir = os.path.join(
            output_path,
            str(
                item.get(
                    "_sample_output_dirname",
                    _sample_output_dirname(item_id, sample_index),
                )
            ),
        )
        if rank == 0:
            os.makedirs(item_output_dir, exist_ok=True)
        base.d_barrier()

        try:
            run_sample(
                reasoner,
                handler,
                item,
                item_output_dir,
                dataset_name=args.dataset_name,
                image_root=args.image_root,
                max_iterations=args.max_iterations,
                verbose_iter=args.verbose_iter,
                verbose_image=args.verbose_image,
                inference_kwargs=INFERENCE_HYPERPARAMS,
            )
        except Exception as exc:
            print(f"Rank {rank} failed to process item {item_id}. Error: {exc}")
            traceback.print_exc()
            continue

    print(f"Rank {rank} has completed all assigned tasks.")
    metric_dumper.flush()  # wait for all async jsonl writes to finish
    base.d_barrier()
    if rank == 0:
        print("\nInference completed across all GPUs!")
