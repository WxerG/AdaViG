#!/usr/bin/env bash
# One GPU per process; set CUDA_VISIBLE_DEVICES to select the device.
set -euo pipefail
ADAVIG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-gated}"
DATASET="${2:-mathvision}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the BAGEL-Canvas checkpoint directory}"
: "${INPUT_PATH:?Set INPUT_PATH to the benchmark JSONL file}"
: "${IMAGE_ROOT:?Set IMAGE_ROOT to the directory containing benchmark images}"
RUN_NAME="${RUN_NAME:-${DATASET}_${MODE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ADAVIG_ROOT}/results}"
INPUT_PATH="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "${INPUT_PATH}")"
IMAGE_ROOT="$(cd "${IMAGE_ROOT}" && pwd)"
CHECKPOINT_DIR="$(cd "${CHECKPOINT_DIR}" && pwd)"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
export BAGEL_ONLINE_GATE_POLICY=""
case "${MODE}" in
  baseline) MODULE=search.inference; DEFAULT_TEMPERATURE=0.7 ;;
  text-only) MODULE=search.inference_text_only; DEFAULT_TEMPERATURE=0.7 ;;
  gated)
    MODULE=search.inference; DEFAULT_TEMPERATURE=0.7
    export BAGEL_ONLINE_GATE_POLICY=adavig_pre
    export BAGEL_ONLINE_GATE_METRIC_PATH="${OUTPUT_DIR}/${DATASET}/${RUN_NAME}/gate_metrics.jsonl"
    ;;
  *) echo "Usage: bash scripts/infer.sh {baseline|text-only|gated} DATASET" >&2; exit 2 ;;
esac
cd "${ADAVIG_ROOT}/BAGEL-Canvas"
torchrun --nproc_per_node=1 --master_port="${MASTER_PORT:-23455}" -m "${MODULE}" \
  --dataset_type uni --dataset_name "${DATASET}" \
  --input_path "${INPUT_PATH}" --image_root "${IMAGE_ROOT}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" --model_path "${CHECKPOINT_DIR}" \
  --checkpoint_file model.safetensors \
  --output_dir "${OUTPUT_DIR}" --run_description "${RUN_NAME}" --add_timestamp false \
  --max_iterations 10 --do_sample true --text_temperature "${TEXT_TEMPERATURE:-${DEFAULT_TEMPERATURE}}" \
  --cfg_text_scale 4.0 --cfg_img_scale 2.0 --cfg_interval 0.0 1.0 \
  --timestep_shift 3.0 --num_timesteps 50 --cfg_renorm_min 0.0 --cfg_renorm_type text_channel \
  --enable_taylorseer false --seed 42
