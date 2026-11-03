<h1 align="center">🎨 AdaViG</h1>

<h2 align="center">Model Guides You How to Draw: Adaptive Visual Gating for Unified Multimodal Reasoning</h2>

<p align="center">
  Wenxi Gao<sup>1</sup>, Guanxi Lu<sup>1</sup>, Didi Zhu<sup>1</sup>, Hao Mark Chen<sup>1</sup>,<br>
  Quan Deng<sup>1,2</sup>, Zhican Wang<sup>1</sup>, Jiankang Deng<sup>1</sup>, Hongxiang Fan<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> Imperial College London &nbsp;&nbsp; <sup>2</sup> Tsinghua University
</p>

[Project page](https://xer-gwx.github.io/AdaViG/) · [Website source](https://github.com/Xer-GWX/AdaViG/tree/gh-pages)

AdaViG is a training-free method that decides whether to continue a visual reasoning step. At an early diffusion step, it measures **Generation Intent** (attention to the preceding text) and **Visual Fidelity** (attention to the input image). When both signals are weak, it stops image generation and resumes text reasoning.

## 🛠️ Setup

Please follow the instructions below to set up the environment required by AdaViG.

```bash
conda create -n adavig python=3.10 -y
conda activate adavig
pip install -r requirements.txt
pip install flash-attn==2.8.2 --no-build-isolation
```

FlashAttention must match your CUDA and PyTorch build. The attention instrumentation also imports its reference implementation at `flash_attn.flash_attn_triton_amd.fwd_ref`; retain that module when building FlashAttention.

### 📥 Download the checkpoint

Download [BAGEL-Canvas](https://huggingface.co/shiwk24/BAGEL-Canvas) from Hugging Face. Run the following command from the AdaViG repository root:

```bash
python - <<'PYTHON'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="shiwk24/BAGEL-Canvas",
    local_dir="checkpoints/BAGEL-Canvas",
    allow_patterns=["*.safetensors", "*.json", "*.txt"],
)
PYTHON
```

## 🚀 Inference

Run commands from the repository root. `INPUT_PATH` is a JSONL file; image paths in that file are resolved relative to `IMAGE_ROOT`. See [data preparation](docs/data.md).

```bash
export CHECKPOINT_DIR="$PWD/checkpoints/BAGEL-Canvas"
export INPUT_PATH=/path/to/mathvision/testmini.jsonl
export IMAGE_ROOT=/path/to/mathvision

bash scripts/infer.sh baseline mathvision
bash scripts/infer.sh text-only mathvision
bash scripts/infer.sh gated mathvision
```

The three modes run the original visual reasoning branch, its no-generated-image counterfactual, and AdaViG respectively. Results are saved under `results/<dataset>/<dataset>_<mode>/`. Use a new `RUN_NAME` for each independent run.

| Setting | Default |
| --- | --- |
| Decision step | 5 |
| Total diffusion steps | 50 |
| Intent threshold (`ADAVIG_PRE_R_ALL_THR`) | 0.60 |
| Fidelity threshold (`ADAVIG_PRE_ORIG_VIT_THR`) | 0.01 |

The runtime continues generation when Intent **exceeds** 0.60 **or** Fidelity **exceeds** 0.01. It aborts otherwise, including exact equality. See [implementation and reproduction notes](docs/reproduction.md) for aggregation details and the distinction from the paper's strict-inequality notation.

## 📊 Evaluation

```bash
cd BAGEL-Canvas
python evaluation/eval_dynamath.py --result_dir /path/to/dynamath/run --gt_path /path/to/test.jsonl
```

Evaluation scripts for the different benchmarks are available under `BAGEL-Canvas/evaluation/`.

## 🎛️ Threshold calibration

We select the gating thresholds using paired baseline and text-only results on MathVision testmini. See [calibration instructions](docs/calibration.md) for details.

## 🤝 Acknowledgments and license

Built on [MathCanvas](https://github.com/shiwk24/MathCanvas) and [BAGEL](https://github.com/ByteDance-Seed/Bagel). Some benchmark scoring functions are adapted from [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## 📜 Citation

If you find our work useful for your research, please consider citing our paper:

```bibtex
@misc{gao2026modelguidesdrawadaptive,
      title={Model Guides You How to Draw: Adaptive Visual Gating for Unified Multimodal Reasoning},
      author={Wenxi Gao and Guanxi Lu and Didi Zhu and Hao Mark Chen and Quan Deng and Zhican Wang and Jiankang Deng and Hongxiang Fan},
      year={2026},
      eprint={2607.10004},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.10004},
}
```
