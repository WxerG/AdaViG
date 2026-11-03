# Data Preparation

Download the benchmarks from the links below. Refer to [LMMS-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks) for their dataset configurations and splits.

| Benchmark | Dataset | Split / subset |
| --- | --- | --- |
| MathVision | [MathLLMs/MathVision](https://huggingface.co/datasets/MathLLMs/MathVision) | `testmini` for calibration |
| MathVista | [AI4Math/MathVista](https://huggingface.co/datasets/AI4Math/MathVista) | `testmini`, GPS subset |
| MathVerse | [CaraJ/MathVerse-lmmseval](https://huggingface.co/datasets/CaraJ/MathVerse-lmmseval) | `testmini` |
| OlympiadBench | [Hothan/OlympiadBench](https://huggingface.co/datasets/Hothan/OlympiadBench) | Math subset |
| WeMath | [We-Math/We-Math](https://huggingface.co/datasets/We-Math/We-Math) | `testmini` |
| DynaMath | [kcz358/DynaMath](https://huggingface.co/datasets/kcz358/DynaMath) | `test` |

## Input format

Save the images locally and prepare a JSONL file with one question per line:

```json
{"id": "example_0", "question": "What is the marked angle?", "options": ["30", "45", "60", "90"], "answer": "B", "image": "images/example_0.png"}
```

`image` is a path relative to `IMAGE_ROOT`, or a list of paths for multiple images. Use `options: []` for questions without separate choices. Preserve the original sample order and benchmark metadata, including WeMath's `key` and DynaMath's `seed_id`, for evaluation.

## Set data paths

For example, with MathVision:

```bash
export INPUT_PATH=/path/to/mathvision/testmini.jsonl
export IMAGE_ROOT=/path/to/mathvision

bash scripts/infer.sh gated mathvision
```

See the [README](../README.md#inference) for the checkpoint setting and other inference modes.
