# Calibrate the early diffusion gate

## 1. Prepare the environment and development samples

Run all commands below from the repository root after setting the checkpoint and data variables shown in the README. Use the same MathVision development split and sample IDs for both branches. Keep a separate evaluation split for reporting performance after the thresholds are fixed.

## 2. Run the baseline and collect metrics

Enable metric collection, then run the visual baseline:

```bash
export BAGEL_METRIC_DUMP=1
export BAGEL_METRIC_JSONL_ALL=1
export BAGEL_METRIC_SAVE_TENSORS=0
export BAGEL_METRIC_DUMP_DIR="$PWD/analysis/runtime_metrics"
export BAGEL_METRIC_EXACT_ATTN=1
export BAGEL_METRIC_SEGMENT_MASS=1
export BAGEL_METRIC_SAVE_IMAGEGEN_INTERNAL_STEPS=1
export BAGEL_METRIC_IMAGEGEN_SAMPLE_STEPS=1,3,5,7,9
bash scripts/infer.sh baseline mathvision
```

The runner configures the metric run group from the output directory name and the run ID from the dataset/run description. With the README defaults, metrics are written under `analysis/runtime_metrics/results/mathvision/mathvision_baseline/`.

## 3. Run the text-only counterfactual and evaluate both outputs

Run the text-only branch on the same development samples with metric collection disabled, then evaluate both branches:

```bash
BAGEL_METRIC_DUMP=0 bash scripts/infer.sh text-only mathvision
python BAGEL-Canvas/evaluation/eval_mathvision.py --result_dir results/mathvision/mathvision_baseline
python BAGEL-Canvas/evaluation/eval_mathvision.py --result_dir results/mathvision/mathvision_text-only
```

## 4. Align the results and build the calibration dataset

Set the baseline result, text-only result, and baseline metric directories. If you changed `OUTPUT_DIR` or `RUN_NAME`, update these three paths accordingly. Check the actual metric directory before alignment.

```bash
export ADAVIG_OFF_RESULTS_DIR="$PWD/results/mathvision/mathvision_baseline"
export ADAVIG_WO_RESULTS_DIR="$PWD/results/mathvision/mathvision_text-only"
export ADAVIG_OFF_METRICS_DIR="$PWD/analysis/runtime_metrics/results/mathvision/mathvision_baseline"
python calibration/build_gate_dataset.py
```

`analysis/adavig_gate_dataset/aligned_dataset.csv` contains aligned correctness, cost estimates, and early attention features. Labels are `helpful` (visual branch correct, text-only wrong), `harmful` (the reverse), `neutral_keep` (both correct), and `neutral_skip` (both wrong).

## 5. Search the two gate thresholds

Run the threshold search on the aligned development dataset:

```bash
python calibration/search_pre_gate_thresholds.py
```

The threshold search varies only the two signals used by the deployed gate. It writes `results.csv` and `summary.json` under `analysis/adavig_pre_gate_search/`, ranked by estimated accuracy and then counterfactual cost. Review these files and select the threshold pair to validate online. It does not fit a model or use gradients. Calibration on the same split used for reporting would bias the reported performance; fix thresholds on a development split before evaluation.

The coarse search uses the 10th, 30th, 50th, 70th, and 90th percentiles of gate-applicable samples, rounded to increments of `0.1` for intent and `0.01` for fidelity, with duplicate candidates removed. It also includes the default thresholds (`0.60`, `0.01`). The three best coarse pairs are refined with offsets of `-0.05`, `0`, and `+0.05` for intent, and `-0.005`, `0`, and `+0.005` for fidelity. All thresholds stay within `[0, 1]`, and repeated pairs are evaluated only once. This evaluates at most 60 distinct pairs. `results.csv` identifies coarse and fine candidates; `summary.json` also reports the default configuration, always-keep, and skip-all-applicable reference estimates.

To validate the top-ranked pair, read the `best` object in `summary.json` and set the runtime environment variables as follows:

| Search result field | Runtime environment variable |
| --- | --- |
| `best.intent_threshold` | `ADAVIG_PRE_R_ALL_THR` |
| `best.fidelity_threshold` | `ADAVIG_PRE_ORIG_VIT_THR` |

Replace both default values in the online validation commands below with these selected values. If you choose another pair from `results.csv`, use its `intent_threshold` and `fidelity_threshold` values with the same mapping.

## 6. Validate the selected thresholds online

Validate the chosen thresholds online with diagnostic logging disabled:

```bash
export BAGEL_METRIC_DUMP=0
export BAGEL_METRIC_EXACT_ATTN=0
export BAGEL_METRIC_SEGMENT_MASS=0
export ADAVIG_PRE_R_ALL_THR=0.60
export ADAVIG_PRE_ORIG_VIT_THR=0.01
RUN_NAME=mathvision_gated_validation bash scripts/infer.sh gated mathvision
```

The two threshold values above are the supplied runtime defaults. Replace them with your selected values when calibrating a different setup.

Evaluate the gated output to check its actual accuracy:

```bash
python BAGEL-Canvas/evaluation/eval_mathvision.py --result_dir results/mathvision/mathvision_gated_validation
```

Update the result path if you changed the output directory or validation run name. Compare the online result with the baseline and text-only results. Once thresholds are fixed, use the held-out evaluation split to report performance.
