"""CPU tests for the single-trajectory runner with model execution mocked."""

import contextlib
import importlib
import io
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "BAGEL-Canvas"))
from search.runner import parse_args, run_sample, run_inference
from online_gate import OnlineGateAbortImageGeneration


class RunnerTests(unittest.TestCase):
    def test_cli_uses_matched_temperature_and_rejects_removed_options(self):
        required = [
            "--dataset_type",
            "uni",
            "--input_path",
            "data.jsonl",
            "--checkpoint_dir",
            "weights",
            "--output_dir",
            "results",
            "--model_path",
            "weights",
            "--image_root",
            "images",
        ]
        args = parse_args(required)
        self.assertEqual(args.text_temperature, 0.7)
        self.assertEqual(args.seed, 42)
        self.assertFalse(hasattr(args, "seed_mode"))
        self.assertFalse(hasattr(args, "seed_stride"))
        for removed in (
            ["--best_of_n", "1"],
            ["--bon_logprob_mode", "answer"],
            ["--bon_seed_offset", "0"],
            ["--seed_mode", "global"],
            ["--seed_mode", "per_sample"],
            ["--seed_stride", "1000"],
        ):
            with (
                self.subTest(option=removed),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    parse_args(required + removed)

    def test_seed_is_set_once_for_the_whole_dataset(self):
        import search

        base = types.ModuleType("search.base")
        base.setup_distributed = lambda: 0
        base.d_world_size = lambda: 1
        base.set_random_seed = Mock()
        base.setup_model = Mock(return_value=(None,) * 6)
        base.InterleaveInferencer = Mock()
        base.DIST_ENABLED = False
        base.d_barrier = lambda: None
        base.timestamp = lambda: "test"
        handler = Mock()
        handler.load_data.return_value = [{"id": "one"}, {"id": "two"}]
        handler.prepare_input.return_value = ["question"]
        handlers = types.ModuleType("data_handlers")
        handlers.UniHandler = lambda: handler
        handlers.MathCanvasHandler = lambda: handler
        reasoner = Mock()
        reasoner.run.return_value = ([{"type": "text", "content": "answer"}], [])
        factory = Mock(return_value=reasoner)
        metrics = types.ModuleType("runtime_metrics")
        dumper = Mock(enabled=False)
        metrics.get_metric_dumper = lambda: dumper
        progress = types.ModuleType("tqdm.auto")
        progress.tqdm = lambda items, **kwargs: items
        substitutes = {
            "search.base": base,
            "torch": types.ModuleType("torch"),
            "data_handlers": handlers,
            "runtime_metrics": metrics,
            "tqdm.auto": progress,
        }
        with tempfile.TemporaryDirectory() as directory:
            args = [
                "--dataset_type",
                "uni",
                "--dataset_name",
                "mathvision",
                "--input_path",
                "data.jsonl",
                "--checkpoint_dir",
                "weights",
                "--output_dir",
                directory,
                "--model_path",
                "weights",
                "--image_root",
                "images",
                "--add_timestamp",
                "false",
                "--seed",
                "123",
            ]
            with (
                patch.dict(sys.modules, substitutes),
                patch.object(search, "base", base, create=True),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                run_inference(factory, args)
        base.set_random_seed.assert_called_once_with(123)
        self.assertEqual(reasoner.run.call_count, 2)
        self.assertEqual(handler.save_results.call_count, 2)

    def test_one_generation_and_one_save(self):
        reasoner = Mock()
        steps = [
            {"type": "text", "content": "answer"},
            {"type": "image", "content": None, "skipped": True},
        ]
        profiling = [{"stage": "text_decode", "elapsed_ms": 10}]
        reasoner.run.return_value = (steps, profiling)
        handler = Mock()
        handler.prepare_input.return_value = ["question"]
        item = {"id": "example"}
        with tempfile.TemporaryDirectory() as directory:
            run_sample(
                reasoner,
                handler,
                item,
                directory,
                dataset_name="mathvision",
                image_root="images",
                inference_kwargs={"text_temperature": 0.7},
            )
            reasoner.run.assert_called_once()
            handler.save_results.assert_called_once_with(
                [steps[0]],
                item,
                directory,
                image_root="images",
                profiling_log=profiling,
                dataset_name="mathvision",
            )
            self.assertEqual(list(Path(directory).iterdir()), [])


class DummyImage:
    pass


class DummyProgress:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, *args):
        pass

    def set_description(self, *args):
        pass


class DummyBase:
    def __init__(self, inferencer):
        self.inferencer = inferencer
        self.tokenizer = inferencer.tokenizer
        self.action_token_map = {99: "image", 100: "text", 101: "end"}

    def _profile_call(self, events, stage, function, *args, **kwargs):
        result = function(*args, **kwargs)
        events.append({"stage": stage, "elapsed_ms": 0.0})
        return result

    def _count_text_tokens(self, text):
        return len(text)

    def _kv_length(self, context):
        return 0

    def _text_prefill_meta(self, *args):
        return {}

    def _visual_prefill_meta(self, *args):
        return {}


class DummyInferencer:
    def __init__(self, actions, abort=False):
        self.actions = iter(actions)
        self.abort = abort
        self.image_calls = 0
        self.text_options = []
        self.tokenizer = types.SimpleNamespace(
            decode=lambda *args, **kwargs: "reasoning"
        )
        self.vae_transform = types.SimpleNamespace(resize_transform=lambda image: image)

    def init_gen_context(self):
        return {"trace": {}, "kv_lens": [0]}

    def update_trace(self, gen_context, **kwargs):
        gen_context["trace"].update(kwargs)

    def update_context_text(self, text, context, **kwargs):
        return context

    def update_context_image(self, image, context, **kwargs):
        return context

    def gen_text(self, context, **kwargs):
        self.text_options.append(kwargs)
        return [0, 1, 98, next(self.actions)]

    def gen_image(self, *args, **kwargs):
        self.image_calls += 1
        if self.abort:
            raise OnlineGateAbortImageGeneration("abort at step 5")
        return DummyImage()


@contextlib.contextmanager
def mocked_model_runtime():
    """Load the real reasoning loops while replacing GPU/model dependencies."""
    torch = types.ModuleType("torch")
    torch.no_grad = lambda: lambda function: function
    torch.autocast = lambda **kwargs: contextlib.nullcontext()
    torch.bfloat16 = "bfloat16"
    torch.cuda = types.SimpleNamespace(
        nvtx=types.SimpleNamespace(
            range_push=lambda message: None, range_pop=lambda: None
        )
    )
    base = types.ModuleType("search.base")
    base.InterleaveReasoner = DummyBase
    base.d_rank = lambda: 0
    dumper = types.SimpleNamespace(should_write_jsonl=lambda sample_id: False)
    metrics = types.ModuleType("runtime_metrics")
    metrics.get_metric_dumper = lambda: dumper
    data_utils = types.ModuleType("data.data_utils")
    data_utils.pil_img2rgb = lambda image: image
    pil = types.ModuleType("PIL")
    pil.Image = types.SimpleNamespace(Image=DummyImage)
    progress = types.ModuleType("tqdm.auto")
    progress.tqdm = DummyProgress
    substitutes = {
        "torch": torch,
        "numpy": types.ModuleType("numpy"),
        "search.base": base,
        "runtime_metrics": metrics,
        "data.data_utils": data_utils,
        "PIL": pil,
        "tqdm.auto": progress,
    }
    with patch.dict(sys.modules, substitutes), patch.dict(os.environ, {}, clear=True):
        for name in ["search.common", "search.inference", "search.inference_text_only"]:
            sys.modules.pop(name, None)
        visual = importlib.import_module("search.inference")
        text_only = importlib.import_module("search.inference_text_only")
        yield visual.VisualReasoner, text_only.TextOnlyReasoner


class ReasoningFlowTests(unittest.TestCase):
    def run_reasoner(self, reasoner, max_iterations=8):
        return reasoner.run(
            inputs=["question"],
            system_prompt=None,
            max_iterations=max_iterations,
            verbose_iter=False,
            verbose_image=False,
            inference_kwargs={},
            sample_id="example",
        )

    def test_visual_trace_generates_one_image_and_uses_temperature_point_seven(self):
        with mocked_model_runtime() as (VisualReasoner, _):
            inferencer = DummyInferencer([99, 101])
            steps, _ = self.run_reasoner(VisualReasoner(inferencer))
            self.assertEqual(
                [step["type"] for step in steps], ["text", "image", "text"]
            )
            self.assertEqual(inferencer.image_calls, 1)
            self.assertEqual(len(inferencer.text_options), 2)
            for options in inferencer.text_options:
                self.assertEqual(options["temperature"], 0.7)
                self.assertNotIn("return_logprobs", options)

    def test_text_only_trace_never_calls_image_generation(self):
        with mocked_model_runtime() as (_, TextOnlyReasoner):
            inferencer = DummyInferencer([99, 101])
            steps, _ = self.run_reasoner(TextOnlyReasoner(inferencer))
            self.assertEqual(inferencer.image_calls, 0)
            self.assertTrue(
                all(
                    step.get("content") is None
                    for step in steps
                    if step["type"] == "image"
                )
            )

    def test_gate_abort_only_skips_current_image_and_matches_text_only_rounds(self):
        with mocked_model_runtime() as (VisualReasoner, TextOnlyReasoner):
            inferencer = DummyInferencer([99, 99, 101], abort=True)
            steps, profiling = self.run_reasoner(VisualReasoner(inferencer))
            self.assertEqual(inferencer.image_calls, 2)
            self.assertEqual([step["type"] for step in steps], ["text", "text", "text"])
            self.assertEqual(
                [event["iteration"] for event in profiling if event.get("gated_abort")],
                [2, 4],
            )
            text_steps, _ = self.run_reasoner(
                TextOnlyReasoner(DummyInferencer([99, 99, 101]))
            )
            self.assertEqual(
                [step["iteration"] for step in steps],
                [step["iteration"] for step in text_steps if step["type"] == "text"],
            )

    def test_later_image_can_succeed_after_gate_abort(self):
        with mocked_model_runtime() as (VisualReasoner, _):
            inferencer = DummyInferencer([99, 99, 101])
            with patch.object(
                inferencer,
                "gen_image",
                side_effect=[OnlineGateAbortImageGeneration("abort"), DummyImage()],
            ) as generate:
                steps, profiling = self.run_reasoner(VisualReasoner(inferencer))
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(
                [(step["type"], step["iteration"]) for step in steps],
                [("text", 1), ("text", 3), ("image", 4), ("text", 5)],
            )
            self.assertEqual(sum(bool(event.get("gated_abort")) for event in profiling), 1)

    def test_only_text_and_successful_images_consume_round_budget(self):
        with mocked_model_runtime() as (VisualReasoner, TextOnlyReasoner):
            for reasoner_type, abort, expected_image_calls, expected_text_calls in [
                (VisualReasoner, False, 1, 2),
                (VisualReasoner, True, 2, 3),
                (TextOnlyReasoner, False, 0, 3),
            ]:
                with self.subTest(mode=reasoner_type.__name__, abort=abort):
                    inferencer = DummyInferencer([99, 99, 101])
                    inferencer.abort = abort
                    steps, _ = self.run_reasoner(
                        reasoner_type(inferencer), max_iterations=3
                    )
                    self.assertEqual(
                        len([step for step in steps if step["type"] == "text"]),
                        expected_text_calls,
                    )
                    self.assertEqual(inferencer.image_calls, expected_image_calls)

    def test_image_generation_failure_propagates_without_resuming_text(self):
        with mocked_model_runtime() as (VisualReasoner, _):
            inferencer = DummyInferencer([99, 101])
            with patch.object(
                inferencer, "gen_image", side_effect=RuntimeError("generation failed")
            ), self.assertRaisesRegex(RuntimeError, "generation failed"):
                self.run_reasoner(VisualReasoner(inferencer))
            self.assertEqual(len(inferencer.text_options), 1)

    def test_image_context_failure_propagates_without_resuming_text(self):
        with mocked_model_runtime() as (VisualReasoner, _):
            inferencer = DummyInferencer([99, 101])
            with patch.object(
                inferencer,
                "update_context_image",
                side_effect=RuntimeError("image prefill failed"),
            ), self.assertRaisesRegex(RuntimeError, "image prefill failed"):
                self.run_reasoner(VisualReasoner(inferencer))
            self.assertEqual(len(inferencer.text_options), 1)


if __name__ == "__main__":
    unittest.main()
