import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "BAGEL-Canvas"))
from online_gate import OnlineGateController
from answer_utils import extract_answer


class GateTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(
            os.environ, {"BAGEL_ONLINE_GATE_POLICY": "adavig_pre"}, clear=True
        ):
            self.gate = OnlineGateController()
        self.trace = {
            "sample_id": "one",
            "trial_id": 0,
            "iteration": 2,
            "stage": "image_generation",
            "query_kind": "vae",
            "context": "gen",
            "diffusion_step_idx": 5,
        }

    def decide(self, text, vit, vae):
        for layer in (0, 14, 27):
            self.gate.record_segment_records(
                self.trace,
                layer,
                [
                    {"kind": "text", "source": "generated_text", "mass_mean": text},
                    {"kind": "vit", "source": "input_image", "mass_mean": vit},
                    {"kind": "vae", "source": "input_image", "mass_mean": vae},
                ],
            )
        return self.gate.maybe_decide_pre(self.trace)

    def test_both_weak_abort(self):
        self.assertFalse(self.decide(0.001, 0.001, 0.2)["choose_off"])

    def test_either_anchor_keeps_generation(self):
        self.assertTrue(self.decide(0.5, 0.001, 0.01)["choose_off"])
        self.setUp()
        self.assertTrue(self.decide(0.001, 0.1, 0.2)["choose_off"])

    def test_boundary_and_zero_masses(self):
        self.assertFalse(self.decide(0, 0.01, 0.5)["choose_off"])
        self.setUp()
        self.assertFalse(self.decide(0, 0, 0)["choose_off"])

    def test_attention_selection(self):
        self.assertTrue(self.gate.exact_attention_needed(self.trace, 14))
        for change, layer in [
            ({"context": "cfg_text"}, 14),
            ({"diffusion_step_idx": 4}, 14),
            ({}, 7),
            ({"stage": "text_decode"}, 27),
        ]:
            self.assertFalse(
                self.gate.exact_attention_needed(dict(self.trace, **change), layer)
            )

    def test_missing_metrics_and_trial_cleanup(self):
        self.assertIsNone(self.gate.maybe_decide_pre(self.trace))
        decision = self.decide(0.1, 0.1, 0.1)
        self.assertIs(self.gate.maybe_decide_pre(self.trace), decision)
        self.gate.finalize_trial("one", 0)
        self.assertIsNone(self.gate.maybe_decide_pre(self.trace))

    def test_disabled_and_invalid_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(OnlineGateController().enabled)
        with patch.dict(os.environ, {"BAGEL_ONLINE_GATE_POLICY": "typo"}, clear=True):
            with self.assertRaises(ValueError):
                OnlineGateController()
        with patch.dict(os.environ, {"ADAVIG_PRE_R_ALL_THR": "nan"}, clear=True):
            with self.assertRaises(ValueError):
                OnlineGateController()

    def test_nested_final_answers(self):
        self.assertEqual(
            extract_answer(r"First \boxed{3}, finally \boxed{\frac{1}{2}}"),
            r"\frac{1}{2}",
        )
        self.assertEqual(extract_answer("<answer>A</answer>"), "A")
        self.assertIsNone(extract_answer("Still reasoning."))


if __name__ == "__main__":
    unittest.main()
