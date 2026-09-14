"""CPU regression checks for the shared navigation/skill action protocol."""

import contextlib
import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from qwen3_vl_interleaved_common import (
    VLN_ACTION_PARSER,
    VLN_ALLOWED_ACTION_IDS,
    action_stopping_criteria,
    parse_vln_action_id,
    predict_action,
)
import skill_eval_intrain_crops
import skill_eval_left_right
import skill_eval_stop
import skill_eval_lr_diagnostics


class ActionPrefixTest(unittest.TestCase):
    def test_all_actions_with_surrounding_whitespace(self):
        for action in VLN_ALLOWED_ACTION_IDS:
            with self.subTest(action=action):
                self.assertEqual(parse_vln_action_id(f" \n{action}\t"), action)

    def test_continuations_do_not_override_leading_action(self):
        cases = {
            "2\n10\n10\n9\n0  # Stop": 2,
            "3 3 9 9 10 10 10": 3,
            "3.5": 3,
            "2.0\n9.0\n2.0\n10.0": 2,
            "2,2,10": 2,
            "2010101010101010": 2,
            "3*/30°, 30°": 3,
            "0 10": 0,
            "1": 1,
            "1 0": 1,
            "1\n0": 1,
            "1.0": 1,
            "10": 10,
            "10 2": 10,
            "10\n2": 10,
            "10.5": 10,
            "100": 10,
            "101010": 10,
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(parse_vln_action_id(raw), expected)

    def test_invalid_prefix_never_becomes_stop_or_a_later_action(self):
        for raw in ("", " \n", "6 2", "7 10", "answer: 2", "-3", "+3", "[2]", "３"):
            with self.subTest(raw=raw):
                self.assertIsNone(parse_vln_action_id(raw))

    def test_predict_decodes_both_tokens_of_ten_and_preserves_raw_output(self):
        # Qwen encodes 10 as [16, 15]. The first two IDs belong to the prompt.
        prompt_ids = torch.tensor([[17, 18]])
        generated_ids = [16, 15, 198, 17]
        processor = Mock()
        processor.return_value = {"input_ids": prompt_ids}
        processor.tokenizer = SimpleNamespace(
            pad_token_id=0, eos_token_id=151645, decode=Mock(return_value="10\n2")
        )
        model = Mock()
        model.generate.return_value = torch.tensor([[17, 18, *generated_ids]])
        self.assertEqual(predict_action(model, processor, [], "cpu", 16), (10, "10\n2"))
        decode_args, decode_kwargs = processor.tokenizer.decode.call_args
        self.assertEqual(decode_args[0].tolist(), generated_ids)
        self.assertTrue(decode_kwargs["skip_special_tokens"])
        self.assertFalse(model.generate.call_args.kwargs["do_sample"])
        processor.tokenizer.decode.return_value = "6 2"
        self.assertEqual(predict_action(model, processor, [], "cpu", 16), (None, "6 2"))
        processor.tokenizer.decode.return_value = ""
        self.assertEqual(predict_action(model, processor, [], "cpu", 16), (None, ""))

    def test_offline_metrics_keep_invalids_in_denominator_and_separate_format(self):
        # Invalid responses on stop targets must not count as correct or errors.
        outputs = ["", "6 2", "0", "3.5", "10\n2", "7 10", "2 10"]
        targets = [0, 0, 0, 3, 10, 0, 2]
        data = [
            {"image_path": f"trajectory_{i}", "action": [gt], "index_list": ["0"],
             "gpt_instruction": "Fly to the doorway."}
            for i, gt in enumerate(targets)
        ]
        for evaluator in (skill_eval_left_right, skill_eval_intrain_crops):
            with self.subTest(evaluator=evaluator.__name__), contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {"OPENFLY_QWEN_STOP_AFTER_ACTION": "0"}))
                stack.enter_context(patch.object(evaluator, "load_trajectory_pil_frames", return_value=[object()]))
                stack.enter_context(patch.object(
                    evaluator, "predict_action",
                    side_effect=[(parse_vln_action_id(raw), raw) for raw in outputs],
                ))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                kwargs = {"limit": 0} if evaluator is skill_eval_left_right else {}
                predictions, metrics = evaluator._evaluate_json(
                    data=data, model=None, processor=SimpleNamespace(), device="cpu",
                    image_roots=[], temporal_past=16, system_prompt="", max_new_tokens=16,
                    **kwargs,
                )
                self.assertEqual(metrics["n_scored"], 7)
                self.assertEqual(metrics["n_errors"], 0)
                self.assertEqual(metrics["n_invalid_actions"], 3)
                self.assertEqual(metrics["n_invalid_format"], 6)
                self.assertEqual(metrics["n_correct"], 4)
                self.assertEqual(metrics["accuracy"], 4 / 7)
                self.assertEqual(metrics["pred_action_counts"]["invalid"], 3)
                self.assertEqual(metrics["confusion_gt_pred"]["0->invalid"], 3)
                self.assertEqual(metrics["action_parser"], VLN_ACTION_PARSER)
                self.assertEqual([r["raw_decode"] for r in predictions], outputs)
                self.assertFalse(predictions[0]["action_valid"])
                self.assertTrue(predictions[3]["action_valid"])
                self.assertFalse(predictions[3]["output_format_valid"])
                json.dumps({"predictions": predictions, "metrics": metrics}, sort_keys=True)
        self.assertIs(skill_eval_stop._evaluate_json, skill_eval_left_right._evaluate_json)

    def test_real_generation_stops_without_changing_the_action(self):
        from transformers import LlamaConfig, LlamaForCausalLM, LogitsProcessorList

        class Tokenizer:
            def decode(self, ids, skip_special_tokens=True):
                chars = {15: "0", 16: "1", 17: "2", 18: "3", 21: "6", 13: ".", 198: "\n", 220: " ", 254: ""}
                return "".join(chars[int(i)] for i in ids)

        class ForceTokens:
            def __init__(self, tokens):
                self.tokens = tokens

            def __call__(self, ids, scores):
                scores[:] = -float("inf")
                scores[:, self.tokens[min(ids.shape[1] - 1, len(self.tokens) - 1)]] = 0
                return scores

        torch.set_num_threads(1)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=256, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, eos_token_id=254, pad_token_id=255,
        )).eval()
        tok = Tokenizer()
        cases = [([17, 16, 15], [17], 2), ([18, 13, 16], [18], 3),
                 ([16, 15, 17], [16, 15], 10), ([16, 220, 15], [16, 220], 1),
                 ([16, 254], [16, 254], 1), ([220, 16, 15], [220, 16, 15], 10),
                 ([21, 17], [21], None)]
        for tokens, expected_ids, expected_action in cases:
            with self.subTest(tokens=tokens), patch.dict(os.environ, {"OPENFLY_QWEN_STOP_AFTER_ACTION": "1"}):
                out = model.generate(
                    input_ids=torch.tensor([[42]]), max_new_tokens=8, do_sample=False,
                    logits_processor=LogitsProcessorList([ForceTokens(tokens)]),
                    stopping_criteria=action_stopping_criteria(tok, 1),
                )[0, 1:].tolist()
                self.assertEqual(out, expected_ids)
                self.assertEqual(parse_vln_action_id(tok.decode(out)), expected_action)
        with patch.dict(os.environ, {"OPENFLY_QWEN_STOP_AFTER_ACTION": "0"}):
            self.assertEqual(len(action_stopping_criteria(tok, 1)), 0)

    def test_early_stopping_does_not_claim_natural_format_success(self):
        with patch.dict(os.environ, {"OPENFLY_QWEN_STOP_AFTER_ACTION": "1"}), \
             patch.object(skill_eval_left_right, "load_trajectory_pil_frames", return_value=[object()]), \
             patch.object(skill_eval_left_right, "predict_action", return_value=(2, "2")), \
             contextlib.redirect_stdout(io.StringIO()):
            predictions, metrics = skill_eval_left_right._evaluate_json(
                data=[{"image_path": "fixture", "action": [2], "index_list": ["0"]}],
                model=None, processor=SimpleNamespace(), device="cpu", image_roots=[],
                temporal_past=16, system_prompt="", max_new_tokens=16, limit=0,
            )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["n_format_scored"], 0)
        self.assertIsNone(metrics["n_invalid_format"])
        self.assertIsNone(predictions[0]["output_format_valid"])
        diagnostic = skill_eval_lr_diagnostics.summarize([{
            "final_action_nll": 0.1, "gt_action": 2, "parsed_action": 2,
            "raw_first_token_correct": True, "generation_first_token_correct": True,
            "parsed_correct": True, "strict_correct": None, "strict_action": None,
        }])
        self.assertEqual(diagnostic["parsed_correct_accuracy"], 1.0)
        self.assertIsNone(diagnostic["strict_correct_accuracy"])
        self.assertIsNone(diagnostic["n_invalid_strict_outputs"])


if __name__ == "__main__":
    unittest.main()
