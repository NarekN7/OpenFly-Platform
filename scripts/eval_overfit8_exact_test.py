"""CPU checks for the overfit probe's shared parser, stopping, and token metrics."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

import eval_overfit8_exact as probe
from train.qwen3_vl_interleaved_common import parse_vln_action_id


class Processor:
    def __init__(self):
        self.tokenizer = self

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(int(t)) for t in tokens if int(t) != 0)

    def apply_chat_template(self, messages, **kwargs):
        return "prompt"

    def __call__(self, **kwargs):
        return {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}


class Model:
    def __init__(self, text):
        self.tokens = [ord(c) for c in text] + [0]

    def __call__(self, **kwargs):
        logits = torch.full((1, 1, 128), -100.0)
        logits[0, 0, self.tokens[0]] = 100
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, max_new_tokens, stopping_criteria, **kwargs):
        for token in self.tokens[:max_new_tokens]:
            input_ids = torch.cat((input_ids, torch.tensor([[token]])), dim=1)
            if token == 0 or stopping_criteria(input_ids, None).all():
                break
        return input_ids


class OverfitProbeTest(unittest.TestCase):
    def predict(self, text, target, *, full=False):
        with patch.dict(os.environ, {"OPENFLY_QWEN_STOP_AFTER_ACTION": "0" if full else "1"}):
            return probe.predict_one(Model(text), Processor(), [], gt_action=target,
                                     max_new_tokens=16, device=torch.device("cpu"))

    def test_shared_parser_and_leading_continuations(self):
        self.assertIs(probe.parse_vln_action_id, parse_vln_action_id)
        for raw, gt, expected in (("2 10", 2, 2), ("3.5", 3, 3), ("201010", 2, 2),
                                  ("6 2", 2, None), ("7 10", 10, None), ("answer 2", 2, None)):
            with self.subTest(raw=raw):
                out = self.predict(raw, gt, full=True)
                self.assertEqual(out["pred"], expected)
                self.assertEqual(out["gen_text"], raw)
                self.assertFalse(out["output_format_valid"])

    def test_action_10_and_first_token_scoring_are_distinct(self):
        correct = self.predict("10", 10)
        self.assertEqual(correct["generated_token_ids"], [ord("1"), ord("0")])
        self.assertEqual(correct["pred"], 10)
        self.assertTrue(correct["ok_first_token"])
        self.assertTrue(correct["ok"])
        ambiguous = self.predict("1", 10)
        self.assertEqual(ambiguous["pred_from_first_token"], 1)
        self.assertTrue(ambiguous["ok_first_token"])
        self.assertFalse(ambiguous["ok"])
        self.assertFalse(self.predict("10", 1)["ok"])
        spaced = self.predict(" 2", 2)
        self.assertTrue(spaced["ok"])
        self.assertFalse(spaced["ok_first_token"])

    def test_stopping_preserves_action_and_format_is_unmeasured(self):
        with patch.dict(os.environ, {"OPENFLY_QWEN_STOP_AFTER_ACTION": "1"}):
            rows = [self.predict("2 10", 2), self.predict("10 3", 10), self.predict("6 2", 0)]
            metrics = probe.summarize(rows)
        self.assertEqual([r["gen_text"] for r in rows], ["2", "10", "6"])
        self.assertEqual(metrics["n_correct"], 2)
        self.assertEqual(metrics["n_invalid_actions"], 1)
        self.assertEqual(metrics["n_scored"], 3)
        self.assertEqual(metrics["n_format_scored"], 0)
        self.assertIsNone(metrics["n_invalid_format"])
        self.assertIsNone(metrics["strict_accuracy"])
        self.assertTrue(all(r["output_format_valid"] is None for r in rows))

    def test_full_response_keeps_action_and_strict_accuracy_separate(self):
        with patch.dict(os.environ, {"OPENFLY_QWEN_STOP_AFTER_ACTION": "0"}):
            rows = [self.predict("3.5", 3, full=True), self.predict("10", 10, full=True),
                    self.predict("6 2", 2, full=True)]
            metrics = probe.summarize(rows)
        self.assertEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["strict_accuracy"], 1 / 3)
        self.assertEqual(metrics["n_invalid_format"], 2)
        self.assertEqual(metrics["generation_policy"], "full_response")

    def test_historical_single_turn_padding_and_index_order_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traj = root / "trajectory"
            traj.mkdir()
            for filename, color in (("9.png", "red"), ("2.png", "blue")):
                Image.new("RGB", (2, 2), color).save(traj / filename)
            item = {"image_path": "trajectory", "gpt_instruction": "Turn left",
                    "action": [1, 2], "index_list": [9, 2]}
            messages = probe.build_messages(item, root)
            self.assertEqual([m["role"] for m in messages], ["system", "user"])
            images = [p["image"] for p in messages[1]["content"] if p["type"] == "image"]
            self.assertEqual(len(images), 17)
            self.assertEqual([im.getpixel((0, 0)) for im in images], [(255, 0, 0)] * 16 + [(0, 0, 255)])

    def test_cli_writes_rows_and_companion_metrics(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            data_path = root / "eval_8.json"
            rows_path = root / "results.json"
            data_path.write_text(json.dumps([{"image_path": "traj", "action": [10], "index_list": [0]}] * 8))
            processor, model = Processor(), Model("10")
            model.to = lambda *_: model
            model.eval = lambda: model
            stack.enter_context(patch.dict(os.environ, {"OPENFLY_QWEN_STOP_AFTER_ACTION": "1"}))
            stack.enter_context(patch.object(probe.AutoProcessor, "from_pretrained", return_value=processor))
            stack.enter_context(patch.object(probe.Qwen3VLForConditionalGeneration, "from_pretrained", return_value=model))
            stack.enter_context(patch.object(probe, "build_messages", return_value=[]))
            stack.enter_context(patch("sys.argv", ["probe", "--model_dir", str(root), "--eval_json", str(data_path),
                                                   "--frames_root", str(root), "--out_json", str(rows_path)]))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            probe.main()
            rows = json.loads(rows_path.read_text())
            metrics = json.loads((root / "results.summary.json").read_text())
            self.assertEqual(len(rows), 8)
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertEqual(metrics["first_token_accuracy"], 1.0)
            self.assertEqual(metrics["max_new_tokens"], 16)
            self.assertEqual(metrics["model_dir"], str(root))

    def test_too_short_generation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be >= 2"):
            probe.predict_one(Model("10"), Processor(), [], gt_action=10,
                              max_new_tokens=1, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
