"""CPU checks for action/EOS labels, weighted loss, and action-only validation."""
import math
import unittest
from types import SimpleNamespace

import torch

from qwen3_vl_sft import Qwen3VlTrajectoryCollator, WeightedTrainer


class TextProcessor:
    """Small chat fixture with multi-token action 10 and separate role/EOS IDs."""
    def __init__(self):
        self.tokenizer = self
        self.pad_token_id = 0
        self.eos_token_id = 3

    def encode(self, text, add_special_tokens=False):
        special = {"<|im_start|>": 1, "assistant": 2, "<|im_end|>": 3, "\n": 4}
        if text in special:
            return [special[text]]
        return [10 + int(c) for c in text]

    def apply_chat_template(self, messages, **kwargs):
        return messages

    def __call__(self, text, **kwargs):
        ids = []
        for message in text:
            if message["role"] == "assistant":
                action = message["content"][0]["text"]
                ids += [1, 2, 4, *self.encode(action), 3, 4]
            else:
                ids += [1, 5, 4, 8, 3, 4]
        ids = torch.tensor([ids])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def item(actions, mode="last_token"):
    messages = []
    for action in actions:
        messages += [{"role": "user", "content": [{"type": "text", "text": "frame"}]},
                     {"role": "assistant", "content": [{"type": "text", "text": str(action)}]}]
    return {"messages": messages, "loss_mode": mode}


def trainer():
    # Exercise the production loss without initializing distributed training.
    obj = object.__new__(WeightedTrainer)
    obj.loss_type = "weighted"
    obj.im_start_id, obj.assistant_id, obj.im_end_id, obj.newline_id = 1, 2, 3, 4
    obj.supervise_eos = True
    obj._loss_mask_dumped = True
    return obj


class LogitModel(torch.nn.Module):
    def __init__(self, batch, eos_probability=0.2):
        super().__init__()
        labels = batch["labels"]
        logits = torch.zeros((*labels.shape, 32))
        for b in range(labels.shape[0]):
            for pos in (labels[b] != -100).nonzero().flatten().tolist():
                target = int(labels[b, pos])
                probability = eos_probability if target == 3 else 0.8
                logits[b, pos - 1] = math.log((1 - probability) / 31)
                logits[b, pos - 1, target] = math.log(probability)
        self.logits = torch.nn.Parameter(logits)

    def forward(self, **kwargs):
        return SimpleNamespace(logits=self.logits)


class EosSupervisionTest(unittest.TestCase):
    def test_collator_includes_eos_but_excludes_user_role_and_padding(self):
        batch = Qwen3VlTrajectoryCollator(TextProcessor(), 1024)([item([10, 2]), item([10])])
        self.assertEqual(batch["labels"][0][batch["labels"][0] != -100].tolist(), [11, 10, 3, 12, 3])
        self.assertEqual(batch["labels"][1][batch["labels"][1] != -100].tolist(), [11, 10, 3])
        self.assertTrue(torch.all(batch["labels"][batch["attention_mask"] == 0] == -100))
        legacy = Qwen3VlTrajectoryCollator(TextProcessor(), 1024, supervise_eos=False)([item([10])])
        self.assertEqual(legacy["labels"][legacy["labels"] != -100].tolist(), [11, 10])

    def test_final_turn_includes_both_digits_and_eos_without_training_history(self):
        batch = Qwen3VlTrajectoryCollator(TextProcessor(), 1024)([item([2, 10])])
        obj = trainer()
        weights = obj._last_turn_action_weight_mask(batch["input_ids"][0], batch["labels"][0], include_eos=True)
        self.assertEqual(batch["input_ids"][0][weights > 0].tolist(), [11, 10, 3])
        model = LogitModel(batch)
        loss = obj.compute_loss(model, dict(batch))
        self.assertAlmostEqual(float(loss.detach()), (-2 * math.log(0.8) - math.log(0.2)) / 3, places=6)
        loss.backward()
        active_logit_positions = (model.logits.grad.abs().sum(-1)[0] > 0).nonzero().flatten().tolist()
        self.assertEqual(active_logit_positions, [p - 1 for p in (weights > 0).nonzero().flatten().tolist()])

    def test_weighted_general_turns_apply_same_weight_to_action_and_eos(self):
        batch = Qwen3VlTrajectoryCollator(TextProcessor(), 1024)([item([2, 10], "weighted")])
        weights = trainer()._turn_weights_for_labels(batch["input_ids"][0], batch["labels"][0], include_eos=True)
        self.assertEqual(weights[weights > 0].tolist(), [0.5, 0.5, 1., 1., 1.])
        model = LogitModel(batch)
        loss = trainer().compute_loss(model, dict(batch))
        expected = -(2.5 * math.log(0.8) + 1.5 * math.log(0.2)) / 4
        self.assertAlmostEqual(float(loss.detach()), expected, places=6)

    def test_validation_loss_is_independent_of_eos_probability(self):
        collator = Qwen3VlTrajectoryCollator(TextProcessor(), 1024)
        for mode in ("last_token", "weighted"):
            batch = collator([item([2, 10], mode)])
            for eos_probability in (0.01, 0.99):
                with self.subTest(mode=mode, eos_probability=eos_probability):
                    model = LogitModel(batch, eos_probability).eval()
                    loss = trainer().compute_loss(model, dict(batch))
                    self.assertAlmostEqual(float(loss.detach()), -math.log(0.8), places=6)
                    loss.backward()
                    eos_positions = (batch["labels"][0] == 3).nonzero().flatten() - 1
                    self.assertTrue(torch.all(model.logits.grad[0, eos_positions] == 0))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
