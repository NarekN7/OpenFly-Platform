from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cosmos3_edge_vl_sft import Cosmos3EdgeTrajectoryCollator
from qwen3_vl_sft import (
    DEFAULT_VLN_SYSTEM_PROMPT,
    VLN_ALLOWED_ACTION_IDS,
    VlnMixedGeneralSkillDataset,
    VlnTrajectoryCropDataset,
    WeightedTrainer,
)

ALLOWED_ACTIONS = tuple(sorted(VLN_ALLOWED_ACTION_IDS))


class FakeTokenizer:
    TOKENS = {
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "assistant": 3,
        "\n": 4,
        "<think>": 5,
        "</think>": 6,
        "system": 7,
        "user": 8,
    }
    ACTION_IDS = {str(i): 20 + i for i in range(10)}

    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "assistant":
            # Real Cosmos3-Edge tokenization is ["ass", "istant"].
            return [self.TOKENS[text], 9]
        if text in self.TOKENS:
            return [self.TOKENS[text]]
        if text == "10":
            return [self.ACTION_IDS["1"], self.ACTION_IDS["0"]]
        if text in self.ACTION_IDS:
            return [self.ACTION_IDS[text]]
        return [90]

    def decode(self, ids, skip_special_tokens=True):
        reverse_actions = {value: key for key, value in self.ACTION_IDS.items()}
        pieces = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in reverse_actions:
                pieces.append(reverse_actions[token_id])
            elif token_id == 2 and not skip_special_tokens:
                pieces.append("<|im_end|>")
            elif not skip_special_tokens:
                pieces.append(f"<{token_id}>")
        return "".join(pieces)


class FakeImageProcessor:
    size = {"shortest_edge": 784, "longest_edge": 57600}


class FakeCosmosProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_processor = FakeImageProcessor()

    def apply_chat_template(self, messages, **kwargs):
        self.last_kwargs = kwargs
        ids = []
        mm_types = []
        image_count = 0

        def append(token_id, mm_type=0):
            ids.append(token_id)
            mm_types.append(mm_type)

        for message in messages:
            role = message["role"]
            append(1)
            for role_id in self.tokenizer.encode(role):
                append(role_id)
            append(4)
            if role == "assistant":
                append(5)
                append(6)
                text = next(part["text"] for part in message["content"] if part["type"] == "text")
                for action_id in self.tokenizer.encode(text):
                    append(action_id)
            else:
                for part in message["content"]:
                    if part["type"] == "image":
                        append(50, mm_type=1)
                        image_count += 1
                    elif part["type"] == "text":
                        append(90)
            append(2)

        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            "mm_token_type_ids": torch.tensor([mm_types], dtype=torch.long),
            "pixel_values": torch.zeros((image_count, 12), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 1, 1]] * image_count, dtype=torch.long),
        }


def _collator(**kwargs):
    defaults = dict(processor=FakeCosmosProcessor(), max_length=16384, supervise_eos=True)
    defaults.update(kwargs)
    return Cosmos3EdgeTrajectoryCollator(**defaults)


def _trainer_from_collator(collator: Cosmos3EdgeTrajectoryCollator) -> WeightedTrainer:
    trainer = object.__new__(WeightedTrainer)
    trainer.im_start_id = collator._im_start_id
    trainer.assistant_id = collator._assistant_id
    trainer.im_end_id = collator._im_end_id
    trainer.newline_id = collator._newline_id
    return trainer


class CosmosCollatorTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.images = []
        for index in range(2):
            path = self.root / f"{index}.png"
            Image.new("RGB", (64, 32), color=(index, 0, 0)).save(path)
            self.images.append(path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def messages(self, actions=("10", "3")):
        turns = []
        for index, action in enumerate(actions):
            content = [{"type": "image", "image_path": str(self.images[min(index, len(self.images) - 1)])}]
            if index == 0:
                content.append({"type": "text", "text": "Fly forward."})
            turns.append({"role": "user", "content": content})
            turns.append({"role": "assistant", "content": [{"type": "text", "text": str(action)}]})
        return turns

    def test_action_and_eos_only_labels(self):
        processor = FakeCosmosProcessor()
        collator = _collator(
            processor=processor,
            system_prompt=DEFAULT_VLN_SYSTEM_PROMPT,
            supervise_eos=True,
        )
        batch = collator([{"messages": self.messages(), "loss_mode": "weighted"}])
        labels = batch["labels"][0]
        labeled = labels[labels != -100].tolist()
        self.assertEqual(labeled, [21, 20, 2, 23, 2])
        self.assertNotIn(5, labeled)
        self.assertNotIn(6, labeled)
        self.assertNotIn(50, labeled)
        self.assertNotIn(90, labeled)
        self.assertNotIn(1, labeled)
        self.assertNotIn(7, labeled)
        self.assertEqual(batch["loss_mode"], ["weighted"])
        self.assertIn("mm_token_type_ids", batch)
        self.assertTrue(processor.last_kwargs["return_mm_token_type_ids"])
        self.assertEqual(int((batch["mm_token_type_ids"] == 1).sum().item()), 2)

    def test_all_allowed_action_ids_and_multitoken_ten(self):
        collator = _collator(supervise_eos=False)
        for action in ALLOWED_ACTIONS:
            batch = collator(
                [{"messages": self.messages(actions=(str(action),)), "loss_mode": "last_token"}]
            )
            labeled = batch["labels"][0]
            decoded = collator.tokenizer.decode(
                labeled[labeled != -100].tolist(), skip_special_tokens=True
            ).strip()
            self.assertEqual(decoded, str(action), msg=f"action={action}")
            self.assertEqual(int((labeled != -100).sum().item()), 1 + int(action == 10))

    def test_assistant_span_count_matches_messages(self):
        collator = _collator()
        messages = self.messages(actions=("8", "9", "10"))
        batch = collator([{"messages": messages, "loss_mode": "weighted"}])
        n_assistant = sum(1 for message in messages if message["role"] == "assistant")
        spans = __import__("qwen3_vl_sft")._qwen3_vl_assistant_supervision_spans(
            batch["input_ids"],
            collator._im_start_id,
            collator._assistant_id,
            collator._im_end_id,
        )
        self.assertEqual(len(spans), n_assistant)
        self.assertEqual(n_assistant, 3)

    def test_no_eos_and_batch_padding(self):
        collator = _collator(supervise_eos=False)
        short = self.messages()[:2]
        batch = collator(
            [
                {"messages": short, "loss_mode": "last_token"},
                {"messages": self.messages(), "loss_mode": "weighted"},
            ]
        )
        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertEqual(batch["labels"].shape, batch["input_ids"].shape)
        self.assertEqual(batch["attention_mask"].shape, batch["input_ids"].shape)
        self.assertEqual(batch["mm_token_type_ids"].shape, batch["input_ids"].shape)
        self.assertNotIn(2, batch["labels"][batch["labels"] != -100].tolist())
        self.assertEqual(batch["pixel_values"].shape[0], 3)
        pad_len = int(batch["input_ids"].shape[1]) - int((batch["attention_mask"][0] == 1).sum().item())
        self.assertGreater(pad_len, 0)
        self.assertTrue(torch.all(batch["labels"][0, -pad_len:] == -100))

    def test_weighted_and_last_turn_masks_match_openfly_contract(self):
        collator = _collator(supervise_eos=True)
        batch = collator([{"messages": self.messages(), "loss_mode": "weighted"}])
        trainer = _trainer_from_collator(collator)

        turn_weights = trainer._turn_weights_for_labels(
            batch["input_ids"][0], batch["labels"][0], include_eos=True
        )
        positive = turn_weights[turn_weights > 0].tolist()
        self.assertEqual(positive, [0.5, 0.5, 0.5, 1.0, 1.0])

        final_only = trainer._last_turn_action_weight_mask(
            batch["input_ids"][0], batch["labels"][0], include_eos=False
        )
        final_positions = final_only.nonzero(as_tuple=False)[:, 0].tolist()
        self.assertEqual(
            [batch["input_ids"][0, pos].item() for pos in final_positions],
            [23],
        )


class DatasetParityTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.frames = self.root / "frames"
        self.frames.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_pool(self, name, count, final_action, history=1):
        trajectories = []
        for index in range(count):
            traj_name = f"{name}-{index}"
            traj_dir = self.frames / traj_name
            traj_dir.mkdir()
            actions = []
            keys = []
            for step in range(history):
                Image.new("RGB", (32, 32)).save(traj_dir / f"{step}.png")
                actions.append(int(final_action) if step == history - 1 else 10)
                keys.append(str(step))
            trajectories.append(
                {
                    "image_path": traj_name,
                    "gpt_instruction": "Navigate.",
                    "action": actions,
                    "index_list": keys,
                }
            )
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(trajectories), encoding="utf-8")
        return str(path)

    def test_message_semantics_are_shared_with_qwen_dataset(self):
        general = self.write_pool("general", 1, 10, history=3)
        dataset = VlnTrajectoryCropDataset(
            json_path=general,
            frames_root=str(self.frames),
            temporal_history_past=16,
            deterministic=True,
        )
        sample = dataset[0]
        roles = [m["role"] for m in sample["messages"]]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant", "user", "assistant"])
        self.assertEqual(sample["messages"][0]["content"][1]["text"], "Navigate.")
        self.assertEqual(sample["messages"][1]["content"][0]["text"], "10")
        self.assertEqual(sample["messages"][3]["content"][0]["text"], "10")
        self.assertEqual(sample["messages"][-1]["content"][0]["text"], "10")
        image_names = [
            Path(part["image_path"]).name
            for message in sample["messages"]
            if message["role"] == "user"
            for part in message["content"]
            if part.get("type") == "image"
        ]
        self.assertEqual(image_names, ["0.png", "1.png", "2.png"])
        self.assertEqual(sample["traj_meta"]["t"], 2)

        collator = _collator(supervise_eos=True)
        batch = collator([sample])
        decoded = collator.tokenizer.decode(
            batch["labels"][0][batch["labels"][0] != -100].tolist(),
            skip_special_tokens=True,
        ).strip()
        self.assertEqual(decoded, "101010")
        trainer = _trainer_from_collator(collator)
        weights = trainer._turn_weights_for_labels(
            batch["input_ids"][0], batch["labels"][0], include_eos=True
        )
        # Three assistant turns: action 10 is two tokens + EOS.
        self.assertEqual(
            [round(w, 4) for w in weights[weights > 0].tolist()],
            [round(1 / 3, 4), round(1 / 3, 4), round(1 / 3, 4),
             round(2 / 3, 4), round(2 / 3, 4), round(2 / 3, 4),
             1.0, 1.0, 1.0],
        )

    def test_lrb60_stop20_cardinality_and_loss_modes(self):
        mixed = VlnMixedGeneralSkillDataset(
            general_json=self.write_pool("general", 10, 1),
            skill_json_left=self.write_pool("left", 4, 2),
            skill_json_right=self.write_pool("right", 4, 3),
            skill_mix_rate=0.60,
            skill_json_stop=self.write_pool("stop", 4, 0),
            skill_mix_rate_stop=0.20,
            frames_root=str(self.frames),
            chat_window_turns=1,
            temporal_history_past=16,
            verify_images_exist=True,
            max_window_sample_attempts=2,
            max_trajectories=None,
            debug_samples=None,
            deterministic=False,
        )
        self.assertEqual(mixed.R_each, 3)
        self.assertEqual(mixed.R_stop, 2)
        self.assertEqual(len(mixed), 18)
        self.assertEqual(mixed[0]["loss_mode"], "weighted")
        self.assertEqual(mixed[10]["loss_mode"], "last_token")
        self.assertEqual(mixed[13]["loss_mode"], "last_token")
        self.assertEqual(mixed[16]["loss_mode"], "last_token")
        self.assertEqual(mixed[10]["traj_meta"]["skill"], "left")
        self.assertEqual(mixed[13]["traj_meta"]["skill"], "right")
        self.assertEqual(mixed[16]["traj_meta"]["skill"], "stop")


class RealProcessorOptionalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from cosmos3_edge_vl_sft import (
                COSMOS3_EDGE_REPO,
                Cosmos3EdgeProcessorFactory,
            )

            cls.processor = Cosmos3EdgeProcessorFactory.from_pretrained(COSMOS3_EDGE_REPO)
            cls.processor.image_processor.size = {
                **dict(cls.processor.image_processor.size),
                "shortest_edge": 784,
                "longest_edge": 57600,
            }
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"Pinned Cosmos processor unavailable: {exc}") from exc

    def test_real_tokenizer_action_ids_and_processor_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "frame.png"
            Image.new("RGB", (64, 32), color=(7, 8, 9)).save(image)
            collator = Cosmos3EdgeTrajectoryCollator(
                self.processor,
                max_length=16384,
                system_prompt=DEFAULT_VLN_SYSTEM_PROMPT,
                supervise_eos=True,
            )
            self.assertGreater(len(collator._assistant_ids), 1)
            for action in ALLOWED_ACTIONS:
                probe = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image_path": str(image)},
                                {"type": "text", "text": "Navigate."},
                            ],
                        },
                        {"role": "assistant", "content": [{"type": "text", "text": str(action)}]},
                    ],
                    "loss_mode": "last_token",
                }
                batch = collator([probe])
                self.assertIn("mm_token_type_ids", batch)
                labeled = batch["labels"][0]
                decoded = self.processor.tokenizer.decode(
                    labeled[labeled != -100].tolist(),
                    skip_special_tokens=True,
                ).strip()
                self.assertEqual(decoded, str(action))
                self.assertTrue(torch.all(labeled[batch["mm_token_type_ids"][0] == 1] == -100))
            saved = Path(tmp) / "processor"
            self.processor.save_pretrained(saved)
            self.assertTrue((saved / "tokenizer.json").is_file() or (saved / "tokenizer_config.json").is_file())


if __name__ == "__main__":
    unittest.main()
