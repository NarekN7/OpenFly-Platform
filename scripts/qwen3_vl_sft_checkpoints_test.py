"""CPU regression checks for best/last persistence and reloadable HF saves."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

from qwen3_vl_sft import BestAndLastCheckpointCallback


def tiny_model():
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    ))


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = SimpleNamespace(output_dir=self.tmp.name)
        self.state = SimpleNamespace(is_world_process_zero=True, global_step=10, epoch=1.0)
        self.model = tiny_model()
        self.trainer = SimpleNamespace(save_model=Mock(side_effect=self.model.save_pretrained))
        self.processor = SimpleNamespace(save_pretrained=Mock(side_effect=lambda path: (
            Path(path) / "processor_config.json"
        ).write_text('{"fixture": true}', encoding="utf-8")))
        self.callback = self.new_callback()

    def new_callback(self, **kwargs):
        return BestAndLastCheckpointCallback(self.processor, [self.trainer], **kwargs)

    def save(self, loss, callback=None):
        (callback or self.callback).on_evaluate(self.args, self.state, None, metrics={"eval_loss": loss})

    def metadata(self):
        return json.loads((self.root / "eval-best" / "best_eval_metric.json").read_text())

    def assert_model_equal(self, path, model=None):
        reloaded = LlamaForCausalLM.from_pretrained(path, local_files_only=True)
        for key, tensor in (model or self.model).state_dict().items():
            torch.testing.assert_close(reloaded.state_dict()[key], tensor, rtol=0, atol=0)

    def test_resume_retains_best_until_a_real_improvement(self):
        self.save(0.14)
        resumed = self.new_callback()
        resumed.on_train_begin(self.args, self.state, None)
        self.assertEqual((resumed.best_loss, resumed.best_step), (0.14, 10))
        self.state.global_step = 20
        self.save(0.30, resumed)
        self.save(0.14, resumed)
        self.trainer.save_model.assert_called_once()
        self.assertEqual(self.metadata()["global_step"], 10)
        with torch.no_grad():
            self.model.model.embed_tokens.weight.add_(1)
        self.save(0.13, resumed)
        self.assertEqual(self.metadata()["global_step"], 20)
        self.assertEqual(resumed.best_loss, 0.13)
        self.assertTrue((self.root / "eval-best" / "processor_config.json").is_file())
        self.assert_model_equal(self.root / "eval-best")

    def test_write_and_install_failures_preserve_the_previous_best(self):
        self.save(0.14)
        old_weights = (self.root / "eval-best" / "model.safetensors").read_bytes()
        original_replace = os.replace
        original_write_text = Path.write_text

        def fail_install(src, dst):
            if Path(src).name.startswith(".eval-best.tmp-"):
                raise OSError("simulated rename failure")
            return original_replace(src, dst)

        def fail_metadata(path, *args, **kwargs):
            if path.name == "best_eval_metric.json":
                raise OSError("simulated metadata write failure")
            return original_write_text(path, *args, **kwargs)

        failures = [
            patch.object(self.trainer, "save_model", side_effect=OSError("simulated disk full")),
            patch.object(self.trainer, "save_model", return_value=None),
            patch.object(self.processor, "save_pretrained", side_effect=OSError("simulated processor failure")),
            patch.object(Path, "write_text", fail_metadata),
            patch("qwen3_vl_sft.os.replace", side_effect=fail_install),
        ]
        for failure in failures:
            with self.subTest(failure=failure), failure:
                self.state.global_step = 20
                self.save(0.13)
            self.assertEqual((self.callback.best_loss, self.callback.best_step), (0.14, 10))
            self.assertEqual(self.metadata()["loss"], 0.14)
            self.assertEqual((self.root / "eval-best" / "model.safetensors").read_bytes(), old_weights)
            self.assertFalse(list(self.root.glob(".eval-best.tmp-*")))
        self.save(0.13)
        self.assertEqual(self.callback.best_step, 20)
        self.assert_model_equal(self.root / "eval-best")

    def test_non_finite_metrics_and_nonzero_ranks_do_not_save(self):
        for loss in (float("nan"), float("inf"), -float("inf")):
            self.save(loss)
        self.state.is_world_process_zero = False
        self.save(0.1)
        self.trainer.save_model.assert_not_called()

    def test_corrupt_or_incompatible_best_metadata_is_not_overwritten(self):
        self.save(0.14)
        path = self.root / "eval-best" / "best_eval_metric.json"
        original = self.metadata()
        for text in ("{broken", json.dumps({**original, "loss": float("nan")}),
                     json.dumps({**original, "metric": "train_loss"})):
            with self.subTest(metadata=text):
                path.write_text(text)
                with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                    self.new_callback().on_train_begin(self.args, self.state, None)
                self.assertEqual(path.read_text(), text)
        path.unlink()
        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
            self.save(0.1, self.new_callback())
        self.trainer.save_model.assert_called_once()

    def test_interrupted_replacement_is_recovered_on_restart(self):
        self.save(0.14)
        os.replace(self.root / "eval-best", self.root / ".eval-best.previous")
        resumed = self.new_callback()
        resumed.on_train_begin(self.args, self.state, None)
        self.assertEqual(resumed.best_loss, 0.14)
        self.assertEqual(self.metadata()["global_step"], 10)
        self.assert_model_equal(self.root / "eval-best")

    def test_sharded_safetensors_save_reloads_and_rejects_broken_indices(self):
        self.trainer.save_model.side_effect = lambda path: self.model.save_pretrained(path, max_shard_size="1KB")
        self.save(0.14)
        root = self.root / "eval-best"
        self.assertTrue(self.callback._verify_best_saved(str(root)))
        self.assert_model_equal(root)
        index_path = root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        self.assertGreater(len(set(index["weight_map"].values())), 1)
        for index_data in ({"weight_map": {}}, {"weight_map": {"bad": "missing.safetensors"}},
                           {"weight_map": {"bad": "../outside.safetensors"}},
                           {"weight_map": {"bad": "config.json"}},
                           {"weight_map": {"missing_tensor": next(iter(index["weight_map"].values()))}}):
            with self.subTest(index=index_data):
                index_path.write_text(json.dumps({"metadata": {}, **index_data}))
                self.assertFalse(self.callback._verify_best_saved(str(root)))
        index_path.write_text(json.dumps({"weight_map": index["weight_map"]}))
        self.assertFalse(self.callback._verify_best_saved(str(root)))
        index_path.write_text(json.dumps(index))
        shard = root / next(iter(index["weight_map"].values()))
        shard.write_bytes(b"truncated")
        self.assertFalse(self.callback._verify_best_saved(str(root)))

    def test_legacy_sharded_bin_manifest_checks_every_file(self):
        root = self.root / "legacy"
        root.mkdir()
        self.model.config.save_pretrained(root)
        state = self.model.state_dict()
        names = list(state)
        weight_map = {}
        for i, keys in enumerate((names[::2], names[1::2]), 1):
            filename = f"pytorch_model-{i:05d}-of-00002.bin"
            torch.save({k: state[k] for k in keys}, root / filename)
            weight_map.update({k: filename for k in keys})
        (root / "pytorch_model.bin.index.json").write_text(json.dumps({
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state.values())},
            "weight_map": weight_map,
        }))
        self.assertTrue(self.callback._verify_best_saved(str(root)))
        self.assert_model_equal(root)
        (root / next(iter(weight_map.values()))).unlink()
        self.assertFalse(self.callback._verify_best_saved(str(root)))

    def test_default_lora_adapter_save_and_reload(self):
        base = self.root / "base-model"
        self.model.save_pretrained(base)
        adapter = get_peft_model(LlamaForCausalLM.from_pretrained(base), LoraConfig(
            r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM",
        ))
        with torch.no_grad():
            for name, param in adapter.named_parameters():
                if "lora_B" in name:
                    param.fill_(0.1)
        self.trainer.save_model.side_effect = adapter.save_pretrained
        self.save(0.14)
        root = self.root / "eval-best"
        self.assertTrue(self.callback._verify_best_saved(str(root)))
        reloaded = PeftModel.from_pretrained(LlamaForCausalLM.from_pretrained(base), root)
        for key, tensor in adapter.state_dict().items():
            torch.testing.assert_close(reloaded.state_dict()[key], tensor, rtol=0, atol=0)
        (root / "adapter_config.json").unlink()
        self.assertFalse(self.callback._verify_best_saved(str(root)))

    def test_last_copy_failure_retains_full_previous_state(self):
        src = self.root / "checkpoint-10"
        self.model.save_pretrained(src)
        (src / "trainer_state.json").write_text('{"global_step": 10}')
        torch.save({"step": 10}, src / "optimizer.pt")
        self.callback.on_save(self.args, self.state, None)
        dst = self.root / "checkpoint-last"
        old = {p.name: p.read_bytes() for p in dst.iterdir()}
        with patch("qwen3_vl_sft.shutil.copytree", side_effect=OSError("simulated copy failure")):
            with self.assertRaises(OSError):
                self.callback.on_save(self.args, self.state, None)
        self.assertEqual({p.name: p.read_bytes() for p in dst.iterdir()}, old)
        (src / "trainer_state.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "Incomplete Trainer checkpoint"):
            self.callback.on_save(self.args, self.state, None)
        self.assertEqual({p.name: p.read_bytes() for p in dst.iterdir()}, old)

    def test_real_trainer_full_state_resume(self):
        data = [{"input_ids": [1, 3, 4, 2], "labels": [1, 3, 4, 2]}] * 2

        def make_trainer(max_steps):
            holder = [None]
            callback = BestAndLastCheckpointCallback(None, holder)
            args = TrainingArguments(
                output_dir=self.tmp.name, use_cpu=True, max_steps=max_steps,
                per_device_train_batch_size=1, per_device_eval_batch_size=1,
                eval_strategy="steps", eval_steps=1, save_strategy="epoch", save_total_limit=1,
                report_to=[], disable_tqdm=True, logging_strategy="no",
            )
            trainer = Trainer(model=tiny_model(), args=args, train_dataset=data,
                              eval_dataset=data, callbacks=[callback])
            holder[0] = trainer
            return trainer, callback

        trainer, _ = make_trainer(2)
        trainer.train()
        last = self.root / "checkpoint-last"
        for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
            self.assertTrue((last / filename).is_file(), filename)
        previous_best = self.metadata()
        os.replace(last, self.root / ".checkpoint-last.previous")
        BestAndLastCheckpointCallback.recover_last_for_resume(str(last))
        resumed, callback = make_trainer(4)
        with patch.object(callback, "_maybe_save_best"):
            resumed.train(resume_from_checkpoint=str(last))
        self.assertEqual(resumed.state.global_step, 4)
        self.assertEqual(callback.best_loss, previous_best["loss"])
        self.assertEqual(self.metadata(), previous_best)
        self.assertEqual(json.loads((last / "trainer_state.json").read_text())["global_step"], 4)
        self.assert_model_equal(last, resumed.model)


if __name__ == "__main__":
    unittest.main()
