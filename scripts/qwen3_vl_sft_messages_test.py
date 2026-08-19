import json
import tempfile
import unittest
from pathlib import Path

from qwen3_vl_sft import VlnTrajectoryCropDataset


class VlnTrajectoryMessagesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.frames_root = self.root / "frames"
        self.trajectory_dir = self.frames_root / "trajectory"
        self.trajectory_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def build_dataset(self, actions: list[int], frame_keys: list[str]) -> VlnTrajectoryCropDataset:
        for key in frame_keys:
            (self.trajectory_dir / f"{key}.png").touch()
        json_path = self.root / "trajectories.json"
        json_path.write_text(
            json.dumps(
                [
                    {
                        "image_path": "trajectory",
                        "gpt_instruction": "Fly to the doorway.",
                        "action": actions,
                        "index_list": frame_keys,
                    }
                ]
            ),
            encoding="utf-8",
        )
        return VlnTrajectoryCropDataset(
            json_path=str(json_path),
            frames_root=str(self.frames_root),
            temporal_history_past=16,
            deterministic=True,
        )

    def test_final_history_turn_contains_only_its_frame(self) -> None:
        messages = self.build_dataset([10, 3], ["frame_6", "frame_7"])[0]["messages"]
        users = [message for message in messages if message["role"] == "user"]

        self.assertEqual(
            users[-1],
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image_path": str(self.trajectory_dir / "frame_7.png"),
                    }
                ],
            },
        )

    def test_single_turn_keeps_instruction_without_action_suffix(self) -> None:
        messages = self.build_dataset([3], ["frame_7"])[0]["messages"]

        self.assertEqual(
            messages[0],
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image_path": str(self.trajectory_dir / "frame_7.png"),
                    },
                    {"type": "text", "text": "Fly to the doorway."},
                ],
            },
        )


if __name__ == "__main__":
    unittest.main()
