"""Lightweight AirSim session for live frame capture (predicted-path replay)."""

from __future__ import annotations

import math
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import msgpack
import numpy as np

# msgpack-rpc default max_bin_len=1MB; AirSim image payloads are larger.
_orig_msgpack_unpacker = msgpack.Unpacker


def _Unpacker_large_bin(*args, **kwargs):
    kwargs.setdefault("max_bin_len", 32 * 1024 * 1024)
    return _orig_msgpack_unpacker(*args, **kwargs)


msgpack.Unpacker = _Unpacker_large_bin

import airsim  # noqa: E402

from physics import replay_trajectory

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WAIT_SEC = int(os.environ.get("OPENFLY_AIRSIM_WAIT_SEC", "40"))
DEFAULT_SETTLE_SEC = float(os.environ.get("OPENFLY_VIZ_POSE_SETTLE_SEC", "0.1"))
DISPLAY_MAX_WIDTH = int(os.environ.get("OPENFLY_VIZ_FRAME_MAX_WIDTH", "960"))

_SHARED_SESSION: Optional["AirSimSession"] = None


def get_shared_session() -> "AirSimSession":
    """Process-wide session so a new Frame Viewer tab can reuse the running sim."""
    global _SHARED_SESSION
    if _SHARED_SESSION is None:
        _SHARED_SESSION = AirSimSession()
    return _SHARED_SESSION


def kill_airsim_processes() -> None:
    for keyword in ("AirVLN", "guangzhou", "shanghai", "CitySample", "CrashReport"):
        result = subprocess.run(["pgrep", "-n", keyword], stdout=subprocess.PIPE, check=False)
        cr_pid = result.stdout.decode().strip()
        if cr_pid:
            subprocess.run(["kill", "-9", cr_pid], check=False)


class AirSimSession:
    """Manage one AirSim env process + client for viz frame capture."""

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self.repo_root = Path(repo_root) if repo_root else _REPO_ROOT
        self.env_name: Optional[str] = None
        self.status: str = "Stopped"  # Stopped | Starting | Connected | Error
        self.error: Optional[str] = None
        self._client = None
        self._process: Optional[subprocess.Popen] = None
        self._launch_thread: Optional[threading.Thread] = None

    def stop(self) -> None:
        kill_airsim_processes()
        self._client = None
        self._process = None
        self.env_name = None
        self.status = "Stopped"
        self.error = None

    def start(self, env_name: str, wait_sec: int = DEFAULT_WAIT_SEC) -> None:
        """Stop any prior sim, launch env, connect. Blocking until connected or error."""
        self.stop()
        self.env_name = env_name
        self.status = "Starting"
        self.error = None

        env_dir = self.repo_root / "envs" / "airsim" / env_name
        start_sh = env_dir / "LinuxNoEditor" / "start.sh"
        if not start_sh.is_file():
            self.status = "Error"
            self.error = f"Missing start script: {start_sh}"
            self.env_name = None
            raise FileNotFoundError(self.error)

        def _launch() -> None:
            self._process = subprocess.Popen(
                ["bash", str(start_sh)],
                cwd=str(env_dir / "LinuxNoEditor"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        self._launch_thread = threading.Thread(target=_launch, daemon=True)
        self._launch_thread.start()
        time.sleep(max(1, int(wait_sec)))

        try:
            self._connect()
            self.status = "Connected"
        except Exception as e:
            self.status = "Error"
            self.error = str(e)
            kill_airsim_processes()
            self._client = None
            self.env_name = None
            raise

    def _connect(self) -> None:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        self._client = client

    def ensure_connected(self) -> None:
        if self._client is None:
            raise RuntimeError("AirSim client not connected")

    def set_camera_pose(self, x: float, y: float, z: float, pitch: float, yaw: float, roll: float = 0.0) -> None:
        self.ensure_connected()
        target_pose = airsim.Pose(
            airsim.Vector3r(x, -y, -z),
            airsim.to_quaternion(math.radians(pitch), 0, math.radians(-yaw)),
        )
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        self._client.simSetVehiclePose(target_pose, True)

    def get_camera_rgb(self) -> np.ndarray:
        """Return RGB HxWx3 uint8 (converted from AirSim BGR-ish Scene buffer)."""
        self.ensure_connected()
        responses = self._client.simGetImages(
            [airsim.ImageRequest("front_custom", airsim.ImageType.Scene, False, False)]
        )
        response = responses[0]
        img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img = img.reshape(response.height, response.width, 3)
        # AirSim returns BGR; convert to RGB for Streamlit/PIL
        return img[:, :, ::-1].copy()

    @staticmethod
    def downscale_rgb(img: np.ndarray, max_width: int = DISPLAY_MAX_WIDTH) -> np.ndarray:
        h, w = img.shape[:2]
        if w <= max_width:
            return img
        scale = max_width / float(w)
        new_w = max_width
        new_h = max(1, int(round(h * scale)))
        try:
            import cv2

            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            from PIL import Image

            pil = Image.fromarray(img)
            pil = pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
            return np.asarray(pil)

    def capture_predicted_frames(
        self,
        *,
        start_xyz: Sequence[float],
        start_yaw: float,
        predicted_actions: Sequence[int],
        image_path: str,
        settle_sec: float = DEFAULT_SETTLE_SEC,
        max_width: int = DISPLAY_MAX_WIDTH,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[np.ndarray], List[List[float]]]:
        """Teleport along predicted poses; return (rgb_frames, poses). No disk I/O."""
        self.ensure_connected()
        poses = replay_trajectory(predicted_actions, start_xyz, start_yaw)
        pitch = -45.0 if "high" in image_path else 0.0
        frames: List[np.ndarray] = []
        n = len(poses)
        for i, pose in enumerate(poses):
            x, y, z, yaw = pose
            self.set_camera_pose(x, y, z, pitch, float(np.rad2deg(yaw)), 0.0)
            if settle_sec > 0:
                time.sleep(settle_sec)
            rgb = self.get_camera_rgb()
            frames.append(self.downscale_rgb(rgb, max_width=max_width))
            if progress_cb is not None:
                progress_cb(i + 1, n)
        return frames, poses
