
from unrealcv import Client  
import cv2  
import numpy as np  
import io
import time
import math
import subprocess, threading
import msgpack
# msgpack-rpc default max_bin_len=1MB; AirSim image payloads are larger (~1920x1080x3).
_orig_msgpack_unpacker = msgpack.Unpacker

def _Unpacker_large_bin(*args, **kwargs):
    kwargs.setdefault("max_bin_len", 32 * 1024 * 1024)
    return _orig_msgpack_unpacker(*args, **kwargs)

msgpack.Unpacker = _Unpacker_large_bin
import airsim
from common import *
import psutil
import requests
import random
import re
import os, json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime

# Ground-truth frame dump: when True, skip Transformers/Torch imports (avoids TF/numpy issues; no VLM).
# Override without editing file: OPENFLY_GT_DUMP=0 for normal VLM eval.
GT_DUMP_MODE = True
if os.environ.get("OPENFLY_GT_DUMP") is not None:
    GT_DUMP_MODE = os.environ.get("OPENFLY_GT_DUMP", "1").strip().lower() in ("1", "true", "yes", "y")
# GT dump JSON: default seen.json; train split: OPENFLY_GT_JSON_PATH=Annotation/train.json
GT_JSON_PATH = os.environ.get("OPENFLY_GT_JSON_PATH", "Annotation/seen.json")
# Output root for GT PNGs: train vs seen. Prefer OPENFLY_NFS_GT_ROOT, else OPENFLY_NFS_SEEN_ROOT (backward compat).
NFS_GT_ROOT = (
    os.environ.get("OPENFLY_NFS_GT_ROOT")
    or os.environ.get("OPENFLY_NFS_SEEN_ROOT")
    or "/nfs/np/mnt/xtb/vln/seen"
)
FRAME_SIZE = (480, 270)
# Prefixes for image_path filter. All six AirSim envs:
# GT_ENV_PREFIXES = (
#     "env_airsim_16/", "env_airsim_18/", "env_airsim_23/", "env_airsim_26/",
#     "env_airsim_gz/", "env_airsim_sh/",
# )
_gt_env_prefixes_env = os.environ.get("OPENFLY_GT_ENV_PREFIXES", "").strip()
if _gt_env_prefixes_env:
    GT_ENV_PREFIXES = tuple(
        p.strip() for p in _gt_env_prefixes_env.split(",") if p.strip()
    )
else:
    GT_ENV_PREFIXES = ("env_airsim_16/",)
GT_MAX_TRAJECTORIES = None  # e.g. 1 for a quick smoke test

GT_IMAGE_PATH_CONTAINS = os.environ.get("OPENFLY_GT_IMAGE_PATH_CONTAINS", "").strip()


def _env_int(name, default=0):
    """Parse int from env; empty or missing -> default."""
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return int(str(v).strip())


def _env_float(name, default=0.0):
    """Parse float from env; empty or missing -> default."""
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return float(str(v).strip())


# Skip first N trajectories after prefix filter (resume after crash / killed sim). Env: OPENFLY_GT_START_INDEX
GT_START_INDEX = _env_int("OPENFLY_GT_START_INDEX", 0) if GT_DUMP_MODE else 0
# GT dump: sleep after each pose update so the sim can render before capturing (no timestamp polling).
GT_AFTER_POSE_SLEEP_SEC = _env_float("OPENFLY_GT_AFTER_POSE_SLEEP_SEC", 0.05) if GT_DUMP_MODE else 0.0

# VLM eval (OPENFLY_GT_DUMP=0): optional sleep after pose before capture (sim render settle).
EVAL_AFTER_POSE_SLEEP_SEC = (
    _env_float("OPENFLY_EVAL_AFTER_POSE_SLEEP_SEC", 0.0) if not GT_DUMP_MODE else 0.0
)
# Resume VLM eval after N trajectories (after env prefix filter if any). Env: OPENFLY_EVAL_START_INDEX
EVAL_START_INDEX = _env_int("OPENFLY_EVAL_START_INDEX", 0) if not GT_DUMP_MODE else 0
# Cap trajectories after start_index (profiling smoke). Env: OPENFLY_EVAL_MAX_TRAJECTORIES, 0 = unlimited
EVAL_MAX_TRAJECTORIES = _env_int("OPENFLY_EVAL_MAX_TRAJECTORIES", 0) if not GT_DUMP_MODE else 0
# Per-step bottleneck logging. Env: OPENFLY_EVAL_TIMING=1
EVAL_TIMING_ENABLED = (
    os.environ.get("OPENFLY_EVAL_TIMING", "").strip().lower() in ("1", "true", "yes", "y")
    if not GT_DUMP_MODE
    else False
)
# When True, never exit early on predicted action 0 — run exactly OPENFLY_EVAL_MAX_STEPS inferences per trajectory (timing / throughput only).
EVAL_DISABLE_EARLY_STOP = (
    os.environ.get("OPENFLY_EVAL_DISABLE_EARLY_STOP", "").strip().lower() in ("1", "true", "yes", "y")
    if not GT_DUMP_MODE
    else False
)


def _env_optional_positive_int(name: str) -> Optional[int]:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return None
    x = int(str(v).strip())
    if x <= 0:
        raise ValueError(f"{name} must be a positive int, got {x!r}")
    return x

# Interleaved nosfx defaults (scripts/qwen3_vl_sft_nosfx.py + --use_default_vln_system_prompt).
QWEN_TEMPORAL_HISTORY_PAST = _env_int("OPENFLY_QWEN_TEMPORAL_HISTORY_PAST", 16) if not GT_DUMP_MODE else 0
if QWEN_TEMPORAL_HISTORY_PAST < 0:
    raise ValueError("OPENFLY_QWEN_TEMPORAL_HISTORY_PAST must be >= 0")


# Run from OpenFly-Platform repo root: conda activate OF3 && python train/eval.py
if not GT_DUMP_MODE:
    import torch
    from PIL import Image

    from qwen3_vl_interleaved_common import (
        bgr_to_pil as _qwen_bgr_to_pil,
        build_interleaved_messages,
        interleaved_window_lo as _qwen_interleaved_window_lo,
        load_processor as _qwen_load_processor,
        parse_vln_action_id as _parse_vln_action_id,
        resolve_prompt_suffix as _resolve_qwen_prompt_suffix,
        resolve_system_prompt as _resolve_qwen_system_prompt,
    )

    def _cuda_sync_all_devices() -> None:
        if not torch.cuda.is_available():
            return
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device=d)
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from extern.hf.configuration_prismatic import OpenFlyConfig
    from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    AutoConfig.register("openvla", OpenFlyConfig)
    AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)

    def _qwen_frames_for_timestep(
        frames_history: List[np.ndarray], timestep: int, temporal_history_past: int
    ) -> List[np.ndarray]:
        """Real frames in [lo, timestep] only (no left-pad; matches interleaved training)."""
        if timestep < 0:
            raise ValueError("timestep must be >= 0")
        if temporal_history_past < 0:
            raise ValueError("temporal_history_past must be >= 0")
        if not frames_history:
            raise ValueError("frames_history empty")
        if timestep >= len(frames_history):
            raise ValueError(f"timestep {timestep} out of range for frames_history len={len(frames_history)}")
        lo = _qwen_interleaved_window_lo(timestep, temporal_history_past)
        return list(frames_history[lo : timestep + 1])

    def _qwen_messages_flat_images(messages: List[Dict[str, Any]]) -> List[Any]:
        flat: List[Any] = []
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    im = part.get("image")
                    if im is not None:
                        flat.append(im)
        return flat

    def _qwen_resize_bgr(bgr: np.ndarray, wh: Tuple[int, int]) -> np.ndarray:
        return cv2.resize(bgr, (wh[0], wh[1]), interpolation=cv2.INTER_AREA)

    def sim_frames_to_pil_rgb(
        frames: List[np.ndarray], resize_wh: Optional[Tuple[int, int]] = None
    ) -> List[Any]:
        pil_images: List[Any] = []
        for bgr in frames:
            if resize_wh is not None:
                bgr = _qwen_resize_bgr(bgr, resize_wh)
            pil_images.append(_qwen_bgr_to_pil(bgr))
        return pil_images

    def _qwen_build_closed_loop_messages(
        system_prompt: str,
        instruction: str,
        frames_bgr: Sequence[np.ndarray],
        past_actions: Sequence[int],
        resize_wh: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        pils = sim_frames_to_pil_rgb(list(frames_bgr), resize_wh=resize_wh)
        return build_interleaved_messages(
            system_prompt,
            instruction,
            _resolve_qwen_prompt_suffix(),
            pils,
            list(past_actions),
        )

    def generate_action_id_qwen3_vl(
        model,
        processor,
        messages: List[Dict[str, Any]],
        device: str,
        max_length: int,
        max_new_tokens: int,
        timing_detail: Optional[Dict[str, float]] = None,
    ) -> int:
        """If timing_detail is a dict, fill sub-second timings (CUDA-synced around GPU segments)."""
        model.eval()

        def _tick() -> float:
            if timing_detail is not None:
                _cuda_sync_all_devices()
            return time.perf_counter()

        def _elapsed(t0: float) -> float:
            if timing_detail is not None:
                _cuda_sync_all_devices()
            return max(0.0, time.perf_counter() - t0)

        if timing_detail is not None:
            timing_detail.clear()

        t0 = _tick()
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tt_chat = _elapsed(t0) if timing_detail is not None else 0.0

        t0 = time.perf_counter()
        all_images = _qwen_messages_flat_images(messages)
        tt_flat = time.perf_counter() - t0 if timing_detail is not None else 0.0

        t0 = _tick()
        inputs = processor(
            text=text,
            images=all_images,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        tt_proc = _elapsed(t0) if timing_detail is not None else 0.0

        tok = processor.tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        def _to_dev(t):
            if isinstance(t, torch.Tensor):
                if t.is_floating_point():
                    return t.to(device, dtype=torch.bfloat16)
                return t.to(device)
            return t

        t0 = _tick()
        inputs = {k: _to_dev(v) for k, v in inputs.items()}
        tt_h2d = _elapsed(t0) if timing_detail is not None else 0.0

        t0 = _tick()
        with torch.inference_mode():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        tt_gen = _elapsed(t0) if timing_detail is not None else 0.0

        in_len = inputs["input_ids"].shape[1]
        new_tokens = gen_ids[0, in_len:]
        t0 = time.perf_counter()
        text_out = tok.decode(new_tokens, skip_special_tokens=True)
        m = re.search(r"[0-9]", text_out)
        tt_dec = time.perf_counter() - t0 if timing_detail is not None else 0.0

        if timing_detail is not None:
            timing_detail["t_chat_template"] = tt_chat
            timing_detail["t_flatten_images"] = tt_flat
            timing_detail["t_processor"] = tt_proc
            timing_detail["t_inputs_to_device"] = tt_h2d
            timing_detail["t_generate"] = tt_gen
            timing_detail["t_decode_action"] = tt_dec

        if not m:
            print(f"Qwen3-VL: no digit in output, raw={text_out!r} -> default 0")
            return 0
        return _parse_vln_action_id(text_out)

    def get_action_qwen3_vl(
        model,
        processor,
        messages_state: List[Dict[str, Any]],
        device: str,
        max_length: int,
        max_new_tokens: int,
        timing_detail: Optional[Dict[str, float]] = None,
    ) -> int:
        aid = generate_action_id_qwen3_vl(
            model,
            processor,
            messages_state,
            device,
            max_length,
            max_new_tokens,
            timing_detail=timing_detail,
        )
        print("Qwen3-VL action id:", aid)
        return aid


def kill_env_process(keyword):
    result = subprocess.run(['pgrep', '-n', keyword], stdout=subprocess.PIPE)
    cr_pid = result.stdout.decode().strip()
    if len(cr_pid) > 0:
        subprocess.run(['kill', '-9', cr_pid])

class AirsimBridge:
    def __init__(self, env_name):
        self.env_name = env_name
        skip_launch = os.environ.get("OPENFLY_SKIP_AIRSIM_LAUNCH", "").strip().lower() in (
            "1", "true", "yes", "y",
        )
        wait_sec = int(os.environ.get("OPENFLY_AIRSIM_WAIT_SEC", "40"))
        if not skip_launch:
            self._sim_thread = threading.Thread(target=self._init_airsim_sim)
            self._sim_thread.start()
            time.sleep(wait_sec)
        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        self._client.enableApiControl(True)
        self._client.armDisarm(True)

        self.distance_to_goal = []
        self.spl = []
        self.success = []
        self.traj_len = 0
        self.pass_len = 1e-3
        self.osr = []

    def _init_airsim_sim(self):
        env_dir = "envs/airsim/" + self.env_name

        if not os.path.exists(env_dir):
            raise ValueError(f"Specified directory {env_dir} does not exist")
        
        command = ["bash", f"{env_dir}/LinuxNoEditor/start.sh"]
        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        # print("Command output:\n", stdout)

    def print_info(self):
        print(f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}")
        return f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}"
    def set_camera_pose(self, x, y, z, pitch, yaw, roll):
        target_pose = airsim.Pose(airsim.Vector3r(x, -y, -z),
                                  airsim.to_quaternion(math.radians(pitch), 0, math.radians(-yaw)))
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        self._client.simSetVehiclePose(target_pose, True)

    def set_drone_pos(self, x, y, z, pitch, yaw, roll):
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        qua = euler_to_quaternion(pitch, -yaw, roll)
        target_pose = airsim.Pose(airsim.Vector3r(x, y, z),
                                  airsim.Quaternionr(qua[0], qua[1], qua[2], qua[3]))
        self._client.simSetVehiclePose(target_pose, True)
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        time.sleep(0.1)

    def _camera_init(self):
        '''Camera initialization'''
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(15), 0, 0))
        self._client.simSetCameraPose("0", camera_pose)
        time.sleep(1)

    def _drone_init(self):
        '''Drone initialization'''
        self.set_drone_pos(0, 0, 0, 0, 0, 0)
        time.sleep(1)

    def get_camera_data(self, camera_type = 'color'):
        valid_types = {'color', 'object_mask', 'depth'}
        if camera_type not in valid_types:
            raise ValueError(f"Invalid camera type. Expected one of {valid_types}, but got '{camera_type}'.")

        if camera_type == 'color':
            image_type = airsim.ImageType.Scene
        elif camera_type == 'depth':
            image_type = airsim.ImageType.DepthPlanar
        else:
            image_type = airsim.ImageType.Segmentation

        responses = self._client.simGetImages([airsim.ImageRequest('front_custom', image_type, False, False)])
        response = responses[0]
        if response.pixels_as_float:
            img_data = np.array(response.image_data_float, dtype=np.float32)
            img_data = np.reshape(img_data, (response.height, response.width))
        else:
            img_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_data = img_data.reshape(response.height, response.width, 3)

        return img_data

    def save_image(self, image_data, file_path):
        cv2.imwrite(file_path, image_data)

    def process_camera_data(self, file_path, camera_type='color'):
        img = self.get_camera_data(camera_type)
        self.save_image(img, file_path)
        print("Image saved")

class UEBridge:
    def __init__(self, ue_ip, ue_port, env_name):
        self.kill_failed_process()
        time.sleep(10)

        # port = self.find_available_port()

        port = random.randint(9000, 9100)
        print(f"Available port: {port}")
        self.modify_port_in_ini(port, env_name)
        ue_port = port

        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_ue_sim)
        self._sim_thread.start()
        time.sleep(15)

        self._client = Client((ue_ip, ue_port))
        self._connection_check()

        self._camera_init()

        # self._drone_init()  
        self.distance_to_goal = []
        self.spl = []
        self.success = []
        self.traj_len = 0
        self.pass_len = 1e-3
        self.osr = []

    def print_info(self):
        print(f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}")
        return f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}"

    def find_available_port(self):
        port = 9000
        while True:
            result = subprocess.run(['lsof', f'-i:{port}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            netstat_output = result.stdout.decode()

            if f'PID' not in netstat_output:
                return port
            port += 1

    def modify_port_in_ini(self, port, ue_env_name):
        ini_file = f"envs/ue/{ue_env_name}/City_UE52/Binaries/Linux/unrealcv.ini"
        with open(ini_file, 'r') as file:
            lines = file.readlines()

        with open(ini_file, 'w') as file:
            for line in lines:
                if line.startswith("Port="):
                    file.write(f"Port={port}\n")
                else:
                    file.write(line)

    def kill_failed_process(self):
        result = subprocess.run(['pgrep', '-n', 'CrashReport'], stdout=subprocess.PIPE)
        cr_pid = result.stdout.decode().strip()
        if len(cr_pid) > 0:
            subprocess.run(['kill', '-9', cr_pid])

        result = subprocess.run(['pgrep', '-n', 'CitySample'], stdout=subprocess.PIPE)
        cr_pid = result.stdout.decode().strip()
        if len(cr_pid) > 0:
            subprocess.run(['kill', '-9', cr_pid])

    def _init_ue_sim(self):
        env_dir = "envs/ue/" + self.env_name
        if not os.path.exists(env_dir):
            raise ValueError(f"Specified directory {env_dir} does not exist")

        command = ["bash", f"{env_dir}/CitySample.sh"]

        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        # print("Command output:\n", stdout)
        time.sleep(2)

    def __del__(self):
        self._client.disconnect()

    def _connection_check(self):
        '''Check if connected'''
        if self._client.connect():
            print('UnrealCV connected successfully')
        else:
            print('UnrealCV is not connected')
            exit()

    def set_camera_pose(self, x, y, z, pitch, yaw, roll):
        '''Set camera position'''
        x = x * 100
        y = - y * 100
        z = z * 100
        camera_settings = {
            'location': {'x': x, 'y': y, 'z': z},
            'rotation': {'pitch': pitch, 'yaw': -yaw, 'roll': roll}
        }

        self._client.request('vset /camera/0/location {x} {y} {z}'.format(**camera_settings['location']))
        self._client.request('vset /camera/1/location {x} {y} {z}'.format(**camera_settings['location']))
        self._client.request('vset /camera/0/rotation {pitch} {yaw} {roll}'.format(**camera_settings['rotation']))
        self._client.request('vset /camera/1/rotation {pitch} {yaw} {roll}'.format(**camera_settings['rotation']))
        print('camera_settings', camera_settings)

    def _camera_init(self):
        '''Camera initialization'''
        time.sleep(2)
        self._client.request('vset /cameras/spawn')
        self._client.request('vset /camera/1/size 1920 1080')
        time.sleep(2)
        self.set_camera_pose(150, 400, 15, 0, 0, 0)  # Initial position
        time.sleep(2)

    def get_camera_data(self, camera_type = 'lit'):
        valid_types = {'lit', 'object_mask', 'depth'}
        if camera_type not in valid_types:
            raise ValueError(f"Invalid camera type. Expected one of {valid_types}, but got '{camera_type}'.")

        if camera_type == 'lit':
            data = self._client.request('vget /camera/1/lit png')
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        elif camera_type == 'object_mask':
            data = self._client.request('vget /camera/1/object_mask png')
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        elif camera_type == 'depth':
            data = self._client.request('vget /camera/1/depth npy')
            depth_np = np.load(io.BytesIO(data))
            return depth_np  # Return depth data

    def save_image(self, image_data, file_path):
        cv2.imwrite(file_path, image_data)

    def process_camera_data(self, file_path, camera_type='lit'):
        img = self.get_camera_data(camera_type)
        self.save_image(img, file_path)

class GSBridge:  
    def __init__(self, env_name):  
        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_gs_sim)
        self._sim_thread.start()
        self.url = "http://localhost:18080/render"
        time.sleep(10)

        self.distance_to_goal = []
        self.spl = []
        self.success = []
        self.traj_len = 0
        self.pass_len = 1e-3
        self.osr = []

    def print_info(self):
        print(f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}")
        return f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}"

    def _init_gs_sim(self):
        # dataset_dir = "envs/gs/" + self.env_name  
        dataset_dir = "/media/pjlabrl/hdd/all_files_relate_to_3dgs/reconstruction_result/nwpu02"
        gs_vis_tool_dir = "envs/gs/SIBR_viewers/"  
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Specified directory {dataset_dir} does not exist")
        command = [
            gs_vis_tool_dir + "install/bin/SIBR_gaussianHierarchyViewer_app",
            "--path", f"{dataset_dir}/camera_calibration/aligned",
            "--scaffold", f"{dataset_dir}/output/scaffold/point_cloud/iteration_30000",
            "--model-path", f"{dataset_dir}/output/merged.hier",
            "--images-path", f"{dataset_dir}/camera_calibration/rectified/images"
        ]
        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        print("Command output:\n", stdout)

    def transform_euler_to_new_frame(self, roll, pitch, yaw):
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        transformation_matrix = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])
        new_R = np.dot(transformation_matrix, R)
        new_roll, new_pitch, new_yaw = rotation_matrix_to_euler_angles(new_R)
        return new_roll, new_pitch, new_yaw
    
    def rotation_matrix_roll(self, roll):
        return np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

    def rotation_matrix_pitch(self, pitch):
        return np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

    def rotation_matrix_yaw(self, yaw):
        return np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

    def transform_to_camera_frame(self, roll, pitch, yaw):
        R_roll = self.rotation_matrix_roll(roll)
        R_pitch = self.rotation_matrix_pitch(pitch)
        R_yaw = self.rotation_matrix_yaw(yaw)
        R_combined = np.dot(R_pitch, np.dot(R_yaw, R_roll))
        QW, QX, QY, QZ = rotation_matrix_to_quaternion(R_combined)
        print(f"QW: {QW}, QX: {QX}, QY: {QY}, QZ: {QZ}")
        transformation_matrix = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        new_R = np.dot(transformation_matrix, R_combined)
        QW_new, QX_new, QY_new, QZ_new = rotation_matrix_to_quaternion(new_R)
        return QW_new, QX_new, QY_new, QZ_new

    def set_camera_pose(self, x, y, z, pitch, yaw, roll, path_params):
        yaw = -yaw
        pitch = -40
        QW, QX, QY, QZ = self.transform_to_camera_frame(math.radians(roll), math.radians(pitch), math.radians(yaw))
        camera_position = world2cam_WXYZ(x, y, z, QW, QX, QY, QZ)
        quat = [QW, QX, QY, QZ]
        camera_id = 0
        image_name = "00000000.png"
        image_data = f"{camera_id} {' '.join(map(str, quat))} {' '.join(map(str, [camera_position[0], camera_position[1], camera_position[2]]))} {0} {image_name}"
        camera_params = f"0 PINHOLE 1436 1077 718.861 718.861 718 538.5"
        data = {
            "camera": camera_params,
            "image": image_data,
            "path": path_params
        }
        print(data)
        try:
            response = requests.post(self.url, data=data)
            if response.status_code == 200:
                print("Request successful!")
                print(response.text) 
            else:
                print(f"Request failed, status code: {response.status_code}")
                print(response.text)
            memory = psutil.virtual_memory()
            print(memory.percent)
            if memory.percent >= 90:
                print("Memory usage is above 90%")
                self.process.terminate()
                self.__init__()
        except requests.RequestException as e:
            print(f"Error during request: {e}")
            time.sleep(20)

    def process_camera_data(self, file_path):
        pass



def get_images(lst,if_his,step):
    if if_his is False:
        return lst[-1]
    else:
        if step == 1:
            if len(lst) >= 2:
                return [lst[-2], lst[-1]]
            elif len(lst) == 1:
                return [lst[0], lst[0]]
        elif step == 2:
            if len(lst) >= 3:
                return lst[-3:]
            elif len(lst) == 2:
                return [lst[0], lst[0], lst[1]]
            elif len(lst) == 1:
                return [lst[0],lst[0], lst[0]]

def convert_to_action_id(action):
    action_dict = {
        "0": np.array([1, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # stop
        "1": np.array([0, 3, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward
        "2": np.array([0, 0, 15, 0, 0, 0, 0, 0]).astype(np.float32),  # turn left 30
        "3": np.array([0, 0, 0, 15, 0, 0, 0, 0]).astype(np.float32),  # turn right 30
        "4": np.array([0, 0, 0, 0, 2, 0, 0, 0]).astype(np.float32),  # go up
        "5": np.array([0, 0, 0, 0, 0, 2, 0, 0]).astype(np.float32),  # go down
        "6": np.array([0, 0, 0, 0, 0, 0, 5, 0]).astype(np.float32),  # move left
        "7": np.array([0, 0, 0, 0, 0, 0, 0, 5]).astype(np.float32),  # move right
        "8": np.array([0, 6, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward 6
        "9": np.array([0, 9, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward 9
    }
    action_values = list(action_dict.values())
    result = 0

    matched = False
    for idx, value in enumerate(action_values):
        if np.array_equal(action, value):
            result = idx
            matched = True
            break
    # If no match is found, default to 0
    if not matched:
        result = 0
    return result

def get_action(policy, processor, image_list, text, his, if_his=False, his_step=0, device="cuda:0"):

    # Otherwise, generate new actions using the policy
    image_list = get_images(image_list, if_his, his_step)

    if isinstance(image_list, np.ndarray):
        img = image_list
        img = Image.fromarray(img)
        images = [img, img, img]
    else:
        images = []
        for img in image_list:
            img = Image.fromarray(img)
            images.append(img)
        
    prompt = text
    inputs = processor(prompt, images).to(device, dtype=torch.bfloat16)
    action = policy.predict_action(**inputs, unnorm_key="vlnv1", do_sample=False)
    print("raw action:", action)
    action = action.round().astype(int)

    # Convert action_chunk to action IDs
    action_id = convert_to_action_id(action)

    cur_action = action_id
    print("Action:", action_id)
    return cur_action

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + 
                     (point2[1] - point1[1])**2 + 
                     (point2[2] - point1[2])**2)

def getPoseAfterMakeAction(new_pose, action):
    x, y, z, yaw = new_pose

    # Define step size
    step_size = 3.0  # Translation step size (units can be adjusted as needed)

    # Update new_pose based on action value
    if action == 0:
        pass
    elif action == 1:
        x += step_size * math.cos(yaw)
        y += step_size * math.sin(yaw)
    elif action == 2:
        yaw += math.radians(30)
    elif action == 3:
        yaw -= math.radians(30)
    elif action == 4:
        z += step_size
    elif action == 5:
        z -= step_size
    elif action == 6:
        x -= step_size * math.sin(yaw)
        y += step_size * math.cos(yaw)
    elif action == 7:
        x += step_size * math.sin(yaw)
        y -= step_size * math.cos(yaw)
    elif action == 8:
        x += step_size * math.cos(yaw) *2
        y += step_size * math.sin(yaw) *2
    elif action == 9:
        x += step_size * math.cos(yaw) *3
        y += step_size * math.sin(yaw) *3
    elif action == 10:
        x += step_size * math.cos(yaw) * 9
        y += step_size * math.sin(yaw) * 9

    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    return [x, y, z, yaw]

def main():
    qwen_ckpt = os.environ.get("OPENFLY_EVAL_QWEN3_CHECKPOINT", "").strip()
    use_qwen_eval = (not GT_DUMP_MODE) and bool(qwen_ckpt)

    if GT_DUMP_MODE:
        eval_info_path = GT_JSON_PATH
    elif use_qwen_eval:
        eval_info_path = (
            os.environ.get("OPENFLY_EVAL_JSON", "data_curated/seen_curated.json").strip()
            or "data_curated/seen_curated.json"
        )
    else:
        eval_info_path = (
            os.environ.get("OPENFLY_EVAL_JSON", "configs/eval_test.json").strip()
            or "configs/eval_test.json"
        )

    with open(eval_info_path, "r") as f:
        all_eval_info = json.loads(f.read())

    if _gt_env_prefixes_env:
        # Explicit OPENFLY_GT_ENV_PREFIXES: apply to both GT dump and VLM eval.
        all_eval_info = [
            x for x in all_eval_info
            if any(x["image_path"].startswith(p) for p in GT_ENV_PREFIXES)
        ]
    elif GT_DUMP_MODE:
        # GT dump with unset env var: default GT_ENV_PREFIXES is ("env_airsim_16/",) only.
        all_eval_info = [
            x for x in all_eval_info
            if any(x["image_path"].startswith(p) for p in GT_ENV_PREFIXES)
        ]
    if GT_DUMP_MODE and GT_IMAGE_PATH_CONTAINS:
        all_eval_info = [
            x for x in all_eval_info
            if GT_IMAGE_PATH_CONTAINS in x.get("image_path", "")
        ]
    if GT_DUMP_MODE and GT_MAX_TRAJECTORIES is not None:
        all_eval_info = all_eval_info[:GT_MAX_TRAJECTORIES]
    if GT_DUMP_MODE and GT_START_INDEX > 0:
        n_before = len(all_eval_info)
        all_eval_info = all_eval_info[GT_START_INDEX:]
        print(
            f"GT resume: OPENFLY_GT_START_INDEX={GT_START_INDEX} "
            f"(skipped {min(GT_START_INDEX, n_before)} trajectories, {len(all_eval_info)} remaining)"
        )
    if (not GT_DUMP_MODE) and EVAL_START_INDEX > 0:
        n_before = len(all_eval_info)
        all_eval_info = all_eval_info[EVAL_START_INDEX:]
        print(
            f"VLM eval resume: OPENFLY_EVAL_START_INDEX={EVAL_START_INDEX} "
            f"(skipped {min(EVAL_START_INDEX, n_before)} trajectories, {len(all_eval_info)} remaining)"
        )
    if (not GT_DUMP_MODE) and EVAL_MAX_TRAJECTORIES > 0:
        _n = len(all_eval_info)
        all_eval_info = all_eval_info[:EVAL_MAX_TRAJECTORIES]
        print(
            f"VLM eval cap: OPENFLY_EVAL_MAX_TRAJECTORIES={EVAL_MAX_TRAJECTORIES} "
            f"(using {_n} trajectories capped to {len(all_eval_info)})"
        )

    vlm_step_limit = _env_int("OPENFLY_EVAL_MAX_STEPS", 40) if not GT_DUMP_MODE else 0
    vlm_device = (
        os.environ.get("OPENFLY_QWEN_DEVICE", "cuda:0").strip() or "cuda:0"
        if not GT_DUMP_MODE
        else "cuda:0"
    )

    processor = None
    policy = None
    qwen_model = None
    eval_ctx: Dict[str, Any] = {}
    qwen_eval_resize_wh: Optional[Tuple[int, int]] = None

    if use_qwen_eval:
        _rw = _env_optional_positive_int("OPENFLY_QWEN_EVAL_IMAGE_WIDTH")
        _rh = _env_optional_positive_int("OPENFLY_QWEN_EVAL_IMAGE_HEIGHT")
        if (_rw is None) ^ (_rh is None):
            raise ValueError(
                "Set both OPENFLY_QWEN_EVAL_IMAGE_WIDTH and OPENFLY_QWEN_EVAL_IMAGE_HEIGHT, or unset both."
            )
        if _rw is not None and _rh is not None:
            qwen_eval_resize_wh = (_rw, _rh)

    if not GT_DUMP_MODE:
        if use_qwen_eval:
            from transformers import AutoConfig
            from transformers import Qwen3VLForConditionalGeneration

            qwen_system = _resolve_qwen_system_prompt()
            qwen_max_length = _env_int("OPENFLY_QWEN_MAX_LENGTH", 1024)
            qwen_max_new = _env_int("OPENFLY_QWEN_MAX_NEW_TOKENS", 16)
            qwen_attn = os.environ.get("OPENFLY_QWEN_ATTN", "sdpa").strip() or "sdpa"
            print(
                f"Qwen3-VL eval: checkpoint={qwen_ckpt} device={vlm_device} "
                f"max_length={qwen_max_length} max_new_tokens={qwen_max_new} attn={qwen_attn} "
                f"system={'on' if qwen_system else 'off'} "
                f"chat_layout=interleaved temporal_past={QWEN_TEMPORAL_HISTORY_PAST}"
            )
            if qwen_eval_resize_wh:
                print(
                    f"Qwen3-VL optional cv2 pre-resize: {qwen_eval_resize_wh[0]}x{qwen_eval_resize_wh[1]} "
                    "(default: native sim frame + checkpoint processor budget)"
                )
            print(f"OPENFLY_EVAL_TIMING={EVAL_TIMING_ENABLED} OPENFLY_EVAL_MAX_TRAJECTORIES={EVAL_MAX_TRAJECTORIES}")
            processor = _qwen_load_processor(qwen_ckpt)
            q_cfg = AutoConfig.from_pretrained(qwen_ckpt, trust_remote_code=True)
            _tc = getattr(q_cfg, "text_config", None)
            if _tc is not None and getattr(_tc, "rope_scaling", None) is None:
                _rp = getattr(_tc, "rope_parameters", None)
                if _rp is not None:
                    _d = _rp if isinstance(_rp, dict) else dict(_rp)
                    _tc.rope_scaling = {
                        "type": _d.get("rope_type", "default"),
                        "rope_theta": _d.get("rope_theta", 5000000),
                        "mrope_section": _d.get("mrope_section", [24, 20, 20]),
                        "mrope_interleaved": _d.get("mrope_interleaved", True),
                    }
            qwen_device_map = os.environ.get("OPENFLY_QWEN_DEVICE_MAP", "").strip()
            if qwen_device_map:
                print(f"Qwen3-VL device_map={qwen_device_map!r}")
                qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    qwen_ckpt,
                    config=q_cfg,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    attn_implementation=qwen_attn,
                    device_map=qwen_device_map,
                )
                # Put inputs on the device of the first parameter (typically cuda:0 when sharded).
                vlm_device = str(next(qwen_model.parameters()).device)
            else:
                qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    qwen_ckpt,
                    config=q_cfg,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    attn_implementation=qwen_attn,
                ).to(vlm_device)
            qwen_model.eval()
            eval_ctx = {
                "system": qwen_system,
                "device": vlm_device,
                "max_len": qwen_max_length,
                "max_new": qwen_max_new,
                "temporal_history_past": QWEN_TEMPORAL_HISTORY_PAST,
                "images_per_step": QWEN_TEMPORAL_HISTORY_PAST + 1,
                "chat_layout": "interleaved",
                "qwen_eval_resize_wh": qwen_eval_resize_wh,
                "eval_timing_enabled": bool(EVAL_TIMING_ENABLED),
            }
        else:
            model_name_or_path = "IPEC-COMMUNITY/openfly-agent-7b"
            processor = AutoProcessor.from_pretrained(model_name_or_path)
            policy = AutoModelForVision2Seq.from_pretrained(
                model_name_or_path,
                attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(vlm_device)

    # Test metrics
    acc = 0
    stop = 0
    data_num = 0

    eval_out_dir = None
    all_predictions: List[Dict[str, Any]] = []
    if not GT_DUMP_MODE:
        eval_out_dir = os.environ.get("OPENFLY_EVAL_OUT_DIR", "").strip()
        if not eval_out_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = "qwen3_seen" if use_qwen_eval else "openvla"
            eval_out_dir = os.path.join("eval_runs", f"{tag}_{ts}")
        os.makedirs(eval_out_dir, exist_ok=True)
        print(f"VLM eval artifacts -> {eval_out_dir}")
        if EVAL_DISABLE_EARLY_STOP:
            print(
                f"OPENFLY_EVAL_DISABLE_EARLY_STOP=1: running exactly {vlm_step_limit} inference "
                "steps per trajectory when not image_error (navigation metrics not meaningful)."
            )

    # Group by environment type
    env_groups = {}
    for item in all_eval_info:
        env_type = item["image_path"].split("/")[0]  # Get environment type
        if env_type not in env_groups:
            env_groups[env_type] = []
        env_groups[env_type].append(item)
    
    # Process each environment type sequentially (sample indices continue across env groups when resuming)
    sample_idx_base = GT_START_INDEX if GT_DUMP_MODE else EVAL_START_INDEX
    for env_name, eval_info in env_groups.items():
        print(f"Starting evaluation of environment: {env_name}, with {len(eval_info)} data entries")
        time.sleep(5)
        
        # Create appropriate environment bridge based on environment type
        if "airsim" in env_name:
            env_bridge = AirsimBridge(env_name)
            pos_ratio = 1.0
        elif "ue" in env_name:
            env_bridge = UEBridge(ue_ip="127.0.0.1", ue_port="9000", env_name=env_name)
            pos_ratio = 1.0
        elif "gs" in env_name:
            env_bridge = GSBridge(env_name)
            pos_ratio = 5.15
        else:
            print(f"Unknown environment type: {env_name}, skipping")
            sample_idx_base += len(eval_info)
            continue

        # Evaluate all data for current environment
        for idx, item in enumerate(eval_info, start=sample_idx_base):
            acts = []  # Reset action list
            data_num += 1
            pos_list = item['pos']
            text = item['gpt_instruction']
            start_postion = pos_list[0]
            start_yaw = item['yaw'][0]
            new_pose = [start_postion[0], start_postion[1], start_postion[2], start_yaw]
            end_position = pos_list[-1]
            print(f"Sample {idx}: {start_postion} -> {end_position}, initial heading: {start_yaw}")
            
            stop_error = 1
            image_error = False

            out_dir = None
            if GT_DUMP_MODE:
                parts = item["image_path"].split("/")
                env_key = parts[0]
                rel = os.path.join(*parts[1:]) if len(parts) > 1 else ""
                out_dir = os.path.join(NFS_GT_ROOT, env_key, rel)
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except OSError as e:
                    raise RuntimeError(f"Cannot create GT dump dir {out_dir}: {e}") from e
            
            # Set camera pose
            pitch = -45.0 if 'high' in item['image_path'] else 0.0
            env_bridge.set_camera_pose(
                start_postion[0]/pos_ratio, 
                start_postion[1]/pos_ratio, 
                start_postion[2]/pos_ratio, 
                pitch, 
                np.rad2deg(start_yaw), 
                0
            )
            if GT_DUMP_MODE and GT_AFTER_POSE_SLEEP_SEC > 0:
                time.sleep(GT_AFTER_POSE_SLEEP_SEC)
            elif (not GT_DUMP_MODE) and EVAL_AFTER_POSE_SLEEP_SEC > 0:
                time.sleep(EVAL_AFTER_POSE_SLEEP_SEC)

            step = 0
            flag_osr = 0
            frames_history: List[np.ndarray] = []
            env_bridge.pass_len = 1e-3
            old_pose = new_pose
            resize_wh = eval_ctx.get("qwen_eval_resize_wh") if use_qwen_eval else None

            if GT_DUMP_MODE:
                step_limit = len(item["action"])
            else:
                step_limit = vlm_step_limit

            timing_on_qwen = bool(
                use_qwen_eval and EVAL_TIMING_ENABLED and (eval_out_dir is not None)
            )
            timing_jsonl_fp = None
            traj_timing_agg: Dict[str, float] = {}
            traj_wall_started = False
            traj_started_at = 0.0
            traj_timing_meta: Optional[Dict[str, Any]] = None

            if timing_on_qwen:
                traj_started_at = time.perf_counter()
                traj_wall_started = True
                timing_jsonl_fp = open(
                    os.path.join(eval_out_dir or ".", "timing_steps.jsonl"), "a", encoding="utf-8", buffering=1
                )
            while step < step_limit:
                try:
                    if timing_on_qwen:
                        ts: Dict[str, Any] = {
                            "event": "eval_timing_step",
                            "env": env_name,
                            "sample_index": idx,
                            "step": step,
                        }
                        ts_wall_start = time.perf_counter()

                    if timing_on_qwen:
                        t0 = time.perf_counter()
                    raw_image = env_bridge.get_camera_data()
                    if timing_on_qwen:
                        ts["t_capture_sec"] = time.perf_counter() - t0

                    if GT_DUMP_MODE:
                        idx_list = item.get("index_list", [])
                        frame_name = (
                            f"{idx_list[step]}.png" if step < len(idx_list) else f"{step:06d}.png"
                        )
                        small = cv2.resize(
                            raw_image, FRAME_SIZE, interpolation=cv2.INTER_AREA
                        )
                        cv2.imwrite(os.path.join(out_dir, frame_name), small)
                    else:
                        if timing_on_qwen:
                            t0 = time.perf_counter()
                        cv2.imwrite("test/cur_img.jpg", raw_image)
                        if timing_on_qwen:
                            ts["t_debug_imwrite_sec"] = time.perf_counter() - t0
                    image = raw_image

                    if timing_on_qwen:
                        t0 = time.perf_counter()
                    frames_history.append(image)
                    if timing_on_qwen:
                        ts["t_history_append_sec"] = time.perf_counter() - t0

                    if GT_DUMP_MODE:
                        model_action = int(item["action"][step])
                    elif use_qwen_eval:
                        past_i = int(eval_ctx["temporal_history_past"])
                        if timing_on_qwen:
                            t0 = time.perf_counter()
                        frames_sel = _qwen_frames_for_timestep(frames_history, step, past_i)
                        if timing_on_qwen:
                            ts["t_qwen_frame_select_sec"] = time.perf_counter() - t0

                        lo = _qwen_interleaved_window_lo(step, past_i)
                        window_past_actions = list(acts[lo:step])

                        if timing_on_qwen:
                            t0 = time.perf_counter()
                        messages_step = _qwen_build_closed_loop_messages(
                            eval_ctx.get("system", ""),
                            text,
                            frames_sel,
                            window_past_actions,
                            resize_wh=resize_wh,
                        )
                        if timing_on_qwen:
                            ts["t_messages_build_sec"] = time.perf_counter() - t0

                        inner_detail: Optional[Dict[str, float]] = {} if timing_on_qwen else None
                        if timing_on_qwen:
                            t0 = time.perf_counter()
                        model_action = get_action_qwen3_vl(
                            qwen_model,
                            processor,
                            messages_step,
                            eval_ctx["device"],
                            eval_ctx["max_len"],
                            eval_ctx["max_new"],
                            timing_detail=inner_detail,
                        )
                        if timing_on_qwen:
                            ts["t_model_forward_total_sec"] = time.perf_counter() - t0
                            if inner_detail is not None:
                                for _k, _v in inner_detail.items():
                                    ts[f"{_k}_sec"] = _v

                        acts.append(model_action)
                    else:
                        model_action = get_action(
                            policy,
                            processor,
                            frames_history,
                            text,
                            acts,
                            if_his=True,
                            his_step=2,
                            device=vlm_device,
                        )
                        acts.append(model_action)

                    new_pose = getPoseAfterMakeAction(new_pose, model_action)

                    print(
                        f"Environment: {env_name}, Sample: {idx}, Step: {step}, Action: {model_action}, New position: {new_pose}"
                    )

                    if timing_on_qwen:
                        t0 = time.perf_counter()
                    env_bridge.set_camera_pose(
                        new_pose[0] / pos_ratio,
                        new_pose[1] / pos_ratio,
                        new_pose[2] / pos_ratio,
                        pitch,
                        np.rad2deg(new_pose[3]),
                        0,
                    )
                    if timing_on_qwen:
                        ts["t_pose_apply_sec"] = time.perf_counter() - t0

                    if timing_on_qwen:
                        t0 = time.perf_counter()
                    if GT_DUMP_MODE and GT_AFTER_POSE_SLEEP_SEC > 0:
                        time.sleep(GT_AFTER_POSE_SLEEP_SEC)
                    elif (not GT_DUMP_MODE) and EVAL_AFTER_POSE_SLEEP_SEC > 0:
                        time.sleep(EVAL_AFTER_POSE_SLEEP_SEC)
                    if timing_on_qwen:
                        ts["t_after_pose_sleep_sec"] = time.perf_counter() - t0

                    if timing_on_qwen:
                        t0 = time.perf_counter()
                    env_bridge.pass_len += calculate_distance(old_pose, new_pose)
                    dis = calculate_distance(end_position, new_pose)
                    if dis < 20 and flag_osr != 2:
                        flag_osr = 2
                        env_bridge.osr.append(1)
                    old_pose = new_pose
                    if timing_on_qwen:
                        ts["t_step_metrics_sec"] = time.perf_counter() - t0

                    if timing_on_qwen:
                        ts["t_step_wall_sec"] = time.perf_counter() - ts_wall_start
                        acc_keys = (
                            "t_capture_sec",
                            "t_debug_imwrite_sec",
                            "t_history_append_sec",
                            "t_qwen_frame_select_sec",
                            "t_pil_convert_sec",
                            "t_resize_sec",
                            "t_messages_build_sec",
                            "t_model_forward_total_sec",
                            "t_pose_apply_sec",
                            "t_after_pose_sleep_sec",
                            "t_step_metrics_sec",
                        )
                        outer_sum = sum(float(ts.get(k, 0.0) or 0.0) for k in acc_keys)
                        ts["t_components_sum_sec"] = outer_sum
                        ts["t_unaccounted_sec"] = float(ts["t_step_wall_sec"]) - outer_sum
                        line_d = {k: ts[k] for k in ts if k != "event"}
                        _parts: List[str] = []
                        for _k in sorted(line_d.keys()):
                            _v = line_d[_k]
                            if _k.endswith("_sec") and isinstance(_v, (int, float)):
                                _parts.append(f"{_k}={float(_v) * 1000.0:.3f}ms")
                            else:
                                _parts.append(f"{_k}={_v}")
                        print("EVAL_TIMING " + " ".join(_parts))
                        timing_jsonl_fp.write(json.dumps(ts, ensure_ascii=False) + "\n")
                        for k, v in ts.items():
                            if k.endswith("_sec") and isinstance(v, (int, float)):
                                traj_timing_agg[k] = traj_timing_agg.get(k, 0.0) + float(v)

                    if model_action == 0:
                        stop_error = 0
                        if not EVAL_DISABLE_EARLY_STOP:
                            break
                    step += 1
                except Exception as e:
                    print(f"Error processing image: {e}")
                    image_error = True
                    break

            if timing_jsonl_fp is not None:
                timing_jsonl_fp.close()
                timing_jsonl_fp = None
            if timing_on_qwen and traj_wall_started:
                traj_wall_sec = max(0.0, time.perf_counter() - traj_started_at)
                n_steps = len(acts)
                mean_wall = traj_wall_sec / max(1, n_steps)
                summary = {
                    "event": "eval_timing_trajectory",
                    "env": env_name,
                    "sample_index": idx,
                    "n_inference_steps": n_steps,
                    "traj_wall_sec": traj_wall_sec,
                    "mean_step_wall_sec": mean_wall,
                    "stage_sums_sec": {k: v for k, v in sorted(traj_timing_agg.items())},
                }
                sum_step_wall = traj_timing_agg.get("t_step_wall_sec", 0.0)
                _coarse = (
                    "t_capture_sec",
                    "t_debug_imwrite_sec",
                    "t_history_append_sec",
                    "t_qwen_frame_select_sec",
                    "t_pil_convert_sec",
                    "t_resize_sec",
                    "t_messages_build_sec",
                    "t_model_forward_total_sec",
                    "t_pose_apply_sec",
                    "t_after_pose_sleep_sec",
                    "t_step_metrics_sec",
                )
                summary["fraction_of_summed_step_wall"] = {
                    k: (traj_timing_agg.get(k, 0.0) / sum_step_wall) if sum_step_wall > 0 else 0.0
                    for k in _coarse
                }
                _inner = (
                    "t_chat_template_sec",
                    "t_flatten_images_sec",
                    "t_processor_sec",
                    "t_inputs_to_device_sec",
                    "t_generate_sec",
                    "t_decode_action_sec",
                )
                _sum_mf = traj_timing_agg.get("t_model_forward_total_sec", 0.0)
                summary["fraction_of_model_forward_total"] = {
                    k: (traj_timing_agg.get(k, 0.0) / _sum_mf) if _sum_mf > 0 else 0.0
                    for k in _inner
                }
                sp = os.path.join(eval_out_dir or ".", "timing_summary.json")
                summaries: List[Dict[str, Any]] = []
                if os.path.isfile(sp):
                    try:
                        with open(sp, "r", encoding="utf-8") as rf:
                            prev = json.load(rf)
                        if isinstance(prev, list):
                            summaries = prev
                        elif isinstance(prev, dict) and "trajectories" in prev:
                            summaries = list(prev["trajectories"])
                    except (json.JSONDecodeError, OSError):
                        summaries = []
                summaries.append(summary)
                with open(sp, "w", encoding="utf-8") as wf:
                    json.dump({"trajectories": summaries}, wf, ensure_ascii=False, indent=2)
                print(
                    f"EVAL_TIMING_TRAJ sample={idx} steps={n_steps} traj_wall_sec={traj_wall_sec:.3f} "
                    f"mean_step_wall_ms={mean_wall * 1000.0:.2f} -> {sp}"
                )
                traj_timing_meta = {
                    "traj_wall_sec": traj_wall_sec,
                    "n_inference_steps": n_steps,
                    "mean_step_wall_sec": mean_wall,
                }

            dis = calculate_distance(end_position, new_pose)
            env_bridge.traj_len = calculate_distance(end_position, start_postion)
            env_bridge.distance_to_goal.append(dis)
            if dis < 20:
                env_bridge.success.append(1)
                env_bridge.spl.append(env_bridge.traj_len / env_bridge.pass_len)
                acc += 1
            else:
                env_bridge.success.append(0)
                env_bridge.spl.append(0)
            if flag_osr == 0:
                env_bridge.osr.append(0)
            env_bridge.print_info()

            if not GT_DUMP_MODE and eval_out_dir is not None:
                stopped_by_model = bool(acts) and acts[-1] == 0
                hit_max = bool(acts) and (not stopped_by_model) and (len(acts) >= vlm_step_limit)
                if EVAL_DISABLE_EARLY_STOP:
                    # Fixed-step runs: trajectory length is capped by OPENFLY_EVAL_MAX_STEPS, not stop action.
                    hit_max = bool(acts) and (len(acts) >= vlm_step_limit)
                all_predictions.append(
                    {
                        "sample_index": idx,
                        "environment": env_name,
                        "image_path": item.get("image_path"),
                        "gpt_instruction": item.get("gpt_instruction"),
                        "predicted_actions": list(acts),
                        "gt_actions": item.get("action"),
                        "num_steps": len(acts),
                        "final_distance": float(dis),
                        "success": 1 if dis < 20 else 0,
                        "spl": float(env_bridge.spl[-1]),
                        "osr_hit": int(env_bridge.osr[-1]),
                        "stopped_by_model": stopped_by_model,
                        "hit_max_steps": hit_max,
                        "image_error": image_error,
                        "qwen_temporal_history_past": eval_ctx.get("temporal_history_past") if use_qwen_eval else None,
                        "qwen_images_per_step": eval_ctx.get("images_per_step") if use_qwen_eval else None,
                        "qwen_chat_layout": eval_ctx.get("chat_layout") if use_qwen_eval else None,
                        "qwen_eval_resize_wh": list(eval_ctx["qwen_eval_resize_wh"])
                        if use_qwen_eval and eval_ctx.get("qwen_eval_resize_wh")
                        else None,
                        "eval_timing_enabled": eval_ctx.get("eval_timing_enabled")
                        if use_qwen_eval
                        else None,
                        "timing_trajectory": traj_timing_meta,
                        "eval_disable_early_stop": EVAL_DISABLE_EARLY_STOP,
                    }
                )

            if image_error:
                continue

        sample_idx_base += len(eval_info)

        # Clean up environment resources
        print(f"Completed evaluation of environment {env_name}")
        kill_env_process("AirVLN")
        kill_env_process("guangzhou")
        kill_env_process("shanghai")
        kill_env_process("CitySample")
        kill_env_process("CrashReport")

        del env_bridge
        import gc
        gc.collect()
    
    # Final results
    final_acc = acc / data_num if data_num > 0 else 0
    final_stop = 1 - stop / data_num if data_num > 0 else 0
    
    print(f"\nEvaluation complete!")
    print(f"Total samples: {data_num}")
    print(f"Final accuracy: {final_acc:.4f}")
    print(f"Final stop rate: {final_stop:.4f}")

    if not GT_DUMP_MODE and eval_out_dir and all_predictions:
        pred_path = os.path.join(eval_out_dir, "predictions.json")
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(all_predictions, f, ensure_ascii=False, indent=2)
        n = len(all_predictions)
        ok = [r for r in all_predictions if not r.get("image_error")]
        n_ok = len(ok)
        mean_sr = sum(r.get("success", 0) for r in ok) / n_ok if n_ok else 0.0
        mean_osr = sum(r.get("osr_hit", 0) for r in ok) / n_ok if n_ok else 0.0
        mean_ne = sum(r.get("final_distance", 0.0) for r in ok) / n_ok if n_ok else 0.0
        mean_spl = sum(float(r.get("spl", 0.0)) for r in ok) / n_ok if n_ok else 0.0
        metrics: Dict[str, Any] = {
            "n_samples": n,
            "n_samples_no_image_error": n_ok,
            "mean_success_rate": mean_sr,
            "mean_oracle_success_rate": mean_osr,
            "mean_navigation_error": mean_ne,
            "mean_spl": mean_spl,
            "eval_json": eval_info_path,
            "max_steps_per_trajectory": vlm_step_limit,
            "use_qwen3_vl": use_qwen_eval,
            "checkpoint": qwen_ckpt if use_qwen_eval else "IPEC-COMMUNITY/openfly-agent-7b",
            "aggregate_success_rate": final_acc,
        }
        if use_qwen_eval:
            metrics["qwen_max_length"] = eval_ctx.get("max_len")
            metrics["qwen_max_new_tokens"] = eval_ctx.get("max_new")
            metrics["qwen_device"] = eval_ctx.get("device")
            metrics["qwen_system_prompt"] = "on" if eval_ctx.get("system") else "off"
            metrics["qwen_temporal_history_past"] = eval_ctx.get("temporal_history_past")
            metrics["qwen_images_per_step"] = eval_ctx.get("images_per_step")
            metrics["qwen_chat_layout"] = eval_ctx.get("chat_layout")
        _rz = eval_ctx.get("qwen_eval_resize_wh") if use_qwen_eval else None
        metrics["openfly_qwen_eval_resize_wh"] = list(_rz) if _rz else None
        metrics["openfly_eval_timing_enabled"] = EVAL_TIMING_ENABLED
        metrics["openfly_eval_max_trajectories"] = (
            EVAL_MAX_TRAJECTORIES if EVAL_MAX_TRAJECTORIES > 0 else None
        )
        metrics["openfly_eval_disable_early_stop"] = EVAL_DISABLE_EARLY_STOP
        met_path = os.path.join(eval_out_dir, "metrics.json")
        with open(met_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Wrote {pred_path}")
        print(f"Wrote {met_path}")


if __name__ == '__main__':
    main()
