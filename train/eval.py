
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
from typing import Any, Dict, List, Sequence

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

# --- Qwen3-VL eval (Phase 1: single live frame, matches `scripts/qwen3_vl_sft.py` 1-frame SFT) ---
# Env when OPENFLY_GT_DUMP=0:
#   OPENFLY_EVAL_QWEN3_CHECKPOINT   Path to HF Trainer output (required for Qwen eval; if unset, OpenFly OpenVLA is used).
#   OPENFLY_QWEN_DEVICE             Default cuda:0
#   OPENFLY_QWEN_MAX_LENGTH         Default 1024 (match training)
#   OPENFLY_QWEN_MAX_NEW_TOKENS     Default 16
#   OPENFLY_QWEN_ATTN               sdpa | flash_attention_2 | eager (default sdpa)
#   OPENFLY_QWEN_NO_SYSTEM_PROMPT   1/true to omit system message (match runs trained with --no_system_prompt)
#   OPENFLY_QWEN_SYSTEM_PROMPT_FILE UTF-8 file for custom system text
#   OPENFLY_QWEN_SYSTEM_PROMPT      Inline system text (overrides default VLN block below when non-empty and no file)
#
# Live AirSim frames are BGR uint8; training PNGs are RGB via PIL — we BGR→RGB before the Qwen processor.
# PNGs on disk (e.g. seen_curated) are already RGB; do not run this path on those without skipping cvtColor.
#
# v1: single live frame only (no multi-frame env; matches SFT --history_frames 1). Tail-N frames for training
# parity = deferred Phase 2 when requested.

# Keep in sync with `scripts/qwen3_vl_sft.py` DEFAULT_VLN_SYSTEM_PROMPT.
DEFAULT_VLN_SYSTEM_PROMPT = """You are an AI assistant controlling a flying drone. Navigate using the current camera view and the human instruction by replying with exactly one action id from 0 to 10 (digits only, no other text). Action meanings:
0. Stop
1. Move forward (×1)
2. Turn left (~30°)
3. Turn right (~30°)
4. Move up
5. Move down
8. Move forward (×2)
9. Move forward (×3)
10. Move forward (×9)
"""

# Must match `VlnActionDataset` default `prompt_suffix` in `scripts/qwen3_vl_sft.py`.
QWEN_VLN_PROMPT_SUFFIX = "\nNext action id (0-10): "

# Run from OpenFly-Platform repo root. Use a `transformers` build with Qwen3-VL for checkpoint eval (e.g. TrainOF venv).
if not GT_DUMP_MODE:
    import torch
    from PIL import Image


def _register_and_load_openfly_vla(device: str = "cuda:0"):
    """Load OpenFly OpenVLA; registers custom HF classes (only call when not using Qwen3-VL checkpoint)."""
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    from extern.hf.configuration_prismatic import OpenFlyConfig
    from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenFlyConfig)
    AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)

    model_name_or_path = "IPEC-COMMUNITY/openfly-agent-7b"
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    policy = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    return processor, policy


def _resolve_qwen_system_prompt() -> str:
    if os.environ.get("OPENFLY_QWEN_NO_SYSTEM_PROMPT", "").strip().lower() in ("1", "true", "yes", "y"):
        return ""
    path = os.environ.get("OPENFLY_QWEN_SYSTEM_PROMPT_FILE", "").strip()
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"OPENFLY_QWEN_SYSTEM_PROMPT_FILE not found: {p}")
        return p.read_text(encoding="utf-8").strip()
    override = os.environ.get("OPENFLY_QWEN_SYSTEM_PROMPT", "").strip()
    if override:
        return override
    return DEFAULT_VLN_SYSTEM_PROMPT.strip()


def latest_sim_frame_to_pil_rgb(image_list: List[np.ndarray]) -> List[Any]:
    """
    Phase 1: latest sim frame only. AirSim `uint8` HxWx3 is treated as BGR -> RGB for PIL (training used RGB PNGs).
    """
    if not image_list:
        raise ValueError("image_list is empty; expected at least one captured frame")
    bgr = image_list[-1]
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape {getattr(bgr, 'shape', None)}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return [Image.fromarray(rgb)]


def build_qwen3_vl_messages(
    instruction: str,
    pil_images: Sequence[Any],
    system_prompt: str,
    prompt_suffix: str = QWEN_VLN_PROMPT_SUFFIX,
) -> List[Dict[str, Any]]:
    """Same chat prefix layout as `Qwen3VlActionCollator` in `scripts/qwen3_vl_sft.py` (no assistant turn)."""
    user_content: List[Dict[str, Any]] = []
    for im in pil_images:
        user_content.append({"type": "image", "image": im})
    user_content.append({"type": "text", "text": f"{instruction}{prompt_suffix}"})

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": user_content})
    return messages


def generate_action_id_qwen3_vl(
    model,
    processor,
    messages: List[Dict[str, Any]],
    pil_images: Sequence[Any],
    device: str,
    max_length: int,
    max_new_tokens: int,
) -> int:
    model.eval()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text,
        images=list(pil_images),
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def _to_dev(t):
        if isinstance(t, torch.Tensor):
            if t.is_floating_point():
                return t.to(device, dtype=torch.bfloat16)
            return t.to(device)
        return t

    inputs = {k: _to_dev(v) for k, v in inputs.items()}

    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    in_len = inputs["input_ids"].shape[1]
    new_tokens = gen_ids[0, in_len:]
    text_out = tok.decode(new_tokens, skip_special_tokens=True)
    m = re.search(r"\b(10|[0-9])\b", text_out)
    if not m:
        print(f"Qwen3-VL: no digit in output, raw={text_out!r} -> default 0")
        return 0
    return int(m.group(1))


def get_action_qwen3_vl(
    model,
    processor,
    image_list: List[np.ndarray],
    instruction: str,
    system_prompt: str,
    device: str,
    max_length: int,
    max_new_tokens: int,
    prompt_suffix: str = QWEN_VLN_PROMPT_SUFFIX,
) -> int:
    pil_images = latest_sim_frame_to_pil_rgb(image_list)
    messages = build_qwen3_vl_messages(instruction, pil_images, system_prompt, prompt_suffix=prompt_suffix)
    aid = generate_action_id_qwen3_vl(
        model, processor, messages, pil_images, device, max_length, max_new_tokens
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
        "10": np.array([0, 27, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward 27 (9x3)
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

def get_action(policy, processor, image_list, text, his, if_his=False, his_step=0):

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
    inputs = processor(prompt, images).to("cuda:0", dtype=torch.bfloat16)
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
    eval_info_path = GT_JSON_PATH if GT_DUMP_MODE else "configs/eval_test.json"
    with open(eval_info_path, "r") as f:
        all_eval_info = json.loads(f.read())

    if GT_DUMP_MODE and GT_ENV_PREFIXES:
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

    use_qwen_eval = False
    qwen_model = None
    processor = None
    policy = None
    eval_ctx: Dict[str, Any] = {}

    if not GT_DUMP_MODE:
        qwen_ckpt = os.environ.get("OPENFLY_EVAL_QWEN3_CHECKPOINT", "").strip()
        qwen_device = os.environ.get("OPENFLY_QWEN_DEVICE", "cuda:0").strip() or "cuda:0"
        qwen_max_length = _env_int("OPENFLY_QWEN_MAX_LENGTH", 1024)
        qwen_max_new = _env_int("OPENFLY_QWEN_MAX_NEW_TOKENS", 16)
        qwen_attn = os.environ.get("OPENFLY_QWEN_ATTN", "sdpa").strip() or "sdpa"

        if qwen_ckpt:
            use_qwen_eval = True
            from transformers import AutoProcessor as HFAutoProcessor
            from transformers import Qwen3VLForConditionalGeneration

            qwen_system = _resolve_qwen_system_prompt()
            print(
                f"Qwen3-VL eval: checkpoint={qwen_ckpt} device={qwen_device} "
                f"max_length={qwen_max_length} max_new_tokens={qwen_max_new} attn={qwen_attn} "
                f"system={'on' if qwen_system else 'off'}"
            )
            processor = HFAutoProcessor.from_pretrained(qwen_ckpt, trust_remote_code=True)
            qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                qwen_ckpt,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=qwen_attn,
            ).to(qwen_device)
            qwen_model.eval()
            eval_ctx = {
                "system": qwen_system,
                "device": qwen_device,
                "max_len": qwen_max_length,
                "max_new": qwen_max_new,
            }
        else:
            processor, policy = _register_and_load_openfly_vla(device=qwen_device)

    # Test metrics
    acc = 0
    stop = 0
    data_num = 0
    MAX_STEP = 100

    # Group by environment type
    env_groups = {}
    for item in all_eval_info:
        env_type = item["image_path"].split("/")[0]  # Get environment type
        if env_type not in env_groups:
            env_groups[env_type] = []
        env_groups[env_type].append(item)
    
    # Process each environment type sequentially (sample indices continue across env groups when resuming)
    sample_idx_base = GT_START_INDEX if GT_DUMP_MODE else 0
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

            step = 0
            flag_osr = 0
            image_list = []
            env_bridge.pass_len = 1e-3
            old_pose = new_pose

            step_limit = len(item["action"]) if GT_DUMP_MODE else MAX_STEP
            while step < step_limit:
                try:
                    raw_image = env_bridge.get_camera_data()
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
                        cv2.imwrite("test/cur_img.jpg", raw_image)
                    image = raw_image
                    
                    image_list.append(image)
                    if GT_DUMP_MODE:
                        model_action = int(item["action"][step])
                    elif use_qwen_eval:
                        model_action = get_action_qwen3_vl(
                            qwen_model,
                            processor,
                            image_list,
                            text,
                            eval_ctx["system"],
                            eval_ctx["device"],
                            eval_ctx["max_len"],
                            eval_ctx["max_new"],
                        )
                    else:
                        model_action = get_action(policy, processor, image_list, text, acts, if_his=True, his_step=2)
                    acts.append(model_action)
                    new_pose = getPoseAfterMakeAction(new_pose, model_action)
                    print(f"Environment: {env_name}, Sample: {idx}, Step: {step}, Action: {model_action}, New position: {new_pose}")
                    env_bridge.set_camera_pose(
                        new_pose[0]/pos_ratio, 
                        new_pose[1]/pos_ratio, 
                        new_pose[2]/pos_ratio, 
                        pitch, 
                        np.rad2deg(new_pose[3]), 
                        0
                    )
                    if GT_DUMP_MODE and GT_AFTER_POSE_SLEEP_SEC > 0:
                        time.sleep(GT_AFTER_POSE_SLEEP_SEC)
                    env_bridge.pass_len += calculate_distance(old_pose, new_pose)
                    dis = calculate_distance(end_position, new_pose)
                    if dis < 20 and flag_osr != 2:
                        flag_osr = 2
                        env_bridge.osr.append(1)
                    old_pose = new_pose

                    if model_action == 0:
                        stop_error = 0
                        break
                    step += 1
                except Exception as e:
                    print(f"Error processing image: {e}")
                    image_error = True
                    break

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


if __name__ == '__main__':
    main()
