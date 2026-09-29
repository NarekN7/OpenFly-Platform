"""Frame viewer: AirSim-captured predicted-path slideshow with prompt + map."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_VIZ_DIR = Path(__file__).resolve().parents[1]
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))

import streamlit as st

from airsim_session import get_shared_session
from common import action_label, dict_to_record, format_actions
from frame_cache import TrajectoryFrames, get_shared_frame_cache, lookup_record
from physics import calculate_distance
from plot import render_pseudo_map_from_record

st.set_page_config(page_title="Frame Viewer", page_icon="🎬", layout="wide")


def _ensure_session() -> None:
    st.session_state.airsim_session = get_shared_session()
    st.session_state.frame_cache = get_shared_frame_cache()
    if "frame_step" not in st.session_state:
        st.session_state.frame_step = 0
    if "autoplay" not in st.session_state:
        st.session_state.autoplay = False
    if "autoplay_fps" not in st.session_state:
        st.session_state.autoplay_fps = 1.0


def _step_action_label(actions: list[int], step: int) -> str:
    if step < 0:
        return "—"
    if step >= len(actions):
        return "end"
    return action_label(actions[step])


def _resolve_record_dict() -> dict | None:
    """Load trajectory from query params (new tab) via process-wide registry."""
    qp = st.query_params
    env = qp.get("env")
    sample_raw = qp.get("sample")
    if env is not None and sample_raw is not None:
        try:
            sample = int(sample_raw)
        except (TypeError, ValueError):
            return None
        found = lookup_record(env, sample)
        if found is not None:
            return found
        # Frames may already carry the record after a prior capture
        entry = get_shared_frame_cache().get(env, sample)
        if entry is not None and entry.record:
            return entry.record
        return None
    return st.session_state.get("viewer_record")


def _capture_frames(record_dict: dict) -> bool:
    session = st.session_state.airsim_session
    cache = st.session_state.frame_cache
    record = dict_to_record(record_dict)
    if session.status != "Connected" or session.env_name != record.environment:
        st.error(
            f"Engine not connected for **{record.environment}**. "
            "Start it from the grid tab, then reload this page."
        )
        return False
    if not record.start_xyz or not record.target_xyz:
        st.error("Missing start/target pose.")
        return False

    progress = st.progress(0.0, text="Capturing frames…")

    def _cb(done: int, total: int) -> None:
        progress.progress(done / max(total, 1), text=f"Loading frame {done}/{total}")

    try:
        frames, poses = session.capture_predicted_frames(
            start_xyz=record.start_xyz,
            start_yaw=record.start_yaw,
            predicted_actions=record.predicted_actions,
            image_path=record.image_path,
            progress_cb=_cb,
        )
    except Exception as e:
        progress.empty()
        st.error(f"Capture failed: {e}")
        return False

    progress.empty()
    cache.put(
        TrajectoryFrames(
            environment=record.environment,
            sample_index=record.sample_index,
            image_path=record.image_path,
            frames_rgb=frames,
            poses=poses,
            predicted_actions=list(record.predicted_actions),
            record=record_dict,
        )
    )
    st.session_state.frame_step = 0
    return True


def main() -> None:
    _ensure_session()

    record_dict = _resolve_record_dict()
    if not record_dict:
        st.warning(
            "No trajectory selected. Open the grid tab first (loads metadata), "
            "start the engine, then click **Open frames**."
        )
        st.page_link("app.py", label="← Back to grid")
        st.stop()

    record = dict_to_record(record_dict)
    cache = st.session_state.frame_cache
    frames_entry = cache.get(record.environment, record.sample_index)

    st.title(f"Frame Viewer — {record.environment} #{record.sample_index}")
    top = st.columns([1, 1, 1, 2])
    with top[0]:
        st.page_link("app.py", label="← Grid (this tab)")
        st.caption("Or close this tab — the grid stays open elsewhere.")
    with top[1]:
        if st.button("Reload frames", use_container_width=True):
            st.session_state.autoplay = False
            if _capture_frames(record_dict):
                st.rerun()
    with top[2]:
        st.session_state.autoplay = st.toggle(
            "Autoplay (GIF)",
            value=bool(st.session_state.autoplay),
        )
    with top[3]:
        st.session_state.autoplay_fps = st.select_slider(
            "Speed (fps)",
            options=[0.5, 1.0, 2.0, 4.0],
            value=float(st.session_state.autoplay_fps),
        )

    if frames_entry is None or frames_entry.n_frames == 0:
        st.info("Capturing predicted-path frames from AirSim…")
        if _capture_frames(record_dict):
            st.rerun()
        st.stop()

    n = frames_entry.n_frames
    step = int(st.session_state.frame_step)
    step = max(0, min(step, n - 1))
    st.session_state.frame_step = step

    left, right = st.columns([1.35, 1.0], gap="large")

    with left:
        st.image(frames_entry.frames_rgb[step], use_container_width=True, caption=f"Frame {step}/{n - 1}")

        nav = st.columns(3)
        with nav[0]:
            if st.button("⟵ Prev", use_container_width=True):
                st.session_state.autoplay = False
                st.session_state.frame_step = (step - 1) % n
                st.rerun()
        with nav[1]:
            if st.session_state.autoplay:
                st.caption(f"Autoplay · step {step}")
            else:
                new_step = st.slider("Step", 0, n - 1, step)
                if new_step != step:
                    st.session_state.frame_step = new_step
                    st.rerun()
        with nav[2]:
            if st.button("Next ⟶", use_container_width=True):
                st.session_state.autoplay = False
                st.session_state.frame_step = (step + 1) % n
                st.rerun()

    with right:
        status = "SUCCESS" if record.success else "FAIL"
        st.markdown(
            f"**{record.environment}** | sample **{record.sample_index}** | "
            f"{status} | NE **{record.final_distance:.1f}m** | "
            f"OSR **{record.osr_hit}** | SPL **{record.spl:.2f}**"
        )

        pred_action_here = _step_action_label(record.predicted_actions, step)
        pose = frames_entry.poses[step] if step < len(frames_entry.poses) else None
        dist = None
        if pose is not None and record.target_xyz:
            dist = calculate_distance(pose[:3], record.target_xyz)
        st.markdown(
            f"**Step {step}/{n - 1}** · action at this view: **{pred_action_here}**"
            + (f" · dist to goal: **{dist:.1f} m**" if dist is not None else "")
        )

        st.subheader("Prompt")
        st.write(record.gpt_instruction)

        st.subheader("Trajectories")
        map_png = render_pseudo_map_from_record(record, current_step=step)
        if map_png is not None:
            st.image(map_png, width=320)
        st.caption("Purple dot = current pose · Blue = GT · Red/Green = Pred · Orange ring = first divergence")
        st.markdown(f"**Pred:** {format_actions(record.predicted_actions, max_items=10_000)}")
        st.markdown(f"**GT:** {format_actions(record.gt_actions, max_items=10_000)}")
        st.caption(f"`{record.image_path}`")

    if st.session_state.autoplay and n > 0:
        time.sleep(1.0 / max(float(st.session_state.autoplay_fps), 0.1))
        st.session_state.frame_step = (step + 1) % n
        st.rerun()


if __name__ == "__main__":
    main()
