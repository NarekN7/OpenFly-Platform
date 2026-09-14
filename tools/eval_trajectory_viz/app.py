"""Streamlit app: grid of GT vs predicted trajectory pseudo maps + AirSim engine panel."""

from __future__ import annotations

import sys
from pathlib import Path

_VIZ_DIR = Path(__file__).resolve().parent
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))

import streamlit as st

from airsim_session import DEFAULT_WAIT_SEC, get_shared_session
from common import (
    dict_to_record,
    format_actions,
    record_key,
    record_to_dict,
    viewer_relative_url,
)
from frame_cache import get_shared_frame_cache, register_records
from loader import TrajectoryRecord, load_eval_run, summarize_run
from physics import (
    DEFAULT_F9_STEP_MULT,
    F9_18M_STEP_MULT,
    f9_distance_m,
    first_divergence_step,
    set_f9_step_mult,
)
from plot import render_pseudo_map_from_record

DEFAULT_GT_X9_JSON = str(Path(__file__).resolve().parents[2] / "data_curated/seenx9.json")
DEFAULT_RUN = (
    Path(__file__).resolve().parents[2]
    / "eval_runs/interleave_2b_lrb10_stop20_reslt_r320x180_evalbest_x9_f918m_20260909_011015"
)


def _gt_has_turn(gt_actions: list[int], action_id: int) -> bool:
    return action_id in gt_actions


def _pred_missed_turn(pred_actions: list[int], action_id: int) -> bool:
    return action_id not in pred_actions


def _left_turn_recall_fail(r: TrajectoryRecord) -> bool:
    return _gt_has_turn(r.gt_actions, 2) and _pred_missed_turn(r.predicted_actions, 2)


def _right_turn_recall_fail(r: TrajectoryRecord) -> bool:
    return _gt_has_turn(r.gt_actions, 3) and _pred_missed_turn(r.predicted_actions, 3)


FILTER_OPTIONS = {
    "All": lambda r: True,
    "Success only": lambda r: r.success == 1,
    "Failure only": lambda r: r.success == 0,
    "OSR hit but failed": lambda r: r.osr_hit == 1 and r.success == 0,
    "Left turn recall fail": _left_turn_recall_fail,
    "Right turn recall fail": _right_turn_recall_fail,
    "Turn recall fail (L or R)": lambda r: _left_turn_recall_fail(r) or _right_turn_recall_fail(r),
}

SORT_OPTIONS = {
    "Final distance (worst first)": lambda r: (-r.final_distance, r.environment, r.sample_index),
    "Sample index": lambda r: (r.environment, r.sample_index),
    "Num steps": lambda r: (-r.num_steps, r.environment, r.sample_index),
}


@st.cache_data(show_spinner=False)
def cached_load_run(run_path_str: str, gt_json_path: str, use_gt_x9: bool) -> tuple[list[dict], dict]:
    records, meta = load_eval_run(Path(run_path_str), gt_json_path=gt_json_path or None, use_gt_x9=use_gt_x9)
    return [record_to_dict(r) for r in records], meta


_MAP_RENDER_VERSION = 4
MAP_DISPLAY_PX = 280


def _infer_f9_mult_from_path(run_path: str) -> int:
    name = Path(run_path).name.lower()
    if "f918m" in name or "f9_18" in name or "18m" in name:
        return F9_18M_STEP_MULT
    return DEFAULT_F9_STEP_MULT


def _inject_grid_css() -> None:
    st.markdown(
        f"""
        <style>
        div[data-testid="stImage"] img {{
            width: {MAP_DISPLAY_PX}px !important;
            height: {MAP_DISPLAY_PX}px !important;
            object-fit: contain !important;
            display: block !important;
            margin: 0 auto !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_render_map(
    record_key_str: str,
    record_dict: dict,
    render_version: int,
    pred_f9_step_mult: int,
) -> bytes | None:
    record = dict_to_record(record_dict)
    return render_pseudo_map_from_record(record, pred_f9_step_mult=pred_f9_step_mult)


def _init_session() -> None:
    # Process-wide singletons so a new browser tab can see frames + the running sim.
    st.session_state.airsim_session = get_shared_session()
    st.session_state.frame_cache = get_shared_frame_cache()
    if "frame_step" not in st.session_state:
        st.session_state.frame_step = 0
    if "autoplay" not in st.session_state:
        st.session_state.autoplay = False


def _engine_panel(envs: list[str]) -> None:
    st.sidebar.header("AirSim engine")
    session = st.session_state.airsim_session
    status = session.status
    if status == "Connected" and session.env_name:
        st.sidebar.success(f"Connected: **{session.env_name}**")
    elif status == "Starting":
        st.sidebar.warning("Starting…")
    elif status == "Error":
        st.sidebar.error(f"Error: {session.error}")
    else:
        st.sidebar.info("Stopped")

    wait_sec = st.sidebar.number_input(
        "Launch wait (sec)",
        min_value=10,
        max_value=120,
        value=DEFAULT_WAIT_SEC,
        step=5,
    )

    cols = st.sidebar.columns(2)
    for i, env in enumerate(envs):
        col = cols[i % 2]
        with col:
            label = env.replace("env_airsim_", "")
            if st.button(f"Start {label}", key=f"start_{env}", use_container_width=True):
                cache = st.session_state.frame_cache
                cache.clear()
                st.session_state.autoplay = False
                with st.spinner(f"Starting {env} (~{wait_sec}s)…"):
                    try:
                        session.start(env, wait_sec=int(wait_sec))
                        st.sidebar.success(f"Connected: {env}")
                    except Exception as e:
                        st.sidebar.error(str(e))
                st.rerun()

    if st.sidebar.button("Stop engine", use_container_width=True):
        session.stop()
        st.session_state.frame_cache.clear()
        st.session_state.autoplay = False
        st.rerun()

    st.sidebar.caption(
        "Open Frames only works for trajectories in the **connected** env. "
        "Opens the simulator viewer in a **new tab** (grid stays here). "
        "Switching envs auto-stops the previous sim and clears cached frames."
    )


def main() -> None:
    st.set_page_config(
        page_title="Eval Trajectory Viz",
        page_icon="🗺️",
        layout="wide",
    )
    _init_session()
    st.title("Eval Trajectory Visualization")
    st.caption("Compare ground-truth vs model trajectories on pseudo maps (20 m success radius).")
    _inject_grid_css()

    with st.sidebar:
        st.header("Settings")
        run_path = st.text_input(
            "Eval run directory",
            value=str(DEFAULT_RUN),
        )
        run_root = Path(run_path)
        if not run_root.is_dir():
            st.error(f"Directory not found: {run_root}")
            st.stop()

        inferred_f9 = _infer_f9_mult_from_path(run_path)
        f9_options = {
            f"27 m (default / eval.py)": DEFAULT_F9_STEP_MULT,
            f"18 m (eval_f9_18m / f918m)": F9_18M_STEP_MULT,
        }
        f9_labels = list(f9_options.keys())
        default_f9_idx = 1 if inferred_f9 == F9_18M_STEP_MULT else 0
        f9_label = st.selectbox(
            "Pred F9 (action 10) distance",
            options=f9_labels,
            index=default_f9_idx,
            help="Must match closed-loop physics. GT maps still use 27 m (seenx9 planning).",
        )
        pred_f9_mult = f9_options[f9_label]
        set_f9_step_mult(pred_f9_mult)
        st.caption(f"Pred F9 step = **{f9_distance_m(pred_f9_mult):.0f} m** (mult={pred_f9_mult})")

        summary = summarize_run(run_root)
        st.metric("Total trajectories", summary["total_samples"])

        use_gt_x9 = st.checkbox("Use tier2/x9 grouped GT (seenx9.json)", value=True)
        gt_json_path = st.text_input(
            "GT x9 JSON path",
            value=DEFAULT_GT_X9_JSON,
            disabled=not use_gt_x9,
        )

        all_envs = sorted({e["name"] for e in summary["envs"]})
        selected_envs = st.multiselect(
            "Environments",
            options=all_envs,
            default=all_envs,
        )
        if not selected_envs:
            st.warning("Select at least one environment.")
            st.stop()

        filter_label = st.selectbox(
            "Filter",
            options=list(FILTER_OPTIONS.keys()),
            index=3,
        )
        sort_label = st.selectbox(
            "Sort by",
            options=list(SORT_OPTIONS.keys()),
            index=0,
        )
        grid_cols = st.slider("Grid columns", min_value=3, max_value=6, value=4)
        page_size = st.selectbox("Page size", options=[12, 24, 48], index=1)

        with st.expander("Run metrics per env"):
            for env_info in summary["envs"]:
                if env_info["name"] not in selected_envs:
                    continue
                m = env_info.get("metrics") or {}
                st.markdown(f"**{env_info['name']}** ({env_info['num_samples']} samples)")
                if m:
                    st.write(
                        f"SR: {100 * m.get('mean_success_rate', 0):.1f}% | "
                        f"OSR: {100 * m.get('mean_oracle_success_rate', 0):.1f}% | "
                        f"NE: {m.get('mean_navigation_error', 0):.1f} m"
                    )

    _engine_panel(all_envs)

    raw_records, load_meta = cached_load_run(str(run_root.resolve()), gt_json_path, use_gt_x9)
    register_records(raw_records)
    if use_gt_x9:
        if load_meta.get("gt_x9_json"):
            st.sidebar.success(
                f"GT overlay: **{load_meta['gt_x9_matched']}** matched, "
                f"**{load_meta['gt_x9_missing']}** missing"
            )
            st.sidebar.caption(f"GT source: `{load_meta['gt_x9_json']}`")
        else:
            st.sidebar.warning("GT x9 JSON not found — using gt_actions from predictions.json")
    logged_eval = load_meta.get("logged_eval_json")
    if logged_eval:
        st.sidebar.caption(f"Closed-loop eval JSON (logged): `{logged_eval}`")

    records = [dict_to_record(d) for d in raw_records]
    records = [r for r in records if r.environment in selected_envs]
    predicate = FILTER_OPTIONS[filter_label]
    records = [r for r in records if predicate(r)]
    records = sorted(records, key=SORT_OPTIONS[sort_label])

    total = len(records)
    if total == 0:
        st.info("No trajectories match the current filters.")
        st.stop()

    st.sidebar.caption(f"**{total}** trajectories match filter: _{filter_label}_")

    num_pages = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=num_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    page_records = records[start:end]

    st.write(f"Showing **{start + 1}–{end}** of **{total}** trajectories (page {page}/{num_pages})")

    rows = [page_records[i : i + grid_cols] for i in range(0, len(page_records), grid_cols)]
    for row in rows:
        cols = st.columns(grid_cols)
        for col, record in zip(cols, row):
            with col:
                _render_cell(record, pred_f9_mult=pred_f9_mult)


def _truncate(text: str, max_len: int = 120) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _render_cell(record: TrajectoryRecord, *, pred_f9_mult: int) -> None:
    d = record_to_dict(record)
    key = record_key(d)

    if not record.start_xyz or not record.target_xyz:
        st.warning(f"{record.environment} #{record.sample_index}: missing pose in console.log")
        return

    png = cached_render_map(key, d, _MAP_RENDER_VERSION, pred_f9_mult)
    if png is None:
        st.warning(f"{record.environment} #{record.sample_index}: could not render map")
        return

    status = "SUCCESS" if record.success else "FAIL"
    status_color = "green" if record.success else "red"
    st.image(png, width=MAP_DISPLAY_PX, output_format="PNG")
    st.markdown(
        f"**{record.environment}** | sample **{record.sample_index}** | "
        f":{status_color}[{status}] | NE **{record.final_distance:.1f}m** | "
        f"OSR **{record.osr_hit}** | SPL **{record.spl:.2f}**"
    )
    st.caption(f"Pred: {format_actions(record.predicted_actions)}")
    st.caption(f"GT:   {format_actions(record.gt_actions)}")
    st.markdown(f"*{_truncate(record.gpt_instruction, 160)}*")

    session = st.session_state.airsim_session
    can_open = (
        session.status == "Connected"
        and session.env_name == record.environment
        and bool(record.start_xyz)
    )
    btn_key = f"open_{record.environment}_{record.sample_index}"
    if can_open:
        st.link_button(
            "Open frames",
            url=viewer_relative_url(record.environment, record.sample_index),
            use_container_width=True,
            help="Opens Frame Viewer in a new tab; this grid stays here.",
        )
    else:
        hint_env = record.environment.replace("env_airsim_", "")
        st.button(
            f"Open frames (start {hint_env})",
            key=btn_key,
            use_container_width=True,
            disabled=True,
        )

    div = first_divergence_step(record.gt_actions, record.predicted_actions)
    with st.expander("Details"):
        st.write(f"**image_path:** `{record.image_path}`")
        st.write(f"**Steps:** {record.num_steps} | **Stopped by model:** {record.stopped_by_model}")
        if div is not None:
            st.write(f"**First divergence at step:** {div}")
        else:
            st.write("**First divergence at step:** none (identical action prefix)")
        st.write("**Full instruction:**")
        st.write(record.gpt_instruction)
        st.write("**Predicted actions:**", format_actions(record.predicted_actions, max_items=10_000))
        st.write("**GT actions:**", format_actions(record.gt_actions, max_items=10_000))


if __name__ == "__main__":
    main()
