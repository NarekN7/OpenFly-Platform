# Eval Trajectory Visualization

Streamlit app for comparing ground-truth vs model trajectories from OpenFly eval runs, plus live AirSim frame viewing along the predicted path.

## Setup

```bash
pip install -r tools/eval_trajectory_viz/requirements.txt
# AirSim Python client must also be available (same env as train/eval.py):
#   airsim, msgpack-rpc-python, msgpack<1, opencv-python
```

## Run

From the repository root:

```bash
streamlit run tools/eval_trajectory_viz/app.py
```

Change the eval run directory in the sidebar as needed.

## Pseudo-map grid

Each environment folder under an eval run should contain:

- `predictions.json` — actions, metrics, instructions
- `console.log` — start/target poses and initial yaw (joined by `sample_index`)

Optional: overlay GT from `data_curated/seenx9.json` (tier2/x9) via the sidebar checkbox.

### Legend

- Green triangle: start
- Red star: target
- Blue line: GT trajectory
- Green/red line: predicted trajectory (success/failure)
- Orange ring: first action divergence
- Dashed red circle: 20 m success zone
- Purple dot (frame viewer): current pose

## AirSim frame viewer

1. In the sidebar **AirSim engine** panel, click **Start 16** (or 18/23/…) — waits ~40s for the sim.
2. Filter the grid as usual; **Open frames** is enabled only for trajectories in the connected env.
3. On open, the app teleports along **predicted** poses, captures RGB into **session memory only** (no disk writes), then opens the Frame Viewer page.
4. Use Prev/Next/slider, or **Autoplay (GIF)** to loop frames while reading the prompt and watching the map.

Switching envs auto-stops the previous engine and clears cached frames.
