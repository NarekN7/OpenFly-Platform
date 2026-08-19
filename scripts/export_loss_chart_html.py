#!/usr/bin/env python3
"""Export train + validation loss as one self-contained HTML/SVG chart (no CDN)."""
from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

Point = Tuple[int, float]


def _load_series(ea: EventAccumulator, primary: str, fallback: str) -> List[Point]:
    tags = set(ea.Tags().get("scalars", []))
    tag = primary if primary in tags else (fallback if fallback in tags else None)
    if tag is None:
        return []
    return [(int(e.step), float(e.value)) for e in ea.Scalars(tag)]


def _svg_polyline(points: Sequence[Point], width: int, height: int, pad: int, ymin: float, ymax: float) -> str:
    if len(points) < 2:
        return ""
    xmin, xmax = points[0][0], points[-1][0]
    if xmax == xmin:
        xmax = xmin + 1

    def xy(step: int, val: float) -> Tuple[float, float]:
        x = pad + (step - xmin) / (xmax - xmin) * (width - 2 * pad)
        y = pad + (1.0 - (val - ymin) / (ymax - ymin)) * (height - 2 * pad)
        return x, y

    coords = " ".join(f"{xy(s, v)[0]:.1f},{xy(s, v)[1]:.1f}" for s, v in points)
    return coords


def _svg_circles(points: Sequence[Point], width: int, height: int, pad: int, ymin: float, ymax: float) -> str:
    if not points:
        return ""
    xmin, xmax = points[0][0], points[-1][0]
    if xmax == xmin:
        xmax = xmin + 1

    def xy(step: int, val: float) -> Tuple[float, float]:
        x = pad + (step - xmin) / (xmax - xmin) * (width - 2 * pad)
        y = pad + (1.0 - (val - ymin) / (ymax - ymin)) * (height - 2 * pad)
        return x, y

    return "\n".join(
        f'  <circle cx="{xy(s, v)[0]:.1f}" cy="{xy(s, v)[1]:.1f}" r="4" fill="#f59e0b" stroke="#111" stroke-width="1"/>'
        for s, v in points
    )


def _y_limits(train: Sequence[Point], validation: Sequence[Point]) -> Tuple[float, float]:
    """Readable y-range: ignore rare train spikes so val is visible with train."""
    vals = [v for _, v in train] + [v for _, v in validation]
    if not vals:
        return 0.0, 1.0
    sorted_vals = sorted(vals)
    p99 = sorted_vals[int(0.99 * (len(sorted_vals) - 1))]
    ymax = min(1.2, max(p99 * 1.08, max(vals[-50:] if len(vals) >= 50 else vals) * 1.05))
    ymin = 0.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    return ymin, ymax


def write_loss_chart_html(
    train: Sequence[Point],
    validation: Sequence[Point],
    out: Path,
    *,
    title: str,
    png_path: Optional[Path] = None,
) -> Path:
    ymin, ymax = _y_limits(train, validation)

    w, h, pad = 1000, 420, 48
    train_line = _svg_polyline(train, w, h, pad, ymin, ymax)
    val_line = _svg_polyline(validation, w, h, pad, ymin, ymax) if len(validation) >= 2 else ""
    val_dots = _svg_circles(validation, w, h, pad, ymin, ymax)

    t = html.escape(title)
    last_train = train[-1][1] if train else float("nan")
    last_val = validation[-1][1] if validation else float("nan")

    png_img = ""
    if png_path is not None and png_path.is_file():
        b64 = base64.standard_b64encode(png_path.read_bytes()).decode("ascii")
        png_img = (
            '\n  <p class="meta">PNG fallback (use if SVG preview is blank):</p>\n'
            f'  <img alt="loss chart" src="data:image/png;base64,{b64}" '
            'style="max-width:100%;border:1px solid #ddd;border-radius:8px"/>'
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{t}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 20px; background: #fafafa; color: #111; }}
    h1 {{ font-size: 1rem; margin: 0 0 4px; }}
    .meta {{ color: #555; font-size: 0.85rem; margin-bottom: 12px; }}
    .legend {{ display: flex; gap: 20px; margin-bottom: 8px; font-size: 0.9rem; }}
    .legend span::before {{ content: ""; display: inline-block; width: 24px; height: 3px; margin-right: 8px; vertical-align: middle; }}
    .train::before {{ background: #2563eb; }}
    .val::before {{ background: #f59e0b; }}
    svg {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>{t}</h1>
  <p class="meta">Final train loss: {last_train:.4f} &nbsp;|&nbsp; Final val loss: {last_val:.4f} &nbsp;|&nbsp; Y-axis capped for readability (early train spikes hidden)</p>
  <div class="legend">
    <span class="train">train (every 10 steps)</span>
    <span class="val">validation (every 500 steps)</span>
  </div>
  <svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">
    <rect x="0" y="0" width="{w}" height="{h}" fill="#fff"/>
    <text x="{pad}" y="24" fill="#666" font-size="12">CE loss (y: {ymin:.2f}–{ymax:.2f})</text>
    <text x="{w - pad}" y="{h - 12}" fill="#666" font-size="12" text-anchor="end">step</text>
    <polyline fill="none" stroke="#2563eb" stroke-width="2" points="{train_line}"/>
    {f'<polyline fill="none" stroke="#f59e0b" stroke-width="2" stroke-dasharray="6 4" points="{val_line}"/>' if val_line else ''}
{val_dots}
  </svg>
{png_img}
</body>
</html>
"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--title", type=str, default="loss")
    args = parser.parse_args()

    ea = EventAccumulator(args.logdir)
    ea.Reload()
    train = _load_series(ea, "loss/train", "train/loss")
    val = _load_series(ea, "loss/validation", "eval/loss")
    if not train and not val:
        raise SystemExit(f"No loss scalars in {args.logdir}")

    path = write_loss_chart_html(train, val, Path(args.out), title=args.title)
    print(path.resolve())


if __name__ == "__main__":
    main()
