"""In-memory frame cache for AirSim trajectory previews (no disk writes).

Process-global singleton so Frame Viewer can open in a separate browser tab
(new Streamlit session) and still see frames / trajectory metadata from the grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

CacheKey = Tuple[str, int]  # (environment, sample_index)

_SHARED_CACHE: Optional["FrameCache"] = None
_SHARED_RECORDS: Dict[CacheKey, dict] = {}


@dataclass
class TrajectoryFrames:
    environment: str
    sample_index: int
    image_path: str
    frames_rgb: List[np.ndarray] = field(default_factory=list)
    poses: List[List[float]] = field(default_factory=list)
    predicted_actions: List[int] = field(default_factory=list)
    # Full TrajectoryRecord as dict — needed when viewer opens in a new tab.
    record: Optional[dict] = None

    @property
    def n_frames(self) -> int:
        return len(self.frames_rgb)


class FrameCache:
    """Keep at most one trajectory's frames in memory."""

    def __init__(self) -> None:
        self._entry: Optional[TrajectoryFrames] = None

    def clear(self) -> None:
        self._entry = None

    def get(self, environment: str, sample_index: int) -> Optional[TrajectoryFrames]:
        if self._entry is None:
            return None
        if self._entry.environment == environment and self._entry.sample_index == sample_index:
            return self._entry
        return None

    def put(self, entry: TrajectoryFrames) -> None:
        self._entry = entry

    @property
    def current(self) -> Optional[TrajectoryFrames]:
        return self._entry


def get_shared_frame_cache() -> FrameCache:
    """Return the process-wide cache shared across Streamlit browser tabs."""
    global _SHARED_CACHE
    if _SHARED_CACHE is None:
        _SHARED_CACHE = FrameCache()
    return _SHARED_CACHE


def register_records(record_dicts: List[dict]) -> None:
    """Index trajectory metadata so a new viewer tab can look up by env/sample."""
    for d in record_dicts:
        key = (str(d["environment"]), int(d["sample_index"]))
        _SHARED_RECORDS[key] = d


def lookup_record(environment: str, sample_index: int) -> Optional[dict]:
    return _SHARED_RECORDS.get((environment, int(sample_index)))


def cache_key(environment: str, sample_index: int) -> CacheKey:
    return (environment, int(sample_index))
