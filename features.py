from __future__ import annotations
import math
from typing import Any
import numpy as np

FEATURE_NAMES = [
    "tap_interval_ms",
    "mean_speed_px_s",
    "mean_abs_accel_px_s2",
    "path_deviation_px",
    "pressure_mean",
]

def _mean(xs):
    return float(np.mean(xs)) if len(xs) else 0.0

def _std(xs):
    return float(np.std(xs)) if len(xs) else 0.0

def extract_features(points: list[dict[str, Any]], tap_interval_ms: float | None = None) -> dict:
    """Extract descriptive features from one pointer gesture.

    Each point: x, y, t_ms, optional pressure. Coordinates are CSS pixels;
    t_ms is a monotonic timestamp in milliseconds from the client.
    """
    if len(points) < 2:
        raise ValueError("A gesture needs at least two points.")

    xy = np.asarray([[float(p["x"]), float(p["y"])] for p in points], dtype=float)
    ts = np.asarray([float(p["t_ms"]) for p in points], dtype=float)
    pressure = np.asarray([float(p.get("pressure", 0.0) or 0.0) for p in points], dtype=float)

    dt = np.diff(ts) / 1000.0
    dt = np.maximum(dt, 1e-3)
    delta = np.diff(xy, axis=0)
    step_dist = np.linalg.norm(delta, axis=1)
    speed = step_dist / dt
    accel = np.diff(speed) / np.maximum(dt[1:], 1e-3) if len(speed) > 1 else np.array([])

    start, end = xy[0], xy[-1]
    line = end - start
    line_len = float(np.linalg.norm(line))
    if line_len > 1e-8:
        rel = xy - start
        cross = line[0] * rel[:, 1] - line[1] * rel[:, 0]
        deviations = np.abs(cross) / line_len
    else:
        deviations = np.linalg.norm(xy - start, axis=1)

    return {
        "tap_interval_ms": float(tap_interval_ms or 0.0),
        "mean_speed_px_s": _mean(speed),
        "mean_abs_accel_px_s2": _mean(np.abs(accel)),
        "path_deviation_px": _mean(deviations),
        "pressure_mean": _mean(pressure),
        "speed_std_px_s": _std(speed),
        "path_length_px": float(np.sum(step_dist)),
        "duration_ms": float(ts[-1] - ts[0]),
        "sample_count": int(len(points)),
    }

def normalize_feature_vector(features: dict, means=None, stds=None) -> np.ndarray:
    x = np.asarray([features.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float32)
    if means is None or stds is None:
        # Only a stable numerical scaling for initial model input.
        scales = np.asarray([1000., 2000., 10000., 100., 1.], dtype=np.float32)
        return np.clip(x / scales, 0.0, 1.0)
    return (x - np.asarray(means, dtype=np.float32)) / np.maximum(
        np.asarray(stds, dtype=np.float32), 1e-6
    )
