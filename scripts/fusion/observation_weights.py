"""Prediction-only observation factors; input alpha is an annotated mask.

These are conservative hypotheses, not calibrated probabilities. Keep a floor
so a silhouette or a grazing ray cannot alone delete a thin structure.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt


def observation_factor(depth, k, mask, mode, boundary_pixels=3.0, floor=0.25):
    valid = mask & np.isfinite(depth) & (depth > 0)
    factor = np.ones_like(depth, dtype=np.float32)
    if mode in ("boundary", "combined"):
        # Padding treats image borders as boundaries too.
        distance = distance_transform_edt(np.pad(valid, 1))[1:-1, 1:-1]
        factor *= floor + (1 - floor) * np.clip(distance / boundary_pixels, 0, 1)
    if mode in ("incidence", "combined"):
        yy, xx = np.indices(depth.shape)
        ray = np.stack([(xx-k[0, 2])/k[0, 0], (yy-k[1, 2])/k[1, 1],
                        np.ones_like(xx)], axis=-1)
        xyz = depth[..., None] * ray
        dy, dx = np.gradient(xyz, axis=(0, 1))
        normal = np.cross(dx, dy)
        cosine = np.abs((normal * ray).sum(-1)) / np.maximum(
            np.linalg.norm(normal, axis=-1) * np.linalg.norm(ray, axis=-1), 1e-12)
        # Boundary derivatives mix foreground and background: leave them alone.
        interior = np.zeros_like(valid)
        interior[1:-1, 1:-1] = (valid[1:-1, 1:-1] & valid[:-2, 1:-1]
                                & valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:])
        factor *= np.where(interior, floor + (1-floor)*cosine**2, 1)
    if mode not in ("confidence", "incidence", "boundary", "combined"):
        raise ValueError(mode)
    return np.where(valid, factor, 0).astype(np.float32)
