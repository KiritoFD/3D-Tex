#!/usr/bin/env python3
"""Sparse gated fusion: replace VGGT-Omega depth with COLMAP PatchMatch depth
only where COLMAP's geometric-consistency gate commits; keep VGGT elsewhere.

The gate IS the colmap geometric depth map (valid pixels passed COLMAP's
geom_consistency filter at a given threshold level). COLMAP depth (800^2,
source gauge, aligned poses from export_aligned_poses.py) is unprojected to
world points, mapped back into the predicted gauge with the frozen train-only
Sim(3) (identical computation to capture_eval.py), and z-buffer-splatted onto
the predicted 512^2 grid. Fusion:

    D_fused = D_vggt * (1 - w) + C * w,   w = 1 inside the gate,
    w = max(0, 1 - dist/blend_px) within blend_px of a gated pixel
    (nearest-valid extrapolation into the feather band), 0 elsewhere.

Output is a standard "omega" geometry dir (cameras.json copied unchanged +
depth/*.npy), so capture_eval.py evaluates it through the identical frozen
protocol as the VGGT baseline. GT depth/normals/mesh are never read here.

    .venv/Scripts/python.exe scripts/gated_fusion.py \
        --omega data/scenes/nerf_lego16/geometry/omega \
        --colmap outputs/colmap/sweep/anchor16/cost4_nc2 --colmap-format bin \
        --poses outputs/e2_anchor16_poses.json \
        --gt data/scenes/nerf_lego_gt16/geometry/gt \
        --out outputs/gated_fusion/geom/cost4_nc2 --blend-px 2
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_scene import umeyama  # noqa: E402


def read_bin(path):
    """COLMAP .bin map: '<W>&<H>&<C>&' header then little-endian float32."""
    with open(path, "rb") as handle:
        header = handle.read(32)
        m = re.match(rb"(\d+)&(\d+)&(\d+)&", header)
        width, height, channels = int(m.group(1)), int(m.group(2)), int(m.group(3))
        handle.seek(m.end())
        data = np.frombuffer(handle.read(), dtype=np.float32)
    return data.reshape(height, width, channels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omega", required=True, help="omega-format geometry dir (predicted gauge)")
    parser.add_argument("--colmap", required=True, help="dir with per-stem colmap depth maps")
    parser.add_argument("--colmap-format", choices=["bin", "npy"], default="bin")
    parser.add_argument("--poses", required=True, help="aligned-poses json (source gauge, 800^2)")
    parser.add_argument("--gt", required=True, help="GT geometry dir; ONLY camera poses are read "
                        "(train-only Sim3, identical to capture_eval)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--blend-px", type=float, default=0.0,
                        help="feather band width (pred-grid px); 0 = hard replace inside gate")
    parser.add_argument("--min-cluster", type=int, default=0,
                        help="drop gate clusters smaller than this many pred-grid pixels "
                             "(isolated splat points act as spikes in the normal metric)")
    args = parser.parse_args()
    omega, out = Path(args.omega), Path(args.out)
    colmap_dir = Path(args.colmap)

    cameras = json.loads((omega / "cameras.json").read_text())
    frames = cameras["frames"]
    gt_frames = json.loads((Path(args.gt) / "cameras.json").read_text())["frames"]
    aligned = {f["stem"]: f for f in json.loads(Path(args.poses).read_text())["frames"]}

    # Frozen train-only Sim(3) on camera centres (same as capture_eval.py).
    pts = np.asarray([f["T_world_to_cam"] for f in frames])
    gts = np.asarray([f["T_world_to_cam"] for f in gt_frames])
    gc = -np.einsum("sji,sj->si", gts[:, :3, :3], gts[:, :3, 3])
    pc = -np.einsum("sji,sj->si", pts[:, :3, :3], pts[:, :3, 3])
    scale, rot, trans = umeyama(pc, gc)  # source ~= scale*R*pred + t

    (out / "depth").mkdir(parents=True, exist_ok=True)
    stats = {"scale": scale, "blend_px": args.blend_px, "levels": {}}
    gate_counts, total_counts = 0, 0
    for frame in frames:
        stem = Path(frame["frame"]).stem
        d0 = np.load(omega / "depth" / f"{stem}.npy").squeeze().astype(np.float64)
        height, width = d0.shape
        k = np.asarray(frame["K_model"], dtype=np.float64)
        if (height, width) != (frame["model_height"], frame["model_width"]):
            k[0] *= width / frame["model_width"]
            k[1] *= height / frame["model_height"]
        t_pred = np.asarray(frame["T_world_to_cam"], dtype=np.float64)
        # COLMAP depth at 800^2 in the source gauge.
        if args.colmap_format == "bin":
            c800 = read_bin(colmap_dir / f"{stem}.png.geometric.bin")[..., 0]
        else:
            c800 = np.load(colmap_dir / f"{stem}.npy").squeeze().astype(np.float64)
        valid = np.isfinite(c800) & (c800 > 0)
        rows, cols = np.nonzero(valid)
        fused = d0.copy()
        gate_frac = 0.0
        if len(rows):
            pa = aligned[stem]
            ka = np.asarray(pa["K_original"], dtype=np.float64)
            ta = np.asarray(pa["T_world_to_cam"], dtype=np.float64)
            z = c800[rows, cols]
            cam_a = np.stack([(cols - ka[0, 2]) / ka[0, 0] * z,
                              (rows - ka[1, 2]) / ka[1, 1] * z, z], axis=-1)
            world_src = (ta[:3, :3].T @ (cam_a - ta[:3, 3]).T).T
            # Source gauge -> predicted gauge: pred = R * (src - t) / scale.
            world_pred = (world_src - trans) @ rot / scale
            cam_p = (t_pred[:3, :3] @ world_pred.T).T + t_pred[:3, 3]
            zp = cam_p[:, 2]
            ok = zp > 1e-6
            uf = cam_p[ok, 0] / zp[ok] * k[0, 0] + k[0, 2]
            vf = cam_p[ok, 1] / zp[ok] * k[1, 1] + k[1, 2]
            ui = np.rint(uf).astype(int)
            vi = np.rint(vf).astype(int)
            keep = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
            ui, vi, zp = ui[keep], vi[keep], zp[ok][keep]
            gate = np.full((height, width), np.inf)
            np.minimum.at(gate, (vi, ui), zp)
            gated = np.isfinite(gate)
            if args.min_cluster > 0:
                from scipy.ndimage import label
                clusters, n_clusters = label(gated)
                sizes = np.bincount(clusters.ravel())
                small = np.isin(clusters, np.nonzero(sizes < args.min_cluster)[0])
                gate[small & (clusters > 0)] = np.inf
                gated = np.isfinite(gate)
            gate_frac = float(gated.mean())
            if args.blend_px > 0:
                # Feather: nearest gated value within blend_px, linear ramp.
                dist, nearest = distance_transform_edt(~gated, return_indices=True)
                filled = gate[nearest[0], nearest[1]]
                w = np.clip(1.0 - dist / args.blend_px, 0.0, 1.0)
                w[~np.isfinite(filled)] = 0.0
                w[gated] = 1.0
            else:
                filled = gate
                w = gated.astype(np.float64)
            apply = (w > 0) & np.isfinite(filled) & (d0 > 0)
            fused[apply] = d0[apply] * (1 - w[apply]) + filled[apply] * w[apply]
        gate_counts += int((w > 0).sum()) if len(rows) else 0
        total_counts += height * width
        stats["levels"][stem] = {"colmap_valid_px": int(len(rows)), "gate_frac": gate_frac}
        np.save(out / "depth" / f"{stem}.npy", fused.astype(np.float32)[..., None])

    cameras["note"] = (f"sparse gated fusion: VGGT depth replaced by COLMAP depth on "
                       f"{gate_counts / max(total_counts, 1):.4f} of pixels "
                       f"(blend_px={args.blend_px}); colmap={args.colmap}")
    (out / "cameras.json").write_text(json.dumps(cameras, indent=2) + "\n")
    stats["gate_fraction_total"] = gate_counts / max(total_counts, 1)
    (out / "fusion_report.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({"gate_fraction_total": stats["gate_fraction_total"],
                      "scale": scale}, indent=2))


if __name__ == "__main__":
    main()
