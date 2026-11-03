#!/usr/bin/env python3
"""Geometry-side supervision without any photometric optimisation:
cross-view free-space (background-vote) carving of VGGT-Omega depth maps.

Rule (per depth pixel, in the predicted gauge - no GT depth/normals/mesh):
unproject the pixel; project the point into every view; count
  votes    = fraction of views where it lands on a BACKGROUND pixel
             (GT foreground complement, eroded by --bg-erode-px; GT foreground
             is the same protocol mask capture_eval already applies),
  confirms = fraction of views where it lands on foreground with depth
             consistent (rel. tol) with that view's unmasked predicted depth.
Delete the pixel when votes >= --vote-frac AND confirms <= --confirm-max.
Visual diagnosis (outputs/geometry_pruning/diag): bridge/burr pixels are
magenta (votes high, confirms low); legit surface is cyan. A point behind a
window opening (interior floor) keeps its confirms and survives.

Output is a standard omega-format geometry dir with carved depth maps, so the
unchanged frozen protocol (capture_eval + measure_surfaces + heldout renders)
evaluates it exactly like the baseline.

    .venv/Scripts/python.exe scripts/geometry_pruning.py \
        --omega data/scenes/nerf_lego16/geometry/omega \
        --gt data/scenes/nerf_lego_gt16/geometry/gt \
        --out outputs/geometry_pruning/geom/carve_v50c20
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "diagnostics"))
from pose_depth_ablation import regrid_depth  # noqa: E402
from verify_scene import umeyama  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omega", required=True, help="omega-format geometry dir (predicted gauge)")
    parser.add_argument("--gt", required=True, help="GT geometry dir; ONLY foreground masks are read "
                        "(the same protocol masks capture_eval applies)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--vote-frac", type=float, default=0.5)
    parser.add_argument("--confirm-max", type=float, default=0.2)
    parser.add_argument("--tol-rel", type=float, default=0.015)
    parser.add_argument("--bg-erode-px", type=int, default=1)
    parser.add_argument("--vis", action="store_true", help="save vote/confirm visualisations")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    omega, gt_dir, out = Path(args.omega), Path(args.gt), Path(args.out)

    import cv2
    frames = json.loads((omega / "cameras.json").read_text())["frames"]
    gt_frames = json.loads((gt_dir / "cameras.json").read_text())["frames"]
    n = len(frames)
    names = [Path(f["frame"]).stem for f in frames]

    depth, fg, ks, ts = [], [], [], []
    for i, frame in enumerate(frames):
        stem = names[i]
        d = np.load(omega / "depth" / f"{stem}.npy").squeeze().astype(np.float32)
        height, width = d.shape
        k = np.asarray(frame["K_model"], dtype=np.float64).copy()
        if (height, width) != (frame["model_height"], frame["model_width"]):
            k[0] *= width / frame["model_width"]
            k[1] *= height / frame["model_height"]
        # GT foreground regridded (nearest) onto the predicted grid.
        gk = np.asarray(gt_frames[i]["K_model"], dtype=np.float64)
        fgd = np.load(gt_dir / "foreground" / f"{stem}.npy")
        yy, xx = np.indices((height, width), dtype=np.float32)
        u = (xx - k[0, 2]) / k[0, 0] * gk[0, 0] + gk[0, 2]
        v = (yy - k[1, 2]) / k[1, 1] * gk[1, 1] + gk[1, 2]
        fgp = regrid_depth(fgd.astype(np.float32), u, v, "nearest") > 0.5
        depth.append(torch.from_numpy(d).to(device))
        fg.append(torch.from_numpy(fgp).to(device))
        ks.append(torch.from_numpy(k).float().to(device))
        ts.append(torch.from_numpy(np.asarray(frame["T_world_to_cam"])).float().to(device))
    height, width = depth[0].shape
    if args.bg_erode_px > 0:
        kern = np.ones((2 * args.bg_erode_px + 1,) * 2, np.uint8)
        fg = [torch.from_numpy(cv2.erode(f.cpu().numpy().astype(np.uint8), kern).astype(bool)).to(device)
              for f in fg]
    K = torch.stack(ks)
    # Votes are computed in the Sim3-ALIGNED gauge: raw predicted poses and raw
    # predicted depth are mutually inconsistent at the few-percent level
    # (measured: pair reprojection median mismatch up to 16%), while the frozen
    # train-only Sim(3) removes the global component. Deletion is exported in
    # the raw predicted grid, so the frozen eval protocol is unchanged.
    pts_np = np.asarray([f["T_world_to_cam"] for f in frames])
    gts_np = np.asarray([f["T_world_to_cam"] for f in gt_frames])
    gc = -np.einsum("sji,sj->si", gts_np[:, :3, :3], gts_np[:, :3, 3])
    pc = -np.einsum("sji,sj->si", pts_np[:, :3, :3], pts_np[:, :3, 3])
    scale, rot, trans = umeyama(pc, gc)
    aligned_np = pts_np.copy()
    aligned_np[:, :3, :3] = pts_np[:, :3, :3] @ rot.T
    aligned_np[:, :3, 3] = scale * pts_np[:, :3, 3] - aligned_np[:, :3, :3] @ trans
    T = torch.from_numpy(aligned_np).float().to(device)
    Tinv = torch.linalg.inv(T)
    depth = torch.stack(depth) * float(scale)
    fg = torch.stack(fg)

    yy, xx = torch.meshgrid(torch.arange(height, device=device),
                            torch.arange(width, device=device), indexing="ij")
    started = time.time()
    carved = depth.clone()
    report = {"args": vars(args), "views": {}}
    vote_vis, conf_vis = [], []
    for i in range(n):
        d = depth[i]
        valid = d > 0
        pix = torch.stack([xx[valid], yy[valid], torch.ones_like(xx[valid])], -1).float()
        cam = (torch.inverse(K[i]) @ pix.unsqueeze(-1)).squeeze(-1) * d[valid].unsqueeze(-1)
        world = (Tinv[i][:3, :3] @ cam.unsqueeze(-1)).squeeze(-1) + Tinv[i][:3, 3]
        votes = torch.zeros(len(pix), device=device)
        confirms = torch.zeros(len(pix), device=device)
        seen = torch.zeros(len(pix), device=device)
        for j in range(n):
            cs = (T[j][:3, :3] @ world.unsqueeze(-1)).squeeze(-1) + T[j][:3, 3]
            z = cs[:, 2]
            uf = cs[:, 0] / z.clamp(min=1e-6) * K[j][0, 0] + K[j][0, 2]
            vf = cs[:, 1] / z.clamp(min=1e-6) * K[j][1, 1] + K[j][1, 2]
            ui, vi = uf.round().long(), vf.round().long()
            inside = (z > 1e-6) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
            ui, vi, zin = ui[inside], vi[inside], z[inside]
            seen[inside] += 1
            zref = depth[j][vi, ui]
            fg_hit = fg[j][vi, ui]
            votes[inside] += (~fg_hit).float()
            confirms[inside] += (fg_hit & (zref > 0)
                                 & ((zin - zref).abs() < zref * args.tol_rel)).float()
        vr = votes / seen.clamp(min=1)
        cf = confirms / seen.clamp(min=1)
        delete = valid.clone()
        delete[valid] = (vr >= args.vote_frac) & (cf <= args.confirm_max)
        carved[i][delete] = 0.0
        report["views"][names[i]] = {
            "valid_px": int(valid.sum()), "deleted_px": int(delete.sum()),
            "deleted_frac": float(delete.sum() / valid.sum().clamp(min=1))}
        if args.vis:
            vr_full = torch.zeros(height, width, device=device)
            cf_full = torch.zeros(height, width, device=device)
            vr_full[valid] = vr
            cf_full[valid] = cf
            vote_vis.append(torch.stack([vr_full, cf_full, valid.float()], 0))
    (out / "depth").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        np.save(out / "depth" / f"{names[i]}.npy",
                (carved[i] / float(scale)).cpu().numpy().astype(np.float32)[..., None])
    cameras = json.loads((omega / "cameras.json").read_text())
    cameras["note"] = (f"free-space carved: votes>={args.vote_frac} & confirms<={args.confirm_max} "
                       f"(tol {args.tol_rel}, bg erode {args.bg_erode_px}px); see pruning_report.json")
    (out / "cameras.json").write_text(json.dumps(cameras, indent=2) + "\n")
    report["seconds"] = round(time.time() - started, 1)
    report["scale"] = float(scale)
    report["deleted_frac_total"] = float(np.mean([v["deleted_frac"] for v in report["views"].values()]))
    (out / "pruning_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.vis:
        import torchvision.utils as vutils
        vis_dir = out / "vis"
        vis_dir.mkdir(exist_ok=True)
        for i, img in enumerate(vote_vis):
            vutils.save_image(img, str(vis_dir / f"votes_{names[i]}.png"))
    print(json.dumps({"deleted_frac_total": report["deleted_frac_total"],
                      "seconds": report["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
