#!/usr/bin/env python3
"""Mixed-trajectory fusion: merge full-frame views (anchor16, optionally
free-space carved) with mixed-run close-up crops into one omega-format
48-frame geometry dir for the frozen eval protocol.

Close-up alignment rule (per crop view, no GT in the optimisation):
  pose      := the matching full view's pose (crop = same physical camera,
               zoomed in the image plane; the crop run's own pose deviates
               ~0.15 median from it, Follow-up D);
  intrinsics:= A_crop @ (full K_original shifted by the crop box);
  depth     := mixed-run crop depth x robust scalar (median full/crop ratio
               in the overlap region). Residual mismatch after the scalar is
               reported; if it stayed structured, a small 6-DOF refinement
               would be needed (reported instead of silently fused).

Fusion weights / conflict adjudication (no simple averaging):
  overlapping pixels are consistent when |crop_full_depth diff| <= tol_rel;
  consistent crop pixels get weight w_crop (structure scale is more trusted);
  conflicting pixels are adjudicated by confidence - crop wins when its
  confidence >= the full confidence there, otherwise the crop pixel is
  dropped (the full view keeps its own pixel).

    .venv/Scripts/python.exe scripts/mixed_fusion.py \
        --full-geom outputs/geometry_pruning/geom/carve_v35c30 \
        --mixed-geom outputs/tiling/crop_mixed/geometry/omega \
        --crop-boxes outputs/tiling/crop_roofwin/crop_boxes.json \
        --crop-boxes outputs/tiling/crop_wheels/crop_boxes.json \
        --out outputs/mixed_fusion/geom --w-crop 4.0 --tol-rel 0.02
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "diagnostics"))
from pose_depth_ablation import regrid_depth  # noqa: E402


def shifted_k(k, dx, dy):
    k = np.asarray(k, dtype=np.float64).copy()
    k[0, 2] -= dx
    k[1, 2] -= dy
    return k


def _rodrigues(rho):
    import torch
    theta = rho.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    k = rho / theta
    zeros = torch.zeros_like(k[:, 0])
    kx = torch.stack([zeros, -k[:, 2], k[:, 1],
                      k[:, 2], zeros, -k[:, 0],
                      -k[:, 1], k[:, 0], zeros], dim=-1).reshape(-1, 3, 3)
    eye = torch.eye(3, device=rho.device).expand_as(kx)
    return eye + torch.sin(theta)[..., None] * kx + (1 - torch.cos(theta))[..., None] * (kx @ kx)


def refine_pose(k_model_np, full_pack, t_full_np, depth, conf, fd_r, scalar_init,
                iters, device):
    """Refine (rho_depth, rotvec, tau) so crop depth registers onto ALL full views.

    Multi-view consistency: sample crop pixels, project into every full view and
    minimise the robust relative depth mismatch against that view's depth
    (single-view matching only constrains the direction perpendicular to its
    ray; measured: residual ~1.1% = several voxels without this).
    """
    import torch
    import torch.nn.functional as F
    k_c = torch.from_numpy(k_model_np).float().to(device)
    t_f = torch.from_numpy(t_full_np).float().to(device)
    t_f_inv = torch.linalg.inv(t_f)
    d = torch.from_numpy(depth).float().to(device)
    h, w = d.shape
    k_all = torch.stack([torch.from_numpy(p[0]).float() for p in full_pack]).to(device)
    t_all = torch.stack([torch.from_numpy(p[1]).float() for p in full_pack]).to(device)
    d_all = torch.stack([torch.from_numpy(p[2]).float() for p in full_pack]).to(device)
    n_full = len(full_pack)
    fd = torch.from_numpy(fd_r).float().to(device)
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    valid0 = (d > 0) & (fd > 0)
    rows, cols = torch.nonzero(valid0, as_tuple=True)
    gen = torch.Generator(device="cpu").manual_seed(0)
    sel = torch.randperm(len(rows), generator=gen)[: min(8000, len(rows))].to(device)
    r, c = rows[sel], cols[sel]
    pix = torch.stack([c.float(), r.float(), torch.ones_like(r, dtype=torch.float32)], -1)
    rho_depth = torch.tensor([float(np.log(max(scalar_init, 1e-6)))], device=device, requires_grad=True)
    rot = torch.zeros(3, device=device, requires_grad=True)
    tau = torch.zeros(3, device=device, requires_grad=True)
    opt = torch.optim.Adam([rho_depth, rot, tau], lr=2e-3)
    d_pts = d[r, c]

    def forward():
        cam = (torch.inverse(k_c) @ pix.unsqueeze(-1)).squeeze(-1) * (d_pts * rho_depth.exp()).unsqueeze(-1)
        world = (t_f_inv[:3, :3] @ cam.unsqueeze(-1)).squeeze(-1) + t_f_inv[:3, 3]
        world = (_rodrigues(rot[None])[0] @ world.unsqueeze(-1)).squeeze(-1) + tau
        return world

    def multi_loss(world):
        total = torch.zeros((), device=device)
        count = torch.zeros((), device=device)
        for j in range(n_full):
            cs = (t_all[j][:3, :3] @ world.unsqueeze(-1)).squeeze(-1) + t_all[j][:3, 3]
            z = cs[:, 2].clamp(min=1e-6)
            uf = cs[:, 0] / z * k_all[j][0, 0] + k_all[j][0, 2]
            vf = cs[:, 1] / z * k_all[j][1, 1] + k_all[j][1, 2]
            hh, ww = d_all[j].shape
            ui, vi = uf.round().long(), vf.round().long()
            inside = (z > 1e-6) & (ui >= 0) & (ui < ww) & (vi >= 0) & (vi < hh)
            ui_c = ui.clamp(0, ww - 1)
            vi_c = vi.clamp(0, hh - 1)
            zref_full = d_all[j][vi_c, ui_c]
            relv = (z - zref_full) / zref_full.clamp(min=1e-6)
            keep = inside & (zref_full > 0) & (relv.abs() < 0.10)
            total = total + F.huber_loss(relv[keep], torch.zeros_like(relv[keep]), delta=0.01) * keep.sum()
            count = count + keep.sum()
        return total / count.clamp(min=1)

    for _ in range(iters):
        world = forward()
        loss = multi_loss(world) \
            + 10.0 * (rot ** 2).mean() / 0.02 ** 2 + 10.0 * (tau ** 2).mean() / 0.02 ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        world = forward()
        rels = []
        for j in range(n_full):
            cs = (t_all[j][:3, :3] @ world.unsqueeze(-1)).squeeze(-1) + t_all[j][:3, 3]
            z = cs[:, 2].clamp(min=1e-6)
            uf = cs[:, 0] / z * k_all[j][0, 0] + k_all[j][0, 2]
            vf = cs[:, 1] / z * k_all[j][1, 1] + k_all[j][1, 2]
            hh, ww = d_all[j].shape
            ui, vi = uf.round().long(), vf.round().long()
            inside = (z > 1e-6) & (ui >= 0) & (ui < ww) & (vi >= 0) & (vi < hh)
            ui_c = ui.clamp(0, ww - 1)
            vi_c = vi.clamp(0, hh - 1)
            zref_full = d_all[j][vi_c, ui_c]
            relv = ((z - zref_full) / zref_full.clamp(min=1e-6)).abs()
            keep = inside & (zref_full > 0)
            rels.append(relv[keep])
        rel = torch.cat(rels)
        rot_np = _rodrigues(rot[None])[0].cpu().numpy()
        return (float(rho_depth.exp()), rot_np, tau.cpu().numpy(),
                float(rel.median()), float(rel.quantile(0.9)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-geom", required=True, help="omega dir with full views (e.g. carved)")
    parser.add_argument("--mixed-geom", required=True, help="mixed run omega dir (full + crops)")
    parser.add_argument("--crop-boxes", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--w-crop", type=float, default=4.0)
    parser.add_argument("--tol-rel", type=float, default=0.02)
    parser.add_argument("--conflict-mode", choices=["coexist", "replace", "adjudicate"],
                        default="adjudicate",
                        help="coexist: crop wins by weight only (double-sheets, measured worse); "
                             "replace: zero the full view's whole crop rectangle, crop pixels rule "
                             "the region; adjudicate: per-pixel mutual exclusion by confidence "
                             "(crop wins -> kill full pixel, full wins -> drop crop pixel)")
    parser.add_argument("--refine-pose", action="store_true",
                        help="small per-crop-view refinement (depth scale + rotvec + translation, "
                             "init at the full pose) minimising robust depth mismatch against the "
                             "full view in the overlap region; needed because a scalar alone leaves "
                             "structured residuals (measured: median 8.4%)")
    parser.add_argument("--refine-iters", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    full_dir, mixed_dir, out = Path(args.full_geom), Path(args.mixed_geom), Path(args.out)

    full_cameras = json.loads((full_dir / "cameras.json").read_text())
    full_frames = full_cameras["frames"]
    mixed_frames = {f["frame"]: f for f in json.loads((mixed_dir / "cameras.json").read_text())["frames"]}
    # pack all full views for the multi-view pose refinement
    full_pack = [(np.asarray(f["K_model"], dtype=np.float64),
                  np.asarray(f["T_world_to_cam"], dtype=np.float64),
                  np.load(full_dir / "depth" / f"{Path(f['frame']).stem}.npy").squeeze().astype(np.float64))
                 for f in full_frames]
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "confidence").mkdir(parents=True, exist_ok=True)

    out_frames = []
    report = {"args": vars(args), "crops": {}}
    # 1. pass through full views (already carved; conf plain); conflict modes
    #    may later zero parts of these maps via the kill masks below.
    full_depth_out = {}
    full_conf_out = {}
    for f in full_frames:
        stem = Path(f["frame"]).stem
        full_depth_out[stem] = np.load(full_dir / "depth" / f"{stem}.npy").squeeze().astype(np.float32)
        full_conf_out[stem] = np.load(full_dir / "confidence" / f"{stem}.npy").squeeze().astype(np.float32)
        out_frames.append(f)
    kill_masks = {stem: np.zeros_like(d, dtype=bool) for stem, d in full_depth_out.items()}

    # 2. crop views
    for boxes_path in args.crop_boxes:
        meta = json.loads(Path(boxes_path).read_text())
        region = boxes_path.replace("\\", "/").split("/")[-2].replace("crop_", "")
        boxes = meta["boxes_800"]
        for f in full_frames:
            stem = Path(f["frame"]).stem
            crop_name = f"{stem}_{region}"
            cf = mixed_frames.get(f"{crop_name}.png")
            if cf is None:
                continue
            bx, by, side = boxes[stem]
            # intrinsics: full K_original shifted, mapped to the crop model grid
            aff = np.asarray(cf["pixel_affine_original_to_model"], dtype=np.float64)
            k_model = aff @ shifted_k(np.asarray(f["K_original"], dtype=np.float64), bx, by)
            cd = np.load(mixed_dir / "depth" / f"{crop_name}.npy").squeeze().astype(np.float64)
            cc = np.load(mixed_dir / "confidence" / f"{crop_name}.npy").squeeze().astype(np.float32)
            ch, cw = cd.shape
            # full depth/conf regridded onto the crop grid (nearest; full is 512^2 at 0.64x800)
            fd = np.load(full_dir / "depth" / f"{stem}.npy").squeeze().astype(np.float64)
            fc = np.load(full_dir / "confidence" / f"{stem}.npy").squeeze().astype(np.float32)
            yy, xx = np.indices((ch, cw), dtype=np.float32)
            # crop model pixel -> crop-image pixel (via inverse affine) -> full model pixel (0.64)
            ainv = np.linalg.inv(aff)
            hom = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], 0)
            ci = (ainv @ hom)  # crop-image coords
            fm_u = (ci[0] + bx) * 0.64
            fm_v = (ci[1] + by) * 0.64
            ui = np.clip(np.rint(fm_u).astype(int), 0, fd.shape[1] - 1).reshape(ch, cw)
            vi = np.clip(np.rint(fm_v).astype(int), 0, fd.shape[0] - 1).reshape(ch, cw)
            fd_r = fd[vi, ui]
            fc_r = fc[vi, ui]
            both = (cd > 0) & (fd_r > 0)
            scalar = float(np.median(fd_r[both] / cd[both])) if both.sum() > 100 else 1.0
            t_export = np.asarray(f["T_world_to_cam"], dtype=np.float64)
            if args.refine_pose and both.sum() > 100:
                rho, rot_np, tau_np, residual_med, residual_p90 = refine_pose(
                    k_model, full_pack, t_export, cd, cc, fd_r, scalar, args.refine_iters, args.device)
                scalar = rho
                # fold the correction into the exported pose:
                # world_corr = R_corr * (T_full^-1 cam) + tau  <=>  T_c = inv([R_corr*R_c2w | R_corr*C + tau])
                t_full_inv = np.linalg.inv(t_export)
                rot_c2w = rot_np @ t_full_inv[:3, :3]
                centre = rot_np @ t_full_inv[:3, 3] + tau_np
                t_export = np.eye(4)
                t_export[:3, :3] = rot_c2w.T
                t_export[:3, 3] = -rot_c2w.T @ centre
            else:
                diff0 = np.abs(cd[both] * scalar - fd_r[both]) / fd_r[both]
                residual_med = float(np.median(diff0))
                residual_p90 = float(np.percentile(diff0, 90))
            cd_s = cd * scalar
            diff = np.zeros_like(cd)
            diff[both] = np.abs(cd_s[both] - fd_r[both]) / fd_r[both]
            conflict = both & (diff > args.tol_rel)
            crop_wins = conflict & (cc >= fc_r)
            if args.conflict_mode == "replace":
                keep = cd > 0
                # zero the full view's whole crop rectangle (model coords)
                kill_masks[stem][vi, ui] = True
            elif args.conflict_mode == "adjudicate":
                # crop wins a conflict -> the full pixel is killed; crop loses -> dropped
                kill_masks[stem][vi[crop_wins], ui[crop_wins]] = True
                keep = (both & ~conflict) | crop_wins
            else:  # coexist (measured: double-sheets the surface)
                keep = (both & ~conflict) | crop_wins
            weight = cc.copy()
            weight[both & ~conflict] *= args.w_crop
            weight[crop_wins] *= args.w_crop
            if args.conflict_mode == "replace":
                weight = cc * args.w_crop
            cd_out = np.where(keep, cd_s, 0.0)
            np.save(out / "depth" / f"{crop_name}.npy", cd_out.astype(np.float32)[..., None])
            np.save(out / "confidence" / f"{crop_name}.npy",
                    np.where(keep, weight, 0.0).astype(np.float32))
            nf = dict(cf)
            nf["frame"] = f"{crop_name}.png"
            nf["file"] = f"{crop_name}.png"
            nf["K_model"] = k_model.tolist()
            nf["K_original"] = shifted_k(np.asarray(f["K_original"], dtype=np.float64), bx, by).tolist()
            nf["T_world_to_cam"] = t_export.tolist()  # full pose (+ optional refinement)
            nf["original_width"], nf["original_height"] = side, side
            nf["model_width"], nf["model_height"] = cw, ch
            out_frames.append(nf)
            report["crops"][crop_name] = {
                "scalar": scalar, "residual_med": residual_med, "residual_p90": residual_p90,
                "overlap_px": int(both.sum()), "conflict_frac": float(conflict.sum() / max(both.sum(), 1)),
                "crop_wins_frac": float(crop_wins.sum() / max(both.sum(), 1)),
                "kept_frac": float(keep.sum() / max((cd > 0).sum(), 1))}
    # 3. write full views with kill masks applied (conflict modes)
    for stem, d in full_depth_out.items():
        d = d.copy()
        d[kill_masks[stem]] = 0.0
        np.save(out / "depth" / f"{stem}.npy", d.astype(np.float32)[..., None])
        np.save(out / "confidence" / f"{stem}.npy", full_conf_out[stem])
        report.setdefault("killed_full_px", {})
        report["killed_full_px"][stem] = int(kill_masks[stem].sum())
    cameras = dict(full_cameras)
    cameras["frames"] = out_frames
    cameras["note"] = ("mixed-trajectory fusion: full views (carved) + close-up crops aligned "
                       "to full poses with robust scalar; conflict adjudicated by confidence; "
                       f"w_crop={args.w_crop}, tol_rel={args.tol_rel}")
    (out / "cameras.json").write_text(json.dumps(cameras, indent=2) + "\n")
    med = [c["residual_med"] for c in report["crops"].values()]
    p90 = [c["residual_p90"] for c in report["crops"].values()]
    report["residual_med_mean"] = float(np.mean(med))
    report["residual_p90_mean"] = float(np.mean(p90))
    (out / "fusion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"crops": len(report["crops"]),
                      "residual_med_mean": report["residual_med_mean"],
                      "residual_p90_mean": report["residual_p90_mean"],
                      "conflict_frac_mean": float(np.mean([c["conflict_frac"] for c in report["crops"].values()])),
                      "kept_frac_mean": float(np.mean([c["kept_frac"] for c in report["crops"].values()]))}, indent=2))


if __name__ == "__main__":
    main()
