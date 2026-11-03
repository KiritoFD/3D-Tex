#!/usr/bin/env python3
"""P1: per-view multiplicative depth-residual refinement for VGGT-Omega depth.

D_i' = D_i * exp(delta_i), where delta_i is a multi-scale set of bilinearly
upsampled low-resolution grids per view. Optimisation inputs are ONLY the
predicted depth/poses and the RGB images (alpha matte as the object mask);
GT depth / GT normals / GT mesh are never read. Evaluation afterwards is the
unchanged frozen protocol (capture_eval.py + measure_surfaces.py).

Losses (per iteration, over all views):
  L_photo   multi-view photometric consistency: ZNCC of 5x5 patches between the
            reference view and its 4 nearest source views, warped through the
            refined depth; occluded samples (z_proj behind the source depth by
            more than --occ-tol relative) are masked out.
  L_depth   multi-view depth consistency: huber of the relative difference
            between the projected refined depth and the source refined depth,
            same occlusion mask.
  L_smooth  edge-aware first-order TV on delta (weights from RGB gradients, so
            real edges are not penalised), plus a small curvature (2nd-order)
            penalty on delta to keep the correction low-order without forcing
            the refined surface itself to be planar.
  L_prior   huber(delta, --prior-beta): bounds the residual magnitude and
            prevents drift away from the VGGT prior.

Outputs a geometry dir in the exact "omega" layout (cameras.json + depth/*.npy)
so capture_eval.py runs the identical default path (Sim3 + nearest regrid +
pred-pose TSDF) on baseline and refined depth alike.

    .venv/Scripts/python.exe scripts/refine_depth.py \
        --omega data/scenes/nerf_lego16/geometry/omega \
        --images data/scenes/nerf_lego16/images \
        --out outputs/p1_refine/run_main
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_frames(omega: Path, images_dir: Path, device):
    """Load model-grid depth/poses/K and alpha-warped RGB for every frame."""
    import cv2
    from PIL import Image

    frames = json.loads((omega / "cameras.json").read_text())["frames"]
    depths, rgbs, masks, ks, ts, names = [], [], [], [], [], []
    for frame in frames:
        stem = Path(frame["frame"]).stem
        depth = np.load(omega / "depth" / f"{stem}.npy").squeeze().astype(np.float32)
        height, width = depth.shape
        k = np.asarray(frame["K_model"], dtype=np.float64).copy()
        if (height, width) != (frame["model_height"], frame["model_width"]):
            k[0] *= width / frame["model_width"]
            k[1] *= height / frame["model_height"]
        with Image.open(images_dir / frame["file"]) as im:
            rgba = np.asarray(im.convert("RGBA")).astype(np.float32) / 255.0
        affine = np.asarray(frame["pixel_affine_original_to_model"], dtype=np.float64).copy()
        affine[0] *= width / frame["model_width"]
        affine[1] *= height / frame["model_height"]
        warped = cv2.warpAffine(rgba, affine[:2], (width, height),
                                flags=cv2.INTER_AREA, borderValue=0.0)
        depths.append(torch.from_numpy(depth).to(device))
        rgbs.append(torch.from_numpy(warped[..., :3]).permute(2, 0, 1).to(device))
        # Alpha matte when present; real photographs (no alpha) fall back to
        # "valid predicted depth" as the object mask.
        alpha = warped[..., 3] if rgba.shape[-1] == 4 and np.ptp(warped[..., 3]) > 0.5 else None
        masks.append(torch.from_numpy(alpha > 0.5 if alpha is not None else depth > 0).to(device))
        ks.append(torch.from_numpy(k).float().to(device))
        ts.append(torch.from_numpy(np.asarray(frame["T_world_to_cam"])).float().to(device))
        names.append(stem)
    return frames, names, depths, rgbs, masks, ks, ts


def edge_weights(rgb):
    """Per-axis smoothness weights from image gradients (1 at flat, ~0 at edges)."""
    gray = rgb.mean(dim=0, keepdim=True)
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    gx[:, :, 1:] = gray[:, :, 1:] - gray[:, :, :-1]
    gy[:, 1:, :] = gray[:, 1:, :] - gray[:, :-1, :]
    sigma = 0.08
    return torch.exp(-gx.abs() / sigma), torch.exp(-gy.abs() / sigma)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omega", required=True, help="omega-format geometry dir (cameras.json + depth/)")
    parser.add_argument("--images", required=True, help="original RGB(A) images dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iters", type=int, default=1200)
    parser.add_argument("--samples", type=int, default=8192, help="pixels per view per iter")
    parser.add_argument("--sources", type=int, default=4, help="nearest source views per ref view")
    parser.add_argument("--patch", type=int, default=5, help="ZNCC patch side (odd)")
    parser.add_argument("--grids", default="16,64,128", help="residual grid resolutions (square)")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lambda-photo", type=float, default=1.0)
    parser.add_argument("--lambda-depth", type=float, default=1.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.5)
    parser.add_argument("--lambda-curv", type=float, default=0.1)
    parser.add_argument("--lambda-prior", type=float, default=1.0)
    parser.add_argument("--joint-pose", action="store_true",
                        help="jointly refine a small 6-DOF pose correction per view; "
                             "predicted-pose misalignment otherwise gets absorbed into "
                             "depth ripples (measured: GT poses raise ZNCC 0.09->0.155 "
                             "even with predicted depth)")
    parser.add_argument("--lambda-pose-prior", type=float, default=1.0,
                        help="L2 anchor on pose corrections (rot 0.02 rad / trans 0.02 gauge units)")
    parser.add_argument("--occ-tol", type=float, default=0.015, help="relative occlusion tolerance")
    parser.add_argument("--photo-cap", type=float, default=0.0,
                        help="if >0, truncate the per-pixel photo residual r=1-ZNCC at this "
                             "value (min(r, cap)): pixels worse than 1-cap ZNCC (specular, "
                             "unfixable) stop pulling, preventing adversarial overshoot")
    parser.add_argument("--depth-beta", type=float, default=0.01, help="huber beta, relative depth")
    parser.add_argument("--prior-beta", type=float, default=0.02, help="huber beta, residual")
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--holdout", type=int, default=0,
                        help="hold out N random views from being used as photo SOURCE views; "
                             "the held-out-transfer ZNCC gap then tests whether the photometric "
                             "signal is geometrically real (generalises) or adversarial (does not)")
    parser.add_argument("--holdout-seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    omega, out = Path(args.omega), Path(args.out)
    frames, names, depth0, rgb, mask, ks, ts = load_frames(omega, Path(args.images), device)
    n_views = len(frames)
    height, width = depth0[0].shape
    half = args.patch // 2
    patch_offsets = torch.stack(torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32), indexing="ij"),
        dim=-1).reshape(-1, 2).to(device)  # (P, 2) as (dv, du)
    n_patch = patch_offsets.shape[0]

    # Source-view selection: nearest camera centres in the predicted gauge.
    centres = torch.stack([-(t[:3, :3].mT @ t[:3, 3]) for t in ts])
    dist = torch.cdist(centres, centres)
    dist.fill_diagonal_(float("inf"))
    holdout_idx = []
    if args.holdout > 0:
        gen = torch.Generator().manual_seed(args.holdout_seed)
        holdout_idx = torch.randperm(n_views, generator=gen)[: args.holdout].tolist()
        dist[:, holdout_idx] = float("inf")
    source_idx = dist.argsort(dim=1)[:, : args.sources]  # (V, S)

    # Residual parameters: one set of grids per view.
    grid_sizes = [int(g) for g in args.grids.split(",")]
    params = []
    for _ in range(n_views):
        for g in grid_sizes:
            p = torch.zeros(1, 1, g, g, device=device, requires_grad=True)
            params.append(p)

    # Optional per-view 6-DOF pose corrections (rotvec rho, translation tau).
    pose_rho = torch.zeros(n_views, 3, device=device, requires_grad=args.joint_pose)
    pose_tau = torch.zeros(n_views, 3, device=device, requires_grad=args.joint_pose)
    if args.joint_pose:
        params += [pose_rho, pose_tau]

    def rodrigues(rho):
        theta = rho.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        k = rho / theta
        zeros = torch.zeros_like(k[:, 0])
        kx = torch.stack([zeros, -k[:, 2], k[:, 1],
                          k[:, 2], zeros, -k[:, 0],
                          -k[:, 1], k[:, 0], zeros], dim=-1).reshape(-1, 3, 3)
        eye = torch.eye(3, device=rho.device).expand_as(kx)
        return (eye + torch.sin(theta)[..., None] * kx
                + (1 - torch.cos(theta))[..., None] * (kx @ kx))

    def posed(view):
        """Corrected (R, t) for one view."""
        rot = rodrigues(pose_rho[view : view + 1])[0] @ t_all[view][:3, :3] \
            if args.joint_pose else t_all[view][:3, :3]
        tra = t_all[view][:3, 3] + pose_tau[view] if args.joint_pose else t_all[view][:3, 3]
        return rot, tra

    def residual(view):
        base = view * len(grid_sizes)
        delta = torch.zeros(1, 1, height, width, device=device)
        for gi, g in enumerate(grid_sizes):
            delta = delta + F.interpolate(params[base + gi], size=(height, width),
                                          mode="bilinear", align_corners=False)
        return delta[0, 0]

    # Precompute edge weights and valid-pixel index lists.
    wx_list = [edge_weights(rgb[v])[0] for v in range(n_views)]
    wy_list = [edge_weights(rgb[v])[1] for v in range(n_views)]
    mask_idx = [torch.nonzero(mask[v], as_tuple=True) for v in range(n_views)]
    kinv = torch.stack([torch.inverse(k) for k in ks])
    k_all = torch.stack(ks)
    t_all = torch.stack(ts)
    depth0_stack = torch.stack(depth0)

    opt = torch.optim.Adam(params, lr=args.lr)
    log = {"args": vars(args), "iters": []}
    started = time.time()

    for it in range(args.iters):
        opt.zero_grad()
        refined = [depth0[v] * torch.exp(residual(v)) for v in range(n_views)]
        refined_stack = torch.stack(refined)
        l_photo = torch.zeros((), device=device)
        l_depth = torch.zeros((), device=device)
        n_photo = torch.zeros((), device=device)
        n_depth = torch.zeros((), device=device)
        for v in range(n_views):
            rows, cols = mask_idx[v]
            sel = torch.randint(0, len(rows), (args.samples,), device=device)
            r, c = rows[sel].float(), cols[sel].float()
            d = refined[v][r.long(), c.long()]
            rot_v, tra_v = posed(v)
            # Reference -> world -> each source view.
            pix = torch.stack([c, r, torch.ones_like(c)], dim=-1)  # (M, 3)
            cam = (kinv[v] @ pix.unsqueeze(-1)).squeeze(-1) * d.unsqueeze(-1)
            world = (rot_v.mT @ (cam - tra_v).unsqueeze(-1)).squeeze(-1)
            # Reference patch samples (constant per view, no grad needed).
            # Per-channel ZNCC: (M, P*3) vectors — grayscale ZNCC is nearly
            # signal-free on glossy plastic (GT ceiling 0.26), per-channel
            # reaches 0.65 (diagnostic: outputs/p1_refine/diag_zncc.py).
            pr = (r.unsqueeze(-1) + patch_offsets[:, 0]).clamp(0, height - 1).long()
            pc = (c.unsqueeze(-1) + patch_offsets[:, 1]).clamp(0, width - 1).long()
            ref_vec = rgb[v][:, pr, pc].permute(1, 2, 0).reshape(len(r), -1)  # (M, P*3)
            for s in source_idx[v]:
                rot_s, tra_s = posed(int(s))
                cam_s = (rot_s @ world.unsqueeze(-1)).squeeze(-1) + tra_s
                z = cam_s[:, 2].clamp(min=1e-6)
                uf = cam_s[:, 0] / z * k_all[s][0, 0] + k_all[s][0, 2]
                vf = cam_s[:, 1] / z * k_all[s][1, 1] + k_all[s][1, 2]
                inside = (uf >= half) & (uf < width - half) & (vf >= half) & (vf < height - half)
                # Bilinear source depth of the refined map at the projection.
                def sample_depth(dmap):
                    gx = (uf / (width - 1) * 2 - 1).view(1, 1, -1)
                    gy = (vf / (height - 1) * 2 - 1).view(1, 1, -1)
                    grid = torch.stack([gx, gy], dim=-1)
                    return F.grid_sample(dmap.view(1, 1, height, width), grid,
                                         mode="bilinear", align_corners=True).view(-1)
                d_src = sample_depth(refined_stack[s])
                d0_src = sample_depth(depth0_stack[s]).detach()
                occluded = (z - d0_src) > args.occ_tol * d0_src
                valid = inside & ~occluded & (d0_src > 0)
                rel = (z - d_src) / d_src.clamp(min=1e-6)
                l_depth = l_depth + (F.huber_loss(rel[valid], torch.zeros_like(rel[valid]),
                                                  delta=args.depth_beta)).nan_to_num() * valid.sum()
                n_depth = n_depth + valid.sum()
                # Patch ZNCC: sample the source image at the warped patch grid.
                pu = (uf.unsqueeze(-1) + patch_offsets[:, 1])  # (M, P)
                pv = (vf.unsqueeze(-1) + patch_offsets[:, 0])
                gx = (pu / (width - 1) * 2 - 1).unsqueeze(0)  # (1, M, P)
                gy = (pv / (height - 1) * 2 - 1).unsqueeze(0)
                grid = torch.stack([gx, gy], dim=-1)
                src_vec = F.grid_sample(rgb[s].unsqueeze(0), grid, mode="bilinear",
                                        align_corners=True)[0].permute(1, 2, 0).reshape(len(r), -1)  # (M, P*3)
                a = ref_vec - ref_vec.mean(dim=1, keepdim=True)
                b = src_vec - src_vec.mean(dim=1, keepdim=True)
                zncc = (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-6)
                contrast = (a.norm(dim=1) > 1e-3) & (b.norm(dim=1) > 1e-3)
                photo_valid = valid & contrast
                r_photo = 1.0 - zncc
                if args.photo_cap > 0:
                    r_photo = torch.clamp(r_photo, max=args.photo_cap)
                l_photo = l_photo + r_photo[photo_valid].sum()
                n_photo = n_photo + photo_valid.sum()
        l_photo = l_photo / n_photo.clamp(min=1)
        l_depth = l_depth / n_depth.clamp(min=1)
        # Smoothness / curvature / prior on the residuals (full image, all views).
        l_smooth = torch.zeros((), device=device)
        l_curv = torch.zeros((), device=device)
        l_prior = torch.zeros((), device=device)
        for v in range(n_views):
            delta = residual(v)
            dx = delta[:, 1:] - delta[:, :-1]
            dy = delta[1:, :] - delta[:-1, :]
            l_smooth = l_smooth + (wx_list[v][0, :, 1:] * dx.abs()).mean() \
                                + (wy_list[v][0, 1:, :] * dy.abs()).mean()
            l_curv = l_curv + (delta[:, 2:] - 2 * delta[:, 1:-1] + delta[:, :-2]).abs().mean() \
                            + (delta[2:, :] - 2 * delta[1:-1, :] + delta[:-2, :]).abs().mean()
            l_prior = l_prior + F.huber_loss(delta, torch.zeros_like(delta), delta=args.prior_beta)
        l_smooth = l_smooth / n_views
        l_curv = l_curv / n_views
        l_prior = l_prior / n_views
        loss = (args.lambda_photo * l_photo + args.lambda_depth * l_depth
                + args.lambda_smooth * l_smooth + args.lambda_curv * l_curv
                + args.lambda_prior * l_prior)
        if args.joint_pose:
            loss = loss + args.lambda_pose_prior * (
                (pose_rho ** 2).mean() / 0.02 ** 2 + (pose_tau ** 2).mean() / 0.02 ** 2)
        loss.backward()
        opt.step()
        if (it + 1) % args.log_every == 0 or it == 0:
            with torch.no_grad():
                deltas = torch.stack([residual(v) for v in range(n_views)])
                entry = {"iter": it + 1, "loss": float(loss), "photo": float(l_photo),
                         "depth": float(l_depth), "smooth": float(l_smooth),
                         "curv": float(l_curv), "prior": float(l_prior),
                         "delta_abs_mean": float(deltas.abs().mean()),
                         "delta_abs_p95": float(deltas.abs().quantile(0.95)),
                         "photo_frac": float(n_photo) / float(n_depth.clamp(min=1)),
                         "rho_deg_mean": float(pose_rho.norm(dim=1).mean() * 57.2958),
                         "tau_mean": float(pose_tau.norm(dim=1).mean())}
                log["iters"].append(entry)
                print(json.dumps(entry), flush=True)

    # Export refined depth in the omega layout.
    out_geo = out / "geometry" / "omega"
    (out_geo / "depth").mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for v in range(n_views):
            d = (depth0[v] * torch.exp(residual(v))).cpu().numpy().astype(np.float32)
            np.save(out_geo / "depth" / f"{names[v]}.npy", d[..., None])
    cameras = json.loads((omega / "cameras.json").read_text())
    if args.joint_pose:
        # Persist corrected poses so capture_eval re-runs its frozen Sim3 on them.
        for v, frame in enumerate(cameras["frames"]):
            rot, tra = posed(v)
            t4 = np.eye(4)
            t4[:3, :3] = rot.detach().cpu().numpy()
            t4[:3, 3] = tra.detach().cpu().numpy()
            frame["T_world_to_cam"] = t4.tolist()
    cameras["note"] = ("P1 refined: D' = D * exp(delta), multi-scale grids "
                       f"{grid_sizes}; joint_pose={args.joint_pose}; losses "
                       f"photo(ZNCC{args.patch})/depth-consistency/edge-aware "
                       "smooth+curv/prior; see refine log.")
    (out_geo / "cameras.json").write_text(json.dumps(cameras, indent=2) + "\n")
    log["seconds"] = round(time.time() - started, 1)
    log["delta_stats"] = log["iters"][-1]
    log["holdout_views"] = [names[i] for i in holdout_idx]
    (out / "refine_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"[ok] refined geometry -> {out_geo} ({log['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
