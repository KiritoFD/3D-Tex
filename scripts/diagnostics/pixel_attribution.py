"""Pixel-level attribution for capture experiments (questions A and B).

For each GT foreground pixel across the capture views, compute a vector of
covariates plus the depth/normal residual of the predicted depth, then aggregate
by region type (A) or run a covariate attribution (B).

Covariates computed per GT foreground pixel (frame i, pixel p):
    view_count           number of capture frames whose frustum contains the
                         GT 3-D point of pixel p (fixed across configs)
    parallax_deg         max angular separation among the frames seeing p
    incidence_deg        angle between the camera ray and the GT normal at p
                         (grazing -> 90)
    image_grad           Sobel gradient magnitude of the source image at p
                         (weak-texture -> 0)
    dist_to_silhouette   pixel distance to the nearest GT foreground boundary
    dist_to_discont      pixel distance to the nearest GT depth discontinuity
                         (rel. jump > 4 * voxel in a 3x3 window)
    local_curvature      mean |d2| of the GT depth second derivative at p
    is_groove            1 if p lies inside a GT interior hole (binary_fill_holes
                         minus foreground), else 0
    highlight            1 if the source pixel luminance >= 245 (near-clip)

Region types (mutually exclusive priority):
    groove > thin_sharp > edge_band > weak_texture > highlight >
    high_curvature > flat
(the residual population after the earlier categories is "flat / other").

Outputs:
    <out>/pixel_attr.npz           full per-pixel array
    <out>/region_summary.json      region x {count, abs_rel, normal_deg, acc@1vox}
    <out>/covariate_corr.json      per-covariate Spearman rho vs abs_rel + binned
                                   mean abs_rel per covariate quartile
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_laplace,
    sobel,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics.pose_depth_ablation import depth_normals, regrid_depth
from verify_scene import umeyama


def spearman(x, y):
    """Spearman rho via rank correlation (scipy.stats not imported to keep deps small)."""
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--p0-out", default="deliverables/lego_p0")
    p.add_argument("--geometry-dir", default=None,
                   help="optional external geometry dir (defaults to capture/geometry/omega)")
    p.add_argument("--max-pixels", type=int, default=400000,
                   help="subsample GT foreground pixels for tractability")
    args = p.parse_args()
    capture = Path(args.capture)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = capture / "geometry" / "gt"
    pred = Path(args.geometry_dir) if args.geometry_dir else capture / "geometry" / "omega"

    p0 = json.loads((Path(args.p0_out) / "ablation_report.json").read_text())
    voxel = float(p0["voxel"])

    records = json.loads((gt / "cameras.json").read_text())["frames"]
    predictions = {f["frame"]: f for f in json.loads((pred / "cameras.json").read_text())["frames"]}
    gts = np.asarray([f["T_world_to_cam"] for f in records])
    if args.geometry_dir:
        scale, rotation, translation = 1.0, np.eye(3), np.zeros(3)
        aligned = gts.copy()
    else:
        pts = np.asarray([predictions[f["frame"]]["T_world_to_cam"] for f in records])
        gc = -np.einsum("sji,sj->si", gts[:, :3, :3], gts[:, :3, 3])
        pc = -np.einsum("sji,sj->si", pts[:, :3, :3], pts[:, :3, 3])
        scale, rotation, translation = umeyama(pc, gc)
        aligned = pts.copy()
        aligned[:, :3, :3] = pts[:, :3, :3] @ rotation.T
        aligned[:, :3, 3] = scale * pts[:, :3, 3] - aligned[:, :3, :3] @ translation

    # Per-frame precomputation
    per_frame = []
    for index, frame in enumerate(records):
        stem = Path(frame["frame"]).stem
        reference = np.load(gt / "depth" / f"{stem}.npy")
        foreground = np.load(gt / "foreground" / f"{stem}.npy")
        raw = np.load(pred / "depth" / f"{stem}.npy").squeeze() * scale
        k = np.asarray(frame["K_model"])
        pk = np.asarray(predictions[frame["frame"]]["K_model"])
        if args.geometry_dir:
            if raw.shape != reference.shape:
                import cv2
                raw = cv2.resize(raw.astype(np.float32), (reference.shape[1], reference.shape[0]),
                                 interpolation=__import__("cv2").INTER_NEAREST)
            d = np.where(foreground, raw, 0).astype(np.float32)
        else:
            yy, xx = np.indices(reference.shape, dtype=np.float32)
            u = (xx - k[0, 2]) / k[0, 0] * pk[0, 0] + pk[0, 2]
            v = (yy - k[1, 2]) / k[1, 1] * pk[1, 1] + pk[1, 2]
            d = np.where(foreground, regrid_depth(raw, u, v, "nearest"), 0).astype(np.float32)

        # GT normal from depth
        gn, gv = depth_normals(reference, k)

        # Covariates computed on GT
        fg = foreground.astype(bool)
        interior_hole = binary_fill_holes(fg) & ~fg
        dist_to_sil = distance_transform_edt(fg)
        # depth discontinuity: |d - shifted| > 4 vox in 3x3 (approx via gradient)
        gy, gx = np.gradient(reference)
        grad_mag = np.hypot(gx, gy)
        discont = (grad_mag > 4 * voxel) & fg
        dist_to_dis = distance_transform_edt(~discont) if discont.any() else np.full_like(
            reference, 99.0)
        # local curvature: |Laplace| of GT depth
        curv = np.abs(gaussian_laplace(np.where(fg, reference, 0), sigma=1.0))
        # texture: Sobel on image (load PNG luminance)
        image_path = capture / "images" / f"{stem}.png"
        from PIL import Image
        img = np.asarray(Image.open(image_path).convert("L")).astype(np.float32) / 255.0
        if img.shape != reference.shape:
            # model grid; resize luminance to GT grid for pixel-aligned queries
            import cv2
            img = cv2.resize(img, (reference.shape[1], reference.shape[0]))
        img_grad = np.hypot(sobel(img, axis=0), sobel(img, axis=1))
        highlight = (img >= 245 / 255.0)

        per_frame.append(dict(
            stem=stem, reference=reference, foreground=fg, d=d, k=k,
            T_gt=gts[index], T_pred=aligned[index],
            interior_hole=interior_hole, dist_to_sil=dist_to_sil,
            dist_to_dis=dist_to_dis, curv=curv, img_grad=img_grad,
            highlight=highlight, gn=gn, gv=gv,
        ))

    # Lift every GT foreground pixel to its 3-D point (GT frame) and count how
    # many of the OTHER frames see it. Approximation: project into the other
    # frames and check it lands inside the (dilated) foreground.
    H, W = records and per_frame[0]["reference"].shape
    for i, f in enumerate(per_frame):
        fg = f["foreground"]
        ref = f["reference"]
        k = f["k"]
        T_g2w = np.linalg.inv(f["T_gt"])  # cam -> world
        ys, xs = np.nonzero(fg)
        z = ref[ys, xs]
        # to camera coords
        x_cam = (xs - k[0, 2]) / k[0, 0] * z
        y_cam = (ys - k[1, 2]) / k[1, 1] * z
        pts_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=0)
        pts_world = (T_g2w @ pts_cam)[:3]
        # check in each other frame
        sees = np.zeros(len(z), dtype=np.int16)
        parallax = np.zeros(len(z), dtype=np.float32)
        cam_centres = []
        for j, g in enumerate(per_frame):
            if j == i:
                continue
            cam_c = -g["T_pred"][:3, :3].T @ g["T_pred"][:3, 3]
            cam_centres.append(cam_c)
            pts_in_g = g["T_pred"][:3, :3] @ pts_world + g["T_pred"][:3, 3:4]
            z_g = pts_in_g[2]
            valid = z_g > 1e-6
            u_g = np.rint(pts_in_g[0] / np.maximum(z_g, 1e-6) * g["k"][0, 0]
                          + g["k"][0, 2]).astype(int)
            v_g = np.rint(pts_in_g[1] / np.maximum(z_g, 1e-6) * g["k"][1, 1]
                          + g["k"][1, 2]).astype(int)
            inside = valid & (u_g >= 0) & (u_g < W) & (v_g >= 0) & (v_g < H)
            inside_idx = np.nonzero(inside)[0]
            if len(inside_idx):
                fg_other = binary_dilation(g["foreground"], iterations=2)
                sees[inside_idx[fg_other[v_g[inside_idx], u_g[inside_idx]]]] += 1
        # parallax: max angle between rays from p to each pair of camera centres
        cam_centres = np.asarray(cam_centres)
        if len(cam_centres) >= 2:
            rays = pts_world[None, :, :] - cam_centres[:, :, None]
            rays = rays / np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-9)
            # pairwise cosine; max parallax = min cosine
            # (N_cam, N_pix) x (N_cam, N_pix) is too big; approximate with mean ray
            mean_ray = rays.mean(axis=0)
            mean_ray /= np.maximum(np.linalg.norm(mean_ray, axis=0, keepdims=True), 1e-9)
            cos = (rays * mean_ray[None, :, :]).sum(axis=1).clip(-1, 1)
            parallax = np.degrees(np.arccos(cos.min(axis=0)))
        # incidence angle vs GT normal
        n = f["gn"][ys, xs]
        n_norm = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.where(n_norm > 1e-9, n / np.maximum(n_norm, 1e-9), 0)
        # camera forward axis in world
        fwd = -f["T_gt"][:3, 2]  # camera looks along -Z in cam coords
        incidence = np.degrees(np.arccos(np.clip(np.abs(n @ fwd), 0, 1)))
        f["pix"] = dict(ys=ys, xs=xs, sees=sees, parallax=parallax,
                        incidence=incidence, z=z)

    # Aggregate per-pixel table
    rng = np.random.default_rng(20261103)
    rows = []
    for f in per_frame:
        d, ref, fg = f["d"], f["reference"], f["foreground"]
        ys, xs = f["pix"]["ys"], f["pix"]["xs"]
        valid = (ref[ys, xs] > 0) & (d[ys, xs] > 0)
        abs_rel = np.abs(d[ys, xs] - ref[ys, xs]) / np.maximum(ref[ys, xs], 1e-9)
        # normal angle
        pn, pv = depth_normals(d, f["k"])
        n_dot = (f["gn"][ys, xs] * pn[ys, xs]).sum(axis=1)
        normal_deg = np.degrees(np.arccos(np.clip(np.abs(n_dot), 0, 1)))
        rows.append(dict(
            abs_rel=abs_rel.astype(np.float32),
            normal_deg=normal_deg.astype(np.float32),
            valid=valid,
            views=f["pix"]["sees"].astype(np.int16),
            parallax=f["pix"]["parallax"].astype(np.float32),
            incidence=f["pix"]["incidence"].astype(np.float32),
            img_grad=f["img_grad"][ys, xs].astype(np.float32),
            dist_to_sil=f["dist_to_sil"][ys, xs].astype(np.float32),
            dist_to_dis=f["dist_to_dis"][ys, xs].astype(np.float32),
            curv=f["curv"][ys, xs].astype(np.float32),
            groove=f["interior_hole"][ys, xs],
            highlight=f["highlight"][ys, xs],
        ))
    keys = rows[0].keys()
    table = {k: np.concatenate([r[k] for r in rows]) for k in keys}
    if len(table["abs_rel"]) > args.max_pixels:
        sel = rng.choice(len(table["abs_rel"]), args.max_pixels, replace=False)
        table = {k: v[sel] for k, v in table.items()}

    np.savez_compressed(out / "pixel_attr.npz", **table)

    # Region summary (question A)
    valid = table["valid"]
    groove = table["groove"] & valid
    thin_sharp = (table["curv"] > np.percentile(table["curv"][valid], 90)) & valid & ~groove
    edge_band = (table["dist_to_dis"] <= 2.0) & valid & ~groove & ~thin_sharp
    weak_tex = (table["img_grad"] <= np.percentile(table["img_grad"][valid], 20)) & valid \
        & ~groove & ~thin_sharp & ~edge_band
    highlight = table["highlight"] & valid & ~groove & ~thin_sharp & ~edge_band & ~weak_tex
    high_curv = (table["curv"] > np.percentile(table["curv"][valid], 70)) & valid \
        & ~groove & ~thin_sharp & ~edge_band & ~weak_tex & ~highlight
    flat = valid & ~groove & ~thin_sharp & ~edge_band & ~weak_tex & ~highlight & ~high_curv
    regions = {
        "groove": groove, "thin_sharp": thin_sharp, "edge_band": edge_band,
        "weak_texture": weak_tex, "highlight": highlight,
        "high_curvature": high_curv, "flat": flat,
    }
    summary = {}
    for name, mask in regions.items():
        if not mask.any():
            summary[name] = {"count": 0}
            continue
        a = table["abs_rel"][mask]
        n = table["normal_deg"][mask]
        summary[name] = {
            "count": int(mask.sum()),
            "abs_rel_mean": float(a.mean()),
            "abs_rel_p90": float(np.percentile(a, 90)),
            "normal_deg_mean": float(n.mean()),
            "within_1vox": float((a * np.mean([r["z"].mean() for r in
                                               [f["pix"] for f in per_frame]]) <= voxel).mean())
            if mask.any() else None,
        }
    (out / "region_summary.json").write_text(json.dumps(summary, indent=2))

    # Covariate attribution (question B)
    covariates = ["views", "parallax", "incidence", "img_grad", "dist_to_sil",
                  "dist_to_dis", "curv", "groove", "highlight"]
    corr = {}
    for c in covariates:
        x = table[c][valid].astype(np.float64)
        y = table["abs_rel"][valid].astype(np.float64)
        rho = spearman(x, y) if np.std(x) > 0 else 0.0
        # quartile binning
        qs = np.percentile(x, [0, 25, 50, 75, 100])
        bins = []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (x >= lo) & (x <= hi)
            bins.append(dict(range=[float(lo), float(hi)],
                             abs_rel=float(y[m].mean()) if m.any() else None,
                             count=int(m.sum())))
        corr[c] = {"spearman_rho": rho, "bins": bins}
    (out / "covariate_corr.json").write_text(json.dumps(corr, indent=2))

    # Console digest
    print("== region summary ==")
    for name, s in summary.items():
        if s.get("count"):
            print(f"  {name:>14}: n={s['count']:>7} abs_rel={s['abs_rel_mean']:.4f} "
                  f"normal={s['normal_deg_mean']:.2f}")
    print("== covariate |rho| with abs_rel ==")
    for c, s in sorted(corr.items(), key=lambda kv: -abs(kv[1]["spearman_rho"])):
        print(f"  {c:>14}: rho={s['spearman_rho']:+.3f}")


if __name__ == "__main__":
    main()
