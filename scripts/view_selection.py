"""View-selection experiment: per-view marginal contribution to VGGT-Omega geometry.

Independent experiment (docs/experiments/VIEW_SELECTION.md). Given a fixed core view
set A and candidate sets B, measure whether adding views changes the model's own
depth/normal predictions on A (interference, E1 vs E0) and/or only the fused mesh
(fusion effect, E2 vs E1), then run leave-one-out over a 32-view set to get each
photo's marginal contribution, and validate deployable no-GT selection signals
against the GT marginal contributions.

Subcommands
    select   choose A / B_core / B1..B6 view sets, write selection.json
    build    materialise run dirs (image copies / blur-noise / tone-shift transforms)
    infer    VGGT-Omega geometry export for one run dir (protocol-identical to
             adapters/vggt_omega_geometry.py; adds --exclude, --save-tokens and a
             free-VRAM gate; skips the ASCII point cloud)
    eval     depth/normal/RPE eval of an omega export against a GT dir (same math
             as capture_eval.py, no fusion; per-view table for LOO)
    score    deployable per-view signals (quality / redundancy / consistency)
    loo      leave-one-out driver over a 32-view run dir + random-deletion controls
    viz      visual review panels (depth/normal/errors, mesh z-buffer views)

Everything is parameterized; paths default to this repo's layout but are not
hard-coded. No existing file or protocol is modified.
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
for extra in (SCRIPTS, SCRIPTS / "diagnostics", SCRIPTS / "adapters"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from verify_scene import umeyama  # noqa: E402
from pose_depth_ablation import depth_normals, regrid_depth  # noqa: E402
from resampling_normal_ablation import gt_edge_mask, gt_flat_mask, normal_angle, summarize  # noqa: E402

DATASET = ROOT / "data" / "scenes" / "nerf_lego"
DATASET_ZIP = ROOT / "data" / ".cache" / "nerf-syn" / "nerf_syn.zip"
ANCHOR_IMAGES = ROOT / "data" / "scenes" / "nerf_lego16" / "images"
ANCHOR_GT = ROOT / "data" / "scenes" / "nerf_lego_gt16" / "geometry" / "gt"
ANCHOR_OMEGA = ROOT / "data" / "scenes" / "nerf_lego16" / "geometry" / "omega"
ETH3D_ROOT = ROOT / "data" / "scenes"
VIEWSEL = ROOT / "data" / "scenes" / "viewsel"
CHECKPOINT = ROOT / "data" / ".cache" / "models" / "sanqilu376" / "VGGT-Omega" / "vggt_omega_1b_512.pt"
IMG_EXTS = (".jpg", ".jpeg", ".png")


# ----------------------------------------------------------------------------- helpers

def load_train_meta(dataset=DATASET):
    dataset = Path(dataset)
    meta = json.loads((dataset / "transforms_train.json").read_text())
    stems = [Path(f["file_path"]).stem for f in meta["frames"]]
    c2w = np.asarray([f["transform_matrix"] for f in meta["frames"]], dtype=np.float64)
    return stems, c2w


def geometry_stats(c2w):
    """Azimuth/elevation of camera centres (degrees) and unit centre vectors."""
    centres = c2w[:, :3, 3]
    radius = np.linalg.norm(centres, axis=1, keepdims=True)
    unit = centres / np.maximum(radius, 1e-12)
    az = np.degrees(np.arctan2(centres[:, 1], centres[:, 0]))
    el = np.degrees(np.arcsin(np.clip(centres[:, 2] / radius[:, 0], -1, 1)))
    return az, el, unit, centres


def angle_matrix(unit_a, unit_b):
    """Pairwise angular distance (deg) between view directions (centre vectors)."""
    dots = np.clip(unit_a @ unit_b.T, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def anchor_stems(anchor_images=ANCHOR_IMAGES):
    return sorted(p.stem for p in Path(anchor_images).iterdir() if p.suffix.lower() in IMG_EXTS)


def image_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fg_fraction_rgba(rgba):
    """Foreground pixel fraction of an RGBA image array (alpha > 0.5)."""
    return float((rgba[..., 3] > 127).mean()) if rgba.shape[-1] == 4 else 1.0


# ----------------------------------------------------------------------------- select

def cmd_select(args):
    from PIL import Image
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    stems, c2w = load_train_meta(args.dataset)
    az, el, unit, centres = geometry_stats(c2w)
    anchor = anchor_stems(args.anchor_images)
    anchor_idx = [stems.index(s) for s in anchor]
    remaining = np.array([i for i in range(len(stems)) if i not in set(anchor_idx)])
    selection = {"anchor16": anchor, "protocol_note": "A = the repo anchor16 set "
                 "(data/scenes/nerf_lego16); all candidates come from the remaining "
                 "84 train views or the 200 heldout test views of nerf-synthetic lego."}

    # B_core: uniform-azimuth-rank pick over the remaining 84 (mirrors build_dataset_subset).
    order = remaining[np.argsort(az[remaining], kind="stable")]
    picks = sorted({int(order[round(i * (len(order) - 1) / (args.n_candidate - 1))])
                    for i in range(args.n_candidate)})
    if len(picks) != args.n_candidate:
        raise SystemExit("B_core selection collided; adjust count")
    selection["B_core"] = [stems[i] for i in picks]

    # B1 near-duplicates: per anchor the angularly nearest remaining train view,
    # assigned greedily by global distance so picks stay unique.
    dist = angle_matrix(unit[anchor_idx], unit[remaining])
    pairs = sorted((dist[a, r], a, r) for a in range(len(anchor_idx)) for r in range(len(remaining)))
    used_a, used_r, b1 = set(), set(), []
    for d, a, r in pairs:
        if a in used_a or r in used_r:
            continue
        used_a.add(a); used_r.add(r)
        b1.append({"anchor": anchor[a], "stem": stems[remaining[r]], "angular_dist_deg": round(float(dist[a, r]), 3)})
        if len(b1) == args.n_candidate:
            break
    selection["B1"] = b1

    # B6 high-value: greedy max-min angular coverage starting from A.
    covered = list(anchor_idx)
    b6 = []
    pool = list(remaining)
    for _ in range(args.n_candidate):
        d = angle_matrix(unit[pool], unit[covered]).min(axis=1)
        best = int(np.argmax(d))
        b6.append({"stem": stems[pool[best]], "min_angular_dist_deg": round(float(d[best]), 3)})
        covered.append(pool.pop(best))
    selection["B6"] = b6

    # B4 extreme: smallest foreground fraction (8, "only a sliver visible / occluded")
    # + lowest |elevation| grazing views (8), from remaining train + all 200 test views.
    candidates = []
    for i in remaining:
        with Image.open(Path(args.dataset) / "images" / f"{stems[i]}.png") as im:
            fg = fg_fraction_rgba(np.asarray(im))
        candidates.append({"stem": stems[i], "source": "train", "fg_fraction": round(fg, 4),
                           "azimuth_deg": round(float(az[i]), 2), "elevation_deg": round(float(el[i]), 2),
                           "c2w": c2w[i].tolist()})
    with zipfile.ZipFile(args.dataset_zip) as archive:
        meta_test = json.loads(archive.read("nerf_synthetic/lego/transforms_test.json"))
        for frame in meta_test["frames"]:
            stem = "t_" + Path(frame["file_path"]).stem
            with archive.open(f"nerf_synthetic/lego/test/{Path(frame['file_path']).stem}.png") as fh:
                from io import BytesIO
                fg = fg_fraction_rgba(np.asarray(Image.open(BytesIO(fh.read()))))
            m = np.asarray(frame["transform_matrix"], dtype=np.float64)
            a2, e2, _, _ = geometry_stats(m[None])
            candidates.append({"stem": stem, "source": "test", "fg_fraction": round(fg, 4),
                               "azimuth_deg": round(float(a2[0]), 2),
                               "elevation_deg": round(float(e2[0]), 2), "c2w": m.tolist()})
    half = args.n_candidate // 2
    b4, seen = [], set()
    for ranked in (sorted(candidates, key=lambda c: c["fg_fraction"]),
                   sorted(candidates, key=lambda c: abs(c["elevation_deg"]))):
        picked = 0
        for c in ranked:
            if c["stem"] in seen:
                continue
            seen.add(c["stem"]); b4.append(c); picked += 1
            if picked >= half:
                break
    selection["B4"] = b4
    selection["B4_rule"] = (f"{half} smallest foreground-fraction + {half} lowest-|elevation| "
                            "views, train remainder + 200 heldout test views")

    # B5 unrelated: deterministic stride pick over the three ETH3D scenes.
    b5 = []
    for scene, count in (("eth3d_pipes", 6), ("eth3d_kicker", 5), ("eth3d_relief", 5)):
        imgs = sorted(p.name for p in (Path(args.eth3d_root) / scene / "images").iterdir()
                      if p.suffix.lower() in IMG_EXTS)
        step = max(1, len(imgs) // count)
        for name in imgs[::step][:count]:
            b5.append({"stem": f"{scene}_{Path(name).stem}", "scene": scene, "file": name})
    selection["B5"] = b5[:args.n_candidate]

    # B2/B3 reuse B1 viewpoints (matched-viewpoint quality/appearance pairs).
    selection["B2_viewpoints"] = "same as B1 (blur+noise+overexposure)"
    selection["B3_viewpoints"] = "same as B1 (gamma/exposure/channel-gain tone shift)"
    (out_root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps({"anchor16": len(anchor), "B_core": selection["B_core"],
                      "B1_mean_dist_deg": float(np.mean([b["angular_dist_deg"] for b in b1])),
                      "B6_min_dist_deg": [b["min_angular_dist_deg"] for b in b6],
                      "B4": [(b["stem"], b["fg_fraction"], b["elevation_deg"]) for b in selection["B4"]],
                      "B5": [b["stem"] for b in selection["B5"]]}, indent=2))


# ----------------------------------------------------------------------------- build

def degrade_low_quality(rgb):
    """B2: gaussian blur + additive noise + overexposure clip (deterministic seed)."""
    import cv2
    rng = np.random.default_rng(20261103)
    out = cv2.GaussianBlur(rgb, (0, 0), 2.0)
    out = out + rng.normal(0.0, 8.0, out.shape)
    return np.clip(out * 1.5, 0, 255)


def shift_appearance(rgb):
    """B3: appearance-inconsistency proxy (NOT real specular variation): gamma 1.8,
    exposure x0.8 and a channel-gain skew to shift the tone curve / white balance."""
    x = np.power(np.clip(rgb / 255.0, 0, 1), 1.8) * 0.8
    x = x * np.array([1.12, 1.00, 0.88])
    return np.clip(x * 255.0, 0, 255)


def copy_frame(src_path, dst_dir, transform=None):
    from PIL import Image
    dst_dir.mkdir(parents=True, exist_ok=True)
    if transform is None:
        shutil.copyfile(src_path, dst_dir / src_path.name)
        return image_sha256(dst_dir / src_path.name)
    with Image.open(src_path) as im:
        rgba = np.asarray(im.convert("RGBA"))
    rgb = transform(rgba[..., :3].astype(np.float32)).astype(np.uint8)
    out = np.dstack([rgb, rgba[..., 3]])
    Image.fromarray(out, "RGBA").save(dst_dir / src_path.name)
    return image_sha256(dst_dir / src_path.name)


def cmd_build(args):
    out_root = Path(args.out_root)
    selection = json.loads((out_root / "selection.json").read_text())
    stems, c2w = load_train_meta(args.dataset)
    az, el, unit, _ = geometry_stats(c2w)
    anchor = selection["anchor16"]
    runs = {"e0": [], "ab": [], "b1": [], "b2": [], "b3": [], "b4": [], "b5": [], "b6": []}
    poses = {}

    def add_anchor(run):
        for stem in anchor:
            src = Path(args.dataset) / "images" / f"{stem}.png"
            sha = copy_frame(src, run / "images")
            runs[run.name].append({"stem": stem, "role": "core", "sha256": sha})

    def add_train(run, stem, role, transform=None, ops=None):
        src = Path(args.dataset) / "images" / f"{stem}.png"
        sha = copy_frame(src, run / "images", transform)
        i = stems.index(stem)
        runs[run.name].append({"stem": stem, "role": role, "transform": ops,
                               "sha256": sha, "azimuth_deg": round(float(az[i]), 2),
                               "elevation_deg": round(float(el[i]), 2)})
        poses.setdefault(run.name, []).append({"stem": stem, "c2w_blender": c2w[i].tolist()})

    for name in runs:
        run = out_root / "runs" / name
        run.mkdir(parents=True, exist_ok=True)
        add_anchor(run)
        poses.setdefault(name, [])
        a_idx = [stems.index(s) for s in anchor]
        poses[name].extend({"stem": s, "c2w_blender": c2w[i].tolist()} for s, i in zip(anchor, a_idx))
    for stem in selection["B_core"]:
        add_train(out_root / "runs" / "ab", stem, "candidate")
    for entry in selection["B1"]:
        add_train(out_root / "runs" / "b1", entry["stem"], "candidate")
        add_train(out_root / "runs" / "b2", entry["stem"], "candidate", degrade_low_quality,
                  "gaussian_blur_sigma2 + noise_sigma8/255 + exposure_x1.5_clip")
        add_train(out_root / "runs" / "b3", entry["stem"], "candidate", shift_appearance,
                  "gamma_1.8 * exposure_0.8 * channel_gain(1.12,1.00,0.88)")
    with zipfile.ZipFile(args.dataset_zip) as archive:
        for entry in selection["B4"]:
            run = out_root / "runs" / "b4"
            if entry["source"] == "train":
                add_train(run, entry["stem"], "candidate")
            else:
                from io import BytesIO
                from PIL import Image
                raw = entry["stem"][2:]
                with archive.open(f"nerf_synthetic/lego/test/{raw}.png") as fh:
                    img = Image.open(BytesIO(fh.read()))
                    img.save(run / "images" / f"{entry['stem']}.png")
                runs["b4"].append({"stem": entry["stem"], "role": "candidate",
                                   "sha256": image_sha256(run / "images" / f"{entry['stem']}.png"),
                                   "azimuth_deg": entry["azimuth_deg"],
                                   "elevation_deg": entry["elevation_deg"]})
                poses["b4"].append({"stem": entry["stem"], "c2w_blender": entry["c2w"]})
    for entry in selection["B5"]:
        run = out_root / "runs" / "b5"
        src = Path(args.eth3d_root) / entry["scene"] / "images" / entry["file"]
        dst = run / "images" / f"{entry['stem']}.png"
        from PIL import Image
        with Image.open(src) as im:
            im.convert("RGB").save(dst)
        runs["b5"].append({"stem": entry["stem"], "role": "candidate", "scene": entry["scene"],
                           "sha256": image_sha256(dst)})
    for entry in selection["B6"]:
        add_train(out_root / "runs" / "b6", entry["stem"], "candidate")
    for name, frames in runs.items():
        run = out_root / "runs" / name
        manifest = {"run": name, "frames": frames,
                    "note": "core frames are byte-identical copies of the dataset images; "
                            "candidate transforms are recorded per frame"}
        (run / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if name in ("ab", "b1", "b4", "b6", "b5"):
            # poses.json for the Blender BVH depth pass (b2/b3 share b1 geometry).
            (run / "poses.json").write_text(json.dumps(poses[name], indent=2) + "\n")
        print(f"[build] {name}: {len(frames)} frames -> {run}")


# ----------------------------------------------------------------------------- assemble

def cmd_assemble(args):
    """Assemble a capture_eval-compatible case dir:
    cases/<case>/geometry/gt (subset of a GT source) + geometry/omega (copy of a run export).
    Omega copying is skipped when --omega-src does not exist yet (GT-only pre-assembly)."""
    case = Path(args.out_root) / "cases" / args.case
    gt_src = Path(args.gt_src)
    stems = args.gt_stems.split(",") if args.gt_stems else None
    gt_dst = case / "geometry" / "gt"
    for sub in ("depth", "foreground"):
        (gt_dst / sub).mkdir(parents=True, exist_ok=True)
    records = json.loads((gt_src / "cameras.json").read_text())["frames"]
    if stems is not None:
        records = [f for f in records if Path(f["frame"]).stem in set(stems)]
        missing = set(stems) - {Path(f["frame"]).stem for f in records}
        if missing:
            raise SystemExit(f"GT source misses {sorted(missing)}")
    for f in records:
        stem = Path(f["frame"]).stem
        for sub in ("depth", "foreground"):
            src_f, dst_f = gt_src / sub / f"{stem}.npy", gt_dst / sub / f"{stem}.npy"
            if src_f.resolve() != dst_f.resolve():
                shutil.copyfile(src_f, dst_f)
    (gt_dst / "cameras.json").write_text(json.dumps({"camera_model": "PINHOLE", "frames": records},
                                                    indent=2) + "\n")
    if args.omega_src and Path(args.omega_src).is_dir():
        om_src = Path(args.omega_src)
        om_dst = case / "geometry" / "omega"
        for sub in ("depth", "confidence"):
            (om_dst / sub).mkdir(parents=True, exist_ok=True)
            for npy in (om_src / sub).glob("*.npy"):
                shutil.copyfile(npy, om_dst / sub / npy.name)
        for extra in ("cameras.json", "manifest.json", "frame_tokens.npy"):
            if (om_src / extra).is_file():
                shutil.copyfile(om_src / extra, om_dst / extra)
        print(f"[assemble] {case}: gt {len(records)} frames + omega from {om_src}")
    else:
        print(f"[assemble] {case}: gt {len(records)} frames (omega pending)")


# ----------------------------------------------------------------------------- infer

def host_commit_free_gib():
    """Free commit charge (GiB): the streaming checkpoint load dies with an access
    violation when the page file is exhausted, so gate on commit, not just VRAM."""
    import ctypes
    perf = type("PERF", (ctypes.Structure,),
                {"_fields_": [("cb", ctypes.c_ulong),
                              ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                              ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                              ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                              ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                              ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                              ("HandleCount", ctypes.c_ulong), ("ProcessCount", ctypes.c_ulong),
                              ("ThreadCount", ctypes.c_ulong)]})()
    perf.cb = ctypes.sizeof(perf)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(perf), perf.cb)
    return (perf.CommitLimit - perf.CommitTotal) * perf.PageSize / 2**30


def wait_for_vram(min_free_gib, timeout_s, poll_s=30.0, min_commit_gib=8.0):
    import torch
    deadline = time.time() + timeout_s
    while True:
        free, total = torch.cuda.mem_get_info()
        commit_free = host_commit_free_gib()
        if free >= min_free_gib * 2**30 and commit_free >= min_commit_gib:
            print(f"[gpu] free {free / 2**30:.2f}/{total / 2**30:.2f} GiB | "
                  f"commit free {commit_free:.1f} GiB -> proceed", flush=True)
            return
        if time.time() > deadline:
            raise SystemExit(f"[gpu] vram {free / 2**30:.2f} GiB / commit {commit_free:.1f} GiB "
                             f"still short after {timeout_s:.0f}s; aborting")
        print(f"[gpu] vram {free / 2**30:.2f} GiB (need {min_free_gib}), commit "
              f"{commit_free:.1f} GiB (need {min_commit_gib}); waiting {poll_s:.0f}s ...",
              flush=True)
        time.sleep(poll_s)


def cmd_infer(args):
    import random
    import torch
    from vggt_omega_geometry import (file_sha256, measure_pixel_affine, pick_aggregator_dtype,
                                     pick_device, read_size)

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "geometry" / "omega"
    if (out / "cameras.json").is_file() and not args.force:
        print(f"[skip] {out} already exported")
        return
    paths = [p for p in sorted(run_dir.joinpath("images").iterdir())
             if p.suffix.lower() in IMG_EXTS]
    exclude = set(args.exclude.split(",")) if args.exclude else set()
    paths = [p for p in paths if p.stem not in exclude]
    if len(paths) < 2:
        raise SystemExit("need >= 2 images after exclusion")
    if args.image_resolution % 16:
        raise SystemExit("--image-resolution must be divisible by 16")
    ckpt = Path(args.checkpoint)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = pick_device(args.device)
    if device == "cuda":
        wait_for_vram(args.min_free_gib, args.wait_seconds)
    agg_dtype = pick_aggregator_dtype(args.aggregator_dtype, device)
    print(f"[plan] {len(paths)} frames -> {out} (device={device}, dtype={agg_dtype})", flush=True)

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    t0 = time.time()
    backbone_low_memory = agg_dtype != torch.float32
    use_autocast = (device == "cuda") and not backbone_low_memory
    model = VGGTOmega(autocast=use_autocast).eval()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.torch_stream_loader import load_state_dict_streaming
    print(f"[load] {load_state_dict_streaming(ckpt, model)} tensors copied", flush=True)
    model = model.to(device=device)
    if backbone_low_memory:
        model.aggregator.to(dtype=agg_dtype)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    load_seconds = time.time() - t0

    frames_str = [str(p) for p in paths]
    original_sizes = [read_size(p) for p in frames_str]
    images = load_and_preprocess_images(frames_str, image_resolution=args.image_resolution)
    model_hw = (int(images.shape[-1]), int(images.shape[-2]))
    images = images.to(device=device, dtype=agg_dtype) if backbone_low_memory else images.to(device)
    t_infer = time.time()
    with torch.inference_mode():
        predictions = model(images)
    infer_seconds = time.time() - t_infer
    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])

    def to_np(value):
        array = value.detach().float().cpu().numpy()
        return array[0] if array.ndim >= 4 and array.shape[0] == 1 else array

    depth = to_np(predictions["depth"])
    conf = to_np(predictions["depth_conf"])
    if conf.ndim == depth.ndim and conf.shape[-1] == 1:
        conf = conf[..., 0]
    extr = to_np(extrinsic)
    intr = to_np(intrinsic)

    out.mkdir(parents=True, exist_ok=True)
    for sub in ("depth", "confidence"):
        (out / sub).mkdir(exist_ok=True)
    cameras = []
    for index, (path, size) in enumerate(zip(paths, original_sizes)):
        stem = Path(path).stem
        np.save(out / "depth" / f"{stem}.npy", depth[index].astype(np.float32))
        np.save(out / "confidence" / f"{stem}.npy", conf[index].astype(np.float32))
        measured = measure_pixel_affine(str(path), model_hw)
        affine = measured["affine"]
        k_model = intr[index].astype(np.float64)
        k_original = np.array([
            [k_model[0, 0] / affine[0, 0], 0.0, (k_model[0, 2] - affine[0, 2]) / affine[0, 0]],
            [0.0, k_model[1, 1] / affine[1, 1], (k_model[1, 2] - affine[1, 2]) / affine[1, 1]],
            [0.0, 0.0, 1.0]])
        cameras.append({
            "frame": stem + Path(path).suffix, "file": Path(path).name,
            "original_width": int(size[0]), "original_height": int(size[1]),
            "model_width": model_hw[0], "model_height": model_hw[1],
            "K_model": k_model.tolist(), "K_original": k_original.tolist(),
            "T_world_to_cam": np.vstack([extr[index], [0.0, 0.0, 0.0, 1.0]]).astype(np.float64).tolist(),
            "pixel_affine_original_to_model": affine.tolist()})
    if args.save_tokens:
        tokens = to_np(predictions["camera_and_register_tokens"])  # (S, T, C)
        np.save(out / "frame_tokens.npy", tokens.mean(axis=1).astype(np.float32))
    sha, size_bytes = file_sha256(ckpt)
    peak = (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else 0.0
    manifest = {"backend": "vggt-omega", "exporter": "scripts/view_selection.py infer",
                "checkpoint": {"path": str(ckpt), "sha256": sha, "bytes": size_bytes},
                "device": device, "aggregator_dtype": str(agg_dtype), "heads_dtype": "float32",
                "image_resolution_requested": args.image_resolution,
                "model_width": model_hw[0], "model_height": model_hw[1],
                "excluded": sorted(exclude), "seed": args.seed,
                "frames": [p.name for p in paths],
                "points_world": "not exported (not needed for this experiment)",
                "timing_seconds": {"load": round(load_seconds, 2), "infer": round(infer_seconds, 2)},
                "peak_cuda_mem_mib": round(peak, 1)}
    (out / "cameras.json").write_text(json.dumps({"camera_model": "PINHOLE", "frames": cameras},
                                                 indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[ok] {len(paths)} frames | infer {infer_seconds:.1f}s | peak {peak:.0f} MiB -> {out}",
          flush=True)
    del model, images, predictions
    if device == "cuda":
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- eval

def eval_geometry(omega_dir, gt_dir, only_stems=None):
    """capture_eval.py's image-space math without fusion: train-only Sim(3) on the
    evaluated frames' camera centres, nearest regrid onto the GT K grid, foreground
    mask, abs-rel / normal (flat/edge) / RPE. Returns per-view rows + aggregates."""
    omega_dir, gt_dir = Path(omega_dir), Path(gt_dir)
    records = json.loads((gt_dir / "cameras.json").read_text())["frames"]
    if only_stems is not None:
        records = [f for f in records if Path(f["frame"]).stem in set(only_stems)]
    predictions = {f["frame"]: f for f in json.loads((omega_dir / "cameras.json").read_text())["frames"]}
    gts = np.asarray([f["T_world_to_cam"] for f in records])
    pts = np.asarray([predictions[f["frame"]]["T_world_to_cam"] for f in records])
    gc = -np.einsum("sji,sj->si", gts[:, :3, :3], gts[:, :3, 3])
    pc = -np.einsum("sji,sj->si", pts[:, :3, :3], pts[:, :3, 3])
    scale, rotation, translation = umeyama(pc, gc)
    aligned = pts.copy()
    aligned[:, :3, :3] = pts[:, :3, :3] @ rotation.T
    aligned[:, :3, 3] = scale * pts[:, :3, 3] - aligned[:, :3, :3] @ translation

    rows = []
    rel_all, ang_all, flat_all, edge_all = [], [], [], []
    for index, frame in enumerate(records):
        stem = Path(frame["frame"]).stem
        reference = np.load(gt_dir / "depth" / f"{stem}.npy")
        foreground = np.load(gt_dir / "foreground" / f"{stem}.npy")
        raw = np.load(omega_dir / "depth" / f"{stem}.npy").squeeze() * scale
        k = np.asarray(frame["K_model"])
        pk = np.asarray(predictions[frame["frame"]]["K_model"])
        yy, xx = np.indices(reference.shape, dtype=np.float32)
        u = (xx - k[0, 2]) / k[0, 0] * pk[0, 0] + pk[0, 2]
        v = (yy - k[1, 2]) / k[1, 1] * pk[1, 1] + pk[1, 2]
        d = np.where(foreground, regrid_depth(raw, u, v, "nearest"), 0).astype(np.float32)
        valid = (reference > 0) & (d > 0)
        rel = np.abs(d[valid] - reference[valid]) / reference[valid]
        gn, gv = depth_normals(reference, k)
        pn, pv = depth_normals(d, k)
        nv = gv & pv
        angle = normal_angle(gn, pn)
        row = {"stem": stem, "valid_px": int(valid.sum()),
               "abs_rel": float(rel.mean()), "abs_rel_median": float(np.median(rel)),
               "normal_deg": float(angle[nv].mean()),
               "normal_deg_flat": float(angle[gt_flat_mask(gn, gv) & nv].mean()),
               "normal_deg_edge": float(angle[gt_edge_mask(reference) & nv].mean())}
        rows.append(row)
        rel_all.append(rel); ang_all.append(angle[nv])
        flat_all.append(angle[gt_flat_mask(gn, gv) & nv]); edge_all.append(angle[gt_edge_mask(reference) & nv])

    # RPE over all unordered pairs (aligned prediction vs GT, source gauge).
    # Relative motion a->b in w2c convention: M = T_b @ inv(T_a); RPE = M_gt^-1 M_pred.
    rot_err, trans_ang, trans_len = [], [], []
    n = len(records)
    for i in range(n):
        for j in range(i + 1, n):
            mg = gts[j] @ np.linalg.inv(gts[i])
            mp = aligned[j] @ np.linalg.inv(aligned[i])
            rd = mg[:3, :3].T @ mp[:3, :3]
            cos = np.clip((np.trace(rd) - 1) / 2, -1, 1)
            rot_err.append(np.degrees(np.arccos(cos)))
            tg, tp = mg[:3, 3], mp[:3, 3]
            ng, np_ = np.linalg.norm(tg), np.linalg.norm(tp)
            if ng > 1e-9 and np_ > 1e-9:
                trans_ang.append(np.degrees(np.arccos(np.clip(tg @ tp / (ng * np_), -1, 1))))
                trans_len.append(abs(np_ - ng) / ng)
    agg = {"frames": len(records),
           "abs_rel": summarize(np.concatenate(rel_all)),
           "normal_deg": summarize(np.concatenate(ang_all)),
           "normal_deg_flat": summarize(np.concatenate(flat_all)),
           "normal_deg_edge": summarize(np.concatenate(edge_all)),
           "rpe_rot_deg": summarize(np.asarray(rot_err)),
           "rpe_trans_angle_deg": summarize(np.asarray(trans_ang)),
           "rpe_trans_len_rel": summarize(np.asarray(trans_len)),
           "sim3_scale": float(scale)}
    return {"per_view": rows, "aggregate": agg}


def cmd_eval(args):
    only = args.frames.split(",") if args.frames else None
    result = eval_geometry(args.omega, args.gt, only)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    agg = result["aggregate"]
    print(json.dumps({"frames": agg["frames"], "abs_rel": agg["abs_rel"]["mean"],
                      "normal_deg": agg["normal_deg"]["mean"],
                      "normal_flat": agg["normal_deg_flat"]["mean"],
                      "normal_edge": agg["normal_deg_edge"]["mean"],
                      "rpe_rot_deg": agg["rpe_rot_deg"]["mean"]}, indent=2))


# ----------------------------------------------------------------------------- score

def cmd_score(args):
    """Deployable per-view signals: image quality, confidence, view redundancy from
    PREDICTED poses, cross-view depth consistency, and optional frame-token similarity."""
    import cv2
    from PIL import Image
    run_dir = Path(args.run_dir)
    omega = Path(args.omega) if args.omega else run_dir / "geometry" / "omega"
    frames = json.loads((omega / "cameras.json").read_text())["frames"]
    names = [Path(f["frame"]).stem for f in frames]
    depths = [np.load(omega / "depth" / f"{s}.npy").squeeze() for s in names]
    confs = [np.load(omega / "confidence" / f"{s}.npy").squeeze() for s in names]
    ts = np.asarray([f["T_world_to_cam"] for f in frames])
    centres = -np.einsum("sji,sj->si", ts[:, :3, :3], ts[:, :3, 3])
    unit = centres / np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-12)
    ang = angle_matrix(unit, unit)
    np.fill_diagonal(ang, np.inf)
    baseline = np.linalg.norm(centres[:, None] - centres[None], axis=-1)
    np.fill_diagonal(baseline, np.inf)

    tokens_path = omega / "frame_tokens.npy"
    token_cos = None
    if tokens_path.is_file():
        tok = np.load(tokens_path).astype(np.float64)
        tok /= np.maximum(np.linalg.norm(tok, axis=1, keepdims=True), 1e-12)
        sim = tok @ tok.T
        np.fill_diagonal(sim, -np.inf)
        token_cos = sim.max(axis=1)

    scores = {}
    height, width = depths[0].shape
    step = max(1, min(height, width) // 64)
    for i, stem in enumerate(names):
        img_path = run_dir / "images" / f"{stem}{Path(frames[i]['frame']).suffix}"
        with Image.open(img_path) as im:
            rgba = np.asarray(im.convert("RGBA"))
        # Quality stats are restricted to the photographed object when an alpha mask
        # exists (the lego renders are transparent-bg; compositing on white would make
        # "overexposure" measure the background, not the image content).
        has_alpha = rgba.shape[-1] == 4 and (rgba[..., 3] < 250).any()
        fg = rgba[..., 3] > 127 if has_alpha else np.ones(rgba.shape[:2], bool)
        comp = rgba[..., :3].astype(np.float32)
        gray = cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        lum = comp.mean(axis=-1) / 255.0
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        reproj = []
        d = depths[i][::step, ::step]
        k = np.asarray(frames[i]["K_model"])
        rows, cols = np.mgrid[0:d.shape[0], 0:d.shape[1]] * step
        cam = np.stack([(cols - k[0, 2]) / k[0, 0] * d, (rows - k[1, 2]) / k[1, 1] * d, d], -1).reshape(-1, 3)
        world = (ts[i][:3, :3].T @ (cam - ts[i][:3, 3]).T).T
        for j in range(len(names)):
            if i == j:
                continue
            cs = (ts[j][:3, :3] @ world.T).T + ts[j][:3, 3]
            z = cs[:, 2]
            kj = np.asarray(frames[j]["K_model"])
            with np.errstate(divide="ignore", invalid="ignore"):
                uf = cs[:, 0] / z * kj[0, 0] + kj[0, 2]
                vf = cs[:, 1] / z * kj[1, 1] + kj[1, 2]
            inside = (z > 1e-6) & (uf >= 0) & (uf < width) & (vf >= 0) & (vf < height)
            if inside.sum() < 32:
                continue
            zref = depths[j][np.clip(np.rint(vf[inside]).astype(int), 0, height - 1),
                             np.clip(np.rint(uf[inside]).astype(int), 0, width - 1)]
            reproj.append(float((np.abs(z[inside] - zref) <= 0.05 * z[inside]).mean()))
        scores[stem] = {
            "laplacian_var": float(lap[fg].var()),
            "exposure_over_frac": float((lum[fg] > 250 / 255).mean()),
            "exposure_under_frac": float((lum[fg] < 5 / 255).mean()),
            "mean_luminance": float(lum[fg].mean()),
            "conf_mean": float(confs[i].mean()), "conf_p10": float(np.percentile(confs[i], 10)),
            "min_angular_deg_pred": float(ang[i].min()),
            "min_baseline_rel": float(baseline[i].min() / np.median(baseline[np.isfinite(baseline)])),
            "reproj_inlier_mean": float(np.mean(reproj)) if reproj else None,
            "reproj_inlier_min": float(np.min(reproj)) if reproj else None,
            "token_cos_max": float(token_cos[i]) if token_cos is not None else None}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2) + "\n")
    print(f"[score] {len(scores)} views -> {out}")


# ----------------------------------------------------------------------------- loo

def cmd_loo(args):
    """Leave-one-out over the views of one base run dir.

    Phase run:      infer S (if missing), then for every stem infer S\\{i} and eval on
                    the retained 31 views. Sequential; resume-safe; VRAM-gated.
    Phase joint:    drop a flagged set H at once, plus --draws seeded random sets of the
                    SAME size (equal-count random control for joint deletion). Each run
                    is evaluated on the retained core (anchor) views.
    Phase tabulate: read all per-run eval JSONs, build the Delta table, and rank-
                    correlate deployable scores with the GT marginal contribution.
    """
    base = Path(args.base)
    gt = Path(args.gt)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    frames = json.loads((base / "geometry" / "omega" / "cameras.json").read_text())["frames"]
    stems = [Path(f["frame"]).stem for f in frames]

    if args.phase == "run":
        todo = [s for s in stems if not (work / f"minus_{s}.json").is_file()]
        print(f"[loo] {len(stems)} views, {len(todo)} deletions to run")
        for s in todo:
            out = work / f"minus_{s}" / "omega"
            cmd = [sys.executable, str(Path(__file__).resolve()), "infer",
                   "--run-dir", str(base), "--out", str(out), "--checkpoint", str(args.checkpoint),
                   "--exclude", s, "--min-free-gib", str(args.min_free_gib),
                   "--wait-seconds", str(args.wait_seconds)]
            print(f"[loo] --- drop {s} ---", flush=True)
            subprocess.run(cmd, check=True)
            eval_out = work / f"minus_{s}.json"
            retained = [x for x in stems if x != s]
            result = eval_geometry(out, gt, retained)
            eval_out.write_text(json.dumps(result, indent=2) + "\n")
            print(f"[loo] {s}: abs_rel {result['aggregate']['abs_rel']['mean']:.5f} "
                  f"normal {result['aggregate']['normal_deg']['mean']:.3f}", flush=True)
        return

    if args.phase == "joint":
        # H from --joint-h (comma stems); random controls: --draws seeded sets of |H|.
        harmful = [s for s in args.joint_h.split(",") if s]
        anchor = [s for s in (args.anchor_stems.split(",") if args.anchor_stems else [])]
        rng = np.random.default_rng(args.seed)
        sets = {"drop_flagged": harmful}
        for k in range(args.draws):
            sets[f"drop_random_{k}"] = sorted(rng.choice(stems, size=len(harmful),
                                                         replace=False).tolist())
        full = eval_geometry(base / "geometry" / "omega", gt, stems)
        full_view = {r["stem"]: r for r in full["per_view"]}
        rows = []
        for name, drop in sets.items():
            tag = "_".join(sorted(drop))[:60]
            out = work / f"joint_{name}" / "omega"
            if not (work / f"joint_{name}.json").is_file():
                cmd = [sys.executable, str(Path(__file__).resolve()), "infer",
                       "--run-dir", str(base), "--out", str(out),
                       "--checkpoint", str(args.checkpoint), "--exclude", ",".join(drop),
                       "--min-free-gib", str(args.min_free_gib),
                       "--wait-seconds", str(args.wait_seconds)]
                print(f"[loo-joint] --- {name}: drop {drop} ---", flush=True)
                subprocess.run(cmd, check=True)
                retained = [x for x in stems if x not in set(drop)]
                result = eval_geometry(out, gt, retained)
                (work / f"joint_{name}.json").write_text(json.dumps(result, indent=2) + "\n")
            result = json.loads((work / f"joint_{name}.json").read_text())
            retained = [x for x in stems if x not in set(drop)]
            ref = [x for x in retained if x in set(anchor)] or retained
            minus_view = {r["stem"]: r for r in result["per_view"]}
            rows.append({"set": name, "dropped": drop, "reference": "retained_core",
                         "n_ref": len(ref),
                         "delta_abs_rel": float(np.mean([full_view[x]["abs_rel"] for x in ref])
                                                - np.mean([minus_view[x]["abs_rel"] for x in ref])),
                         "delta_normal_deg": float(np.mean([full_view[x]["normal_deg"] for x in ref])
                                                   - np.mean([minus_view[x]["normal_deg"]
                                                              for x in ref]))})
        flagged = [r for r in rows if r["set"] == "drop_flagged"][0]
        randoms = [r for r in rows if r["set"].startswith("drop_random")]
        report = {"flagged": flagged, "random_controls": randoms,
                  "random_delta_mean": float(np.mean([r["delta_abs_rel"] for r in randoms])),
                  "random_delta_std": float(np.std([r["delta_abs_rel"] for r in randoms])),
                  "note": "Delta = E(S) - E(S\\set) on retained core views; Delta > 0 = removed "
                          "views were net negative contributors"}
        (work / "joint_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return

    # tabulate
    full = eval_geometry(base / "geometry" / "omega", gt, stems)
    (work / "full.json").write_text(json.dumps(full, indent=2) + "\n")
    full_view = {r["stem"]: r for r in full["per_view"]}
    anchor = set(args.anchor_stems.split(",")) if args.anchor_stems else set()
    rows = []
    for s in stems:
        path = work / f"minus_{s}.json"
        if not path.is_file():
            continue
        minus = json.loads(path.read_text())
        retained = [x for x in stems if x != s]
        full_ref = [full_view[x] for x in retained]
        minus_view = {r["stem"]: r for r in minus["per_view"]}
        delta_rel = (np.mean([r["abs_rel"] for r in full_ref])
                     - np.mean([minus_view[x]["abs_rel"] for x in retained]))
        delta_nrm = (np.mean([r["normal_deg"] for r in full_ref])
                     - np.mean([minus_view[x]["normal_deg"] for x in retained]))
        ref_a = [x for x in retained if x in anchor]
        delta_rel_a = (np.mean([full_view[x]["abs_rel"] for x in ref_a])
                       - np.mean([minus_view[x]["abs_rel"] for x in ref_a])) if ref_a else None
        rows.append({"dropped": s, "is_core": s in anchor,
                     "delta_abs_rel": float(delta_rel), "delta_normal_deg": float(delta_nrm),
                     "delta_abs_rel_core_ref": float(delta_rel_a) if ref_a else None,
                     "full_abs_rel_on_retained": float(np.mean([r["abs_rel"] for r in full_ref])),
                     "minus_abs_rel_on_retained": float(np.mean([minus_view[x]["abs_rel"]
                                                                 for x in retained]))})
    deltas = np.array([r["delta_abs_rel"] for r in rows])
    rng = np.random.default_rng(args.seed)
    boot = [float(np.mean(rng.choice(deltas, size=len(deltas), replace=True))) for _ in range(2000)]
    report = {"protocol": "Delta_i = E(S) - E(S\\{i}) on the retained 31 views; "
                          "Delta > 0 means the dropped view was a net NEGATIVE contributor; "
                          "the 31 deletions are the exhaustive k=1 distribution, so the "
                          "equal-count random control is their own mean/spread (bootstrap CI)",
              "n_views": len(stems), "n_deletions": len(rows),
              "delta_abs_rel": {"mean": float(deltas.mean()), "std": float(deltas.std()),
                                "min": float(deltas.min()), "max": float(deltas.max()),
                                "bootstrap_mean_ci95": [float(np.percentile(boot, 2.5)),
                                                        float(np.percentile(boot, 97.5))]},
              "rows": sorted(rows, key=lambda r: -r["delta_abs_rel"])}
    if args.scores and Path(args.scores).is_file():
        from scipy.stats import spearmanr
        scores = json.loads(Path(args.scores).read_text())
        corr = {}
        for key in ("laplacian_var", "exposure_over_frac", "exposure_under_frac",
                    "mean_luminance", "conf_mean", "conf_p10", "min_angular_deg_pred",
                    "min_baseline_rel", "reproj_inlier_mean", "reproj_inlier_min",
                    "token_cos_max"):
            pairs = [(scores[r["dropped"]][key], r["delta_abs_rel"])
                     for r in rows if r["dropped"] in scores and scores[r["dropped"]][key] is not None]
            if len(pairs) >= 8:
                rho, p = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
                corr[key] = {"spearman_rho": float(rho), "p_value": float(p), "n": len(pairs)}
        report["score_correlations_vs_delta"] = corr
    (work / "loo_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["delta_abs_rel"], indent=2))
    if "score_correlations_vs_delta" in report:
        print(json.dumps(report["score_correlations_vs_delta"], indent=2))


# ----------------------------------------------------------------------------- viz

def cmd_viz(args):
    from PIL import Image
    import cv2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = Path(args.gt)
    stems = args.stems.split(",")
    omegas = [Path(p) for p in args.omegas.split(",")]
    gt_records = json.loads((gt / "cameras.json").read_text())["frames"]
    # Protocol scale per omega dir: train-only Sim(3) over ALL GT frames (same rule
    # as capture_eval/eval), so depth panels share the GT gauge.
    scales = []
    for omega in omegas:
        preds = {f["frame"]: f for f in json.loads((omega / "cameras.json").read_text())["frames"]}
        gts = np.asarray([f["T_world_to_cam"] for f in gt_records])
        pts = np.asarray([preds[f["frame"]]["T_world_to_cam"] for f in gt_records])
        gc = -np.einsum("sji,sj->si", gts[:, :3, :3], gts[:, :3, 3])
        pc = -np.einsum("sji,sj->si", pts[:, :3, :3], pts[:, :3, 3])
        scale, _, _ = umeyama(pc, gc)
        scales.append(scale)
    print(f"[viz] sim3 scales: {[round(s, 4) for s in scales]}")
    rows_png = []
    for stem in stems:
        reference = np.load(gt / "depth" / f"{stem}.npy")
        foreground = np.load(gt / "foreground" / f"{stem}.npy")
        imgs = [np.asarray(Image.open(Path(p) / f"{stem}.png").convert("RGBA"))
                for p in args.image_dirs.split(",")]
        rgba = imgs[0]
        comp = (rgba[..., :3].astype(np.float32) * (rgba[..., 3:4] / 255.0)
                + 255 * (1 - rgba[..., 3:4] / 255.0)).astype(np.uint8)
        if comp.shape[:2] != reference.shape:
            comp = cv2.resize(comp, (reference.shape[1], reference.shape[0]),
                              interpolation=cv2.INTER_AREA)
        lo = reference[foreground].min() if foreground.any() else 0
        hi = reference[foreground].max() if foreground.any() else 1

        def depth_panel(d):
            vis = cv2.applyColorMap(np.uint8(np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1) * 255),
                                    cv2.COLORMAP_TURBO)
            vis[~foreground] = 24
            return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        tiles = [comp, depth_panel(reference)]
        for omega, scale in zip(omegas, scales):
            d = np.load(omega / "depth" / f"{stem}.npy").squeeze() * scale
            if d.shape != reference.shape:
                d = cv2.resize(d.astype(np.float32), (reference.shape[1], reference.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            d = np.where(foreground, d, 0)
            err = np.zeros_like(d)
            valid = (reference > 0) & (d > 0)
            err[valid] = np.abs(d[valid] - reference[valid]) / reference[valid]
            evis = cv2.applyColorMap(np.uint8(np.clip(err / 0.05, 0, 1) * 255), cv2.COLORMAP_TURBO)
            evis[~valid] = 96
            tiles += [depth_panel(d), cv2.cvtColor(evis, cv2.COLOR_BGR2RGB)]
        rows_png.append(np.concatenate(tiles, axis=1))
    panel = np.concatenate(rows_png, axis=0)
    Image.fromarray(panel).save(out / args.name)
    print(f"[viz] {out / args.name} {panel.shape}")


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("select")
    p.add_argument("--dataset", default=str(DATASET))
    p.add_argument("--dataset-zip", default=str(DATASET_ZIP))
    p.add_argument("--anchor-images", default=str(ANCHOR_IMAGES))
    p.add_argument("--eth3d-root", default=str(ETH3D_ROOT))
    p.add_argument("--out-root", default=str(VIEWSEL))
    p.add_argument("--n-candidate", type=int, default=16)
    p.set_defaults(fn=cmd_select)

    p = sub.add_parser("build")
    p.add_argument("--dataset", default=str(DATASET))
    p.add_argument("--dataset-zip", default=str(DATASET_ZIP))
    p.add_argument("--eth3d-root", default=str(ETH3D_ROOT))
    p.add_argument("--out-root", default=str(VIEWSEL))
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("assemble")
    p.add_argument("--out-root", default=str(VIEWSEL))
    p.add_argument("--case", required=True)
    p.add_argument("--gt-src", required=True)
    p.add_argument("--gt-stems", default=None, help="comma-separated subset; default all")
    p.add_argument("--omega-src", default=None)
    p.set_defaults(fn=cmd_assemble)

    p = sub.add_parser("infer")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--exclude", default=None, help="comma-separated stems to drop")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--aggregator-dtype", choices=["auto", "fp32", "bfloat16"], default="auto")
    p.add_argument("--image-resolution", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-free-gib", type=float, default=5.0)
    p.add_argument("--wait-seconds", type=float, default=3600)
    p.add_argument("--save-tokens", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_infer)

    p = sub.add_parser("eval")
    p.add_argument("--omega", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--frames", default=None, help="comma-separated stems; default all GT frames")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("score")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--omega", default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("loo")
    p.add_argument("--base", required=True, help="run dir with the full 32-view omega export")
    p.add_argument("--gt", required=True, help="GT dir covering all base views")
    p.add_argument("--work", required=True)
    p.add_argument("--phase", choices=["run", "joint", "tabulate"], required=True)
    p.add_argument("--joint-h", default=None, help="comma stems flagged for joint deletion")
    p.add_argument("--draws", type=int, default=6, help="random equal-count control draws")
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--anchor-stems", default=None, help="comma-separated core-view stems")
    p.add_argument("--scores", default=None, help="score JSON for the Spearman analysis")
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--min-free-gib", type=float, default=5.0)
    p.add_argument("--wait-seconds", type=float, default=3600)
    p.set_defaults(fn=cmd_loo)

    p = sub.add_parser("viz")
    p.add_argument("--gt", required=True)
    p.add_argument("--omegas", required=True, help="comma-separated omega dirs (compared)")
    p.add_argument("--image-dirs", required=True, help="comma-separated dirs holding <stem>.png (first is shown)")
    p.add_argument("--stems", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="panel.png")
    p.set_defaults(fn=cmd_viz)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
