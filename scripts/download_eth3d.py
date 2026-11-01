#!/usr/bin/env python3
"""Download a small, reproducible ETH3D validation suite; keep held-out views isolated.

Official ETH3D high-resolution multi-view training archive URLs:
https://www.eth3d.net/datasets
Requires 7z/7zz/7za. No Python image/GPU dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from common import ROOT, scene_path

BASE_URL = "https://www.eth3d.net/data/"
SCENES = {
    "pipes": {"views": 14, "images_gb": 0.1, "eval_gb": 0.1},
    "relief": {"views": 31, "images_gb": 0.3, "eval_gb": 0.2},
    "kicker": {"views": 31, "images_gb": 0.4, "eval_gb": 0.2},
}
DEFAULT_SCENES = ("pipes", "relief", "kicker")
GIB = 1024**3


def archive_names(scene: str, geometry: bool) -> list[str]:
    if scene not in SCENES:
        raise ValueError(f"Unknown ETH3D scene: {scene}")
    names = [f"{scene}_dslr_undistorted.7z"]
    if geometry:
        names.append(f"{scene}_dslr_scan_eval.7z")
    return names


def find_7z() -> str:
    binary = next((shutil.which(n) for n in ("7z", "7zz", "7za") if shutil.which(n)), None)
    if not binary:
        raise RuntimeError("7z missing: sudo apt-get install p7zip-full")
    return binary


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def download_archive(url: str, dest: Path, max_bytes: int) -> None:
    """Resumable HTTP downloader with hard byte cap and no silent error pages."""
    if dest.is_file():
        if not dest.stat().st_size or dest.stat().st_size > max_bytes:
            raise RuntimeError(f"Invalid cached archive size: {dest}")
        print(f"[cache] {dest.name}: {dest.stat().st_size / GIB:.3f} GiB")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(4):
        pos = part.stat().st_size if part.exists() else 0
        if pos >= max_bytes:
            raise RuntimeError(f"Download byte budget exceeded for {url}")
        headers = {"User-Agent": "3D-Tex-ETH3D-dataset-fetch/1.0"}
        if pos:
            headers["Range"] = f"bytes={pos}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                status = response.status
                if pos and status != 206:
                    part.unlink()
                    pos = 0
                size_hint = response.headers.get("Content-Length")
                if size_hint and pos + int(size_hint) > max_bytes:
                    raise RuntimeError(f"Archive exceeds max-byte budget: {url}")
                if response.headers.get("Content-Type", "").startswith("text/html"):
                    raise RuntimeError(f"Received HTML rather than 7z: {url}")
                with part.open("ab" if pos else "wb") as output:
                    size = pos
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > max_bytes:
                            raise RuntimeError(f"Download byte budget exceeded: {url}")
                        output.write(chunk)
                if size_hint and size - pos != int(size_hint):
                    raise IOError(f"Incomplete HTTP body for {url}")
            part.replace(dest)
            print(f"[downloaded] {dest.name}: {dest.stat().st_size / GIB:.3f} GiB")
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError, IOError) as e:
            if attempt == 3:
                raise RuntimeError(f"Download failed: {url}: {e}") from e
            time.sleep(min(2 ** attempt, 8))


def extract_archives(archives: list[Path], target: Path, sevenzip: str) -> None:
    for archive in archives:
        subprocess.run([sevenzip, "t", str(archive)], stdout=subprocess.DEVNULL, check=True)
        subprocess.run([sevenzip, "x", "-y", f"-o{target}", str(archive)],
                       stdout=subprocess.DEVNULL, check=True)


def read_gt_records(path: Path) -> list[dict]:
    """COLMAP images.txt: pose line, then optional observations line (may be empty)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    output: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) < 10:
            raise ValueError(f"Malformed COLMAP image pose line in {path}: {line[:120]}")
        try:
            int(tokens[0])
            qvec = [float(x) for x in tokens[1:5]]
            tvec = [float(x) for x in tokens[5:8]]
            camera_id = int(tokens[8])
        except ValueError as e:
            raise ValueError(f"Malformed COLMAP pose in {path}: {line[:120]}") from e
        name = Path(tokens[9]).name
        output.append({"image": name, "image_id": int(tokens[0]),
                       "qvec": qvec, "tvec": tvec, "camera_id": camera_id})
        # The next line is a COLMAP 2D-observations line, possibly empty.
        # Do not skip it before consuming it; it can contain many numbers.
        if i < len(lines):
            i += 1
    if len({x["image"] for x in output}) != len(output):
        raise ValueError("Duplicate image basename in GT camera records")
    if len(output) < 8:
        raise ValueError(f"Too few GT camera records in {path}: {len(output)}")
    return output


def choose_split(records: list[dict], stride: int) -> tuple[list[str], list[str]]:
    if stride < 3:
        raise ValueError("test stride must be >=3")
    names = sorted(x["image"] for x in records)
    test = [name for i, name in enumerate(names) if i % stride == stride - 1]
    train = [name for name in names if name not in set(test)]
    if len(test) < 2 or len(train) < 6:
        raise ValueError("Insufficient train/test views")
    return train, test


def locate_images(root: Path) -> dict[str, Path]:
    paths = [p for p in root.rglob("*")
             if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
             and any(part == "dslr_images_undistorted" for part in p.parts)]
    if not paths:
        raise FileNotFoundError(f"No ETH3D dslr_images_undistorted files under {root}")
    result = {}
    for path in paths:
        if path.name in result:
            raise ValueError(f"Duplicate image basename: {path.name}")
        result[path.name] = path
    return result


def locate_calibration(root: Path) -> tuple[Path, Path]:
    candidates = [p for p in root.rglob("cameras.txt")
                  if p.parent.name == "dslr_calibration_undistorted"]
    if len(candidates) != 1:
        raise ValueError(f"Expected one ETH3D undistorted calibration, found {candidates}")
    cameras = candidates[0]
    images = cameras.parent / "images.txt"
    if not images.is_file():
        raise FileNotFoundError(images)
    return cameras, images


def build_scene(extracted: Path, name: str, geometry: bool, stride: int,
                dest: Path, hashes: dict[str, str], force: bool = False) -> dict:
    """Atomic-ish staged scene creation. GT is NEVER copied into training/images."""
    if dest.exists():
        if (dest / "dataset_manifest.json").exists() and not force:
            print(f"[exists] {dest}; pass --force to replace")
            return json.loads((dest / "dataset_manifest.json").read_text())
        if not force:
            raise FileExistsError(f"Refusing to replace existing scene: {dest}")
    cameras, gt_poses = locate_calibration(extracted)
    images = locate_images(extracted)
    records = read_gt_records(gt_poses)
    train, test = choose_split(records, stride)
    missing = sorted(set(train + test) - set(images))
    if missing:
        raise ValueError(f"Images missing for official poses: {missing[:10]}")
    if geometry:
        gt_points = [p for p in extracted.rglob("*.ply")
                     if "scan_eval" in "/".join(p.parts) or "eval" in p.stem.lower()]
        if not gt_points:
            raise FileNotFoundError("No ETH3D scan_eval PLY in the geometry archive")
    else:
        gt_points = []
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eth3d-scene-", dir=dest.parent) as temp:
        tmp = Path(temp) / dest.name
        train_dir = tmp / "images"
        test_dir = tmp / "heldout" / "images"
        train_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)
        for basename in train:
            shutil.copy2(images[basename], train_dir / basename)
        for basename in test:
            shutil.copy2(images[basename], test_dir / basename)
        calib = tmp / "ground_truth" / "calibration"
        calib.mkdir(parents=True)
        shutil.copy2(cameras, calib / "cameras.txt")
        shutil.copy2(gt_poses, calib / "images.txt")
        (calib / "poses.json").write_text(json.dumps(records, indent=2) + "\n")
        if gt_points:
            gt = tmp / "ground_truth" / "scan_eval"
            gt.mkdir(parents=True)
            for file in gt_points:
                shutil.copy2(file, gt / file.name)
        manifest = {
            "dataset": "ETH3D high-resolution multi-view TRAINING",
            "scene": name,
            "reconstruction_scene": dest.name,
            "source": "https://www.eth3d.net/datasets",
            "undistorted": True,
            "poses_ground_truth": "ground_truth/calibration/poses.json",
            "images_used_by_vggt": "images/",
            "heldout_images_not_used_by_vggt": "heldout/images/",
            "evaluation_scan": "ground_truth/scan_eval/" if geometry else None,
            "split": {"method": "sorted filenames, every stride-th held out",
                      "test_stride": stride, "train": train, "test": test},
            "archives_sha256": hashes,
            "warning": "Do not use GT calibration/scan as input to VGGT or Gaussian training.",
        }
        (tmp / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)
    print(f"[prepared] {dest.name}: {len(train)} train, {len(test)} heldout, {len(gt_points)} GT PLY")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="ETH3D small (<10GB) reconstruction evaluation suite")
    parser.add_argument("--scenes", nargs="+", choices=sorted(SCENES), default=list(DEFAULT_SCENES))
    parser.add_argument("--no-geometry", action="store_true", help="Only images and GT camera poses; no GT scan")
    parser.add_argument("--test-stride", type=int, default=5)
    parser.add_argument("--max-download-gib", type=float, default=3.0,
                        help="Maximum total compressed archive bytes for selected scenes")
    parser.add_argument("--force", action="store_true", help="Replace prepared scene")
    parser.add_argument("--delete-archives", action="store_true", help="Delete downloaded 7z after extraction")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    archives = [x for name in args.scenes
                for x in archive_names(name, not args.no_geometry)]
    estimated = sum(
        (SCENES[name]["images_gb"] + (0 if args.no_geometry else SCENES[name]["eval_gb"]))
        for name in args.scenes
    )
    print(f"ETH3D scenes: {args.scenes}; official approximate compressed size: {estimated:.2f} GB")
    for file in archives:
        print(f"  {BASE_URL}{file}")
    if args.dry_run:
        return
    if args.max_download_gib <= 0 or args.test_stride < 3:
        parser.error("Invalid byte budget or test stride")
    sevenzip = find_7z()
    cache = ROOT / "data" / ".cache" / "eth3d"
    cache.mkdir(parents=True, exist_ok=True)
    budget = int(args.max_download_gib * GIB)
    for name in args.scenes:
        dest = scene_path("eth3d_" + name)
        if (dest / "dataset_manifest.json").exists() and not args.force:
            print(f"[exists] {dest}; skipping")
            continue
        paths = []
        for archive in archive_names(name, not args.no_geometry):
            target = cache / archive
            existing = sum(p.stat().st_size for p in cache.glob("*.7z") if p.is_file())
            if existing > budget:
                raise RuntimeError(f"Total cached archives exceed {args.max_download_gib:.2f} GiB")
            download_archive(BASE_URL + archive, target, budget - (existing - (target.stat().st_size if target.exists() else 0)))
            paths.append(target)
        if sum(p.stat().st_size for p in cache.glob("*.7z")) > budget:
            raise RuntimeError(f"Total cached archives exceed {args.max_download_gib:.2f} GiB")
        hashes = {p.name: digest(p) for p in paths}
        with tempfile.TemporaryDirectory(prefix="extract-", dir=cache) as tmp:
            stage = Path(tmp)
            extract_archives(paths, stage, sevenzip)
            build_scene(stage, name, not args.no_geometry, args.test_stride, dest, hashes, args.force)
        if args.delete_archives:
            for path in paths:
                path.unlink()
    print("Done. Train with: python3 scripts/run_vggt.py --scene eth3d_pipes --ba")


if __name__ == "__main__":
    main()
