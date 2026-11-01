#!/usr/bin/env python3
"""Clone pinned upstreams, vendor only reconstruction-critical code (no submodules).

Use on a trusted developer machine or the one-shot GitHub Actions workflow.
Does NOT install dependencies or execute upstream Python/CUDA code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "upstreams.json"
REQUIRED_SUBMODULES = [
    "gaussian_splatting_2d/submodules/diff-surfel-rasterization",
    "gaussian_splatting_2d/submodules/simple-knn",
    "gaussian_splatting/submodules/diff-gaussian-rasterization",
]
KEEP = {
    "vggt": [
        "vggt",
        "demo_colmap.py", "requirements.txt", "requirements_demo.txt",
        "LICENSE.txt", "README.md",
    ],
    "2dsugar": [
        "gaussian_splatting", "gaussian_splatting_2d",
        "sugar_extractors", "sugar_scene", "sugar_trainers",
        "sugar_utils", "gsplat_wrapper",
        "train.py", "train_sugar.py", "train_refined.py",
        "extract_mesh.py", "extract_refined_mesh_with_texture.py",
        "estimate_depth_normal_priors.py", "environment.yml",
        "LICENSE.md", "README.md",
    ],
}
# Trim galleries, demos, benchmark harnesses and UI, *not* runtime code.
SKIP_NAMES = {".git", ".gitmodules", "__pycache__", ".pytest_cache",
              "SIBR_viewers", "sugar_viewer", ".github", "tests", "docs",
              "assets", "profiling", "lint", "benchmarks", "scripts",
              "node_modules", ".DS_Store"}
# Preserve upstream runtime imports; strip only irrelevant demos and media.


def invoke(*argv: str, cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), cwd=cwd, check=True)


def clone(url: str, sha: str, dest: Path) -> None:
    invoke("git", "clone", "--quiet", "--depth", "1", url, str(dest))
    invoke("git", "fetch", "--quiet", "--depth", "1", "origin", sha, cwd=dest)
    invoke("git", "checkout", "--quiet", "--detach", sha, cwd=dest)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip()
    if head != sha:
        raise RuntimeError(f"Upstream pin mismatch for {url}: {head} != {sha}")


def ignored(src: str, names: list[str]) -> set[str]:
    return {name for name in names if name in SKIP_NAMES or name.endswith((".pyc", ".pyo"))}


def copy_selected(src: Path, dest: Path, name: str) -> None:
    if not src.is_dir():
        raise FileNotFoundError(src)
    dest.mkdir(parents=True, exist_ok=True)
    for rel in KEEP[name]:
        origin = src / rel
        if not origin.exists():
            raise FileNotFoundError(f"Required {name} path absent: {rel}")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if origin.is_dir():
            shutil.copytree(origin, target, ignore=ignored, dirs_exist_ok=True)
        elif origin.is_file():
            shutil.copy2(origin, target)
        else:
            raise RuntimeError(f"Unsupported upstream entry: {origin}")


def check_vendor(base: Path) -> None:
    needed = [
        "vggt/demo_colmap.py",
        "vggt/vggt/models/vggt.py",
        "2dsugar/train.py",
        "2dsugar/gaussian_splatting_2d/train.py",
        "2dsugar/sugar_extractors/refined_mesh.py",
    ] + ["2dsugar/" + path + "/setup.py" for path in REQUIRED_SUBMODULES]
    missing = [p for p in needed if not (base / p).is_file()]
    if missing:
        raise RuntimeError("Vendoring incomplete: " + ", ".join(missing))
    for sub in ["vggt", "2dsugar"]:
        if (base / sub / ".git").exists() or (base / sub / ".gitmodules").exists():
            raise RuntimeError(f"Nested git metadata leaked into {sub}")


def manifest_for(base: Path, upstreams: dict) -> dict:
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "source_mode": "vendored-allowlist-no-submodules",
                "upstreams": upstreams, "files": {}, "counts": {}}
    for name in KEEP:
        files = sorted(p for p in (base / name).rglob("*") if p.is_file())
        manifest["counts"][name] = len(files)
        manifest["files"][name] = {
            str(f.relative_to(base / name)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in files
        }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, help="Pre-existing pinned clones (for offline testing)")
    args = parser.parse_args()
    upstreams = json.loads(CONFIG.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="3dtex-vendor-") as tmp:
        temp = Path(tmp)
        src_root = args.source_root if args.source_root is not None else temp / "clone"
        stage = temp / "stage"
        if args.source_root is None:
            src_root.mkdir()
            for name, item in upstreams.items():
                clone(item["url"], item["sha"], src_root / name)
            sugar = src_root / "2dsugar"
            invoke("git", "submodule", "update", "--init", "--depth", "1",
                   *REQUIRED_SUBMODULES, cwd=sugar)
        else:
            if not src_root.is_dir():
                raise FileNotFoundError(src_root)
        for name in KEEP:
            copy_selected(src_root / name, stage / name, name)
        check_vendor(stage)
        manifest = manifest_for(stage, upstreams)
        # Stage everything before replacing old sources. Never partially vendor one backend.
        for name in KEEP:
            target = ROOT / name
            if target.exists():
                if target.is_symlink():
                    raise RuntimeError(f"Refusing to replace symlink: {target}")
                shutil.rmtree(target)
            shutil.move(str(stage / name), str(target))
        out = ROOT / "configs" / "vendored_manifest.json"
        out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("Vendored", manifest["counts"], "with pins", {k:v["sha"] for k,v in upstreams.items()})


if __name__ == "__main__":
    main()
