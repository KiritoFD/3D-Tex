#!/usr/bin/env python3
"""Fetch the pinned pretrained weights into the project cache (resumable, verified).

Nothing here needs GPU or torch: it only moves bytes. Wrappers call resolve() so a
local Windows laptop with no conda and a slow uplink can pre-stage weights once and
then run fully offline.

    python3 scripts/fetch_weights.py --list
    python3 scripts/fetch_weights.py --dry-run
    python3 scripts/fetch_weights.py                      # everything the pipeline needs
    python3 scripts/fetch_weights.py --only vggt-1b --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from common import ROOT

GIB = 1024**3
CACHE_DIR = Path(os.environ.get("3DTEX_WEIGHTS_DIR") or (ROOT / "data" / ".cache" / "models"))

# Each asset lists failover URLs (the second entry is the mainland China mirror).
# sha256 is optional: the first successful fetch records it into the sidecar.
ASSETS: dict[str, dict] = {
    "vggt-omega-512": {
        "label": "VGGT-Omega 1B @512 (CVPR 2026 successor of VGGT-1B; feeds the vggt-omega model code)",
        "target": "sanqilu376/VGGT-Omega/vggt_omega_1b_512.pt",
        "filename": "sanqilu376/VGGT-Omega/vggt_omega_1b_512.pt",
        "urls": [
            "https://www.modelscope.cn/models/sanqilu376/VGGT-Omega/resolve/master/vggt_omega_1b_512.pt",
            "https://huggingface.co/spaces/modelscope/VGGT-Omega/resolve/main/vggt_omega_1b_512.pt",
        ],
        "used_by": "scripts/run_vggt.py -> facebookresearch/vggt-omega inference",
    },
    "vggt-omega-256-text": {
        "label": "VGGT-Omega 1B 256 text-conditioned variant (optional)",
        "target": "sanqilu376/VGGT-Omega/vggt_omega_1b_256_text.pt",
        "filename": "sanqilu376/VGGT-Omega/vggt_omega_1b_256_text.pt",
        "urls": [
            "https://www.modelscope.cn/models/sanqilu376/VGGT-Omega/resolve/master/vggt_omega_1b_256_text.pt",
        ],
        "optional": True,
        "used_by": "alternative VGGT-Omega checkpoint",
    },
    "vggt-1b": {
        "label": "VGGT-1B feedforward reconstruction weights (legacy pinned upstream)",
        "target": "facebook/VGGT-1B/model.pt",
        "filename": "facebook/VGGT-1B/model.pt",
        "urls": [
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            "https://hf-mirror.com/facebook/VGGT-1B/resolve/main/model.pt",
        ],
        "optional": True,
        "used_by": "scripts/run_vggt.py -> vggt/demo_colmap.py",
    },
    "vggt-1b-ba": {
        "label": "VGGSfM tracking weights (only needed for the --ba path)",
        "target": "facebook/VGGSfM checkpoint",
        "filename": "facebook/VGGSfM/vggsfm_tracker_v1.pt",
        "urls": [
            "https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_tracker_v1.pt",
            "https://hf-mirror.com/facebook/VGGSfM/resolve/main/vggsfm_tracker_v1.pt",
        ],
        "optional": True,
        "used_by": "vggt/demo_colmap.py --use_ba (predict_tracks)",
    },
}
DEFAULT_ONLY = [k for k, v in ASSETS.items() if not v.get("optional")]


def resolve(name: str) -> Path:
    """Path of a cached weight; raises with the exact fetch command if missing."""
    spec = ASSETS.get(name)
    if spec is None:
        raise KeyError(f"Unknown weight asset: {name}. Known: {', '.join(sorted(ASSETS))}")
    path = CACHE_DIR / spec["filename"]
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(
            f"{path} is missing. Run: python3 scripts/fetch_weights.py --only {name}"
        )
    return path


def sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "3D-Tex-weight-fetch/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            value = r.headers.get("Content-Length")
            return int(value) if value else None
    except Exception:
        return None


def fetch_one(url: str, dest: Path, expected: int | None, timeout: int, quiet: bool) -> int:
    """Append-to-.part download with Range resume. Returns total bytes on disk."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    pos = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "3D-Tex-weight-fetch/1.0"}
    if pos:
        headers["Range"] = f"bytes={pos}-"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        if pos and r.status != 206:
            pos = 0
            part.unlink(missing_ok=True)
        total = expected
        if r.headers.get("Content-Length"):
            total = pos + int(r.headers["Content-Length"])
        ctype = r.headers.get("Content-Type", "")
        if ctype.startswith("text/html"):
            raise RuntimeError(f"{url} returned HTML ({ctype}), not a weight file")
        mode = "ab" if pos else "wb"
        written = pos
        last_report = time.time()
        with part.open(mode) as out:
            while True:
                chunk = r.read(256 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
                if not quiet and time.time() - last_report > 15:
                    done = written / 2**20
                    rate = done / max(1e-6, (time.time() - start_ref[0]))
                    of = f"/{total / 2**20:.0f} MiB" if total else ""
                    print(f"  {dest.name}: {done:.0f}{of} MiB ({rate:.2f} MiB/s)", flush=True)
                    last_report = time.time()
        if total and written < total:
            raise IOError(f"Incomplete body: {written} < {total}")
    part.replace(dest)
    return written


start_ref = [0.0]


def download(name: str, spec: dict, retries: int, timeout: int, quiet: bool) -> Path:
    dest = CACHE_DIR / spec["filename"]
    if dest.is_file() and dest.stat().st_size:
        note = sidecar(dest)
        cached = json.loads(note.read_text()) if note.is_file() else {}
        if cached.get("size") == dest.stat().st_size and (not cached.get("sha256") or cached["sha256"] == sha256(dest)):
            print(f"[cache] {dest.name}: {dest.stat().st_size / GIB:.3f} GiB verified")
            return dest
    expected = None
    for url in spec["urls"]:
        expected = remote_size(url)
        if expected:
            break
    if expected:
        print(f"[target] {dest.name}: {expected / GIB:.3f} GiB from {spec['urls'][0]}")
    last_error = None
    for attempt in range(retries):
        for url in spec["urls"]:
            start_ref[0] = time.time()
            try:
                fetch_one(url, dest, expected, timeout, quiet)
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError, IOError, RuntimeError, OSError) as exc:
                last_error = exc
                wait = min(2 ** attempt, 20)
                print(f"[retry {attempt + 1}/{retries}] {type(exc).__name__}: {str(exc)[:160]} (sleep {wait}s)", flush=True)
                time.sleep(wait)
            except Exception as exc:  # keep partial state, abort this URL
                last_error = exc
                print(f"[skip url] {url}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        else:
            raise RuntimeError(f"Failed to fetch {name}: {last_error}")
        if dest.is_file() and (not expected or dest.stat().st_size == expected):
            break
    digest = sha256(dest)
    if expected and dest.stat().st_size != expected:
        raise RuntimeError(f"{dest.name} size {dest.stat().st_size} != expected {expected}")
    sidecar(dest).write_text(json.dumps({
        "asset": name, "source": spec["target"], "path": str(dest),
        "size": dest.stat().st_size, "sha256": digest,
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "\n")
    print(f"[ok] {name}: {dest.stat().st_size / GIB:.3f} GiB sha256={digest[:16]}... -> {dest}")
    return dest


def main() -> None:
    global CACHE_DIR
    p = argparse.ArgumentParser(description="Stage pretrained weights into the Git-ignored project cache")
    p.add_argument("--only", nargs="+", choices=sorted(ASSETS), default=DEFAULT_ONLY,
                   help="Assets to fetch (default: everything non-optional)")
    p.add_argument("--cache-dir", default=None, help=f"Override cache root (default {CACHE_DIR})")
    p.add_argument("--retries", type=int, default=8)
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--verify-only", action="store_true", help="Hash existing files, download nothing")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if a.cache_dir:
        CACHE_DIR = Path(a.cache_dir).resolve()

    if a.list:
        for name, spec in ASSETS.items():
            state = "cached" if (CACHE_DIR / spec["filename"]).is_file() else "missing"
            flag = " (optional)" if spec.get("optional") else ""
            print(f"{name:12s} [{state}]{flag} {spec['label']}\n{'':14s}{spec['urls'][0]}")
        print(f"cache dir: {CACHE_DIR}")
        return

    for name in a.only:
        spec = ASSETS[name]
        dest = CACHE_DIR / spec["filename"]
        if a.dry_run:
            print(f"[dry-run] {name} -> {dest}\n          from {spec['urls'][0]}")
            continue
        if a.verify_only:
            if not dest.is_file():
                print(f"[missing] {name}: {dest}")
                continue
            print(f"[verify] {name}: size={dest.stat().st_size} sha256={sha256(dest)}")
            continue
        download(name, spec, a.retries, a.timeout, a.quiet)


if __name__ == "__main__":
    main()
