#!/usr/bin/env python3
"""Download selected original arXiv PDFs to a Git-ignored local cache.

CPU-only/stdlib; no papers are redistributed in the source repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from common import ROOT

PAPERS = {
    "vggt": ("2503.11651", "VGGT: Visual Geometry Grounded Transformer"),
    "omega": ("2605.15195", "VGGT-Omega"),
    "exmesh": ("2606.07288", "ExMesh"),
    "exmeshpp": ("2608.24109", "ExMesh++"),
    "geosvr": ("2509.18090", "GeoSVR"),
    "meshsplatting": ("2512.06818", "MeshSplatting"),
    "gaussian3d": ("2308.04079", "3D Gaussian Splatting"),
    "gsplat": ("2409.06765", "gsplat library"),
    "gaussian2d": ("2403.17888", "2D Gaussian Splatting"),
    "sugar": ("2311.12775", "SuGaR"),
    "directuv": ("2609.34651", "DirectUV"),
    "texgen": ("2411.14740", "TEXGen"),
    "pointuv": ("2308.10490", "Point-UV Diffusion"),
    "paint3d": ("2312.13913", "Paint3D"),
    "texfusion": ("2310.13772", "TexFusion"),
    "matlat": ("2512.17302", "MatLat"),
}
DEFAULT = ("omega", "exmesh", "geosvr", "matlat")
CHUNK = 1 << 20


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_pdf(path: Path) -> None:
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise ValueError(f"Not a PDF: {path}")


def download_one(slug: str, out: Path, byte_limit: int, force: bool = False) -> dict:
    if slug not in PAPERS:
        raise ValueError(f"Unknown paper: {slug}")
    arxiv, title = PAPERS[slug]
    url = f"https://arxiv.org/pdf/{arxiv}"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{slug}-{arxiv}.pdf"
    if dest.exists() and not force:
        check_pdf(dest)
        if dest.stat().st_size > byte_limit:
            raise RuntimeError(f"Total PDF budget exceeded by cached {dest}")
        print(f"[cache] {dest.name}")
    else:
        if byte_limit < 5:
            raise RuntimeError("PDF byte budget exhausted")
        part = dest.with_suffix(".pdf.part")
        part.unlink(missing_ok=True)
        req = urllib.request.Request(url, headers={
            "User-Agent": "3D-Tex-paper-references/1.0", "Accept": "application/pdf"
        })
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                hint = response.headers.get("Content-Length")
                if hint and int(hint) > byte_limit:
                    raise RuntimeError(f"PDF exceeds byte budget: {url}")
                if response.headers.get("Content-Type", "").lower().startswith("text/html"):
                    raise RuntimeError(f"Received HTML instead of PDF: {url}")
                used = 0
                with part.open("wb") as handle:
                    while chunk := response.read(CHUNK):
                        used += len(chunk)
                        if used > byte_limit:
                            raise RuntimeError(f"PDF exceeds byte budget: {url}")
                        handle.write(chunk)
            check_pdf(part)
            part.replace(dest)
            print(f"[downloaded] {dest.name}")
        finally:
            part.unlink(missing_ok=True)
    return {"slug": slug, "title": title, "arxiv": arxiv, "url": url,
            "filename": dest.name, "bytes": dest.stat().st_size, "sha256": checksum(dest)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache original arXiv PDFs (no GPU)")
    parser.add_argument("--papers", nargs="+", choices=sorted(PAPERS), default=list(DEFAULT))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace cached PDF")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "papers" / "pdf")
    parser.add_argument("--max-total-mib", type=float, default=200.0)
    args = parser.parse_args()
    if args.list:
        for slug, (arxiv, title) in PAPERS.items():
            print(f"{slug:12} {arxiv} {title}")
        return
    if args.max_total_mib <= 0:
        parser.error("--max-total-mib must be positive")
    wanted = list(dict.fromkeys(args.papers))
    for slug in wanted:
        print(f"https://arxiv.org/pdf/{PAPERS[slug][0]} -> {slug}")
    if args.dry_run:
        return
    budget = int(args.max_total_mib * 1024**2)
    rows = []
    for slug in wanted:
        row = download_one(slug, args.output, budget, args.force)
        budget -= row["bytes"]
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(rows)} PDF(s); manifest: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
