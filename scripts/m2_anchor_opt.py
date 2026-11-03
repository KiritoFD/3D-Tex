#!/usr/bin/env python3
"""M2 entry point and legacy bucket-renderer validation helpers.

Training uses mesh_opt.runner with explicit diagnostic/prediction provenance,
voxel-scaled offsets, averaged per-view gradients and a tolerance depth prior.
Run scripts/m2_anchor_opt.py --help; nvdiffrast training runs in WSL.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))


# ---------------- differentiable soft rasterizer (from the M1 spike) ----------------
def project(verts3d, k, t):
    cam = (t[:3, :3] @ verts3d.transpose(-1, -2)).transpose(-1, -2) + t[:3, 3]
    z = cam[..., 2]
    u = cam[..., 0] / z.clamp(min=1e-6) * k[0, 0] + k[0, 2]
    v = cam[..., 1] / z.clamp(min=1e-6) * k[1, 1] + k[1, 2]
    return torch.stack([u, v], -1), z


def build_buckets(uv, z, width, height, tile, sigma, cache_path=None, max_span=8):
    """Tile -> candidate triangle indices. Long sliver triangles can span
    thousands of tiles and explode host RAM, so triangles whose tile span
    exceeds max_span are kept in a global list under key (-1, -1) that is
    appended to every tile's candidates by pixels_by_tile."""
    if cache_path and Path(cache_path).is_file():
        blob = np.load(cache_path)
        return {tuple(int(x) for x in kk.split("_")): torch.from_numpy(blob[kk]).to(uv.device)
                for kk in blob.files}
    with torch.no_grad():
        lo = uv.min(dim=1).values - sigma
        hi = uv.max(dim=1).values + sigma
        in_front = (z > 1e-6).any(dim=1)
        tile_lo = torch.floor(lo / tile).long().clamp(0, width // tile - 1)
        tile_hi = torch.floor(hi / tile).long().clamp(0, height // tile - 1)
        span = tile_hi - tile_lo  # (F,2)
        large = in_front & (span.amax(dim=1) > max_span)
        buckets = {(-1, -1): torch.nonzero(large).squeeze(-1).cpu().numpy().astype(np.int64)}
        for i in torch.nonzero(in_front & ~large).squeeze(-1).tolist():
            for ty in range(tile_lo[i, 1].item(), tile_hi[i, 1].item() + 1):
                for tx in range(tile_lo[i, 0].item(), tile_hi[i, 0].item() + 1):
                    buckets.setdefault((tx, ty), []).append(i)
        # nearest-first ordering: with the max_cand cap in soft_render, the kept
        # subset must be the closest triangles, not an arbitrary index range
        zmin = z.min(dim=1).values.cpu().numpy()
        out = {}
        for k, v in buckets.items():
            v = np.asarray(v, dtype=np.int64)
            out[k] = v[np.argsort(zmin[v], kind="stable")]
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **{f"{k[0]}_{k[1]}": v for k, v in out.items()})
    return {k: torch.from_numpy(v).to(uv.device) for k, v in out.items()}


def soft_render(vertices, faces, buckets, k, t, pixels, sigma=1.5, max_cand=256,
                temperature=30.0):
    """Soft depth + alpha at arbitrary pixels (differentiable in vertices)."""
    tri = vertices[faces]
    uv, z = project(tri, k, t)
    depth_out = torch.zeros(len(pixels), device=vertices.device)
    alpha_out = torch.zeros(len(pixels), device=vertices.device)
    for key, sel_list in pixels_by_tile(pixels, buckets).items():
        cand, sel = sel_list
        if len(cand) > max_cand:
            cand = cand[:max_cand]
        p = pixels[sel]
        uv_c = uv[cand]
        z_c = z[cand]
        v0, v1, v2 = uv_c[:, 0], uv_c[:, 1], uv_c[:, 2]
        den = (v1[:, 1] - v2[:, 1]) * (v0[:, 0] - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (v0[:, 1] - v2[:, 1])
        den = torch.where(den.abs() < 1e-9, torch.full_like(den, 1e-9), den)
        w0 = ((v1[:, 1] - v2[:, 1]) * (p[:, None, 0] - v2[None, :, 0])
              + (v2[:, 0] - v1[:, 0]) * (p[:, None, 1] - v2[None, :, 1])) / den
        w1 = ((v2[:, 1] - v0[:, 1]) * (p[:, None, 0] - v2[None, :, 0])
              + (v0[:, 0] - v2[:, 0]) * (p[:, None, 1] - v2[None, :, 1])) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        d_edge = torch.stack([w0, w1, w2], -1).clamp(max=0.0).abs().amax(dim=-1)
        weight = torch.where(inside, torch.ones_like(d_edge),
                             torch.exp(-((d_edge * 20.0 * sigma / 1.5) ** 2)))
        z_b = w0 * z_c[None, :, 0] + w1 * z_c[None, :, 1] + w2 * z_c[None, :, 2]
        score = weight / z_b.clamp(min=1e-6)
        attn = torch.softmax(score * temperature, dim=1)
        depth_out[sel] = (attn * z_b).sum(dim=1)
        alpha_out[sel] = weight.amax(dim=1)
    return depth_out, alpha_out


def pixels_by_tile(pixels, buckets):
    """Group pixel indices by their tile, returning {tile: (cand, sel)} lists.
    Large-span triangles live under key (-1, -1) and are appended everywhere."""
    groups = {}
    glob = buckets.get((-1, -1))
    with torch.no_grad():
        tile = (pixels / 16).long()
        keys, inv = torch.unique(tile, dim=0, return_inverse=True)
        for gi, key in enumerate(keys.tolist()):
            sel = torch.nonzero(inv == gi).squeeze(-1)
            cand = buckets.get(tuple(key))
            if glob is not None and len(glob):
                cand = glob if cand is None else torch.cat([cand, glob])
            if cand is not None and len(sel):
                groups[tuple(key)] = (cand, sel)
    return groups


from mesh_opt.render import NvdRenderer  # compatibility for renderer validation


def main():
    from mesh_opt.runner import main as run
    run()


if __name__ == "__main__":
    main()
