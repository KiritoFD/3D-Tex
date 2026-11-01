# Selected upstream source provenance (2026-11-01)

- **`vggt/`** — [original VGGT](https://github.com/facebookresearch/vggt) pinned in `configs/upstreams.json`. Includes original inference + COLMAP export, NOT VGGT-Ω.
- **`2dsugar/`** — [2D-SuGaR](https://github.com/prajwalcr/2d-sugar), preserved **only as historical/reference vendor**; it retains its Gaussian/2DGS dependencies. Not part of new mainline.
- **`3dgs/`** — [gsplat](https://github.com/nerfstudio-project/gsplat) **removed from current tree**. Retrieve earlier versions via the Git commit history if needed, but do not re-vendor it.
- **Planned**: [VGGT-Ω](https://github.com/facebookresearch/vggt-omega) is separately pinned and minimally vendored **after checkpoint/access/license review and adapter tests**. [ExMesh](https://github.com/Fan-Treasure/ExMesh), [GeoSVR](https://github.com/Fictionarry/GeoSVR), [xatlas](https://github.com/jpcy/xatlas) are not yet incorporated.

Current `scripts/vendor_sources.py` includes only `vggt` and `2dsugar` and excludes nested `.git`/submodules in the *resulting repository*. 2D-SuGaR requires fetching its CUDA submodule source during vendoring, but no Git submodule is committed. Licenses are retained in vendored directories.

```bash
# Re-vendor existing pinned historical/runtime sources only.
python3 scripts/vendor_sources.py
python3 -m unittest discover -s tests -v
```

No large source repository clones, model weights, captured dataset images, or raw paper PDF binaries should be committed. Check upstream code and checkpoint licenses before reproduction/commercial usage.

The `configs/vendored_manifest.json` lists the retained pinned sources. Old 3DGS provenance remains recoverable via past commits (e.g., commit `b7f0c78`).
