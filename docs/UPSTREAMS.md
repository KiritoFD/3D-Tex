# Selected upstream source provenance (2026-11-12)

- **`vggt/`** — [original VGGT](https://github.com/facebookresearch/vggt) pinned in `configs/upstreams.json`. Includes original inference + COLMAP export, NOT VGGT-Ω.
- **`2dsugar/`** — [2D-SuGaR](https://github.com/prajwalcr/2d-sugar), preserved **only as historical/reference vendor**; it retains its Gaussian/2DGS dependencies. Not part of new mainline.
- **`3dgs/`** — [gsplat](https://github.com/nerfstudio-project/gsplat) **removed from current tree**. Retrieve earlier versions via the Git commit history if needed, but do not re-vendor it.
- **`third_party/supra_tex/`** — [SUPRA-Tex](https://github.com/Supra-3D/supra-tex) @ `cee4435652d0f23737bd28d5704e0fae9830b411`, vendored without `.git`, pin record in `third_party/supra_tex_pin.json`. Runtime only for Tex-T1 (fp32/bf16 texture completion arms); weights are fetched separately (`scripts/experiments/tex_t1_weights.sh`) and never committed.
- **`third_party/hunyuan3d/`** — [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan3D/Hunyuan3D-2.1) @ `82920d643c0dc2f7bfd7255f45f62d386edfe60c` (2025-10-17). Kept as an **upstream clone outside version control** (`.git` included, ~315 MB), listed here because Tex-T2 uses it as the paint baseline; the thin `bpy` stub lives in `third_party/stubs/`. Do not commit the clone.
- **Planned**: [VGGT-Ω](https://github.com/facebookresearch/vggt-omega) is separately pinned and minimally vendored **after checkpoint/access/license review and adapter tests**. [GeoSVR](https://github.com/Fictionarry/GeoSVR), [xatlas](https://github.com/jpcy/xatlas) are not yet incorporated.

Current `scripts/vendor_sources.py` includes only `vggt` and `2dsugar` and excludes nested `.git`/submodules in the *resulting repository*. 2D-SuGaR requires fetching its CUDA submodule source during vendoring, but no Git submodule is committed. Licenses are retained in vendored directories.

```bash
# Re-vendor existing pinned historical/runtime sources only.
python3 scripts/vendor_sources.py
python3 -m unittest discover -s tests -v
```

No large source repository clones, model weights, captured dataset images, or raw paper PDF binaries should be committed. Check upstream code and checkpoint licenses before reproduction/commercial usage.

The `configs/vendored_manifest.json` lists the retained pinned sources. Old 3DGS provenance remains recoverable via past commits (e.g., commit `b7f0c78`).
