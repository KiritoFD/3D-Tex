"""Official Mesh Splatting CUDA component probe, not a full method reproduction.

Run from WSL render venv with CPATH pointing at GLM headers. Source lives outside
tracked code; the only build patch is an explicit missing <cstdint> include.
"""
import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root, out = Path(args.repo).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    header = root / "TetrahedronRender/include/rasterizer_impl.h"
    text = header.read_text()
    if "#include <cstdint>" not in text:
        header.write_text(text.replace("#pragma once", "#pragma once\n#include <cstdint>"))
    sys.path.insert(0, str(root))
    os.chdir(root)
    official = importlib.import_module("custom_functions")
    torch.cuda.reset_peak_memory_stats()
    vertices = torch.tensor([[-1.,-1.,3.], [1.,-1.,3.], [0.,1.,3.]], device="cuda", requires_grad=True)
    faces = torch.tensor([[0,1,2]], dtype=torch.int32, device="cuda")
    proj = torch.tensor([[1.,0,0,0], [0,1.,0,0], [0,0,1.,0], [0,0,1.,0]], device="cuda")
    pose = torch.eye(4, device="cuda")
    feature = torch.tensor([[[.2,.4,.8],[.8,.3,.2],[.3,.8,.4]]], device="cuda", requires_grad=True)
    opacity = torch.full((1,3,1), .5, device="cuda", requires_grad=True)
    settings = official.TetrahedronRasterizationSettings(64,64,0.)

    def render():
        inv, depths, order, bounds, rect, tiles, visible = official.preprocess_tet(
            vertices, faces, proj, pose, 64, 64)
        _, alpha, color, _ = official.render_tet(-order, inv, feature, -depths[faces.long()],
                                                opacity, bounds, rect, tiles, visible, settings)
        return alpha, color

    alpha, color = render()
    loss = alpha.mean() + color.mean()
    loss.backward()
    gradients = {name: {"finite": bool(torch.isfinite(t.grad).all()),
                        "nonzero": int(torch.count_nonzero(t.grad))}
                 for name, t in [("vertices",vertices), ("features",feature), ("opacity",opacity)]}
    if not all(v["finite"] and v["nonzero"] > 0 for v in gradients.values()):
        raise FloatingPointError(gradients)
    analytic = opacity.grad.detach().clone()
    fd = []
    for i in range(3):
        losses = []
        for sign in (1,-1):
            with torch.no_grad():
                opacity[0,i,0] = .5 + sign * .001
                a, c = render()
                losses.append(float(a.mean() + c.mean()))
        with torch.no_grad():
            opacity[0,i,0] = .5
        numeric = (losses[0] - losses[1]) / .002
        exact = float(analytic[0,i,0])
        fd.append({"axis": i, "autograd": exact, "finite_difference": numeric,
                   "relative_error": abs(exact - numeric) / max(abs(numeric),1e-7)})
    if max(r["relative_error"] for r in fd) > .05:
        raise AssertionError(fd)
    result = {"scope": "official triangle splatting CUDA forward/backward + opacity FD; not full training",
              "source": "https://github.com/HKBU-VSComputing/mesh-splatting",
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "build_patch": "rasterizer_impl.h: add missing cstdint", "gradients": gradients,
              "opacity_fd": fd, "covered_pixels": int((alpha > 0).sum()),
              "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
              "sources_sha256": {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in sorted(root.rglob("*")) if f.suffix in (".py", ".cu", ".cpp", ".h")}}
    (out / "spike.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
