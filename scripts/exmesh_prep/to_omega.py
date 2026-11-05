"""Convert source-gauge meshes to the Omega gauge using the frozen Sim3 (CPU only)."""
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def main() -> None:
    report = json.loads(Path(sys.argv[1]).read_text())
    s = float(report["scale"])
    R = np.asarray(report["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(report["translation"], dtype=np.float64)

    for src_arg, dst_arg in zip(sys.argv[2::2], sys.argv[3::2]):
        src, dst = Path(src_arg), Path(dst_arg)
        ply = PlyData.read(str(src))
        v = ply["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        omega = (xyz - t) @ R / s
        names = [p.name for p in v.properties]
        if {"nx", "ny", "nz"} <= set(names):
            n = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float64)
            n_out = n @ R
        v["x"], v["y"], v["z"] = omega[:, 0], omega[:, 1], omega[:, 2]
        if {"nx", "ny", "nz"} <= set(names):
            v["nx"], v["ny"], v["nz"] = n_out[:, 0], n_out[:, 1], n_out[:, 2]
        dst.parent.mkdir(parents=True, exist_ok=True)
        PlyData([v] + [ply[e.name] for e in ply.elements if e.name != "vertex"], text=True).write(str(dst))
        print(f"{src} -> {dst}  verts={len(v.data)}  bbox={np.round(omega.min(0), 3)}..{np.round(omega.max(0), 3)}")


if __name__ == "__main__":
    main()
