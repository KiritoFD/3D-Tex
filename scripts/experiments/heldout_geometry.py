"""Evaluation-only silhouette transfer on frozen cameras, using CUDA rasterization."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from mesh_opt.render import NvdRenderer
from ply_to_obj import read_ply


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="outputs/lego_heldout_val16")
    p.add_argument("--out", default="outputs/m2/heldout_val16")
    p.add_argument("--meshes", nargs="+", required=True, help="name=source-gauge PLY")
    args = p.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = json.loads((data / "cameras.json").read_text())["frames"]
    renderer = NvdRenderer(near=.01, far=20.)
    rows = {}
    for item in args.meshes:
        name, path = item.split("=",1)
        vertices, faces, _ = read_ply(Path(path))
        vertices = torch.tensor(vertices, device="cuda", dtype=torch.float32)
        faces = torch.tensor(faces, device="cuda", dtype=torch.int32).contiguous()
        folder = out / name
        folder.mkdir(exist_ok=True)
        results = []
        with torch.no_grad():
            for i, frame in enumerate(frames):
                k = np.asarray(frame["K_original"]).copy()
                k[0] *= 512 / frame["original_width"]
                k[1] *= 512 / frame["original_height"]
                t = torch.tensor(frame["T_world_to_cam"], device="cuda", dtype=torch.float32)
                _, alpha = renderer.render_frame(vertices, faces, torch.tensor(k, device="cuda", dtype=torch.float32), t, 512)
                predicted = alpha.cpu().numpy() > .5
                with Image.open(data / "images" / frame["frame"]) as image:
                    target = np.asarray(image.getchannel("A").resize((512,512), Image.Resampling.NEAREST)) > 127
                iou = float((target & predicted).sum() / max((target | predicted).sum(),1))
                results.append({"frame": frame["frame"], "silhouette_iou": iou})
                # Gray = agreement on foreground; red = extra surface; blue = missed surface.
                colors = np.full((512,512,3), 40, np.uint8)
                colors[target & predicted] = [170,170,170]
                colors[predicted & ~target] = [235,65,65]
                colors[target & ~predicted] = [60,130,245]
                Image.fromarray(colors).save(folder / f"view_{i:02d}.png")
        rows[name] = {"mean_iou": float(np.mean([r["silhouette_iou"] for r in results])), "views": results}
        print(name, rows[name]["mean_iou"], flush=True)
        (out / "metrics.json").write_text(json.dumps({"scope": "heldout silhouette geometry only; no RGB/normal quality claim",
            "split": json.loads((data / "split.json").read_text()), "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
