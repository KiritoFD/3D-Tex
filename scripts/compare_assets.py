"""Compare four matching training-camera renders; these are not heldout metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def composite(image):
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (96, 96, 96, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--scene", default="data/scenes/nerf_lego16")
    args = parser.parse_args()
    scene, before, after = Path(args.scene), Path(args.before), Path(args.after)
    frames = json.loads((scene / "geometry/omega/cameras.json").read_text())["frames"]
    comparison = Image.new("RGB", (1536, 1080), (36, 36, 36))
    fourviews = Image.new("RGB", (1024, 1080), (36, 36, 36))
    draw, fourdraw = ImageDraw.Draw(comparison), ImageDraw.Draw(fourviews)
    metrics = []
    for i, fi in enumerate((0, 4, 8, 12)):
        reference = Image.open(scene / "images" / frames[fi]["file"]).convert("RGBA").resize((512, 512))
        old = Image.open(before / f"view_{i}.png").convert("RGBA")
        new = Image.open(after / f"view_{i}.png").convert("RGBA")
        target = np.asarray(reference).astype(float) / 255
        mask = target[..., 3] > .5
        item = {"frame": frames[fi]["frame"]}
        for name, image in (("before", old), ("after", new)):
            pixels = np.asarray(image).astype(float) / 255
            predicted = pixels[..., 3] > .5
            item[name] = {
                "silhouette_iou": float((mask & predicted).sum() / (mask | predicted).sum()),
                "foreground_psnr": float(-10 * np.log10(np.maximum(
                    np.mean((pixels[..., :3][mask] * pixels[..., 3:4][mask] - target[..., :3][mask]) ** 2), 1e-12))),
            }
        metrics.append(item)
        x, y = (i % 2) * 512, (i // 2) * 540
        fourviews.paste(composite(new), (x, y + 28))
        fourdraw.text((x+12, y+8), f"Corrected - camera {fi}", fill="white")
        if i in (0, 2):
            row = i // 2
            for col, (label, image) in enumerate(zip(
                    ("Input RGB", "Before (700 iterations)", "Corrected (10000 iterations)"),
                    (reference, old, new))):
                comparison.paste(composite(image), (col*512, row*540+28))
                draw.text((col*512+12, row*540+8), label, fill="white")
    comparison.save(after / "comparison.png")
    fourviews.save(after / "render_4views.png")
    report = {"protocol": "four training views, predicted Omega cameras, foreground RGB PSNR and silhouette IoU; not a benchmark",
              "views": metrics}
    for name in ("before", "after"):
        report[name] = {key: float(np.mean([v[name][key] for v in metrics]))
                        for key in metrics[0][name]}
    (after / "comparison_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ("before", "after")}, indent=2))


if __name__ == "__main__":
    main()
