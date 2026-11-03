"""Small fixed-count capture pilot; warm-up predicted poses never use GT selection."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VS = ROOT / "scripts/view_selection.py"


def prepare(config):
    import numpy as np
    from PIL import Image
    root = Path(config["root"])
    anchor, pool = Path(config["anchor"]), Path(config["pool"])
    frames = json.loads((pool / "geometry/omega/cameras.json").read_text())["frames"]
    stems = [Path(f["frame"]).stem for f in frames]
    cameras = {Path(f["frame"]).stem: f for f in frames}
    centers = {}
    for stem, frame in cameras.items():
        t = np.asarray(frame["T_world_to_cam"])
        centers[stem] = -t[:3,:3].T @ t[:3,3]
    probes = config["probes"]
    remaining = [s for s in stems if s not in probes]
    near = sorted(remaining, key=lambda s: (min(np.linalg.norm(centers[s]-centers[p]) for p in probes),s))[:8]
    chosen = list(probes)
    for _ in range(8):
        candidates = [s for s in remaining if s not in chosen]
        chosen.append(max(candidates, key=lambda s: min(np.linalg.norm(centers[s]-centers[p]) for p in chosen)))
    anchor_stems = sorted(p.stem for p in (anchor / "images").glob("*.png"))
    groups = {name: anchor_stems for name in config["groups"]}
    groups["near_baseline16"] = sorted(probes + near)
    groups["diverse_baseline16"] = sorted(chosen)
    contract = {"config": config, "groups": groups,
        "selection": "predicted AB camera-centre distance is a baseline/coverage proxy, not GT parallax",
        "warmup_camera_sha256": hashlib.sha256((pool / "geometry/omega/cameras.json").read_bytes()).hexdigest(),
        "background": "gray=128; image-fixed checker=64/192, tile=32px; visible means recorded after occlusion"}
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "selection.json"
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError("capture protocol changed; use a fresh root")
    manifest.write_text(json.dumps(contract, indent=2) + "\n")
    background_stats, input_hashes = {}, {}
    for name, selected in groups.items():
        background_stats[name], input_hashes[name] = [], {}
        scene = root / name
        (scene / "images").mkdir(parents=True, exist_ok=True)
        gt_source = Path(config["pool_gt"] if "baseline" in name else config["anchor_gt"])
        gt = scene / "geometry/gt"
        for folder in ("depth", "foreground"):
            (gt / folder).mkdir(parents=True, exist_ok=True)
        gt_cameras = json.loads((gt_source / "cameras.json").read_text())
        gt_cameras["frames"] = [f for f in gt_cameras["frames"] if Path(f["frame"]).stem in selected]
        assert len(selected) == len(gt_cameras["frames"]) == 16 and set(probes) <= set(selected)
        (gt / "cameras.json").write_text(json.dumps(gt_cameras, indent=2))
        for stem in selected:
            source = (pool if "baseline" in name else anchor) / "images" / f"{stem}.png"
            target = scene / "images" / source.name
            if name in ("gray16", "static_texture16"):
                with Image.open(source) as image:
                    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
                h, w = rgba.shape[:2]
                yy, xx = np.indices((h,w))
                bg = np.full((h,w),128.,dtype=np.float32)
                if name == "static_texture16":
                    bg += np.where((xx//32 + yy//32)%2,64.,-64.)
                alpha = rgba[...,3:4] / 255
                rgb = np.rint(rgba[...,:3]*alpha + bg[...,None]*(1-alpha)).clip(0,255).astype(np.uint8)
                Image.fromarray(rgb).save(target)
                visible_bg, solid_fg = rgba[...,3] == 0, rgba[...,3] == 255
                unchanged = int(np.abs(rgb[solid_fg].astype(int)-rgba[solid_fg,:3].astype(int)).max())
                assert unchanged == 0
                background_stats[name].append({"stem": stem,
                    "visible_background_mean": float(rgb[visible_bg].mean()),
                    "visible_background_std": float(rgb[visible_bg].std()),
                    "solid_foreground_max_change": unchanged})
            else:
                shutil.copyfile(source, target)
            input_hashes[name][stem] = hashlib.sha256(target.read_bytes()).hexdigest()
            for folder in ("depth", "foreground"):
                shutil.copyfile(gt_source / folder / f"{stem}.npy", gt / folder / f"{stem}.npy")
        if name == "white16" and not (scene / "geometry/omega").exists():
            target = scene / "geometry/omega"
            target.mkdir()
            for filename in ("cameras.json", "manifest.json"):
                shutil.copyfile(anchor / "geometry/omega" / filename, target / filename)
            for folder in ("depth", "confidence"):
                shutil.copytree(anchor / "geometry/omega" / folder, target / folder)
    (root / "background_stats.json").write_text(json.dumps(background_stats, indent=2) + "\n")
    (root / "input_hashes.json").write_text(json.dumps(input_hashes, indent=2) + "\n")
    print(json.dumps({name: len(selected) for name,selected in groups.items()}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/local_capture.json")
    p.add_argument("--phase", choices=["prepare", "infer", "evaluate", "all"], required=True)
    p.add_argument("--alignment", choices=["probes", "all"], default="probes",
                   help="all is a sensitivity audit; primary protocol stays fixed-probe")
    p.add_argument("--out", help="override result root for a separately labelled protocol")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.phase in ("prepare", "all"):
        prepare(config)
    root, out = Path(config["root"]), Path(args.out or config["out"])
    probes = ",".join(config["probes"])
    if args.phase in ("infer", "all"):
        for name in config["groups"]:
            subprocess.run([sys.executable,str(VS),"infer","--run-dir",str(root/name),
                "--image-resolution",str(config.get("resolution_overrides",{}).get(name,config["model_resolution"])),
                "--seed",str(config["model_seed"]),
                "--aggregator-dtype",config["aggregator_dtype"]], cwd=ROOT,check=True)
    if args.phase in ("evaluate", "all"):
        jobs = [(root/name,out/name) for name in config["groups"]]
        if args.alignment == "probes":
            jobs += [(Path(source),out/"resolution"/name) for name,source in config.get("reuse_resolution",{}).items()]
        for scene, result in jobs:
            result.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable,str(VS),"eval","--omega",str(scene/"geometry/omega"),
                "--gt",str(scene/"geometry/gt"),"--frames",probes,"--out",str(result/"probe.json")],cwd=ROOT,check=True)
            with (result/"fusion.log").open("w") as log:
                cmd=[sys.executable,str(ROOT/"scripts/capture/capture_eval.py"),"--capture",str(scene),"--out",str(result)]
                if args.alignment == "probes":
                    cmd += ["--alignment-stems",probes]
                subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            with (result/"measure.log").open("w") as log:
                subprocess.run([config["blender"],"-b","--disable-autoexec","--python",
                    str(ROOT/"scripts/diagnostics/measure_surfaces.py"),"--","--out",str(result),
                    "--source-flags",str(ROOT/"outputs/lego_source_edge_flags.npy")],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(f"[capture] evaluated {scene.name}",flush=True)


if __name__ == "__main__":
    main()
