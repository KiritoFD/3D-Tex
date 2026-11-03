"""Small serial M2 launcher. Run inside the documented WSL render environment."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/local_mesh.json")
    p.add_argument("--only", help="comma-separated run names; default all")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    names = args.only.split(",") if args.only else list(config["runs"])
    summary = {}
    for name in names:
        options = config["common"] | config["runs"][name]
        out = Path(options["out"])
        finished = out / "m2_log.json"
        if finished.is_file():
            previous = json.loads(finished.read_text())["args"]
            for key, value in options.items():
                if previous[key.replace("-", "_")] != value:
                    raise ValueError(f"existing {name} has different {key}; choose fresh output")
        else:
            out.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "scripts/m2_anchor_opt.py"]
            for key, value in options.items():
                cmd += [f"--{key}", str(value)]
            print(f"[local] running {name}", flush=True)
            with (out / "run.log").open("w") as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(finished.read_text())
        summary[name] = {"mesh": str(out / "mesh_m2.ply"), "seconds": result["seconds"],
                         "peak_vram_gb": result["peak_vram_gb"],
                         "gt_assisted": result["provenance"]["gt_assisted"]}
        print(name, summary[name], flush=True)
    target = ROOT / "outputs/m2/local_runs.json"
    if target.exists():
        summary = json.loads(target.read_text()) | summary
    target.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
