"""Standard-library-only pipeline helpers."""
from __future__ import annotations
import json, re, shlex, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SCENES=ROOT/"data"/"scenes"
OUTPUTS=ROOT/"outputs"
def scene_path(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",name) or name in (".",".."):
        raise ValueError(f"Invalid scene name: {name}")
    return SCENES/name
def images_for(scene,minimum=3):
    path=scene/"images"
    files=sorted(p for p in path.glob("*") if p.suffix.lower() in (".jpg",".jpeg",".png")) if path.is_dir() else []
    if len(files)<minimum: raise FileNotFoundError(f"Need {minimum} images in {path}")
    return files
def require_colmap(scene):
    for folder in (scene/"sparse"/"0",scene/"sparse"):
        if all((folder/x).is_file() for x in ("cameras.bin","images.bin","points3D.bin")):
            return folder
    raise FileNotFoundError(f"No VGGT COLMAP export in {scene}")
def conda(env,*args):
    return ["conda","run","--no-capture-output","-n",env,*map(str,args)]
def execute(stage,name,args,cwd,dry_run=False):
    print(f"[{stage}] cwd={cwd}\n+ {shlex.join(args)}",flush=True)
    if dry_run: return
    if not cwd.is_dir(): raise FileNotFoundError(f"Missing vendored source: {cwd}")
    logs=OUTPUTS/"logs"
    logs.mkdir(parents=True,exist_ok=True)
    path=logs/(name+"-"+stage+"-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+".log")
    start=time.perf_counter()
    with path.open("w") as fp:
        fp.write(shlex.join(args)+"\n")
        fp.flush()
        with subprocess.Popen(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,errors="replace") as proc:
            for line in proc.stdout:
                print(line,end="",flush=True)
                fp.write(line)
            status=proc.wait()
    path.with_suffix(".json").write_text(json.dumps({
        "stage":stage,"scene":name,"command":args,"returncode":status,
        "seconds":round(time.perf_counter()-start,3),"log":str(path)},indent=2))
    if status: raise subprocess.CalledProcessError(status,args)
