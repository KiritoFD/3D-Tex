"""Mirror VGGT sparse/ binaries into 2D-SuGaR's required sparse/0/ directory."""
from __future__ import annotations
import argparse, shutil
from pathlib import Path
from common import images_for,scene_path
def prepare(scene: Path):
    images_for(scene)
    src=scene/"sparse"
    dst=src/"0"
    if not src.is_dir(): raise FileNotFoundError(src)
    dst.mkdir(exist_ok=True)
    for filename in ("cameras.bin","images.bin","points3D.bin"):
        a,b=src/filename,dst/filename
        if not a.is_file() or a.stat().st_size==0: raise FileNotFoundError(a)
        if b.is_symlink() and b.resolve()==a.resolve(): continue
        if b.exists():
            if b.read_bytes()!=a.read_bytes(): raise FileExistsError(b)
            continue
        try: b.symlink_to(Path("..")/filename)
        except OSError: shutil.copy2(a,b)
    return dst
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--scene",required=True)
    prepare(scene_path(p.parse_args().scene))
