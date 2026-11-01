"""gsplat 3DGS per-scene fit with VGGT COLMAP initialization."""
import argparse
from common import ROOT,OUTPUTS,conda,execute,images_for,require_colmap,scene_path
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scene",required=True)
    p.add_argument("--steps",type=int,default=7000)
    p.add_argument("--test-every",type=int,default=8)
    p.add_argument("--data-factor",type=int,default=1)
    p.add_argument("--save-ply",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    if a.steps<1 or a.test_every<2 or a.data_factor<1: p.error("Invalid training arguments")
    scene=scene_path(a.scene)
    images_for(scene);require_colmap(scene)
    cmd=conda("recon-3dgs","python","examples/simple_trainer.py","default",
              "--data_dir",str(scene),"--result_dir",str(OUTPUTS/"3dgs"/a.scene),
              "--data_factor",str(a.data_factor),"--test_every",str(a.test_every),
              "--max_steps",str(a.steps),"--eval_steps",str(a.steps),
              "--save_steps",str(a.steps),"--init_type","sfm",
              "--disable_viewer","--disable_video")
    if a.save_ply: cmd+=["--save_ply","--ply_steps",str(a.steps)]
    execute("3dgs",a.scene,cmd,ROOT/"3dgs",a.dry_run)
if __name__=="__main__": main()
