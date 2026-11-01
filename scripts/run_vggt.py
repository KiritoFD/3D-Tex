"""VGGT feedforward cameras and point cloud, optional bundle adjustment."""
import argparse
from common import ROOT,conda,execute,images_for,scene_path
from prepare_scene import prepare
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scene",required=True)
    p.add_argument("--ba",action="store_true")
    p.add_argument("--query-frames",type=int,default=5)
    p.add_argument("--max-query-points",type=int,default=2048)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    scene=scene_path(a.scene)
    images=images_for(scene)
    cmd=conda("recon-vggt","python","demo_colmap.py","--scene_dir",str(scene))
    if a.ba:
        cmd+=["--use_ba","--max_query_pts",str(a.max_query_points),
              "--query_frame_num",str(min(a.query_frames,len(images)))]
    execute("vggt",a.scene,cmd,ROOT/"vggt",a.dry_run)
    if not a.dry_run: prepare(scene)
if __name__=="__main__": main()
