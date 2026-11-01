import argparse
from common import ROOT,OUTPUTS,conda,execute,images_for,scene_path
from prepare_scene import prepare
from scene_scale import robust_extent
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--scene",required=True)
 p.add_argument("--iterations",type=int,default=7000)
 p.add_argument("--refine-iterations",type=int,default=2000)
 p.add_argument("--resolution",type=int,default=2)
 p.add_argument("--priors",choices=["none","metric3d"],default="none")
 p.add_argument("--no-uv",action="store_true")
 p.add_argument("--dry-run",action="store_true")
 a=p.parse_args()
 scene=scene_path(a.scene)
 images_for(scene)
 sparse=prepare(scene)
 size=robust_extent(sparse)
 if a.priors=="metric3d":
  execute("metric3d",a.scene,conda("2d-sugar","python","estimate_depth_normal_priors.py","-s",str(scene)),ROOT/"2dsugar",a.dry_run)
  execute("colmap-text",a.scene,conda("recon-vggt","python","-c","import pycolmap,sys;pycolmap.Reconstruction(sys.argv[1]).write_text(sys.argv[1])",str(sparse)),ROOT,a.dry_run)
 output=OUTPUTS/"2dsugar"/a.scene
 cmd=conda("2d-sugar","python","train.py","-s",str(scene),"-m",str(output/"model"),"-o",str(output/"mesh"),"--eval","--test_iterations","-1","--save_iterations",str(a.iterations),"--iterations",str(a.iterations),"--refinement_iterations",str(a.refine_iterations),"--cluster_prune_iterations",str(max(1,a.iterations//2)),"--initialization_prior","monocular_depth_normal" if a.priors=="metric3d" else "","--depth_ratio","1" if a.priors=="metric3d" else "0","-r",str(a.resolution),"--lambda_dist","1000","--num_cluster","1","--voxel_size",str(size/256),"--sdf_trunc",str(size/64),"--depth_trunc",str(size*3),"--export_uv_textured_mesh","False" if a.no_uv else "True")
 execute("2dsugar",a.scene,cmd,ROOT/"2dsugar",a.dry_run)
if __name__=="__main__":main()
