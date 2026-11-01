"""Download only example photos from pinned VGGT (no weights, no full dataset)."""
import argparse,json,subprocess,shutil
from common import ROOT,scene_path
MAP={"fern":"llff_fern","flower":"llff_flower","kitchen":"kitchen"}
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--scenes",nargs="+",choices=sorted(MAP),default=["fern","flower"])
 p.add_argument("--limit",type=int,default=0)
 a=p.parse_args()
 spec=json.loads((ROOT/"configs"/"upstreams.json").read_text())["vggt"]
 cache=ROOT/"data"/".cache"/"vggt-examples"
 if not (cache/".git").exists():
  cache.parent.mkdir(parents=True,exist_ok=True)
  subprocess.run(["git","clone","--depth","1","--filter=blob:none","--sparse",spec["url"],str(cache)],check=True)
 subprocess.run(["git","-C",str(cache),"fetch","--depth","1","origin",spec["sha"]],check=True)
 subprocess.run(["git","-C",str(cache),"checkout","--detach",spec["sha"]],check=True)
 subprocess.run(["git","-C",str(cache),"sparse-checkout","set"]+["examples/"+MAP[x]+"/images" for x in a.scenes],check=True)
 for name in a.scenes:
  source=cache/"examples"/MAP[name]/"images"
  images=sorted(x for x in source.iterdir() if x.suffix.lower() in (".png",".jpg",".jpeg"))
  if a.limit and a.limit<len(images):
   if a.limit<3:raise ValueError("At least three images")
   n=a.limit;images=[images[round(i*(len(images)-1)/(n-1))] for i in range(n)]
  target=scene_path(name)/"images";target.mkdir(parents=True,exist_ok=True)
  for image in images:shutil.copy2(image,target/image.name)
  (target.parent/"sample_manifest.json").write_text(json.dumps({"sample":name,"images":[i.name for i in images],"benchmark":False},indent=2))
  print(name,len(images))
if __name__=="__main__":main()
