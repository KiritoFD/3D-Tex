import shutil
from common import ROOT
needed={"vggt":["demo_colmap.py","vggt/models/vggt.py","LICENSE.txt"],"3dgs":["setup.py","config.yaml","examples/simple_trainer.py","gsplat/__init__.py"],"2dsugar":["train.py","gaussian_splatting_2d/train.py","sugar_extractors/refined_mesh.py","gaussian_splatting_2d/submodules/diff-surfel-rasterization/setup.py","gaussian_splatting_2d/submodules/simple-knn/setup.py","gaussian_splatting/submodules/diff-gaussian-rasterization/setup.py"]}
missing=[str(ROOT/name/path) for name,paths in needed.items() for path in paths if not (ROOT/name/path).is_file()]
for cmd in ("conda","nvcc","nvidia-smi"):print(cmd,shutil.which(cmd))
if missing:print("Missing source:","; ".join(missing));raise SystemExit(1)
print("Vendored source entrypoints OK; CUDA unverified")
