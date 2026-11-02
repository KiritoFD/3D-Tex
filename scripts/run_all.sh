#!/usr/bin/env bash
set -euo pipefail
# The old 3DGS/SuGaR pipeline is retired; the VGGT-Omega -> Mesh -> UV route is the only one.
# This script now drives the implemented stages end to end instead of failing closed:
#   Omega geometry -> depth consistency -> TSDF mesh -> ExMesh adapter -> OBJ/GLB export
SCENE="${1:-}"
if [ -z "$SCENE" ]; then
  echo "usage: bash scripts/run_all.sh <scene>   (e.g. eth3d_kicker)" >&2
  echo "stages: omega geometry | consistency | tsdf mesh | exmesh adapter | export" >&2
  echo "The retired 3DGS/SuGaR pipeline is no longer part of this repository." >&2
  echo "See docs/MESH_UV_PIPELINE.md and docs/VGGT_OMEGA_ADAPTER.md." >&2
  exit 2
fi
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY="python3"
CKPT="${3DTEX_OMEGA_CHECKPOINT:-data/.cache/models/sanqilu376/VGGT-Omega/vggt_omega_1b_512.pt}"

"$PY" scripts/adapters/vggt_omega_geometry.py --scene-dir "data/scenes/$SCENE" \
      --checkpoint "$CKPT" --device "${DEVICE:-auto}"
"$PY" scripts/consistency_filter.py --scene "$SCENE" --min-agree 1 --tau-rel 0.12
"$PY" scripts/tsdf_mesh.py --scene "$SCENE" --voxel-div 256 --trunc-voxels 4 --stride 2
"$PY" scripts/exmesh_adapter.py --scene "$SCENE"
echo "Next: cd ExMesh && python train.py -s \"$PWD/data/scenes/$SCENE/exmesh_workdir\" \\"
echo "        -m \"$PWD/data/scenes/$SCENE/exmesh_workdir/out\""
echo "Then: $PY scripts/export_exmesh.py --checkpoint <...>/model_state_dict.pt"
