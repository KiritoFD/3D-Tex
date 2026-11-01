#!/usr/bin/env bash
set -euo pipefail
SCENE="${1:-fern}"
python3 scripts/run_vggt.py --scene "$SCENE" --ba
python3 scripts/run_3dgs.py --scene "$SCENE" --steps 7000
python3 scripts/run_2dsugar.py --scene "$SCENE" --iterations 7000 --refine-iterations 2000
