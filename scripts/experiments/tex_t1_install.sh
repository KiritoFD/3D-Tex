#!/bin/bash
# Tex-T1: SUPRA-Tex 独立环境安装（不污染 /opt/conda/envs/3dtex）
set -uo pipefail

ENV=supra-tex
PY=/opt/conda/envs/$ENV/bin/python
SRC=/root/3dtex/third_party/supra_tex
LOG=/root/3dtex/outputs/tex_t1_supra_20261105/logs/install.log
export PATH=/opt/conda/bin:$PATH
export MAX_JOBS=8
mkdir -p "$(dirname "$LOG")"

say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "=== create env $ENV ==="
conda create -y -n "$ENV" python=3.10 >> "$LOG" 2>&1 || { say "conda create failed"; exit 1; }
say "env created"

# 第三方源码按官方仓库固定 commit 落地（快照无 .git 时补一次 clone 便于复现）
if [ ! -d "$SRC/supratex" ]; then
  say "=== clone supra-tex @cee4435 ==="
  mkdir -p /root/3dtex/third_party
  git clone https://github.com/Supra-3D/supra-tex.git "$SRC" >> "$LOG" 2>&1
  git -C "$SRC" checkout cee4435652d0f23737bd28d5704e0fae9830b411 >> "$LOG" 2>&1
fi

"$PY" -m pip install pip==25.1.1 setuptools==80.9.0 wheel==0.45.1 packaging==25.0 >> "$LOG" 2>&1

say "=== cuda toolkit 12.4: reuse existing /opt/conda/envs/nvcc124 ==="
export CUDA_HOME=/opt/conda/envs/nvcc124
export PATH="$CUDA_HOME/bin:$PATH"
"$CUDA_HOME/bin/nvcc" --version >> "$LOG" 2>&1 || { say "nvcc124 missing"; exit 1; }

say "=== torch 2.6.0+cu124 ==="
"$PY" -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124 >> "$LOG" 2>&1 || { say "torch install failed"; exit 1; }

say "=== python requirements ==="
"$PY" -m pip install -r "$SRC/deployment/runtime/requirements-python.txt" >> "$LOG" 2>&1

say "=== supratex editable + bundled moge/utils3d ==="
"$PY" -m pip install --no-build-isolation --no-deps -e "$SRC" >> "$LOG" 2>&1
"$PY" -m pip install --no-build-isolation --no-deps "$SRC/deployment/runtime/moge" "$SRC/deployment/runtime/utils3d" >> "$LOG" 2>&1

say "=== verify base imports ==="
"$PY" - <<'PY' >> "$LOG" 2>&1
import torch, torchvision
print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
say "install stage1 done"
