#!/bin/bash
# Tex-T1: 下载 SUPRA-Tex 完成模型 + TRELLIS.2 VAE 权重（经 hf-mirror）
set -uo pipefail
export HF_ENDPOINT=https://hf-mirror.com
LOG=/root/3dtex/outputs/tex_t1_supra_20261105/logs/weights.log
mkdir -p "$(dirname "$LOG")"
PY=/opt/conda/envs/3dtex/bin/python

say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "=== completion: supra3d/supra-tex :: completion/* ==="
"$PY" - <<'PY' >> "$LOG" 2>&1
from huggingface_hub import snapshot_download
p = snapshot_download("supra3d/supra-tex",
                      allow_patterns=["completion/config.json", "completion/partial_normalization.json",
                                      "completion/denoiser_step0030000.safetensors", "completion/LICENSE*",
                                      "LICENSE*", "README.md"],
                      local_dir="/root/3dtex/models/supra_tex", max_workers=8)
print("completion ->", p)
PY

say "=== vae: microsoft/TRELLIS.2-4B :: ckpts/* ==="
"$PY" - <<'PY' >> "$LOG" 2>&1
from huggingface_hub import snapshot_download
p = snapshot_download("microsoft/TRELLIS.2-4B", revision="af44b45f2e35a493886929c6d786e563ec68364d",
                      allow_patterns=["ckpts/shape_enc_next_dc_f16c32_fp16.*",
                                      "ckpts/tex_enc_next_dc_f16c32_fp16.*",
                                      "ckpts/shape_dec_next_dc_f16c32_fp16.*",
                                      "ckpts/tex_dec_next_dc_f16c32_fp16.*", "LICENSE*"],
                      local_dir="/root/3dtex/models/trellis2", max_workers=8)
print("vae ->", p)
PY

say "=== tree ==="
du -sh /root/3dtex/models/supra_tex /root/3dtex/models/trellis2 >> "$LOG" 2>&1
find /root/3dtex/models/supra_tex/completion -maxdepth 2 -type f | head -20 >> "$LOG" 2>&1
say "weights done"
