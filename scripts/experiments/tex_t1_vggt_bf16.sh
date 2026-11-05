#!/usr/bin/env bash
# 受限规格（bf16 权重）旁路臂：vggt 固定 Mesh 单视图生成，降分辨率/降简化目标以避开显存竞争。
set -u
ROOT=/root/3dtex
OUT=$ROOT/outputs/tex_t1_supra_20261105
TEX=${TEX_SIZE:-1024}
DEC=${DEC_TARGET:-300000}
FREE_THRESHOLD=${FREE_THRESHOLD:-6500}

free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "free=${free_mb}MiB threshold=${FREE_THRESHOLD}MiB tex=${TEX} dec=${DEC}"
if [ "$free_mb" -lt "$FREE_THRESHOLD" ]; then echo "SKIP_LOW_FREE"; exit 3; fi

source /opt/conda/etc/profile.d/conda.sh
conda activate supra-tex
cd "$ROOT/third_party/supra_tex"
export SUPRATEX_ABI_IGNORE=dtype SUPRATEX_FORCE_BF16=1
mkdir -p "$OUT/logs"
flock -w 300 "$ROOT/.locks/gpu4090.lock" -c \
  "python texturing.py \
     --model-config $ROOT/models/supra_tex/completion_bf16/config.json \
     --checkpoint $ROOT/models/supra_tex/completion/denoiser_step0030000.safetensors \
     --trellis-root $ROOT/models/trellis2 \
     --input-dir $OUT/inputs/vggt_v0 \
     --output $OUT/glb_bf16/supra_vggt_v0_s42_bf16.glb \
     --attention sdpa --low-vram --seed 42 \
     --texture-size $TEX --decimation-target $DEC --texture-format png"
rc=$?
echo "rc=$rc"
exit $rc
