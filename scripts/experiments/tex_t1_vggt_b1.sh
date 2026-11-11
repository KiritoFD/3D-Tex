#!/usr/bin/env bash
# Tex-T1 B1：VGGT / proxy Mesh 的 SUPRA bf16 同规格推理（短持锁轮询 + 峰值/耗时落盘）
# MODE=base        1024/250000，沿用 remesh + simplify
# MODE=no_simplify 跳过 simplify（remesh 输出直接导出）
# MODE=no_remesh   跳过 remesh 与 simplify（直接以输入面数展开 UV 并烘焙）
set -u
ROOT=/root/3dtex
OUT=${OUT_ROOT:-$ROOT/outputs/tex_t1_supra_20261105}
TEX=${TEX_SIZE:-1024}
DEC=${DEC_TARGET:-250000}
MODE=${MODE:-base}
MESH=${MESH:-vggt}
GLB_DIR=${GLB_DIR:-$OUT/glb_bf16}
FREE_THRESHOLD=${FREE_THRESHOLD:-6500}
WAIT_WINDOW=${WAIT_WINDOW_S:-1200}
if [ "$MODE" = "base" ]; then
  GLB=$GLB_DIR/supra_${MESH}_v0_s42_bf16.glb
else
  GLB=$GLB_DIR/supra_${MESH}_v0_s42_bf16_${MODE}.glb
fi
LOG=$OUT/logs/b1_${MESH}_${MODE}.log
mkdir -p "$OUT/logs" "$GLB_DIR"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

deadline=$(( $(date +%s) + WAIT_WINDOW ))
exec 9>"$ROOT/.locks/gpu4090.lock"
while :; do
  if ! flock -n 9; then
    flock -u 9 2>/dev/null || true
    [ "$(date +%s)" -ge "$deadline" ] && { say "giveup: lock busy"; exit 2; }
    say "lock busy, retry in 45s"; sleep 45; continue
  fi
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -ge "$FREE_THRESHOLD" ]; then say "acquired, free=${free}MiB"; break; fi
  say "free=${free}MiB < ${FREE_THRESHOLD}MiB, release and retry"
  flock -u 9 2>/dev/null || true
  [ "$(date +%s)" -ge "$deadline" ] && { say "giveup: vram free=${free}MiB"; exit 2; }
  sleep 45
done

BASE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
say "baseline used=${BASE}MiB mode=${MODE} tex=${TEX} dec=${DEC}"
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" >> "$OUT/logs/vram_b1_${MESH}_${MODE}.txt"; sleep 2; done ) &
MON=$!
trap 'kill $MON 2>/dev/null; flock -u 9' EXIT

case "$MODE" in
  no_simplify) export SUPRATEX_SKIP_SIMPLIFY=1 ;;
  no_remesh)   export SUPRATEX_SKIP_SIMPLIFY=1 SUPRATEX_NO_REMESH=1 ;;
esac

source /opt/conda/etc/profile.d/conda.sh
conda activate supra-tex
cd "$ROOT/third_party/supra_tex"
export SUPRATEX_ABI_IGNORE=dtype SUPRATEX_FORCE_BF16=1
t0=$(date +%s)
python texturing.py \
  --model-config "$ROOT/models/supra_tex/completion_bf16/config.json" \
  --checkpoint "$ROOT/models/supra_tex/completion/denoiser_step0030000.safetensors" \
  --trellis-root "$ROOT/models/trellis2" \
  --input-dir "$OUT/inputs/${MESH}_v0" --output "$GLB" \
  --attention sdpa --low-vram --seed 42 \
  --texture-size "$TEX" --decimation-target "$DEC" --texture-format png >> "$LOG" 2>&1
rc=$?
t1=$(date +%s)
peak=$(awk -v a="$t0" -v b="$t1" '$1>=a && $1<=b { if ($2+0>m) m=$2+0 } END { print m+0 }' "$OUT/logs/vram_b1_${MESH}_${MODE}.txt")
say "MESH=${MESH} MODE=${MODE} rc=${rc} elapsed=$((t1-t0))s peak_all=${peak}MiB peak_delta=$((peak-BASE))MiB glb=$(du -h "$GLB" 2>/dev/null | cut -f1)"
flock -u 9 2>/dev/null || true
exit $rc
