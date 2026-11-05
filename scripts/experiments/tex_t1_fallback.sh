#!/bin/bash
# Tex-T1 旁路臂：fp32 因 co-tenant 显存竞争无法运行时的 bf16 受限规格验证（非基线）
set -uo pipefail
ROOT=/root/3dtex
OUT=$ROOT/outputs/tex_t1_supra_20261105
SPY=/opt/conda/envs/supra-tex/bin/python
SRC=$ROOT/third_party/supra_tex
LOG=$OUT/logs/fallback.log
NEED=${NEED_FREE_MB:-7000}
mkdir -p "$OUT/logs" "$OUT/glb_bf16" "$OUT/eval" "$ROOT/models/supra_tex/completion_bf16"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

exec 9>"$ROOT/.locks/gpu4090.lock"
NEED_EVAL=${NEED_EVAL_MB:-4500}
NEED_BF16=${NEED_BF16_MB:-6500}
deadline=$(( $(date +%s) + ${WAIT_WINDOW_S:-900} ))
while :; do
  if flock -n 9; then
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$NEED_EVAL" ]; then
      say "lock acquired, free=${free}MiB >= ${NEED_EVAL}MiB"
      break
    fi
    flock -u 9 2>/dev/null || true
    say "free=${free}MiB < ${NEED_EVAL}MiB, release and retry"
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then say "eval window expired"; exit 2; fi
  sleep 30
done
BASE_VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" >> "$OUT/logs/vram_samples_fallback.txt"; sleep 2; done ) &
MON=$!
trap 'kill $MON 2>/dev/null; flock -u 9' EXIT

# ---- 1) T0 单视角四轴补全（LPIPS + 跨视角一致性），同一渲染口径 ----
if grep -q 't0:proxy' "$OUT/eval/summary.json" 2>/dev/null; then
  say "eval T0 already done, skip"
else
  say "eval T0 four-axis"
  "$SPY" "$ROOT/scripts/experiments/tex_t1_eval.py" --glb t0:proxy --glb t0:vggt \
    --out "$OUT/eval" >> "$OUT/logs/eval_t0.log" 2>&1
  say "eval T0 rc=$?"
fi

# ---- 2) bf16 受限规格配置（denoiser dtype float32 -> bfloat16） ----
CFG=$ROOT/models/supra_tex/completion_bf16
$SPY - <<PY
import json
src = "/root/3dtex/models/supra_tex/completion/config.json"
dst = "$CFG/config.json"
d = json.load(open(src, encoding="utf-8"))
d["models"]["denoiser"]["args"]["dtype"] = "bfloat16"
json.dump(d, open(dst, "w", encoding="utf-8"), indent=2)
print("bf16 config dtype:", d["models"]["denoiser"]["args"]["dtype"])
PY
[ -f "$CFG/config.json" ] || { say "bf16 config failed"; exit 3; }

# ---- 3) 两个 Mesh × seed42，2048 分辨率，sdpa + low-vram ----
free2=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free2" -lt "$NEED_BF16" ]; then
  say "free=${free2}MiB < ${NEED_BF16}MiB, skip bf16 arm"
  say "FALLBACK DONE"
  exit 0
fi
for mesh in proxy vggt; do
  name="supra_${mesh}_v0_s42_bf16"
  glb="$OUT/glb_bf16/${name}.glb"
  if [ -f "$glb" ]; then say "skip $name (exists)"; continue; fi
  t0=$(date +%s)
  say "run $name"
  ( cd "$SRC" && SUPRATEX_ABI_IGNORE=dtype SUPRATEX_FORCE_BF16=1 "$SPY" texturing.py \
      --model-config "$CFG/config.json" \
      --checkpoint "$ROOT/models/supra_tex/completion/denoiser_step0030000.safetensors" \
      --trellis-root "$ROOT/models/trellis2" \
      --input-dir "$OUT/inputs/${mesh}_v0" --output "$glb" \
      --attention sdpa --low-vram --seed 42 --texture-size 2048 --texture-format png ) \
    >> "$OUT/logs/${name}.log" 2>&1
  rc=$?
  t1=$(date +%s)
  peak=$(awk -v a="$t0" -v b="$t1" '$1>=a && $1<=b { if ($2+0>m) m=$2+0 } END { print m+0 }' "$OUT/logs/vram_samples_fallback.txt")
  if [ $rc -eq 0 ] && [ -f "$glb" ]; then
    say "OK $name elapsed=$((t1-t0))s peak_all=${peak}MiB peak_delta=$((peak-BASE_VRAM))MiB"
  else
    say "FAIL $name rc=$rc elapsed=$((t1-t0))s peak_all=${peak}MiB"
  fi
done

# ---- 4) 旁路臂四轴评测 ----
specs=""
for mesh in proxy vggt; do
  f="$OUT/glb_bf16/supra_${mesh}_v0_s42_bf16.glb"
  [ -f "$f" ] && specs="$specs --glb supra_${mesh}_v0_s42_bf16=$f"
done
if [ -n "$specs" ]; then
  say "eval bf16 arm"
  "$SPY" "$ROOT/scripts/experiments/tex_t1_eval.py" $specs --out "$OUT/eval" >> "$OUT/logs/eval_bf16.log" 2>&1
  say "eval bf16 rc=$?"
fi

say "FALLBACK DONE"
