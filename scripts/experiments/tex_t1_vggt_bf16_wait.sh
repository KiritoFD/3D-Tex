#!/usr/bin/env bash
# 受限规格（bf16）vggt 臂：在 6 次探测窗口内等待显存，满足则串行跑一次；不持锁等待。
set -u
ROOT=/root/3dtex
for i in 1 2 3 4 5 6; do
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "[$(date +%H:%M:%S)] probe $i free=${free_mb}MiB"
  if [ "$free_mb" -ge 6500 ]; then
    bash "$ROOT/scripts/experiments/tex_t1_vggt_bf16.sh"
    exit $?
  fi
  sleep 60
done
echo "GIVEUP: free stayed below 6500MiB"
exit 4
