#!/usr/bin/env bash
# Tex-T1：vggt 同规格 2048 臂（预注册 2026-11-12，尚未运行；判据见 TEX_T1_SUPRA.md §9.2）
#
# 口径 = proxy bf16 主行完全一致（tex_t1_fallback.sh：--texture-size 2048，
# 未传 decimation-target 即上游默认 500000）：TEX 2048 + remesh + decimate 500000
# + seed 42 + bf16。
# 2026-11-05 的 b1 base/no_simplify 失败根因是 co-tenant 显存（当时全卡 22.9GiB，
# 本臂空卡重试预计 ≈8.5GiB 增量）；若空卡下 remesh 仍 OOM，按 §9.2 判据登记为
# blocked-by-VRAM，不实现"分块 simplify"替代口径。
#
# 用法：bash scripts/experiments/tex_t1_vggt_base2048.sh
#   （所有参数已固定；等待窗口 40min，短持锁轮询复用 tex_t1_vggt_b1.sh）
set -u
export MESH=vggt
export MODE=base
export TEX_SIZE=2048
export DEC_TARGET=500000
export GLB_DIR=${OUT_ROOT:-/root/3dtex/outputs/tex_t1_supra_20261105}/glb_bf16_2048_base
export WAIT_WINDOW_S=${WAIT_WINDOW_S:-2400}
exec bash "$(dirname "$0")/tex_t1_vggt_b1.sh"
