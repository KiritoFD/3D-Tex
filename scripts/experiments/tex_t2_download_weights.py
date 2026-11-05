#!/usr/bin/env python3
"""直连 hf-mirror 的 CDN 下载 Hunyuan3D-2.1 Paint 权重（绕过 API 限流）。

只下载 Paint 阶段所需文件：hunyuan3d-paintpbr-v2-1 + facebook/dinov2-giant。
"""
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://hf-mirror.com"
DEST = Path("/root/hunyuan_weights")
PAINT_SUB = "hunyuan3d-paintpbr-v2-1"

PAINT = [
    "README.md", "model_index.json",
    "feature_extractor/preprocessor_config.json",
    "image_encoder/config.json", "image_encoder/model.safetensors",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json", "text_encoder/pytorch_model.bin",
    "tokenizer/merges.txt", "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json", "tokenizer/vocab.json",
    "unet/attn_processor.py", "unet/config.json",
    "unet/diffusion_pytorch_model.bin", "unet/model.py", "unet/modules.py",
    "vae/config.json", "vae/diffusion_pytorch_model.bin",
]
DINO = ["config.json", "model.safetensors", "preprocessor_config.json"]


def fetch(repo: str, rel: str, out: Path) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{repo}/resolve/main/{rel}?download=true"
    cmd = ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "3",
           "--continue-at", "-", "-sS", "-o", str(out), url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return f"FAIL {repo}/{rel}: {r.stderr.strip()[:200]}"
    return f"OK {out} {out.stat().st_size / 1e9:.3f}GB"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    repo_paint = "tencent/Hunyuan3D-2.1"
    repo_dino = "facebook/dinov2-giant"
    jobs = [(repo_paint, f"{PAINT_SUB}/{rel}", DEST / PAINT_SUB / rel) for rel in PAINT]
    jobs += [(repo_dino, rel, DEST / "dinov2-giant" / rel) for rel in DINO]
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(fetch, repo, rel, out) for repo, rel, out in jobs]
        for fut in futs:
            print(fut.result(), flush=True)
    print("DOWNLOAD_DONE", flush=True)


if __name__ == "__main__":
    main()
