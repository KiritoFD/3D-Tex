"""Local Torch 2.14 build compatibility and actual CUDA dependency probe.

Not a Mesh Splatting training reproduction. Run in its dedicated WSL venv.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

TCNN_COMMIT = "3daa6e5b8b8e0fa17971a1a20acfbb3754494b38"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="recursive tiny-cuda-nn checkout")
    p.add_argument("--out", required=True)
    p.add_argument("--install", action="store_true")
    args = p.parse_args()
    repo = Path(args.repo).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if commit != TCNN_COMMIT:
        raise ValueError(f"Unreviewed tiny-cuda-nn revision: {commit}")
    setup = repo / "bindings/torch/setup.py"
    if args.install:
        source = setup.read_text().replace("cpp_standard = 17", "cpp_standard = 20")
        # Torch host bindings require C++20. Pure CUDA sources retain upstream
        # C++17: nvcc 12.8 hits fmt consteval errors and an internal compiler error at 20.
        source = source.replace('f"-std=c++{cpp_standard}",', '"-std=c++17",', 1)
        setup.write_text(source)
        env = dict(os.environ, MAX_JOBS="1", TCNN_CUDA_ARCHITECTURES="89",
                   CUDA_HOME="/usr/local/cuda")
        env["PATH"] = "/usr/local/cuda/bin:" + env["PATH"]
        env["LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + env.get("LIBRARY_PATH", "")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation",
                        "--no-deps", "."], cwd=setup.parent, env=env, check=True)
    import torch
    import torch_scatter
    import tinycudann
    x = torch.rand(1024, 3, device="cuda", requires_grad=True)
    encoding = tinycudann.Encoding(3, {"otype": "HashGrid", "n_levels": 4,
        "n_features_per_level": 2, "log2_hashmap_size": 12, "base_resolution": 16,
        "per_level_scale": 2.0})
    y = encoding(x)
    y.float().sum().backward()
    for grad in (x.grad, encoding.params.grad):
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    result = torch_scatter.scatter_mean(torch.tensor([1., 3., 2.], device="cuda"),
                                       torch.tensor([0, 0, 1], device="cuda"))
    torch.testing.assert_close(result, torch.tensor([2., 2.], device="cuda"))
    report = {"scope": "CUDA HashGrid and scatter only; not full training",
        "tcnn_commit": commit, "submodules": subprocess.check_output(
            ["git", "submodule", "status"], cwd=repo, text=True).strip().splitlines(),
        "setup_sha256": hashlib.sha256(setup.read_bytes()).hexdigest(),
        "setup_diff": subprocess.check_output(["git", "diff", "--", "bindings/torch/setup.py"],
                                               cwd=repo, text=True),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "encoding_shape": list(y.shape),
        "input_grad_norm": x.grad.norm().item(), "parameter_grad_norm": encoding.params.grad.norm().item(),
        "scatter_mean": result.tolist()}
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
