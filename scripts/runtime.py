"""Machine-facing runtime layer: interpreter resolution, hardware detection, memory profiles.

Why this exists
---------------
The pipeline used to hard-code ``conda run -n recon-vggt`` and CUDA-only assumptions, which
means a machine without conda (or without much VRAM) could not run any stage at all. This
module turns those into data:

* ``launcher(stage)``   -> the argv prefix used to start a stage (venv python, an absolute
  interpreter path, a conda env, or the current interpreter).
* ``detect()``          -> what the box actually has (torch build, GPU name, free VRAM/RAM/disk).
* ``resolve_profile()`` -> "full" / "low" / "tiny" memory ladder, either explicit or auto.
* ``profile_for()``     -> that profile's per-stage knobs from configs/runtime.json.
* ``child_env()``       -> environment overrides for the child process (allocator, threads).

configs/runtime.json holds the policy; nothing here reads the network or needs a GPU.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from common import ROOT, OUTPUTS

CONFIG_PATH = ROOT / "configs" / "runtime.json"
EVIDENCE_PATH = OUTPUTS / "hardware.json"
VALID_PROFILES = ("full", "low", "tiny")


def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


# --------------------------------------------------------------------------- hardware ---
def _windows_memory() -> tuple[float, float]:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullTotalPhys / 2**30, status.ullAvailPhys / 2**30


def _sys_memory() -> tuple[float, float]:
    try:
        pages = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        return pages / 2**30, avail_pages / 2**30
    except (ValueError, OSError):
        return 0.0, 0.0


def _torch_probe() -> dict:
    """Import torch only if present; never let detection crash a CPU-only box."""
    try:
        import torch
    except Exception as exc:  # torch absent or broken
        return {"torch": None, "torch_error": f"{type(exc).__name__}: {str(exc)[:120]}",
                "cuda_available": False, "device_name": None,
                "vram_total_mib": 0, "vram_free_mib": 0, "compute_capability": None}
    info = {"torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()),
            "device_name": None, "vram_total_mib": 0, "vram_free_mib": 0,
            "compute_capability": None}
    if info["cuda_available"]:
        try:
            free, total = torch.cuda.mem_get_info()
            info["vram_free_mib"] = int(free / 2**20)
            info["vram_total_mib"] = int(total / 2**20)
            info["device_name"] = torch.cuda.get_device_name(0)
            info["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        except Exception as exc:
            info["cuda_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return info


def detect(refresh: bool = False) -> dict:
    """Describe the machine. Cheap enough to call before every stage launch."""
    cache = getattr(detect, "_cache", None)
    if cache and not refresh:
        return cache
    if platform.system() == "Windows":
        ram_total, ram_avail = _windows_memory()
    else:
        ram_total, ram_avail = _sys_memory()
    try:
        disk_free = shutil.disk_usage(ROOT).free / 2**30
    except OSError:
        disk_free = 0.0
    hw = {"platform": platform.system(), "python": platform.python_version(),
          "exe": sys.executable, "ram_total_gib": round(ram_total, 1),
          "ram_available_gib": round(ram_avail, 1), "disk_free_gib": round(disk_free, 1),
          "conda": bool(shutil.which("conda")), "nvcc": bool(shutil.which("nvcc"))}
    hw.update(_torch_probe())
    detect._cache = hw
    return hw


def pick_device(preference: str = "auto", hw: dict | None = None) -> str:
    hw = hw or detect()
    if preference in ("cuda", "cpu"):
        if preference == "cuda" and not hw.get("cuda_available"):
            raise SystemExit("--device cuda requested but torch found no usable CUDA device "
                             f"({hw.get('torch_error') or hw.get('cuda_error') or 'cuda unavailable'})")
        return preference
    if preference not in ("auto", None):
        raise SystemExit(f"--device must be auto|cuda|cpu, got {preference!r}")
    return "cuda" if hw.get("cuda_available") else "cpu"


def pick_dtype(preference: str, device: str, hw: dict | None = None) -> str:
    hw = hw or detect()
    if preference != "auto":
        return preference
    if device == "cpu":
        return "float32"          # fp16/bf16 matmuls on CPU are unsupported or badly slow
    capability = hw.get("compute_capability")
    try:
        major = int(str(capability).split(".")[0])
    except (AttributeError, ValueError):
        major = 8
    return "bfloat16" if major >= 8 else "float16"


def auto_profile(device: str, hw: dict, config: dict | None = None) -> str:
    config = config or load_config()
    rules = config.get("auto_profile", {})
    if device == "cpu":
        return rules.get("cpu", "tiny")
    free = int(hw.get("vram_free_mib") or 0)
    for threshold, name in rules.get("vram_mib", []):
        if free >= int(threshold):
            return name
    return "tiny"


def profile_for(stage: str, profile: str, config: dict | None = None) -> dict:
    """Stage knobs of a named profile, merged over the 'full' baseline so nothing is None."""
    config = config or load_config()
    profiles = config.get("profiles", {})
    base = dict(profiles.get("full", {}).get(stage, {}))
    if profile and profile != "full":
        base.update(profiles.get(profile, {}).get(stage, {}))
    base.setdefault("profile", profile or "full")
    return base


# -------------------------------------------------------------------------- interpreter ---
def _candidate_python(explicit: str | None, config: dict) -> list[Path]:
    interpreters = config.get("interpreters", {})
    out: list[Path] = []
    if explicit:
        out.append(Path(explicit))
    for value in (os.environ.get("3DTEX_PYTHON"), os.environ.get("TEX3D_PYTHON")):
        if value:
            out.append(Path(value))
    out.extend(ROOT / rel for rel in interpreters.get("venv_candidates", []))
    return out


def launcher(stage: str, python: str | None = None, conda_env: str | None = None,
             config: dict | None = None) -> list[str]:
    """Argv prefix for a stage: absolute/venv python, else conda env, else this interpreter.

    ``python`` and ``conda_env`` come from CLI flags; configs/runtime.json supplies the
    per-machine defaults (conda env names on the Linux workstation, .venv on a laptop).
    """
    config = config or load_config()
    interpreters = config.get("interpreters", {})
    envs = interpreters.get("conda_envs", {})
    overrides = interpreters.get("python_overrides", {})
    prefer = interpreters.get("prefer", "auto")
    env = conda_env or os.environ.get("3DTEX_CONDA_ENV") or envs.get(stage)

    explicit = python or overrides.get(stage)
    if explicit:
        return [str(explicit)]
    if prefer in ("auto", "venv"):
        for candidate in _candidate_python(None, config):
            if candidate.is_file():
                return [str(candidate)]
    if prefer in ("auto", "conda") and env and shutil.which("conda"):
        return ["conda", "run", "--no-capture-output", "-n", env]
    if prefer == "system":
        return [sys.executable]
    # Last resort: run with whatever interpreter is executing this script, so a
    # single-env machine still works. Failures then come from the stage itself.
    return [sys.executable]


def child_env(device: str, threads: int | None = None, gpus: str | None = None,
              config: dict | None = None) -> dict:
    """Extra environment variables for the child process (not applied yet)."""
    config = config or load_config()
    env = dict(config.get("env", {}).get("common", {}))
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    if device == "cpu" or gpus == "":
        env["CUDA_VISIBLE_DEVICES"] = ""
    limit = threads or os.cpu_count() or 1
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(max(1, int(limit)))
    return env


def apply_env(extra: dict) -> None:
    for key, value in extra.items():
        os.environ[key] = str(value)


# --------------------------------------------------------------------------- launching ---
def check_vram_budget(required_mib: int, device: str, hw: dict, reserve: int = 512) -> tuple[bool, str]:
    """Compare a stage's estimated need with what is actually free right now."""
    if device != "cuda":
        return True, f"cpu stage: needs ~{required_mib / 1024:.1f} GiB host RAM (free {hw.get('ram_available_gib')} GiB)"
    free = int(hw.get("vram_free_mib") or 0)
    ok = free >= required_mib + reserve
    return ok, (f"needs ~{required_mib} MiB (+{reserve} reserve), {free} MiB free"
                f" on {hw.get('device_name') or 'GPU'}")


def record_evidence(stage: str, hw: dict, device: str, dtype: str, profile: str,
                    note: str = "") -> Path:
    """Append a machine-readable line of evidence; docs quote this file, not prose."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    entry = {"stage": stage, "device": device, "dtype": dtype, "profile": profile,
             "vram_free_mib": hw.get("vram_free_mib"), "vram_total_mib": hw.get("vram_total_mib"),
             "torch": hw.get("torch"), "gpu": hw.get("device_name"), "note": note}
    history = []
    if EVIDENCE_PATH.is_file():
        try:
            history = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            history = []
    history.append(entry)
    EVIDENCE_PATH.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return EVIDENCE_PATH


def describe(hw: dict | None = None) -> str:
    hw = hw or detect()
    gpu = hw.get("device_name") or "no CUDA device"
    return (f"{hw['platform']} py{hw['python']} | torch {hw.get('torch')} | {gpu} "
            f"(cap {hw.get('compute_capability')}) | VRAM {hw.get('vram_free_mib')}/{hw.get('vram_total_mib')} MiB free "
            f"| RAM {hw.get('ram_available_gib')}/{hw.get('ram_total_gib')} GiB | "
            f"disk {hw.get('disk_free_gib')} GiB | conda={hw.get('conda')} nvcc={hw.get('nvcc')}")


def add_arguments(parser, default_profile: str = "auto") -> None:
    """Shared hardware flags for every wrapper."""
    group = parser.add_argument_group("hardware/runtime")
    group.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    group.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    group.add_argument("--profile", choices=["auto", *VALID_PROFILES], default=default_profile,
                       help="memory/time ladder from configs/runtime.json (auto = fit to free VRAM)")
    group.add_argument("--python", default=None, help="interpreter for this stage (overrides conda/venv)")
    group.add_argument("--conda-env", default=None, help="explicit conda env name if you use conda")
    group.add_argument("--threads", type=int, default=None, help="CPU thread cap for the child process")
    group.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES value for the child process")
    group.add_argument("--print-hardware", action="store_true", help="show detection results and exit")


def resolve(args, stage: str) -> dict:
    """Turn parsed CLI args + config into the concrete launch decision for a stage."""
    if getattr(args, "print_hardware", False):
        hw = detect(refresh=True)
        print(describe(hw))
        print(json.dumps(hw, indent=2))
        raise SystemExit(0)
    config = load_config()
    hw = detect()
    device = pick_device(args.device, hw)
    profile = args.profile
    if profile == "auto":
        profile = auto_profile(device, hw, config)
    dtype = pick_dtype(args.dtype, device, hw)
    reserve = int(config.get("vram_reserve_mib", 512))
    return {"config": config, "hw": hw, "device": device, "dtype": dtype, "profile": profile,
            "reserve": reserve, "knobs": profile_for(stage, profile, config),
            "prefix": launcher(stage, getattr(args, "python", None), getattr(args, "conda_env", None), config),
            "env": child_env(device, getattr(args, "threads", None), getattr(args, "gpus", None), config)}


def conda(env: str, *args) -> list[str]:
    """Back-compatible prefix builder: maps legacy env names to stage launch decisions."""
    stage = {"recon-vggt": "vggt", "recon-3dgs": "3dgs", "2d-sugar": "2dsugar",
             "metric3d": "metric3d"}.get(env, env)
    return launcher(stage, conda_env=env) + [str(a) for a in args]


def selftest(argv=None) -> int:
    hw = detect(refresh=True)
    print(describe(hw))
    for device in ("cuda", "cpu"):
        try:
            picked = pick_device(device, hw) if device != "cuda" or hw.get("cuda_available") else "cpu"
        except SystemExit as exc:
            print(f"device={device:5s} -> refused: {exc}")
            continue
        profile = auto_profile(picked, hw)
        print(f"device={picked:4s} dtype={pick_dtype('auto', picked, hw):9s} auto_profile={profile:5s} "
              f"vggt={profile_for('vggt', profile).get('max_frames')} "
              f"3dgs={profile_for('3dgs', profile).get('steps')} "
              f"2dsugar={profile_for('2dsugar', profile).get('iterations')}")
    for stage in ("vggt", "3dgs", "2dsugar", "metric3d"):
        print(f"launcher[{stage:8s}] = {launcher(stage)}")
    return 0


if __name__ == "__main__":
    sys.exit(selftest())
