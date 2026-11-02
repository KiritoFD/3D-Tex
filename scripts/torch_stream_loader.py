"""Read a torch .pt checkpoint tensor-by-tensor without ever holding the whole state dict.

Why this exists
---------------
`torch.load()` on the 4.6 GiB VGGT-Omega checkpoint fails on two different platforms for two
different reasons, and both surface as a hard crash rather than an exception:

  * WSL DrvFs (/mnt/...)  -> OSError, because torch mmaps the zip archive;
  * native Windows        -> access violation inside `load_tensor`, reproducibly, once the
                             process already holds the 4.6 GiB fp32 model (peak host RAM then
                             approaches 9 GiB and the allocation fails deep inside an
                             unrelated later `nn.Linear`).

The zip layout torch writes is stable and simple: `data.pkl` holds the pickled structure and
`data/<n>` holds the raw tensor bytes. We read `data.pkl` alone (small), keep the storage-key ->
file mapping, then stream one `data/<n>` at a time. Peak host memory is one tensor.

    reader = CheckpointReader("model.pt")
    reader.names()                      # tensor names
    reader.get("aggregator.patch_embed.x")   # one tensor, float32 CPU
"""
from __future__ import annotations

import pickle
import struct
import zipfile
from pathlib import Path

import numpy as np
import torch

_DTYPE_SIZE = {
    "FloatStorage": 4, "DoubleStorage": 8, "HalfStorage": 2, "BFloat16Storage": 2,
    "LongStorage": 8, "IntStorage": 4, "ShortStorage": 2, "CharStorage": 1,
}


class _StorageType:
    def __init__(self, name: str):
        self.name = name


class _StorageRef:
    def __init__(self, storage_type: _StorageType, key: str):
        self.storage_type = storage_type
        self.key = key


class CheckpointReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._zip = zipfile.ZipFile(self.path)
        self._prefix = self._find_prefix()
        self._meta = self._read_pickle()

    def _find_prefix(self) -> str:
        for name in self._zip.namelist():
            if name.endswith("/data.pkl"):
                return name[: -len("data.pkl")]
        raise RuntimeError(f"{self.path} has no data.pkl")

    def _read_pickle(self) -> dict:
        """Unpickle the structure only; storage payloads become _StorageRef placeholders."""
        data = self._zip.read(f"{self._prefix}data.pkl")

        class Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name in _DTYPE_SIZE:
                    return _StorageType(name)
                if module.startswith("torch") and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                    return rebuild_tensor
                if module.startswith("collections") and name in ("OrderedDict",):
                    import collections
                    return collections.OrderedDict
                # only allow plain containers/scalars through
                return super().find_class(module, name)

            def persistent_load(self, pid):
                # torch's zip format stores ("storage", dtype, key, location, numel)
                if isinstance(pid, tuple) and pid and pid[0] == "storage":
                    return _StorageRef(pid[1], pid[2])
                raise pickle.UnpicklingError(f"unsupported persistent id: {pid!r}")

        return Unpickler(_BytesReader(data)).load()

    def names(self) -> list[str]:
        return list(self._meta.keys()) if isinstance(self._meta, dict) else []

    def get(self, key: str) -> torch.Tensor:
        spec = self._meta[key]
        if not isinstance(spec, _TensorSpec):
            raise TypeError(f"{key} is not a tensor in this checkpoint")
        return self._materialise(spec)

    def _materialise(self, spec: "_TensorSpec") -> torch.Tensor:
        # torch stores payloads as `<prefix>data/<n>`, while the persistent id carries only "n"
        raw = self._zip.read(f"{self._prefix}data/{spec.storage.key}")
        dtype = _torch_dtype(spec.storage.storage_type.name)
        offset = spec.storage_offset * _DTYPE_SIZE[spec.storage.storage_type.name]
        count = int(np.prod(spec.shape))
        buf = np.frombuffer(raw, dtype=_numpy_dtype(spec.storage.storage_type.name),
                            count=count, offset=offset)
        tensor = torch.from_numpy(buf.reshape(spec.shape).copy())
        if spec.stride != _contiguous_stride(spec.shape):
            tensor = tensor.as_strided(spec.shape, spec.stride, spec.storage_offset)
        return tensor.to(dtype)

    def close(self) -> None:
        self._zip.close()


class _TensorSpec:
    __slots__ = ("storage", "storage_offset", "shape", "stride")

    def __init__(self, storage, storage_offset, shape, stride):
        self.storage, self.storage_offset = storage, storage_offset
        self.shape, self.stride = shape, stride


def rebuild_tensor(storage, storage_offset, shape, stride, *_):
    return _TensorSpec(storage, storage_offset, tuple(shape), tuple(stride))


def _torch_dtype(name: str):
    return {"FloatStorage": torch.float32, "DoubleStorage": torch.float64,
            "HalfStorage": torch.float16, "BFloat16Storage": torch.bfloat16,
            "LongStorage": torch.int64, "IntStorage": torch.int32,
            "ShortStorage": torch.int16, "CharStorage": torch.int8}[name]


def _numpy_dtype(name: str) -> str:
    return {"FloatStorage": "<f4", "DoubleStorage": "<f8", "HalfStorage": "<f2",
            "BFloat16Storage": "<u2", "LongStorage": "<i8", "IntStorage": "<i4",
            "ShortStorage": "<i2", "CharStorage": "u1"}[name]


def _contiguous_stride(shape) -> tuple:
    stride, acc = [], 1
    for dim in reversed(tuple(shape)):
        stride.append(acc)
        acc *= dim
    return tuple(reversed(stride))


class _BytesReader:
    """Minimal file object so pickle can read from an in-memory buffer."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            chunk = self._data[self._pos:]
            self._pos = len(self._data)
            return chunk
        chunk = self._data[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk

    def readinto(self, target) -> int:
        chunk = self.read(len(target))
        target[: len(chunk)] = chunk
        return len(chunk)

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: len(self._data)}[whence]
        self._pos = base + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readline(self) -> bytes:
        end = self._data.find(b"\n", self._pos)
        if end < 0:
            end = len(self._data)
        line = self._data[self._pos:end + 1]
        self._pos = end + 1
        return line


def load_state_dict_streaming(checkpoint: str | Path, model) -> int:
    """Copy every matching tensor into `model`, holding only one tensor at a time."""
    reader = CheckpointReader(checkpoint)
    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    copied, unused, mismatched = 0, [], []
    try:
        for name in reader.names():
            if name not in targets:
                unused.append(name)
                continue
            value = reader.get(name)
            if tuple(value.shape) != tuple(targets[name].shape):
                mismatched.append((name, tuple(value.shape), tuple(targets[name].shape)))
                continue
            with torch.no_grad():
                targets[name].copy_(value)
            copied += 1
            del value
    finally:
        reader.close()
    if unused:
        print(f"[load] {len(unused)} checkpoint keys not in the model (e.g. {unused[:2]})", flush=True)
    if mismatched:
        raise RuntimeError(f"shape mismatch for {len(mismatched)} tensors: {mismatched[:2]}")
    return copied
