"""Minimal bpy stub: Hunyuan3D only needs `import bpy` for its optional GLB export.

We run Paint with save_glb=False, so none of the Blender API is exercised.
"""
version = (4, 0, 0)


class _Unsupported:
    def __getattr__(self, name):
        raise RuntimeError("bpy stub: Blender export path is not available")


class _App:
    version = version


app = _App()
data = _Unsupported()
context = _Unsupported()
ops = _Unsupported()
