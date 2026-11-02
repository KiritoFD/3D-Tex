"""CPU-only unit tests for the runtime/hardware layer and the scene verifier math.

No GPU, no torch execution: runtime.detect() is stubbed with fake hardware dicts so these
assertions hold identically on a CI runner and on a laptop.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import runtime
import verify_scene

CUDA_8GIB = {"cuda_available": True, "vram_free_mib": 7000, "vram_total_mib": 8188,
             "compute_capability": "8.9", "device_name": "Fake 4070", "ram_available_gib": 12.0}
CPU_ONLY = {"cuda_available": False, "vram_free_mib": 0, "vram_total_mib": 0,
            "compute_capability": None, "device_name": None, "ram_available_gib": 12.0}


class TestDeviceAndProfile(unittest.TestCase):
    def test_auto_device_follows_torch(self):
        self.assertEqual(runtime.pick_device("auto", CPU_ONLY), "cpu")
        self.assertEqual(runtime.pick_device("auto", CUDA_8GIB), "cuda")

    def test_explicit_cuda_is_refused_when_absent(self):
        with self.assertRaises(SystemExit):
            runtime.pick_device("cuda", CPU_ONLY)

    def test_dtype_never_goes_low_on_cpu(self):
        self.assertEqual(runtime.pick_dtype("auto", "cpu", CPU_ONLY), "float32")
        self.assertEqual(runtime.pick_dtype("auto", "cuda", CUDA_8GIB), "bfloat16")
        old_gpu = dict(CUDA_8GIB, compute_capability="7.5")
        self.assertEqual(runtime.pick_dtype("auto", "cuda", old_gpu), "float16")

    def test_profile_ladder_by_free_vram(self):
        config = runtime.load_config()
        self.assertEqual(runtime.auto_profile("cpu", CPU_ONLY, config), "tiny")
        self.assertEqual(runtime.auto_profile("cuda", CUDA_8GIB, config), "low")
        self.assertEqual(runtime.auto_profile("cuda", dict(CUDA_8GIB, vram_free_mib=2000), config), "tiny")
        self.assertEqual(runtime.auto_profile("cuda", dict(CUDA_8GIB, vram_free_mib=24000), config), "full")

    def test_profiles_get_tighter_as_they_descend(self):
        config = runtime.load_config()
        frames = [runtime.profile_for("vggt", name, config)["max_frames"] for name in ("full", "low", "tiny")]
        self.assertEqual(frames[0], 0, "full must keep every view")
        self.assertGreater(frames[1], frames[2])
        self.assertLessEqual(frames[2], 4, "tiny must stay small enough for integrated GPUs")
        steps = [runtime.profile_for("3dgs", name, config)["steps"] for name in ("full", "low", "tiny")]
        self.assertEqual(steps, sorted(steps, reverse=True))

    def test_profile_merges_over_the_full_baseline(self):
        config = runtime.load_config()
        tiny = runtime.profile_for("3dgs", "tiny", config)
        self.assertIn("data_factor", tiny)      # inherited, not re-declared
        self.assertIn("steps", tiny)


class TestLauncher(unittest.TestCase):
    def test_venv_candidate_is_preferred_when_present(self):
        config = {"interpreters": {"prefer": "auto", "venv_candidates": ["__no_such__.venv/bin/python"],
                                   "conda_envs": {"vggt": "recon-vggt"}}}
        # Nothing matches on this box -> we must still return a usable interpreter rather than
        # silently falling back to a hard-coded conda env name.
        fallback = runtime.launcher("vggt", config=config)
        self.assertTrue(fallback, "launcher must never return an empty command")
        self.assertNotEqual(fallback[0], "conda", "conda must not be assumed when absent")

    def test_missing_venv_falls_back_to_running_interpreter(self):
        config = {"interpreters": {"prefer": "auto", "venv_candidates": ["__no_such__/python"],
                                   "conda_envs": {}}}
        self.assertEqual(runtime.launcher("3dgs", config=config), [sys.executable])

    def test_explicit_python_and_conda_env_override(self):
        config = {"interpreters": {"prefer": "auto", "venv_candidates": [], "conda_envs": {}}}
        self.assertEqual(runtime.launcher("3dgs", python="/opt/x/bin/python", config=config),
                         ["/opt/x/bin/python"])
        if runtime.shutil.which("conda"):
            self.assertEqual(runtime.launcher("3dgs", conda_env="recon-3dgs", config=config)[:3],
                             ["conda", "run", "--no-capture-output"])


class TestChildEnv(unittest.TestCase):
    def test_cpu_run_hides_the_gpu_and_caps_threads(self):
        env = runtime.child_env("cpu", threads=4)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["OMP_NUM_THREADS"], "4")

    def test_allocator_fragmentation_hint_is_inherited(self):
        env = runtime.child_env("cuda", threads=2, gpus="0", config=runtime.load_config())
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF", env)


class TestVramGuard(unittest.TestCase):
    def test_tight_budget_is_reported_not_silently_ignored(self):
        # 7000 MiB free, 6000 needed, 512 reserved -> 6512 fits, so this is tight-but-OK
        ok, detail = runtime.check_vram_budget(6000, "cuda", CUDA_8GIB, reserve=512)
        self.assertTrue(ok)
        self.assertIn("7000 MiB free", detail)
        # ask for more than exists and the guard must say no
        ok2, detail2 = runtime.check_vram_budget(8000, "cuda", CUDA_8GIB, reserve=512)
        self.assertFalse(ok2)
        self.assertIn("needs ~8000 MiB", detail2)
        self.assertTrue(runtime.check_vram_budget(1000, "cuda", CUDA_8GIB, reserve=512)[0])

    def test_cpu_stage_never_claims_gpu_memory(self):
        ok, detail = runtime.check_vram_budget(6000, "cpu", CPU_ONLY, reserve=512)
        self.assertTrue(ok)
        self.assertIn("host RAM", detail)


class TestVerifySceneMath(unittest.TestCase):
    def test_umeyama_recovers_a_known_similarity(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(9, 3))
        angle = 0.7
        rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        scale, truth_rot, truth_trans = 2.5, rot, np.array([1.0, -2.0, 0.5])
        dst = scale * src @ truth_rot.T + truth_trans
        s, r, t = runtime_umeyama(src, dst)
        self.assertAlmostEqual(s, scale, places=5)
        self.assertTrue(np.allclose(r, truth_rot, atol=1e-6))
        self.assertTrue(np.allclose(t, truth_trans, atol=1e-6))

    def test_quat_to_mat_is_orthonormal_and_matches_rotation_error(self):
        mat = verify_scene.quat_to_mat(np.array([1.0, 0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(mat, np.eye(3)))
        self.assertAlmostEqual(verify_scene.angle_deg(mat, mat), 0.0, places=6)
        quarter = verify_scene.quat_to_mat(np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]))
        self.assertAlmostEqual(verify_scene.angle_deg(np.eye(3), quarter), 90.0, places=3)


def runtime_umeyama(src, dst):
    return verify_scene.umeyama(src, dst)


if __name__ == "__main__":
    unittest.main()
