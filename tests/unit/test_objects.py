"""
R12-503: Dynamic object tracking tests.

Requires `mujoco` (native side); skipped in the Python-3.8 container / CI.
Covers scene->model composition, pose streaming, MuJoCo/stream coherence,
and deterministic seeded reset.
"""

import math
import os
import sys

import pytest

mujoco = pytest.importorskip("mujoco")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../native_mujoco"))
from objects import ObjectTracker, build_scene_model, build_scene_model_xml  # noqa: E402
from scene_compiler import tracked_object_ids  # noqa: E402


SCENE = {
    "objects": [
        {"id": "red_cube",
         "geometry": {"kind": "box", "size": [0.04, 0.04, 0.04]},
         "pose": {"position": [0.3, -0.1, 1.2], "rpy": [0, 0, 0.15]},
         "material": {"rgba": [0.9, 0.1, 0.1, 1]},
         "physics": {"dynamic": True, "collision": True, "mass": 0.05,
                     "friction": [0.8, 0.01, 0.001]},
         "tracked": True},
        {"id": "blue_cyl",
         "geometry": {"kind": "cylinder", "radius": 0.03, "height": 0.1},
         "pose": {"position": [0.3, 0.1, 1.2], "orientation_wxyz": [1, 0, 0, 0]},
         "material": {"rgba": [0.1, 0.1, 0.9, 1]},
         "physics": {"dynamic": True, "collision": True, "mass": 0.04},
         "tracked": True},
        {"id": "static_shelf",
         "geometry": {"kind": "box", "size": [0.4, 0.3, 0.02]},
         "pose": {"position": [0.5, 0, 0.8]},
         "material": {"rgba": [0.6, 0.4, 0.2, 1]},
         "physics": {"dynamic": False, "collision": True}},
    ]
}


@pytest.fixture(scope="module")
def model():
    return build_scene_model(SCENE)


@pytest.fixture
def sim(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    mujoco.mj_forward(model, d)
    tr = ObjectTracker(model, d, tracked_ids=tracked_object_ids(SCENE))
    tr.capture_initial(d)
    return model, d, tr


class TestModelComposition:
    def test_objects_added(self, model):
        # robot (26) + 3 scene bodies
        assert model.nbody == 29

    def test_dynamic_objects_have_freejoints(self, model):
        # 21 robot hinges + 2 free joints
        n_free = sum(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                     for j in range(model.njnt))
        assert n_free == 2

    def test_robot_still_valid(self, model):
        assert model.ncam == 2
        for name in ("red_cube", "blue_cyl", "static_shelf"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0


class TestTracking:
    def test_only_tracked_dynamic_objects(self, sim):
        _, _, tr = sim
        assert set(tr.object_ids) == {"red_cube", "blue_cyl"}
        assert "static_shelf" not in tr.object_ids

    def test_pose_coherent_with_mujoco_xpos(self, sim):
        m, d, tr = sim
        for p in tr.poses(d):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, p.object_id)
            assert np.allclose(p.pos_xyz, d.xpos[bid], atol=1e-9)

    def test_rpy_pose_becomes_quat(self, sim):
        _, d, tr = sim
        cube = next(p for p in tr.poses(d) if p.object_id == "red_cube")
        # yaw 0.15 -> w=cos(0.075), z=sin(0.075)
        assert cube.quat_wxyz[0] == pytest.approx(math.cos(0.075), abs=1e-3)
        assert cube.quat_wxyz[3] == pytest.approx(math.sin(0.075), abs=1e-3)

    def test_poses_as_dicts_shape(self, sim):
        _, d, tr = sim
        dicts = tr.poses_as_dicts(d)
        assert all({"object_id", "pos_xyz", "quat_wxyz"} <= set(x) for x in dicts)

    def test_objects_fall_under_gravity(self, sim):
        m, d, tr = sim
        z0 = tr.poses(d)[0].pos_xyz[2]
        for _ in range(600):
            mujoco.mj_step(m, d)
        z1 = tr.poses(d)[0].pos_xyz[2]
        assert z1 < z0  # fell


class TestSeededReset:
    def test_reset_restores_initial(self, sim):
        m, d, tr = sim
        start = tr.poses(d)[0].pos_xyz
        for _ in range(500):
            mujoco.mj_step(m, d)
        assert tr.poses(d)[0].pos_xyz[2] != pytest.approx(start[2], abs=1e-3)
        tr.reset(d)
        mujoco.mj_forward(m, d)
        assert np.allclose(tr.poses(d)[0].pos_xyz, start, atol=1e-9)

    def test_seeded_jitter_deterministic(self, sim):
        m, d, tr = sim
        tr.reset(d, seed=42, jitter_m=0.01)
        mujoco.mj_forward(m, d)
        a = tr.poses(d)[0].pos_xyz
        tr.reset(d, seed=42, jitter_m=0.01)
        mujoco.mj_forward(m, d)
        b = tr.poses(d)[0].pos_xyz
        assert np.allclose(a, b)

    def test_different_seed_differs(self, sim):
        m, d, tr = sim
        tr.reset(d, seed=1, jitter_m=0.02)
        mujoco.mj_forward(m, d)
        a = tr.poses(d)[0].pos_xyz
        tr.reset(d, seed=2, jitter_m=0.02)
        mujoco.mj_forward(m, d)
        b = tr.poses(d)[0].pos_xyz
        assert not np.allclose(a, b)

    def test_reset_zeroes_velocity(self, sim):
        m, d, tr = sim
        for _ in range(300):
            mujoco.mj_step(m, d)
        tr.reset(d, seed=7, jitter_m=0.005)
        # object dof velocities zeroed
        for o in tr._objects:
            assert np.allclose(d.qvel[o.qvel_adr:o.qvel_adr + 6], 0.0)
