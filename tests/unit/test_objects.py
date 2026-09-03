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
        # robot (25, furniture-free) + 3 scene bodies
        assert model.nbody == 28

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


class TestWorldAppearance:
    """`world.background_rgba` and `world.floor.material` were in the scene
    format from the start and nothing read them, so FWDCenterLabMCC declared a
    pale lab and rendered as MuJoCo's blue demo checkerboard.  Cosmetic while
    people were watching the renders; not cosmetic once the frames became
    detector training data, where the background is most of every image.
    """

    BASE = (
        '<mujoco><asset>\n'
        '  <texture name="skybox" type="skybox" builtin="gradient"\n'
        '           rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>\n'
        '  <texture name="groundtex" type="2d" builtin="checker" mark="edge"\n'
        '           rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"\n'
        '           width="300" height="300"/>\n'
        '  <material name="groundplane" texture="groundtex" texuniform="true"\n'
        '            texrepeat="5 5" reflectance="0.2"/>\n'
        '</asset></mujoco>'
    )

    def test_the_background_repaints_the_sky(self):
        from objects import apply_world_appearance
        out = apply_world_appearance(
            self.BASE, {"world": {"background_rgba": [0.86, 0.88, 0.90, 1.0]}})
        assert 'rgb1="0.8600 0.8800 0.9000"' in out
        assert 'rgb1="0.3 0.5 0.7"' not in out

    def test_the_horizon_stays_darker_than_the_sky(self):
        """A flat flood reads as a void; a room is darker low in the view."""
        from objects import apply_world_appearance
        out = apply_world_appearance(
            self.BASE, {"world": {"background_rgba": [0.8, 0.8, 0.8, 1.0]}})
        sky = float(out.split('rgb1="')[1].split()[0])
        horizon = float(out.split('rgb2="')[1].split()[0])
        assert 0.0 < horizon < sky

    def test_the_floor_checker_goes_near_uniform(self):
        """The real lab floor is large pale tiles: a seam, not a chessboard."""
        from objects import apply_world_appearance
        out = apply_world_appearance(
            self.BASE,
            {"world": {"floor": {"material": {"rgba": [0.82, 0.83, 0.85, 1.0]}}}})
        block = out.split('name="groundtex"')[1].split("/>")[0]
        a = float(block.split('rgb1="')[1].split()[0])
        b = float(block.split('rgb2="')[1].split()[0])
        assert abs(a - b) < 0.06 and a > 0.5

    def test_roughness_becomes_reflectance(self):
        from objects import apply_world_appearance
        glossy = apply_world_appearance(
            self.BASE, {"world": {"floor": {"material": {"roughness": 0.15}}}})
        matte = apply_world_appearance(
            self.BASE, {"world": {"floor": {"material": {"roughness": 0.9}}}})
        g = float(glossy.split('reflectance="')[1].split('"')[0])
        m = float(matte.split('reflectance="')[1].split('"')[0])
        assert g > m

    def test_a_scene_with_no_world_block_is_untouched(self):
        from objects import apply_world_appearance
        assert apply_world_appearance(self.BASE, {"objects": []}) == self.BASE

    def test_a_partial_world_block_only_changes_what_it_names(self):
        from objects import apply_world_appearance
        out = apply_world_appearance(
            self.BASE, {"world": {"background_rgba": [0.9, 0.9, 0.9, 1.0]}})
        assert 'rgb1="0.2 0.3 0.4"' in out          # ground untouched
        assert 'reflectance="0.2"' in out


class TestFreejointsAreNamed:
    def test_every_dynamic_object_joint_is_addressable_by_id(self, model):
        """joint_name(obj_id) promised f"{id}__j" while the compiler emitted
        <freejoint/> unnamed, so mj_name2id returned -1 and every lookup that
        trusted the promise failed by doing nothing.  The dataset generator's
        placement randomiser was one: it skipped all four objects and rendered
        the scene's default layout for every frame of every run.
        """
        import mujoco
        from scene_compiler import joint_name

        free = [j for j in range(model.njnt)
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        assert free, "fixture scene has no dynamic objects to check"
        for jid in free:
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                     int(model.jnt_bodyid[jid]))
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            assert name == joint_name(body)
            assert mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name(body)) == jid
