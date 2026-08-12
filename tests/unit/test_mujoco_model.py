"""
R12-500: Collision and inertial audit tests for reachy_1_2.xml.

Requires the `mujoco` package (native macOS arm64 side).  Skipped gracefully
in the Python-3.8 Docker container / CI where mujoco is not installed.

Collision channel contract (bitmask: collide iff (ct_a & ca_b) | (ct_b & ca_a)):
    world  contype=1 conaffinity=6
    robot  contype=2 conaffinity=5
    object contype=4 conaffinity=7
  => robot never self-collides; robot collides with world and objects.
"""

import os

import pytest

mujoco = pytest.importorskip("mujoco")
import numpy as np  # noqa: E402

_MODEL = os.path.join(
    os.path.dirname(__file__), "../../native_mujoco/model/reachy_1_2.xml"
)


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(_MODEL)


@pytest.fixture
def data(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    return d


def _gid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _can_collide(model, a, b):
    ga, gb = _gid(model, a), _gid(model, b)
    return bool(
        (model.geom_contype[ga] & model.geom_conaffinity[gb])
        or (model.geom_contype[gb] & model.geom_conaffinity[ga])
    )


class TestModelLoads:
    def test_dof_count(self, model):
        assert model.nq == 21
        assert model.nu == 21

    def test_two_cameras(self, model):
        assert model.ncam == 2

    def test_home_keyframe_exists(self, model):
        assert model.nkey >= 1


class TestInertials:
    def test_total_mass_reasonable(self, model):
        # ~66 kg incl. static table + pedestal (auto-density); robot links > 0.
        assert model.body_mass.sum() > 10.0

    def test_all_inertias_positive_definite(self, model):
        # Every body's principal inertia must be strictly positive (the URDF
        # antenna failure that blocked Epic 4 must not reappear).
        for i in range(model.nbody):
            if model.body_mass[i] > 0:
                assert np.all(model.body_inertia[i] > 0), (
                    f"body {i} has non-positive inertia {model.body_inertia[i]}"
                )

    def test_key_link_masses(self, model):
        expected = {
            "r_upper_arm": (1.0, 1.5),
            "r_forearm": (0.8, 1.2),
            "torso": (11.9, 12.1),
        }
        for name, (lo, hi) in expected.items():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            assert lo <= model.body_mass[bid] <= hi


class TestVisualCollisionSeparation:
    def test_visual_geoms_have_no_collision(self, model):
        # Group-2 (visual) geoms must not participate in contacts.
        for i in range(model.ngeom):
            if model.geom_group[i] == 2:
                assert model.geom_contype[i] == 0
                assert model.geom_conaffinity[i] == 0

    def test_collision_geoms_add_no_mass(self, model):
        # Group-3 (collision) geoms are declared mass="0"; body inertia comes
        # from the visual geoms.  Proven by the invariant that total mass equals
        # the visual-only reference: had collision geoms carried density-based
        # mass, the total would rise.  Reference updated to 66.119 kg after the
        # tabletop surface was lowered to z=0.74 (legs shortened to match the
        # scene YAML / scene-awareness table height).
        assert model.body_mass.sum() == pytest.approx(66.119, abs=0.01)

    def test_collision_geoms_exist_for_major_links(self, model):
        for name in ("torso_col", "r_upper_arm_col", "r_forearm_col",
                     "r_thumb_col", "r_finger_col", "head_col"):
            assert _gid(model, name) >= 0, f"missing collision geom {name}"


class TestCollisionChannels:
    def test_robot_does_not_self_collide(self, model):
        assert not _can_collide(model, "r_upper_arm_col", "torso_col")
        assert not _can_collide(model, "r_finger_col", "l_finger_col")

    def test_robot_collides_with_world(self, model):
        assert _can_collide(model, "r_forearm_col", "floor")
        assert _can_collide(model, "torso_col", "tabletop")

    def test_robot_collision_contype(self, model):
        # robot collision geoms: contype=2, conaffinity=5
        gid = _gid(model, "r_finger_col")
        assert model.geom_contype[gid] == 2
        assert model.geom_conaffinity[gid] == 5

    def test_world_collision_contype(self, model):
        gid = _gid(model, "tabletop")
        assert model.geom_contype[gid] == 1
        assert model.geom_conaffinity[gid] == 6


class TestBaselineStability:
    def test_zero_baseline_contacts(self, model, data):
        mujoco.mj_forward(model, data)
        assert data.ncon == 0

    def test_long_run_no_nan(self, model, data):
        for _ in range(3000):  # 6 s
            mujoco.mj_step(model, data)
        assert not np.any(np.isnan(data.qpos))
        # Gravity sag from finite-gain hold should stay small.
        assert np.abs(data.qpos).max() < 0.2

    def test_no_self_collision_when_arm_driven_into_torso(self, model, data):
        data.ctrl[1] = -3.0  # r_shoulder_roll inward, sweeps toward torso
        self_hits = 0
        for _ in range(1000):
            mujoco.mj_step(model, data)
            for i in range(data.ncon):
                c = data.contact[i]
                if (model.geom_contype[c.geom1] == 2
                        and model.geom_contype[c.geom2] == 2):
                    self_hits += 1
        assert self_hits == 0
