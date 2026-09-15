"""E1 readiness (assignment 2026-09-14, work item 4):
native_mujoco/contact_accumulator.py.

Offline throughout: synthetic `ContactRecord`s exercise the accumulator's
own bookkeeping directly (no MuJoCo needed), and the one MuJoCo-touching
test only ever calls `mj_forward` on a compiled model -- never `mj_step`,
never a server.
"""

import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))

mujoco = pytest.importorskip("mujoco")

from contact_accumulator import ARM_CONTACT_GEOMS, ContactAccumulator  # noqa: E402
from evaluation_snapshot import ContactRecord  # noqa: E402
from simulation_core import contact_records  # noqa: E402
from objects import build_scene_model_xml  # noqa: E402
from scene_io import load_scene  # noqa: E402

_B2_SCENE = os.path.join(_HERE, "../../scenes/e1_boards/B2_foam_r3c3.yaml")
_ROBOT_MODEL = os.path.join(_HERE, "../../native_mujoco/model/reachy_1_2.xml")


def _record(geom1, geom2, body1, body2, dist=-0.001, force=1.0, pos=(0.0, 0.0, 0.0)):
    return ContactRecord(
        geom1=geom1, geom2=geom2, body1=body1, body2=body2,
        contype1=0, contype2=0, conaffinity1=0, conaffinity2=0,
        pos=pos, normal_force=force, dist=dist,
    )


class TestAccumulation:
    def test_two_steps_of_the_same_pair_tracks_steps_and_max_min(self):
        acc = ContactAccumulator()
        r1 = _record("r_finger_col", "foam_block_geom", "world", "foam_block",
                    dist=-0.002, force=1.0, pos=(1, 1, 1))
        r2 = _record("r_finger_col", "foam_block_geom", "world", "foam_block",
                    dist=-0.005, force=3.0, pos=(2, 2, 2))
        acc.add_step([r1], sim_step=10, object_ids=["foam_block"])
        acc.add_step([r2], sim_step=11, object_ids=["foam_block"])
        [pair] = acc.drain()
        assert pair["arm_geom"] == "r_finger_col"
        assert pair["object_id"] == "foam_block"
        assert pair["steps"] == 2
        assert pair["min_dist_m"] == pytest.approx(-0.005)
        assert pair["max_normal_force_n"] == pytest.approx(3.0)
        assert pair["pos_at_max_force"] == [2, 2, 2]
        assert pair["first_sim_step"] == 10
        assert pair["last_sim_step"] == 11

    def test_distinct_pairs_are_kept_apart(self):
        acc = ContactAccumulator()
        acc.add_step([
            _record("r_finger_col", "geom_a", "world", "foam_block"),
            _record("r_thumb_col", "geom_b", "world", "soda_can"),
        ], sim_step=1, object_ids=["foam_block", "soda_can"])
        pairs = {(p["arm_geom"], p["object_id"]) for p in acc.drain()}
        assert pairs == {("r_finger_col", "foam_block"), ("r_thumb_col", "soda_can")}

    def test_cleared_on_drain(self):
        acc = ContactAccumulator()
        acc.add_step([_record("r_finger_col", "g", "world", "foam_block")],
                    sim_step=1, object_ids=["foam_block"])
        first = acc.drain()
        assert len(first) == 1
        second = acc.drain()
        assert second == []

    def test_a_pair_present_on_one_step_only_still_emits(self):
        acc = ContactAccumulator()
        acc.add_step([_record("r_finger_col", "g", "world", "foam_block")],
                    sim_step=5, object_ids=["foam_block"])
        [pair] = acc.drain()
        assert pair["steps"] == 1
        assert pair["first_sim_step"] == pair["last_sim_step"] == 5

    def test_non_arm_geom_pairs_are_ignored(self):
        acc = ContactAccumulator()
        acc.add_step([_record("table_col", "g", "table", "foam_block")],
                    sim_step=1, object_ids=["foam_block"])
        assert acc.drain() == []

    def test_arm_arm_self_collision_pairs_are_ignored(self):
        acc = ContactAccumulator()
        acc.add_step([_record("r_finger_col", "r_thumb_col", "hand", "hand")],
                    sim_step=1, object_ids=["foam_block"])
        assert acc.drain() == []

    def test_arm_geom_against_untracked_body_is_ignored(self):
        """An arm geom touching something that is NOT in the tracked
        object_ids (a rail, the torso, the table body) is out of scope --
        assignment's own "out of scope" list."""
        acc = ContactAccumulator()
        acc.add_step([_record("r_forearm_col", "rail_geom", "world", "rig_rail_outer_right")],
                    sim_step=1, object_ids=["foam_block"])
        assert acc.drain() == []

    def test_reversed_geom_order_still_matches(self):
        """MuJoCo does not guarantee which side is geom1 -- the arm geom
        may be either side of the pair."""
        acc = ContactAccumulator()
        acc.add_step([_record("foam_block_geom", "r_finger_col", "foam_block", "world")],
                    sim_step=1, object_ids=["foam_block"])
        [pair] = acc.drain()
        assert pair["arm_geom"] == "r_finger_col"
        assert pair["object_id"] == "foam_block"

    def test_all_arm_contact_geoms_are_named(self):
        assert ARM_CONTACT_GEOMS == (
            "r_upper_arm_col", "r_forearm_col", "r_thumb_col", "r_finger_col")


class TestAgainstCompiledMjcf:
    """The one MuJoCo-touching pair of tests: mj_forward only, no
    mj_step, no server."""

    def test_penetrating_pose_yields_exactly_one_arm_object_pair(self):
        scene_doc = load_scene(_B2_SCENE)
        xml = build_scene_model_xml(scene_doc, _ROBOT_MODEL)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        finger_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_finger_col")
        finger_world_pos = data.geom_xpos[finger_gid].copy()

        # Move foam_block's free joint to sit exactly at the finger
        # collision geom's world position -- guarantees deep overlap
        # regardless of either geom's own orientation or extents.
        obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foam_block")
        obj_joint = model.body_jntadr[obj_body]
        qadr = model.jnt_qposadr[obj_joint]
        data.qpos[qadr:qadr + 3] = finger_world_pos
        data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

        records = contact_records(model, data)
        acc = ContactAccumulator()
        acc.add_step(records, sim_step=1, object_ids=["foam_block"])
        pairs = acc.drain()
        matches = [p for p in pairs
                  if p["arm_geom"] == "r_finger_col" and p["object_id"] == "foam_block"]
        assert len(matches) == 1, (
            f"expected exactly one r_finger_col/foam_block pair, got {pairs}")
        assert matches[0]["min_dist_m"] < 0.0

    def test_rest_pose_yields_no_arm_object_contact(self):
        scene_doc = load_scene(_B2_SCENE)
        xml = build_scene_model_xml(scene_doc, _ROBOT_MODEL)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)  # object at its board (r3c3) pose

        records = contact_records(model, data)
        acc = ContactAccumulator()
        acc.add_step(records, sim_step=1, object_ids=["foam_block"])
        pairs = acc.drain()
        assert pairs == []


class TestAddStepFromDataMatchesAddStep:
    """The production fast path (add_step_from_data) must produce the
    IDENTICAL result to the offline-testable path
    (add_step(contact_records(...))) it is meant to replace -- on the same
    model/data, penetrating and at rest."""

    def _model_and_data(self):
        scene_doc = load_scene(_B2_SCENE)
        xml = build_scene_model_xml(scene_doc, _ROBOT_MODEL)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def _penetrate(self, model, data):
        finger_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_finger_col")
        finger_world_pos = data.geom_xpos[finger_gid].copy()
        obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foam_block")
        obj_joint = model.body_jntadr[obj_body]
        qadr = model.jnt_qposadr[obj_joint]
        data.qpos[qadr:qadr + 3] = finger_world_pos
        data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

    def test_equivalent_when_penetrating(self):
        model, data = self._model_and_data()
        self._penetrate(model, data)

        slow = ContactAccumulator()
        slow.add_step(contact_records(model, data), sim_step=7,
                      object_ids=["foam_block"])
        fast = ContactAccumulator()
        fast.add_step_from_data(model, data, sim_step=7,
                                object_ids=["foam_block"])

        key = lambda p: (p["arm_geom"], p["object_id"])  # noqa: E731
        assert sorted(slow.drain(), key=key) == sorted(fast.drain(), key=key)

    def test_equivalent_at_rest(self):
        model, data = self._model_and_data()

        slow = ContactAccumulator()
        slow.add_step(contact_records(model, data), sim_step=1,
                      object_ids=["foam_block"])
        fast = ContactAccumulator()
        fast.add_step_from_data(model, data, sim_step=1,
                                object_ids=["foam_block"])

        assert slow.drain() == fast.drain() == []
