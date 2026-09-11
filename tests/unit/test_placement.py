"""R12-607 / issue #38: runtime object pool and grid-cell placement.

Split deliberately into two tiers.  Everything that is geometry — cell
resolution, resting height, reachability, the refusal messages — is pure and
runs in the offline suite.  Only the tests that need contacts or a compiled
model import mujoco, so a CI box without a GL stack still checks the part most
likely to be wrong.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../native_mujoco"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from placement import (  # noqa: E402
    GRASP_APPROACH_M,
    UNREACHABLE_TAG,
    Cell,
    PlacementError,
    cells_from_scene,
    pool_ids,
    resolve_cell,
    shoulder_distance,
    yaw_to_quat,
)
from scene_io import load_scene  # noqa: E402

_SCENES = os.path.join(os.path.dirname(__file__), "../../scenes")
_POOL_SCENE = os.path.join(_SCENES, "FWDCenterLabSivaPool.yaml")
_SIVA_SCENE = os.path.join(_SCENES, "FWDCenterLabSiva.yaml")


@pytest.fixture(scope="module")
def pool_doc():
    return load_scene(_POOL_SCENE)


@pytest.fixture(scope="module")
def cells(pool_doc):
    return cells_from_scene(pool_doc)


class TestSceneInheritanceStillHolds:
    """The pool is a CHILD of FWDCenterLabSiva on purpose: the detector handoff
    set was rendered from that scene at seed 2026 and its README promises the
    set can be regenerated.  Adding bodies to it would break that quietly."""

    def test_the_dataset_scene_is_unchanged(self):
        doc = load_scene(_SIVA_SCENE)
        assert len(doc["objects"]) == 27
        assert not [o for o in doc["objects"] if "pool" in (o.get("tags") or [])]

    def test_the_pool_scene_inherits_everything_and_adds_six(self, pool_doc):
        assert len(pool_doc["objects"]) == 33
        # 6 new slots + the 4 inherited manipulables, now parked on the floor
        assert len(pool_ids(pool_doc)) == 10
        # inherited, not restated
        ids = {o["id"] for o in pool_doc["objects"]}
        assert {"table_top", "rig_rail_back", "cell_r2c2", "red_cube"} <= ids
        assert pool_doc["world"]["background_rgba"]

    def test_the_board_starts_empty(self, pool_doc):
        """FWDCenterLabSiva puts its four manipulables on the grid; this scene
        parks them, so anything on the board has a caller who put it there."""
        cells = cells_from_scene(pool_doc)
        for obj in pool_doc["objects"]:
            if "manipulable" not in (obj.get("tags") or []):
                continue
            x, y, _ = obj["pose"]["position"]
            for c in cells.values():
                assert not (abs(x - c.x) <= c.half_extent
                            and abs(y - c.y) <= c.half_extent), obj["id"]

    def test_the_pool_parks_on_the_floor_not_below_it(self, pool_doc):
        """A body stowed under an infinite ground plane is penetrating it and
        gets ejected — measured, one came back to z=+0.005 at 2.4 m/s."""
        for obj in pool_doc["objects"]:
            if "pool" not in (obj.get("tags") or []):
                continue
            assert obj["pose"]["position"][2] > 0.0, obj["id"]

    def test_the_pool_is_out_of_the_arms_reach_where_it_sits(self, pool_doc):
        for obj in pool_doc["objects"]:
            if "pool" not in (obj.get("tags") or []):
                continue
            x, y, z = obj["pose"]["position"]
            assert shoulder_distance(x, y, z) > 0.609, obj["id"]


class TestReachability:
    """Distances are computed; reachable/not is READ from the scene.

    MCC measured reachability per cell with 40-restart DLS IK.  This module
    reports the distance and defers the verdict, which also sidesteps what used
    to read as an inconsistency: MCC's header stated a 0.609 m "maximum" while
    its own table labelled cell_r3c3 at 0.622 m reachable and 0.649 m
    unreachable.  Issue #41 reproduced the sweep and confirmed the per-cell
    verdicts below are correct (reach is direction-dependent, not spherical);
    only the header's "maximum" framing was wrong, and it has been corrected.
    """

    # FWDCenterLabMCC's published shoulder-to-cell table.
    PUBLISHED = {
        "r1c1": 0.489, "r1c2": 0.398, "r1c3": 0.352,
        "r2c1": 0.590, "r2c2": 0.516, "r2c3": 0.482,
        "r3c1": 0.709, "r3c2": 0.649, "r3c3": 0.622,
    }

    def test_distances_still_match_the_published_table(self, cells):
        """Ties the code to the measurement: move the board and the recorded
        reachability sweep is silently invalidated, so fail here instead."""
        assert set(cells) == set(self.PUBLISHED)
        for name, expected in self.PUBLISHED.items():
            assert cells[name].shoulder_distance_m == pytest.approx(
                expected, abs=0.002), name

    def test_exactly_the_two_ik_verified_cells_are_unreachable(self, cells):
        assert {n for n, c in cells.items() if not c.reachable} == {"r3c1", "r3c2"}

    def test_the_approach_height_is_what_reproduces_the_table(self):
        """The sweep targeted 5 cm above the surface, not the surface."""
        assert GRASP_APPROACH_M == 0.05
        at_surface = shoulder_distance(0.5842, 0.1524, 0.741)
        assert at_surface < self.PUBLISHED["r3c1"]

    def test_an_unreachable_cell_is_refused_by_name_with_the_reason(self, cells):
        with pytest.raises(PlacementError) as exc:
            resolve_cell(cells, "r3c1")
        msg = str(exc.value)
        assert "r3c1" in msg and "71 cm" in msg and "UNREACHABLE" in msg

    def test_it_can_be_overridden_deliberately(self, cells):
        assert resolve_cell(cells, "r3c1", allow_unreachable=True).name == "r3c1"

    def test_an_unknown_cell_lists_the_ones_that_exist(self, cells):
        with pytest.raises(PlacementError) as exc:
            resolve_cell(cells, "r9c9")
        assert "r2c2" in str(exc.value)

    def test_both_the_short_name_and_the_object_id_resolve(self, cells):
        assert resolve_cell(cells, "cell_r2c2") is resolve_cell(cells, "r2c2")


class TestCellGeometry:
    def test_the_top_surface_is_the_centre_plus_half_the_thickness(self, cells):
        """Scene sizes are FULL extents; scene_compiler halves them for MJCF.
        Getting this backwards puts every object 0.5 mm into the board."""
        assert cells["r2c2"].top_z == pytest.approx(0.7405 + 0.0010 / 2)

    def test_cell_centres_match_the_measured_grid(self, cells):
        assert cells["r2c2"].x == pytest.approx(0.4318)
        assert cells["r2c2"].y == pytest.approx(0.0)
        # col 1 is the robot's LEFT (+y), col 3 its right
        assert cells["r1c1"].y > 0 > cells["r1c3"].y
        # row 1 is nearest the robot
        assert cells["r1c2"].x < cells["r3c2"].x

    def test_yaw_becomes_a_z_quaternion(self):
        w, x, y, z = yaw_to_quat(90.0)
        assert (x, y) == (0.0, 0.0)
        assert w == pytest.approx(0.70710678)
        assert z == pytest.approx(0.70710678)


class TestCellsFromScene:
    def test_a_cell_without_the_tag_is_not_a_cell(self):
        doc = {"objects": [{"id": "cell_r1c1", "tags": [],
                            "pose": {"position": [0, 0, 0]},
                            "geometry": {"size": [1, 1, 1]}}]}
        assert cells_from_scene(doc) == {}

    def test_a_tagged_object_that_is_not_named_like_a_cell_is_skipped(self):
        doc = {"objects": [{"id": "grid_tape_row_near", "tags": ["grid-cell"],
                            "pose": {"position": [0, 0, 0]},
                            "geometry": {"size": [1, 1, 1]}}]}
        assert cells_from_scene(doc) == {}

    def test_the_unreachable_tag_is_what_flips_it(self):
        base = {"id": "cell_r1c1", "pose": {"position": [0.3, 0.1, 0.74]},
                "geometry": {"size": [0.127, 0.127, 0.001]}}
        assert cells_from_scene(
            {"objects": [dict(base, tags=["grid-cell"])]})["r1c1"].reachable
        assert not cells_from_scene(
            {"objects": [dict(base, tags=["grid-cell", UNREACHABLE_TAG])]}
        )["r1c1"].reachable


# ── Everything below needs a compiled model ──────────────────────────────────

mujoco = pytest.importorskip("mujoco")
import numpy as np  # noqa: E402

from objects import build_scene_model_xml  # noqa: E402
from placement import ObjectPlacer  # noqa: E402


@pytest.fixture(scope="module")
def scene_xml(pool_doc):
    return build_scene_model_xml(pool_doc)


@pytest.fixture
def placer(scene_xml, pool_doc):
    """A FRESH model per test.  reshape() mutates geom_size/rgba/mass on the
    model itself, so a shared one would leak a resized object into whichever
    test ran next — and the compile is ~10 ms, which is not worth the coupling.
    """
    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    # Free joints start at their MJCF scene pose, which mj_resetData zeroes.
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            adr = model.jnt_qposadr[jid]
            data.qpos[adr:adr + 7] = model.qpos0[adr:adr + 7]
    mujoco.mj_forward(model, data)
    return ObjectPlacer(model, data, pool_doc), model, data


def _settle(model, data, steps=600):
    for _ in range(steps):
        mujoco.mj_step(model, data)


def _z_of(model, data, oid):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, oid)
    return float(data.xpos[bid][2])


def _touching(model, data, a, b):
    ga = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, a)
    gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])}
        if pair == {ga, gb}:
            return True
    return False


class TestPlacement:
    def test_every_pool_slot_is_placeable(self, placer):
        p, _, _ = placer
        assert set(p.pool_ids) <= set(p.placeable_ids)
        # the scene's own manipulables are placeable too — the tag advertises
        # spare slots, it does not restrict the API
        assert "red_cube" in p.placeable_ids

    def test_a_placed_object_rests_on_the_board(self, placer):
        p, model, data = placer
        p.place("pool_box_1", "r2c2")
        _settle(model, data)
        # contact, not just height: an object hovering 1 mm up also passes a
        # z check, and an object 1 mm inside the board passes it too
        assert _touching(model, data, "pool_box_1", "table_top")
        assert _z_of(model, data, "pool_box_1") == pytest.approx(0.741 + 0.02,
                                                                abs=0.004)

    def test_it_lands_over_the_cell_it_was_given(self, placer):
        p, model, data = placer
        cell = p.cells["r1c3"]
        p.place("pool_box_2", "r1c3")
        _settle(model, data)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pool_box_2")
        x, y, _ = data.xpos[bid]
        assert abs(float(x) - cell.x) < cell.half_extent
        assert abs(float(y) - cell.y) < cell.half_extent

    def test_placing_zeroes_any_velocity_it_had(self, placer):
        p, model, data = placer
        p.place("pool_cyl_1", "r1c1")
        _settle(model, data, steps=50)
        p.place("pool_cyl_1", "r2c1")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pool_cyl_1__j")
        adr = model.jnt_dofadr[jid]
        assert np.allclose(data.qvel[adr:adr + 6], 0.0)

    def test_an_unreachable_cell_is_refused(self, placer):
        p, _, _ = placer
        with pytest.raises(PlacementError, match="UNREACHABLE"):
            p.place("pool_box_1", "r3c1")

    def test_an_occupied_cell_is_refused_rather_than_exploding(self, placer):
        """Two bodies spawned into the same space start interpenetrating, and
        MuJoCo resolves that by launching them — which reads as a physics bug
        rather than as the caller's mistake."""
        p, model, data = placer
        p.place("pool_box_1", "r2c3")
        _settle(model, data, steps=200)
        with pytest.raises(PlacementError, match="already holds"):
            p.place("pool_box_2", "r2c3")
        assert p.place("pool_box_2", "r2c3", allow_occupied=True) is not None

    def test_replacing_the_same_object_on_its_own_cell_is_allowed(self, placer):
        p, model, data = placer
        p.place("pool_box_3", "r1c2")
        _settle(model, data, steps=200)
        p.place("pool_box_3", "r1c2", yaw_deg=45)   # must not raise

    def test_stow_returns_it_to_the_floor_and_it_stays(self, placer):
        p, model, data = placer
        p.place("pool_cyl_2", "r2c2")
        _settle(model, data, steps=200)
        home = p.stow("pool_cyl_2")
        assert home.stowed and home.cell is None
        _settle(model, data, steps=3000)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pool_cyl_2")
        x, y, z = (float(v) for v in data.xpos[bid])
        assert z > 0.0
        assert shoulder_distance(x, y, z) > 0.609   # still out of reach

    def test_an_unknown_object_names_the_ones_that_exist(self, placer):
        p, _, _ = placer
        with pytest.raises(PlacementError, match="pool_box_1"):
            p.place("no_such_thing", "r2c2")


class TestReshape:
    def test_size_colour_and_mass_are_writable_on_a_live_model(self, placer):
        p, model, data = placer
        gid = model.body_geomadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pool_box_1")]
        p.reshape("pool_box_1", size=[0.08, 0.08, 0.08],
                  rgba=[0.9, 0.2, 0.1, 1.0], mass=0.3)
        assert list(model.geom_size[gid][:3]) == pytest.approx([0.04, 0.04, 0.04])
        assert list(model.geom_rgba[gid]) == pytest.approx([0.9, 0.2, 0.1, 1.0])

    def test_growing_a_geom_updates_rbound_so_it_still_collides(self, placer):
        """THE TRAP.  geom_rbound is the broadphase bounding radius, computed by
        the compiler.  It is NOT recomputed when geom_size changes and
        mj_setConst does not fix it (verified: 0.015 -> 0.05 left rbound at
        0.02598).  Grow a geom without updating rbound and the broadphase culls
        the pair before narrowphase runs, so contacts silently stop being
        generated — presenting as the gripper passing through the object, with
        nothing in any log."""
        p, model, data = placer
        gid = model.body_geomadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pool_box_3")]
        p.reshape("pool_box_3", size=[0.10, 0.10, 0.10])
        assert model.geom_rbound[gid] == pytest.approx(0.05 * 3 ** 0.5)
        p.place("pool_box_3", "r1c1")
        _settle(model, data)
        assert _touching(model, data, "pool_box_3", "table_top")

    def test_resting_height_follows_the_new_size(self, placer):
        """Half-height is read from the LIVE model, not the scene doc, so a
        reshaped object still lands on the board instead of inside it."""
        p, model, data = placer
        p.reshape("pool_box_2", size=[0.10, 0.10, 0.10])
        p.place("pool_box_2", "r2c1")
        _settle(model, data)
        assert _z_of(model, data, "pool_box_2") == pytest.approx(0.741 + 0.05,
                                                                 abs=0.005)

    def test_a_cylinder_resizes_by_radius_and_length(self, placer):
        p, model, data = placer
        gid = model.body_geomadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pool_cyl_3")]
        p.reshape("pool_cyl_3", radius=0.03, length=0.12)
        assert model.geom_size[gid][0] == pytest.approx(0.03)
        assert model.geom_size[gid][1] == pytest.approx(0.06)
        assert model.geom_rbound[gid] == pytest.approx((0.03 ** 2 + 0.06 ** 2) ** 0.5)

    def test_size_on_a_cylinder_is_refused(self, placer):
        """geom_size means different things per primitive: [hx, hy, hz] for a
        box but [radius, half-length] for a cylinder.  Writing `size` to a
        cylinder sets its radius from hx and its half-length from hy — a
        plausible-looking object of entirely the wrong shape."""
        p, _, _ = placer
        with pytest.raises(PlacementError, match="not a box"):
            p.reshape("pool_cyl_1", size=[0.04, 0.04, 0.04])

    def test_radius_on_a_box_is_refused(self, placer):
        p, _, _ = placer
        with pytest.raises(PlacementError, match="is a box"):
            p.reshape("pool_box_1", radius=0.03)

    def test_bad_arguments_are_refused(self, placer):
        p, _, _ = placer
        with pytest.raises(PlacementError):
            p.reshape("pool_box_1", size=[0.1, 0.1])
        with pytest.raises(PlacementError):
            p.reshape("pool_box_1", rgba=[1, 1, 1])
        with pytest.raises(PlacementError):
            p.reshape("pool_box_1", mass=0.0)


class TestServerPath:
    """The route a client actually takes: submit from the websocket task, apply
    on the sim thread, ack back.  Placement writes qpos and mutates the model,
    so doing it inline in the recv handler would race a step in progress — the
    same reason zoom_command defers to the render thread."""

    @pytest.fixture
    def sim_state(self, scene_xml, pool_doc):
        from server import SimState
        model = mujoco.MjModel.from_xml_string(scene_xml)
        return SimState(model, scene_doc=pool_doc)

    def test_a_placement_survives_the_round_trip(self, sim_state):
        sim_state.submit_place({"object_id": "pool_box_1", "cell": "r2c2",
                                "request_id": "abc"})
        assert sim_state.drain_place_results() == []   # not applied yet
        sim_state.apply_pending()
        res = sim_state.drain_place_results()
        assert len(res) == 1 and res[0]["accepted"]
        assert res[0]["request_id"] == "abc"
        assert res[0]["placement"]["cell"] == "r2c2"

    def test_several_placements_in_one_tick_all_run(self, sim_state):
        """A list, not a slot.  Joint commands may coalesce because only the
        newest target matters; setting up a board is a sequence of events, and
        dropping all but the last would silently build the wrong scene."""
        for i, cell in enumerate(("r1c1", "r1c2", "r1c3")):
            sim_state.submit_place({"object_id": f"pool_box_{i+1}",
                                    "cell": cell, "request_id": str(i)})
        sim_state.apply_pending()
        res = sim_state.drain_place_results()
        assert [r["placement"]["cell"] for r in res] == ["r1c1", "r1c2", "r1c3"]

    def test_a_refusal_comes_back_as_an_ack_not_an_exception(self, sim_state):
        """A bad placement must not take the sim thread down with it."""
        sim_state.submit_place({"object_id": "pool_box_1", "cell": "r3c1",
                                "request_id": "z"})
        sim_state.apply_pending()
        res = sim_state.drain_place_results()
        assert not res[0]["accepted"] and "UNREACHABLE" in res[0]["error"]

    def test_stow_is_a_null_cell(self, sim_state):
        sim_state.submit_place({"object_id": "pool_box_1", "cell": None})
        sim_state.apply_pending()
        assert sim_state.drain_place_results()[0]["placement"]["stowed"]

    def test_reshape_rides_along_with_the_placement(self, sim_state):
        sim_state.submit_place({"object_id": "pool_box_1", "cell": "r2c2",
                                "reshape": {"size": [0.1, 0.1, 0.1],
                                            "rgba": [1, 0, 0, 1]}})
        sim_state.apply_pending()
        assert sim_state.drain_place_results()[0]["accepted"]
        gid = sim_state.model.body_geomadr[mujoco.mj_name2id(
            sim_state.model, mujoco.mjtObj.mjOBJ_BODY, "pool_box_1")]
        assert sim_state.model.geom_size[gid][2] == pytest.approx(0.05)

    def test_an_unknown_reshape_key_is_refused_rather_than_ignored(self, sim_state):
        """Silently dropping a key the caller believed in is how you get a
        'why did nothing happen' bug with nothing in the log."""
        sim_state.submit_place({"object_id": "pool_box_1", "cell": "r2c2",
                                "reshape": {"colour": [1, 0, 0, 1]}})
        sim_state.apply_pending()
        res = sim_state.drain_place_results()
        assert not res[0]["accepted"]
        assert "colour" in res[0]["error"] and "rgba" in res[0]["error"]

    def test_a_child_scene_resolves_itself_when_validated(self):
        """Every caller gets inheritance, not just the ones that know to ask.

        ros/scene_marker_publisher.py draws RViz markers through scene_loader
        and cannot reach the resolver on its own; without this, an inherited
        scene renders its own objects with no board under them.
        """
        from scene_loader import load_scene as validate
        direct = validate(_POOL_SCENE)
        assert len(direct.objects) == 33            # the parent's, not just 10
        assert any(o.id == "table_top" for o in direct.objects)
        # and passing an already-resolved document agrees with it
        assert len(validate(_POOL_SCENE, document=load_scene(_POOL_SCENE)).objects) == 33

    def test_each_result_names_the_connection_that_asked(self, sim_state):
        """The ack has to go back to the client that requested it.

        The server takes concurrent connections, each with its own send loop.
        A single shared ack queue meant whichever loop polled first took the
        ack — a notebook's placement ack delivered to the Docker bridge, which
        discards it, leaving the notebook waiting forever for a placement that
        had already happened.  Observed live before this was routed.
        """
        sim_state.submit_place({"object_id": "pool_box_1", "cell": "r2c2",
                                "_conn_id": 7, "request_id": "a"})
        sim_state.submit_place({"object_id": "pool_box_2", "cell": "r1c1",
                                "_conn_id": 9, "request_id": "b"})
        sim_state.apply_pending()
        got = {r["request_id"]: r["_conn_id"]
               for r in sim_state.drain_place_results()}
        assert got == {"a": 7, "b": 9}

    def test_a_refusal_also_names_the_connection(self, sim_state):
        """Otherwise a rejected placement is the one that hangs the client."""
        sim_state.submit_place({"object_id": "pool_box_1", "cell": "r3c1",
                                "_conn_id": 4, "request_id": "z"})
        sim_state.apply_pending()
        res = sim_state.drain_place_results()[0]
        assert res["_conn_id"] == 4 and not res["accepted"]

    def test_the_pending_queue_is_capped(self, sim_state):
        """Network-facing: a client submitting faster than the sim drains would
        otherwise grow the list without bound."""
        from server import _MAX_PENDING_PLACES
        for i in range(_MAX_PENDING_PLACES + 5):
            sim_state.submit_place({"object_id": "pool_box_1", "cell": "r2c2",
                                    "request_id": str(i)})
        rejected = [r for r in sim_state.drain_place_results()
                    if "queue full" in r.get("error", "")]
        assert len(rejected) == 5

    def test_a_server_with_no_scene_says_so(self, scene_xml):
        from server import SimState
        model = mujoco.MjModel.from_xml_string(scene_xml)
        bare = SimState(model)                 # no scene_doc
        bare.submit_place({"object_id": "pool_box_1", "cell": "r2c2"})
        bare.apply_pending()
        res = bare.drain_place_results()
        assert not res[0]["accepted"] and "without a scene" in res[0]["error"]

    def test_the_ack_carries_the_step_it_landed_on(self, sim_state):
        """What makes this a perception check rather than a demo: a client that
        knows the step can line the next camera frame up with a placement whose
        pose it already knows."""
        from protocol import PlaceAck
        sim_state.submit_place({"object_id": "pool_box_1", "cell": "r2c2"})
        sim_state.apply_pending()
        res = sim_state.drain_place_results()[0]
        res.pop("_conn_id")
        ack = PlaceAck(sim_step=sim_state.step, **res)
        assert ack.accepted and ack.sim_step == sim_state.step
        assert ack.encode()


class TestResetAckRouting:
    """#43 / #84: reset_ack had the identical bug place_ack was fixed for —
    one server-level queue drained by every connection's send loop, so
    whichever loop polled first took the ack regardless of who asked. Mirrors
    TestServerPath's place_ack routing tests, against the same contract:
    apply_pending() must carry the submitting connection through so the ack
    can be routed rather than broadcast.
    """

    @pytest.fixture
    def sim_state(self, scene_xml, pool_doc):
        from server import SimState
        model = mujoco.MjModel.from_xml_string(scene_xml)
        return SimState(model, scene_doc=pool_doc)

    def test_reset_info_names_the_connection_that_asked(self, sim_state):
        sim_state.submit_reset({"request_id": "r1", "_conn_id": 7})
        info = sim_state.apply_pending()
        assert info == {"request_id": "r1", "_conn_id": 7}

    def test_a_second_reset_names_its_own_connection(self, sim_state):
        """Two clients, sequential resets — each ack must carry the asker
        that requested it, not whichever connection happened to be first."""
        sim_state.submit_reset({"request_id": "r1", "_conn_id": 7})
        first = sim_state.apply_pending()
        sim_state.submit_reset({"request_id": "r2", "_conn_id": 9})
        second = sim_state.apply_pending()
        assert first["_conn_id"] == 7 and second["_conn_id"] == 9
        assert first["request_id"] == "r1" and second["request_id"] == "r2"

    def test_no_pending_reset_returns_none(self, sim_state):
        assert sim_state.apply_pending() is None

    def test_reset_without_a_conn_id_still_reports_its_request_id(self, sim_state):
        """A reset submitted without _conn_id (e.g. a unit test constructing
        the message directly) must not raise — the ack is simply unroutable,
        which server.py's sim-thread loop already treats as a dropped receipt,
        not an error."""
        sim_state.submit_reset({"request_id": "bare"})
        info = sim_state.apply_pending()
        assert info == {"request_id": "bare", "_conn_id": None}


class TestBroadcastFanOut:
    """State and camera frames are broadcasts: every client is entitled to all
    of them.  They were single shared queues drained by whichever send loop
    polled first, which SPLIT the stream — measured on a live server, one client
    got 13.2 camera_frame/s and two got 7.8 each.  So opening a browser panel or
    a second notebook silently halved the Docker bridge's frame rate.
    """

    @staticmethod
    def _run(coro):
        import asyncio
        return asyncio.new_event_loop().run_until_complete(coro)

    def _server(self):
        from server import ReachyMujocoServer
        return ReachyMujocoServer.__new__(ReachyMujocoServer)

    def test_every_client_gets_every_message(self):
        import asyncio
        srv = self._server()
        qs = {1: asyncio.Queue(maxsize=4), 2: asyncio.Queue(maxsize=4)}
        self._run(srv._broadcast(qs, "frame-a"))
        self._run(srv._broadcast(qs, "frame-b"))
        assert [qs[1].get_nowait(), qs[1].get_nowait()] == ["frame-a", "frame-b"]
        assert [qs[2].get_nowait(), qs[2].get_nowait()] == ["frame-a", "frame-b"]

    def test_a_full_queue_drops_the_oldest_not_the_newest(self):
        """A late frame is worth less than the current one."""
        import asyncio
        srv = self._server()
        qs = {1: asyncio.Queue(maxsize=2)}
        for msg in ("one", "two", "three"):
            self._run(srv._broadcast(qs, msg))
        assert [qs[1].get_nowait(), qs[1].get_nowait()] == ["two", "three"]

    def test_a_slow_client_cannot_stall_the_others(self):
        """The old shared queue used a blocking put(), so one wedged consumer
        backed the producer up for everyone."""
        import asyncio
        srv = self._server()
        slow, fast = asyncio.Queue(maxsize=1), asyncio.Queue(maxsize=8)
        qs = {1: slow, 2: fast}
        for i in range(5):
            self._run(srv._broadcast(qs, i))       # must not hang
        assert fast.qsize() == 5
        assert slow.qsize() == 1 and slow.get_nowait() == 4

    def test_no_clients_is_not_an_error(self):
        srv = self._server()
        self._run(srv._broadcast({}, "frame"))
