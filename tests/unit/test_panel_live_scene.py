"""Issue #50: live object poses, destination references, and invalidation.

Two tiers again.  The snapshot ingest and the geometry are pure; the tests at
the bottom load the real FWDCenterLabSivaPool document, because "a can in the
pool is not a tabletop candidate" is a claim about where that file parks its
pool objects relative to where it puts its grid.
"""

import math
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../web"))
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))

from panel_planner import (  # noqa: E402
    POSE_TOLERANCE_M,
    DeterministicPlanner,
    LiveProposalValidator,
)
from panel_scene import (  # noqa: E402
    LEGACY_DESTINATIONS,
    CellView,
    DestinationRef,
    SceneView,
    apply_snapshot,
    scene_view_from_doc,
)
from panel_sim_link import SimLink, SimSnapshot  # noqa: E402
from tasks import ConversationEvent, PlannerRequest  # noqa: E402


class _Cell:
    """Stand-in for placement.Cell, matching the attributes read from it."""

    def __init__(self, name, x, y, reachable=True):
        self.name = name
        self.x = x
        self.y = y
        self.top_z = 0.80
        self.half_extent = 0.06
        self.reachable = reachable
        self.shoulder_distance_m = 0.4


#: A 3x3 grid at 0.15 m spacing, with the two measured-unreachable cells.
def _cells():
    out = {}
    for r in (1, 2, 3):
        for c in (1, 2, 3):
            name = f"r{r}c{c}"
            out[name] = _Cell(name, 0.20 + 0.15 * (r - 1), 0.15 * (c - 2),
                              reachable=name not in ("r3c1", "r3c2"))
    return out


def make_scene(*, destinations=False):
    doc = {
        "name": "TestScene",
        "objects": [
            {"id": "soda_can", "semantic_class": "can",
             "tags": ["pool", "manipulable", "pickable", "recyclable"]},
            {"id": "foam_block", "semantic_class": "block",
             "tags": ["pool", "manipulable", "pickable", "non-recyclable"]},
            {"id": "red_cube", "semantic_class": "cube",
             "tags": ["pool", "manipulable", "pickable"]},
        ],
    }
    if destinations:
        doc["objects"].append(
            {"id": "left_tray", "semantic_class": "tray",
             "tags": ["tray", "destination", "placement-target"]}
        )
    return scene_view_from_doc(
        doc, _cells(), placeable=["soda_can", "foam_block", "red_cube"]
    )


def snap(**objects):
    return SimSnapshot(
        sim_step=1000, sim_time_s=2.0, scene_revision="rev-1", seq=7,
        objects=dict(objects), received_at=__import__("time").monotonic(),
    )


def on_cell(scene, name, z_offset=0.03):
    c = scene.cells[name]
    return (c.x, c.y, c.top_z + z_offset)


IN_POOL = (0.55, 0.95, 0.0575)      # where the pool scene parks objects


def plan(scene, text, answers=()):
    history = [ConversationEvent(role="user", text=text)]
    for a in answers:
        history.append(ConversationEvent(role="reachy", text="?", question_id="q"))
        history.append(ConversationEvent(role="user", text=a))
    latest = answers[-1] if answers else text
    return DeterministicPlanner(lambda: scene)(
        PlannerRequest(text=latest, history=history)
    )


# ---------------------------------------------------------------------------
# Snapshot ingest
# ---------------------------------------------------------------------------

def test_ingest_keeps_object_poses():
    link = SimLink()
    link._ingest_state({
        "sim_step": 42, "sim_time_s": 0.084, "scene_revision": "rev-1", "seq": 3,
        "objects": [{"object_id": "soda_can", "pos_xyz": [0.1, 0.2, 0.83]}],
    })
    got = link.snapshot()
    assert got.objects == {"soda_can": (0.1, 0.2, 0.83)}
    assert got.sim_step == 42
    assert got.scene_revision == "rev-1"


@pytest.mark.parametrize("pos", [
    [float("nan"), 0.0, 0.8],
    [0.0, float("inf"), 0.8],
    [0.0, 0.0, float("-inf")],
    [0.0, 0.0],                 # wrong arity
    "not a list",
    None,
])
def test_nonfinite_or_malformed_poses_are_dropped_not_passed_on(pos):
    link = SimLink()
    link._ingest_state({"objects": [
        {"object_id": "soda_can", "pos_xyz": pos},
        {"object_id": "red_cube", "pos_xyz": [0.1, 0.1, 0.8]},
    ]})
    got = link.snapshot()
    assert "soda_can" not in got.objects
    assert got.dropped_objects == ("soda_can",)
    assert got.objects["red_cube"] == (0.1, 0.1, 0.8)


def test_a_stale_snapshot_is_reported_as_no_snapshot():
    link = SimLink(stale_after_s=0.0)
    link._ingest_state({"objects": []})
    # Never present an old world as the current one.
    assert link.snapshot() is None
    assert link.status["live"] is False


def test_status_without_a_snapshot_is_not_live():
    link = SimLink()
    assert link.snapshot() is None
    status = link.status
    assert status["live"] is False
    # "never_started", not "stopped": the request that lazily builds the panel
    # also starts the link, so a link that is coming up must not report itself
    # as one that has been shut down.
    assert status["state"] == "never_started"


def test_the_link_never_sends_anything_but_hello_and_heartbeat_ack():
    """A read-only link is a claim about the source, so check the source."""
    import pathlib
    src = pathlib.Path(_HERE, "../../web/panel_sim_link.py").read_text()
    for forbidden in ("place_object", "joint_command", '"reset"', '"pause"',
                      "scene_load"):
        # They may appear in prose explaining why they are absent; what must
        # not appear is a send of one.
        assert f'"type": {forbidden}' not in src
        assert f"'type': {forbidden}" not in src


# ---------------------------------------------------------------------------
# Applying a snapshot
# ---------------------------------------------------------------------------

def test_apply_snapshot_places_objects_on_cells():
    scene = make_scene()
    apply_snapshot(scene, snap(soda_can=on_cell(scene, "r2c2"),
                               foam_block=IN_POOL))
    assert scene.live is True
    assert scene.objects["soda_can"].on_board is True
    assert scene.objects["soda_can"].cell == "r2c2"
    assert scene.objects["foam_block"].on_board is False
    assert scene.objects["foam_block"].cell is None
    assert scene.cells["r2c2"].occupant == "soda_can"
    assert scene.cells["r1c1"].occupant is None


def test_an_object_missing_from_the_snapshot_stays_unknown():
    scene = make_scene()
    apply_snapshot(scene, snap(soda_can=on_cell(scene, "r2c2")))
    # Not reported is not the same as not there.
    assert scene.objects["red_cube"].on_board is None
    assert scene.objects["red_cube"].cell is None


def test_no_snapshot_leaves_the_view_unknown_and_not_live():
    scene = make_scene()
    apply_snapshot(scene, None)
    assert scene.live is False
    assert all(o.on_board is None for o in scene.objects.values())


def test_cell_contains_matches_the_simulator_predicate():
    cell = CellView(name="r2c2", x=0.5, y=0.0, top_z=0.8, half_extent=0.06)
    assert cell.contains((0.5, 0.0, 0.83))
    assert cell.contains((0.559, 0.059, 0.83))       # inside the extent
    assert not cell.contains((0.57, 0.0, 0.83))      # outside in x
    assert not cell.contains((0.5, 0.0, 0.78))       # below the surface
    assert cell.contains((0.5, 0.0, 0.795))          # the 1 cm tolerance


def test_available_cells_excludes_occupied_and_unreachable():
    scene = make_scene()
    apply_snapshot(scene, snap(soda_can=on_cell(scene, "r2c2")))
    available = scene.available_cells()
    assert "r2c2" not in available          # occupied
    assert "r3c1" not in available          # measured unreachable
    assert "r1c1" in available
    # Before a snapshot, occupancy is unknown and nothing is excluded for it.
    assert "r2c2" in make_scene().available_cells()


# ---------------------------------------------------------------------------
# Destination references and the legacy adapter
# ---------------------------------------------------------------------------

def test_a_grid_cell_has_no_legacy_destination():
    ref = DestinationRef(kind="cell", ref_id="r2c2")
    # None is the answer.  Substituting left_tray here is the specific bug the
    # design brief calls out.
    assert ref.legacy_destination() is None
    assert ref.as_str() == "cell:r2c2"
    assert ref.describe() == "grid cell r2c2"


def test_a_tagged_tray_maps_onto_the_legacy_enum():
    ref = DestinationRef(kind="destination", ref_id="left_tray", label="tray")
    assert ref.legacy_destination() == "left_tray"
    assert ref.as_str() == "destination:left_tray"


def test_a_destination_outside_the_legacy_enum_maps_to_none():
    ref = DestinationRef(kind="destination", ref_id="recycling_bin")
    assert ref.legacy_destination() is None
    assert "recycling_bin" not in LEGACY_DESTINATIONS


def test_proposal_carries_the_full_contract():
    scene = make_scene()
    apply_snapshot(scene, snap(soda_can=on_cell(scene, "r1c1")))
    p = plan(scene, "put soda_can on r2c2").proposal
    assert p.destination == "cell:r2c2"
    assert p.destination_kind == "cell"
    assert p.destination_label == "grid cell r2c2"
    assert p.legacy_destination is None
    assert p.task_type == "pick_place"
    assert p.semantic_source == "scene_data"
    assert p.scene_name == "TestScene"
    assert p.requires_confirmation is True
    assert p.state_evidence["scene_revision"] == "rev-1"
    assert p.state_evidence["target_cell"] == "r1c1"
    assert p.state_evidence["destination_occupant"] is None
    assert "It is on r1c1 now." in p.brief_reason


def test_a_non_live_plan_records_no_evidence():
    p = plan(make_scene(), "put soda_can on r2c2").proposal
    assert p.state_evidence == {}
    assert "has not been verified" in p.brief_reason


# ---------------------------------------------------------------------------
# Live semantics
# ---------------------------------------------------------------------------

def test_a_can_in_the_pool_is_not_a_tabletop_candidate():
    scene = make_scene()
    apply_snapshot(scene, snap(soda_can=IN_POOL))
    out = plan(scene, "put the recycle item on r2c2")
    assert out.kind == "clarification"
    assert "No recyclable object is on the board" in out.message


def test_a_placed_can_resolves_and_foam_block_does_not():
    scene = make_scene()
    apply_snapshot(scene, snap(soda_can=on_cell(scene, "r1c1"),
                               foam_block=on_cell(scene, "r1c2")))
    out = plan(scene, "put the recycle item on r2c2")
    assert out.kind == "proposal"
    assert out.proposal.target_id == "soda_can"


def test_two_tabletop_recyclables_are_disambiguated_by_cell():
    scene = make_scene()
    scene.objects["red_cube"].tags.append("recyclable")
    apply_snapshot(scene, snap(soda_can=on_cell(scene, "r1c1"),
                               red_cube=on_cell(scene, "r1c3")))
    out = plan(scene, "put the recycle item on r2c2")
    assert out.kind == "clarification"
    assert "soda_can (r1c1)" in out.choices
    assert "red_cube (r1c3)" in out.choices


def test_an_occupied_destination_is_refused_from_live_state():
    scene = make_scene()
    apply_snapshot(scene, snap(soda_can=on_cell(scene, "r1c1"),
                               red_cube=on_cell(scene, "r2c2")))
    out = plan(scene, "put soda_can on r2c2")
    assert out.kind == "clarification"
    assert "already occupied by red_cube" in out.message
    assert "r2c2" not in out.choices


def test_missing_bin_offers_only_free_reachable_cells():
    scene = make_scene()
    apply_snapshot(scene, snap(soda_can=on_cell(scene, "r1c1"),
                               red_cube=on_cell(scene, "r2c2")))
    out = plan(scene, "Put the recycle item in the bin.")
    assert out.kind == "clarification"
    assert "no bin or tray configured" in out.message
    assert "r1c1" not in out.choices and "r2c2" not in out.choices
    assert "r3c1" not in out.choices
    assert "r1c2" in out.choices


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------

def _live_scene_with(can_cell="r1c1"):
    scene = make_scene()
    apply_snapshot(scene, snap(soda_can=on_cell(scene, can_cell)))
    return scene


def _proposal_and_validator(scene_now, plan_scene=None):
    plan_scene = plan_scene or scene_now
    proposal = plan(plan_scene, "put soda_can on r2c2").proposal
    return proposal, LiveProposalValidator(lambda: scene_now)


def test_an_unchanged_world_still_validates():
    scene = _live_scene_with()
    proposal, validate = _proposal_and_validator(scene)
    assert validate(proposal) == (True, "")


def test_advancing_simulation_time_alone_does_not_invalidate():
    planned = _live_scene_with()
    proposal = plan(planned, "put soda_can on r2c2").proposal

    later = make_scene()
    apply_snapshot(later, SimSnapshot(
        sim_step=999_999,               # hundreds of thousands of steps later
        sim_time_s=2000.0,
        scene_revision="rev-1",
        objects={"soda_can": on_cell(later, "r1c1")},
        received_at=__import__("time").monotonic(),
    ))
    ok, why = LiveProposalValidator(lambda: later)(proposal)
    assert ok, why


def test_tiny_physics_jitter_does_not_invalidate():
    planned = _live_scene_with()
    proposal = plan(planned, "put soda_can on r2c2").proposal
    pos = list(proposal.state_evidence["target_pos"])
    pos[0] += POSE_TOLERANCE_M / 2

    later = make_scene()
    apply_snapshot(later, snap(soda_can=tuple(pos)))
    assert LiveProposalValidator(lambda: later)(proposal)[0]


def test_moving_the_target_invalidates():
    planned = _live_scene_with()
    proposal = plan(planned, "put soda_can on r2c2").proposal

    later = make_scene()
    apply_snapshot(later, snap(soda_can=on_cell(later, "r1c3")))
    ok, why = LiveProposalValidator(lambda: later)(proposal)
    assert not ok
    assert "has moved" in why or "no longer on" in why


def test_recalling_the_target_to_the_pool_invalidates():
    planned = _live_scene_with()
    proposal = plan(planned, "put soda_can on r2c2").proposal

    later = make_scene()
    apply_snapshot(later, snap(soda_can=IN_POOL))
    ok, why = LiveProposalValidator(lambda: later)(proposal)
    assert not ok


def test_something_else_taking_the_destination_invalidates():
    planned = _live_scene_with()
    proposal = plan(planned, "put soda_can on r2c2").proposal

    later = make_scene()
    apply_snapshot(later, snap(soda_can=on_cell(later, "r1c1"),
                               red_cube=on_cell(later, "r2c2")))
    ok, why = LiveProposalValidator(lambda: later)(proposal)
    assert not ok
    assert "now holds red_cube" in why


def test_a_scene_reload_invalidates():
    planned = _live_scene_with()
    proposal = plan(planned, "put soda_can on r2c2").proposal

    later = make_scene()
    apply_snapshot(later, SimSnapshot(
        scene_revision="rev-2",
        objects={"soda_can": on_cell(later, "r1c1")},
        received_at=__import__("time").monotonic(),
    ))
    ok, why = LiveProposalValidator(lambda: later)(proposal)
    assert not ok
    assert "reloaded" in why


def test_losing_the_link_invalidates_rather_than_assuming():
    planned = _live_scene_with()
    proposal = plan(planned, "put soda_can on r2c2").proposal
    ok, why = LiveProposalValidator(lambda: make_scene())(proposal)   # not live
    assert not ok
    assert "lost my link" in why


def test_a_scene_error_invalidates():
    planned = _live_scene_with()
    proposal = plan(planned, "put soda_can on r2c2").proposal
    ok, why = LiveProposalValidator(lambda: SceneView(error="bad YAML"))(proposal)
    assert not ok
    assert "bad YAML" in why


def test_a_plan_with_no_evidence_is_not_invalidated_by_a_live_scene():
    """A planning-only plan claimed no position, so nothing contradicts it."""
    proposal = plan(make_scene(), "put soda_can on r2c2").proposal
    assert proposal.state_evidence == {}
    assert LiveProposalValidator(lambda: _live_scene_with())(proposal)[0]


def test_the_target_vanishing_from_the_scene_invalidates():
    planned = _live_scene_with()
    proposal = plan(planned, "put soda_can on r2c2").proposal

    later = make_scene()
    del later.objects["soda_can"]
    apply_snapshot(later, snap(red_cube=on_cell(later, "r1c2")))
    ok, why = LiveProposalValidator(lambda: later)(proposal)
    assert not ok
    assert "no longer in this scene" in why


def test_a_target_whose_pose_went_nonfinite_invalidates():
    planned = _live_scene_with()
    proposal = plan(planned, "put soda_can on r2c2").proposal

    later = make_scene()
    # The link drops a non-finite pose, so the object simply has no position.
    apply_snapshot(later, snap(red_cube=on_cell(later, "r1c2")))
    ok, why = LiveProposalValidator(lambda: later)(proposal)
    assert not ok
    assert "no longer see where" in why


# ===========================================================================
# Against the real scene file
# ===========================================================================

def _real_pool_scene():
    from placement import cells_from_scene, pool_ids
    from scene_io import load_scene
    path = os.path.join(_HERE, "../../scenes/FWDCenterLabSivaPool.yaml")
    doc = load_scene(path)
    return scene_view_from_doc(doc, cells_from_scene(doc), placeable=pool_ids(doc))


def test_real_pool_scene_parks_everything_off_the_grid():
    """The pool scene's own promise: the board starts empty."""
    scene = _real_pool_scene()
    poses = {}
    for oid, obj in scene.objects.items():
        # Read the compiled-in pose straight out of the document.
        poses[oid] = None
    from scene_io import load_scene
    doc = load_scene(os.path.join(_HERE, "../../scenes/FWDCenterLabSivaPool.yaml"))
    for entry in doc["objects"]:
        if entry["id"] in scene.objects:
            poses[entry["id"]] = tuple(entry["pose"]["position"])

    apply_snapshot(scene, SimSnapshot(
        scene_revision="initial",
        objects={k: v for k, v in poses.items() if v},
        received_at=__import__("time").monotonic(),
    ))
    assert scene.live is True
    assert scene.tabletop_recyclables() == []
    assert all(c.occupant is None for c in scene.cells.values())


def test_real_pool_scene_a_placed_soda_can_becomes_the_candidate():
    scene = _real_pool_scene()
    cell = scene.cells["r2c2"]
    apply_snapshot(scene, SimSnapshot(
        scene_revision="initial",
        objects={"soda_can": (cell.x, cell.y, cell.top_z + 0.06),
                 "foam_block": (cell.x, cell.y - 0.15, cell.top_z + 0.03)},
        received_at=__import__("time").monotonic(),
    ))
    assert scene.objects["soda_can"].cell == "r2c2"
    assert [o.object_id for o in scene.tabletop_recyclables()] == ["soda_can"]
    # foam_block is on the board too, and is still not recyclable.
    assert scene.objects["foam_block"].on_board is True
    assert not scene.objects["foam_block"].is_recyclable


def test_real_pool_scene_still_has_no_bin_to_offer():
    scene = _real_pool_scene()
    cell = scene.cells["r2c2"]
    apply_snapshot(scene, SimSnapshot(
        scene_revision="initial",
        objects={"soda_can": (cell.x, cell.y, cell.top_z + 0.06)},
        received_at=__import__("time").monotonic(),
    ))
    out = plan(scene, "Put the recycle item in the bin.")
    assert out.kind == "clarification"
    assert "no bin or tray configured" in out.message
    assert "r2c2" not in out.choices        # the can is standing on it
