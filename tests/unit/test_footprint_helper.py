"""`motion.footprint.route_footprint_clearances` -- the guard's pure geometry,
extracted from `panel_executor._footprint_refusal` for #56.

The panel's own refusal tests (`test_panel_executor.TestFootprintCheck`,
`TestLiftToPresentFootprint`) are the behavioural pin and are unmodified; these
cover the helper's own contract.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.motion.footprint import route_footprint_clearances  # noqa: E402
from reachy_ai.motion.kinematics import link_capsules  # noqa: E402
from reachy_ai.scene.awareness import SceneModel  # noqa: E402

_SCENE = os.path.join(_HERE, "..", "..", "scenes", "FWDCenterLabMCC.yaml")


def _model():
    return SceneModel.from_yaml(_SCENE)


def test_a_route_with_no_footprint_entry_contributes_nothing():
    assert route_footprint_clearances(_model(), ["WAVE", "NOT_A_ROUTE"]) == {}
    assert route_footprint_clearances(_model(), []) == {}


def test_it_reports_manipulables_with_the_link_that_is_closest():
    model = _model()
    out = route_footprint_clearances(model, ["RAISE_TO_SIDE"])
    assert set(out) == set(model.manipulable_ids())
    for oid, c in out.items():
        assert c.object_id == oid
        assert c.link in ("upper_arm", "forearm", "hand")


def test_an_object_under_the_rest_pose_reads_as_inside_the_footprint():
    model = _model()
    oid = model.manipulable_ids()[0]
    caps = link_capsules([R.REST[j] for j in R.ARM7], "right", R.REST["r_gripper"])
    forearm = next(c for c in caps if c[0] == "forearm")
    mid = tuple(0.5 * (a + b) for a, b in zip(forearm[1], forearm[2]))
    model.update_poses({oid: mid})
    c = route_footprint_clearances(model, ["RAISE_TO_SIDE"])[oid]
    assert c.distance < R.FOOTPRINT_MARGIN
    assert c.link == "forearm"


def test_the_stow_legs_are_the_ones_the_table_lists():
    # STOW_FROM_SIDE is PRESENT -> REST_SHUT -> HOVER, and nothing else.
    assert R.FOOTPRINT_LEGS["STOW_FROM_SIDE"] == (R.PRESENT, R.REST_SHUT, R.HOVER)
    assert R.FOOTPRINT_LEGS["RAISE_TO_SIDE"] == (R.HOVER, R.REST_SHUT, R.REST)
