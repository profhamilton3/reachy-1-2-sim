"""#2: motion driven against the scene, not beside it.

An integration check the scene technology had unit coverage for but no
motion-level test of:

(a) PLANNING REFUSES A PATH THROUGH THE TABLE.  `CartesianPlanner.plan_segment`
    against the panel scene (FWDCenterLabSivaPool) raises `CollisionError` for
    a segment whose pad dips under the tabletop -- before any IK is solved, so
    nothing could be commanded from it.

(b), a physics carry from the recorded 2026-10-06 crane hold, went with the
    August mount (#192): the recorded joints no longer put the hand round the
    cube.  The grasp itself is covered by tests/unit/test_gripper.py.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (_ROOT / "native_mujoco", _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reachy_ai.motion.kinematics import (  # noqa: E402
    CartesianPlanner, CollisionError, IKPolicy,
)
from reachy_ai.scene.awareness import SceneModel  # noqa: E402

_SCENE = _ROOT / "scenes" / "FWDCenterLabSivaPool.yaml"
_MODEL = _ROOT / "native_mujoco" / "model" / "reachy_1_2.xml"


# ── (a) planning a below-table path ──────────────────────────────────────────

@pytest.fixture(scope="module")
def planner():
    return CartesianPlanner(arm=None, scene=SceneModel.from_yaml(str(_SCENE)))


def _never_solve(*_a, **_k):
    raise AssertionError("IK was solved for a path the pad check should refuse")


def test_a_segment_through_the_tabletop_is_refused_before_any_ik(planner, monkeypatch):
    surface = planner.scene.table_surface_z
    above = (0.42, -0.10, surface + 0.10)
    below = (0.42, -0.10, surface - 0.06)          # inside the table's footprint
    monkeypatch.setattr(planner, "solve", _never_solve)
    with pytest.raises(CollisionError) as info:
        planner.plan_segment(above, below, 8, [0.0] * 7, ik=IKPolicy.FAST,
                             segment="descend through the board")
    text = str(info.value)
    assert "below_table" in text and "table_top" in text


def test_the_same_descent_stopping_above_the_surface_passes_the_pad_check(planner):
    surface = planner.scene.table_surface_z
    cart = planner.interpolate((0.42, -0.10, surface + 0.10),
                               (0.42, -0.10, surface + 0.03), 8)
    planner.check_collisions(cart)                 # raises on a violation
