"""#180: the side-on grasp flown offline, end to end.

`tools.place_sweep.fly` flies `execute_crane_pick(..., place=...)` against the
#172 offline world (see test_place_offline.py).  Here the grasp is the
side-on one (`reachy_ai.tasks.side_grasp_live`):

  * red_cube r2c2 -> r1c1 with grasp "best": the crane is refused (no grasp
    pose), so the planner choice flies side-on, and the placement meets D6;
  * a pool cylinder lifted (picked, held, put back) at r2c2 with the side-on
    grasp meets every lift criterion.

Needs MuJoCo and scipy; each flight takes ~1-3 min.
"""

import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytest.importorskip("mujoco")
pytest.importorskip("scipy")

import tools.place_sweep  # noqa: E402,F401  (paths)
import tools.withdrawal_replay  # noqa: E402,F401
from tools.place_sweep.fly import fly  # noqa: E402


def test_red_cube_r2c2_to_r1c1_is_placed_side_on():
    run = fly(oid="red_cube", origin="r2c2", dest="r1c1", grasp="best")
    assert "refused" not in run, run.get("refused")
    assert run["grasp"] == "side-on"
    assert run["criteria"] and all(run["criteria"].values()), run["criteria"]
    assert run["error_mm"] <= 10.0
    assert run["tilt_deg"] <= 5.0
    assert run["others_moved_max_mm"] < 0.1
    assert run["final"] == "PRESENT"


def test_pool_cylinder_is_lifted_side_on_at_r2c2():
    run = fly(oid="pool_cyl_1", origin="r2c2", dest=None, grasp="side")
    assert "refused" not in run, run.get("refused")
    assert run["grasp"] == "side-on"
    assert run["criteria"] and all(run["criteria"].values()), run["criteria"]
