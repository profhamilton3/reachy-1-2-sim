#!/usr/bin/env python3
"""Which shape gate, if any, refuses each PART design in each grasp planner.

    python3 gates.py [--design 110 ...]

No physics and no motion.  Each part is posed upright on cell r2c2.  The planners get an arm whose first IK request
raises, so whatever is refused before it was refused by the shape checks (kind,
class, roll model, jaw opening) and not by reach or clearance.
"""
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "native_mujoco")]

from reachy_ai.motion.kinematics import CartesianPlanner     # noqa: E402
from reachy_ai.scene.awareness import SceneModel               # noqa: E402
from reachy_ai.tasks import crane_pick_live as CP              # noqa: E402
from reachy_ai.tasks import side_grasp_live as SG              # noqa: E402

SCENE = str(REPO / "scenes" / "FWDCenterLabSivaPool.yaml")
DESIGNS = ["110", "111", "120", "121", "210", "211", "220", "221"]


class PastTheGates(BaseException):
    """BaseException, so the planners' own `except Exception` around IK cannot swallow it."""


class NoIKArm:
    def inverse_kinematics(self, *_a, **_k):
        raise PastTheGates

    def forward_kinematics(self, *_a, **_k):
        raise PastTheGates


def on_cell(scene, oid, cell="cell_r2c2"):
    x, y, _ = scene.cell_center(cell)
    pos = (x, y, scene.table_surface_z + scene.get(oid).size[2] / 2.0)
    return CP.Observation(pos, (1.0, 0.0, 0.0, 0.0), 0.0, False, 0.0)


def verdict(plan, scene, oid):
    try:
        out = plan(CartesianPlanner(NoIKArm(), scene=scene), scene, oid, on_cell(scene, oid))
    except PastTheGates:
        return "passes the shape gates (reaches IK)"
    except CP.CraneRefused as exc:
        return f"REFUSED [{exc.stage}] {exc.reason}"
    # side_candidates can finish without IK when every approach is refused first
    reasons = sorted({c.refusal for c in out if c.refusal})
    return f"REFUSED before IK, all {len(out)} approaches: {reasons[0] if reasons else '?'}"


def main():
    scene = SceneModel.from_yaml(SCENE)
    for design in DESIGNS:
        oid = f"part{design}_a"
        o = scene.get(oid)
        print(f"{oid}: kind={o.kind} class={o.semantic_class} size_mm="
              f"{[round(1000 * v, 2) for v in o.size]}")
        print("   crane  :", verdict(CP.plan_crane_pick, scene, oid))
        print("   side-on:", verdict(SG.side_candidates, scene, oid))


if __name__ == "__main__":
    main()
