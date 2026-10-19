"""Record pad points for fixed joint sets, for #187's pad-invariance test.

Run against a checkout of the commit BEFORE #187 (the SDK FK frame was then the
wrist), which is what makes the values a "before":

    python3 make_pad_invariance_187.py <checkout> <out.json> <commit>

The joint sets are the crane's r3c2 -> r3c3 placement plan (#195/#197: pickup
and destination ladders, made with the host IK mirror), PRESENT, the zero pose,
and the real robot's B2_SAFE_JOINTS (docs/reviews/2026-10-14-taught-board-poses-185/
taught_poses.json) with its wrist roll swept.
Each pad is the planner's own `CartesianPlanner.fk_world` (through the SDK-shaped
host mirror) and `kinematics.pad_point` (the local chain); they agree.
"""

import json
import os
import sys

root, out_path, commit = sys.argv[1:4]
for p in ("src", "native_mujoco", "tests/unit"):
    sys.path.insert(0, os.path.join(root, p))

from host_ik_arm import HostIKArm  # noqa: E402
from reachy_ai.motion import rig_routes as RR  # noqa: E402
from reachy_ai.motion.kinematics import R_ARM_JOINTS, CartesianPlanner, pad_point  # noqa: E402
from reachy_ai.scene.awareness import SceneModel  # noqa: E402
from reachy_ai.tasks import crane_pick_live as CP  # noqa: E402
from reachy_ai.tasks import crane_place_live as PL  # noqa: E402

scene = SceneModel.from_yaml(os.path.join(root, "scenes/FWDCenterLabSivaPool.yaml"))
x, y, z = scene.cell_center("cell_r3c2")
pos = (x, y, z + 0.03)
scene.update_poses({"red_cube": pos})
planner = CartesianPlanner(HostIKArm(), scene=scene)
pp = PL.plan_crane_place(planner, scene, "red_cube", "r3c3",
                         CP.Observation(pos, (1.0, 0.0, 0.0, 0.0), 0.0, False))

taught = json.load(open(os.path.join(
    root, "docs/reviews/2026-10-14-taught-board-poses-185/taught_poses.json")))
sets = {"present": [float(RR.PRESENT[j]) for j in R_ARM_JOINTS],
        "b2_safe": taught["fk_pair"]["joints_deg"]}
# The tip <-> pad offset turns with the wrist roll: sweep it at B2_SAFE.
for roll in (-45.0, -20.0, 0.0, 20.0, 45.0):
    sets[f"b2_safe_roll_{roll:+.0f}"] = sets["b2_safe"][:6] + [roll]
sets["zero"] = [0.0] * 7
for k, q in enumerate(pp.pick.rungs):
    sets[f"crane_pick_rung_{k:02d}"] = [float(v) for v in q]
for k, q in enumerate(pp.dest.rungs):
    sets[f"crane_dest_rung_{k:02d}"] = [float(v) for v in q]

poses = {}
for name, q in sets.items():
    a = planner.fk_world(q)
    b = [float(v) for v in pad_point(q)]
    assert max(abs(u - v) for u, v in zip(a, b)) < 1e-9, name
    poses[name] = {"joints_deg": [float(v) for v in q], "pad_world_m": list(a)}

json.dump({"provenance": {
    "issue": "profhamilton3/reachy-1-2-sim#187",
    "commit": commit,
    "generator": "tests/fixtures/kinematics/make_pad_invariance_187.py",
    "scene": "scenes/FWDCenterLabSivaPool.yaml, red_cube upright on cell_r3c2",
    "plan": "crane_place_live.plan_crane_place(red_cube, r3c3), host IK mirror",
    "frame": "pad_world_m: world metres, CartesianPlanner.fk_world"},
    "poses": poses}, open(out_path, "w"), indent=1)
print(len(poses), "joint sets")
