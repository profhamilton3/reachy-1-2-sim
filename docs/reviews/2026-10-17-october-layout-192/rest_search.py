"""The REST search: drop the arm motors-off from random starts on the right margin,
keep fixed points, score against tall objects on column 3 (rest and the descent
from PRESENT).  Args: <repo> <samples> <seed>.

With the hand collision boxes the hand rests on its palm, not on the finger
pad, and a dropped gripper can fall open; a fixed point must also keep the
gripper shut (>= 15 deg), since the footprint guard sizes the hand by its
aperture.  REST came from seeds 41-44, 1500 samples each (the pre-box REST
came from seeds 31 and 32, 3000 samples each)."""
import sys, pathlib
import numpy as np
S = pathlib.Path(__file__).parent; sys.path.insert(0, str(S))
# usage: PYTHONPATH=src python3 docs/reviews/2026-10-17-october-layout-192/rest_search.py . [args]
from settle import World
REPO = pathlib.Path(sys.argv[1]); w = World(REPO)
from reachy_ai.motion import kinematics as K, rig_routes as R
from reachy_ai.scene.awareness import SceneModel, segment_object_distance
SCENE = str(REPO / "scenes/FWDCenterLabSivaPool.yaml")
probes = []
for cell in ("cell_r1c3", "cell_r2c3", "cell_r3c3"):
    for oid in ("soda_can", "blue_cylinder", "red_cube", "pool_box_3"):
        mm = SceneModel.from_yaml(SCENE); x, y, z = mm.cell_center(cell)
        mm.update_poses({oid: (x, y, z + mm.get(oid).size[2] / 2)}); probes.append((cell, oid, mm.get(oid)))
PRES = [R.PRESENT[j] for j in R.ARM7]
def score(q):
    worst = (9, "", "")
    for qq in K.joint_path(PRES, q, 61):
        caps = K.link_capsules(qq, "right", 20.0)
        for c, oid, o in probes:
            for _, p0, p1, r in caps:
                d = segment_object_distance(o, p0, p1, r)[0]
                if d < worst[0]: worst = (d, c, oid)
    return worst
rng = np.random.default_rng(int(sys.argv[3]))
for i in range(int(sys.argv[2])):
    q0 = [rng.uniform(15, 45), rng.uniform(-10, 15), rng.uniform(-75, -15), rng.uniform(-120, -70),
          rng.uniform(-60, 60), rng.uniform(-40, 40), rng.uniform(-45, 10)]
    pad = K.pad_point(q0)
    if not (0.28 < pad[0] < 0.62 and -0.345 < pad[1] < -0.25 and 0.74 < pad[2] < 0.80): continue
    q1, _, _, _ = w.settle(q0, grip=20.0, seconds=5.0)
    q2, cons, tip, sp = w.settle(q1[:7], grip=20.0, seconds=5.0)
    drift = max(abs(a - b) for a, b in zip(q1[:4], q2[:4]))
    if drift > 2.0 or sp > 0.01 or tip[2] < 0.735 or q2[7] < 15.0: continue
    rest_only = min((segment_object_distance(o, p0, p1, r)[0], c[5:], oid, n) for c, oid, o in probes for n, p0, p1, r in K.link_capsules(q2[:7], "right", 20.0))
    sc = score(q2[:7])
    print(f"q {np.round(q2[:8],1).tolist()} tip {np.round(tip,3).tolist()} drift {drift:.1f} speed {sp:.4f} rest {rest_only[0]*100:+.1f} {rest_only[1:]} path {sc[0]*100:+.1f}cm {sc[1]} {sc[2]}", flush=True)
