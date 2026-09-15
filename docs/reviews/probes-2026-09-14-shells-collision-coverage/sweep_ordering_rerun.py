"""Re-run of probes-2026-09-14/sweep_ordering.py's 224-position PLACE_ROUTE-
tail sweep (test_footprint_boards.py::TestSweepOrdering's own method,
_run_sweep, called directly so this prints exactly what the test asserts),
now that "shells" covers the collision pads (Slice 2).  Run from the repo
root:

    PYTHONPATH=src:native_mujoco python3 \
        docs/reviews/probes-2026-09-14-shells-collision-coverage/sweep_ordering_rerun.py

Offline: mj_forward only, no server, no SDK, no motion.
"""
import sys
sys.path.insert(0, "tests/unit"); sys.path.insert(0, "src"); sys.path.insert(0, "native_mujoco")
import mujoco
import test_footprint_boards as T
from objects import build_scene_model_xml
from scene_io import load_scene
from placement import ObjectPlacer

doc = load_scene(T._SCENE_PATH)
m = mujoco.MjModel.from_xml_string(build_scene_model_xml(doc))
d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)
for jid in range(m.njnt):
    if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
        a = m.jnt_qposadr[jid]; d.qpos[a:a + 7] = m.qpos0[a:a + 7]
mujoco.mj_forward(m, d)
placer = ObjectPlacer(m, d, doc)

n, ts_gaps, sr_gaps, tr_gaps = T._run_sweep((m, d, placer))
tol = 0.002
print(f"positions: {n}")
print(f"tube>reference (>2mm):  {sum(1 for g in tr_gaps if g > tol)}  max {max(tr_gaps)*100:.4f} cm")
print(f"tube>shells (>2mm):     {sum(1 for g in ts_gaps if g > tol)}  max {max(ts_gaps)*100:.4f} cm")
print(f"shells>reference (>2mm): {sum(1 for g in sr_gaps if g > tol)}  max {max(sr_gaps)*100:.6f} cm  "
      "(before Slice 2: 62 positions, max 1.0 cm)")
print(f"shells tighter than tube by >1cm: {sum(1 for g in ts_gaps if g < -0.01)}  (unchanged by Slice 2)")
print(f"tube over-conservatism vs reference, max: {max(-g for g in tr_gaps)*100:.4f} cm")
