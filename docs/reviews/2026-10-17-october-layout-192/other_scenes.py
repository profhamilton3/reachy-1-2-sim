"""The start keyframe (REST) dropped motors-off for 5 s in every scene (#192).

The mount and the start pose are global; the board REST rests on exists only
in the FWDCenterLab scenes.  For each scene: the gross-joint drift from REST,
whether `posture_of` still calls it REST, and where the tip ends up.

    PYTHONPATH=src python3 docs/reviews/2026-10-17-october-layout-192/other_scenes.py .
"""
import math
import pathlib
import sys

REPO = pathlib.Path(sys.argv[1])
sys.path[:0] = [str(REPO / "native_mujoco"), str(REPO / "src")]
import mujoco  # noqa: E402
from simulation_core import SimulationCore  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402

for scene in sorted((REPO / "scenes").glob("*.yaml")):
    try:
        core = SimulationCore.from_paths(str(REPO / "native_mujoco/model/reachy_1_2.xml"), str(scene))
    except Exception as exc:  # a scene this backend does not load
        print(f"| {scene.name} | not loadable: {type(exc).__name__} |")
        continue
    m, d = core.model, core.data
    core.reset()
    for _ in range(int(5.0 / m.opt.timestep)):
        core.advance()
    now = {n: math.degrees(d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]])
           for n in R.R_JOINTS}
    drift = max(abs(now[n] - R.REST[n]) for n in R.GROSS_JOINTS)
    tip = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "r_tip")]
    print(f"| {scene.name} | {drift:.1f} deg | {R.posture_of(now)} | "
          f"({tip[0]:.2f}, {tip[1]:.2f}, {tip[2]:.2f}) |")
