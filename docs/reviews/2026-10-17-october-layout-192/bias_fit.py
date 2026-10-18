"""Repeat gaze.PITCH_BIAS_DEG's fit: aim at red_cube with extra in {0..20}, row% of the cube in the left image."""
import sys, math, pathlib
REPO = pathlib.Path(sys.argv[1])
sys.path[:0] = [str(REPO), str(REPO / "native_mujoco"), str(REPO / "src")]
import numpy as np, mujoco
from simulation_core import SimulationCore
from reachy_ai.motion import gaze
from reachy_ai.scene.awareness import SceneModel
core = SimulationCore.from_paths(str(REPO / "native_mujoco/model/reachy_1_2.xml"), str(REPO / "scenes/FWDCenterLabSivaPool.yaml"))
m, d = core.model, core.data
sm = SceneModel.from_yaml(str(REPO / "scenes/FWDCenterLabSivaPool.yaml"))
cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "left_camera")
m.cam_fovy[cam] = math.degrees(2 * math.atan(240 / 407.0))
r = mujoco.Renderer(m, 480, 640)
b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "red_cube"); adr = m.jnt_qposadr[m.body_jntadr[b]]
cube_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
for cell in sys.argv[2:]:
    x, y, z = sm.cell_center(cell); c = (x, y, z + 0.03)
    rows = []
    for extra in (0.0, 5.0, 10.0, 15.0, 20.0):
        core.reset(); d.qpos[adr:adr+3] = c
        dx, dy, dz = (c[i] - gaze.NECK_PIVOT_IN_WORLD[i] for i in range(3))
        pitch = math.degrees(math.atan2(-dz, math.hypot(dx, dy))) - gaze.BUILT_IN_PITCH_DEG + extra
        yaw = math.degrees(math.atan2(dy, dx))
        for n, v in (("neck_pitch", pitch), ("neck_yaw", yaw)):
            d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = math.radians(v)
        mujoco.mj_forward(m, d); r.update_scene(d, camera="left_camera"); r.enable_segmentation_rendering(); seg = r.render(); r.disable_segmentation_rendering()
        mask = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], cube_geoms)
        row = 100 * np.nonzero(mask)[0].mean() / 480 if mask.any() else float("nan")
        rows.append((extra, pitch, row))
    ex = np.array([e for e, _, rr in rows if not math.isnan(rr)]); rr = np.array([rr for _, _, rr in rows if not math.isnan(rr)])
    k, c0 = np.polyfit(ex, rr, 1)
    print(cell, " ".join(f"{e:+.0f}:{p:.1f}deg->{q:.1f}%" for e, p, q in rows), f"| fit row% = {k:.2f}*extra + {c0:.1f} -> centred at {(50 - c0) / k:+.1f}")
