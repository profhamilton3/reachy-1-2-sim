"""Render both sim cameras at gaze.BOARD_VIEW, arm at REST, measured fovy, pinhole."""
import sys, math, pathlib
REPO = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2])
sys.path[:0] = [str(REPO), str(REPO / "native_mujoco"), str(REPO / "src")]
import numpy as np, mujoco
from PIL import Image
from simulation_core import SimulationCore
from reachy_ai.motion import gaze
core = SimulationCore.from_paths(str(REPO / "native_mujoco/model/reachy_1_2.xml"), str(REPO / "scenes/FWDCenterLabSivaPool.yaml"))
m, d = core.model, core.data
core.reset()
for n, v in gaze.BOARD_VIEW.items():
    d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = math.radians(v)
mujoco.mj_forward(m, d)
for cam, fy in (("left_camera", 407.0), ("right_camera", 398.8)):
    m.cam_fovy[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)] = math.degrees(2 * math.atan(240 / fy))
r = mujoco.Renderer(m, 480, 640)
for cam in ("left_camera", "right_camera"):
    r.update_scene(d, camera=cam)
    Image.fromarray(r.render()).save(OUT / f"board_view_{cam.split('_')[0]}.png")
print("rendered", gaze.BOARD_VIEW)
