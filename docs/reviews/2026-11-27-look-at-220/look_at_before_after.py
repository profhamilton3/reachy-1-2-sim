"""#220: every look_at caller's aim, before and after, and a before/after render of the left eye.

usage: python3 look_at_before_after.py REPO OUT_DIR   (needs mujoco, numpy, PyYAML, Pillow; no SDK, no server)

"after" runs primitives.look_at on a stub robot. "before" is the formula it had at d889a2a, restated here. Targets are
built with the same scene calls as each call site. Pixels are the target through the compiled model's own cameras
(gaze.CAMERAS focal lengths), midway between the two eyes, as offsets from the centre of a 640x480 frame.
"""
import math
import pathlib
import sys

REPO, OUT = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
sys.path[:0] = [str(REPO / "src"), str(REPO / "native_mujoco")]

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from reachy_ai.motion import gaze, primitives as P  # noqa: E402
from reachy_ai.scene.awareness import SceneModel  # noqa: E402
from simulation_core import SimulationCore  # noqa: E402

MODEL = str(REPO / "native_mujoco/model/reachy_1_2.xml")
model = mujoco.MjModel.from_xml_path(MODEL)


def scene(name):
    return SceneModel.from_yaml(str(REPO / "scenes" / (name + ".yaml")))


def before(t):
    dx, dy, dz = (t[i] - gaze.NECK_PIVOT_IN_WORLD[i] for i in range(3))
    clamp = lambda v, r: min(max(v, r[0]), r[1])  # noqa: E731
    return (clamp(math.degrees(math.atan2(-dz, math.hypot(dx, dy))), gaze.PITCH_LIMITS),
            clamp(math.degrees(math.atan2(dy, dx)), gaze.YAW_LIMITS))


class Joint:
    goal_position = present_position = 0.0
    compliant = True


class Robot:
    def __init__(self):
        self.head = type("Head", (), {})()
        for n in ("neck_roll", "neck_pitch", "neck_yaw"):
            setattr(self.head, n, Joint())

    def turn_on(self, part):
        pass


def after(t):
    robot, sleep = Robot(), P.time.sleep
    P.time.sleep = lambda s: None
    try:
        P.look_at(robot, t, duration=0.1)
    finally:
        P.time.sleep = sleep
    return robot.head.neck_pitch.goal_position, robot.head.neck_yaw.goal_position


def posed(m, d, angles):
    for name, deg in zip(("neck_pitch", "neck_yaw"), angles):
        d.qpos[m.joint(name).qposadr[0]] = math.radians(deg)
    mujoco.mj_forward(m, d)


def aim(t, angles):
    d = mujoco.MjData(model)
    posed(model, d, angles)
    px = []
    for name, _, fx, fy in gaze.CAMERAS:
        c = model.camera(name + "_camera").id
        v = d.cam_xmat[c].reshape(3, 3).T @ (np.asarray(t) - d.cam_xpos[c])
        px.append((320 + fx * v[0] / -v[2], 240 - fy * v[1] / -v[2]) if v[2] < 0 else (math.nan, math.nan))
    u, v = np.mean(px, axis=0) - (320, 240)
    return u, v, all(0 <= a < 640 and 0 <= b < 480 for a, b in px)


def callers():
    cp = scene("control_panel")
    ctrls = cp.controls()
    yield ("demo_control_panel", "console (initial)",
           (sum(c.center[0] for c in ctrls) / len(ctrls), 0.0, max(c.center[2] for c in ctrls) - 0.03))
    for c in ctrls:
        yield "demo_control_panel", c.id, cp.control_target(c.id).point
    td = scene("tabletop_demo")
    ids = td.manipulable_ids()
    yield ("demo_pick_place", "workspace centroid",
           (sum(td.get(i).center[0] for i in ids) / len(ids), sum(td.get(i).center[1] for i in ids) / len(ids),
            td.table_surface_z))
    for i in ids:
        yield "pick_place_live grasp", i, td.grasp_point(i)
    lab = scene("FWDCenterLabSivaPool")
    for cell in sorted(lab.grid_cells()):
        x, y, _ = lab.cell_center(cell)
        yield "pick_place_live place", cell, (x, y, lab.table_surface_z)
        yield "crane_pick_live obj.center", cell + " (50.8 mm cube)", (x, y, lab.table_surface_z + 0.0254)
    yield "v2 notebook fixed point", "(0.42, 0.15, 0.77)", (0.42, 0.15, 0.77)
    yield "v2 notebook fixed point", "(0.50, 0, 0.90)", (0.50, 0.0, 0.90)


def cell_text(t, angles):
    u, v, both = aim(t, angles)
    return "(%5.1f, %+6.1f)  off (%+4.0f, %+4.0f)  %s" % (angles[0], angles[1], u, v, "in " if both else "OUT"), both


rows = ["%-27s %-27s %-44s %s" % ("caller", "target", "before: (pitch, yaw) off(u,v) px, both eyes", "after")]
totals = [0, 0, 0]
for caller, label, t in callers():
    (a, in_a), (b, in_b) = cell_text(t, before(t)), cell_text(t, after(t))
    rows.append("%-27s %-27s %-44s %s" % (caller, label, a, b))
    totals = [totals[0] + 1, totals[1] + in_a, totals[2] + in_b]
rows.append("targets %d; in both frames: before %d, after %d" % tuple(totals))
(OUT / "callers.txt").write_text("\n".join(rows) + "\n")
print("\n".join(rows))

# The left eye, with the red cube standing on the cell that look_at is given. Arm at REST, measured fovy, pinhole.
core = SimulationCore.from_paths(MODEL, str(REPO / "scenes/FWDCenterLabSivaPool.yaml"))
m, d = core.model, core.data
lab = scene("FWDCenterLabSivaPool")
m.cam_fovy[m.camera("left_camera").id] = math.degrees(2 * math.atan(240 / gaze.CAMERAS[0][3]))
renderer = mujoco.Renderer(m, 480, 640)
adr = m.jnt_qposadr[m.body_jntadr[m.body("red_cube").id]]
cells = ("cell_r1c1", "cell_r2c1", "cell_r3c3")
sheet = Image.new("RGB", (640, 24 + 240 * len(cells)), "white")
ImageDraw.Draw(sheet).text((8, 6), "left eye, cross = frame centre. before (d889a2a)          after (#220)", fill="black")
for row, cell in enumerate(cells):
    x, y, z = lab.cell_center(cell)
    target = (x, y, z + 0.03)
    for col, angles in enumerate((before(target), after(target))):
        core.reset()
        d.qpos[adr:adr + 3] = target
        posed(m, d, angles)
        renderer.update_scene(d, camera="left_camera")
        tile = Image.fromarray(renderer.render()).resize((320, 240))
        draw = ImageDraw.Draw(tile)
        draw.line([(150, 120), (170, 120)], fill="white", width=1)
        draw.line([(160, 110), (160, 130)], fill="white", width=1)
        draw.text((6, 4), "%s  (%.1f, %+.1f)" % (cell, *angles), fill="white")
        sheet.paste(tile, (320 * col, 24 + 240 * row))
    ImageDraw.Draw(sheet).line([(320, 24), (320, sheet.height)], fill="white", width=2)
sheet.save(OUT / "left_eye_before_after.png", optimize=True)
