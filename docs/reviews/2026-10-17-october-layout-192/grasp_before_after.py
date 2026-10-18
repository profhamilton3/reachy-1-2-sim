"""The hand collision boxes leave the pinch alone: pads-only contact, same grip force.

The R12-502 grasp fixture (tests/unit/test_gripper.py: arm straight down, a cube
on a pillar in the open gap, close, lift, release), run with and without the
two hand boxes, for a 20 mm cube (the fixture's) and a 40 mm cube (the scenes'
red_cube and pool boxes, gripper opened to -45 deg).  Reports every robot geom
that touched the cube and the grip force at close and at lift.

usage: PYTHONPATH=src:native_mujoco python3 docs/reviews/2026-10-17-october-layout-192/grasp_before_after.py .
"""
import math
import pathlib
import re
import sys

import mujoco

REPO = pathlib.Path(sys.argv[1])
sys.path[:0] = [str(REPO / "native_mujoco"), str(REPO / "src")]
from actuator import ActuatorController  # noqa: E402
from gripper import GripperModel  # noqa: E402
from reachy_ai.scene.awareness import ARM_BASE_IN_WORLD  # noqa: E402

MODEL = REPO / "native_mujoco/model/reachy_1_2.xml"
HAND_BOXES = ("r_hand_back_col", "r_palm_col")


def scene(half, open_rad, with_boxes, friction="1.0 0.1 0.01"):
    xml = MODEL.read_text()
    if not with_boxes:
        for name in HAND_BOXES:
            xml, n = re.subn(r'<geom name="%s".*?/>\s*' % name, "", xml, flags=re.DOTALL)
            assert n == 1, name
    xml = re.sub(r'<body name="table".*?</body>\s*', "", xml, flags=re.DOTALL)
    px, py, pz = ARM_BASE_IN_WORLD[0], -0.20, ARM_BASE_IN_WORLD[2] - 0.8153
    extra = f"""
    <body name="support" pos="{px} {py} {pz}">
      <geom name="support" type="box" size="0.06 0.06 0.15"
            contype="8" conaffinity="4"/>
    </body>
    <body name="cube" pos="0 0 0">
      <freejoint name="cube_free"/>
      <geom name="cube" type="box" size="{half} {half} {half}"
            contype="4" conaffinity="15" mass="0.02" friction="{friction}"/>
    </body>
  </worldbody>"""
    m = mujoco.MjModel.from_xml_string(xml.replace("  </worldbody>", extra, 1),
                                       {})
    return m, mujoco.MjData(m)


def roll(m, d, ctl, n, touched):
    cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube")
    for _ in range(n):
        ctl.apply(d, m.opt.timestep)
        mujoco.mj_step(m, d)
        for c in d.contact[:d.ncon]:
            if cube in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == cube else c.geom1
                if m.geom_contype[other] == 2:
                    touched.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other))


def run(half, open_rad, with_boxes):
    m, d = scene(half, open_rad, with_boxes)
    ctl, grip = ActuatorController(m), GripperModel(m)
    ctl.sync_targets_to_current(d)
    fg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "r_finger_col")
    tg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "r_thumb_col")
    sg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "support")
    cadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
    touched = set()
    for i in range(7):
        ctl.set_compliant(i, False)
        ctl.set_goal_position(i, 0.0)
    ctl.set_compliant("act_r_gripper", False)
    # Ramp the gripper (the SDK's goto does): a step from 0 to -45 deg with no
    # ramp saturates the 8 N limit on the light finger and chatters in place.
    ctl.set_speed_limit("act_r_gripper", 1.5)
    ctl.set_goal_position("act_r_gripper", open_rad)
    roll(m, d, ctl, 500, set())
    fp, tp = d.geom_xpos[fg], d.geom_xpos[tg]
    cube_y = ((fp[1] + 0.008) + (tp[1] - 0.008)) / 2
    # Centre the cube on the thumb pad's height (the r_tip height, where the
    # planners aim), whatever its size: drop the pillar to meet it.
    cube_z = float(tp[2])
    sb = m.geom_bodyid[sg]
    m.body_pos[sb][2] = cube_z - half - 0.15
    d.qpos[cadr:cadr + 7] = [tp[0], cube_y, cube_z, 1, 0, 0, 0]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    placed = int(d.ncon)
    z0 = float(d.qpos[cadr + 2])
    ctl.set_goal_position("act_r_gripper", 0.35)
    roll(m, d, ctl, 700, touched)
    close = grip.update(d)["right"]
    ctl.set_goal_position(0, -0.7)
    ctl.set_speed_limit(0, 0.6)
    roll(m, d, ctl, 2200, touched)
    lift = grip.update(d)["right"]
    z1 = float(d.qpos[cadr + 2])
    return dict(placed=placed, touched=sorted(touched), close_n=close.grip_force_n,
                lift_n=lift.grip_force_n, grasp=close.grasping and lift.grasping,
                rise_mm=1000 * (z1 - z0))


print("| cube | gripper open | hand boxes | contacts at placement | robot geoms touching the cube"
      " | grasp held | grip force at close (N) | at lift (N) | cube rise (mm) |")
print("|---|---|---|---|---|---|---|---|---|")
for half, open_rad in ((0.010, -0.6), (0.020, -0.785)):
    for boxes in (False, True):
        r = run(half, open_rad, boxes)
        print(f"| {int(2000 * half)} mm | {math.degrees(open_rad):.0f} deg | {'with' if boxes else 'without'}"
              f" | {r['placed']} | {', '.join(r['touched'])} | {r['grasp']} | {r['close_n']:.3f}"
              f" | {r['lift_n']:.3f} | {r['rise_mm']:.1f} |")
