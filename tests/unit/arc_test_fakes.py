"""Fakes shared by the #56 arc and preflight tests.  Offline; nothing moves.

* `RecordingArm` -- the host IK mirror plus joints that record every goal write
  and can be scripted to lag.
* `RecordingRobot` -- records `turn_on`.
* `Spies` -- replaces the primitives that would command hardware with
  recorders, so a test can assert that NONE ran.
* `board` -- a FWDCenterLabMCC board with chosen objects on chosen cells and
  every other manipulable parked far off the board.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

from host_ik_arm import HostIKArm  # noqa: E402
from reachy_ai.motion import primitives as P  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.motion.kinematics import R_ARM_JOINTS  # noqa: E402
from reachy_ai.scene.awareness import SceneModel  # noqa: E402

SCENES = os.path.join(_HERE, "..", "..", "scenes")
LAB_SCENE = os.path.join(SCENES, "FWDCenterLabMCC.yaml")
DEMO_SCENE = os.path.join(SCENES, "tabletop_demo.yaml")


class _Joint:
    def __init__(self, arm, name, value):
        self._arm = arm
        self._name = name
        self.present_position = value
        self._goal = value

    @property
    def goal_position(self):
        return self._goal

    @goal_position.setter
    def goal_position(self, v):
        self._goal = v
        self._arm.writes.append((self._name, v))
        offset = (self._arm.lag_offset.get(self._name, 0.0)
                  if self._arm.lagging else 0.0)
        self.present_position = v + offset


class RecordingArm(HostIKArm):
    """IK like the simulator's; joints that record writes and can lag."""

    def __init__(self, start=None):
        super().__init__()
        self.writes = []
        self.lagging = False
        self.lag_offset = {"r_shoulder_pitch": 25.0}
        start = start if start is not None else R.PRESENT
        for name in R.R_JOINTS:
            setattr(self, name, _Joint(self, name, float(start.get(name, 0.0))))

    def place_at(self, q7):
        """Teleport the reading (not a command): where the arm 'is'."""
        for name, v in zip(R_ARM_JOINTS, q7):
            getattr(self, name).present_position = float(v)


class RecordingRobot:
    def __init__(self, arm=None):
        self.r_arm = arm if arm is not None else RecordingArm()
        self.turn_on_calls = []
        self.events = None        # a shared, ordered event list, when wanted

    def turn_on(self, part):
        self.turn_on_calls.append(part)
        if self.events is not None:
            self.events.append(("turn_on", part))

    def turn_off(self, part):
        self.turn_on_calls.append(("off", part))


class Spies:
    """Recorders standing in for the primitives that command hardware."""

    NAMES = ("look_at", "close_gripper", "open_gripper", "raise_to_side",
             "go_home", "smooth_move", "converge")

    def __init__(self, monkeypatch):
        self.calls = []
        self.streams = 0
        self.stream_hook = None
        self.sleeps = []
        monkeypatch.setattr(P.time, "sleep", self.sleeps.append)
        for name in self.NAMES:
            monkeypatch.setattr(P, name, self._recorder(name))
        real_stream = P.execute_trajectory

        def stream(arm, traj, names, rate_hz=25, on_step=None):
            self.streams += 1
            self.calls.append(("stream", self.streams, len(traj)))
            if self.stream_hook is not None:
                self.stream_hook(self.streams)
            return real_stream(arm, traj, names, rate_hz=rate_hz, on_step=on_step)

        monkeypatch.setattr(P, "execute_trajectory", stream)

    def _recorder(self, name):
        def rec(*a, **k):
            self.calls.append((name,))
        return rec

    def names(self):
        return [c[0] for c in self.calls]

    def count(self, name):
        return self.names().count(name)


def board(positions, scene_path=LAB_SCENE):
    """A copy of the lab scene: the named objects on the named cells, every
    other manipulable parked far off the board."""
    scene = SceneModel.from_yaml(scene_path)
    cells = {c: scene.cell_center(c) for c in scene.grid_cells()}
    scene.update_poses({k: (5.0 + i, 5.0, 0.0)
                        for i, k in enumerate(scene.manipulable_ids())})
    for oid, cell in positions.items():
        obj = scene.get(oid)
        c = cells[cell]
        scene.update_poses({oid: (c[0], c[1],
                                  scene.table_surface_z + obj.half_height)})
    return scene, cells
