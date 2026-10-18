"""Print a Reachy URDF with its `pedestal` joint moved to the simulator's mount.

RViz draws the robot from Pollen's `reachy_description` URDF, whose `pedestal`
joint puts the torso at (0, 0, 1).  The scene markers RViz draws beside it are
in the scene frame, also called `pedestal`.  Since the October layout (#192)
the torso sits at `ARM_BASE_IN_WORLD` = (0.206, 0, 1.15) in that frame, so
the stock URDF would draw the robot 0.206 m behind and 0.15 m below where the
simulator has it.  supervisord.conf pipes the URDF through this before handing
it to robot_state_publisher; the URDF file itself, a pinned upstream asset, is
not edited.

    python3 scripts/urdf_with_mount.py <path/to/reachy.URDF>

The mount comes from src/reachy_ai/scene/awareness.py, the one place it is
stated for the Python side (tests/unit/test_robot_mount.py holds the MJCF to
it).  Exits non-zero, printing nothing, if the joint is not found exactly
once, so the caller can fall back to the stock file.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
sys.path.insert(0, "/opt/src")           # the container's mount of src/

_PEDESTAL_JOINT = re.compile(
    r'(<joint\s+name="pedestal"[^>]*>.*?<origin\s+xyz=")([^"]*)(")',
    re.DOTALL)


def with_mount(urdf: str, xyz) -> str:
    """`urdf` with the `pedestal` joint's origin xyz replaced by `xyz`."""
    text = " ".join(f"{v:g}" for v in xyz)
    out, n = _PEDESTAL_JOINT.subn(lambda m: m.group(1) + text + m.group(3),
                                  urdf)
    if n != 1:
        raise ValueError(f"expected one `pedestal` joint origin, found {n}")
    return out


def main(argv) -> int:
    from reachy_ai.scene.awareness import ARM_BASE_IN_WORLD

    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as f:
        urdf = f.read()
    try:
        sys.stdout.write(with_mount(urdf, ARM_BASE_IN_WORLD))
    except ValueError as exc:
        print(f"urdf_with_mount: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
