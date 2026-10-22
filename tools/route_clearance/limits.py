"""The tabletop's limits and rig identity, transcribed (not imported: this repo does not ship the tabletop package).

Source: profhamilton3/reachy-tabletop-ai @ 7468ccbe731f382976f4f35cc60a0a107981e795
- ARM_LIMITS_DEG: src/reachy_ai/motion/arm_routes.py (SHA-256 adcdfed022dbaf26c649949ca9bba201926d956f98d24190666c39298ebc3af2).
  It is part of the route hash, so a copy that drifts gives a hash no tabletop route has, and the record is refused
  there as stale.
- NECK_LIMITS_DEG, ARM_RIG_LAYOUT_ID, ARM_MOUNT_ID: src/reachy_ai/config.py
  (SHA-256 77cb9719c6f186461951b68d451a2a63c3052bcefd2bae637ea0581538cbc57c).
"""

# Degrees, (lower, upper): the URDF rounded inward, plus the sim's deliberate r_shoulder_roll +20.3 (ADR-0010).
ARM_LIMITS_DEG = {
    "r_shoulder_pitch": (-150.0, 89.95),
    "r_shoulder_roll": (-179.90, 20.3),
    "r_arm_yaw": (-89.95, 89.95),
    "r_elbow_pitch": (-125.01, 0.0),
    "r_forearm_yaw": (-99.98, 99.98),
    "r_wrist_pitch": (-44.97, 44.97),
    "r_wrist_roll": (-44.97, 44.97),
    "r_gripper": (-68.75, 20.05),
}

# The owner-set neck limits (2026-10-20): positive pitch looks down, positive yaw turns to the robot's left.
NECK_LIMITS_DEG = {
    "neck_roll": (-15.0, 15.0),
    "neck_pitch": (-20.0, 64.0),
    "neck_yaw": (-45.0, 45.0),
}

ARM_RIG_LAYOUT_ID = "fwd-center-lab"
ARM_MOUNT_ID = "october-2026"
