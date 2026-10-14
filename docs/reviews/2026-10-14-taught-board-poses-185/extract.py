"""Extract the hand-taught poses from reachy-tabletop-ai tests/CalibrateBoard.ipynb.

Usage: python3 extract.py <CalibrateBoard.ipynb> <out.json> <source-commit>

Reads only literal np.array(...) / joint-dict cells; nothing in the notebook is executed.
"""
import ast, hashlib, json, re, sys

path, out, commit = sys.argv[1], sys.argv[2], sys.argv[3]
raw = open(path, "rb").read()
nb = json.loads(raw)
cells = [("".join(c["source"]), i) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]


def arrays(src):
    return {m.group(1): ast.literal_eval(m.group(2))
            for m in re.finditer(r'"?\b([A-Z]\d(?:_[A-Z_]+)?)\b"?\s*[:=]\s*np\.array\(\s*(\[.*?\])\s*\)', src, re.S)}


board, board_cell = next((arrays(s[s.index("BOARD_POSES"):]), i)
                         for s, i in cells if re.search(r"BOARD_POSES\s*=\s*\{", s))
named, named_cells = {}, {}
for s, i in cells:
    for k, v in arrays(s).items():
        if "_" in k and k not in named:
            named[k], named_cells[k] = v, i
joint_order = ["r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw", "r_elbow_pitch",
               "r_forearm_yaw", "r_wrist_pitch", "r_wrist_roll"]
safe_joints, safe_cell = next(
    ([float(dict(re.findall(r"reachy\.r_arm\.(\w+):\s*(-?[\d.]+)", s))[j]) for j in joint_order], i)
    for s, i in cells if re.search(r"B2_SAFE_JOINTS\s*=\s*\{", s))
height = next(float(m.group(1)) for s, _ in cells for m in [re.search(r"SAFE_HEIGHT\s*=\s*([\d.]+)", s)] if m)

json.dump({
    "provenance": {
        "source_repo": "profhamilton3/reachy-tabletop-ai", "source_commit": commit,
        "notebook": "tests/CalibrateBoard.ipynb", "sha256": hashlib.sha256(raw).hexdigest(),
        "board_poses_cell": board_cell, "named_pose_cells": named_cells,
        "b2_safe_joints_cell": safe_cell,
        "how_taught": "by hand on the real robot in Siva's lab, read back with r_arm.forward_kinematics()",
        "frame": "robot torso frame, metres; 4x4 homogeneous poses of the right_tip frame",
    },
    "board_poses": board,
    "named_poses": named,
    "fk_pair": {"note": "B2_SAFE_JOINTS are the joints of BOARD_POSES['B2'] raised by SAFE_HEIGHT",
                "joint_order": joint_order, "joints_deg": safe_joints, "safe_height_m": height},
}, open(out, "w"), indent=1)
print("squares:", sorted(board), "named:", sorted(named), "pair cell:", safe_cell, "SAFE_HEIGHT:", height)
