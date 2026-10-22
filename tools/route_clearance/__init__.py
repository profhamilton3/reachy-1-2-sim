"""Offline sim clearance records for tabletop arm routes (reachy-tabletop-ai#66, docs/design/arm-motion.md sections
3.2 and 10; A0b). It reads a tabletop route document (arm_routes.v1) as a wire contract, flies one route in the
October model, kinematic and physics, and writes the record the tabletop commits under
src/reachy_ai/motion/arm_route_records/. No server, SDK or Docker: MuJoCo in this process only.

Run: ``python -m tools.route_clearance --routes <arm_routes.json> --route B2_GRIP_TUNE
--scene FWDCenterLabSivaPool --out <dir>``
"""

import pathlib as _pathlib
import sys as _sys

_REPO = _pathlib.Path(__file__).resolve().parents[2]
for _p in (_REPO / "src", _REPO / "native_mujoco"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
