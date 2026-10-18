"""REST held motors-off for 10 minutes: the arm stays, and so does the shut gripper.

The second REST search kept fixed points over 10 s; one of them was found on
the live panel with its gripper at the open stop after 11 minutes (its finger
hangs open).  Every candidate was then held for 120 s, and REST is checked
here over 600 s.

    PYTHONPATH=src python3 docs/reviews/2026-10-17-october-layout-192/rest_hold.py .
"""
import pathlib, sys
import numpy as np
S = pathlib.Path(__file__).parent; sys.path.insert(0, str(S))
from settle import World, NAMES
REPO = pathlib.Path(sys.argv[1]); w = World(REPO)
from reachy_ai.motion import rig_routes as R
q = [R.REST[n] for n in NAMES]
for seconds in (10.0, 120.0, 600.0):
    q1, cons, tip, sp = w.settle(q[:7], grip=R.SHUT, seconds=seconds)
    print(f"{seconds:5.0f} s motors off: drift", np.round(np.array(q1) - np.array(q), 1).tolist(),
          "tip", np.round(tip, 3).tolist(), "speed", round(sp, 4), "contacts", cons, flush=True)
