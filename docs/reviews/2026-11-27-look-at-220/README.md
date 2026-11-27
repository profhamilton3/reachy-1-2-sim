# `primitives.look_at` through `gaze.neck_angles_for` (#220)

Sim only: offline MuJoCo on macOS arm64, with no server, SDK or robot.

| File | What it is |
|---|---|
| `look_at_before_after.py` | Makes the two files below: `python3 look_at_before_after.py REPO OUT_DIR`. |
| `callers.txt` | Raw output: each caller's target, the angles before (the formula at `d889a2a`) and after, and where the target lands in the model's cameras. |
| `left_eye_before_after.png` | The left eye, aimed by each formula at the red cube on three cells, arm at REST (pinhole, measured fovy). |

**Reading them.** The offsets in `callers.txt` are measured midway between the two eyes. In one eye alone, a centred target sits to one side by that eye's offset from the head's centre line: in the left eye it appears **right** of the cross. At r3c3 the left eye alone therefore looks about as good before as after, while the midpoint moved from 60 px off to 0.

The `control_panel` and `tabletop_demo` targets are computed from those scene files. The demos themselves were not run: they need the SDK and a server, and have no dry-run mode.
