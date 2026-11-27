# Container checks (sim #212, PR #219), 2026-11-27

Image `reachy-1-2-sim:latest` = `sha256:ad65ee229ae2ef4f48fbd96240e98d130e505170fae3449d7efc8c7e1e21308f` (linux/amd64 on Docker 29.6.2, arm64 Mac, emulated), built with `docker compose build` from the branch after merging `main` 970ab3c. Inside: Python 3.8.10, ROS 2 Foxy, reachy-sdk 0.7.0, grpcio 1.70.0, pytest 4.6.9, no MuJoCo. The stack ran with `REACHY_PANEL_LANGUAGE=0 REACHY_PANEL_EXECUTOR=0` and an empty `ANTHROPIC_API_KEY`.

## Compile (code and scenes baked into the image)

```
FWDCenterLabSivaPool         objects=53 PART=16 compound_primitives=144 seed=None mjcf=19cb409eef47064c
FWDCenterLabSivaPoolBlack    objects=53 PART=16 compound_primitives=144 seed=None mjcf=41bd153379684a02
FWDCenterLabSivaPoolRandom   objects=53 PART=16 compound_primitives=144 seed=2026 mjcf=b3e978ab4d1e27a7
python 3.8.10 x86_64 | scene_io from /opt/scene_io.py
jsonschema schema validation: on | mujoco: not in the image
FWDCenterLabSivaPool         host MJCF == container MJCF: True   host MuJoCo 3.11.0 compile: nbody=78 ngeom=227
FWDCenterLabSivaPoolBlack    host MJCF == container MJCF: True   host MuJoCo 3.11.0 compile: nbody=78 ngeom=227
FWDCenterLabSivaPoolRandom   host MJCF == container MJCF: True   host MuJoCo 3.11.0 compile: nbody=78 ngeom=227
```
`scene_compiler.py` is not shipped in the image; it was imported from the source copy. Without MuJoCo the container cannot build an MjModel; its MJCF text equals the host's byte for byte, and the host compiles that with MuJoCo 3.11.0.

## Tests

The repo documents no container test run. All of `tests/` was run with the image's Python, one file at a time with a 600 s limit, on read-only copies of this branch and of `main` 970ab3c (no `.git`, no ignored files). Running the whole tree in ONE pytest process stalled at about 71 % until a 3000 s limit, on both; file by file, every file finishes.

| | files | passed | failed | error | skipped |
|---|---|---|---|---|---|
| `main` 970ab3c | 119 | 2558 | 117 | 60 | 43 |
| branch | 122 | 2701 | 118 | 60 | 116 |

The one extra branch failure was a test in this PR that imported a MuJoCo module before skipping; fixed (the three new test files then gave 143 passed, 74 skipped). **Every other failure or error also happens on `main`, with the same error messages.** Main causes: `datetime.UTC` (needs Python 3.11), MuJoCo not installed, `pytest.MonkeyPatch` (not in pytest 4.6.9).

The SDK/gRPC files the host skips (same on both): `test_arm_kinematics_tip` 17 passed; `test_mujoco_camera_adapter` 5 passed; `test_sdk_compat` 7 passed; `test_camera_service` 13 passed, 1 failed (`test_stream_yields_frames`, 2 != 3); `test_goal_reporting_sdk_path` does not collect (`rig_routes` has no `HOVER`).

## smoke_test_host.py, in the container, against the fake server (kinematic, Pool scene)

`RESULT: PASS`: connection, 21 joints, compliance toggle, a bounded command, sensors and fans, and 30 manipulable objects (all 16 parts) with their grasp points.

## RViz markers (RViz itself was not looked at)

`ros2 topic info`: one publisher (`scene_marker_publisher`), one subscriber (`rviz`), reliable, transient local; `rviz/reachy.rviz` displays that topic. `check_rviz_markers.py`:

```
scene FWDCenterLabSivaPool: 189 markers on /reachy_sim/scene/markers (expected 189), ids unique: True
  part110_a  box      primitives  1/ 1  rgb (0.812, 0.627, 0.137)
  part110_b  box      primitives  1/ 1  rgb (0.812, 0.627, 0.137)
  part111_a  compound primitives 20/20  rgb (0.812, 0.627, 0.137)
  part111_b  compound primitives 20/20  rgb (0.812, 0.627, 0.137)
  part120_a  box      primitives  1/ 1  rgb (0.784, 0.353, 0.09)
  part120_b  box      primitives  1/ 1  rgb (0.784, 0.353, 0.09)
  part121_a  compound primitives 20/20  rgb (0.784, 0.353, 0.09)
  part121_b  compound primitives 20/20  rgb (0.784, 0.353, 0.09)
  part210_a  cylinder primitives  1/ 1  rgb (0.42, 0.51, 0.6)
  part210_b  cylinder primitives  1/ 1  rgb (0.42, 0.51, 0.6)
  part211_a  compound primitives 16/16  rgb (0.42, 0.51, 0.6)
  part211_b  compound primitives 16/16  rgb (0.42, 0.51, 0.6)
  part220_a  cylinder primitives  1/ 1  rgb (0.306, 0.604, 0.204)
  part220_b  cylinder primitives  1/ 1  rgb (0.306, 0.604, 0.204)
  part221_a  compound primitives 16/16  rgb (0.306, 0.604, 0.204)
  part221_b  compound primitives 16/16  rgb (0.306, 0.604, 0.204)
PART bodies: 16; PART primitives matched: 152/152
override of part121_a to [0.432, 0.0, 0.774925]: 20/20 markers moved together
RESULT: PASS
exit 0

scene FWDCenterLabSivaPoolBlack: 189 markers on /reachy_sim/scene/markers (expected 189), ids unique: True
  part110_a  box      primitives  1/ 1  rgb (0.05, 0.05, 0.05)
  part110_b  box      primitives  1/ 1  rgb (0.05, 0.05, 0.05)
  part111_a  compound primitives 20/20  rgb (0.05, 0.05, 0.05)
  part111_b  compound primitives 20/20  rgb (0.05, 0.05, 0.05)
  part120_a  box      primitives  1/ 1  rgb (0.05, 0.05, 0.05)
  part120_b  box      primitives  1/ 1  rgb (0.05, 0.05, 0.05)
  part121_a  compound primitives 20/20  rgb (0.05, 0.05, 0.05)
  part121_b  compound primitives 20/20  rgb (0.05, 0.05, 0.05)
  part210_a  cylinder primitives  1/ 1  rgb (0.05, 0.05, 0.05)
  part210_b  cylinder primitives  1/ 1  rgb (0.05, 0.05, 0.05)
  part211_a  compound primitives 16/16  rgb (0.05, 0.05, 0.05)
  part211_b  compound primitives 16/16  rgb (0.05, 0.05, 0.05)
  part220_a  cylinder primitives  1/ 1  rgb (0.05, 0.05, 0.05)
  part220_b  cylinder primitives  1/ 1  rgb (0.05, 0.05, 0.05)
  part221_a  compound primitives 16/16  rgb (0.05, 0.05, 0.05)
  part221_b  compound primitives 16/16  rgb (0.05, 0.05, 0.05)
PART bodies: 16; PART primitives matched: 152/152
RESULT: PASS
publisher log: 0 traceback(s); Scene loaded: 'FWDCenterLabSivaPoolBlack', 53 objects, frame_id='pedestal'

scene FWDCenterLabSivaPoolRandom: 189 markers on /reachy_sim/scene/markers (expected 189), ids unique: True
  part110_a  box      primitives  1/ 1  rgb (0.784, 0.353, 0.09)
  part110_b  box      primitives  1/ 1  rgb (0.306, 0.604, 0.204)
  part111_a  compound primitives 20/20  rgb (0.812, 0.627, 0.137)
  part111_b  compound primitives 20/20  rgb (0.784, 0.353, 0.09)
  part120_a  box      primitives  1/ 1  rgb (0.306, 0.604, 0.204)
  part120_b  box      primitives  1/ 1  rgb (0.812, 0.627, 0.137)
  part121_a  compound primitives 20/20  rgb (0.784, 0.353, 0.09)
  part121_b  compound primitives 20/20  rgb (0.306, 0.604, 0.204)
  part210_a  cylinder primitives  1/ 1  rgb (0.42, 0.51, 0.6)
  part210_b  cylinder primitives  1/ 1  rgb (0.784, 0.353, 0.09)
  part211_a  compound primitives 16/16  rgb (0.306, 0.604, 0.204)
  part211_b  compound primitives 16/16  rgb (0.784, 0.353, 0.09)
  part220_a  cylinder primitives  1/ 1  rgb (0.42, 0.51, 0.6)
  part220_b  cylinder primitives  1/ 1  rgb (0.784, 0.353, 0.09)
  part221_a  compound primitives 16/16  rgb (0.306, 0.604, 0.204)
  part221_b  compound primitives 16/16  rgb (0.784, 0.353, 0.09)
PART bodies: 16; PART primitives matched: 152/152
RESULT: PASS
publisher log: 0 traceback(s); Scene loaded: 'FWDCenterLabSivaPoolRandom', 53 objects, frame_id='pedestal'

```
The Random colours in the container equal the host's for all 16 (seed 2026).

## `test_panel_routes.py` with the full stack up

Native server (`mjpython`, bound to 127.0.0.1:8765, reached from the container through `host.docker.internal`) plus the container in `mujoco-remote`; tests run on the host. Branch f239a2d: 2 failed, 36 passed. `main` 970ab3c (with a server from `main`'s source; the container still ran the branch image): 2 failed, 36 passed. Same 409 `plan_invalidated`. They do not need a live simulator. Filed as #223; not fixed here.
