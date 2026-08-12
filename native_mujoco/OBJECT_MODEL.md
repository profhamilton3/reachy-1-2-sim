# Reachy 1.2 Dynamic Object State (R12-503)

How scene objects become simulated bodies and how their poses stay coherent
across MuJoCo, the SDK/ROS/browser consumers, and camera frames.

Implemented in `scene_compiler.py` (scene → MJCF) and `objects.py`
(`ObjectTracker`, `build_scene_model`), wired into `server.py` (`SimState`),
consumed on the Docker side by `mujoco_remote_backend.py`.  Tested in
`tests/unit/test_scene_compiler.py` and `tests/unit/test_objects.py`.

## Scene → model

`build_scene_model_xml(scene_doc, robot_model_path)` compiles the validated
scene document into MJCF and merges it with `reachy_1_2.xml`:

- object `<body>` elements are inserted before the robot's `</worldbody>`;
- object meshes are merged into the robot's `<asset>`.

Object classification (from the scene schema):

| field | effect |
|---|---|
| `physics.dynamic: true` | body gets a `<freejoint>` + `mass` → simulated |
| `physics.dynamic: false` (default) | body welded to world (static) |
| `physics.collision` | `contype=4 conaffinity=7` (R12-500 object channel) or no contact |
| `physics.friction` | MuJoCo geom friction |
| `tracked: true` | pose is streamed in the `state` message |

Field names follow `scenes/scene.schema.json`: `geometry.size|radius|height`,
`pose.position` + `orientation_wxyz|rpy`, `material.rgba`.  `rpy` is converted
to a `wxyz` quaternion.

## Tracking & streaming

`ObjectTracker` discovers free-joint bodies (optionally filtered to the tracked
set) and reads their pose straight from `MjData.qpos`:

    state.objects = [{object_id, pos_xyz, quat_wxyz}, ...]

Because poses come from the same `MjData` as joints, grippers and camera frames,
every consumer sees one coherent `sim_step` — MuJoCo, RViz markers, and camera
images agree by construction.  Test `test_pose_coherent_with_mujoco_xpos`
asserts the streamed pose equals `data.xpos` exactly.

## Reset & seeded placement

`ObjectTracker.reset(data, seed=None, jitter_m=0.0)`:

- restores each object to its captured initial pose (deterministic);
- with a `seed` and `jitter_m > 0`, applies deterministic ±`jitter_m` uniform
  position jitter (same seed → identical placement; different seed → different).

Reset also zeroes object velocities.  The server's `reset` protocol message
carries optional `seed` and `jitter_m`.

Note: the home keyframe only covers the 21 robot DOFs, so `SimState._reset_physics`
restores free-joint objects from `model.qpos0` (their MJCF scene poses) after a
keyframe reset, then calls `mj_forward` so `xpos`/contacts are immediately
coherent.

## Docker-side consumption

`mujoco_remote_backend.RemoteSnapshot.objects` maps object id → `{pos_xyz,
quat_wxyz}`, ingested from the `state` message.  The existing ROS
scene-marker publisher and browser overlay read these to move dynamic markers;
static objects keep their scene-defined transforms.

## Limitations

- Scene hot-reload rebuilds the whole model; the current server loads the scene
  at startup (`--scene`).  Runtime `scene_load` swap is a follow-on.
- `restitution` is approximated via a fixed `solref`; per-object restitution
  tuning is not yet exposed.
