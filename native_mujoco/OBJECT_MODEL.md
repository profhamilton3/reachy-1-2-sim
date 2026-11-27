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

## Compound geometry (#212)

For shapes one primitive cannot express, above all a through-hole (MuJoCo collides a mesh as its convex hull, which fills it):

```yaml
geometry:
  kind: compound
  parts:        # 1..64; each in the OBJECT frame
    - {kind: box, size: [x, y, z], pos: [x, y, z], quat_wxyz: [w, x, y, z]}
    - {kind: cylinder, radius: r, length: l, pos: [x, y, z], quat_wxyz: [w, x, y, z]}
```

Rules (schema and `scene_io.check_compound_object`, which also runs without `jsonschema`): unknown or missing fields fail; numbers are finite and sizes positive; quaternions are unit; **the bounding box is centred on the object origin**; a dynamic compound needs `physics.mass`; `articulation` is not supported.

| Consumer | What it does with a compound |
|---|---|
| compiler | one geom per part (same colour, contact channel and friction), no per-geom mass; a dynamic body gets an explicit `<inertial>` with the total mass, inertia computed from the parts at uniform density (overlaps count twice) |
| `ObjectPlacer.place` | rest height from the lowest point of any part |
| `ObjectPlacer.reshape` | refused, for every argument |
| `GripperModel` | finger and thumb on different parts of one free body count as a pinch |
| planner `SceneModel` | the solid bounding box, holes included |
| `truth.v1` | outer box or cylinder, chosen by `semantic_class` (`block`/`cube` or `cylinder`); anything else is refused; no hole field |
| RViz markers | one marker per part |
| side-on and crane grasps | refuse the kind `compound` (not validated) |

**Contact cost.** Many coplanar parts resting on one face are expensive with `noslip_iterations` on: the pass grows with the square of the contacts on one body. Let only as many parts reach the face as the footprint needs and recess the rest slightly, as the PART rings do (`docs/scene-object-dimensions.md`).

## Colour randomisation (#212)

A top-level `color_randomization: {tag, seed, palette}` repaints every object carrying `tag`, choosing `sha256("<seed>:<id>")` modulo the palette. `scene_io.load_scene` applies it once, after `extends` is resolved, so all consumers of one file agree. A child may override just `seed`; `scripts/make_part_colour_scene.py` writes such a child per run. The recorder manifest stores `color_randomization: {tag, seed, colors}` (null otherwise).

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
