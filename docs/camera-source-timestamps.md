# Camera source timestamps: robot, simulator and stereo viewer

Investigated 2026-10-17. This is a gap in the camera protocol, not a defect.
It is written to become the basis of an issue, and to support a decision about
changing the robot's camera server. The two simulator problems found along the
way are fixed in the same change as this document (§5).

## 1. Summary

| | Stamp exists at the source | Reaches gRPC clients | Who drops it |
|---|---|---|---|
| **Physical robot** | Yes: ROS `header.stamp` per frame, robot system clock | No | the robot's `camera_server.py` keeps only `msg.data` |
| **Simulator** | Yes: frame `seq`, `sim_step`, `sim_time_s`, render time | No | the Docker bridge writes only the JPEG to `/tmp/reachy_{left,right}.jpg` |
| **Either source** | | | the camera API itself: `Image` has no field for it |

The tabletop stereo viewer logs `source timestamps unavailable` once per eye
and reports `capture_time_ns: null`. That line is a fixed statement. Every age
and pair skew it shows is measured when the **workstation** noticed the frame
(`observed_ns`), not when the camera captured it.

## 2. The protocol gap

- `reachy-sdk-api` 0.7.3, `camera_reachy.proto`: `message Image { bytes data = 1; }`.
  It carries the JPEG and nothing else.
- `reachy-sdk` 0.7.0, `reachy_sdk/camera.py` (`_start_sync_in_bg`): for each
  `StreamImage` message it does `cv.imdecode(resp.data)` and keeps only the
  pixels.
- So no SDK client can see a source timestamp, from the robot or the sim,
  whatever the server knows.
- By contrast, `JointsState` and `SensorsState` do carry a `timestamp`, and the
  robot fills it (§3). The SDK ignores those too.

## 3. Physical robot (as published by Pollen)

Source: `pollen-robotics/reachy_2023` (Reachy 1 workspace), default branch
`develop`. `camera_server.py` was last changed in 8f8cbe2 (2023-02-03).
**Not yet verified to be what our robot runs.** Check on the robot with `git -C
<reachy_ws>/src/reachy_2023 log -1` or by comparing the installed file.

| Step | File | What happens |
|---|---|---|
| Capture | `camera_controllers/camera_publisher.py` | `v4l2py` (`>=0.6.1`) reads MJPG frames from `/dev/{left,right}_camera` at 30 fps. One thread per eye; the cameras are free-running with no hardware sync. |
| Rotate | same, `_rotate` | Each frame is piped through a new `jpegtran -rotate 90/270` **subprocess**. |
| Stamp | same, `publish_img` | `header.stamp = self.clock.now()`: ROS time, which is the robot's system clock. It is taken **after** dequeue and rotation, so it is late by the subprocess time, which varies. |
| Serve | `reachy_sdk_server/camera_server.py` | It subscribes to `{left,right}_image/image_raw/compressed` (queue 1). `on_image_update` stores `msg.data.tobytes()` and **drops the header**. `StreamImage` waits on a per-side `Event`, so it sends only new images, never repeats. The camera server is separate from the joints server (50057 vs 50055), each with 10 threads. |
| Joints | `reachy_sdk_server/reachy_sdk_server.py` | `joints_state.timestamp.GetCurrentTime()` at send time: the same robot system clock as the camera stamp. |

One minor observation, unchanged by any edit below: one `Event` per side is
shared by all streaming clients. With two clients, one client's `clear()` can
make the other wait for the next frame.

## 4. Simulator

- The native MuJoCo server (`native_mujoco/server.py`, `protocol.CameraFrame`)
  stamps every frame with:
  - `seq`, `sim_step` and `sim_time_s`;
  - `wall_time_ns`, which is `time.monotonic_ns()` despite the name;
  - `render_us`.
- Left and right are rendered in the **same simulation step**, so a sim pair is
  exactly simultaneous.
- One caveat: with the latency sensor effect enabled, `sim_step` is the send
  step, not the render step.
- The bridge (`mujoco_remote_backend._ingest_camera_frame`) writes only the
  JPEG. `_MujocoRemoteCameraAdapter` in `fake_reachy_server.py` re-reads that
  file, and `StreamImage` sends `Image(data=...)`.

## 5. Simulator fixes in this change

1. **Repeated frames.** The adapter counted every re-read of the frame file as
   a new frame. `StreamImage`, which sends whenever the sequence changes,
   therefore resent the same image on every 1/15 s poll.
   - Measured on the live sim: the right frame file was replaced 9.7 times/s,
     while the stereo viewer reported 14.9 fps. About a third of its "frames"
     were repeats, and its age reading reset on each one.
   - Now a new frame is counted only when the bridge has written a new file
     (new inode/mtime, read with `fstat` on the open file), under a lock shared
     by the stream threads. This matches the robot, which never repeats.
2. **Only two SDK clients.** The sim serves joints, sensors and cameras from
   one gRPC server with `ThreadPoolExecutor(max_workers=10)`.
   - Each reachy-sdk 0.7.0 client holds 5 streaming RPCs for its whole
     connection: `StreamJointsState`, `StreamSensorStates`,
     `StreamJointsCommands` and one `StreamImage` per camera.
   - So the panel's motion worker plus the stereo viewer used every thread. Any
     further call, such as a notebook connecting or a `GetImage`, waited forever
     (measured: `GetImage` `DEADLINE_EXCEEDED` after 5 s).
   - Now `_GRPC_MAX_WORKERS = 32`, which fits six clients plus unary calls.

Tests: `tests/unit/test_mujoco_camera_adapter.py`. Each fix fails its own tests
when reverted, e.g. "only 10 of 15 streams were served".

## 6. Proposed edits to the robot's camera server

Both options touch only the camera path: no motion code and no safety gate. In
each case the stamp travels with the frame, so the left and right stamps of a
pair are on the same robot clock.

### Option A (recommended first): a JPEG comment, no API change

Put the metadata in a JPEG COM segment (`FF FE`) right after the start-of-image
marker. Every JPEG decoder skips COM segments, so the SDK's `cv.imdecode`,
Siva's notebooks and any other client keep working byte-for-byte as before. The
only file that changes is `reachy_sdk_server/camera_server.py`:

```python
import json, struct

STAMP_FORMAT = 1  # bump if the payload changes

def _with_comment(jpeg: bytes, payload: bytes) -> bytes:
    if jpeg[:2] != b"\xff\xd8" or len(payload) > 65533:
        return jpeg                       # not a JPEG we understand: send it unchanged
    return jpeg[:2] + b"\xff\xfe" + struct.pack(">H", len(payload) + 2) + payload + jpeg[2:]

# in __init__:
#     self.frame_seq = {'left': 0, 'right': 0}

def on_image_update(self, msg, side):
    """Get data from image. Callback for "/'side'_image "subscriber."""
    self.frame_seq[side] += 1
    stamp = msg.header.stamp
    payload = json.dumps({"reachy_frame": STAMP_FORMAT, "side": side, "seq": self.frame_seq[side],
                          "stamp_ns": stamp.sec * 1_000_000_000 + stamp.nanosec,
                          "clock": "robot_ros"}).encode()
    self.cam_img[side] = _with_comment(msg.data.tobytes(), payload)
    self.image_published[side].set()
```

- **Checked on a real sim frame (640×480):** `cv2.imdecode` gives
  pixel-identical output with and without the comment, PIL opens it, the comment
  adds 104 bytes, and the payload parses back.
- **Cost:** about 15 lines, and one extra copy of a roughly 40 KB JPEG per
  frame (2 × 30 fps).
- **Off switch:** an environment variable (e.g. `REACHY_CAMERA_STAMP=0`) can
  disable it without a rebuild.
- **Sim:** the same comment can be written by the sim bridge, with `"clock":
  "sim"` and `sim_time_s`/`sim_step`. The viewer then has one reader for both
  sources.

### Option B: a proper field in the API

Add `google.protobuf.Timestamp timestamp = 2; uint64 sequence = 3;` to
`Image` in `reachy-sdk-api`, regenerate its Python stubs on the robot, and set
the fields in `GetImage`.

- **Compatibility:** adding fields is wire-compatible. Old clients ignore them.
- **Cost:** it changes a Pollen package on the robot and needs the new stubs
  wherever the field is read. That is more moving parts than Option A for the
  same information.

### Optional: a better stamp at the source

The current stamp includes the `jpegtran` subprocess time. In `camera_publisher.py`:
- Read the V4L2 buffer timestamp of each `v4l2py` frame, which is set by the
  kernel when the frame is dequeued. Use it for `header.stamp` instead of
  `clock.now()` after rotation.
- V4L2 timestamps are normally `CLOCK_MONOTONIC`. Convert with a per-process
  offset to the wall clock.
- **To verify on the robot:** that the installed `v4l2py` exposes the buffer
  timestamp on its frame object. This is a separate, second change; Option A is
  useful without it.

### Deploying and reverting (robot)

1. Note the installed version, and copy `camera_server.py` (and
   `camera_publisher.py` if it changes) aside.
2. Edit in a git branch of the robot's workspace, `colcon build --packages-select
   reachy_sdk_server`, and restart the robot's SDK server service.
3. Check:
   - a notebook still gets frames;
   - `cv2.imdecode` and PIL open a saved `resp.data`;
   - the comment parses;
   - the `seq` values have no gaps at idle.
4. Revert: restore the file, rebuild, restart. A few minutes, no data at risk.

## 7. Viewer-side work (tabletop `reachy-tabletop-ai`), needed for any option

The SDK discards everything but pixels, so the viewer must read the raw stream
itself. Either open its own `StreamImage` per eye on the camera port, or wrap
the SDK camera loop the way `sdk_compat.py` wraps the command poll.

It then:
- fills `capture_time_ns`, `source_seq` and `clock`;
- reports **source** pair skew (`|left.stamp - right.stamp|`; no clock sync
  needed, both are robot clock);
- counts gaps in `seq` as source drops.

Frame age on the workstation needs the robot and workstation clocks related.
Either use NTP/chrony on both, or estimate the offset as the minimum of
(`observed_wall - stamp`) over a window. That minimum is offset plus the smallest
network delay, and it is good to a few ms on a LAN.

## 8. Risk assessment (for the PI)

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | The edited camera server fails to start or crashes: no images for anyone | Low (about 15 lines, no new dependencies) | Medium: camera work stops until reverted; **no motion impact** | Back up the file and test in a branch; revert in minutes (§6) |
| R2 | A client rejects the modified JPEG | Very low (COM is standard JPEG; decoders skip it) | Low–medium | Check `cv2.imdecode`, PIL and the notebooks on the robot before relying on it; env-var off switch |
| R3 | CPU or latency cost | Very low (one ~40 KB copy per frame) | Low | Measure fps before and after |
| R4 | The stamp is read as exposure time | Medium | Medium: wrong conclusions about sync | It is publish time after dequeue + `jpegtran`; label it so; follow-up kernel stamp (§6) |
| R5 | Robot clock steps (NTP) | Low | Low: one bad age sample | Use stamps relatively (pair skew, `seq`); clamp |
| R6 | The change is lost on a robot software reinstall | Medium over time | Low | Keep the patch and this doc in our repos |
| R7 | API skew (Option B only) | Medium | Low | Prefer Option A |
| R8 | Policy: changes on the robot computer | n/a | n/a | The PI decides who may change the robot computer, and how |

Motion safety is unaffected: the change touches only the camera
publish/serve path, and `safety.gate_check()` and the motion primitives are
untouched.

**What we would gain:**
1. **True pair synchrony.** The real capture-time skew between left and right
   for every captured pair, without clock sync. Today the viewer can only see
   when the frames reached the workstation, which mixes in network jitter. Pairs
   above a threshold could be refused for depth, which matters whenever the head
   or the scene moves.
2. **The head pose at capture time.** Joint states already carry robot-clock
   stamps (§3). With image stamps on the same clock, each pair can use the head
   pose from when it was taken for the camera→torso transform, instead of the
   latest pose.
3. **Latency and drop accounting** for perception feeding motion (#13 EPIC 9):
   end-to-end frame age (with clock sync) and source drops from `seq`.
4. **One code path for sim and robot.** The sim can emit the same comment with
   simulation time, and its pairs are exactly synchronous, a reference to
   validate against.

**Cost estimate:**
- robot: Option A, about 15 lines plus deploy and check;
- sim bridge: about 20 lines;
- viewer: about 100 lines plus tests.

**Decisions for the PI:**
- Is a camera-server edit on the robot acceptable?
- Option A or B?
- Should the kernel-stamp follow-up be in scope?
