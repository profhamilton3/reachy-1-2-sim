# Research links — authoritative sources for Reachy 1.2

Hardware here is **Reachy 2021 = Reachy v1.2**. Pollen's current docs site
(`docs.pollen-robotics.com`) covers **Reachy 2**, which is a different robot:
different actuators, different cameras, different kinematics. Do not use Reachy 2
numbers for this project. The v1 documentation lives on the archived GitHub Pages
site below.

Retrieved 2026-08-27.

## Robot description (kinematics — the authority)

- `pollen-robotics/reachy_description` — Foxy package, Reachy 2021 URDF + `.dae`
  meshes. Note the **underscore**; `reachy-description` (hyphen) does not exist.
  A copy is vendored at `native_mujoco/model/reachy.urdf`.
- Verified 2026-08-27: the right-arm chain in `native_mujoco/model/reachy_1_2.xml`
  matches this URDF exactly — `r_elbow_pitch` at `0 0 -0.28`, `r_wrist_pitch` at
  `0 0 -0.25`, `r_wrist_roll` at `0 0 -0.0325`, `r_gripper` at `0 -0.037 -0.03998`.
  Cumulative torso → `r_gripper_finger` = 0.6025 m.

  **Known deviation:** the URDF's `neck_roll` origin carries `rpy="0 0.174 0"`
  (a 10° forward head pitch). `reachy_1_2.xml` drops it — `head_x` is declared
  `pos="0.015 0 0.095"` with no rotation. Effect on the camera is small
  (z −1.5 mm, x +10.5 mm) but it shifts the head's neutral gaze by 10°, which is
  why the sim's home-pose depression is 25.8° where the real captures cluster
  around 36°. Not yet fixed — it touches the shared robot model.

## Documentation (Reachy 2021 / v1.2)

- Full docs: https://pollen-robotics.github.io/reachy-2021-docs/
- Source repo: https://github.com/pollen-robotics/reachy-2021-docs
- Cameras: https://pollen-robotics.github.io/reachy-2021-docs/sdk/first-moves/cameras/
- Arm: https://github.com/pollen-robotics/reachy-2021-docs/blob/master/content/SDK/first-moves/arm/index.md

## Confirmed specifications

| Item | Value | Source |
|---|---|---|
| Arm workspace | **65 cm radius sphere centred on the shoulder** | Reachy 2021 docs |
| Arm DoF | 7 + 1 gripper, 8 joints | Reachy 2021 docs |
| Camera module | **Kurokesu C1 Pro** | Reachy 2021 docs |
| Camera sensor | Sony IMX290, 1/2.8" (6.46 mm), 2.9 µm pixels, 1920×1080 @30fps | kurokesu.com |
| Camera lens | **Not supplied with the module** — Pollen fits a motorised zoom | kurokesu.com |
| Zoom levels | `'in'`, `'out'`, `'inter'`; speed 4000–40000, default 10000 | Reachy 2021 docs |
| SDK frame shape | `(720, 1280, 3)` in the docs example | Reachy 2021 docs |
| Our kit | Starter Kit + VR teleop, white, **right arm only**, v1.2 | vendor email, 2026-07-09 |

The 65 cm figure corroborates the sim: with the table at 0.74 m and the torso at
1.0 m, a 0.65 m sphere puts `cell_r3c1` and `cell_r3c2` outside and `cell_r3c3`
right on the boundary — matching Siva's physical check. See
`scenes/FWDCenterLabMCC.yaml`.

## No published field of view

Pollen does not state a camera FOV, and cannot meaningfully: the C1 Pro ships
without a lens and Pollen's is a **motorised zoom**, so FOV is a function of zoom
position. Our own calibration is therefore the only FOV figure available, and it
is only valid for the zoom level in force when the images were taken:

- 76.2° across the long (640 px) image axis, 61.1° across the short (480 px) axis
- `k1 ≈ −0.32` (left) / `−0.39` (right) — strong barrel
- derived from `IITG-Reachy-Project/tests/Pictures/`, 20 stereo pairs,
  reprojection RMS 0.97 / 0.84 px

Caveat: those captures are 480×640, i.e. a rotated **640×480 (4:3)** frame, while
the docs show a 1280×720 (16:9) SDK frame. So the capture path resizes, crops or
rotates somewhere. Pin that down before matching the sim renderer to it — if the
4:3 frame is a crop of a 16:9 sensor, the full sensor sees wider than 76.2°.

## Reachy 2 — reference only, do not apply here

- https://docs.pollen-robotics.com/ — Reachy 2 hardware guide
- Different arm actuators (Orbita 2D/3D), IMX296 global-shutter cameras, an
  Orbbec Gemini 336 depth camera. None of it transfers to v1.2.
