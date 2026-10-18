# ADR-0010: The October layout — the robot moves, REST becomes HOME

- Status: Accepted. Covers #192. Simulator only.
- Date: 2026-10-17; amended 2026-10-18 (the owner's answers to the second STOP)
- Relates to:
  - #185 and PR #188 (`docs/reviews/2026-10-14-taught-board-poses-185/`: the evidence);
  - #187 (the FK tip convention, still open);
  - ADR-0002 and ADR-0006 (the rig corridor, which this retires);
  - ADR-0003 (the hand tube the footprint guard uses).

## Context

Since the October repositioning the real Reachy reaches all nine squares; Siva confirmed it: "reposition. Yes robot can reach all squares now". The sim modelled the August layout:
- torso at (0, 0, 1.0), from the URDF's `pedestal` joint;
- `r3c1` and `r3c2` tagged `no-right-arm`;
- the arm stored hanging in the rig's rail pocket (HOME), and rested over grid column 3 (REST).

#185 measured the board from the camera (17 views) and from Siva's taught poses. Both put the board surface about 0.41 m below the shoulder line, not 0.26 m. The taught poses also put the October board about 0.24 m closer to the robot than in August.

**Owner decisions, 2026-10-17:**
1. Adopt a 1.15 m shoulder line and the October layout.
2. Move the robot, not the scene. The board, grid, cells and rig keep their coordinates, which tests, notebooks and evidence use.
3. On the three problems the move exposed (STOP A, below): REST is HOME, the left arm becomes a phantom, and the right shoulder-roll stop becomes +20.3°.

**Owner decisions, 2026-10-18** (the second STOP: the crane's red_cube pick fails offline at the new mount, and 254 tests failed):
4. Land #192 without the red_cube live check. The crane is re-tuned at the October mount in a follow-up issue.
5. Retire August entirely: no August routes, poses or replays in the code, and no August-mount test helper. docs/reviews keeps its history.
6. Re-fly WAVE and POINT from the new REST and PRESENT, offline, as LIFT_TO_PRESENT and LOWER_TO_REST were. They must not stay refused.
7. Give the hand a simplified collision model, clear of the pinch, and re-settle REST on it.

## STOP A: what the move exposed

At torso (0.206, 0, 1.15) the shoulders are directly over the board (x 0.16–0.747), 0.41 m above it. The arm hangs about 0.53 m before the hand.
- **HOME is inside the board.** An arm at zero, the sim's start pose, runs both forearms through the 25 mm slab.
- **The left arm is inside the board in every pose.** The real robot has no left arm, but the sim does.
- **The pocket is gone.** The rail pocket is behind the shoulders, so the corridor out of it no longer exists: PLACE_ROUTE, STOW_ROUTE, RAISE_TO_SIDE and STOW_FROM_SIDE, measured to the millimetre (ADR-0002, ADR-0006).
- **r3c1 is out of reach** by 48.9 mm at 5 cm above the surface, with the shoulder roll on the URDF's +10° stop.

## Decision

### 1. The mount

| | August | October |
|---|---|---|
| Torso origin in the scene frame | (0, 0, 1.0) | **(0.206, 0, 1.15)** |
| Right shoulder | (0, −0.19, 1.0) | (0.206, −0.19, 1.15) |
| Neck pivot | (0.015, 0, 1.095) | (0.221, 0, 1.245) |

- **The MJCF.** `pedestal` stays at the scene origin, because the scene's `frame_id` and RViz's fixed frame are both `pedestal`. The torso carries the whole offset, which is one translation, as in the URDF.
- **Python.** `ARM_BASE_IN_WORLD` in `scene/awareness.py` states the mount. `gaze.NECK_PIVOT_IN_WORLD`, `primitives._HEAD_WORLD` and `crane_pick_live._SHOULDER_XY` are derived from it.
- **The native server.** `native_mujoco/placement.py` cannot import `reachy_ai`, so it restates the mount.
- **RViz.** `scripts/urdf_with_mount.py` rewrites the URDF's `pedestal` joint, and supervisord pipes the stock URDF through it. This takes effect on an image rebuild.
- **The check.** `tests/unit/test_robot_mount.py` holds all of these to the compiled MJCF.

**The pedestal stand-in is a render-only stub.** It runs from 2 cm above the board into the bust, with radius 0.05. A cylinder to the floor at x = 0.206 would pierce the board, and how the real mast meets the rig is not modelled. The stub clears the table by 2.0 cm and the inner rails by 4.5 cm, and no camera ray reaches it at the board view.

### 2. REST is HOME

Siva confirms the arm rests on the table, as in the lab photos. So REST is where the arm is stored, what "stow", "store" and "put your arm away" mean, and the sim's start pose (the MJCF `home` keyframe).
- `POSTURE_HOME` is kept as a name; it is `POSTURE_REST`.
- The posture graph is REST ⇄ PRESENT, plus WAVE at PRESENT.
- "Rest your arm" no longer asks which rest: there is one.

**REST is a pose the arm stays in with the motors off.** That matters because the sim starts with the motors off, exactly like the robot.
- An IK pose on the margin with the elbow swung out slid off the board's edge within two seconds.
- So REST was found by letting go: the arm was dropped motors-off in MuJoCo from many starts on the margin.
- Only fixed points were kept: poses that a second drop moved by less than 2°, and then only those whose gripper was still on its shut stop after a 120 s drop.
- They were then scored against six objects, up to the 12.2 cm soda can, standing on each column-3 cell, by the footprint guard's own number.
- The chosen pose: joints (10.5, 15, −55, −81, 84.5, −4.5, −17.5), gripper shut. The hand rests on its palm and back on the board's right margin, fingertips just over the board's edge (tip at about (0.35, −0.40); the grid's outer border is at y = −0.2286, the board's edge at −0.349), with the forearm along the margin.
- Motors off for 10 minutes from there: the gross joints drift at most 1.3°, the gripper stays on its shut stop, and the arm is still. The forearm yaw (84.5°) is what holds the hand shut: it turns the hand so gravity closes the finger.
- The shoulder roll does not lean on the new +20.3° stop: the search's best scores did, and were passed over so REST does not depend on that deviation.

**REST was searched three times.**
1. The first October REST (13.5, 13.5, −55, −81, −74, 44, 44) rested on the finger pad, because the hand had no collision geometry but the two pads, and its visual shell sank 38 mm into the board.
2. With the hand boxes (section 6) the hand rests on them instead. That pose no longer stayed put, so REST was searched again: (12, 14, −54.5, −81, −5.5, −21.5, 44). It held for the 10 s the search checked, but on the live panel the gripper was found at its open stop after 11 minutes (offline: +18° at 5 s, −15° at 120 s). Its finger hangs open, and the motors are off.
3. The third search added the 120 s gripper check. An open hand on the margin is the wide hand the footprint guard assumes away, so a REST whose gripper creeps open is wrong even though the arm stays put.

**The gripper is shut at REST.** The footprint guard sizes the hand to its aperture. An open hand's tube (8–11.5 cm radius) on a 10.8 cm margin reaches column 3 whatever the arm does; shut, it is 3.9–6.7 cm. `FOOTPRINT_MARGIN` is unchanged: the arm lands somewhere else; the rule did not move.

**The routes.**
- LIFT_TO_PRESENT: REST → REST_OUT → PRESENT_SHUT → PRESENT.
- LOWER_TO_REST: the reverse. It is also what "stow" means now, and what `primitives.raise_to_side` / `stow_from_side` fly (they used to fly the pocket corridor).
- REST_OUT is REST's arm shape at PRESENT's roll, with the hand off the board's edge. The hand therefore comes onto the margin from outside the board, not across column 3. The straight joint line PRESENT → REST crossed a soda can on r1c3 by −2.2 cm.

**Validation, offline physics** (the simulator's MuJoCo model and contact settings, the arm streamed with the motors on, no server, SDK or bridge). Flown with tall objects on r1c3, r2c3 and r3c3, and as the approach and return of every WAVE and POINT flight (section 7):
- no object moved;
- the worst MJCF distance was 1.8 cm (blue_cylinder on r1c3);
- the arm arrived within 1°;
- the only robot contacts were the forearm and the hand's palm and back on the board, at REST.

The footprint guard's own number, the worst of six objects on each column-3 cell: r1c3 +1.8, r2c3 +12.0, r3c3 +20.8 cm (August: +2.3, −5.7, −9.4).

Route records carry the mount they were flown at, and `check_route` accepts only the current one.

### 3. The phantom left arm

The joints, bodies and actuators stay, so every `l_*` name answers the SDK as before. Its geoms do not collide (contype and conaffinity 0) or render (group 4), and its mass is unchanged.

### 4. The right shoulder-roll stop: +20.3°

This is a deliberate, SDK-visible deviation from the URDF's +0.174 rad. It is now +0.354 rad in:
- the MJCF joint range and actuator ctrlrange;
- `joint_map`;
- `kinematics.JOINT_LIMITS_DEG`;
- the fake server's IK bounds.

The physical robot held +20.3° at B2_SAFE in Siva's taught poses (#185). With it, r3c1 is reachable: it needs at least 15.3° of roll at 5 cm above the surface and 19.7° at 3 cm. The left arm keeps its limits.

What an SDK client sees differently:
- `r_shoulder_roll` goals up to +20.3° are followed instead of clipped at +10°;
- IK solutions may use that range.

### 5. The board view

`gaze.BOARD_VIEW` is the named head pose: neck (roll 0, pitch 64.0, yaw 0), commanded as neck joint goals.
- **How it is computed.** `gaze.board_view_pose` uses a pinhole model of both eyes, built from the MJCF offsets and the measured focal lengths. It searches the pitch for the largest worst-case margin around the nine cells.
- **The neck stop binds.** The best view sits on the clamp (1° inside +64.7°), with 66 px to spare. The pinhole agrees with MuJoCo's own cameras to 0.5 px.
- **What it shows.** All nine squares are in both frames. The modelled chest box hides the near tape border and about the bottom quarter of the near row.

`PITCH_BIAS_DEG` was re-fitted from 10.8 to 8.5. At the new mount the centring offset depends on distance: +5.7 to +11.3 over the cells the neck can centre. See its docstring.

### 6. The hand's collision model

Two boxes fill the hand's visual shell (`r_thumb_body`) in `r_gripper_thumb`: `r_hand_back_col` (the wrist side) and `r_palm_col`, whose underside stays 5 mm above the pads' tops. They are in `r_gripper_thumb`, not `r_wrist2hand`, so they turn with wrist roll as the shell does (−17.5° at REST). The R12-502 pads are unchanged.

**The pinch is unchanged** (`grasp_before_after.py`, the R12-502 fixture with and without the boxes): a 20 mm cube and a 40 mm cube (the gripper open at −45°), each centred at the pad height. Only the two pads touched the cube, the grasp held through the lift, and the grip force was identical with and without the boxes: 28.0 N at the close and 30.4 N at the lift for the 20 mm cube, 53.6 N and 85.8 N for the 40 mm cube.

At REST the visual shells now sit at the board (−0.6 mm and −3.2 mm, contact depth), not 38 mm and 25 mm inside it.

**The pedestal stub stopped colliding.** It stands over the board's near edge, under the chest, where nothing real is. As a collider it was an obstacle only the sim had: pointing at r1c1, r1c2 and r2c2 drove the forearm and hand into it. It is render-only now, as section 1 always described it.

### 7. WAVE and POINT, re-flown

Flown offline from REST through the panel's own code (`rig_motion.travel`, `rig_motion.wave`, the motion worker's `_point_at_target`), with only the SDK's `goto` replaced by a minimum-jerk streamer (`fly_wave_point.py`).
- **WAVE**, three layouts (tall objects on column 3 twice; a can on the centre cell with cylinders on r1c1 and r3c1): all three cycles every time, back at PRESENT within 1°, nothing moved, no contact during the wave, worst MJCF distance 3.6 cm.
- **POINT, empty board:** all nine cells, miss 0.4–1.4 cm, nothing touched. August managed 8 of 9 (r3c1 out of reach), miss 2.2–8.8 cm.
- **POINT, one object at a time on the centre cell** (six types), every cell and then the object: 57 of 60 cells pointed; r1c2 was refused three times by the guard (once mid-flight, twice before moving). Pointing at the object worked for the three boxes and blue_cylinder. The escort stopped pool_cyl_1 and soda_can after its last leg: the tube hand read 0.1–0.3 cm inside them while the MJCF hand was 3 cm clear. **Nothing moved in any of the 69 pointings.**
- Two or more objects on the board are still not covered, and `_ability_available` still refuses them.

The rows are in `ROUTE_COMPATIBILITY`, for FWDCenterLabSivaPool and FWDCenterLabMCC. The panel's `wave`, `point_cell` and `point_object` abilities are available again, at `route_version` 2.

### 8. August, retired

Removed from the code, with the mount they belong to:
- the rail-pocket poses (HOME as a place, GRIP_SHUT, BACK, CURL, CURL_HIGH, TUCK, SWING_1–3, SWING_1_NOTEBOOK, HOVER), REST_AUGUST and REST_SHUT_AUGUST;
- the routes PLACE_ROUTE, STOW_ROUTE, RAISE_TO_SIDE, STOW_FROM_SIDE and their *_AUGUST lift and lowering;
- the corridor margin (#74's CORRIDOR_* figures, `corridor_shortfall`, `corridor_entry_refusal`, `with_notebook_swing_1`) and the stranded-pocket recovery (`stranded_at`, `resume_to_home`);
- every August route record and rejected attempt;
- the tools that only replay or fly August: `tools/goalfix_cmp` (retired 2026-10-08), the E1 Stage 1 campaign scripts and `scripts/e1_tail_check.py`, with their tests and fixtures. The E1 reset client (`reset.sh`, `reset_verify.py`) stays: `reset_watcher.py` uses it.
- the tests that replayed August recordings or measured August geometry.

`rest_forearm` and `stow_arm` both fly LOWER_TO_REST now; their recipes are at `route_version` 3, recipe version 2. docs/reviews keeps the August evidence, unchanged.

## Consequences

- **Reach.** All nine cells are reachable; `no-right-arm` was removed from r3c1 and r3c2.
- **REST footprint.** Objects on column 3 no longer trigger REST refusals:
  - before, every object on r2c3 and r3c3 was refused (−3.1 to −9.4 cm), including #192's pool_box_3 on r2c3 at −3.7 cm;
  - now the worst case is +1.8 cm (blue_cylinder on r1c3, beside the forearm).
- **The pocket is gone** from the code (section 8), not only from the posture graph.
- **WAVE and POINT are validated again** at the October mount (section 7).
- **The crane is not re-tuned.** `place_sweep` plans placements only from r3c2 now (it was r2c2), and the offline red_cube pick halts at the hover: the lateral correction it needs is past the 6 mm cap, and a wider cap does not make the placement clean. #192 lands without the red_cube live check; the re-tune is a follow-up issue.
- **Other scenes.** The move is global. Measured by dropping the start keyframe, motors off, for 5 s (`other_scenes.py`):
  - the three FWDCenterLab scenes: 0.6° of drift, recognised as REST.
  - `tabletop_demo.yaml` and `tabletop.example.yaml`: their table is narrower (edge at y = ±0.33), and the hand settles over its edge; 2.2° of drift, still recognised as REST.
  - `control_panel.yaml` and the two calibration scenes: there is no table under REST, so the arm falls to a hang (tip at z 0.50), which is no named posture. `scripts/demo_control_panel.py` flies its own poses.
  - None of these scenes was re-tuned. The joint poses in `primitives.py` (OVER_RED, …) are the August tabletop_demo tuning.
- **The kinematic backend** still starts at zero. It has no physics or panel, and its start pose is unchanged.

## Known limitations

- **The pedestal stub is a render gap.**
- **The crane pick is not re-tuned at the October mount** (section "Consequences"; a follow-up issue). Until it is, `lift_object` and `place_on_cell` plan, and are refused or halt, as the crane's own checks decide.
- **The hand's collision boxes are simplified.** Two boxes, not the mesh; the finger body still has only its pad.
- **r3c1 is reachable with little margin.** Near the surface it needs 0.6° short of the new stop.
- **REST's footprint is clear only for column 3, measured with the tube hand.** `right_tray`, a non-colliding marking, lies beside the hand at REST; nothing has been placed into it at the October mount.
- **The taught spacing (0.84× / 0.92×) is still open (#185).**
- **#187** (the FK tip convention) is still pending.
- **The supply spot is a follow-up.**
