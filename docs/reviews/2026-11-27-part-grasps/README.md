# Grasping the PART set (Part D; refs #212, #216, #214) — 2026-11-27

Environment: macOS arm64, Python 3.14.0, MuJoCo 3.11.0. **Offline physics only** (`tools.place_sweep.fly`): the crane module flies unchanged against the simulator's MuJoCo model and contact settings through the host mirror of its IK, with no server, SDK, bridge or latency. Nothing touched the robot.

| File | What it is |
|---|---|
| `gates.py` → `gates_before.txt`, `gates_after.txt` | Which shape gate refuses each design in each planner, on `main` `c46f4e4` and with this change. IK is cut off, so only shape checks can refuse. |
| `fly_grasps.py` → `results/part*.json` | One pick-and-place per design. The protocol (planner, cell pairs and their order, one flight each) is in its docstring and was fixed before flying. |
| `summarise.py` → `results_table.md` | The table below, generated from the JSON. |

## 1. Why each planner refused (on `main`)

| Design | Crane | Side-on | Refused only because it is `compound`? |
|---|---|---|---|
| 110, 120 solid blocks | passes the shape gates | class is `block`; side-on is for `cube` and `cylinder` | no |
| 111, 121 holed blocks | `compound` ("closes on two flat faces") | `compound`, class `block` | **crane: yes.** Side-on: no (the class refuses them either way) |
| 210 short solid cylinder | a cylinder | pads at 34 mm would not be 4 mm below a 25 mm top | no |
| 211 short holed cylinder | `compound` | `compound` | the gate was; the real rule behind it (cylinder / too short) still refuses |
| 220 tall solid cylinder | a cylinder | passes | — |
| 221 tall holed cylinder | `compound` | `compound` | **side-on: yes** |

There are two shape gates (`side_shape` and the crane's `kind != "box"`); the panel and executor add none.

## 2. What changed

`SceneModel.grasp_target(id)` gives the planners the object **being grasped** through `reachy_ai/scene/grasp_envelope.py`: a compound becomes a box or an enclosing cylinder **only if its outer surface is checked solid all round and over the full height**; otherwise it is refused at the `shape` stage with the reason. Obstacles are unchanged (a compound is still its bounding box), and a plain object is returned as itself. See `native_mujoco/OBJECT_MODEL.md`.

After it: 111 and 121 pass the crane's shape gates and 221 passes side-on's. Every other refusal is **kept**: the crane refuses every cylinder (it closes on two flat faces), side-on refuses the 25 mm cylinders on height (its pads sit 34 mm up so that the forearm clears the table), and side-on refuses blocks by class (it was measured on cubes and cylinders).

## 3. One flight per design

| Design | Planner | Envelope | Cells flown | Lift | Place | Release drag | Refusal / note |
|---|---|---|---|---|---|---|---|
| PART110 (block, short, solid) | crane | — | not flown (7 pair(s)) | — | — | — | planning: IK found no grasp pose in the bounded start set; IK found no pose for rung +13 |
| PART111 (block, short, holed) | crane | box | not flown (7 pair(s)) | — | — | — | planning: IK found no grasp pose in the bounded start set; IK found no pose for rung +13 |
| PART120 (block, tall, solid) | crane | — | r3c2 → r3c3 | **not lifted** | not placed | — | in flight: refused at the hover -- palm_route: palm route clearance -6.13 mm < 0.494 mm from rung +9 |
| PART121 (block, tall, holed) | crane | box | r3c2 → r3c3 | **not lifted** | not placed | — | in flight: refused at the hover -- palm_route: palm route clearance -6.13 mm < 0.494 mm from rung +9 |
| PART210 (cylinder, short, solid) | side-on | — | not flown (1 pair(s)) | — | — | — | planning: part210_a is 25 mm tall; the pads at 34 mm would not be 4 mm below its top |
| PART211 (cylinder, short, holed) | side-on | cylinder | not flown (1 pair(s)) | — | — | — | planning: part211_a is 25 mm tall; the pads at 34 mm would not be 4 mm below its top |
| PART220 (cylinder, tall, solid) | side-on | — | r2c2 → r1c1 | yes: rose 46.2 mm, slip 0.002 mm; carried drift ≤ 0.43 mm; grip 83 N | on target: 0.7 mm, tilt 0.08° | 0.9 mm | failed clean_placement, clean_withdrawal, returned_to_present: the grip did not let go: 0.5 N still on part220_a with the hand open to -44 deg (the plan's opening), so the hand was not opened further; D6 evaluator: success |
| PART221 (cylinder, tall, holed) | side-on | cylinder | r2c2 → r1c1 | yes: rose 46.9 mm, slip 0.002 mm; carried drift ≤ 0.49 mm; grip 97 N | on target: 0.9 mm, tilt 0.00° | 1.12 mm | all 6 criteria met; back at PRESENT; D6 evaluator: success |

Two runs of all eight from the same code gave identical refusals, flight results and final object positions (the simulation is deterministic, so this checks repeatability, **not** robustness). After those runs the side-on placement's own read of the object was changed to refuse a compound with no envelope (it raised instead). 220 and 221 were flown again: every result is identical; 221 records one more `GRASP_ENVELOPE` event. `results/` holds these last 220/221 files.

### Reading it

- **Only PART221 completed a full pick-and-place.**
- **PART220** was lifted and placed on target, then the withdrawal was refused: with the hand open to the plan's widest opening (−44.5°) the grip read **0.537 N**, above `release_force_n` = 0.5 N. The hand stayed beside the part. PART221 read 0.42 N at let-go, was judged free, backed off to 0 N and completed. **The two readings sit just either side of the threshold (0.04 N above and 0.08 N below), on one flight each: that does not show that the solid part is harder than the holed one.** The threshold is unchanged. The D6 evaluator calls both placements a success; it reports 220's withdrawal as refused.
- **PART120 / PART121** passed planning and preflight, then stopped at the hover: re-checked from the measured joints, the palm's route to the grasp came **6.13 mm into the part**. The hand never reached it and returned to PRESENT. Solid and holed stop identically, so the envelope is not the cause. Planning succeeded at the first pair, so by the protocol no other pair was tried: **whether any other cell works is unknown.**
- **PART110 / PART111**: the crane found no plan on any of the 7 pairs tried (6 × no grasp pose from its bounded IK start set, 1 × no pose for a lift rung). Not all 72 ordered pairs were tried.
- **PART210 / PART211 cannot be picked by either planner**, on any cell. This blocks "every design pickable" for #216 and #214 until a grasp for 25 mm cylinders exists.

### What these runs cannot show

- **The sim's hand geometry is known to be off** (#210: the sim's palm hits pieces at real grasp poses). The PART120/121 palm refusal is exactly that kind of result: it may not happen on the robot, and nothing here says it would not.
- **PART masses are synthetic** (0.150–0.270 kg), and friction is the pool pieces' value.
- The holed cylinders are 16-sided in the sim, so 221's envelope is 50.18 mm across against 220's 49.21 mm, and its planned opening is 2° wider.
- One flight per design, yaw 0, empty board, offline. No live server, SDK or bridge run.
- A flight bug found here: the first version of the `GRASP_ENVELOPE` event used a field named `kind`, which collides with the `on_event(kind, **kw)` callbacks the motion worker and this harness use, and raised `TypeError` in the side-on pick. The unit tests' callback had another parameter name and missed it; it now has the real signature. The crane flights did not hit it because `fly()` does not pass `on_event` to the crane's placement planner.
