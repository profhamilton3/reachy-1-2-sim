# Offline feasibility sweep — placement (2026-10-19; #174, side-on #180)

reachy-1-2-sim at `worktree`.  Preflight only (`crane_place_live.plan_crane_place`): no motion, no simulator.  IK: the host mirror of the simulator's service.  Scene FWDCenterLabSivaPool, the object upright at the origin cell's centre, nothing else on the board.  Reachable cells: the 9 without the `no-right-arm` tag.

## red_cube

Feasible: **3 of 72** ordered pairs.  r3c2→r2c1, r3c2→r2c2, r3c2→r3c3

| from → to | feasible | grasp used | margin mm (side-on / crane) | refused at | stage | reason |
|---|---|---|---|---|---|---|
| r1c1 → r1c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r1c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r2c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r2c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r2c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r3c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r3c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r3c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r1c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r1c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r2c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r2c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r2c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r3c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r3c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r3c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c3 → r1c1 | no |  |  | pick | plan | IK found no pose for rung +1 |
| r1c3 → r1c2 | no |  |  | pick | plan | IK found no pose for rung +1 |
| r1c3 → r2c1 | no |  |  | pick | plan | IK found no pose for rung +1 |
| r1c3 → r2c2 | no |  |  | pick | plan | IK found no pose for rung +1 |
| r1c3 → r2c3 | no |  |  | pick | plan | IK found no pose for rung +1 |
| r1c3 → r3c1 | no |  |  | pick | plan | IK found no pose for rung +1 |
| r1c3 → r3c2 | no |  |  | pick | plan | IK found no pose for rung +1 |
| r1c3 → r3c3 | no |  |  | pick | plan | IK found no pose for rung +1 |
| r2c1 → r1c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r2c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r2c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r3c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r3c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r3c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c3 → r1c1 | no |  |  | pick | plan | IK found no pose for rung +10 |
| r2c3 → r1c2 | no |  |  | pick | plan | IK found no pose for rung +10 |
| r2c3 → r1c3 | no |  |  | pick | plan | IK found no pose for rung +10 |
| r2c3 → r2c1 | no |  |  | pick | plan | IK found no pose for rung +10 |
| r2c3 → r2c2 | no |  |  | pick | plan | IK found no pose for rung +10 |
| r2c3 → r3c1 | no |  |  | pick | plan | IK found no pose for rung +10 |
| r2c3 → r3c2 | no |  |  | pick | plan | IK found no pose for rung +10 |
| r2c3 → r3c3 | no |  |  | pick | plan | IK found no pose for rung +10 |
| r3c1 → r1c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r1c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r1c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r3c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r3c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c2 → r1c1 | no |  |  | destination | destination | IK found no pose that sets the object down on the target with the grasp's attitude |
| r3c2 → r1c2 | no |  |  | destination | destination | IK found no pose that sets the object down on the target with the grasp's attitude |
| r3c2 → r1c3 | no |  |  | destination | plan | IK found no pose for rung +2 |
| r3c2 → r2c1 | **yes** | crane | 0.00 (joints) |  |  |  |
| r3c2 → r2c2 | **yes** | crane | 0.00 (joints) |  |  |  |
| r3c2 → r2c3 | no |  |  | destination | destination | IK found no pose that sets the object down on the target with the grasp's attitude |
| r3c2 → r3c1 | no |  |  | destination | destination | IK found no pose that sets the object down on the target with the grasp's attitude |
| r3c2 → r3c3 | **yes** | crane | 0.00 (joints) |  |  |  |
| r3c3 → r1c1 | no |  |  | pick | straddle | predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r1c2 | no |  |  | pick | straddle | predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r1c3 | no |  |  | pick | straddle | predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r2c1 | no |  |  | pick | straddle | predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r2c2 | no |  |  | pick | straddle | predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r2c3 | no |  |  | pick | straddle | predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r3c1 | no |  |  | pick | straddle | predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r3c2 | no |  |  | pick | straddle | predicted grasp pose does not straddle: ['finger_above_top'] |

Refusals by reason:

- 40 × pick / plan: IK found no grasp pose in the bounded start set
- 8 × pick / plan: IK found no pose for rung +1
- 8 × pick / plan: IK found no pose for rung +10
- 8 × pick / straddle: predicted grasp pose does not straddle: ['finger_above_top']
- 4 × destination / destination: IK found no pose that sets the object down on the target with the grasp's attitude
- 1 × destination / plan: IK found no pose for rung +2

## pool_box_1

Feasible: **3 of 72** ordered pairs.  r3c2→r2c1, r3c2→r2c2, r3c2→r3c3

| from → to | feasible | grasp used | margin mm (side-on / crane) | refused at | stage | reason |
|---|---|---|---|---|---|---|
| r1c1 → r1c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r1c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r2c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r2c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r2c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r3c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r3c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c1 → r3c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r1c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r1c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r2c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r2c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r2c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r3c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r3c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → r3c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c3 → r1c1 | no |  |  | pick | plan | IK found no pose for rung +2 |
| r1c3 → r1c2 | no |  |  | pick | plan | IK found no pose for rung +2 |
| r1c3 → r2c1 | no |  |  | pick | plan | IK found no pose for rung +2 |
| r1c3 → r2c2 | no |  |  | pick | plan | IK found no pose for rung +2 |
| r1c3 → r2c3 | no |  |  | pick | plan | IK found no pose for rung +2 |
| r1c3 → r3c1 | no |  |  | pick | plan | IK found no pose for rung +2 |
| r1c3 → r3c2 | no |  |  | pick | plan | IK found no pose for rung +2 |
| r1c3 → r3c3 | no |  |  | pick | plan | IK found no pose for rung +2 |
| r2c1 → r1c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r1c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r2c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c1 → r3c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r1c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r2c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r2c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r3c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r3c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → r3c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c3 → r1c1 | no |  |  | pick | plan | IK found no pose for rung +11 |
| r2c3 → r1c2 | no |  |  | pick | plan | IK found no pose for rung +11 |
| r2c3 → r1c3 | no |  |  | pick | plan | IK found no pose for rung +11 |
| r2c3 → r2c1 | no |  |  | pick | plan | IK found no pose for rung +11 |
| r2c3 → r2c2 | no |  |  | pick | plan | IK found no pose for rung +11 |
| r2c3 → r3c1 | no |  |  | pick | plan | IK found no pose for rung +11 |
| r2c3 → r3c2 | no |  |  | pick | plan | IK found no pose for rung +11 |
| r2c3 → r3c3 | no |  |  | pick | plan | IK found no pose for rung +11 |
| r3c1 → r1c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r1c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r1c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r2c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r3c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c1 → r3c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c2 → r1c1 | no |  |  | destination | destination | IK found no pose that sets the object down on the target with the grasp's attitude |
| r3c2 → r1c2 | no |  |  | destination | destination | IK found no pose that sets the object down on the target with the grasp's attitude |
| r3c2 → r1c3 | no |  |  | destination | plan | IK found no pose for rung +3 |
| r3c2 → r2c1 | **yes** | crane | 0.00 (joints) |  |  |  |
| r3c2 → r2c2 | **yes** | crane | 0.00 (joints) |  |  |  |
| r3c2 → r2c3 | no |  |  | destination | plan | IK found no pose for rung +12 |
| r3c2 → r3c1 | no |  |  | destination | destination | IK found no pose that sets the object down on the target with the grasp's attitude |
| r3c2 → r3c3 | **yes** | crane | 0.00 (joints) |  |  |  |
| r3c3 → r1c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r1c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r1c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r2c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r2c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r2c3 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r3c1 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c3 → r3c2 | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |

Refusals by reason:

- 48 × pick / plan: IK found no grasp pose in the bounded start set
- 8 × pick / plan: IK found no pose for rung +2
- 8 × pick / plan: IK found no pose for rung +11
- 3 × destination / destination: IK found no pose that sets the object down on the target with the grasp's attitude
- 1 × destination / plan: IK found no pose for rung +3
- 1 × destination / plan: IK found no pose for rung +12

