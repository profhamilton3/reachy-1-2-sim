# Offline feasibility sweep — placement (2026-10-19; #174, side-on #180)

reachy-1-2-sim at `worktree`.  Preflight only (`crane_place_live.plan_crane_place`): no motion, no simulator.  IK: the host mirror of the simulator's service.  Scene FWDCenterLabSivaPool, the object upright at the origin cell's centre, nothing else on the board.  Reachable cells: the 9 without the `no-right-arm` tag.

## red_cube

Feasible: **1 of 9** ordered pairs.  r3c2→

| from → to | feasible | grasp used | margin mm (side-on / crane) | refused at | stage | reason |
|---|---|---|---|---|---|---|
| r1c1 → (lift) | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → (lift) | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c3 → (lift) | no |  |  | pick | plan | IK found no pose for rung +1 |
| r2c1 → (lift) | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → (lift) | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c3 → (lift) | no |  |  | pick | plan | IK found no pose for rung +10 |
| r3c1 → (lift) | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c2 → (lift) | **yes** | crane | 0.00 (joints) |  |  |  |
| r3c3 → (lift) | no |  |  | pick | straddle | predicted grasp pose does not straddle: ['finger_above_top'] |

Refusals by reason:

- 5 × pick / plan: IK found no grasp pose in the bounded start set
- 1 × pick / plan: IK found no pose for rung +1
- 1 × pick / plan: IK found no pose for rung +10
- 1 × pick / straddle: predicted grasp pose does not straddle: ['finger_above_top']

## pool_box_1

Feasible: **1 of 9** ordered pairs.  r3c2→

| from → to | feasible | grasp used | margin mm (side-on / crane) | refused at | stage | reason |
|---|---|---|---|---|---|---|
| r1c1 → (lift) | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c2 → (lift) | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r1c3 → (lift) | no |  |  | pick | plan | IK found no pose for rung +2 |
| r2c1 → (lift) | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c2 → (lift) | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r2c3 → (lift) | no |  |  | pick | plan | IK found no pose for rung +11 |
| r3c1 → (lift) | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |
| r3c2 → (lift) | **yes** | crane | 0.00 (joints) |  |  |  |
| r3c3 → (lift) | no |  |  | pick | plan | IK found no grasp pose in the bounded start set |

Refusals by reason:

- 6 × pick / plan: IK found no grasp pose in the bounded start set
- 1 × pick / plan: IK found no pose for rung +2
- 1 × pick / plan: IK found no pose for rung +11

