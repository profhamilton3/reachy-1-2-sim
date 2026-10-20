# Offline feasibility sweep — placement (2026-10-19; #174, side-on #180)

reachy-1-2-sim at `worktree`.  Preflight only (`crane_place_live.plan_crane_place`): no motion, no simulator.  IK: the host mirror of the simulator's service.  Scene FWDCenterLabSivaPool, the object upright at the origin cell's centre, nothing else on the board.  Reachable cells: the 9 without the `no-right-arm` tag.

## red_cube

Feasible: **56 of 72** ordered pairs.  r1c1→r1c2, r1c1→r1c3, r1c1→r2c1, r1c1→r2c2, r1c1→r2c3, r1c1→r3c2, r1c1→r3c3, r1c2→r1c1, r1c2→r1c3, r1c2→r2c1, r1c2→r2c2, r1c2→r2c3, r1c2→r3c2, r1c2→r3c3, r1c3→r1c1, r1c3→r1c2, r1c3→r2c1, r1c3→r2c2, r1c3→r2c3, r1c3→r3c2, r1c3→r3c3, r2c1→r1c1, r2c1→r1c2, r2c1→r1c3, r2c1→r2c2, r2c1→r2c3, r2c1→r3c2, r2c1→r3c3, r2c2→r1c1, r2c2→r1c2, r2c2→r1c3, r2c2→r2c1, r2c2→r2c3, r2c2→r3c2, r2c2→r3c3, r2c3→r1c1, r2c3→r1c2, r2c3→r1c3, r2c3→r2c1, r2c3→r2c2, r2c3→r3c2, r2c3→r3c3, r3c2→r1c1, r3c2→r1c2, r3c2→r1c3, r3c2→r2c1, r3c2→r2c2, r3c2→r2c3, r3c2→r3c3, r3c3→r1c1, r3c3→r1c2, r3c3→r1c3, r3c3→r2c1, r3c3→r2c2, r3c3→r2c3, r3c3→r3c2

| from → to | feasible | grasp used | margin mm (side-on / crane) | refused at | stage | reason |
|---|---|---|---|---|---|---|
| r1c1 → r1c2 | **yes** | side-on | 1.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c1 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c1 → r2c1 | **yes** | side-on | 1.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c1 → r2c2 | **yes** | side-on | 1.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c1 → r2c3 | **yes** | side-on | 1.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c1 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no grasp pose in the bounded start set |
| r1c1 → r3c2 | **yes** | side-on | 1.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c1 → r3c3 | **yes** | side-on | 1.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → r1c1 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → r2c1 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → r2c2 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → r2c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → r3c2 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → r3c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c3 → r1c1 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +1 |
| r1c3 → r1c2 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +1 |
| r1c3 → r2c1 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +1 |
| r1c3 → r2c2 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +1 |
| r1c3 → r2c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +1 |
| r1c3 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +1 |
| r1c3 → r3c2 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +1 |
| r1c3 → r3c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +1 |
| r2c1 → r1c1 | **yes** | side-on | 0.99 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c1 → r1c2 | **yes** | side-on | 0.99 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c1 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c1 → r2c2 | **yes** | side-on | 0.99 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c1 → r2c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c1 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no grasp pose in the bounded start set |
| r2c1 → r3c2 | **yes** | side-on | 0.99 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c1 → r3c3 | **yes** | side-on | 0.99 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → r1c1 | **yes** | side-on | 1.01 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → r1c2 | **yes** | side-on | 1.01 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → r2c1 | **yes** | side-on | 1.01 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → r2c3 | **yes** | side-on | 0.63 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → r3c2 | **yes** | side-on | 1.01 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → r3c3 | **yes** | side-on | 0.63 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c3 → r1c1 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +10 |
| r2c3 → r1c2 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +10 |
| r2c3 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +10 |
| r2c3 → r2c1 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +10 |
| r2c3 → r2c2 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +10 |
| r2c3 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +10 |
| r2c3 → r3c2 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +10 |
| r2c3 → r3c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +10 |
| r3c1 → r1c1 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set |
| r3c1 → r1c2 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set |
| r3c1 → r1c3 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set |
| r3c1 → r2c1 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set |
| r3c1 → r2c2 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set |
| r3c1 → r2c3 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set |
| r3c1 → r3c2 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set |
| r3c1 → r3c3 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set |
| r3c2 → r1c1 | **yes** | side-on | 1.02 / — |  |  | crane: destination: IK found no pose that sets the object down on the target with the grasp's attitude |
| r3c2 → r1c2 | **yes** | side-on | 1.02 / — |  |  | crane: destination: IK found no pose that sets the object down on the target with the grasp's attitude |
| r3c2 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +2 |
| r3c2 → r2c1 | **yes** | side-on | 0.61 / 0.00 |  |  |  |
| r3c2 → r2c2 | **yes** | side-on | 1.02 / 0.00 |  |  |  |
| r3c2 → r2c3 | **yes** | side-on | 1.02 / — |  |  | crane: destination: IK found no pose that sets the object down on the target with the grasp's attitude |
| r3c2 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: destination: IK found no pose that sets the object down on the target with the grasp's attitude |
| r3c2 → r3c3 | **yes** | side-on | 1.02 / 0.00 |  |  |  |
| r3c3 → r1c1 | **yes** | side-on | 0.97 / — |  |  | crane: straddle: predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r1c2 | **yes** | side-on | 0.97 / — |  |  | crane: straddle: predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: straddle: predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r2c1 | **yes** | side-on | 0.97 / — |  |  | crane: straddle: predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r2c2 | **yes** | side-on | 0.88 / — |  |  | crane: straddle: predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r2c3 | **yes** | side-on | 0.00 / — |  |  | crane: straddle: predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: straddle: predicted grasp pose does not straddle: ['finger_above_top'] |
| r3c3 → r3c2 | **yes** | side-on | 0.97 / — |  |  | crane: straddle: predicted grasp pose does not straddle: ['finger_above_top'] |

Refusals by reason:

- 8 × pick / plan: side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set
- 4 × destination / destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no grasp pose in the bounded start set
- 1 × destination / destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +1
- 1 × destination / destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +10
- 1 × destination / destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: destination: IK found no pose that sets the object down on the target with the grasp's attitude
- 1 × destination / destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: straddle: predicted grasp pose does not straddle: ['finger_above_top']

