# Offline feasibility sweep — placement (2026-10-19; #174, side-on #180)

reachy-1-2-sim at `worktree`.  Preflight only (`crane_place_live.plan_crane_place`): no motion, no simulator.  IK: the host mirror of the simulator's service.  Scene FWDCenterLabSivaPool, the object upright at the origin cell's centre, nothing else on the board.  Reachable cells: the 9 without the `no-right-arm` tag.

## pool_box_1

Feasible: **56 of 72** ordered pairs.  r1c1→r1c2, r1c1→r1c3, r1c1→r2c1, r1c1→r2c2, r1c1→r2c3, r1c1→r3c2, r1c1→r3c3, r1c2→r1c1, r1c2→r1c3, r1c2→r2c1, r1c2→r2c2, r1c2→r2c3, r1c2→r3c2, r1c2→r3c3, r1c3→r1c1, r1c3→r1c2, r1c3→r2c1, r1c3→r2c2, r1c3→r2c3, r1c3→r3c2, r1c3→r3c3, r2c1→r1c1, r2c1→r1c2, r2c1→r1c3, r2c1→r2c2, r2c1→r2c3, r2c1→r3c2, r2c1→r3c3, r2c2→r1c1, r2c2→r1c2, r2c2→r1c3, r2c2→r2c1, r2c2→r2c3, r2c2→r3c2, r2c2→r3c3, r2c3→r1c1, r2c3→r1c2, r2c3→r1c3, r2c3→r2c1, r2c3→r2c2, r2c3→r3c2, r2c3→r3c3, r3c2→r1c1, r3c2→r1c2, r3c2→r1c3, r3c2→r2c1, r3c2→r2c2, r3c2→r2c3, r3c2→r3c3, r3c3→r1c1, r3c3→r1c2, r3c3→r1c3, r3c3→r2c1, r3c3→r2c2, r3c3→r2c3, r3c3→r3c2

| from → to | feasible | grasp used | margin mm (side-on / crane) | refused at | stage | reason |
|---|---|---|---|---|---|---|
| r1c1 → r1c2 | **yes** | side-on | 0.97 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c1 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c1 → r2c1 | **yes** | side-on | 0.97 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c1 → r2c2 | **yes** | side-on | 0.97 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c1 → r2c3 | **yes** | side-on | 0.97 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c1 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no grasp pose in the bounded start set |
| r1c1 → r3c2 | **yes** | side-on | 0.97 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c1 → r3c3 | **yes** | side-on | 0.97 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → r1c1 | **yes** | side-on | 0.73 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → r2c1 | **yes** | side-on | 0.73 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → r2c2 | **yes** | side-on | 0.73 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → r2c3 | **yes** | side-on | 0.73 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → r3c2 | **yes** | side-on | 0.73 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c2 → r3c3 | **yes** | side-on | 0.73 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r1c3 → r1c1 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +4 |
| r1c3 → r1c2 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +4 |
| r1c3 → r2c1 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +4 |
| r1c3 → r2c2 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +4 |
| r1c3 → r2c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +4 |
| r1c3 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +4 |
| r1c3 → r3c2 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +4 |
| r1c3 → r3c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +4 |
| r2c1 → r1c1 | **yes** | side-on | 0.91 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c1 → r1c2 | **yes** | side-on | 0.91 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c1 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c1 → r2c2 | **yes** | side-on | 0.91 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c1 → r2c3 | **yes** | side-on | 0.91 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c1 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no grasp pose in the bounded start set |
| r2c1 → r3c2 | **yes** | side-on | 0.91 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c1 → r3c3 | **yes** | side-on | 0.91 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → r1c1 | **yes** | side-on | 1.01 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → r1c2 | **yes** | side-on | 1.01 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → r2c1 | **yes** | side-on | 1.01 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → r2c3 | **yes** | side-on | 1.01 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → r3c2 | **yes** | side-on | 0.95 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c2 → r3c3 | **yes** | side-on | 0.61 / — |  |  | crane: plan: IK found no grasp pose in the bounded start set |
| r2c3 → r1c1 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +12 |
| r2c3 → r1c2 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +12 |
| r2c3 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +12 |
| r2c3 → r2c1 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +12 |
| r2c3 → r2c2 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +12 |
| r2c3 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +12 |
| r2c3 → r3c2 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +12 |
| r2c3 → r3c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: IK found no pose for rung +12 |
| r3c1 → r1c1 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set |
| r3c1 → r1c2 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set |
| r3c1 → r1c3 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set |
| r3c1 → r2c1 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set |
| r3c1 → r2c2 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set |
| r3c1 → r2c3 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set |
| r3c1 → r3c2 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set |
| r3c1 → r3c3 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set |
| r3c2 → r1c1 | **yes** | side-on | 0.98 / — |  |  | crane: table: finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded) |
| r3c2 → r1c2 | **yes** | side-on | 0.98 / — |  |  | crane: table: finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded) |
| r3c2 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: table: finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded) |
| r3c2 → r2c1 | **yes** | side-on | 0.98 / — |  |  | crane: table: finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded) |
| r3c2 → r2c2 | **yes** | side-on | 0.98 / — |  |  | crane: table: finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded) |
| r3c2 → r2c3 | **yes** | side-on | 0.98 / — |  |  | crane: table: finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded) |
| r3c2 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: table: finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded) |
| r3c2 → r3c3 | **yes** | side-on | 0.98 / — |  |  | crane: table: finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded) |
| r3c3 → r1c1 | **yes** | side-on | 0.98 / — |  |  | crane: table: thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded) |
| r3c3 → r1c2 | **yes** | side-on | 0.70 / — |  |  | crane: table: thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded) |
| r3c3 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: table: thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded) |
| r3c3 → r2c1 | **yes** | side-on | 0.98 / — |  |  | crane: table: thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded) |
| r3c3 → r2c2 | **yes** | side-on | 0.98 / — |  |  | crane: table: thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded) |
| r3c3 → r2c3 | **yes** | side-on | 0.96 / — |  |  | crane: table: thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded) |
| r3c3 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: table: thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded) |
| r3c3 → r3c2 | **yes** | side-on | 0.98 / — |  |  | crane: table: thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded) |

Refusals by reason:

- 8 × pick / plan: side-on: plan: IK found no level grasp pose for this approach; crane: plan: IK found no grasp pose in the bounded start set
- 4 × destination / destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no grasp pose in the bounded start set
- 1 × destination / destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +4
- 1 × destination / destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: IK found no pose for rung +12
- 1 × destination / destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: table: finger_pad -4.4 mm into the table on 'lift grasp->+lift' (commanded)
- 1 × destination / destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: table: thumb_pad -1.9 mm into the table on 'descent hover->grasp' (commanded)

