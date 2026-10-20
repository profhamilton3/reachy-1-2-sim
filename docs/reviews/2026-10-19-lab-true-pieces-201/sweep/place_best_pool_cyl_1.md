# Offline feasibility sweep — placement (2026-10-19; #174, side-on #180)

reachy-1-2-sim at `worktree`.  Preflight only (`crane_place_live.plan_crane_place`): no motion, no simulator.  IK: the host mirror of the simulator's service.  Scene FWDCenterLabSivaPool, the object upright at the origin cell's centre, nothing else on the board.  Reachable cells: the 9 without the `no-right-arm` tag.

## pool_cyl_1

Feasible: **56 of 72** ordered pairs.  r1c1→r1c2, r1c1→r1c3, r1c1→r2c1, r1c1→r2c2, r1c1→r2c3, r1c1→r3c2, r1c1→r3c3, r1c2→r1c1, r1c2→r1c3, r1c2→r2c1, r1c2→r2c2, r1c2→r2c3, r1c2→r3c2, r1c2→r3c3, r1c3→r1c1, r1c3→r1c2, r1c3→r2c1, r1c3→r2c2, r1c3→r2c3, r1c3→r3c2, r1c3→r3c3, r2c1→r1c1, r2c1→r1c2, r2c1→r1c3, r2c1→r2c2, r2c1→r2c3, r2c1→r3c2, r2c1→r3c3, r2c2→r1c1, r2c2→r1c2, r2c2→r1c3, r2c2→r2c1, r2c2→r2c3, r2c2→r3c2, r2c2→r3c3, r2c3→r1c1, r2c3→r1c2, r2c3→r1c3, r2c3→r2c1, r2c3→r2c2, r2c3→r3c2, r2c3→r3c3, r3c2→r1c1, r3c2→r1c2, r3c2→r1c3, r3c2→r2c1, r3c2→r2c2, r3c2→r2c3, r3c2→r3c3, r3c3→r1c1, r3c3→r1c2, r3c3→r1c3, r3c3→r2c1, r3c3→r2c2, r3c3→r2c3, r3c3→r3c2

| from → to | feasible | grasp used | margin mm (side-on / crane) | refused at | stage | reason |
|---|---|---|---|---|---|---|
| r1c1 → r1c2 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c1 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c1 → r2c1 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c1 → r2c2 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c1 → r2c3 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c1 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c1 → r3c2 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c1 → r3c3 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c2 → r1c1 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c2 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c2 → r2c1 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c2 → r2c2 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c2 → r2c3 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c2 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c2 → r3c2 | **yes** | side-on | 0.92 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c2 → r3c3 | **yes** | side-on | 0.92 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c3 → r1c1 | **yes** | side-on | 0.00 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c3 → r1c2 | **yes** | side-on | 0.00 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c3 → r2c1 | **yes** | side-on | 0.00 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c3 → r2c2 | **yes** | side-on | 0.00 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c3 → r2c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c3 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c3 → r3c2 | **yes** | side-on | 0.00 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r1c3 → r3c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c1 → r1c1 | **yes** | side-on | 0.98 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c1 → r1c2 | **yes** | side-on | 0.98 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c1 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c1 → r2c2 | **yes** | side-on | 0.98 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c1 → r2c3 | **yes** | side-on | 0.98 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c1 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c1 → r3c2 | **yes** | side-on | 0.98 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c1 → r3c3 | **yes** | side-on | 0.98 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c2 → r1c1 | **yes** | side-on | 0.90 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c2 → r1c2 | **yes** | side-on | 0.90 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c2 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c2 → r2c1 | **yes** | side-on | 0.90 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c2 → r2c3 | **yes** | side-on | 0.90 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c2 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c2 → r3c2 | **yes** | side-on | 0.90 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c2 → r3c3 | **yes** | side-on | 0.90 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c3 → r1c1 | **yes** | side-on | 0.69 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c3 → r1c2 | **yes** | side-on | 0.18 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c3 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c3 → r2c1 | **yes** | side-on | 0.69 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c3 → r2c2 | **yes** | side-on | 0.69 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c3 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c3 → r3c2 | **yes** | side-on | 0.69 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r2c3 → r3c3 | **yes** | side-on | 0.69 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c1 → r1c1 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c1 → r1c2 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c1 → r1c3 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c1 → r2c1 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c1 → r2c2 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c1 → r2c3 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c1 → r3c2 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c1 → r3c3 | no |  |  | pick | plan | side-on: plan: IK found no level grasp pose for this approach; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c2 → r1c1 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c2 → r1c2 | **yes** | side-on | 0.95 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c2 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c2 → r2c1 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c2 → r2c2 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c2 → r2c3 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c2 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c2 → r3c3 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c3 → r1c1 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c3 → r1c2 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c3 → r1c3 | **yes** | side-on | 0.00 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c3 → r2c1 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c3 → r2c2 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c3 → r2c3 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c3 → r3c1 | no |  |  | destination | destination | side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |
| r3c3 → r3c2 | **yes** | side-on | 0.96 / — |  |  | crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces |

Refusals by reason:

- 8 × destination / destination: side-on: destination: IK found no level pose that sets the object down on the target; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces
- 8 × pick / plan: side-on: plan: IK found no level grasp pose for this approach; crane: plan: pool_cyl_1 is a cylinder; the crane pick closes on two flat faces

