# ADR-0005: The simulator runs MuJoCo's no-slip contact pass by default

- Status: Accepted (simulator contact model; simulator-only, no physical-robot validation)
- Date: 2026-10-07
- Decision owners: IITG Reachy 1.2 simulation project
- Relates to: #55 (**addresses part of #55; does not close it**), ADR-0004

**Scope.** This changes the *simulator's* contact model. It does not validate
the physical Reachy 1.2 gripper, its forces or its friction, and it does not
by itself make a grasp, release or withdrawal reliable.

## Context

In a recorded 2026-10-06 hold, the cube crept downward in the grasp (about
0.75 mm/s) while the hand was still; the 2026-10-07 panel trial measured
1.69 mm of in-hand slip over its 2.1 s hold. MuJoCo's documentation describes the *NoSlip
post-processing pass* as a PGS sweep over the friction dimensions with R = 0
(hard constraints), run after the main solver, which "suppresses the contact
slip that is inherent to soft-constraint models". It also warns that the
pass is an ad-hoc correction that can occasionally destabilise models with
complex multi-contact interactions.

## Evidence (offline, MuJoCo 3.11, 2 ms step, recorded states)

Replay of the recorded hold, cube drop over 2 s by `noslip_iterations`:

| noslip_iterations | cube z change |
|---|---|
| 0 | -2.436 mm (-38.75 mm over 10 s) |
| 1 | -0.063 mm |
| 2 | +0.032 mm |
| 3 .. 50 | +0.0073 mm (identical) |

At 10 for 10 s: +0.019 mm. Step cost in the hold rose from about 0.056 to
0.10 ms; for the idle board (35 contacts), 0.045 to 0.069 ms. The resting
cube-table contact was identical, the idle board showed no movement either
way, and on the documented HOME->PRESENT forearm-rest route the joint paths
differ by at most 0.22 degrees with the same final pose.

These comparisons **strongly support numerical soft-contact creep as the
cause of the hold slip.** They do not show that slip would occur "at any
force", and force ratios alone do not prove the cause.

## Force interpretation (correction)

The earlier "~380 N" figure was wrong: it came from a static replay without
actuator control. The server's `grip_force_n` is the sum of |normal force|
over all contact points of both pads. In the reported hold it was about
250 N summed over both pads (about 125 N per side). Whether that is
physically realistic is **unvalidated**.

## Release and withdrawal (open items)

In an offline open-loop comparison using the same support-stop rule, no-slip
*increased* cube displacement on opening (10.7 mm versus 5.3 mm; the live
trial with no-slip off measured 7.6 mm), and the open-loop raise touched the
cube (about 1.08 N). The magnitude under the live controller was unverified when this was decided
(one live run since: below).
The finger closing edge-on and the reported ~13 mm overlap are findings that
need careful interpretation of geometry and contact state; they are **not**
proof of physical pad penetration. A withdrawal refused by the existing 6 mm
correction cap is a safe outcome and remains unfinished work.

**One live panel trial with no-slip on (2026-10-07, `lift_object`, #155
code):**
- **Hold:** in-hand slip 0.03 mm over the 2 s hold (1.69 mm with no-slip off).
- **Rise:** 48.8 mm, unchanged at the end of the hold.
- **Release:** opening moved the cube 9.1 mm (8.5 mm sideways and a 3.1 mm
  drop as the tilted cube settled flat), against 7.6 mm in the no-slip-off
  trial. The offline replay shows the opening finger dragging the cube while
  it still clamps.
- **Withdrawal:** judged clear without a correction.
- **Return:** to PRESENT.

That is one run: it measures this configuration once and does not establish
a release or withdrawal rate.

## Decision

- `native_mujoco/model/reachy_1_2.xml` sets `noslip_iterations="10"`;
  `noslip_tolerance` stays at the MuJoCo default (1e-6). Nothing else about
  friction, cone, impratio, force limits or geometry changes.
- Override: `--noslip-iterations N` (integer 0..50; 0 restores the prior
  behaviour) on `native_mujoco/server.py` and `native_mujoco/cli/run_episode.py`,
  and `noslip_iterations=` on `simulation_core.load_world` /
  `SimulationCore.from_paths`. `scripts/start_sim.sh` passes
  `REACHY_SIM_NOSLIP_ITERATIONS` through. Helper: `native_mujoco/contact_model.py`.
- Where the effective setting is reported: the server startup log;
  `hello_ack.capabilities.contact_model`; the recorder manifest
  (`contact_model` and `physics_profile_id`); the panel `/capabilities`
  `sim_link.contact_model` (absent/`null` for older servers); `lift_object`
  evidence where that hook is present.
- Identity: `model_sha256` hashes only the model file, so an override is also
  recorded as `physics_profile_id = "noslip_iterations=<N>"` (`"default"`
  when there is no override or it equals the model default).
- Recordings made with noslip 0 and noslip 10 are **different physics
  configurations**. Label and compare them accordingly; older recordings are
  not unusable.

## Activation

Restart the native MuJoCo server (this resets the board to the scene's
initial poses). No container recreation is needed.

## Consequences

- A hold-slip criterion now measures the grasp rather than solver creep, in
  the simulator only.
- The synthetic exit-gate scenario in `tests/unit/test_gripper.py`
  (`TestGraspScenario`: a 20 g cube with friction 2.5 on a support pillar)
  no longer grasps under the default: the closing finger tips the cube and
  wedges it, and the thumb never touches it. With no-slip off it grasps only
  at the tuned friction 2.5 (at 1.0 the cube flips); with no-slip on it grasps
  at the geom default 1.0 and passes every exit-gate assertion unchanged.
  The fixture's friction was tuned to the soft model. How that test is
  configured is an owner decision recorded on the PR.
- Release behaviour under no-slip is the first acceptance item of the next
  delivery, not a reason to switch the pass off.
- Tests: `tests/unit/test_contact_model_noslip.py` (including a recorded-hold
  creep reproduction: default drift <= 0.1 mm, override 0 <= -1 mm).
