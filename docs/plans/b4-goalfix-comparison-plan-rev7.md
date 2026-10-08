# B4 simulator comparison: baseline 8c0dad2 vs fixed 67730a1 (plan, revision 7, 2026-09-29)

**Status: plan only. Not authorized.** Nothing for this comparison has been built, started, moved or executed. **D1 is unauthorized.** (Revision 6 status note: the separate seven-run harness validation batch was authorized and executed on 2026-09-28 at P `f97dc59`; see `assessment-2026-09-28-harness-validation-batch.md`. The process-separated C2 diagnostic is authorized for preparation only, not execution.)

**What revision 7 changes (2026-09-29; reporting wording only, owner-adopted):**
- **§7.6 and §8: how a STOP is reported.** A C2 τ-spread violation is reported as a **synchronization STOP**. Exact-match and classifier evidence keep their stated limits. Neither establishes, by itself, a defect caused by PR #141/#143.
- **Enforcement is unchanged:** every STOP condition, C2 at 30 ms, N1, the echo classifier, the segment definition, and which cycles STOP.
- **Source:** owner decision 2026-09-29, from `decisions-2026-09-29-cmp-stage1-first-command-host-wording.md` §3 (item W).
- **Revision 6 is kept unchanged** as `b4-goalfix-comparison-plan-2026-09-24.rev6.md` (sha256 `c00ac01e60375cece6e0adcadc5507bc4ea701a8f367219d5d0a9ae7f3014e13`).

**What revision 6 changes (2026-09-29; pin amendment only, owner-authorized):**
- **Host/analysis pin M → M′.** The owner authorized merging PR #147 at its reviewed head `e432dd65434a51e56fb3944b95eb51d797f8068b`. The merge commit is **M′ = `dbc878cb1429d240b17edb8dc5b04a5cbc459be6`** (parents M `155fc15…` and `e432dd6…`; root tree `8e9fe827ed98aacf917220f33e15725a197f62df`). Every place this plan names M as the host, kernel, `GENERATED_AT_SHA`, analysis package or frozen `main` now names M′ (§0, §3.1–§3.3, §4.1–§4.2, §7, §7.1, §7.5, §10, §11 D9).
- **Why:** M′ adds the C7 repair (constant-gripper branch and gripper-only-waypoint arm branch now apply R-const/R-carry/R-carry′, as N1 does). No definition, tolerance or threshold changed. Review: `review-2026-09-28-pr147-e432dd6-whole-pr.md`; checkpoint: `checkpoint-2026-09-28-pr147-e432dd6-c2-rev2.md` §2; merge log: `~/goalfix-cmp-pr147-merge-2026-09-29/` and `handoff-2026-09-29-pr147-merge-mprime.md`.
- **Allowed M → M′ difference: exactly five paths** (§10). `tools/goalfix_cmp` tree `cd75ed56…` → `9ea1a47f63d351ca050b7699cc5eea68c6f7d720`; `tests` tree `d655f309…` → `905eae327f22c72bebba1b7c8e74752c684cbc73`.
- **Unchanged:** the A/B image source pins (A `8c0dad2d62dde791c8912fa723b8c2b7f190e1e8`, B `67730a1ecf646544825fb60c00129cf46de6d307`); all 16 runtime trees (§10, re-verified equal at M′); `MERGE_TIME_ISO` (`2026-09-16T02:20:18Z`); every definition, threshold, gate and stop rule; the physical procedure.
- **Report-only neck/seed extractor:** stays at PR #146 head P `f97dc59f1c567289b4aea58a8615163421dd3e85`, unmerged, run from a `git archive P` export (§7). Both PR branches are left in place.
- **Line citations** that moved at M′: `cycle.py:1188–1310` → `1190–1312`, `1188–1210` → `1190–1212`; `pathcheck.py:400–413` → `428–441` (content verified identical). Others are unchanged.
- **Not changed and still open:** the §4.2 "first `joint_command` after `turn_on`" reading; host-worktree creation (I-5b); D1.
- **Revision 5 is kept unchanged** as `b4-goalfix-comparison-plan-2026-09-24.rev5.md` (sha256 `f2e72692ba0eb36a13f0a69d3c21ed3c77ac21611a6b73df6ccc26a28e34c5df`).

**What revision 5 changes (2026-09-28; consolidation only, no new definition):**
- **Superseded wording replaced with the approved definitions**, each cited to its ruling (§0):
  - §7.0: the retired "C4 T − 10 ms" timing item is removed (C4′ is value-based); the W-blk settle-gap rule is added to the sim-time list.
  - §7.1: the goal-assigned segment ("found from the goals … (C0 segmentation)") is replaced by the W-blk window (D-1), with its validity conditions; the Q-echo, R-carry′ and control-run scoping are stated.
  - §7.2: report §4's C1/C4 are replaced by N1–N3, C1′ and C4′; the segmentation and hold rules in force are listed.
  - §7.5–§7.6: W1–W4 as amended by D-1(b) are stated; the metrics that still use the goal-assigned boundary at M are named.
- **§10: owner-approved exception** (2026-09-28) for PR #145's two documentation files. Runtime-tree equality stays mandatory, and no other documentation change is allowed.
- **Unchanged:** the 21-joint compliance vector (§4.1), `MERGE_TIME_ISO` (`2026-09-16T02:20:18Z`, `scripts/e1_stage1/plan.py:199–200`), every threshold, the physical procedure, and every stop rule.
- **Revision 4 is kept unchanged** as `b4-goalfix-comparison-plan-2026-09-24.rev4.md` (sha256 `8f981aa4…f167`).

## 0. Definitions in force and precedence (revision 5; pin updated in revision 6)

**Precedence:** where this plan's text and a listed ruling differ, the ruling governs. A later ruling on the same item supersedes an earlier one only where it says so. Nothing outside this table changes.

| item | definition in force | source | authority |
|---|---|---|---|
| C1 → N1–N3 + C1′ | N1 bit-constant joint with S = G; N2 never > 1 float32 ULP beyond G; N3 first setpoint never > 1 ULP behind S; C1′ first setpoint within **0.05°** of S | `rulings-2026-09-25-owner-comparison-definitions.md` (proposal §2.1 wording) | owner |
| C4 → C4′ | last setpoint within **0.01°** of G and not beyond it (N2); replaces "residual at T − 10 ms" | same | owner |
| C2 | 30 ms τ spread outside the 0.05° end bands; **not widened** | same | owner |
| R-over | a setpoint > 1 ULP beyond `wp_k` toward goto k+1 never belongs to goto k | same | owner |
| Blind bands | echoed start ≤ 0.05° ahead of S on a reversing joint; last sample ≤ 0.01° short of G: **reported, never shown as zero** | same | owner |
| R-tie, R-const | goto-boundary membership; constant-joint reference = nominal S exactly | `rulings-2026-09-25-pr144-stage1-coordinator.md` §1 | coordinator (accepted in the merged tooling) |
| R-carry | carries are bit-equal repeats of the preceding command; excluded from goto setpoints; leading-carry length reported | `review-2026-09-25-pr144-stage-a-slice-and-stage-b-authorization.md` §3 | coordinator, flagged to owner |
| **R-carry′** | a repeat is a proven carry only if its chain origin was not `genuine_echo`; otherwise `echo_carry` (reported; not excluded from τ/N1/N3/C1′; not added to `genuine_echo`). **Replaces R-carry item 1 for timing and echo analysis.** | `rulings-2026-09-26-owner-option-c.md` (D-2; `decision-2026-09-26-…-window-proof.md` §6) | owner |
| Q-echo | an exact match within 0.05° of a goto end, lying in [start, goal], is a path coincidence | stage-1 coordinator rulings §3 | coordinator |
| Q-hold / C6 | settle-hold interval located by assignment; every command in it must be a carry; drift = max vs last-k | stage-1 coordinator rulings §2, addendum A6 | coordinator |
| A1 edge windows | lead-in bounded by the `turn_on` command; parked tail from the final waypoint's last assigned command | addendum A1 | coordinator |
| A2 C3 | raw-value monotonicity toward the goal, no near-end exemption | addendum A2 | coordinator |
| A3 C5 | C5 as shipped (every setpoint after the waypoint's first exact match is exact) | addendum A3 | coordinator |
| **W-blk window (D-1)** | the §7.1 segment and the wrist-ball window are located by settle-gap blocks, not goal assignment (§7.1) | `rulings-2026-09-26-owner-option-c.md` (D-1) | owner |
| W1–W4 | `wrist_ball_delta_cm` definition; W2 boundaries and W4 as amended by D-1 (§7.5–§7.6) | owner 2026-09-25 rulings; D-1 | owner |
| Four-endpoint sensitivity | report-only | owner 2026-09-25 rulings (W section) | owner |
| Compliance vector | §4.1 | `rulings-2026-09-28-owner-compliance-vector.md` | owner |
| §10 docs exception | §10 | owner, 2026-09-28 (this revision's authorization) | owner |

**Implementation reference:** all of the above are implemented in `tools/goalfix_cmp` at M′ `dbc878c` (revision 6; M `155fc15` before) (`pathcheck.py:34, 40–41` for C2/C1′/C4′; `window.py:44–55` for W-blk; `segments.py` for R-tie/R-const/R-over/`carry_mask`; `cycle.py:1190–1312` for the segment, W4 and verdict logic). If the plan text and M′'s code were ever to differ, that is a finding to report, not a licence to follow either silently.

**What revision 4 changes (2026-09-28; owner ruling `rulings-2026-09-28-owner-compliance-vector.md`):**
- **Expected compliance vector corrected (§4.1).** It is now:
  - stiff: the right arm's 8 joints, `neck_roll`, `neck_pitch`, `neck_yaw` and both antennas;
  - compliant: the left arm's 8 joints.

  The native state is still compared exactly on all 21 joints. Revision 3 left the neck out, and the P4 gate would then have stopped every cycle in both arms.
- **Where it applies:** P1, P4, P8, the §4 carry-over check, the `prep_<cycle>.json` records and §8.
- **The physical procedure is unchanged.**
- **Removed:** the unsupported "no physical effect" statement about seeding the non-right-arm targets.
- **Added:** separate report-only neck reporting (§4.2). **Its tooling does not exist yet.**
- **Basis:** a code trace at pinned revisions, not a live observation (`review-2026-09-27-comparison-compliance-trace.md`).
- **Still valid:** everything else from revision 3. Revision 3 is kept unchanged as `b4-goalfix-comparison-plan-2026-09-24.rev3.md`.

**What revision 3 changed:**
- **Initial state.** The 1e-6 rad initial-state match is report-only. The existing `stiff-zero` and compliance checks remain mandatory (§5).
- **Echo movement threshold.** The 0.1° target-movement figure is descriptive only (§7.6).
- **Timing.** All comparison timing uses recorded simulation time (§7.0).
- **Baseline without echoes.** A baseline cycle with no echoes is inconclusive, not a failed fix (§7.6).
- **Tooling.** The offline tooling is specified in `assignment-2026-09-24-sonnet-b4-comparison-offline-tooling.md`. The host SHA becomes that tooling's merge commit, **M** (§3, §10).

**What revision 2 changed:**
- identical stiff-zero preparation for every cycle, with the checks recorded (§5);
- carry-over analysis for reconnects and resets (§4);
- gate compatibility for the mixed provenance, with both versions recorded (§3);
- an echo criterion scoped to the affected `PLACE_ROUTE` segment (§7.1);
- an inconclusive baseline separated from a fixed-version failure (§7.6);
- decisions updated to the owner's proposals (§11).

**Inputs read:**
- `analysis-2026-09-23-goal-feedback-verification-report.md` (§4 checks C0–C9, §5 proposal);
- the PR #141 reviews (`6bb3979`, and `aa83903` = MERGE);
- the PR #143 review (`d23e9d8` = MERGE);
- `e1-stage2-b4-execution-plan-2026-09-18.md`;
- the repo at `67730a1`, read only.

| arm | bridge/reset code in the container image | commit |
|---|---|---|
| **A** (baseline) | pre-PR-#141, echoes present | `8c0dad2d62dde791c8912fa723b8c2b7f190e1e8` |
| **B** (fixed) | PR #141 + PR #143 | `67730a1ecf646544825fb60c00129cf46de6d307` |

## 1. Question and limits

**Question:** with echoes removed, does HOVER → REST_SHUT start within about 1° of HOVER? Does the planned-vs-realised Δ collapse to a tracking term?

**Limits:**
- **No margin is inferred.** No guard, threshold, tolerance, route, budget or margin is set, changed or recommended. With n = 6 per arm the results are descriptive.
- **Nothing is said about the physical robot.**
- **No pooling.** Nothing is pooled with Stage 1 or Stage 2 data. Nothing is compared with historical images or records either: image A is rebuilt (§3.2), so it is not byte-identical to the image behind the 2026-09 evidence.

**How the runner behaves under B.** This is inherent to the fix and stated up front.
- **A re-stream pass holds the waypoint.** Every setpoint of a `fly_route`/`converge` pass equals `wp.pose`, so a pass is a constant hold.
- **Each goto starts from the last goal**, not from the streamed present position.
- **Some code comments are stale.** The ones in `rig_motion.py`/`primitives.py` about moving "only while setpoints stream" are wrong under B. They are left unedited.
- **Known limitation L1 is avoided** by a fresh client plus `turn_on` before every post-reset motion (§6).

## 2. The fix is isolated in the diff (verified, read only)

- **Only three product files change** between `8c0dad2` and `67730a1` outside `tests/`: `fake_reachy_server.py`, `mujoco_remote_backend.py` and `reset_watcher.py` (+362/−75).
- **They are exactly what was reviewed:** byte-identical to the reviewed `aa83903`, and `d814621..67730a1` changes only tests.
- **Everything else is the same tree in both arms:** `native_mujoco/`, `scripts/`, `src/`, `scenes/`, `notebooks/`, `web/`, `ros/`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `supervisord.conf` and `entrypoint.sh` have identical tree/blob hashes.

| file | sha256 (first 16 hex digits) A | B |
|---|---|---|
| `fake_reachy_server.py` | `1dc208d79076d913` | `e26c8da577f69f69` |
| `reset_watcher.py` | `12be2adf4c74b91f` | `c256287e43aba86a` |
| `mujoco_remote_backend.py` | `950be6a977cd1535` | `3dc0573209c40072` |

- **All three files are baked into the image** (`Dockerfile:73-76` → `/opt`), not bind-mounted. The arm is the container image.
- **The reset protocol is compatible across arms.** Both `reset_watcher` versions use the same sentinel and ack files.
  - B publishes nothing on a timeout or refusal.
  - A publishes the ack anyway (L6).
  - `reset.sh` verifies every reset from the server's own record (reset count +1, `sim_step` restart, ack = generation), so A's unconditional ack cannot pass a reset that did not happen.

## 3. Provenance: mixed native server and container

### 3.1 The existing gates pass unchanged, and cannot tell the arms apart

| gate (existing, unchanged) | what it checks | value in both arms |
|---|---|---|
| Binding: kernel tree == `GENERATED_AT_SHA` | `$REPO` = the host worktree at **M′** (revision 6), the analysis-pin merge commit | **M′**, clean |
| `check_binding_provenance` | manifest `code_sha`: clean, a descendant of `3d7fc74`, == `GENERATED_AT_SHA`; `started_at` > `MERGE_TIME_ISO` | **M′** (the native server is started from the host worktree). `native_mujoco/` at M′ must be tree-identical to `67730a1` and `8c0dad2`; this is checked by `git rev-parse <sha>:native_mujoco` before the start. |
| `verify_simulator_identity` | scene sha chain; `/status` backend `mujoco-remote` and `frames_stale` false; the live run directory; SDK-vs-run-directory joint agreement | pass (same scene, same server run) |
| `start_variant_gate`, `require_compliance`, `experiment_gate`, linker, tail check | native states and evidence only | pass |

- **None of these gates reads the container's code**, so every gate will report **M′** in A cycles too.
- **This is correct for the host side and the native server**, which really are identical in both arms (M′, whose runner, native and scene trees equal `67730a1`'s). It says nothing about the bridge.
- **The arm is therefore proved only by the comparison-specific checks in §3.3**, which are hard stops. Nothing is added to or changed in the gates or the tooling.

### 3.2 Builds (before the server starts; decision D3)

- **Worktrees.** These are new, detached and clean. The dirty primary checkout is not used.
  - `~/reachy-1-2-sim-cmp-8c0dad2` is used only to build image A.
  - `~/reachy-1-2-sim-cmp-67730a1` is used only to build image B.
  - `~/reachy-1-2-sim-cmp-host` at **M′** runs the native server, compose, the notebooks, the scripts and `tools/goalfix_cmp`.
- **Order.** Build A first with `docker build --platform linux/amd64 -t reachy-1-2-sim:cmp-A-8c0dad2 .`, then B on A's layer cache with the tag `cmp-B-67730a1`.
  - `:latest`, `:demo-*` and `:rollback-*` are never retagged or pruned.
- **Dependency parity.** This is a hard precondition; on failure, stop and consult the owner. All of the following must hold:
  - the same base `ros:foxy` digest;
  - identical layer digests up to and including the pip-install layer from `requirements.txt`, with the first differing layer no earlier than `COPY fake_reachy_server.py`;
  - identical `pip freeze`, `dpkg-query -W` and `python3 --version` output from a throwaway container of each image.
  - This matters because 20 of the 21 lines in `requirements.txt` are unpinned.
- **Image manifest diff.** Run `sha256sum` over every file under `/opt` (excluding `__pycache__`), `/etc/supervisor/conf.d` and `/entrypoint.sh`.
  - The two manifests must differ in **exactly** the three `/opt/*.py` files.
  - Each of those hashes must equal `git show <sha>:<file> | sha256sum`.
- **Recorded:** `control/build_{A,B}.txt` (image ID, layers, base digest, build log) and `control/image_manifest_{A,B}.txt`.

### 3.3 Both versions recorded per cycle (comparison-specific, hard stop)

**Frozen before the server starts:** `control/arm_map.json` maps each rep to its arm, image tag, image ID and the three expected hashes.

**Each cycle, after the container is recreated and before the reset**, `control/versions_<cycle>.json` records:
- `host_native_kernel_sha = M′…` (`dbc878c…`), dirty = false;
- `bridge_arm` and `bridge_sha` (`8c0dad2…` or `67730a1…`);
- the running container's image ID from `docker inspect`;
- the three `/opt` file hashes, read with `docker compose exec`;
- the `supervisorctl status` start times, which must be later than the recreate.

**Any mismatch with `arm_map.json` is a STOP.** The ledger, the per-cycle summaries and `REPORT.md` all carry **both** SHAs: the host/native one and the bridge one.

**Image selection** needs no code change:
- `COMPOSE_FILE=docker-compose.yml:$EV/control/compose.<X>.yml` is exported for `start_sim.sh`, for the recreate, **and for `reset.sh`**, which calls `docker compose exec`.
- Each override file sets only `image:` and `pull_policy: never`.
- The recreate is `docker compose up -d --no-build --force-recreate reachy-sim`.

## 4. Reconnects and resets cannot carry commands or preparation across cycles

**What a native reset restores** (`server.py:338-351`):
- keyframe `qpos`, and zeroed `qvel`/warm-start;
- `step = 0`;
- every joint's goal and effective target, re-synced to the keyframe;
- object poses (seeded, jitter 0);
- interactive controls.

A command pending at the reset is dropped (#116), and nothing is applied in the reset tick.

**What survives a reset:** only the per-joint actuator configuration, which is `compliant`, `speed_limit` and `torque_limit`.
- **Compliance.** It is the same in every cycle.
  - The native server's defaults (antennas and neck stiff; every other joint compliant) are applied once at server start.
  - `turn_on("r_arm")`, in Stage 0 and every leg, makes the right arm stiff.
  - Nothing in this plan changes the neck, antennas or left arm.
  - The resulting expected vector is in §4.1.
- **Speed and torque limits.** The runner, the routes and the generated notebooks never set them (checked by grep), so they stay at the server defaults.

**What survives a container recreate:** nothing on the bridge side. The bridge is a new process with:
- no `_last_target`;
- no `_pending_cmds`;
- no pending reset;
- no in-memory reset generation;
- a fresh container `/tmp`, so no stale sentinel or ack file.

L4 (queues surviving a reconnect) cannot arise across a process restart.

**Why the order is recreate → reset:**
- Any command the outgoing bridge had merged into native's pending slot is applied before the reset and then erased by it.
- The incoming bridge sends no `joint_command` until the cycle's notebook calls `turn_on`. Neither version sends commands on connect, on reset, or on seeding.

**One arm-attributable difference, part of the fix itself, recorded rather than removed.** The first command's 21-joint vector seeds the joints outside the right arm differently:
- A seeds them from the present pose at first build (after settle);
- B seeds them from the first post-reset state.

The neck and both antennas are **stiff** (§4.1), so this seeding difference can move the neck. Whether it does, and by how much, is **unmeasured**.
- Per cycle, record `max |target − keyframe|` over the 13 non-right-arm joints at the first command.
- Report the neck separately (§4.2).

Neither record is evidence that the two arms are equivalent.

**Recorded carry-over checks.** Each is a STOP if it fails.

| check | when | pass |
|---|---|---|
| No SDK client is connected (`lsof -iTCP:50051 -sTCP:ESTABLISHED` shows only the container side), and the previous nbconvert kernel has exited | before the recreate | true |
| The `commands.jsonl` line count and native `cmd_seq` are unchanged from the pre-recreate snapshot to the setup leg's baseline `cmd_seq` | at the setup leg's `turn_on` | no command in between |
| The bridge log from the recreate to the first `turn_on` shows no `joint_command` sent | from `bridge_log_<cycle>.txt` | 0 |
| Every joint's native `compliant` flag equals the §4.1 expected vector (all 21 joints, exact) | pre-recreate, post-reset and post-settle snapshots | equal |

### 4.1 Expected native compliance vector (revision 4)

**Joint order:** the tooling's `_JOINT_ORDER` (`tools/goalfix_cmp/evidence.py:285–288` at M′; unchanged from M), which is `native_mujoco/joint_map.py:49–71` sorted by `mjcf_index`.

| indices | joints | expected native `compliant` |
|---|---|---|
| 0–7 | `r_shoulder_pitch`, `r_shoulder_roll`, `r_arm_yaw`, `r_elbow_pitch`, `r_forearm_yaw`, `r_wrist_pitch`, `r_wrist_roll`, `r_gripper` | `False` (stiff) |
| 8–15 | `l_shoulder_pitch`, `l_shoulder_roll`, `l_arm_yaw`, `l_elbow_pitch`, `l_forearm_yaw`, `l_wrist_pitch`, `l_wrist_roll`, `l_gripper` | `True` (compliant) |
| 16–18 | `neck_roll`, `neck_pitch`, `neck_yaw` | `False` (stiff) |
| 19–20 | `l_antenna`, `r_antenna` | `False` (stiff) |

`expected_compliant21 = [False]*8 + [True]*8 + [False]*5`.

**The same expected vector applies in both arms and at every checkpoint:**
- P1, P4 and P8 (the state after Stage 0's or the leg's `turn_on`);
- the carry-over check above;
- every `prep_<cycle>.json` record.

It is compared with the **native** `compliant` field from the run directory's state stream, not with what the SDK reports. Under A the SDK reports all 21 joints as stiff, whatever the native state.

**Basis (code trace at pinned revisions, not a live observation):**
- The native default stiff set is `native_mujoco/actuator.py:38–41` at M′ `dbc878c` (and M `155fc15`). `native_mujoco` has the same tree (`4f5a8c5`) at A, B, M and M′.
- The set is applied once per server process (`server.py:181–183`). A reset does not change it (`server.py:340–351`), and only `joint_command` compliance fields change it (`server.py:125–147, 369–370`).
- Both bridges forward compliance only for joints an SDK command names. This plan's notebooks only call `turn_on("r_arm")`.
- Full trace: `review-2026-09-27-comparison-compliance-trace.md`.
- If `native_mujoco` at M′ ever changes, this vector must be re-derived before D1.

### 4.2 Neck reporting (revision 4; report-only)

For every cycle, report separately **per arm (A and B)**:
- **Neck targets:** `neck_roll`, `neck_pitch` and `neck_yaw` (indices 16–18) in the first `joint_command` after `turn_on` on each leg. C8 already requires them to stay constant from the second command onward.
- **Actual neck positions:** those three joints' `position_rad` in the native states, at:
  - the P4 first post-reset state;
  - the P5 fixed-`sim_step` state;
  - each leg's first-command bracket `t_hi`;
  - each leg's last state.
- **The differences:** target − keyframe, and actual − keyframe, for each neck joint.

**Limits:**
- The size and relevance of any neck difference between arms are **unmeasured**. No threshold is set.
- These figures never change a verdict, and **reporting them does not show the arms are equivalent**.

**Tooling gap (open before D1):** at M′, `tools/goalfix_cmp` produces none of these figures.
- It does not report neck targets or positions separately.
- The shipped `cycle` CLI calls `pathcheck.run_all` without `seed_keyframe21` (`cycle.py:431–432`). So even the 13-joint `max |target − keyframe|` above (`pathcheck.py:428–441` at M′) is `None` in its output.
- Producing §4.2, and that 13-joint figure, needs either new reviewed report-only tooling or a reviewed offline extraction step. That work is not authorized by this revision.

## 5. Identical stiff-zero preparation for every cycle, including cycle 1

**Stage 0** runs once, before cycle 1, on the first server start with image A. It comprises settle, then a recorded armon (`turn_on("r_arm")` + `require_compliance`, the `INIT` tail check, the gate).
- **Its only lasting effect** is the actuator configuration described in §4, which every later cycle's `turn_on` re-asserts identically.
- **Its pose is erased** by cycle 1's reset, exactly as each earlier cycle's final PRESENT pose is erased by the next reset.
- **Cycle 1 therefore has no separate preparation path.** It goes through the same **P1–P8** as every other cycle, and it includes the container recreate even though Stage 0 already runs image A.

| step | action | recorded check, all cycles |
|---|---|---|
| **P1** | Carry-over checks, pre-recreate part (§4); capture the outgoing bridge log | `prep_<cycle>.json` part 1: the native 21-joint compliance vector (`actual_compliant21`) and the §4.1 `expected_compliant21`, compared exactly (a mismatch is a carry-over STOP), `cmd_seq`, the `commands.jsonl` count, the posture of the 8 r_arm joints |
| **P2** | Recreate the container with `arm_map[r]`; poll `/status` until ready (fixed 60 s cap; elapsed time recorded) | `versions_<cycle>.json` (§3.3); `env_<cycle>.txt` (executor 0/unset, backend, scene) |
| **P3** | `reset.sh … <gen>` | `reset_<gen>.txt`, verified from the server record |
| **P4** | Initial state | **Gate:** the native 21-joint compliance vector (`actual_compliant21`) must equal the §4.1 `expected_compliant21` exactly; both are written to `prep_<cycle>.json` part 2, which is what `cycle.py` reads. **Report only:** the first post-reset state's (the first with `sim_step ≥ state_every`) deviation from cycle 1's across all 21 joints, with 1e-6 rad noted as the expected size. It is never a stop. This is `prep_<cycle>.json` part 2. |
| **P5** | `settle_wait.py $RUN 15 180` | `settle_<cycle>.json`. The **state at a fixed `sim_step`** is also recorded (the first with `sim_step ≥ 5000`, about 10 s at 500 Hz), and its deviation from cycle 1's is reported. |
| **P6** | `parked.sh … <cycle>-parked B4 PLACE_ROUTE <cycle>` | It must classify as **`stiff-zero`**: all gross joints, the gripper and wrist_roll within 1°. The native compliance vector is recorded (the §4 post-settle carry-over check compares it with §4.1). |
| **P7** | Binding cell in a fresh-kernel notebook: provenance, identity, start variant, fresh `require_compliance(compliant=False)` on all 8 joints | `binding_ok_<cycle>` |
| **P8** | Per leg: `check_start` → baseline `cmd_seq` → `turn_on("r_arm")` → `require_compliance(min_cmd_seq)` → route | `<leg>_done`, including the compliance check. This is the fresh client and `turn_on` before any post-reset motion. The 8-joint `require_compliance` gate is unchanged. After `turn_on` the native vector is expected to equal §4.1: the right arm is re-asserted stiff and nothing touches the other 13 joints. |

**Matched in every cycle:**
- the same steps in the same order;
- the same fixed waits: 60 s ready cap, settle quiescence of 15 s, 3 s parked, `LEAD_IN_S` 3 s;
- the same generated notebook template;
- the same interpreter (`e1venv`, reachy-sdk 0.7.0).

## 6. Design

| item | value |
|---|---|
| Board | **B4** `scenes/e1_boards/B4_pool_box_1_r2c3.yaml`. Its sha256 is recorded and must equal `818d8e57…`. |
| Cycle | E1 **shape c**: a recorded **setup** `PLACE_ROUTE` (HOME → … → HOVER → REST_SHUT → REST), then the **flight** `LIFT_TO_PRESENT` (REST → PRESENT). Tooling, waypoints, tolerances and budgets are unchanged. |
| Count | 12 cycles, **6 per arm** (6 setups and 6 flights each) |
| Order | **A B B A · B A A B · A B B A**, frozen in `arm_map.json` |
| Server | One native run for the whole comparison |
| Cycle ids | `S2-B4-c-r1 … r12`: the generator's fixed prefix, with `rep` unbounded; kept in a separate evidence directory |
| Evidence directory | `~/b4-goalfix-cmp-s1-<date>/`, outside every checkout, with session id `cmp1` |

## 7. Predefined analysis

The analysis is the reviewed `tools/goalfix_cmp/` package at **M′** (revision 6; see the assignment). It is hashed into `control/tools/` before the start and never edited during the session. Both arms go through the same code.

**Report-only extraction (§4.2), revision 5 note:** the neck/seed extractor is being prepared as a separate package, `tools/goalfix_cmp_report/`, in its own PR. It does not change `tools/goalfix_cmp/`, whose tree must stay equal to M′'s. **Revision 6:** PR #146 was reviewed (ACCEPTED, not merged); the extractor stays pinned at P `f97dc59f1c567289b4aea58a8615163421dd3e85`, runs from a `git archive P` export, records P in its output and never gates. Merging PR #146 would break the §10 diff-scope check. PR #147 does not touch anything the extractor calls (`evidence.py`, `holds.py`, `_io.py` unchanged; `resolve_leg` and `check_c8` untouched). The host pin is M′.

### 7.0 Timing is recorded simulation time

**Every measured interval uses `sim_time_s` / `sim_step`**, keyed by reset epoch, because both restart at each reset. That covers:
- the 0.5 s echo lookback;
- the 30 ms C2 skew;
- the W-blk settle-gap classification (0.28 s = `settle_s` − one 20 ms merge period, `window.py:44–55`; §7.1);
- the hold windows (`settle_s`, 3 s parked, 3 s `LEAD_IN_S`);
- the fixed-step state comparison;
- the −20 s control shift;
- the echo age.

**Wall or monotonic time (`wall_time_ns`) is used only to locate leg windows**, through the existing linker sidecars, and for operational logs. The W-blk wall-clock partition is report-only (D-1(c)).

**Revision 5:** report §4's "C4 residual at T − 10 ms" is **retired** (owner ruling 2026-09-25). Its replacement C4′ (§7.2) is a value bound (0.01°), so it adds no timing item here.

**Commands carry no timestamp.** Each command's application time is the bracket `[t_lo, t_hi]` between the last state that has not yet reported it and the first state that has (by `cmd_seq`). A command that is never reported, or whose bracket spans a reset, is counted as unplaceable.

### 7.1 Echo criterion on the affected `PLACE_ROUTE` segment

**The segment (revision 5: W-blk window, owner ruling D-1, `rulings-2026-09-26-owner-option-c.md`).** It is located by **settle-gap blocks**, never by goal assignment, and is the same method in both arms:
- The leg's commands are located by the sidecar alignment. The leading run of commands carrying a non-null right-arm `compliant` flag (the SDK `turn_on`, which the bridge may split into 1–2 commands) is removed first.
- For consecutive commands i, i+1, the arrival-gap bounds are `lb = t_lo(i+1) − t_hi(i)` and `ub = t_hi(i+1) − t_lo(i)` from the §7.0 brackets, in **sim time**. A gap is **settle** if `lb ≥ 0.28 s`, **within** if `ub < 0.28 s`, otherwise **ambiguous**.
- **Blocks** are the runs between settle gaps, labelled ordinally GRIP_SHUT … REST.
- **Segment:** from the **last command of the HOVER block** (inclusive) up to, but excluding, the **first command of the REST block**. So it contains the HOVER hold, the HOVER → REST_SHUT goto and its re-stream passes, and the REST_SHUT hold. Every command in it counts, echoes and carries included.
- **Validity:** the window is **indeterminate** if any of these fails:
  - V-a: the leg is located and every command in it is placeable;
  - V-b: exactly 11 blocks;
  - V-c: each block's last command is nearest (L∞ over 8 joints, no tolerance) to its ordinal waypoint;
  - V-d: the realised window is non-empty, in one epoch, with contiguous state `seq`;
  - V-e: no gap is ambiguous.
- **Not validity conditions:** echoes, C0/C1′/C2/C4′ failures and unassigned commands. They are reported, and in B they remain STOP conditions (§7.6).
- **B:** a W-blk-indeterminate B cycle is a **STOP** (D-1(e)).
- **Superseded:** revision 4's "from the last setpoint of the goto to HOVER … up to the REST_SHUT waypoint check … found from the goals in `commands.jsonl` (C0 segmentation)". On clean B the goal-assigned boundaries sit one command late at each end (decision report §3.2).
- **Code at M′:** `window.identify_window` (`window.py`), used by `cycle.py:1190–1212` for `genuine_echo_count`, `segment_echo_carry_count`, the control run and `segment_indeterminate`.

A new right-arm target (≠ the previous target for that joint) is an **exact match** if it is bit-equal to `float32(position_rad)` of a recorded state in the preceding 500 ms. Each exact match is then classified:

| class | rule | echo? |
|---|---|---|
| start coincidence | The first setpoint after `turn_on` equals the goal set at `turn_on`. This is legitimate under C1. It cannot occur inside the segment, but is classified wherever it appears. | no, listed separately |
| path coincidence | It equals the minimum-jerk setpoint from (last commanded goal → `wp.pose`) within 1 float32 ULP, at an implied τ that agrees with the command's other moving joints within 30 ms (C2). | no, listed separately |
| **genuine echo** | any other exact match | **yes** |

**Timing-ambiguous matches.** A command has no timestamp; its application time is known only as the bracket `[t_lo, t_hi]` between two states (§7.0). An exact match whose matching state lies only inside that bracket is classed **timing-ambiguous** and reported separately. It is not counted as an echo.

**Near-end matches (Q-echo, coordinator ruling, stage-1 §3):** an exact match on a joint within 0.05° of either end of its goto is a **path coincidence** if the value lies in [start, goal] of that goto (start = the last commanded target before the goto). The near-end joint is exempt from the τ-agreement comparison, as in C2. A value outside [start, goal] stays a genuine echo. The rule "B: genuine > 0 in the segment → STOP" is unchanged.

**Repeats (R-carry′, owner ruling D-2):** a repeat is bit-equal to the preceding same-epoch command for that joint. It is a proven carry only if its chain's first value was not labelled `genuine_echo`. Otherwise it is an **`echo_carry`**: counted and reported (per leg, and per segment as `segment_echo_carry_count`), never added to `genuine_echo`, so manipulation status is unchanged.

**Control:** the same test is run against states shifted −20 s of simulation time within the same reset epoch, per arm, over the **same W-blk segment**. It is report-only.

Per cycle, the segment metric is the number of genuine echoes and their fraction of new targets. Also reported, as secondary figures: whole-leg counts for both legs, and net shoulder-pitch **target** movement over the HOVER hold.

### 7.2 Commanded-path checks

- **C0–C8** (report §4, **as amended; revision 5**) run on both legs of every cycle, with each route's own guards:
  - **C1** is replaced by **N1–N3 + C1′** (first setpoint within 0.05° of the reported start S, never > 1 ULP behind it). S is the previous waypoint's `route_rad` pose, or for a route's first goto the `turn_on` target `float32(present)`. Report §4's 1e-6 rad is withdrawn.
  - **C2**: 30 ms τ spread outside the 0.05° end bands, computed only over joints whose value is a goto setpoint (R-carry point 5, with R-carry′ withdrawals). Not widened.
  - **C3**: raw-value monotonicity toward the goal on every moving joint, with no near-end exemption (A2).
  - **C4** is replaced by **C4′** (last setpoint within 0.01° of G, not beyond it). Report §4's "residual at T − 10 ms" is withdrawn.
  - **C5** as shipped (A3). **C6** uses the Q-hold interval rule. **C7**, **C8** as shipped.
  - **Segmentation** under R-tie, R-const (nominal-S reference), R-over and R-carry/R-carry′. **Edge windows** under A1.
  - The **blind bands** (§0) are reported for every cycle, never shown as zero.
- **B must pass.** A is reported only; C1′, C2 and C5 are expected to fail.
- **Also reported per waypoint:** re-stream pass counts, arrival errors, and the `r_wrist_pitch` shortfall (descriptive).

### 7.3 Clearance decomposition

This is per link and per leg, with the HOVER → REST_SHUT segment reported separately. It uses the 2026-09-23 FK and hand models and the corrected planned-aperture rule (PR #136).

| term | definition |
|---|---|
| **Δcmd** = planned − commanded | the command term |
| **Δtrk** = commanded − realised | the tracking term |

Each is reported as min / median / max per arm, with per-cycle values and no pooling across arms.

### 7.4 Hold drift

**Windows:** every `settle_s` hold (the HOVER hold first), the 3 s parked window, and the 3 s `LEAD_IN_S`.

**Reported per window:** target drift (must be 0 in B by C6), realised drift per joint, and the static offset.

### 7.5 Pre-stated outcomes for the segment (unchanged from report §5; evaluated only on valid cycles, §7.6)

**Supports echo → offset:**
- every valid B cycle has a leg-start shoulder-pitch error ≤ about 1.0°;
- B's Δcmd is ≤ 0.05 cm on every link;
- **(W1, owner ruling 2026-09-25)** **every** valid B cycle has `wrist_ball_delta_cm ≤ median_A − 0.5 cm`, where `median_A` is over the valid, manipulated A cycles (§7.6). The B median is also reported.

**`wrist_ball_delta_cm` (owner rulings W1–W3, proposal §4.1, window per D-1):**
- `planned_min − realised_min` for B4 `pool_box_1`, shells hand, `wrist_ball` capsule; positive = closer than planned. `Δcmd_wb` and `Δtrk_wb` are also reported.
- **Window (W2 as amended by D-1(a)):** realised states in [`t_lo` of the W-blk segment's first command, `t_lo` of the REST block's first command), same epoch; commanded samples = every command in the W-blk segment.
- **Realised source (W3):** checksum-verified server `states.jsonl`.
- **Report-only:** the four-endpoint (`t_lo`/`t_hi`) sensitivity of the minimum and Δ.

**Metrics still on the goal-assigned boundary at M′ (recorded as implemented; not changed here).** The 2026-09-26 ruling deferred "moving other `affected`-based metrics onto the window". So at M′, the leg-start shoulder-pitch error, the post-arrival rise and the segment Δcmd maximum still use `seg.find_affected_segment` (`cycle.py:681–704`). That segment can differ from the W-blk segment by one command at each end on clean B, and can be indeterminate on A. It no longer drives validity or the wrist-ball window. Moving them would be a definition change: owner decision, not made here.

**Refutes it:**
- B shows a leg-start error ≥ 2°, or a post-arrival rise ≥ 1.5°, while its targets are constant (C5 and C6 pass).

**Otherwise** the result is inconclusive.

### 7.6 Inconclusive baseline vs fixed-version path failure (kept separate)

| class | definition | effect |
|---|---|---|
| **Fixed-version path failure** | In a B cycle: any genuine echo in the W-blk segment, **or** any C0–C8 failure (as amended, §7.2) on either leg, **or** non-zero target drift in any hold, **or** a lead-in/edge-window violation, **or** an unassigned non-carry command, **or** a W-blk-indeterminate window (D-1(e)) | **STOP** (unchanged), reported by its cause, not as a comparison outcome. §7.5 is not evaluated. (Revision 7) A C2 τ-spread violation (> 30 ms) is a **synchronization STOP**: the server-visible commanded point was not on the ideal joint-space line within 30 ms. It is not, by itself, evidence of a defect caused by PR #141/#143, since the same mixing is possible in either bridge. An exact float32 match to a recorded state is echo-consistent evidence, and a `genuine_echo` label is a classifier result. Neither is causal proof. Attribution to PR #141/#143 requires evidence that shows the mechanism, which this plan does not collect. |
| **Inconclusive baseline cycle (W4 as amended by D-1(b))** | An A cycle whose **W-blk window is indeterminate** (V-a…V-e), or whose W-blk segment has **0 genuine echoes**. C0 violations and unassigned or non-carry runs **no longer** invalidate an A baseline; they are still reported. The manipulation is absent or unmeasurable, which is **not** evidence about the fix. | **Not a stop, and not a failed fix.** The cycle is labelled, excluded from A's median in §7.5, and still reported in full with its reason and counts. Independent evidence and provenance gates stay mandatory on A. |
| *(descriptive only)* | The net shoulder-pitch **target** movement over the HOVER hold, reported for every cycle with 0.1° as a reference value | It never changes a verdict. |
| **Inconclusive comparison** | Fewer than 4 of 6 A cycles are valid and manipulated | §7.5's relative criterion (W1) is reported as inconclusive. B's absolute criteria are still reported. |

**Selection bias (W4, stated in every report):** excluded A cycles are not a random sample. The exclusion reason is "excluded for window-identification failure (V-a…V-e)" or "0 genuine echoes". The direction of any bias in `median_A` is not established.

At the checkpoint, if both A cycles so far are inconclusive, hold the server idle and ask the owner. This is an inconclusive baseline, not a failure of the fix. A B cycle is never re-labelled as inconclusive.

### 7.7 Reset tripwires (per arm)

| counter | B expected | A expected |
|---|---|---|
| "Reset ack timed out" (bridge) or "reset ack timed out" (watcher) | 0 | 0 |
| "Unexpected reset_ack" | 0 | often non-zero (L6, pre-existing; not a stop) |
| `reset.sh` ack mismatch or other STOP | 0 | 0 |
| native `control_held` refusals, lease acquisitions, `pause` messages | 0 | 0 |

## 8. Stop rules

Any of these ends **the entire comparison**:

**Carried from the Stage 2 plan §5:**
- a non-zero exit;
- `control/stop`;
- `binding_FAIL_*`;
- `check_start` false;
- a `RouteError` or any exception;
- a gate rejection;
- a tail check of `NO`;
- a D2/D3/D4 miss of more than 20°.

A D1 wrist droop counts as a completed leg, with its residual recorded.

**Any contact with a board object, or board displacement > 1 mm, in either arm, ends the entire comparison.** It cannot be resumed without a new authorization.

**Specific to this comparison:**
- a failed version or hash check (§3.3);
- a failed carry-over check (§4);
- a failed `stiff-zero` classification (P6, the existing gate) or a compliance-vector mismatch against the §4.1 expected vector (P4, or the §4 carry-over check). The 1e-6 rad initial-state deviation is report-only and never a stop;
- any reset timeout in either arm (still counted);
- a fixed-version path failure (§7.6), reported by cause per §7.6 (revision 7). The STOP itself is unchanged.

**After a stop:**
- nothing is re-flown, redone, retried or recovered, and there is no `turn_off`;
- the state is recorded, then the server is shut down;
- the arms stay unbalanced and are reported as incomplete.

## 9. Runtime and storage limits

| block | estimate / cap |
|---|---|
| Worktrees, builds, parity and manifest checks (before server start) | A 20–60 min, B a few minutes. **Cap: 90 min.** |
| Server start and Stage 0 | about 6 min |
| One cycle, P1–P8 plus both legs plus between-cycle checks (about 30 s) | about 8–9 min |
| **Server uptime** | about 2–2.3 h. **Hard cap 3 h.** |
| Offline analysis | about 30 min |

**Storage:**
- about 2 GB raw (about 16 MB/min of states), about 0.5 GB as `.tar.zst`;
- about 3 GB for the new image tags;
- **at least 15 GB must be free.** On 2026-09-24, 101 GiB was free.

**Archive:** a new file, `~/e1-evidence-archive/b4-goalfix-cmp-s1-<date>.tar.zst` plus its `.sha256`. Nothing existing is overwritten.

## 10. Preservation

These are never touched: `~/reachy-1-2-sim-stage2` (pinned **4ad0be7**), `~/e1-stage2-*`, every archive in `~/e1-evidence-archive/`, `docs/reviews/probes-*`, the primary checkout, the existing Docker tags and `~/e1venv` (read-only use).

- **Stage 2 evidence stays unread** until this comparison has been reviewed.
- **`main` is frozen at M′**, the merge commit of the reviewed C7 repair (PR #147) on top of the tooling merge M, from the builds until the end of the session. M′ = `dbc878cb1429d240b17edb8dc5b04a5cbc459be6` (revision 6; M = `155fc1549812146a8579080c72560c2c1875262c`). Required at M′: root tree `8e9fe827ed98aacf917220f33e15725a197f62df`; parents M and `e432dd65434a51e56fb3944b95eb51d797f8068b`; `tools/goalfix_cmp` tree `9ea1a47f63d351ca050b7699cc5eea68c6f7d720`; `tests` tree `905eae327f22c72bebba1b7c8e74752c684cbc73`; `git diff --name-only M M′` exactly `tools/goalfix_cmp/pathcheck.py`, `tools/goalfix_cmp/cycle.py`, `tests/unit/test_goalfix_cmp_c7_r_carry.py`, `tests/fixtures/goalfix_cmp/c7_run04_r3_setup.json`, `tests/fixtures/goalfix_cmp/extract_c7_run04_r3.py`. Verified 2026-09-29 by enforced assertions.
- **Diff-scope check before the start (revision 5).** `git diff --name-only 67730a1 M′` may list only:
  - paths under `tools/goalfix_cmp/`;
  - paths under `tests/`;
  - **exactly these two documentation files**, under the owner-approved exception of 2026-09-28 (PR #145, merge `fecc33f`): `docs/ROADMAP.md` and `docs/roadmap/continuous-whole-arm-protection.md`.

  Any other path, **including any other documentation file**, is a STOP.
- **Runtime-tree equality stays mandatory**, whatever the diff lists: `git rev-parse 67730a1:<p>` must equal `git rev-parse M′:<p>` for `native_mujoco`, `scripts`, `src`, `scenes`, `notebooks`, `web`, `ros`, `fake_reachy_server.py`, `mujoco_remote_backend.py`, `reset_watcher.py`, `kinematic_backend.py`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `supervisord.conf` and `entrypoint.sh`. Checked 2026-09-28 at M and 2026-09-29 at M′: all equal (`native_mujoco` `4f5a8c5`, bridge blobs `614663d`/`8577818`/`a3dcd2e`; full hashes in `checkpoint-2026-09-28-c7-c2-pin-readiness.md` §2 item 4).
- **The images are still built from `8c0dad2` and `67730a1`.** `tools/` is not copied into either image.
- **The comparison worktrees and image tags are kept** until the report is reviewed.

## 11. Decisions (the owner's proposals, adopted)

| # | decision |
|---|---|
| D1 | Authorize simulator time for this plan. **Still required.** |
| D2 | Each cycle is a `PLACE_ROUTE` recorded setup plus a `LIFT_TO_PRESENT` flight (shape c). |
| D3 | The tooling is unchanged. The arm is proved by §3.3 outside the gates. |
| D4 | Builds are cached, with dependency parity verified (§3.2) as a hard precondition. |
| D5 | Documented checkpoint after cycles 1–4. The operator records the §7.6 manipulation status, the §4/§5 checks, the tripwires and the gate results verbatim in `control/checkpoint_1.txt`. If all pass, continue in-session; otherwise hold or stop per §7.6/§8. |
| D6 | Any contact stops the entire comparison. |
| D7 | B2 is excluded. |
| D8 | **Approved (narrow), 2026-09-24.** Option (b): one read-only run of the new classifier over the three sessions already analysed on 2026-09-23 (B4 s1, B1 s1, B2 s2), limited to `commands.jsonl`/`states.jsonl` checked against their `SHA256SUMS`, no other file read. It must reproduce the published echo counts (85 372 / 83 750 / 87 294) under the original `verify_goal_feedback` definition; any difference introduced by the new coincidence/timing-ambiguous classifications is reported separately, never used to tune the classifier into agreement. This authorizes reading those two files from those three sessions only, for tooling validation — it does not touch `~/reachy-1-2-sim-stage2` (still pinned 4ad0be7, still unopened) or any other Stage 2 archive, and it does not authorize D1 (simulator time) or comparison execution. |
| D9 | **Prerequisite.** The tooling PR must be reviewed and merged (to M) before D1 can be granted. **Met;** the analysis pin is now M′ (revision 6) after the reviewed C7 repair merge. |

**Not in scope:** physical-robot motion or probes; comment edits; any guard, threshold, tolerance, route, budget or margin change; other boards; re-reading Stage 2 evidence beyond the narrow D8 grant above (B4 s1/B1 s1/B2 s2, `commands.jsonl`+`states.jsonl` only).
