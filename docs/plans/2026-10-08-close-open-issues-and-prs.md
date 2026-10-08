# Plan: close out open PRs and issues (2026-10-08)

Status: **DRAFT for owner review. No code has been written under this plan.**
Source: owner proposal "Proposals to Close Issues and PRs 2026-10-08" + `CLAUDE.md`.
Verified against GitHub on 2026-10-08: 6 open PRs (#132, #133, #146, #148, #149, #150) and 7 open issues (#2, #28, #74, #107, #115, #127, #130). `main` was at `fb67972`.

---

## 1. How the work runs

### Roles

| Role | Who | Does |
|---|---|---|
| **Owner (human in the loop)** | profhamilton3 | Makes every decision marked **HITL**. Approves merges. Runs the hardware/Mac/Docker steps that agents cannot run. |
| **Lead** | Claude Code session, Opus | Holds this plan. Sends each work item to one subagent, reviews what comes back, opens PRs, asks the owner at the gates. |
| **Opus subagent** | `Plan` or `general-purpose`, model `opus` | Root-cause and design work: Phase 1 diagnosis, #74 clearance design, #28 scope, reviewing Sonnet diffs that touch safety gates. |
| **Sonnet subagent** | `general-purpose`, model `sonnet` | Small, well-scoped fixes with a clear test: #130, #107, #127, #115, doc edits, notebook migration (only after the owner allows it). |

### Rules for keeping token use low

1. One subagent per work item. Each prompt names the issue number, the exact files, the done criteria from this plan, and "do not search outside these paths unless blocked".
2. Use `Explore` (read-only) only when a file list is not already known. Otherwise go straight to implementation.
3. Continue an existing agent with `SendMessage` for follow-ups instead of starting a new one.
4. Agents return a short report (files changed, commands run, results, open questions). They do not paste whole files.
5. The Lead does not re-read files the agent already summarised unless reviewing the diff.
6. Batch the GitHub housekeeping calls (labels, draft state, comments) for Phase 0 into a single pass.

### Rules every agent follows (from `CLAUDE.md`)

- Run `git status --short` first. Never touch `notebooks/test_motion.ipynb` or `notebooks/tlh_motion-routine.ipynb` unless the owner has said yes at gate **H-2c**.
- Write a test first where practical. Run the targeted tests, then the full suite (`python3 -m py_compile fake_reachy_server.py joint_state_bridge.py` plus `pytest`).
- One issue per PR. Put `Closes #N` in the PR body. Use the PR sections required by `CLAUDE.md`: problem, design, compatibility, tests, environments, limitations.
- Never claim a Docker, Mac, or hardware result that was not run in that environment. Those steps go to the owner.
- Do not loosen any safety guard to get a test to pass. In particular, `FEEDBACK_STALE_S` stays at 0.5 s.

---

## 2. Decision gates (HITL)

The Lead stops and asks the owner at each gate below. Each gate has a recommended default so the owner can answer quickly.

| Gate | Question | Recommended default |
|---|---|---|
| **H-0a** | The next B4 attempt needs a new host pin, wrapper bundle and reservation, and an amendment to B4 plan rev 7 §10. Merge #132, #133 and #149 before that new pin is set? | **Yes.** Merge all three, then set the new pin once (see the note below this table). |
| **H-0b** | #133 stopped at reset 18 (17/18 cycles). Merge it as a partial-session record? | Merge, with "partial: 17/18" kept in the title and the stop reason in the evidence README. |
| **H-0c** | #146 is marked *ready for review*, but the proposal says to keep it as a prep branch. Convert it to draft? | Convert to draft, add a `prep-only` label. |
| **H-0d** | #150: change the host pin (a004 STOP repair)? | Owner decision. The agent writes a one-page summary of the trade-offs and does nothing else. |
| **H-1** | Phase 1 root cause found. Approve the proposed fix before it is implemented? | Required. |
| **H-2a** | #74: which clearance model is authoritative at `rig_rail_outer_right` (planner model or MuJoCo physics), and is a torso-clearance term in scope? | Physics is ground truth. The model is made more conservative to match it. Torso term in scope. |
| **H-2b** | #107: the smoke test must stay "non-destructive". Is it acceptable for it to fail when `gate_check()` refuses a move? | Yes. A refused gate makes the smoke test fail loudly. |
| **H-2c** | #2: may agents edit `notebooks/test_motion.ipynb` (protected user work)? | Owner decides. If no, migrate scripts only, and the owner migrates the notebook. |
| **H-3** | #127 and #115 list several items (M1–M3, items 1–7). Fix them all, or close some as won't-fix? | Fix everything that has a reproducer. Close the others with a reason. |
| **H-4** | #28: amend `EPIC-8.md` to CLI/exporter scope, or build the browser research UI? | Amend the scope now. Open a new issue for the UI later. |
| **H-5** | Before each merge: does the PR pass review, and is CI green? | Owner approves every merge. |

**Note on H-0a (verified 2026-10-08).**
- **What the B4 preflight checks.** The B4 comparison wrapper (bundle r6) requires `main == origin/main == M′` (`dbc878c`). B4 plan rev 7 §10 also requires the host pin to differ from `67730a1` only in `tools/goalfix_cmp/`, `tests/` and two docs, and requires `scripts/` to be tree-equal to `67730a1`.
- **Where this is recorded.** The rules are quoted in the bodies of PR #149 and PR #150. The pin `M_PRIME_SHA` is in `tests/fixtures/goalfix_cmp/c2ps_common.py` on the #148 branch. The rev 7 plan itself is not in this repository.
- **The attempt #149 was waiting for has already happened.** That was a004, on 2026-10-02. It ended in a STOP, and its grant is used up.
- **`main` already fails both checks.** `main` is at `fb67972`, 43 commits past M′, after #151–#158 were merged. Since `67730a1`, those merges also changed `scripts/`, `native_mujoco/` and `scenes/`.
- **Conclusion.** Holding #132, #133 or #149 no longer protects any B4 attempt.
- **#133.** Its stop was a harness bug: `reset.sh` ran `tail -1` on a file that was still being written. That fix goes with #127/#115.

---

## 3. Phases

### Phase 0: PR queue housekeeping
**Agent:** Lead only. No subagent needed.

| PR | Action | Gate | Done when |
|---|---|---|---|
| #132 E1 Stage 2 B4, 18/18 | Review the evidence manifest (hashes, seeds, raw output present). Merge. | H-0a, H-5 | Merged |
| #133 E1 Stage 2 B1, 17/18 | Same review. Merge as partial. | H-0a, H-0b, H-5 | Merged |
| #150 a004 STOP repair | Leave as draft. Post the trade-off summary for the owner. | H-0d | Owner has decided. Then it is carried through Phase 1 (it overlaps #130). |
| #149 historical toolkit docs | Lift the hold: a004 has run, and `main` is already past M′. Merge it, then verify its 14 file hashes on `main`. | H-0a, H-5 | Merged and verified |
| #146, #148 goalfix C2 | Keep unmerged. Mark #146 as draft. | H-0c | Both draft, labelled `prep-only` |

**Owner action:** after the merges, approve the new host pin and the rev 7 §10 amendment (combine this with the #150 decision, H-0d). Then reserve the next B4 attempt.

### Phase 1: Observation delivery (top technical priority)
**Problem:** in the 2026-10-08 measurement, attempt 5, an observation reached the motion worker 0.67 s old. The live-feedback guard (`FEEDBACK_STALE_S` = 0.5 s) correctly halted the crane lift. The details are in `docs/roadmap/continuous-whole-arm-protection.md`. The root cause has not been investigated yet.

| Step | Agent | Output |
|---|---|---|
| 1.1 Trace the panel → motion-worker path. Find every queue, poll interval, lock, and serialization step, and estimate the worst-case latency of each. Find where an observation can be read stale (unbounded queue, poll period, blocked worker, lock held during I/O). | Opus (read-only) | A short root-cause note with ranked hypotheses and the evidence for each |
| 1.2 Write a deterministic reproducer: a fixture backend plus an injected delay that recreates an observation older than 0.5 s. | Sonnet | Failing test |
| **H-1** Owner approves the fix design. | Owner | — |
| 1.3 Implement the fix: bounded queue where the latest frame wins, a monotonic-clock age stamp on every observation, and an explicit stale status. The guard stays at 0.5 s. | Sonnet, reviewed by Opus | PR `core:` |
| 1.4 **#130**: `wall_time_ns` validation rejects `bool`, NaN, inf, and negative values (same class of bug as #119 H3). If #150 is going ahead, coordinate with it, because both change `e1_identity`. | Sonnet | PR with `Closes #130` and unit tests |
| 1.5 Owner re-runs the five-attempt measurement on the real setup. | Owner | Evidence that 5/5 attempts finish with no stale-guard halts |

**Phase exit:** the reproducer passes, #130 is closed, and the owner's re-measurement is recorded.

### Phase 2: Destination transfer ("put cube at R2C1")

| Item | Agent | Scope | Gate | Done when |
|---|---|---|---|---|
| **#107** | Sonnet | Add a helper to `primitives.py` (for example `nudge_joint`) that calls `gate_check()`. Route `scripts/smoke_test_host.py` through it. Add a test that a refused gate fails the smoke test. | H-2b | `Closes #107`, and the smoke test still exits 0 against a healthy sim |
| **#74** | Opus designs, Sonnet implements | Reconcile the planner model with physics on clearance at `rig_rail_outer_right` (`SWING_1`). Add torso-clearance modelling. Add regression tests for the rejected rig routes in `FWDCenterLabSivaPool`. | H-2a | `Closes #74`, and the routes are either accepted with a stated margin or rejected for a documented reason |
| **#2** | Sonnet | List every raw `goal_position` write in scripts and notebooks. Migrate scripts to `SceneModel`. Add table/relocation tests. Notebook edits only if the owner allows them. | H-2c | `Closes #2`, with the audit table in the PR |
| Placement slice | Opus designs, Sonnet implements | Geometry-based placement to cell R2C1, using the gated primitives and the #74 clearance model | H-1-style design approval | A placement test passes in `fixture`/`kinematic` mode. The owner runs MuJoCo on the Mac. |

### Phase 3: General withdrawal and recovery

| Item | Agent | Scope | Gate | Done when |
|---|---|---|---|---|
| Recovery planner | Opus designs, Sonnet implements | Plan a checked exit from the *measured* state, with no hard-coded waypoints and no sim reset | Design approval | Tests pass from at least 3 seeded start states |
| **#127** | Sonnet | Stage 1 leg-chain residuals M1–M3: checked archive copies and the tail-check pointer. Do this during the recovery-script refactor. | H-3 | `Closes #127` |
| **#115** | Sonnet | Items 1–7: gate test pinning, `scene_path` refusal, the shell-test runner, torn-flag visibility | H-3 | `Closes #115`. Each item is fixed or closed with a reason. |

### Phase 4: Experience-based planning and Epic 8

| Item | Agent | Scope | Gate | Done when |
|---|---|---|---|---|
| **#28** | Sonnet (if the scope is amended) or Opus (if a UI is built) | Default: amend `EPIC-8.md` to the CLI/exporter scope that was actually delivered, and open a follow-up issue for the UI | H-4 | `Closes #28` |
| Plan memory | Opus designs | Record successful plans with scene and model hashes and seeds, then reuse them as starting points. Write it up as a design doc first. | Design approval | ADR merged. Implementation is tracked in a new issue. |

---

## 4. Order and dependencies

```
Phase 0 (merge #132/#133/#149) ─┬─> new pin + §10 amendment (with #150) ─> next B4 attempt
         └─> Phase 1 (1.1 → 1.2 → H-1 → 1.3; #130 in parallel) ─> owner re-measure
                 └─> Phase 2 (#107 first → #74 → #2 → placement)
                         └─> Phase 3 (recovery → #127, #115)
                                 └─> Phase 4 (#28, plan memory)
```

Items that can run in parallel: #130 alongside 1.1–1.3; #107 and #2 (the scripts part) alongside #74's design; #28 at any time after H-4.

## 5. Tracking

The Lead keeps a checklist comment on a single tracking issue (to be created when the owner approves this plan), with one line per PR or issue. Each line is updated when that item's state changes. Nothing is closed without a merged PR or a written won't-fix reason.

## 6. Out of scope and risks

- Docker, Apple-Silicon, and hardware results come only from the owner.
- #150 and #130 both change `e1_identity`, so there is a merge-conflict risk. Sequence them at H-0d.
- The Phase 1 root cause may be in the panel/hardware path, which the sim cannot reproduce. If that happens, the agent reports it and the owner collects traces.
- Every merge after the new pin is set breaks the B4 preflight again. Freeze `main` from the moment the pin is set until the B4 attempt has finished.
