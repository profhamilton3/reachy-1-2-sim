# B4 goalfix comparison: RETIRED (2026-10-08)

**Decision (owner, H-0d, 2026-10-08):** the B4 goalfix comparison (plan revision 7, baseline `8c0dad2` vs fixed `67730a1`) is **retired**, so that work can move to the current physics on `main`.

**No B4 result is claimed. The comparison is INCOMPLETE.** No cycle of any attempt completed, and no "supports", "refutes" or "inconclusive" verdict exists.

The plan is kept unchanged at [`b4-goalfix-comparison-plan-rev7.md`](b4-goalfix-comparison-plan-rev7.md) (sha256 `3ede8c0dc0feeff4a5770d5e88d2cf016d6dc0f2c68bcf378b0fce6ab0b6d6b3`).

## Why it could not continue as written

1. **The §10 runtime-tree equality no longer holds on `main`.** Rev 7 §10 requires `git rev-parse 67730a1:<p>` to equal the host pin's tree for every runtime path. At `main` (`49d5da5`, 49 commits past M′ `dbc878c`), **7 trees differ**: `native_mujoco` (PR #156, the no-slip contact model, which is a physics change), `scripts`, `src`, `scenes`, `notebooks`, `web` and `docker-compose.yml`. The diff-scope rule (only `tools/goalfix_cmp/`, `tests/` and two named docs) fails too. Any host pin on current physics therefore breaks the plan.
2. **The design reproduces the a004 STOP.** §6 runs one native server for the whole comparison, and §5 recreates the container in every cycle (P2, including cycle 1). A recreated bridge restarts its command sequence while the native `cmd_seq` does not. P8's `require_compliance(min_cmd_seq)` then fails closed, which is exactly the a004 STOP (H4 caveat, #119). PR #150 addresses this, but it changes `scripts/`, so a pin that includes it also fails §10.

## Attempt history

Records are in `~/Reachy-Lab/outputs/sim/` and are cited, not copied. Every attempt used plan rev 7 with the stage-1 four-cycle plan, and every grant was "once, no retry".

| attempt | date (UTC) | grant | outcome |
|---|---|---|---|
| r4 (bundle `767b968c…`) | 2026-09-30 | `from-tabletop-project/authorization-2026-09-30-cmp-stage1-r4-GRANTED.md` (`803583bb51f0cfbe39f57ec7147dc9727f186676b5257728e0e511f52afddf54`) | **STOP at phase 4 `start_sim`** (wrapper 300 s liveness bound; the server had started in about 15 s). 0 cycles. Report `from-tabletop-project/report-2026-09-30-b4-cmp-stage1-r4-STOP-start-sim.md` (`15247c780b4ef3ff02271b7554d16141c157887d0e844ef4ca70d465522cc848`). Sealed `~/b4-goalfix-cmp-s1-20260930`, archive `b1b623ca5aa5151a5a8728c589aa67785477b0febcd574b2e4be5c7982aff85f`. |
| r5 (bundle `88117ebb…`) | 2026-09-30 / 2026-10-01 | `from-tabletop-project/authorization-2026-09-30-cmp-stage1-r5-GRANTED.md` (`e661c738d22bb8f328c4a7663407b6006d5e018187d05e85aa90be5f0c18d5c9`), with an option-1 re-grant | First try: **STOP at step 1 preflight** (date-derived paths), nothing created: `from-tabletop-project/checkpoint-2026-09-30-cmp-stage1-r5-preflight-STOP.md` (`1ff6dc6ca693513fd3acea74254733aa0a7c82c9d861ad65cfbdd8394775ab51`). Re-grant 2026-10-01: **STOP at Stage 0 `leg.sh` rc 127** (git mode 100644). 0 cycles; CLEANUP VERIFIED. Report `from-tabletop-project/report-2026-10-01-b4-cmp-stage1-r5-STOP-stage0-leg.md` (`9329ee18d8395478a10cca34c3c3687189ae11f1a8cfb014d3f60b9b56ffbad7`). Sealed `~/b4-goalfix-cmp-s1-20261001` (`SHA256SUMS` `6929c50bfcd96198f4efd665f4181f59a2dd6c4c827602ae6740c9370c850e12`), archive `e9d4397d22364d588d07130c09d170941e9fe03d0935a5abf95c3e3dc7028979`. |
| a004 (bundle r6 `3a930301…`) | 2026-10-02 | `working/authorization-2026-10-02-cmp-stage1-r6-a004-GRANTED.md` (`7a2fe9b72435e5fea2d30b3ec78ad628be1067d40c45a6d32252e8166e1d5b93`) | Stage 0 passed for the first time. **STOP in cycle r1 (A) setup leg** at the `turn_on` compliance gate (`min_cmd_seq=2`, sample `cmd_seq=1`). 0 cycles completed; CLEANUP VERIFIED. Report `working/report-2026-10-02-b4s1-a004-STOP-r1-setup-compliance-gate.md` (`07d4d4793ce6104b943214254d903b705402f44b6f15c93c3a52c77c76b5f4c7`); closure `working/checkpoint-2026-10-02-b4s1-a004-closed.md` (`43295a92fd18cfe595e8c0bbd9e96702a67257c3e5c52e85cd8e4036667c57e0`). Sealed tree `SHA256SUMS` `8b92d1431b10b578089abaf18a4d61949b257fd740f4b9c2adfd2ead50bd3524`, archive `1dfbc17dd570af778e03d110cc7c0da145d5e5b0f20441d8fe83bfd1e360d5e6`. Grant exhausted. |

An earlier attempt on 2026-09-29 stopped at build parity before the server started (`from-tabletop-project/report-2026-09-29-b4-cmp-stage1-STOP-build-parity.md`, `866ccddf2a3aee11c9a962a668a6e431797b6c39110bae67f303fb3e38757628`).

## Disposition (decided 2026-10-08)

- **#146 and #148:** closed as not pursued. Their branches are kept for reference.
- **#150:** to be merged into `main` as a general E1 freshness fix, no longer as a B4 host-pin repair.
- **`tools/goalfix_cmp/` and its tests:** left unchanged.
- **Plans rev 3–7 and all evidence** (sessions, archives, bundles, run specs) stay in `~/Reachy-Lab`. Rev 7 alone is also in this directory.

## Stale `IITG-Reachy-Project/outputs/…` citations (not changed)

The 2026-10-01 cleanup moved these files to `~/Reachy-Lab/outputs/sim/from-tabletop-project/`, keeping the same file names (see `~/Reachy-Lab/catalog/path-map.csv`). All targets exist there.

| cited at | cited file |
|---|---|
| `docs/reviews/probes-2026-09-19-e1-stage2-b1/REPORT.md:8`, `REPRODUCTION.md:11`, `control/checkpoint_1.txt:2` | `e1-stage2-b1-execution-plan-2026-09-19.md` |
| `docs/reviews/probes-2026-09-19-e1-stage2-b4/REPORT.md:4`, `REPRODUCTION.md:13` | `e1-stage2-b4-execution-plan-2026-09-18.md` |
| `tests/integration/test_goal_reporting_sdk_path.py:11` | `analysis-2026-09-23-goal-feedback-verification-report.md` |
| `tests/unit/test_goalfix_cmp_historical.py:5`, `tools/goalfix_cmp/__init__.py:3` | `b4-goalfix-comparison-plan-2026-09-24.md` (rev 7) |
| `tools/goalfix_cmp/_io.py:5`, `tools/goalfix_cmp/echo.py:3` | `analysis-2026-09-23-goal-feedback-verification/verify_goal_feedback.py` |
| `tools/goalfix_cmp/clearance.py:18` | `analysis-2026-09-23-corrected-campaign/hover_wristball.py` |

Other `IITG-Reachy-Project/…` mentions in this repository (for example `docs/pics/`, `tests/Pictures/`, `data/annotations/`, `src/`) point to tabletop-ai paths that were not moved. They are not stale.
