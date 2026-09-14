# Reviews and their evidence

Review documents that an ADR, a commit message, or a test cites as evidence
live here, next to the ADRs they support, so the citation resolves from a
checkout of this repository rather than from one person's machine.

| review | supports | probes |
|---|---|---|
| `2026-09-12-repair-review-and-56-74-design.md` | ADR-0003; `fix/bounded-review-followups-56-74` (A1–A6, R4–R6, Slice 1 of #56/#74) | `probes-2026-09-12/` |
| (PR #108 review, in the PR and ADR-0003 "Correcting the tube") | ADR-0003's tube correction: coverage, newly refused boards, the shells thumb-pad gap | `probes-2026-09-14/` |

What goes here: the review text, the probe scripts it ran, and their
human-readable summaries. What does not: raw regenerated output (`.json`
written by the probes — gitignored), test-run logs, and per-session work
orders or assignments addressed to a model run. A review committed here may
have such a section replaced with a short note saying so.

Probes are offline: `mj_forward` on the compiled scene model, never
`mj_step`, no server, no SDK, no motion. See each probes directory's README
for how to re-run them.
