"""Offline, read-only verification tooling for the B4 goal-fix comparison.

See IITG-Reachy-Project/outputs/b4-goalfix-comparison-plan-2026-09-24.md (§3-7)
and outputs/assignment-2026-09-24-sonnet-b4-comparison-offline-tooling.md for
the specification this package implements. Every submodule is importable and
also runnable as ``python -m tools.goalfix_cmp.<name>``.

Hard limits (assignment §0): offline only -- no docker, no simulator, no
network, no motion. Nothing here builds an image, starts a service, or moves
the robot. Evidence is read only from paths a caller passes in; this package
never hard-codes a path to recorded evidence.
"""
