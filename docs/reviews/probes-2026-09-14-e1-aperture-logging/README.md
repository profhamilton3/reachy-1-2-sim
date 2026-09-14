# Evidence for the E1 aperture-logging PR (2026-09-14)

`synthetic_recording_demo.py` builds a hand-built joint log in exactly the
schema `record_joint_log`/`save_log` now produce (ARM7 + `r_gripper`,
degrees, `t` from a monotonic clock, `schema_version: 2`) — a plausible
LOWER_TO_REST flight, gripper closing from open (PRESENT) to shut (REST) —
and runs it through `measure_route_clearance.report()`. No hardware, no
simulator, no SDK import exercised, no motion. Run from the repo root:

    PYTHONPATH=src:native_mujoco:scripts python3 \
        docs/reviews/probes-2026-09-14-e1-aperture-logging/synthetic_recording_demo.py

`synthetic_recording_demo.txt` is its recorded output. It shows, in order:

1. the schema (first/last sample),
2. `validated_samples()` accepting a fully-valid log,
3. the full `report()` output (realised at the actual per-sample aperture,
   vs planned; each half's `..._source`/`..._policy` field names where its
   aperture actually comes from — added in the 2026-09-14 review follow-up),
4. **schema-aware loading** (added in the follow-up): the same log stripped
   of `r_gripper` still raises even with `allow_missing_aperture=True` when
   its `schema_version` is omitted (defaults to current) or an unrecognised
   value (`3`) is passed — only a log explicitly *declared*
   `schema_version=1` is eligible for the fallback,
5. the same log reprocessed as if it were a pre-2026-09-14 (schema 1) log,
   correctly declared as such — every sample's aperture assumed open,
   explicitly via `allow_missing_aperture=True`, named in
   `realised_aperture_assumed_samples`,
6. the resulting per-object difference: actual-aperture clearance reads
   +1.36 cm more than assumed-open at every object here (the flight's
   apertures are all narrower than fully open, so the correct capsule is
   smaller and less conservative — the assumed-open path is always at
   least this conservative, never the reverse),
7. one sample corrupted to `NaN`, refused by `report()` with
   `ApertureDataError` rather than silently treated as open,
8. **preserved-invalid-recording** (added in the follow-up): the corrupted
   log is saved via `save_invalid_log` (`_INVALID.json`, `"valid": false`,
   the failure reason, every sample including the offending one kept) —
   and reading it back the normal, schema-aware way raises
   `UnsupportedSchemaVersionError` outright, so it can never be mistaken
   for usable E1 data.

This is a demonstration of the mechanism, not an E1 result: the SivaPool
scene's objects sit far (~0.7–1.3 m) from this particular flight's path, so
the absolute clearances are not close calls. E1 itself — an operator flying
a real route while this recorder streams — is unrun and unaffected by this
PR; it needs a live arm.
