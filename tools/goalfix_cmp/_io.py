"""SHA256SUMS integrity checking and the shared JSON result contract.

The integrity-check trio (``sha256_of``/``parse_sha256sums``/``verified``) is
the same pattern as ``sha``/``sums_of``/``verified`` in
IITG-Reachy-Project/outputs/analysis-2026-09-23-goal-feedback-verification/verify_goal_feedback.py
-- see tests/unit/test_goalfix_cmp_equivalence.py, which pins that file's own
sha256 against a vendored copy and checks this reimplementation agrees with
it on synthetic input. Reimplemented rather than imported because the
original is a throwaway analysis script, not a package this can depend on.

Exit-code contract (assignment §1): every CLI in this package writes JSON to
a given path and exits 0 (pass), 2 (fail/STOP) or 3 (inconclusive/evidence
incomplete) -- never 0 on missing, malformed or ambiguous input.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Union

PathLike = Union[str, os.PathLike]

RC_OK = 0
RC_STOP = 2
RC_INCONCLUSIVE = 3


class IntegrityError(ValueError):
    """A file failed, or could not be checked against, its SHA256SUMS entry."""


def sha256_of(path: PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sha256sums(path: PathLike) -> Dict[str, str]:
    """{normalized relative path: lowercase hex digest}, ``sha256sum`` format."""
    out: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            digest, rel = line.split(None, 1)
            out[os.path.normpath(rel.lstrip("*"))] = digest.lower()
    return out


def verified(evidence_dir: PathLike, rel_path: str, sums: Dict[str, str]) -> Path:
    """The path to ``rel_path`` under ``evidence_dir``, after checking its
    sha256 against ``sums``. Raises ``IntegrityError`` -- naming the file --
    on a missing manifest entry, a missing file, or a hash mismatch. Never
    returns a path whose bytes were not checked."""
    key = os.path.normpath(rel_path)
    p = Path(evidence_dir) / rel_path
    if key not in sums:
        raise IntegrityError(f"{rel_path}: no entry in SHA256SUMS")
    if not p.is_file():
        raise IntegrityError(f"{rel_path}: file is missing")
    actual = sha256_of(p)
    if actual != sums[key]:
        raise IntegrityError(
            f"{rel_path}: sha256 mismatch (expected {sums[key]}, got {actual})")
    return p


def write_result(path: PathLike, rc: int, payload: Dict[str, Any]) -> int:
    """Write ``payload`` (plus its own ``rc``) as JSON to ``path``. Returns
    ``rc`` so a CLI's ``main`` can end with ``return write_result(...)`` and
    ``raise SystemExit(main())``."""
    if rc not in (RC_OK, RC_STOP, RC_INCONCLUSIVE):
        raise ValueError(f"rc must be 0, 2 or 3, got {rc!r}")
    body = dict(payload)
    body["rc"] = rc
    Path(path).write_text(json.dumps(body, indent=2, sort_keys=True, default=str))
    return rc


def read_result(path: PathLike) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())
