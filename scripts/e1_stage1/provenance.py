"""Pure function for the W4 binding-time provenance check (assignment
section 5.4): does a run directory's manifest prove the native server was
started from a tree that descends from the sha this branch is built on,
cleanly, after that sha's merge landed. No git process is invoked here --
callers inject `git_is_ancestor` so this stays testable offline."""
from __future__ import annotations

import datetime
from typing import Callable, List, Optional, Tuple

GitIsAncestor = Callable[[str, str], bool]


def _parse_iso(value: object) -> Optional[datetime.datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def check_binding_provenance(
    manifest: dict, *, git_is_ancestor: GitIsAncestor,
    required_sha: str, merge_time_iso: str,
) -> Tuple[bool, List[str]]:
    """A bare `code_sha` is not proof of what actually ran if the tree was
    dirty when the server started -- uncommitted changes are invisible to
    `git rev-parse HEAD` -- so `code_sha_dirty` must be exactly `False`,
    never just falsy-or-missing, for the sha to count as evidence at all.
    """
    reasons: List[str] = []
    code_sha = manifest.get("code_sha")
    dirty = manifest.get("code_sha_dirty")
    if not isinstance(code_sha, str) or not code_sha:
        reasons.append(
            "manifest has no code_sha -- cannot prove which tree the "
            "server ran")
    elif dirty is not False:
        reasons.append(
            f"code_sha_dirty={dirty!r} -- a dirty or unknown tree makes "
            "the sha alone unprovable as runtime identity; refusing")
    elif not git_is_ancestor(required_sha, code_sha):
        reasons.append(
            f"code_sha {code_sha} is not a descendant of {required_sha}")

    started_at = manifest.get("started_at")
    started_dt, merge_dt = _parse_iso(started_at), _parse_iso(merge_time_iso)
    if started_dt is None or merge_dt is None or started_dt <= merge_dt:
        reasons.append(
            f"manifest.started_at={started_at!r} is not after the merge "
            f"at {merge_time_iso}")

    return (not reasons, reasons)
