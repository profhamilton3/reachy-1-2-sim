"""R12-801: Epic 8 experience system — public API surface.

Imports are intentionally flat so callers do not need to know which sub-module
owns each type.
"""

from .compatibility import (
    CompatibilityError,
    Mismatch,
    ReuseCandidate,
    ReuseDecision,
    ReusePolicy,
    ReuseRequest,
    check_reuse,
    select_reusable,
)
from .identity import (
    ResearchContextError,
    assert_research_context,
    build_simulator_identity,
)
from .models import (
    ControlPanelTaskSpec,
    EpisodeConfig,
    EpisodeResult,
    EpisodeStatus,
    PanelRouteTaskSpec,
    PickPlaceTaskSpec,
    SimulatorIdentity,
    TaskSpec,
    TrialRecord,
)
from .store import (
    ExperienceStore,
    ExperienceStoreError,
    IdentityMismatchError,
)

__all__ = [
    "CompatibilityError",
    "Mismatch",
    "ReuseCandidate",
    "ReuseDecision",
    "ReusePolicy",
    "ReuseRequest",
    "check_reuse",
    "select_reusable",
    "ResearchContextError",
    "assert_research_context",
    "build_simulator_identity",
    "ControlPanelTaskSpec",
    "EpisodeConfig",
    "EpisodeResult",
    "EpisodeStatus",
    "PanelRouteTaskSpec",
    "PickPlaceTaskSpec",
    "SimulatorIdentity",
    "TaskSpec",
    "TrialRecord",
    "ExperienceStore",
    "ExperienceStoreError",
    "IdentityMismatchError",
]
