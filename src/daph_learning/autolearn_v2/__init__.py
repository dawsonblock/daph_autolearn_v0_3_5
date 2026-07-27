"""AutoLearn v2: empirical, counterfactual, utility-driven policy learning.

This package implements the redesign specified in the AutoLearn v2 design
brief. It coexists with the legacy :mod:`daph_learning.autolearn.loop`
(which is preserved for backwards compatibility) and does not modify its
behaviour.

The v2 core loop is::

    task
      -> capture representation
      -> execute candidate backends (counterfactual during learning)
      -> verify outputs (typed; no substring matching)
      -> measure reward/utility per backend
      -> derive optimal action / reward gap
      -> store experience
      -> sample replay / hard examples
      -> produce candidate policy update (incremental, trust-region)
      -> evaluate candidate on immutable validation set
      -> accept or reject
      -> version + retain + rollback

See :mod:`daph_learning.autolearn_v2.engine` for the orchestrator.
"""

from .experience import BackendOutcome, Experience, fingerprint_task, make_experience_id
from .reward import UtilityConfig, backend_reward, optimal_action, reward_gap
from .replay import ReplayBuffer, ReplayConfig, ReplaySample
from .policies import (
    Policy,
    SingleVectorPolicy,
    MultiVectorPolicy,
    ConditionalSteeringPolicy,
)
from .updater import UpdateConfig, compute_candidate_update, trust_region_clip
from .acceptance import AcceptanceConfig, CandidateDecision, evaluate_candidate
from .registry import PolicyRecord, PolicyRegistry, RegistryError
from .checkpoint import CheckpointState, load_checkpoint, save_checkpoint
from .counterfactual import (
    CounterfactualConfig,
    ExecutionMode,
    collect_counterfactual_experience,
)
from .engine import AutoLearnV2Config, AutoLearnV2Result, run_autolearn_v2

__all__ = [
    "AcceptanceConfig",
    "AutoLearnV2Config",
    "AutoLearnV2Result",
    "BackendOutcome",
    "CandidateDecision",
    "CheckpointState",
    "ConditionalSteeringPolicy",
    "CounterfactualConfig",
    "ExecutionMode",
    "Experience",
    "MultiVectorPolicy",
    "Policy",
    "PolicyRecord",
    "PolicyRegistry",
    "ReplayBuffer",
    "ReplayConfig",
    "ReplaySample",
    "RegistryError",
    "SingleVectorPolicy",
    "UpdateConfig",
    "UtilityConfig",
    "backend_reward",
    "collect_counterfactual_experience",
    "compute_candidate_update",
    "evaluate_candidate",
    "fingerprint_task",
    "load_checkpoint",
    "make_experience_id",
    "optimal_action",
    "reward_gap",
    "run_autolearn_v2",
    "save_checkpoint",
    "trust_region_clip",
]
