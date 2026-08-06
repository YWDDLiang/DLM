"""Stratified Wyckoff-quotient co-diffusion.

The package is intentionally split into a dependency-light formal core and
optional crystallography / PyTorch adapters.  Importing :mod:`wqcodiff` never
imports torch, pymatgen, spglib, or pyxtal, which keeps contract and formal
audits runnable on login/CPU nodes.
"""

import os as _os


# The cluster account has a strict process/thread ceiling.  Default these
# before any optional NumPy/PyTorch import so a missing shell prefix cannot
# make OpenBLAS fan out to the login node's CPU count.  Explicit values are
# preserved because Slurm jobs audit and set their rank-local thread budgets.
for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_thread_variable, "1")
del _os, _thread_variable

from .bridge import BridgeResult, ChartCatalog, ChartSpec, TargetStratumBridge
from .contracts import (
    AttemptAudit,
    ArtifactLedger,
    AttemptLedger,
    AttemptRecord,
    AttemptStatus,
    SeedDeriver,
)
from .events import TopologyEvent, TopologyEventType
from .kernel import TopologyEventKernel, TransitionError
from .state import GeometryEvidence, OrbitState, StratifiedState

__all__ = [
    "AttemptAudit",
    "ArtifactLedger",
    "AttemptLedger",
    "AttemptRecord",
    "AttemptStatus",
    "BridgeResult",
    "ChartCatalog",
    "ChartSpec",
    "GeometryEvidence",
    "OrbitState",
    "SeedDeriver",
    "StratifiedState",
    "TargetStratumBridge",
    "TopologyEvent",
    "TopologyEventKernel",
    "TopologyEventType",
    "TransitionError",
]
