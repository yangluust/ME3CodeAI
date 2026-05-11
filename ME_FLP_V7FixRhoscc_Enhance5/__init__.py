"""Reusable Python port of ME_FLP_V7FixRhoscc.m (constant-reputation, rho'=rho).

Phase 5-equivalent: numba prange parallel state worker, JIT'd argmax/tie
kernels. Skips V6's bilinear (rho', mu) interpolation since rho'=rho fixed.
"""

from .config import load_run_config
from .solver import run_model

__all__ = ["load_run_config", "run_model"]
