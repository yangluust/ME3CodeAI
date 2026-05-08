"""Lightweight parity checks against MATLAB output artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.io import loadmat

from .solver import SolverResult


@dataclass
class ParitySummary:
    compared_files: int
    metrics: Dict[str, float]


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return float(np.max(np.abs(a - b)))


def check_against_matlab(result: SolverResult, matlab_dir: str | Path, out_loop: int, in_loop: int) -> ParitySummary:
    """Compare key outputs with MATLAB MPE artifact for run_no=51 workflow."""
    p = Path(matlab_dir)
    mpe_file = p / f"MPE{out_loop}W{in_loop}.mat"
    if not mpe_file.exists():
        raise FileNotFoundError(f"Missing MATLAB artifact: {mpe_file}")
    m = loadmat(str(mpe_file))
    metrics: Dict[str, float] = {
        "a_dr_max_abs_diff": _max_abs_diff(result.a_dr, np.asarray(m["a_dr"])),
        "alpha_dr_max_abs_diff": _max_abs_diff(result.alpha_dr, np.asarray(m["alpha_dr"])),
        "einfl_dr_max_abs_diff": _max_abs_diff(result.einfl_dr, np.asarray(m["einfl_dr"])),
        "muprime_dr_max_abs_diff": _max_abs_diff(result.muprime_dr, np.asarray(m["muprime_dr"])),
        "Unew_max_abs_diff": _max_abs_diff(result.Unew, np.asarray(m["Unew"])),
    }
    return ParitySummary(compared_files=1, metrics=metrics)
