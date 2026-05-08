"""Diagnostics ports for boundary and multiple-maxima logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class BoundaryEvent:
    st: int
    delta_boundary: bool
    mup_boundary: bool


@dataclass
class MultipleMaximaEvent:
    out_loop: int
    in_loop: int
    squig_st: float
    rho_st: float
    mu_lp: float
    i_mu: int
    candidate_count: int


def boundary_check(
    *,
    ptkeep: np.ndarray,
    n_squig: int,
    n_st: int,
    n_delta: int,
    n_mup: int,
    d_delta: np.ndarray,
    d_mup: np.ndarray,
    make_plots: bool = False,
) -> List[BoundaryEvent]:
    """Check boundary hits in (delta, muprime) index pointers."""
    events: List[BoundaryEvent] = []
    for st in range(n_squig, n_st - n_squig):
        rvals = ptkeep[st, 0, :]
        cvals = ptkeep[st, 1, :]
        test_delta = int(np.min(rvals)) == 1 or int(np.max(rvals)) == n_delta
        test_mup = int(np.min(cvals)) == 1 or int(np.max(cvals)) == n_mup
        if test_delta or test_mup:
            events.append(BoundaryEvent(st=st + 1, delta_boundary=test_delta, mup_boundary=test_mup))
            if make_plots:
                r_idx = np.clip(rvals.astype(int) - 1, 0, n_delta - 1)
                c_idx = np.clip(cvals.astype(int) - 1, 0, n_mup - 1)
                plt.figure()
                plt.plot(d_mup[c_idx], d_delta[r_idx], "k.")
                plt.plot(d_mup, np.max(d_delta) * np.ones_like(d_mup), "r--")
                plt.plot(d_mup, np.min(d_delta) * np.ones_like(d_mup), "r--")
                plt.xlabel(r"$\mu'$")
                plt.ylabel(r"$\delta$")
                plt.title(f"Boundary problem at st={st + 1}")
    return events


def _unique_stable_indices(vals: np.ndarray) -> np.ndarray:
    _, idx = np.unique(vals, return_index=True)
    return np.sort(idx)


def resolve_multiple_maxima(
    *,
    loc: np.ndarray,
    aic: np.ndarray,
    einfl: np.ndarray,
    n_delta: int,
    d_delta: np.ndarray,
    d_mup: np.ndarray,
    out_loop: int,
    in_loop: int,
    squig_st: float,
    rho_st: float,
    mu_lp: float,
    i_mu: int,
    bugcount: int,
    out_loop_max: int,
    in_loop_max: int,
    run_dir: Path | None = None,
    make_plots: bool = False,
    vaic: np.ndarray | None = None,
    veinfl: np.ndarray | None = None,
) -> Tuple[int, int, int, MultipleMaximaEvent | None]:
    """Port of multiple_maximaYL.m tie-resolution semantics.

    Phase 2: callers may pass precomputed ``vaic`` (= aic.reshape(-1, order='F'))
    and ``veinfl`` (= einfl.reshape(-1, order='F')) to avoid repeating the
    F-order reshape copy on every call within a single state.
    """
    if vaic is None:
        vaic = aic.reshape(-1, order="F")
    if veinfl is None:
        veinfl = einfl.reshape(-1, order="F")

    loc0 = np.asarray(loc, dtype=int) - 1
    if loc0.size == 0:
        raise ValueError("resolve_multiple_maxima called with empty loc")

    aidx = _unique_stable_indices(vaic[loc0])
    eidx = _unique_stable_indices(veinfl[loc0])
    keep = np.unique(np.concatenate((aidx, eidx)))
    loc0 = loc0[keep]

    cloc = np.ceil((loc0 + 1) / n_delta).astype(int)
    rloc = (loc0 + 1 - n_delta * (cloc - 1)).astype(int)

    event: MultipleMaximaEvent | None = None
    if rloc.size > 1:
        bugcount += 1
        event = MultipleMaximaEvent(
            out_loop=out_loop,
            in_loop=in_loop,
            squig_st=squig_st,
            rho_st=rho_st,
            mu_lp=mu_lp,
            i_mu=i_mu,
            candidate_count=int(rloc.size),
        )

    if rloc.size == 2 and bugcount < 2 and out_loop == out_loop_max and in_loop == in_loop_max and make_plots:
        c0 = cloc[0] - 1
        c_show = min(c0 + 21, einfl.shape[1])
        plt.figure()
        plt.subplot(311)
        plt.plot(d_mup[:c_show], einfl[rloc[0] - 1, :c_show], "-", label="choice1")
        plt.plot(d_mup[:c_show], einfl[rloc[1] - 1, :c_show], "-.", label="choice2")
        plt.subplot(312)
        plt.subplot(313)
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            fig_name = run_dir / f"Bug_{out_loop}_{in_loop}_{i_mu}.png"
            plt.savefig(fig_name)
        plt.close()

    chosen_r = int(rloc[0])
    chosen_c = int(cloc[0])
    return bugcount, chosen_r, chosen_c, event
