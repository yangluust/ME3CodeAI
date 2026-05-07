"""mu grid boundary calibration, ported from mu_bound.m."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .rho_one import RhoOneCarry, solve_rho_one


@dataclass
class MuBoundResult:
    d_mu: np.ndarray
    i_muzero: int
    rho_one_carry: RhoOneCarry
    min_mu: float
    max_mu: float


def compute_mu_bounds(
    *,
    d_mu: np.ndarray,
    n_mu: int,
    i_muzero: int,
    qq: float,
    n_squig: int,
    z_dr: np.ndarray,
    n_rho: int,
    squig_trans: np.ndarray,
    vtheta_x1: float,
    kappa: float,
    vtheta_pi1: float,
    bet: float,
    bet1: float,
    kconst: float,
    ivec: np.ndarray,
    xvec1: np.ndarray,
    zetax1: float,
    zetae1: float,
    pistar: float,
    A2: float,
    B2: np.ndarray,
    progress_fn: Callable[[str], None] | None = None,
) -> MuBoundResult:
    """Reproduce mu_bound.m iterative bound adjustment."""
    z1vec = np.zeros((n_squig, 1), dtype=float)
    mubound_error = 1.0
    carry = RhoOneCarry()

    work_d_mu = np.asarray(d_mu, dtype=float).reshape(-1)
    while mubound_error > 1e-5:
        if progress_fn is not None:
            progress_fn(f"mu_bound: start outer round with error={mubound_error:.6e}")
        zround = 0
        z1vec_error = 1.0
        while z1vec_error > 1e-6:
            zround += 1
            q = 1.0 if zround == 1 else qq
            rho_res = solve_rho_one(
                endog_rho=-1,
                z_dr=z_dr,
                n_rho=n_rho,
                q=q,
                qq=qq,
                d_mu=work_d_mu,
                n_mu=n_mu,
                n_squig=n_squig,
                squig_trans=squig_trans,
                vtheta_x1=vtheta_x1,
                kappa=kappa,
                vtheta_pi1=vtheta_pi1,
                bet=bet,
                bet1=bet1,
                kconst=kconst,
                ivec=ivec,
                xvec1=xvec1,
                zetax1=zetax1,
                zetae1=zetae1,
                pistar=pistar,
                A2=A2,
                B2=B2,
                carry=carry,
                z1vec_override=z1vec,
            )
            carry = rho_res.carry
            z1vec_old = z1vec.copy()
            z1vec = rho_res.a1mat[:, i_muzero].reshape(-1, 1)
            z1vec_error = float(np.sum(np.abs(z1vec_old - z1vec)))
            if progress_fn is not None and (zround == 1 or zround % 10 == 0):
                progress_fn(f"mu_bound: inner round={zround}, z1vec_error={z1vec_error:.6e}")
        min_mu = float(np.min(rho_res.mup1mat))
        max_mu = float(np.max(rho_res.mup1mat))
        mubound_error = abs(min_mu - work_d_mu[0]) + abs(max_mu - work_d_mu[-1])
        work_d_mu = np.linspace(min_mu, max_mu, n_mu)
        if progress_fn is not None:
            progress_fn(
                "mu_bound: updated bounds "
                f"min_mu={min_mu:.6e}, max_mu={max_mu:.6e}, error={mubound_error:.6e}"
            )

    i_muzero = int(np.argmin(np.abs(work_d_mu)))
    work_d_mu[i_muzero] = 1e-6

    return MuBoundResult(
        d_mu=work_d_mu.reshape(-1, 1),
        i_muzero=i_muzero,
        rho_one_carry=carry,
        min_mu=float(work_d_mu[0]),
        max_mu=float(work_d_mu[-1]),
    )
