"""Analytical rho=1 block ported from rhoone_new.m."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RhoOneCarry:
    lamb_amu_new: float = 0.0
    bavec_new: np.ndarray | None = None
    Umumu_new: float = 0.0
    Dmuvec_new: np.ndarray | None = None
    Dvec_new: np.ndarray | None = None


@dataclass
class RhoOneResult:
    a1mat: np.ndarray
    alpha1mat: np.ndarray
    e1mat: np.ndarray
    mup1mat: np.ndarray
    U1mat: np.ndarray
    bmuvec_new: np.ndarray
    lamb_mumu_new: float
    carry: RhoOneCarry


def solve_rho_one(
    *,
    endog_rho: int,
    z_dr: np.ndarray,
    n_rho: int,
    q: float,
    qq: float,
    d_mu: np.ndarray,
    n_mu: int,
    n_squig: int,
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
    carry: RhoOneCarry | None = None,
    z1vec_override: np.ndarray | None = None,
) -> RhoOneResult:
    """Compute analytical rho=1 decision/value objects."""
    if carry is None:
        carry = RhoOneCarry()
    if carry.bavec_new is None:
        carry.bavec_new = np.zeros((n_squig, 1), dtype=float)
    if carry.Dmuvec_new is None:
        carry.Dmuvec_new = np.zeros((n_squig, 1), dtype=float)
    if carry.Dvec_new is None:
        carry.Dvec_new = np.zeros((n_squig, 1), dtype=float)

    if endog_rho == 1:
        z1vec = z_dr[:, n_rho - 1].reshape(-1, 1)
    elif endog_rho == 0:
        if z_dr.ndim != 2:
            raise ValueError(
                f"solve_rho_one expects 2-D z_dr (n_squig, n_rho); got shape {z_dr.shape}"
            )
        z1vec = np.asarray(z_dr[:, n_rho - 1], dtype=float).reshape(-1, 1)
    elif endog_rho == -1:
        if z1vec_override is None:
            raise ValueError("z1vec_override is required when endog_rho == -1")
        z1vec = np.asarray(z1vec_override, dtype=float).reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported endog_rho value: {endog_rho}")

    if q < 1.0:
        lamb_amu = float(carry.lamb_amu_new)
        bavec = np.asarray(carry.bavec_new, dtype=float).reshape(-1, 1)
        Umumu = float(carry.Umumu_new)
        Dmuvec = np.asarray(carry.Dmuvec_new, dtype=float).reshape(-1, 1)
        Dvec = np.asarray(carry.Dvec_new, dtype=float).reshape(-1, 1)
    else:
        lamb_amu = 0.0
        bavec = np.zeros((n_squig, 1), dtype=float)
        Umumu = 0.0
        Dmuvec = np.zeros((n_squig, 1), dtype=float)
        Dvec = np.zeros((n_squig, 1), dtype=float)

    vthetacomb = vtheta_x1 / (kappa**2)
    h = np.array(
        [
            [0.0, 1.0, -bet * (1.0 - q) / kconst * lamb_amu],
            [vthetacomb, -vthetacomb, 1.0],
            [-(vtheta_pi1 + vthetacomb), vthetacomb, 0.0],
        ],
        dtype=float,
    )
    bl1vec = -bet * (squig_trans @ ((1.0 - q) * bavec + q * z1vec))
    bl2vec = -vthetacomb * (ivec + kappa * xvec1) - zetax1 / kappa - zetae1
    bl3vec = vthetacomb * (ivec + kappa * xvec1) + zetax1 / kappa + vtheta_pi1 * pistar

    lamb_new = np.linalg.solve(h, np.array([0.0, 0.0, 1.0 - q], dtype=float))
    lamb_amu_new = float(lamb_new[0])
    lamb_emu_new = float(lamb_new[1])
    lamb_mumu_new = float(lamb_new[2] / kconst)
    lamb_alphamu_new = lamb_emu_new * A2

    rhs = np.vstack([-bl1vec.T, -bl2vec.T, -bl3vec.T])
    betamat = np.linalg.solve(h, rhs)
    bavec_new = betamat[0, :].reshape(-1, 1)
    bevec_new = betamat[1, :].reshape(-1, 1)
    bmuvec_new = (betamat[2, :].reshape(-1, 1)) / kconst
    balphavec_new = A2 * bevec_new + B2.reshape(-1, 1)

    vmumu = -0.5 * vtheta_pi1 * lamb_amu_new**2 - 0.5 * vthetacomb * (lamb_amu_new - lamb_emu_new) ** 2
    vec_comb = bavec_new - (bevec_new + ivec + kappa * xvec1)
    vmuvec = (
        -zetae1 * lamb_emu_new
        + (zetax1 / kappa) * (lamb_amu_new - lamb_emu_new)
        - vtheta_pi1 * lamb_amu_new * bavec_new
        - vthetacomb * (lamb_amu_new - lamb_emu_new) * vec_comb
        + vtheta_pi1 * pistar * lamb_amu_new
    )
    vvec = (
        -0.5 * vtheta_pi1 * bavec_new**2
        - zetae1 * bavec_new
        - 0.5 * vthetacomb * vec_comb**2
        + (zetax1 / kappa) * vec_comb
        + vtheta_pi1 * pistar * bavec_new
    )

    Umumu_new = float(vmumu + bet1 * (1.0 - q) * Umumu * lamb_mumu_new**2)
    Dmuvec_new = vmuvec + bet1 * (1.0 - q) * lamb_mumu_new * (2.0 * Umumu * bmuvec_new + squig_trans @ Dmuvec)
    Dvec_new = (
        vvec
        + bet1 * (1.0 - q) * Umumu * bmuvec_new**2
        + bet1 * (1.0 - q) * (bmuvec_new * (squig_trans @ Dmuvec) + squig_trans @ Dvec)
    )

    d_mu_row = d_mu.reshape(1, -1)
    a1mat = lamb_amu_new * np.repeat(d_mu_row, n_squig, axis=0) + np.repeat(bavec_new, n_mu, axis=1)
    alpha1mat = lamb_alphamu_new * np.repeat(d_mu_row, n_squig, axis=0) + np.repeat(balphavec_new, n_mu, axis=1)
    e1mat = lamb_emu_new * np.repeat(d_mu_row, n_squig, axis=0) + np.repeat(bevec_new, n_mu, axis=1)
    mup1mat = lamb_mumu_new * np.repeat(d_mu_row, n_squig, axis=0) + np.repeat(bmuvec_new, n_mu, axis=1)
    d_musq = d_mu**2
    U1mat = (
        Umumu_new * np.repeat(d_musq.reshape(1, -1), n_squig, axis=0)
        + Dmuvec_new @ d_mu.reshape(1, -1)
        + np.repeat(Dvec_new, n_mu, axis=1)
    )

    out_carry = RhoOneCarry(
        lamb_amu_new=lamb_amu_new,
        bavec_new=bavec_new,
        Umumu_new=Umumu_new,
        Dmuvec_new=Dmuvec_new,
        Dvec_new=Dvec_new,
    )
    return RhoOneResult(
        a1mat=a1mat,
        alpha1mat=alpha1mat,
        e1mat=e1mat,
        mup1mat=mup1mat,
        U1mat=U1mat,
        bmuvec_new=bmuvec_new,
        lamb_mumu_new=lamb_mumu_new,
        carry=out_carry,
    )
