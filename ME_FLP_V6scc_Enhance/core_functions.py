"""Core helper ports from MATLAB subroutines/utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.interpolate import interpn


def vec(x: np.ndarray) -> np.ndarray:
    """Column-major vectorization (MATLAB x(:))."""
    return np.asarray(x).reshape(-1, order="F")


def onesz(x: np.ndarray) -> np.ndarray:
    """Return an array of ones with same shape."""
    return np.ones_like(np.asarray(x), dtype=float)


def rouwenhorst(nu: float, sigmat_innov: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Port of MATLAB rouwenhorst.m."""
    if n < 2:
        raise ValueError("n must be >= 2 for Rouwenhorst approximation")
    q = (nu + 1.0) / 2.0
    scale = np.sqrt((n - 1.0) / (1.0 - nu**2)) * sigmat_innov
    vtheta = np.array([[q, 1.0 - q], [1.0 - q, q]], dtype=float)
    for i in range(2, n):
        z = np.zeros((i, 1), dtype=float)
        vtheta = (
            q * np.block([[vtheta, z], [np.zeros((1, i + 1)),]])
            + (1.0 - q) * np.block([[z, vtheta], [np.zeros((1, i + 1)),]])
            + (1.0 - q) * np.block([[np.zeros((1, i + 1))], [vtheta, z]])
            + q * np.block([[np.zeros((1, i + 1))], [z, vtheta]])
        )
        vtheta[1:i, :] = vtheta[1:i, :] / 2.0
    zgrid = np.linspace(-scale, scale, n)
    return zgrid.reshape(-1, 1), vtheta


def npdf(x: np.ndarray, mn: np.ndarray | float, sig: float, trunc: str | None = None) -> np.ndarray:
    """Port of npdf.m with optional MATLAB-style trunc normalization."""
    x_arr = np.asarray(x, dtype=float).reshape(-1, 1)
    mn_arr = np.asarray(mn, dtype=float).reshape(-1, 1)
    if sig <= 0:
        raise ValueError("sig must be positive")
    if mn_arr.size > 1:
        x_mat = np.repeat(x_arr, mn_arr.size, axis=1)
        mn_mat = np.repeat(mn_arr.T, x_arr.shape[0], axis=0)
    else:
        x_mat = x_arr
        mn_mat = np.full_like(x_arr, float(mn_arr.item()))
    factor = 1.0 / (sig * np.sqrt(2.0 * np.pi))
    f = factor * np.exp(-0.5 * ((x_mat - mn_mat) / sig) ** 2)
    if trunc == "trunc":
        col_sum = np.sum(f, axis=0, keepdims=True)
        if np.any(col_sum == 0):
            raise ValueError("Cannot normalize npdf columns with zero sum")
        f = f / col_sum
    return f


def br_normal(
    infl: np.ndarray,
    a: np.ndarray | float,
    alph: np.ndarray | float,
    repstate: np.ndarray | float,
    siga: float,
    sigalph: float | None = None,
) -> np.ndarray:
    """Port of br_normal.m."""
    if sigalph is None:
        sigalph = siga
    infl_arr = np.asarray(infl, dtype=float)
    a_arr = np.asarray(a, dtype=float)
    alph_arr = np.asarray(alph, dtype=float)
    rep_arr = np.asarray(repstate, dtype=float)
    num = rep_arr * npdf(infl_arr, a_arr, siga)
    den = num + (1.0 - rep_arr) * npdf(infl_arr, alph_arr, sigalph)
    if np.any(den == 0):
        raise ValueError("Zero denominator encountered in br_normal")
    return num / den


def deltabound(
    rhoprime1: np.ndarray, rhoprime2: np.ndarray, fdens: np.ndarray, n_rho: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Port of deltabound core numerical output."""
    fd = np.asarray(fdens, dtype=float).reshape(-1, 1)
    e1 = np.zeros((n_rho, rhoprime1.shape[1]), dtype=float)
    e2 = np.zeros_like(e1)
    for i_rho in range(n_rho):
        e1[i_rho, :] = (fd.T @ rhoprime1[:, :, i_rho]).reshape(-1)
        e2[i_rho, :] = (fd.T @ rhoprime2[:, :, i_rho]).reshape(-1)
    return e1, e2


def get_mhat(dr: np.ndarray, z: np.ndarray, squig_trans: np.ndarray, q: float) -> np.ndarray:
    """Port of getMhat.m."""
    n_squig, n_rho, n_mu = dr.shape
    mhat = (1.0 - q) * dr + q * np.repeat(z[:, :, None], n_mu, axis=2)
    mhat2 = mhat.reshape((n_squig, n_rho * n_mu), order="F")
    mhat2 = (squig_trans @ mhat2).reshape((n_squig, n_rho, n_mu), order="F")
    return mhat2


def ii_fhat(
    rh: np.ndarray,
    mu: np.ndarray,
    f: np.ndarray,
    rhq: np.ndarray,
    mupq: np.ndarray,
    fdens: np.ndarray,
    n_eps: int,
    n_delta: int,
    n_mup: int,
) -> np.ndarray:
    """Port of iiFhat.m interpolation+integration routine."""
    rh_grid = np.asarray(rh, dtype=float).reshape(-1)
    mu_grid = np.asarray(mu, dtype=float).reshape(-1)
    points = (rh_grid, mu_grid)
    vals = np.asarray(f, dtype=float)
    # MATLAB iiFhat uses interpn on ndgrid arrays, then reshape/sum in column-major order.
    # Preserve that ordering by flattening query grids in Fortran order.
    rhq_arr = np.asarray(rhq, dtype=float)
    mupq_arr = np.asarray(mupq, dtype=float)
    query = np.column_stack((rhq_arr.reshape(-1, order="F"), mupq_arr.reshape(-1, order="F")))
    interp_vals = interpn(points=points, values=vals, xi=query, method="linear", bounds_error=True)
    interp_cube = np.asarray(interp_vals, dtype=float).reshape((n_eps, n_delta, n_mup), order="F")
    dens = np.asarray(fdens, dtype=float).reshape(n_eps, 1, 1)
    return np.sum(dens * interp_cube, axis=0)


def bilinear_axis_table(
    grid: np.ndarray, query: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (idx, w) for 1-D linear interpolation on a strictly increasing grid.

    For any function f sampled on `grid`, the linear-interpolated value at `query`
    is `(1.0 - w) * f[idx] + w * f[idx + 1]`.

    Parameters
    ----------
    grid : 1-D array, strictly monotone increasing, size >= 2.
    query : array of any shape; every element must lie in [grid[0], grid[-1]].

    Raises
    ------
    ValueError
        If `grid` has fewer than 2 points, is not strictly monotone increasing,
        or any query lies outside `[grid[0], grid[-1]]`. Mirrors the
        `bounds_error=True` behaviour of `scipy.interpolate.interpn`.
    """
    grid_arr = np.asarray(grid, dtype=float).reshape(-1)
    query_arr = np.asarray(query, dtype=float)
    if grid_arr.size < 2:
        raise ValueError("bilinear_axis_table: grid must have at least 2 points")
    if not np.all(np.diff(grid_arr) > 0):
        raise ValueError("bilinear_axis_table: grid must be strictly increasing")
    g_min = grid_arr[0]
    g_max = grid_arr[-1]
    if np.any(query_arr < g_min) or np.any(query_arr > g_max):
        raise ValueError(
            f"bilinear_axis_table: query out of grid bounds [{g_min}, {g_max}]"
        )
    idx = np.searchsorted(grid_arr, query_arr, side="right") - 1
    np.clip(idx, 0, grid_arr.size - 2, out=idx)
    span = grid_arr[idx + 1] - grid_arr[idx]
    w = (query_arr - grid_arr[idx]) / span
    return idx, w


def ii_fhat_factored(
    f: np.ndarray,
    rho_idx: np.ndarray,
    rho_w: np.ndarray,
    mu_idx: np.ndarray,
    mu_w: np.ndarray,
    fdens: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation + epsilon integration using precomputed axis tables.

    Mathematically equivalent to `ii_fhat(rh, mu, f, rhq, mupq, fdens, n_eps, n_delta, n_mup)`
    when the precomputed tables are derived as
        ``rho_idx, rho_w = bilinear_axis_table(rh, rhprime_slab)``  with
        ``rhprime_slab.shape == (n_eps, n_delta)`` (i.e. ``rhq[:, j]`` for any ``j``)
    and
        ``mu_idx, mu_w = bilinear_axis_table(mu, d_mup)`` with
        ``d_mup.shape == (n_mup,)`` (i.e. ``mupq[i, :]`` for any ``i``).

    The factoring exploits the fact that `rhq` is constant across the `n_mup`
    columns of the legacy interface and `mupq` is constant across the
    `n_eps * n_delta` rows, so the bilinear gather reduces from a 4.29M-point
    scattered query to two precomputed 1-D tables.

    Parameters
    ----------
    f : (n_rho, n_mu) array, the source samples.
    rho_idx : (n_eps, n_delta) integer array, lower-corner index along axis 0.
    rho_w   : (n_eps, n_delta) float array, fractional offset along axis 0.
    mu_idx  : (n_mup,) integer array, lower-corner index along axis 1.
    mu_w    : (n_mup,) float array, fractional offset along axis 1.
    fdens   : (n_eps,) array, the epsilon-density weights to integrate against.

    Returns
    -------
    (n_delta, n_mup) array.
    """
    f_arr = np.asarray(f, dtype=float)
    if f_arr.ndim != 2:
        raise ValueError(f"ii_fhat_factored: f must be 2D (n_rho, n_mu); got {f_arr.shape}")
    fdens_arr = np.asarray(fdens, dtype=float).reshape(-1)
    if rho_idx.shape != rho_w.shape:
        raise ValueError("ii_fhat_factored: rho_idx and rho_w shape mismatch")
    if mu_idx.shape != mu_w.shape:
        raise ValueError("ii_fhat_factored: mu_idx and mu_w shape mismatch")
    n_eps_table = rho_idx.shape[0]
    if fdens_arr.size != n_eps_table:
        raise ValueError(
            f"ii_fhat_factored: fdens size {fdens_arr.size} must match n_eps {n_eps_table}"
        )

    fdens_col = fdens_arr.reshape(-1, 1)
    weight_lo = (1.0 - rho_w) * fdens_col
    weight_hi = rho_w * fdens_col
    f_lo = f_arr[rho_idx]
    f_hi = f_arr[rho_idx + 1]
    g = np.einsum("ed,edm->dm", weight_lo, f_lo) + np.einsum("ed,edm->dm", weight_hi, f_hi)
    return (1.0 - mu_w) * g[:, mu_idx] + mu_w * g[:, mu_idx + 1]


def ufcn_yl(
    a: np.ndarray | float,
    e: np.ndarray | float,
    ivec: np.ndarray | float,
    vthetapi1: float,
    vthetax1: float,
    zetax1: float,
    zetae1: float,
    kappa: float,
    xvec: np.ndarray | float,
    pistar: float,
) -> np.ndarray:
    """Port of ufcnYL.m."""
    a_arr = np.asarray(a, dtype=float)
    e_arr = np.asarray(e, dtype=float)
    ivec_arr = np.asarray(ivec, dtype=float)
    xvec_arr = np.asarray(xvec, dtype=float)
    ecomb = e_arr + ivec_arr + kappa * xvec_arr
    return (
        -0.5 * (vthetapi1 + vthetax1 / (kappa**2)) * a_arr**2
        - zetae1 * e_arr
        + (vthetax1 / (kappa**2)) * ecomb * a_arr
        + (zetax1 / kappa) * (a_arr - ecomb)
        - 0.5 * (vthetax1 / (kappa**2)) * ecomb**2
        + vthetapi1 * pistar * a_arr
    )


def omega(a: np.ndarray, alpha: np.ndarray, z: float, rho: float, q: float) -> np.ndarray:
    """Port of omega.m."""
    if rho == 0:
        raise ValueError("omega undefined for rho==0; boundary state must be handled separately")
    return -((1.0 - q) * (a + (1.0 - rho) * alpha / rho) + q * z / rho)


def getmax(a: np.ndarray, atol: float = 0.0, rtol: float = 0.0) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Port of getmax.m returning MATLAB-like 1-based indices."""
    arr = np.asarray(a, dtype=float)
    maxval = float(np.max(arr))
    flat = arr.reshape(-1, order="F")
    if atol == 0.0 and rtol == 0.0:
        mask = flat == maxval
    else:
        mask = np.isclose(flat, maxval, atol=atol, rtol=rtol)
    idx0 = np.flatnonzero(mask)
    loc = idx0 + 1
    r_a, _ = arr.shape
    cloc = np.ceil(loc / r_a).astype(int)
    rloc = (loc - r_a * (cloc - 1)).astype(int)
    return maxval, loc, rloc, cloc
