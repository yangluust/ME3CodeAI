"""Numba-compiled inner-loop kernels for Phase 3.

These kernels reproduce, bit-for-bit, the NumPy expressions in Phase 2's
inner-loop block:

    flat[i_mu, F] = d_mu[i_mu] * om_st_f[F] + urhs_f[F]
    if d_mu[i_mu] == 0.0 and invalid_F_mask[F]: flat[i_mu, F] = -inf
    flat_argmax[i_mu] = argmax(flat[i_mu, :])      # first occurrence
    max_per_mu[i_mu]  = flat[i_mu, flat_argmax[i_mu]]
    tie_count[i_mu]   = sum(flat[i_mu, :] == max_per_mu[i_mu])

The fused kernel avoids materializing the (n_mu, N) flat buffer entirely
and replaces 5 full passes over flat with 2 streaming passes per row.
Bit-exactness vs the NumPy implementation is verified by
`scripts/temporary/parity_enhance2_vs_enhance3.py` at strict tolerance 0.

IMPORTANT:
  - We deliberately do not enable `fastmath` because that would license FMA
    contraction (mul+add -> fma) and re-association, breaking bit-exact
    parity with NumPy.
  - We split `tmp = mu * om_st_f[F]; v = tmp + urhs_f[F]` so the LLVM
    backend cannot legally contract into a single fma instruction.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, boundscheck=False)
def fused_argmax_tie(
    d_mu: np.ndarray,
    om_st_f: np.ndarray,
    urhs_f: np.ndarray,
    invalid_F_mask: np.ndarray,
    has_zero_mu: bool,
):
    """Fused per-row argmax + max + tie-count over flat[i_mu, F] = mu * om_st_f[F] + urhs_f[F].

    Parameters
    ----------
    d_mu : (n_mu,) float64
    om_st_f : (N,) float64, F-order ravel of om_st (N = n_delta * n_mup)
    urhs_f : (N,) float64, F-order ravel of urhs
    invalid_F_mask : (N,) bool, True where d_delta[i_delta] > 0 (only used for rows with mu == 0.0)
    has_zero_mu : bool, fast-skip flag; True iff any d_mu[i] == 0.0

    Returns
    -------
    flat_argmax : (n_mu,) int64, 0-based index of first maximum per row
    max_per_mu : (n_mu,) float64, value at the argmax
    tie_count : (n_mu,) int64, number of F where flat[i_mu, F] == max
    """
    n_mu = d_mu.shape[0]
    N = om_st_f.shape[0]
    flat_argmax = np.empty(n_mu, np.int64)
    max_per_mu = np.empty(n_mu, np.float64)
    tie_count = np.empty(n_mu, np.int64)

    for i_mu in range(n_mu):
        mu = d_mu[i_mu]
        is_zero = has_zero_mu and (mu == 0.0)

        best = -np.inf
        best_idx = 0
        if is_zero:
            for F in range(N):
                tmp = mu * om_st_f[F]
                v = tmp + urhs_f[F]
                if invalid_F_mask[F]:
                    v = -np.inf
                if v > best:
                    best = v
                    best_idx = F
        else:
            for F in range(N):
                tmp = mu * om_st_f[F]
                v = tmp + urhs_f[F]
                if v > best:
                    best = v
                    best_idx = F

        max_per_mu[i_mu] = best
        flat_argmax[i_mu] = best_idx

        c = 0
        if is_zero:
            for F in range(N):
                tmp = mu * om_st_f[F]
                v = tmp + urhs_f[F]
                if invalid_F_mask[F]:
                    v = -np.inf
                if v == best:
                    c += 1
        else:
            for F in range(N):
                tmp = mu * om_st_f[F]
                v = tmp + urhs_f[F]
                if v == best:
                    c += 1
        tie_count[i_mu] = c

    return flat_argmax, max_per_mu, tie_count


@njit(cache=True, boundscheck=False)
def collect_tied_F(
    mu: float,
    om_st_f: np.ndarray,
    urhs_f: np.ndarray,
    invalid_F_mask: np.ndarray,
    is_zero: bool,
    best: float,
    n_ties: int,
):
    """Return F-flat (1-based) indices where flat[i_mu, F] == best, in ascending order.

    Used only for rows with tie_count > 1, which is rare.
    """
    N = om_st_f.shape[0]
    out = np.empty(n_ties, np.int64)
    j = 0
    if is_zero:
        for F in range(N):
            tmp = mu * om_st_f[F]
            v = tmp + urhs_f[F]
            if invalid_F_mask[F]:
                v = -np.inf
            if v == best:
                out[j] = F + 1  # 1-based F-flat index
                j += 1
    else:
        for F in range(N):
            tmp = mu * om_st_f[F]
            v = tmp + urhs_f[F]
            if v == best:
                out[j] = F + 1
                j += 1
    return out


@njit(cache=True, boundscheck=False)
def ii_fhat_factored_kernel(
    f: np.ndarray,
    rho_idx: np.ndarray,
    rho_w: np.ndarray,
    mu_idx: np.ndarray,
    mu_w: np.ndarray,
    fdens: np.ndarray,
):
    """Fused bilinear-gather + epsilon integration + mu-axis bilinear interp.

    Mathematically equivalent to Phase 3's NumPy implementation:

        weight_lo = (1.0 - rho_w) * fdens[:, None]                 # (n_eps, n_delta)
        weight_hi =        rho_w  * fdens[:, None]
        f_lo      = f[rho_idx]                                     # (n_eps, n_delta, n_mu)
        f_hi      = f[rho_idx + 1]
        g         = einsum('ed,edm->dm', weight_lo, f_lo)
                  + einsum('ed,edm->dm', weight_hi, f_hi)
        out[:, j] = (1 - mu_w[j]) * g[:, mu_idx[j]] + mu_w[j] * g[:, mu_idx[j]+1]

    The kernel never materializes f_lo / f_hi (which can be 30+ MB per call)
    and never materializes g_lo / g_hi separately. It computes g[d, m] in two
    sequential loops over e (one for the lo term, one for the hi term, then
    sum) so the FP accumulation order matches Phase 3's two-einsum-then-add.

    fastmath stays off and `tmp = w * f; acc = acc + tmp` is split so LLVM
    cannot legally contract mul+add into FMA. Bit-exact parity vs Phase 3
    is enforced by `parity_enhance3_vs_enhance4.py`.
    """
    n_eps = rho_idx.shape[0]
    n_delta = rho_idx.shape[1]
    n_rho = f.shape[0]
    n_mu = f.shape[1]
    n_mup = mu_idx.shape[0]

    g = np.empty((n_delta, n_mu), np.float64)

    for d in range(n_delta):
        for m in range(n_mu):
            acc_lo = 0.0
            for e in range(n_eps):
                w_lo = (1.0 - rho_w[e, d]) * fdens[e]
                idx_lo = rho_idx[e, d]
                tmp = w_lo * f[idx_lo, m]
                acc_lo = acc_lo + tmp
            acc_hi = 0.0
            for e in range(n_eps):
                w_hi = rho_w[e, d] * fdens[e]
                idx_hi = rho_idx[e, d] + 1
                tmp = w_hi * f[idx_hi, m]
                acc_hi = acc_hi + tmp
            g[d, m] = acc_lo + acc_hi

    out = np.empty((n_delta, n_mup), np.float64)
    for d in range(n_delta):
        for j in range(n_mup):
            mi = mu_idx[j]
            mwj = mu_w[j]
            tmp_lo = (1.0 - mwj) * g[d, mi]
            tmp_hi = mwj * g[d, mi + 1]
            out[d, j] = tmp_lo + tmp_hi
    return out


def warmup() -> None:
    """Trigger AOT compilation of all kernels with a tiny representative call."""
    d_mu = np.array([0.0, 1.0], dtype=np.float64)
    om = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    ur = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    inv = np.array([False, True, False], dtype=np.bool_)
    fa, mp, tc = fused_argmax_tie(d_mu, om, ur, inv, True)
    _ = collect_tied_F(1.0, om, ur, inv, False, mp[1], int(tc[1]))

    f = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    rho_idx = np.array([[0, 1], [1, 0]], dtype=np.int64)
    rho_w = np.array([[0.25, 0.5], [0.75, 0.1]], dtype=np.float64)
    mu_idx = np.array([0], dtype=np.int64)
    mu_w = np.array([0.3], dtype=np.float64)
    fdens = np.array([0.4, 0.6], dtype=np.float64)
    _ = ii_fhat_factored_kernel(f, rho_idx, rho_w, mu_idx, mu_w, fdens)
