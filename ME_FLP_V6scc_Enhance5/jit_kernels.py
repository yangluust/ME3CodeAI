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
from numba import njit, prange


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


@njit(cache=True, parallel=True, boundscheck=False)
def process_body_states_parallel(
    st_lo: int,
    st_hi: int,
    small_state: np.ndarray,
    small_lookup: np.ndarray,
    z_dr: np.ndarray,
    Mhat1: np.ndarray,
    Mhat2: np.ndarray,
    Uhat: np.ndarray,
    rho_idx_stack_1: np.ndarray,
    rho_w_stack_1: np.ndarray,
    rho_idx_stack_2: np.ndarray,
    rho_w_stack_2: np.ndarray,
    mu_idx: np.ndarray,
    mu_w: np.ndarray,
    fdens: np.ndarray,
    A2: float,
    B2: np.ndarray,
    d_delta_flat: np.ndarray,
    d_mup: np.ndarray,
    d_mu: np.ndarray,
    xvec1_flat: np.ndarray,
    bet: float,
    bet1: float,
    kappa: float,
    vtheta_pi1: float,
    vtheta_x1: float,
    zetax1: float,
    zetae1: float,
    pistar: float,
    q: float,
    optimizing: int,
    alpha0vec_flat: np.ndarray,
    has_zero_mu_global: bool,
    invalid_F_mask: np.ndarray,
    n_delta: int,
    n_mup: int,
    fast_path: bool,
    PTkeep: np.ndarray,
    EINFL: np.ndarray,
    a_dr_new: np.ndarray,
    alpha_dr_new: np.ndarray,
    einfl_dr_new: np.ndarray,
    muprime_dr_new: np.ndarray,
    Unew: np.ndarray,
    tie_record: np.ndarray,
    aic_buf: np.ndarray,
    einfl_buf: np.ndarray,
    om_st_f_buf: np.ndarray,
    urhs_f_buf: np.ndarray,
    save_intermediates: bool,
):
    """Parallel per-state body worker for the body block of the inner loop.

    Replaces the Python ``for st in range(n_squig, n_stend)`` loop in
    Phase 4's solver. Inlines the per-state numerics (omega, ufcn_yl,
    a_ic / alpha_ic build, om_st F-flat ravel) and reuses the existing
    ``ii_fhat_factored_kernel`` and ``fused_argmax_tie`` JIT helpers.

    Bit-exact parity with Phase 4 is required and enforced by
    ``parity_enhance4_vs_enhance5.py``. The mathematical operations are
    arranged in the same order and grouped the same way as Phase 4's
    NumPy expressions (e.g. ``einfl = bet * rho * ii_M1 + bet * (1-rho) * ii_M2``
    and ``urhs = ufcn_yl(...) + bet1*(1-q) * ii_U`` are kept as additions
    of two precomputed (n_delta, n_mup) tensors so that LLVM's per-element
    addition order matches NumPy's).

    `tie_record` is a per-(st, i_mu) bool array set to True for rows that
    had ``tie_count > 1``. The caller reconstructs ``MultipleMaximaEvent``
    objects serially from this record after the parallel block; PTkeep is
    NOT modified by the resolve step in Phase 4 (provably -- the
    `_unique_stable_indices` rule always picks the smallest tied F-flat,
    matching the default argmax).
    """
    n_mu = d_mu.shape[0]
    one_m_q = 1.0 - q
    bet1_1mq = bet1 * one_m_q
    c1 = -0.5 * (vtheta_pi1 + vtheta_x1 / (kappa * kappa))
    c2 = vtheta_x1 / (kappa * kappa)
    c3 = zetax1 / kappa
    c4 = -0.5 * (vtheta_x1 / (kappa * kappa))
    c5 = vtheta_pi1 * pistar

    for st in prange(st_lo, st_hi):
        squig_st = small_state[st, 0]
        rho_st = small_state[st, 1]
        i_squig = small_lookup[st, 0] - 1
        i_rhoi = small_lookup[st, 1] - 1
        zloc = z_dr[i_squig, i_rhoi]

        rho_idx_1 = rho_idx_stack_1[i_rhoi]
        rho_w_1 = rho_w_stack_1[i_rhoi]
        rho_idx_2 = rho_idx_stack_2[i_rhoi]
        rho_w_2 = rho_w_stack_2[i_rhoi]

        Mhat1_slice = Mhat1[i_squig]
        Mhat2_slice = Mhat2[i_squig]
        Uhat_slice = Uhat[i_squig]

        ii_M1 = ii_fhat_factored_kernel(
            Mhat1_slice, rho_idx_1, rho_w_1, mu_idx, mu_w, fdens
        )
        ii_M2 = ii_fhat_factored_kernel(
            Mhat2_slice, rho_idx_2, rho_w_2, mu_idx, mu_w, fdens
        )
        ii_U = ii_fhat_factored_kernel(
            Uhat_slice, rho_idx_1, rho_w_1, mu_idx, mu_w, fdens
        )

        # einfl = bet * rho * ii_M1 + bet * (1-rho) * ii_M2 (element-wise)
        einfl = np.empty((n_delta, n_mup), np.float64)
        for d in range(n_delta):
            for j in range(n_mup):
                einfl[d, j] = bet * rho_st * ii_M1[d, j] + bet * (1.0 - rho_st) * ii_M2[d, j]
                EINFL[d, j, st] = einfl[d, j]

        # alpha_ic, a_ic
        alpha_ic = np.empty((n_delta, n_mup), np.float64)
        a_ic = np.empty((n_delta, n_mup), np.float64)
        if optimizing == 1:
            B2_val = B2[i_squig]
            for d in range(n_delta):
                d_delta_d = d_delta_flat[d]
                for j in range(n_mup):
                    alpha_ic[d, j] = A2 * einfl[d, j] + B2_val
                    a_ic[d, j] = alpha_ic[d, j] + d_delta_d
        else:
            v = alpha0vec_flat[i_squig]
            for d in range(n_delta):
                d_delta_d = d_delta_flat[d]
                for j in range(n_mup):
                    alpha_ic[d, j] = v
                    a_ic[d, j] = v + d_delta_d

        # urhs = ufcn_yl(a_ic, einfl, squig_st, ...) + bet1*(1-q) * ii_U.
        # Reproduce NumPy ufcn_yl's element-wise evaluation order EXACTLY:
        # - `ecomb = e + ivec + kappa*xvec` evaluates per element as
        #   `(e[i,j] + ivec) + kxv` (left-associative + with kxv = kappa*xvec
        #   computed once). Pre-summing `ivec + kxv` and adding to `e` gives a
        #   sub-ULP-different result that accumulates across inner iterations.
        # - Parens on `a*a` / `ecomb*ecomb` match NumPy's `a**2` / `ecomb**2`
        #   precedence.
        kxv = kappa * xvec1_flat[i_squig]
        ivec_val = squig_st
        urhs = np.empty((n_delta, n_mup), np.float64)
        for d in range(n_delta):
            for j in range(n_mup):
                a_val = a_ic[d, j]
                e_val = einfl[d, j]
                ecomb = (e_val + ivec_val) + kxv
                u = (
                    c1 * (a_val * a_val)
                    - zetae1 * e_val
                    + c2 * ecomb * a_val
                    + c3 * (a_val - ecomb)
                    + c4 * (ecomb * ecomb)
                    + c5 * a_val
                )
                urhs[d, j] = u + bet1_1mq * ii_U[d, j]

        # om_st = -((1-q) * (a + (1-rho) * alpha / rho) + q * z / rho).
        # Match NumPy's left-to-right element-wise evaluation EXACTLY:
        # `(1.0 - rho) * alpha / rho` evaluates as `((1.0 - rho) * alpha) / rho`
        # per element. Pre-computing `(1-rho)/rho` and multiplying by alpha
        # later gives a different sub-ULP result.
        # `q * zloc / rho` are all scalars so left-to-right gives a single
        # scalar that is bit-equal to NumPy's broadcast version.
        q_z_over_rho = q * zloc / rho_st
        om_st = np.empty((n_delta, n_mup), np.float64)
        for d in range(n_delta):
            for j in range(n_mup):
                om_st[d, j] = -(
                    one_m_q * (a_ic[d, j] + (1.0 - rho_st) * alpha_ic[d, j] / rho_st)
                    + q_z_over_rho
                )

        # F-flat ravels of om_st and urhs (column-major: F = j * n_delta + d)
        N = n_delta * n_mup
        om_st_f = np.empty(N, np.float64)
        urhs_f = np.empty(N, np.float64)
        for j in range(n_mup):
            for d in range(n_delta):
                F = j * n_delta + d
                om_st_f[F] = om_st[d, j]
                urhs_f[F] = urhs[d, j]

        # Fused argmax + max + tie_count
        flat_argmax, max_per_mu, tie_count = fused_argmax_tie(
            d_mu, om_st_f, urhs_f, invalid_F_mask, has_zero_mu_global,
        )

        # Decode argmax to (rloc, cloc) and write PTkeep + record ties.
        # tie_record stores the tie count (0 if no tie or fast_path); the
        # caller uses non-zero entries to drive the serial post-pass that
        # reconstructs MultipleMaximaEvent objects.
        for i_mu in range(n_mu):
            loc_F = flat_argmax[i_mu] + 1
            cloc = (loc_F - 1) // n_delta + 1
            rloc = loc_F - n_delta * (cloc - 1)
            PTkeep[st, 0, i_mu] = rloc
            PTkeep[st, 1, i_mu] = cloc
            if (not fast_path) and tie_count[i_mu] > 1:
                tie_record[st, i_mu] = tie_count[i_mu]

        # When `save_intermediates` is true (cautious / startup iters), copy
        # a_ic, einfl, om_st_f, urhs_f into per-state buffers so the serial
        # event post-pass can reuse them without recomputing ii_fhat_factored.
        if save_intermediates:
            for d in range(n_delta):
                for j in range(n_mup):
                    aic_buf[st, d, j] = a_ic[d, j]
                    einfl_buf[st, d, j] = einfl[d, j]
            for F in range(N):
                om_st_f_buf[st, F] = om_st_f[F]
                urhs_f_buf[st, F] = urhs_f[F]

        # Final per-i_mu writes using the decoded PTkeep
        for i_mu in range(n_mu):
            r0 = PTkeep[st, 0, i_mu] - 1
            c0 = PTkeep[st, 1, i_mu] - 1
            a_dr_new[i_squig, i_rhoi, i_mu] = a_ic[r0, c0]
            alpha_dr_new[i_squig, i_rhoi, i_mu] = alpha_ic[r0, c0]
            einfl_dr_new[i_squig, i_rhoi, i_mu] = einfl[r0, c0]
            muprime_dr_new[i_squig, i_rhoi, i_mu] = d_mup[c0]
            Unew[i_squig, i_rhoi, i_mu] = urhs[r0, c0]


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

    # Warm up the parallel state worker with a 1-state run.
    n_squig = 1
    n_rho = 2
    n_mu_w = 2
    n_delta_w = 2
    n_mup_w = 2
    n_eps_w = 2
    n_st_w = n_squig * n_rho
    small_state_w = np.array([[0.1, 0.0], [0.1, 0.5]], dtype=np.float64)
    small_lookup_w = np.array([[1, 1], [1, 2]], dtype=np.int64)
    z_dr_w = np.zeros((n_squig, n_rho), dtype=np.float64)
    Mhat_w = np.zeros((n_squig, n_rho, n_mu_w), dtype=np.float64)
    rho_idx_stack_w = np.zeros((n_rho, n_eps_w, n_delta_w), dtype=np.int64)
    rho_w_stack_w = np.full((n_rho, n_eps_w, n_delta_w), 0.5, dtype=np.float64)
    mu_idx_w = np.zeros(n_mup_w, dtype=np.int64)
    mu_w_w = np.full(n_mup_w, 0.5, dtype=np.float64)
    fdens_w = np.array([0.5, 0.5], dtype=np.float64)
    A2 = 0.5
    B2_w = np.array([0.1], dtype=np.float64)
    d_delta_flat_w = np.array([-0.1, 0.1], dtype=np.float64)
    d_mup_w = np.array([0.0, 1.0], dtype=np.float64)
    d_mu_w = d_mup_w.copy()
    xvec1_flat_w = np.array([0.0], dtype=np.float64)
    alpha0vec_flat_w = np.array([0.1], dtype=np.float64)
    invalid_F_w = np.zeros(n_delta_w * n_mup_w, dtype=np.bool_)
    PTkeep_w = np.zeros((n_st_w, 2, n_mu_w), dtype=np.int64)
    EINFL_w = np.zeros((n_delta_w, n_mup_w, n_st_w), dtype=np.float64)
    a_dr_w = np.zeros((n_squig, n_rho, n_mu_w), dtype=np.float64)
    alpha_dr_w = np.zeros((n_squig, n_rho, n_mu_w), dtype=np.float64)
    einfl_dr_w = np.zeros((n_squig, n_rho, n_mu_w), dtype=np.float64)
    muprime_dr_w = np.zeros((n_squig, n_rho, n_mu_w), dtype=np.float64)
    Unew_w = np.zeros((n_squig, n_rho, n_mu_w), dtype=np.float64)
    tie_record_w = np.zeros((n_st_w, n_mu_w), dtype=np.int64)
    aic_buf_w = np.zeros((n_st_w, n_delta_w, n_mup_w), dtype=np.float64)
    einfl_buf_w = np.zeros((n_st_w, n_delta_w, n_mup_w), dtype=np.float64)
    om_st_f_buf_w = np.zeros((n_st_w, n_delta_w * n_mup_w), dtype=np.float64)
    urhs_f_buf_w = np.zeros((n_st_w, n_delta_w * n_mup_w), dtype=np.float64)
    process_body_states_parallel(
        n_squig, n_st_w,
        small_state_w, small_lookup_w, z_dr_w,
        Mhat_w, Mhat_w, Mhat_w,
        rho_idx_stack_w, rho_w_stack_w, rho_idx_stack_w, rho_w_stack_w,
        mu_idx_w, mu_w_w, fdens_w,
        A2, B2_w, d_delta_flat_w, d_mup_w, d_mu_w,
        xvec1_flat_w,
        0.99, 0.99, 1.0,
        1.0, 1.0, 0.0, 0.0, 0.0,
        1.0,
        1,
        alpha0vec_flat_w,
        False, invalid_F_w,
        n_delta_w, n_mup_w,
        False,
        PTkeep_w, EINFL_w,
        a_dr_w, alpha_dr_w, einfl_dr_w, muprime_dr_w, Unew_w,
        tie_record_w,
        aic_buf_w, einfl_buf_w, om_st_f_buf_w, urhs_f_buf_w,
        True,
    )
