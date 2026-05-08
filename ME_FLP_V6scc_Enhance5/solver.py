"""Main solver port for ME_FLP_V6scc.m."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.io import loadmat
from scipy.stats import beta as beta_dist

from .config import RunConfig
from .core_functions import (
    bilinear_axis_table,
    br_normal,
    deltabound,
    get_mhat,
    ii_fhat_factored,
    npdf,
    omega,
    onesz,
    rouwenhorst,
    ufcn_yl,
    vec,
)
from .diagnostics import BoundaryEvent, MultipleMaximaEvent, boundary_check, resolve_multiple_maxima
from .io_utils import create_run_dir, save_iteration_mat, save_mat, write_log_line
from .jit_kernels import (
    collect_tied_F,
    fused_argmax_tie,
    process_body_states_parallel,
    warmup as jit_warmup,
)
from .mu_bounds import compute_mu_bounds
from .rho_one import RhoOneCarry, solve_rho_one


@dataclass
class SolverOptions:
    cautious: int = 1
    flat_loud: int = 0
    make_plots: bool = False
    output_root: Path | None = None
    workbook_path: Path | None = None
    debug_output_dir: Path | None = None
    debug_trace_out_loop: int = 1
    debug_trace_in_loop: int = 1
    debug_trace_state_1based: tuple[int, int, int] | None = None
    override_d_mu_mat_path: Path | None = None
    override_ptkeep_mat_path: Path | None = None
    override_ptkeep_out_loop: int = 1
    override_ptkeep_in_loop: int = 1
    getmax_atol: float = 1e-16
    getmax_rtol: float = 0.0
    strict_matlab_tie: bool = False


@dataclass
class SolverResult:
    run_dir: Path | None
    run_name: str | None
    a_dr: np.ndarray
    alpha_dr: np.ndarray
    einfl_dr: np.ndarray
    muprime_dr: np.ndarray
    Unew: np.ndarray
    z_dr: np.ndarray
    Zupdates: np.ndarray
    Zchange: np.ndarray
    boundary_events: List[BoundaryEvent]
    multi_max_events: List[MultipleMaximaEvent]
    metadata: Dict[str, object]


def _build_rho_grid(rho_version: str, n_rho: int) -> np.ndarray:
    if rho_version == "equal":
        return np.linspace(0.0, 1.0, n_rho).reshape(-1, 1)
    if rho_version == "lowdense":
        n_rholow = int((n_rho - 1) / 2)
        d_rholow = np.linspace(0.0, 0.1, n_rholow).reshape(-1, 1)
        d_rhorest = np.linspace(0.1, 0.99, n_rho - n_rholow).reshape(-1, 1)
        out = np.vstack((d_rholow, d_rhorest[1:, :], np.array([[1.0]])))
        return out
    raise ValueError(f"Unsupported rho_version: {rho_version}")


def _parse_variant_and_alt(config: RunConfig) -> tuple[int, int, int]:
    if config.platform == "cluster":
        scc = 1
    elif config.platform == "desktop":
        scc = 0
    else:
        raise ValueError(f"Invalid PLATFORM value: {config.platform}")

    if config.variant == "endog_rho":
        endog_rho = 1
    elif config.variant == "exo_rho":
        endog_rho = 0
    else:
        raise ValueError(f"Invalid VARIANT value: {config.variant}")

    if config.alternative == "optimizing":
        optimizing = 1
    elif config.alternative == "mechanical":
        optimizing = 0
    else:
        raise ValueError(f"Invalid ALTERNATIVE value: {config.alternative}")

    return scc, endog_rho, optimizing


def run_model(config: RunConfig, options: SolverOptions | None = None) -> SolverResult:
    """Behavior-preserving translation of ME_FLP_V6scc.m core flow."""
    opt = options or SolverOptions()
    scc, endog_rho, optimizing = _parse_variant_and_alt(config)

    run_dir: Path | None = None
    run_name: str | None = None
    log_path: Path | None = None
    if config.svopt == 1:
        run_dir, run_name = create_run_dir(config, opt.output_root)
        log_path = run_dir / f"{run_name}.out"
        write_log_line(log_path, f"run_no={config.run_no}")

    def stage(msg: str) -> None:
        print(msg, flush=True)
        if log_path is not None:
            write_log_line(log_path, msg)

    debug_dir: Path | None = None
    if opt.debug_output_dir is not None:
        debug_dir = Path(opt.debug_output_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    def write_debug_npz(name: str, **arrays: object) -> None:
        if debug_dir is None:
            return
        np.savez(debug_dir / f"{name}.npz", **arrays)

    stage("stage: begin run_model")

    stage("stage: build exogenous and rho grids")
    d_squig, squig_trans = rouwenhorst(config.squig_nu, config.squig_sd * np.sqrt(1.0 - config.squig_nu**2), config.n_squig)
    d_rho = _build_rho_grid(config.rho_version, config.n_rho)

    kconst = (config.bet1 * (1.0 - config.qq)) / config.bet
    A1 = config.vtheta_x1 / (config.vtheta_pi1 * config.kappa**2 + config.vtheta_x1)
    A2 = config.vtheta_x2 / (config.vtheta_pi2 * config.kappa**2 + config.vtheta_x2)
    ivec = d_squig.copy()
    xvec1 = config.xstar1 * np.ones((config.n_squig, 1), dtype=float)
    xvec2 = config.xstar2 * np.ones((config.n_squig, 1), dtype=float)
    B1 = (
        A1 * (ivec + config.kappa * xvec1)
        + config.kappa * config.zetax1 / (config.vtheta_pi1 * config.kappa**2 + config.vtheta_x1)
        + (1.0 - A1) * config.pistar
    )
    B2 = (
        A2 * (ivec + config.kappa * xvec2)
        + config.kappa * config.zetax2 / (config.vtheta_pi2 * config.kappa**2 + config.vtheta_x2)
        + (1.0 - A2) * config.pistar
    )

    rhoevengrid = np.linspace(0.0, 1.0, 101)
    rhomean = config.rho_intercept / (1.0 - config.rho_slope)
    betapara1 = 3.0
    rhodist = beta_dist.pdf(rhoevengrid, betapara1, betapara1 * (1.0 - rhomean) / rhomean)
    rhodist = rhodist / np.sum(rhodist)

    if config.mu_version == "gam_fx":
        xgrid = np.arange(config.x_lower, config.xstar1 + config.x_step / 2.0, config.x_step).reshape(-1, 1)
        gamgrid = -(config.vtheta_x1 / config.kappa) * (xgrid - config.xstar1)
        d_mu = (gamgrid / kconst).reshape(-1)
        d_mu = np.sort(d_mu)
        i_muzero = int(np.argmin(np.abs(d_mu)))
        d_mu[i_muzero] = 1e-6
    elif config.mu_version == "gam_fx_quad":
        xgrid = np.arange(config.x_lower, config.xstar1 + config.x_step / 2.0, config.x_step).reshape(-1, 1)
        gamgrid = -(config.vtheta_x1 / config.kappa) * (xgrid - config.xstar1)
        d_mu = (gamgrid / kconst).reshape(-1)
        n_mu = d_mu.size
        pts = np.linspace(0.0, 1.0, n_mu)
        d_mu = np.min(d_mu) * pts + (np.max(d_mu) - np.min(d_mu)) * pts**2
        i_muzero = int(np.argmin(np.abs(d_mu)))
    else:
        raise ValueError(f"Unsupported mu_version: {config.mu_version}")

    stage("stage: start mu_bound")
    tmp_z = np.zeros((config.n_squig, config.n_rho), dtype=float)
    mub = compute_mu_bounds(
        d_mu=d_mu,
        n_mu=len(d_mu),
        i_muzero=i_muzero,
        qq=config.qq,
        n_squig=config.n_squig,
        z_dr=tmp_z,
        n_rho=config.n_rho,
        squig_trans=squig_trans,
        vtheta_x1=config.vtheta_x1,
        kappa=config.kappa,
        vtheta_pi1=config.vtheta_pi1,
        bet=config.bet,
        bet1=config.bet1,
        kconst=kconst,
        ivec=ivec,
        xvec1=xvec1,
        zetax1=config.zetax1,
        zetae1=config.zetae1,
        pistar=config.pistar,
        A2=A2,
        B2=B2.reshape(-1),
        progress_fn=stage,
    )
    stage("stage: completed mu_bound")
    d_mu = mub.d_mu.reshape(-1)
    i_muzero = mub.i_muzero
    n_mu = len(d_mu)

    if opt.override_d_mu_mat_path is not None:
        dm = loadmat(str(opt.override_d_mu_mat_path))
        if "d_mu" not in dm:
            raise ValueError(f"override_d_mu_mat_path missing d_mu: {opt.override_d_mu_mat_path}")
        d_mu_override = np.asarray(dm["d_mu"], dtype=float).reshape(-1)
        if d_mu_override.size != n_mu:
            raise ValueError(
                "override_d_mu size mismatch: "
                f"{d_mu_override.size} vs expected {n_mu}"
            )
        d_mu = d_mu_override.copy()
        i_muzero = int(np.argmin(np.abs(d_mu)))
        stage(
            "debug: override_d_mu active "
            f"(path={opt.override_d_mu_mat_path}, min={float(np.min(d_mu)):.6e}, max={float(np.max(d_mu)):.6e})"
        )

    d_mup = d_mu.copy()
    n_mup = len(d_mup)
    # Phase 3: warm up the JIT kernels once outside the timed loops so the
    # first inner iteration does not pay the AOT compilation cost.
    jit_warmup()
    d_delta = np.linspace(-config.nsd_delta * config.sigma_1, config.nsd_delta * config.sigma_1, config.n_delta).reshape(-1, 1)
    # Phase 3: precompute the F-flat invalid mask once. It is only consulted
    # for rows where d_mu[i_mu] == 0 (boundary case). When no row has
    # mu == 0 we set has_zero_mu_global = False and the kernel skips the
    # mask check entirely on the hot path.
    invalid_F_mask = np.zeros(config.n_delta * n_mup, dtype=np.bool_)
    has_zero_mu_global = bool(np.any(d_mu == 0.0))
    if has_zero_mu_global:
        invalid_delta_idx_pf = np.flatnonzero(d_delta.reshape(-1) > 0.0)
        if invalid_delta_idx_pf.size > 0:
            invalid_F_pf = (
                invalid_delta_idx_pf[:, None]
                + np.arange(n_mup)[None, :] * config.n_delta
            ).ravel()
            invalid_F_mask[invalid_F_pf] = True
    d_eps = np.linspace(-config.nsd_eps, config.nsd_eps, config.n_eps).reshape(-1, 1)
    fdens = npdf(d_eps, 0.0, 1.0, "trunc")

    squig_grid, rho_grid = np.meshgrid(d_squig.reshape(-1), d_rho.reshape(-1), indexing="ij")
    small_state = np.column_stack((vec(squig_grid), vec(rho_grid)))
    i_s, i_r = np.meshgrid(np.arange(1, config.n_squig + 1), np.arange(1, config.n_rho + 1), indexing="ij")
    small_lookup = np.column_stack((vec(i_s), vec(i_r))).astype(int)
    n_st = small_state.shape[0]

    z3 = np.zeros((config.n_squig, config.n_rho, n_mu), dtype=float)
    a_dr = z3.copy()
    alpha_dr = z3.copy()
    einfl_dr = z3.copy()
    a_dr_new = z3.copy()
    alpha_dr_new = z3.copy()
    einfl_dr_new = z3.copy()
    muprime_dr = z3.copy()
    muprime_dr_new = z3.copy()
    W = z3.copy()
    U = z3.copy()
    Uhat = z3.copy()
    Unew = z3.copy()
    Mhat1 = z3.copy()
    Mhat2 = z3.copy()
    z2 = np.zeros((config.n_squig, config.n_rho), dtype=float)
    z_dr = z2.copy()
    z_dr_new = z2.copy()
    Zchange = np.zeros((config.out_loop_max,), dtype=float)
    EINFL = np.zeros((config.n_delta, n_mup, config.n_squig * config.n_rho), dtype=float)
    Zupdates = np.zeros((config.n_squig, config.n_rho, config.out_loop_max + 1), dtype=float)
    PTkeep = np.zeros((n_st, 2, n_mu), dtype=np.int64)

    write_debug_npz(
        "checkpoint_A_post_mu_bound",
        d_mu=d_mu.reshape(-1, 1),
        d_mup=d_mup.reshape(-1, 1),
        i_muzero=np.array([[i_muzero + 1]]),
        n_mu=np.array([[n_mu]]),
        n_mup=np.array([[n_mup]]),
    )

    if config.svopt == 1 and run_dir is not None:
        initial_stuff: Dict[str, object] = {
            "run_no": np.array([[config.run_no]]),
            "PLATFORM": np.array([config.platform], dtype=object),
            "VARIANT": np.array([config.variant], dtype=object),
            "ALTERNATIVE": np.array([config.alternative], dtype=object),
            "scc": np.array([[scc]]),
            "endog_rho": np.array([[endog_rho]]),
            "optimizing": np.array([[optimizing]]),
            "kconst": np.array([[kconst]]),
            "A1": np.array([[A1]]),
            "A2": np.array([[A2]]),
            "B1": B1,
            "B2": B2,
            "d_squig": d_squig,
            "squig_trans": squig_trans,
            "d_rho": d_rho,
            "d_mu": d_mu.reshape(-1, 1),
            "d_mup": d_mup.reshape(-1, 1),
            "d_delta": d_delta,
            "d_eps": d_eps,
            "fdens": fdens,
            "small_state": small_state,
            "small_lookup": small_lookup,
            "i_muzero": np.array([[i_muzero + 1]]),
            "n_st": np.array([[n_st]]),
            "n_squig": np.array([[config.n_squig]]),
            "n_rho": np.array([[config.n_rho]]),
            "n_mu": np.array([[n_mu]]),
            "n_mup": np.array([[n_mup]]),
            "n_delta": np.array([[config.n_delta]]),
            "n_eps": np.array([[config.n_eps]]),
        }
        stuff_path = run_dir / "stuff.mat"
        save_mat(stuff_path, initial_stuff, append=False)
        stage(f"save: wrote {stuff_path.name} with initial snapshot keys={len(initial_stuff)}")

    rhoprime1 = np.zeros((config.n_eps, config.n_delta, config.n_rho), dtype=float)
    rhoprime2 = np.zeros_like(rhoprime1)
    if endog_rho == 1:
        stage("stage: precompute rhoprime arrays")
        for i_rho in range(config.n_rho):
            rhoprime1[:, :, i_rho] = br_normal(
                config.sigma_1 * d_eps,
                0.0 * d_delta,
                -d_delta,
                d_rho[i_rho],
                config.sigma_1,
                config.sigma_2,
            )
            rhoprime2[:, :, i_rho] = br_normal(
                config.sigma_2 * d_eps,
                d_delta,
                0.0 * d_delta,
                d_rho[i_rho],
                config.sigma_1,
                config.sigma_2,
            )
        _ = deltabound(rhoprime1, rhoprime2, fdens, config.n_rho)
        stage("stage: completed rhoprime precompute")

    stage("stage: build bilinear interp tables")
    mu_idx, mu_w = bilinear_axis_table(d_mu, d_mup)
    rho_tables_1: List[tuple[np.ndarray, np.ndarray]] = []
    rho_tables_2: List[tuple[np.ndarray, np.ndarray]] = []
    for i_rho in range(config.n_rho):
        rho_tables_1.append(bilinear_axis_table(d_rho, rhoprime1[:, :, i_rho]))
        rho_tables_2.append(bilinear_axis_table(d_rho, rhoprime2[:, :, i_rho]))
    stage("stage: completed bilinear interp tables")

    # Phase 5: stack rho tables once for the parallel state worker. Each
    # rho_tables_*[i_rho] entry is a (rho_idx, rho_w) pair of shape
    # (n_eps, n_delta). Stacking along a leading axis allows numba to take
    # contiguous slices `rho_idx_stack[i_rhoi]` inside the prange loop.
    rho_idx_stack_1 = np.ascontiguousarray(
        np.stack([t[0].astype(np.int64) for t in rho_tables_1], axis=0)
    )
    rho_w_stack_1 = np.ascontiguousarray(
        np.stack([t[1].astype(np.float64) for t in rho_tables_1], axis=0)
    )
    rho_idx_stack_2 = np.ascontiguousarray(
        np.stack([t[0].astype(np.int64) for t in rho_tables_2], axis=0)
    )
    rho_w_stack_2 = np.ascontiguousarray(
        np.stack([t[1].astype(np.float64) for t in rho_tables_2], axis=0)
    )
    mu_idx_int = np.ascontiguousarray(np.asarray(mu_idx, dtype=np.int64))
    mu_w_f64 = np.ascontiguousarray(np.asarray(mu_w, dtype=np.float64))
    fdens_flat = np.ascontiguousarray(np.asarray(fdens, dtype=np.float64).reshape(-1))
    B2_flat = np.ascontiguousarray(np.asarray(B2, dtype=np.float64).reshape(-1))
    d_delta_flat = np.ascontiguousarray(np.asarray(d_delta, dtype=np.float64).reshape(-1))
    d_mup_flat = np.ascontiguousarray(np.asarray(d_mup, dtype=np.float64).reshape(-1))
    d_mu_flat = np.ascontiguousarray(np.asarray(d_mu, dtype=np.float64).reshape(-1))
    xvec1_flat = np.ascontiguousarray(np.asarray(xvec1, dtype=np.float64).reshape(-1))
    small_state_c = np.ascontiguousarray(np.asarray(small_state, dtype=np.float64))
    small_lookup_c = np.ascontiguousarray(np.asarray(small_lookup, dtype=np.int64))

    z_dr = z_dr + config.initial_z
    Zupdates[:, :, 0] = z_dr
    boundary_events: List[BoundaryEvent] = []
    multi_max_events: List[MultipleMaximaEvent] = []
    rho_carry = mub.rho_one_carry if isinstance(mub.rho_one_carry, RhoOneCarry) else RhoOneCarry()
    override_ptkeep: np.ndarray | None = None
    if opt.override_ptkeep_mat_path is not None:
        pm = loadmat(str(opt.override_ptkeep_mat_path))
        if "PTkeep" not in pm:
            raise ValueError(f"override_ptkeep_mat_path missing PTkeep: {opt.override_ptkeep_mat_path}")
        override_ptkeep = np.asarray(pm["PTkeep"], dtype=int)
        if override_ptkeep.shape != PTkeep.shape:
            raise ValueError(
                "override PTkeep shape mismatch: "
                f"{override_ptkeep.shape} vs expected {PTkeep.shape}"
            )
        stage(f"debug: override PTkeep active from {opt.override_ptkeep_mat_path}")

    stage("stage: enter outer loop")
    for out_loop in range(1, config.out_loop_max + 1):
        stage(f"stage: out_loop={out_loop} start")
        a_dr_future = np.zeros_like(z3)
        alpha_dr_future = np.zeros_like(z3)
        Uhat = np.zeros_like(z3)

        for in_loop in range(1, config.in_loop_max + 1):
            if log_path is not None:
                write_log_line(log_path, f"BPE iteration={out_loop}; W iteration={in_loop}")
            if in_loop % 10 == 0:
                milestone_msg = f"InnerLoop milestone reached: out_loop={out_loop}, in_loop={in_loop}"
                print(milestone_msg, flush=True)
                if log_path is not None:
                    write_log_line(log_path, milestone_msg)
            q = 1.0 if in_loop == 1 else config.qq

            Mhat1 = get_mhat(a_dr_future, z_dr, squig_trans, q)
            Mhat2 = get_mhat(alpha_dr_future, z_dr, squig_trans, q)

            for st in range(0, config.n_squig):
                i_squig = small_lookup[st, 0] - 1
                i_rho0 = small_lookup[st, 1] - 1
                einfl0 = config.bet * (squig_trans[i_squig, :].reshape(1, -1) @ ((1.0 - q) * alpha_dr_future[:, i_rho0, 0] + q * z_dr[:, i_rho0]))
                einfl0 = float(einfl0.item())
                alpha0 = float((A2 * einfl0 + B2[i_squig]).item())
                a0 = float((A1 * einfl0 + B1[i_squig]).item())
                u0 = float(
                    ufcn_yl(
                        a0,
                        einfl0,
                        ivec[i_squig],
                        config.vtheta_pi1,
                        config.vtheta_x1,
                        config.zetax1,
                        config.zetae1,
                        config.kappa,
                        xvec1[i_squig],
                        config.pistar,
                    ).item()
                )
                a_dr_new[i_squig, i_rho0, :] = a0
                alpha_dr_new[i_squig, i_rho0, :] = alpha0
                einfl_dr_new[i_squig, i_rho0, :] = einfl0
                Unew[i_squig, i_rho0, :] = u0 + config.bet1 * (1.0 - q) * Uhat[i_squig, i_rho0, :]
                EINFL[:, :, st] = einfl0

            bz = config.bet * ((1.0 - config.qq) * B2 + config.qq * z_dr[:, 0].reshape(-1, 1))
            bzweight = np.linalg.solve(np.eye(config.n_squig) - config.bet * (1.0 - config.qq) * A2 * squig_trans, squig_trans)
            e0vec = bzweight @ bz
            alpha0vec = A2 * e0vec + B2

            if config.go == 1:
                rhores = solve_rho_one(
                    endog_rho=endog_rho,
                    z_dr=z_dr,
                    n_rho=config.n_rho,
                    q=q,
                    qq=config.qq,
                    d_mu=d_mu,
                    n_mu=n_mu,
                    n_squig=config.n_squig,
                    squig_trans=squig_trans,
                    vtheta_x1=config.vtheta_x1,
                    kappa=config.kappa,
                    vtheta_pi1=config.vtheta_pi1,
                    bet=config.bet,
                    bet1=config.bet1,
                    kconst=kconst,
                    ivec=ivec,
                    xvec1=xvec1,
                    zetax1=config.zetax1,
                    zetae1=config.zetae1,
                    pistar=config.pistar,
                    A2=A2,
                    B2=B2.reshape(-1),
                    carry=rho_carry,
                )
                rho_carry = rhores.carry
                mussvec = rhores.bmuvec_new.reshape(-1) / (1.0 - rhores.lamb_mumu_new)
                n_stend = n_st - config.n_squig
                for st in range(n_st - config.n_squig, n_st):
                    i_squig = small_lookup[st, 0] - 1
                    i_rho1 = small_lookup[st, 1] - 1
                    a_dr_new[i_squig, i_rho1, :] = rhores.a1mat[i_squig, :]
                    alpha_dr_new[i_squig, i_rho1, :] = rhores.alpha1mat[i_squig, :]
                    einfl_dr_new[i_squig, i_rho1, :] = rhores.e1mat[i_squig, :]
                    muprime_dr_new[i_squig, i_rho1, :] = rhores.mup1mat[i_squig, :]
                    Unew[i_squig, i_rho1, :] = rhores.U1mat[i_squig, :]
                    EINFL[:, :, st] = np.repeat(rhores.e1mat[i_squig, :].reshape(1, -1), config.n_delta, axis=0)
            else:
                mussvec = None
                n_stend = n_st

            # Phase 5: parallel @njit body-block worker. Replaces the
            # per-state Python loop and the per-state JIT-call dispatch
            # overhead with a single prange loop in compiled code that
            # writes directly to PTkeep / EINFL / *_dr_new / Unew.
            #
            # The resolve_multiple_maxima call from Phase 4 is provably a
            # no-op for PTkeep (the dedup rule always picks the smallest
            # tied F-flat, equal to the default argmax+1). So the parallel
            # kernel skips that call entirely; we reconstruct the
            # diagnostic MultipleMaximaEvent objects in a serial post-pass
            # below using a per-(st, i_mu) tie_count record.
            fast_path_flag = bool(opt.cautious == 0 and in_loop > 2)
            # Save per-state intermediates (a_ic, einfl, om_st_f, urhs_f)
            # only when the post-pass might need them, i.e. when tie
            # detection is active. In fast_path iters there is no post-pass
            # so we pass dummy 1-row buffers.
            tie_record = np.zeros((n_st, n_mu), dtype=np.int64)
            if fast_path_flag:
                aic_buf = np.empty((1, config.n_delta, n_mup), dtype=np.float64)
                einfl_buf = np.empty((1, config.n_delta, n_mup), dtype=np.float64)
                om_st_f_buf = np.empty((1, config.n_delta * n_mup), dtype=np.float64)
                urhs_f_buf = np.empty((1, config.n_delta * n_mup), dtype=np.float64)
                save_intermediates = False
            else:
                aic_buf = np.empty((n_st, config.n_delta, n_mup), dtype=np.float64)
                einfl_buf = np.empty((n_st, config.n_delta, n_mup), dtype=np.float64)
                om_st_f_buf = np.empty((n_st, config.n_delta * n_mup), dtype=np.float64)
                urhs_f_buf = np.empty((n_st, config.n_delta * n_mup), dtype=np.float64)
                save_intermediates = True

            B2_squigvec = np.ascontiguousarray(np.asarray(B2, dtype=np.float64).reshape(-1))
            alpha0vec_flat = np.ascontiguousarray(
                np.asarray(alpha0vec, dtype=np.float64).reshape(-1)
            )
            Mhat1_c = np.ascontiguousarray(np.asarray(Mhat1, dtype=np.float64))
            Mhat2_c = np.ascontiguousarray(np.asarray(Mhat2, dtype=np.float64))
            Uhat_c = np.ascontiguousarray(np.asarray(Uhat, dtype=np.float64))
            z_dr_c = np.ascontiguousarray(np.asarray(z_dr, dtype=np.float64))

            process_body_states_parallel(
                int(config.n_squig), int(n_stend),
                small_state_c, small_lookup_c, z_dr_c,
                Mhat1_c, Mhat2_c, Uhat_c,
                rho_idx_stack_1, rho_w_stack_1,
                rho_idx_stack_2, rho_w_stack_2,
                mu_idx_int, mu_w_f64, fdens_flat,
                float(A2), B2_squigvec,
                d_delta_flat, d_mup_flat, d_mu_flat,
                xvec1_flat,
                float(config.bet), float(config.bet1), float(config.kappa),
                float(config.vtheta_pi1), float(config.vtheta_x1),
                float(config.zetax1), float(config.zetae1), float(config.pistar),
                float(q),
                int(optimizing),
                alpha0vec_flat,
                bool(has_zero_mu_global), invalid_F_mask,
                int(config.n_delta), int(n_mup),
                fast_path_flag,
                PTkeep, EINFL,
                a_dr_new, alpha_dr_new, einfl_dr_new, muprime_dr_new, Unew,
                tie_record,
                aic_buf, einfl_buf, om_st_f_buf, urhs_f_buf,
                save_intermediates,
            )

            # Serial post-pass: override_ptkeep, debug_trace, and event
            # reconstruction for tied states. Tied states are rare in
            # practice (~7 events per 3x50 production run) so the cost is
            # negligible.
            if (
                override_ptkeep is not None
                and out_loop == opt.override_ptkeep_out_loop
                and in_loop == opt.override_ptkeep_in_loop
            ):
                for st in range(config.n_squig, n_stend):
                    PTkeep[st, 0, :] = override_ptkeep[st, 0, :]
                    PTkeep[st, 1, :] = override_ptkeep[st, 1, :]
                    i_squig = small_lookup[st, 0] - 1
                    i_rhoi = small_lookup[st, 1] - 1
                    # Re-derive a_dr_new etc. from the overridden PTkeep.
                    # We need a_ic / alpha_ic / einfl / urhs for this st;
                    # recompute them here (rare path).
                    rho_st = small_state[st, 1]
                    rho_idx_1, rho_w_1 = rho_tables_1[i_rhoi]
                    rho_idx_2, rho_w_2 = rho_tables_2[i_rhoi]
                    einfl = config.bet * rho_st * ii_fhat_factored(
                        f=Mhat1[i_squig, :, :], rho_idx=rho_idx_1, rho_w=rho_w_1,
                        mu_idx=mu_idx, mu_w=mu_w, fdens=fdens,
                    ) + config.bet * (1.0 - rho_st) * ii_fhat_factored(
                        f=Mhat2[i_squig, :, :], rho_idx=rho_idx_2, rho_w=rho_w_2,
                        mu_idx=mu_idx, mu_w=mu_w, fdens=fdens,
                    )
                    if optimizing == 1:
                        alpha_ic = A2 * einfl + B2[i_squig]
                    else:
                        alpha_ic = np.full((config.n_delta, n_mup), float(alpha0vec[i_squig]), dtype=float)
                    a_ic = alpha_ic + d_delta @ onesz(d_mup.reshape(1, -1))
                    urhs = ufcn_yl(
                        a_ic, einfl, small_state[st, 0],
                        config.vtheta_pi1, config.vtheta_x1, config.zetax1,
                        config.zetae1, config.kappa, xvec1[i_squig], config.pistar,
                    ) + config.bet1 * (1.0 - q) * ii_fhat_factored(
                        f=Uhat[i_squig, :, :], rho_idx=rho_idx_1, rho_w=rho_w_1,
                        mu_idx=mu_idx, mu_w=mu_w, fdens=fdens,
                    )
                    r0_all = PTkeep[st, 0, :].astype(int) - 1
                    c0_all = PTkeep[st, 1, :].astype(int) - 1
                    a_dr_new[i_squig, i_rhoi, :] = a_ic[r0_all, c0_all]
                    alpha_dr_new[i_squig, i_rhoi, :] = alpha_ic[r0_all, c0_all]
                    einfl_dr_new[i_squig, i_rhoi, :] = einfl[r0_all, c0_all]
                    muprime_dr_new[i_squig, i_rhoi, :] = d_mup[c0_all]
                    Unew[i_squig, i_rhoi, :] = urhs[r0_all, c0_all]

            # Reconstruct MultipleMaximaEvent diagnostics for tied states.
            # The parallel kernel saved the per-state a_ic / einfl / om_st_f /
            # urhs_f intermediates in `*_buf` arrays (cautious / startup iters
            # only). The post-pass reads them directly without recomputing
            # ii_fhat_factored.
            if save_intermediates:
                tied_states_mask = (tie_record != 0).any(axis=1)
                tied_st_indices = np.flatnonzero(tied_states_mask)
                for st in tied_st_indices:
                    if st < config.n_squig or st >= n_stend:
                        continue
                    squig_st = float(small_state[st, 0])
                    rho_st = float(small_state[st, 1])
                    a_ic_st = aic_buf[st]
                    einfl_st = einfl_buf[st]
                    om_st_f_post = om_st_f_buf[st]
                    urhs_f_post = urhs_f_buf[st]
                    vaic_state = a_ic_st.reshape(-1, order="F")
                    veinfl_state = einfl_st.reshape(-1, order="F")
                    bugcount = 0
                    tied_i_mus = np.flatnonzero(tie_record[st, :] > 0)
                    for i_mu_int in tied_i_mus:
                        i_mu = int(i_mu_int)
                        mu_val = float(d_mu[i_mu])
                        is_zero_row = has_zero_mu_global and (mu_val == 0.0)
                        n_ties = int(tie_record[st, i_mu])
                        r0 = int(PTkeep[st, 0, i_mu]) - 1
                        c0 = int(PTkeep[st, 1, i_mu]) - 1
                        F_chosen = c0 * config.n_delta + r0
                        max_val = mu_val * om_st_f_post[F_chosen] + urhs_f_post[F_chosen]
                        tied_F_flat = collect_tied_F(
                            mu_val, om_st_f_post, urhs_f_post, invalid_F_mask,
                            is_zero_row, max_val, n_ties,
                        )
                        bugcount, _r, _c, evt = resolve_multiple_maxima(
                            loc=tied_F_flat, aic=a_ic_st, einfl=einfl_st,
                            n_delta=config.n_delta,
                            d_delta=d_delta.reshape(-1), d_mup=d_mup.reshape(-1),
                            out_loop=out_loop, in_loop=in_loop,
                            squig_st=squig_st, rho_st=rho_st,
                            mu_lp=mu_val, i_mu=i_mu + 1,
                            bugcount=bugcount,
                            out_loop_max=config.out_loop_max,
                            in_loop_max=config.in_loop_max,
                            run_dir=run_dir, make_plots=opt.make_plots,
                            vaic=vaic_state, veinfl=veinfl_state,
                        )
                        if evt is not None:
                            multi_max_events.append(evt)

            # Optional debug_trace_state_1based dump (rare path).
            if (
                opt.debug_trace_state_1based is not None
                and out_loop == opt.debug_trace_out_loop
                and in_loop == opt.debug_trace_in_loop
            ):
                target_squig_1, target_rhoi_1, target_mu_1 = opt.debug_trace_state_1based
                target_st = -1
                for st_search in range(config.n_squig, n_stend):
                    if (
                        small_lookup[st_search, 0] == target_squig_1
                        and small_lookup[st_search, 1] == target_rhoi_1
                    ):
                        target_st = st_search
                        break
                target_i_mu = int(target_mu_1) - 1
                if target_st >= 0 and 0 <= target_i_mu < n_mu:
                    squig_st_dbg = small_state[target_st, 0]
                    rho_st_dbg = small_state[target_st, 1]
                    i_squig_dbg = small_lookup[target_st, 0] - 1
                    i_rhoi_dbg = small_lookup[target_st, 1] - 1
                    zloc_dbg = z_dr[i_squig_dbg, i_rhoi_dbg]
                    rho_idx_1_dbg, rho_w_1_dbg = rho_tables_1[i_rhoi_dbg]
                    rho_idx_2_dbg, rho_w_2_dbg = rho_tables_2[i_rhoi_dbg]
                    einfl_dbg = config.bet * rho_st_dbg * ii_fhat_factored(
                        f=Mhat1[i_squig_dbg, :, :], rho_idx=rho_idx_1_dbg, rho_w=rho_w_1_dbg,
                        mu_idx=mu_idx, mu_w=mu_w, fdens=fdens,
                    ) + config.bet * (1.0 - rho_st_dbg) * ii_fhat_factored(
                        f=Mhat2[i_squig_dbg, :, :], rho_idx=rho_idx_2_dbg, rho_w=rho_w_2_dbg,
                        mu_idx=mu_idx, mu_w=mu_w, fdens=fdens,
                    )
                    if optimizing == 1:
                        alpha_ic_dbg = A2 * einfl_dbg + B2[i_squig_dbg]
                    else:
                        alpha_ic_dbg = np.full((config.n_delta, n_mup), float(alpha0vec[i_squig_dbg]), dtype=float)
                    a_ic_dbg = alpha_ic_dbg + d_delta @ onesz(d_mup.reshape(1, -1))
                    urhs_dbg = ufcn_yl(
                        a_ic_dbg, einfl_dbg, squig_st_dbg,
                        config.vtheta_pi1, config.vtheta_x1, config.zetax1,
                        config.zetae1, config.kappa, xvec1[i_squig_dbg], config.pistar,
                    ) + config.bet1 * (1.0 - q) * ii_fhat_factored(
                        f=Uhat[i_squig_dbg, :, :], rho_idx=rho_idx_1_dbg, rho_w=rho_w_1_dbg,
                        mu_idx=mu_idx, mu_w=mu_w, fdens=fdens,
                    )
                    om_st_dbg = omega(a_ic_dbg, alpha_ic_dbg, zloc_dbg, rho_st_dbg, q)
                    om_st_mu_dbg = d_mu[target_i_mu] * om_st_dbg + urhs_dbg
                    write_debug_npz(
                        f"checkpoint_BC_state_s{i_squig_dbg+1}_r{i_rhoi_dbg+1}_m{target_i_mu+1}",
                        out_loop=np.array([[out_loop]]),
                        in_loop=np.array([[in_loop]]),
                        st_index=np.array([[target_st + 1]]),
                        mu_lp=np.array([[d_mu[target_i_mu]]]),
                        om_st=om_st_dbg,
                        urhs=urhs_dbg,
                        om_st_mu=om_st_mu_dbg,
                        ptkeep_pair=PTkeep[target_st, :, target_i_mu].reshape(1, 2),
                        d_mup=d_mup.reshape(-1, 1),
                        chosen_muprime=np.array([[muprime_dr_new[i_squig_dbg, i_rhoi_dbg, target_i_mu]]]),
                    )

            boundary_events.extend(
                boundary_check(
                    ptkeep=PTkeep,
                    n_squig=config.n_squig,
                    n_st=n_st,
                    n_delta=config.n_delta,
                    n_mup=n_mup,
                    d_delta=d_delta.reshape(-1),
                    d_mup=d_mup.reshape(-1),
                    make_plots=opt.make_plots,
                )
            )

            a_dr = a_dr_new.copy()
            alpha_dr = alpha_dr_new.copy()
            einfl_dr = einfl_dr_new.copy()
            muprime_dr = muprime_dr_new.copy()
            Uhat = (squig_trans @ Unew.reshape((config.n_squig, config.n_rho * n_mu), order="F")).reshape(
                (config.n_squig, config.n_rho, n_mu), order="F"
            )
            a_dr_future = a_dr.copy()
            alpha_dr_future = alpha_dr.copy()

            if config.svopt == 1 and run_dir is not None:
                payload: Dict[str, object] = {
                    "a_dr": a_dr,
                    "alpha_dr": alpha_dr,
                    "einfl_dr": einfl_dr,
                    "muprime_dr": muprime_dr,
                    "Unew": Unew,
                    "EINFL": EINFL,
                    "PTkeep": PTkeep,
                }
                if mussvec is not None:
                    payload["mussvec"] = mussvec.reshape(-1, 1)
                save_iteration_mat(run_dir=run_dir, out_loop=out_loop, in_loop=in_loop, arrays=payload)
                stage(f"save: wrote MPE{out_loop}W{in_loop}.mat keys={sorted(payload.keys())}")

        if endog_rho == 1:
            for j in range(config.n_squig):
                a_line = a_dr[j, :, i_muzero]
                alpha_line = alpha_dr[j, :, i_muzero]
                a_interp = np.interp(rhoevengrid, d_rho.reshape(-1), a_line)
                alpha_interp = np.interp(rhoevengrid, d_rho.reshape(-1), alpha_line)
                z_dr_new_random = float(np.sum((rhoevengrid * a_interp + (1.0 - rhoevengrid) * alpha_interp) * rhodist))
                z_dr_new_inherit = d_rho.reshape(-1) * a_line + (1.0 - d_rho.reshape(-1)) * alpha_line
                z_dr_new[j, :] = config.rho_slope * z_dr_new_inherit + (1.0 - config.rho_slope) * z_dr_new_random
        else:
            z_dr_new = a_dr[:, :, i_muzero] * d_rho.reshape(1, -1) + alpha_dr[:, :, i_muzero] * (1.0 - d_rho.reshape(1, -1))

        Zupdates[:, :, out_loop] = z_dr_new
        Zchange[out_loop - 1] = float(np.max(np.abs(z_dr - z_dr_new)))
        z_dr = z_dr_new.copy()

    if config.svopt == 1 and run_dir is not None:
        d_mu_save = d_mu.copy()
        d_mu_save[i_muzero] = 0.0
        stuff_path = run_dir / "stuff.mat"
        save_mat(stuff_path, {"d_mu": d_mu_save.reshape(-1, 1)}, append=True)
        stage("save: appended d_mu to stuff.mat")
        saved = loadmat(str(stuff_path))
        if "d_mu" not in saved:
            raise ValueError("stuff.mat append verification failed: missing d_mu key")
        saved_d_mu = np.asarray(saved["d_mu"], dtype=float).reshape(-1)
        if saved_d_mu.shape != d_mu_save.shape:
            raise ValueError(
                "stuff.mat append verification failed: "
                f"shape mismatch {saved_d_mu.shape} vs {d_mu_save.shape}"
            )
        max_abs = float(np.max(np.abs(saved_d_mu - d_mu_save)))
        if max_abs != 0.0:
            raise ValueError(
                "stuff.mat append verification failed: "
                f"d_mu mismatch after append (max_abs_diff={max_abs})"
            )
        stage("save: verified stuff.mat d_mu append parity")
        save_mat(run_dir / f"MPE{config.out_loop_max}Zinfo.mat", {"Zupdates": Zupdates}, append=False)
        stage(f"save: wrote MPE{config.out_loop_max}Zinfo.mat")

    meta: Dict[str, object] = {
        "scc": scc,
        "endog_rho": endog_rho,
        "optimizing": optimizing,
        "i_muzero_python0": i_muzero,
        "run_no": config.run_no,
    }
    return SolverResult(
        run_dir=run_dir,
        run_name=run_name,
        a_dr=a_dr,
        alpha_dr=alpha_dr,
        einfl_dr=einfl_dr,
        muprime_dr=muprime_dr,
        Unew=Unew,
        z_dr=z_dr,
        Zupdates=Zupdates,
        Zchange=Zchange,
        boundary_events=boundary_events,
        multi_max_events=multi_max_events,
        metadata=meta,
    )
