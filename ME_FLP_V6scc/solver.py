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
    br_normal,
    deltabound,
    get_mhat,
    getmax,
    ii_fhat,
    npdf,
    omega,
    onesz,
    rouwenhorst,
    ufcn_yl,
    vec,
)
from .diagnostics import BoundaryEvent, MultipleMaximaEvent, boundary_check, resolve_multiple_maxima
from .io_utils import create_run_dir, save_iteration_mat, save_mat, write_log_line
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
    d_delta = np.linspace(-config.nsd_delta * config.sigma_1, config.nsd_delta * config.sigma_1, config.n_delta).reshape(-1, 1)
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
    PTkeep = np.zeros((n_st, 2, n_mu), dtype=int)

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

    i_rho = 0
    rhoq1 = vec(rhoprime1[:, :, i_rho]).reshape(-1, 1) @ np.ones((1, n_mup))
    rhoq2 = vec(rhoprime2[:, :, i_rho]).reshape(-1, 1) @ np.ones((1, n_mup))
    mupq = np.ones((config.n_eps * config.n_delta, 1)) @ d_mup.reshape(1, -1)

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

            for st in range(config.n_squig, n_stend):
                squig_st = small_state[st, 0]
                rho_st = small_state[st, 1]
                i_squig = small_lookup[st, 0] - 1
                i_rhoi = small_lookup[st, 1] - 1
                zloc = z_dr[i_squig, i_rhoi]

                rhoq1 = vec(rhoprime1[:, :, i_rhoi]).reshape(-1, 1) @ np.ones((1, n_mup))
                rhoq2 = vec(rhoprime2[:, :, i_rhoi]).reshape(-1, 1) @ np.ones((1, n_mup))
                einfl = config.bet * rho_st * ii_fhat(
                    d_rho,
                    d_mu,
                    Mhat1[i_squig, :, :],
                    rhoq1,
                    mupq,
                    fdens,
                    config.n_eps,
                    config.n_delta,
                    n_mup,
                ) + config.bet * (1.0 - rho_st) * ii_fhat(
                    d_rho,
                    d_mu,
                    Mhat2[i_squig, :, :],
                    rhoq2,
                    mupq,
                    fdens,
                    config.n_eps,
                    config.n_delta,
                    n_mup,
                )
                EINFL[:, :, st] = einfl

                if optimizing == 1:
                    alpha_ic = A2 * einfl + B2[i_squig]
                else:
                    alpha_ic = np.full((config.n_delta, n_mup), float(alpha0vec[i_squig]), dtype=float)
                a_ic = alpha_ic + d_delta @ onesz(d_mup.reshape(1, -1))

                urhs = ufcn_yl(
                    a_ic,
                    einfl,
                    squig_st,
                    config.vtheta_pi1,
                    config.vtheta_x1,
                    config.zetax1,
                    config.zetae1,
                    config.kappa,
                    xvec1[i_squig],
                    config.pistar,
                ) + config.bet1 * (1.0 - q) * ii_fhat(
                    d_rho,
                    d_mu,
                    Uhat[i_squig, :, :],
                    rhoq1,
                    mupq,
                    fdens,
                    config.n_eps,
                    config.n_delta,
                    n_mup,
                )

                om_st = omega(a_ic, alpha_ic, zloc, rho_st, q)
                bugcount = 0
                for i_mu in range(n_mu):
                    mu_lp = d_mu[i_mu]
                    om_st_mu = mu_lp * om_st + urhs

                    if opt.cautious == 0 and in_loop > 2:
                        loc = int(np.argmax(om_st_mu.reshape(-1, order="F"))) + 1
                        cloc = int(np.ceil(loc / config.n_delta))
                        rloc = int(loc - (cloc - 1) * config.n_delta)
                        PTkeep[st, :, i_mu] = [rloc, cloc]
                    else:
                        if mu_lp == 0:
                            valid = (d_delta.reshape(-1) <= 0.0)
                            _, loc_raw, rloc_arr_raw, cloc_arr = getmax(
                                om_st_mu[valid, :], atol=opt.getmax_atol, rtol=opt.getmax_rtol
                            )
                            rloc_map = np.where(valid)[0] + 1
                            rloc_arr = rloc_map[rloc_arr_raw - 1]
                            if opt.strict_matlab_tie:
                                # MATLAB loc is defined on full OM_st_mu(:). Reconstruct full loc
                                # after the restricted-row search so tie cleanup sees full-matrix pointers.
                                loc = rloc_arr + config.n_delta * (cloc_arr - 1)
                            else:
                                loc = loc_raw
                        else:
                            _, loc, rloc_arr, cloc_arr = getmax(om_st_mu, atol=opt.getmax_atol, rtol=opt.getmax_rtol)
                        if len(rloc_arr) > 1:
                            bugcount, rloc, cloc, evt = resolve_multiple_maxima(
                                loc=loc,
                                aic=a_ic,
                                einfl=einfl,
                                n_delta=config.n_delta,
                                d_delta=d_delta.reshape(-1),
                                d_mup=d_mup.reshape(-1),
                                out_loop=out_loop,
                                in_loop=in_loop,
                                squig_st=float(squig_st),
                                rho_st=float(rho_st),
                                mu_lp=float(mu_lp),
                                i_mu=i_mu + 1,
                                bugcount=bugcount,
                                out_loop_max=config.out_loop_max,
                                in_loop_max=config.in_loop_max,
                                run_dir=run_dir,
                                make_plots=opt.make_plots,
                            )
                            if evt is not None:
                                multi_max_events.append(evt)
                        else:
                            rloc = int(rloc_arr[0])
                            cloc = int(cloc_arr[0])
                        PTkeep[st, :, i_mu] = [rloc, cloc]

                    if (
                        override_ptkeep is not None
                        and out_loop == opt.override_ptkeep_out_loop
                        and in_loop == opt.override_ptkeep_in_loop
                    ):
                        rloc = int(override_ptkeep[st, 0, i_mu])
                        cloc = int(override_ptkeep[st, 1, i_mu])
                        PTkeep[st, :, i_mu] = [rloc, cloc]

                    r0 = int(PTkeep[st, 0, i_mu] - 1)
                    c0 = int(PTkeep[st, 1, i_mu] - 1)
                    a_dr_new[i_squig, i_rhoi, i_mu] = a_ic[r0, c0]
                    alpha_dr_new[i_squig, i_rhoi, i_mu] = alpha_ic[r0, c0]
                    einfl_dr_new[i_squig, i_rhoi, i_mu] = einfl[r0, c0]
                    muprime_dr_new[i_squig, i_rhoi, i_mu] = d_mup[c0]
                    Unew[i_squig, i_rhoi, i_mu] = urhs[r0, c0]

                    if (
                        opt.debug_trace_state_1based is not None
                        and out_loop == opt.debug_trace_out_loop
                        and in_loop == opt.debug_trace_in_loop
                        and (i_squig + 1, i_rhoi + 1, i_mu + 1) == opt.debug_trace_state_1based
                    ):
                        write_debug_npz(
                            f"checkpoint_BC_state_s{i_squig+1}_r{i_rhoi+1}_m{i_mu+1}",
                            out_loop=np.array([[out_loop]]),
                            in_loop=np.array([[in_loop]]),
                            st_index=np.array([[st + 1]]),
                            mu_lp=np.array([[mu_lp]]),
                            om_st=om_st,
                            urhs=urhs,
                            om_st_mu=om_st_mu,
                            ptkeep_pair=PTkeep[st, :, i_mu].reshape(1, 2),
                            d_mup=d_mup.reshape(-1, 1),
                            chosen_muprime=np.array([[muprime_dr_new[i_squig, i_rhoi, i_mu]]]),
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
