"""Workbook-driven configuration loading for ME_FLP_V6scc."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class RunConfig:
    run_no: int
    user: str
    platform: str
    variant: str
    alternative: str
    # NK params
    vtheta_pi1: float
    vtheta_x1: float
    zetae1: float
    zetax1: float
    xstar1: float
    vtheta_pi2: float
    vtheta_x2: float
    zetae2: float
    zetax2: float
    xstar2: float
    bet1: float
    bet: float
    kappa: float
    pistar: float
    # Shocks
    sigma_1: float
    sigma_2: float
    # Replacement
    rho_intercept: float
    rho_slope: float
    qq: float
    # Loadings
    ivec_setting: str
    xvec1_setting: str
    xvec2_setting: str
    # Startup and loops
    initial_z: float
    out_loop_max: int
    in_loop_max: int
    # Grids
    n_squig: int
    squig_nu: float
    squig_sd: float
    rho_version: str
    n_rho: int
    rho_break: float | None
    mu_version: str
    n_mu: int | None
    x_upper: float
    x_lower: float
    x_step: float
    n_mup: int | None
    mup_version: str | None
    n_delta: int
    nsd_delta: float
    n_eps: int
    nsd_eps: float
    # switches
    go: int
    svopt: int


def _as_str(x: Any) -> str:
    return str(x).strip()


def _nan_to_none(x: Any) -> Any:
    if pd.isna(x):
        return None
    return x


def _row_by_case(df: pd.DataFrame, case_idx_1based: int) -> pd.Series:
    row_idx = case_idx_1based - 1
    if row_idx < 0 or row_idx >= len(df):
        raise ValueError(f"Case index {case_idx_1based} out of bounds for sheet")
    return df.iloc[row_idx]


def _read_all_sheets(workbook_path: Path) -> dict[str, pd.DataFrame]:
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")
    alias_map = {
        "Run_Info": ["Run_Info", "RUN_info"],
        "NK_params": ["NK_params"],
        "Shocks": ["Shocks"],
        "Replacement": ["Replacement"],
        "Loadings": ["Loadings"],
        "Startup": ["Startup", "StartUp"],
        "Loops": ["Loops"],
        "Grids": ["Grids"],
        "Accelerate": ["Accelerate"],
        "File_Storage": ["File_Storage"],
    }
    xls = pd.ExcelFile(workbook_path)
    available = set(xls.sheet_names)
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for canonical, aliases in alias_map.items():
        found = next((name for name in aliases if name in available), None)
        if found is None:
            missing.append(canonical)
        else:
            resolved[canonical] = found
    if missing:
        raise ValueError(f"Missing required workbook sheets (canonical names): {missing}; available={sorted(available)}")
    out: dict[str, pd.DataFrame] = {}
    for canonical, actual in resolved.items():
        out[canonical] = pd.read_excel(workbook_path, sheet_name=actual)
    return out


def _read_run_case_ids(run_row: pd.Series) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
    # MATLAB range F:O on Run_Info row gives these ten case selectors.
    vals: List[Any] = run_row.iloc[5:15].tolist()
    if len(vals) != 10:
        raise ValueError("Run_Info columns F:O are required")
    try:
        return tuple(int(v) for v in vals)  # type: ignore[return-value]
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Invalid case selector values in Run_Info F:O: {vals}") from exc


def load_run_config(workbook_path: str | Path, run_no: int = 51) -> RunConfig:
    """Load one run configuration from workbook, equivalent to LoadModel.m."""
    wb = Path(workbook_path)
    sheets = _read_all_sheets(wb)

    run_info = sheets["Run_Info"]
    run_idx = run_no - 1
    if run_idx < 0 or run_idx >= len(run_info):
        raise ValueError(f"run_no={run_no} out of bounds for Run_Info")
    run_row = run_info.iloc[run_idx]

    run_no_in_sheet = int(run_row.iloc[0])
    if run_no_in_sheet != run_no:
        raise ValueError(f"run_no mismatch: requested {run_no}, sheet has {run_no_in_sheet} in first column")

    user = _as_str(run_row.iloc[1])
    platform = _as_str(run_row.iloc[2])
    variant = _as_str(run_row.iloc[3])
    alternative = _as_str(run_row.iloc[4])

    (
        nk_case,
        shocks_case,
        replacement_case,
        loadings_case,
        startup_case,
        loops_case,
        grids_case,
        accelerate_case,
        _tuning_case,
        file_storage_case,
    ) = _read_run_case_ids(run_row)

    nk_row = _row_by_case(sheets["NK_params"], nk_case)
    nk_vals = nk_row.iloc[2:16].tolist()
    if len(nk_vals) != 14:
        raise ValueError("NK_params columns C:P are required")

    shocks_row = _row_by_case(sheets["Shocks"], shocks_case)
    sigma_1 = float(shocks_row.iloc[2])
    sigma_2 = float(shocks_row.iloc[3])

    replacement_row = _row_by_case(sheets["Replacement"], replacement_case)
    rho_intercept = float(replacement_row.iloc[3])
    rho_slope = float(replacement_row.iloc[4])
    qq = float(replacement_row.iloc[5])

    loadings_row = _row_by_case(sheets["Loadings"], loadings_case)
    ivec_setting = _as_str(loadings_row.iloc[1])
    xvec1_setting = _as_str(loadings_row.iloc[2])
    xvec2_setting = _as_str(loadings_row.iloc[3])

    startup_row = _row_by_case(sheets["Startup"], startup_case)
    initial_z = float(startup_row.iloc[2])

    loops_row = _row_by_case(sheets["Loops"], loops_case)
    out_loop_max = int(loops_row.iloc[1])
    in_loop_max = int(loops_row.iloc[2])

    grids_row = _row_by_case(sheets["Grids"], grids_case)
    g = grids_row.iloc[1:18].tolist()
    if len(g) != 17:
        raise ValueError("Grids columns B:R are required")

    accelerate_row = _row_by_case(sheets["Accelerate"], accelerate_case)
    go = int(accelerate_row.iloc[1])

    file_storage_row = _row_by_case(sheets["File_Storage"], file_storage_case)
    svopt = int(file_storage_row.iloc[1])

    return RunConfig(
        run_no=run_no,
        user=user,
        platform=platform,
        variant=variant,
        alternative=alternative,
        vtheta_pi1=float(nk_vals[0]),
        vtheta_x1=float(nk_vals[1]),
        zetae1=float(nk_vals[2]),
        zetax1=float(nk_vals[3]),
        xstar1=float(nk_vals[4]),
        vtheta_pi2=float(nk_vals[5]),
        vtheta_x2=float(nk_vals[6]),
        zetae2=float(nk_vals[7]),
        zetax2=float(nk_vals[8]),
        xstar2=float(nk_vals[9]),
        bet1=float(nk_vals[10]),
        bet=float(nk_vals[11]),
        kappa=float(nk_vals[12]),
        pistar=float(nk_vals[13]),
        sigma_1=sigma_1,
        sigma_2=sigma_2,
        rho_intercept=rho_intercept,
        rho_slope=rho_slope,
        qq=qq,
        ivec_setting=ivec_setting,
        xvec1_setting=xvec1_setting,
        xvec2_setting=xvec2_setting,
        initial_z=initial_z,
        out_loop_max=out_loop_max,
        in_loop_max=in_loop_max,
        n_squig=int(g[0]),
        squig_nu=float(g[1]),
        squig_sd=float(g[2]),
        rho_version=_as_str(g[3]),
        n_rho=int(g[4]),
        rho_break=float(g[5]) if _nan_to_none(g[5]) is not None else None,
        mu_version=_as_str(g[6]),
        n_mu=int(g[7]) if _nan_to_none(g[7]) is not None else None,
        x_upper=float(g[8]),
        x_lower=float(g[9]),
        x_step=float(g[10]),
        n_mup=int(g[11]) if _nan_to_none(g[11]) is not None else None,
        mup_version=_as_str(g[12]) if _nan_to_none(g[12]) is not None else None,
        n_delta=int(g[13]),
        nsd_delta=float(g[14]),
        n_eps=int(g[15]),
        nsd_eps=float(g[16]),
        go=go,
        svopt=svopt,
    )
