"""Deterministic parameter sweep focused on a_dr - alpha_dr vs rho."""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def _build_rho_grid(rho_version: str, n_rho: int) -> np.ndarray:
    if rho_version == "equal":
        return np.linspace(0.0, 1.0, n_rho, dtype=float)
    if rho_version == "lowdense":
        n_rholow = int((n_rho - 1) / 2)
        d_rholow = np.linspace(0.0, 0.1, n_rholow, dtype=float)
        d_rhorest = np.linspace(0.1, 0.99, n_rho - n_rholow, dtype=float)
        out = np.concatenate((d_rholow, d_rhorest[1:], np.array([1.0], dtype=float)))
        return out
    raise ValueError(f"Unsupported rho_version: {rho_version}")


def _cast_value(raw: str, base_value: Any) -> Any:
    if isinstance(base_value, bool):
        norm = raw.strip().lower()
        if norm in {"1", "true", "t", "yes", "y"}:
            return True
        if norm in {"0", "false", "f", "no", "n"}:
            return False
        raise ValueError(f"Cannot parse boolean value from '{raw}'")
    if isinstance(base_value, int):
        return int(raw)
    if isinstance(base_value, float):
        return float(raw)
    if isinstance(base_value, str):
        return raw
    raise TypeError(
        f"Parameter type for value '{base_value}' is unsupported for CLI casting: {type(base_value)}"
    )


def _safe_tag(value: Any) -> str:
    return str(value).replace(" ", "_").replace(".", "p").replace("-", "m")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic sweep for a_dr-alpha_dr vs rho")
    parser.add_argument("--workbook", type=Path, default=WORKSPACE_ROOT / "KLexperiments_ME_FLP_PD.xlsx")
    parser.add_argument("--run-no", type=int, default=51)
    parser.add_argument("--package", type=str, default="ME_FLP_V6scc_Enhance5")
    parser.add_argument("--param", type=str, required=True, help="RunConfig field name to vary")
    parser.add_argument(
        "--values",
        type=str,
        nargs="+",
        required=True,
        help="Explicit values to sweep, in evaluation order",
    )
    parser.add_argument("--out-loop-max", type=int, default=1)
    parser.add_argument("--in-loop-max", type=int, default=5)
    parser.add_argument("--svopt", type=int, choices=[0, 1], default=0)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--squig-rows", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--mu-cols", type=int, nargs="+", default=[1, 85, 168])
    parser.add_argument("--out-dir", type=Path, default=WORKSPACE_ROOT / "scripts" / "robustness")
    parser.add_argument("--tag", type=str, default="run51")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    config_mod = importlib.import_module(f"{args.package}.config")
    solver_mod = importlib.import_module(f"{args.package}.solver")
    load_run_config = config_mod.load_run_config
    SolverOptions = solver_mod.SolverOptions
    run_model = solver_mod.run_model

    base_cfg = load_run_config(args.workbook, run_no=args.run_no)
    field_names = {f.name for f in fields(type(base_cfg))}
    if args.param not in field_names:
        raise ValueError(
            f"Unknown parameter '{args.param}'. Available RunConfig fields: {sorted(field_names)}"
        )

    base_param_value = getattr(base_cfg, args.param)
    sweep_values = [_cast_value(raw=v, base_value=base_param_value) for v in args.values]
    rho_grid = _build_rho_grid(base_cfg.rho_version, base_cfg.n_rho)

    first_run_shape: tuple[int, int, int] | None = None
    runs: list[dict[str, Any]] = []
    for v in sweep_values:
        cfg = replace(
            base_cfg,
            **{
                args.param: v,
                "out_loop_max": int(args.out_loop_max),
                "in_loop_max": int(args.in_loop_max),
                "svopt": int(args.svopt),
            },
        )
        options_kwargs = {
            "make_plots": False,
            "workbook_path": args.workbook,
            "output_root": args.output_root,
        }
        result = run_model(cfg, SolverOptions(**options_kwargs))
        diff = np.asarray(result.a_dr, dtype=float) - np.asarray(result.alpha_dr, dtype=float)

        if diff.ndim != 3:
            raise ValueError(f"a_dr-alpha_dr must be 3D, got shape {diff.shape}")
        if first_run_shape is None:
            first_run_shape = diff.shape
        elif diff.shape != first_run_shape:
            raise ValueError(f"Shape mismatch across runs: {diff.shape} vs {first_run_shape}")

        runs.append({"value": v, "diff": diff})

    assert first_run_shape is not None
    n_squig, n_rho, n_mu = first_run_shape

    for idx in args.squig_rows:
        if idx < 1 or idx > n_squig:
            raise ValueError(f"squig index {idx} out of bounds, valid range is 1..{n_squig}")
    for idx in args.mu_cols:
        if idx < 1 or idx > n_mu:
            raise ValueError(f"mu index {idx} out of bounds, valid range is 1..{n_mu}")

    csv_path = args.out_dir / f"sweep_{args.param}_a_minus_alpha_vs_rho_{args.tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "param_name",
                "param_value",
                "squig_idx",
                "mu_idx",
                "rho_idx",
                "rho_value",
                "a_minus_alpha",
            ]
        )
        for run in runs:
            for squig_idx in args.squig_rows:
                for mu_idx in args.mu_cols:
                    for rho_idx in range(1, n_rho + 1):
                        writer.writerow(
                            [
                                args.param,
                                run["value"],
                                squig_idx,
                                mu_idx,
                                rho_idx,
                                float(rho_grid[rho_idx - 1]),
                                float(run["diff"][squig_idx - 1, rho_idx - 1, mu_idx - 1]),
                            ]
                        )
    print(f"wrote {csv_path}")

    fig, axes = plt.subplots(len(args.squig_rows), len(args.mu_cols), figsize=(12, 9))
    axes_2d = np.atleast_2d(axes)
    for r, squig_idx in enumerate(args.squig_rows):
        for c, mu_idx in enumerate(args.mu_cols):
            ax = axes_2d[r, c]
            for run in runs:
                curve = run["diff"][squig_idx - 1, :, mu_idx - 1]
                ax.plot(rho_grid, curve, label=f"{args.param}={run['value']}")
            ax.set_title(f"squig={squig_idx}, mu_idx={mu_idx}")
            ax.set_xlabel("rho")
            ax.set_ylabel("a_dr - alpha_dr")
            if r == 0 and c == 0:
                ax.legend()

    fig.tight_layout()
    png_path = args.out_dir / f"sweep_{args.param}_a_minus_alpha_vs_rho_{args.tag}.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
