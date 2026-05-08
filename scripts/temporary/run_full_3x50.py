"""Production-style full run at out=3, in=50.

Loads run-51 config, overrides loop maxes, keeps svopt=1 so MPE/stuff
.mat files are written to the default YL/desktop output root. After the
run finishes, runs the in-package parity check against the MATLAB
reference at the user-supplied directory.

`--package` selects which Python port to drive (Phase 1 = ME_FLP_V6scc_Enhance,
Phase 2 = ME_FLP_V6scc_Enhance2, Phase 3 = ME_FLP_V6scc_Enhance3,
Phase 4 = ME_FLP_V6scc_Enhance4, Phase 5 = ME_FLP_V6scc_Enhance5).
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from dataclasses import replace
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="Run 3x50 production then parity-check.")
    p.add_argument("--workbook", type=Path, default=WORKSPACE_ROOT / "KLexperiments_ME_FLP_PD.xlsx")
    p.add_argument("--run-no", type=int, default=51)
    p.add_argument("--out-loop-max", type=int, default=3)
    p.add_argument("--in-loop-max", type=int, default=50)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument(
        "--package",
        type=str,
        default="ME_FLP_V6scc_Enhance",
        choices=("ME_FLP_V6scc_Enhance", "ME_FLP_V6scc_Enhance2", "ME_FLP_V6scc_Enhance3", "ME_FLP_V6scc_Enhance4", "ME_FLP_V6scc_Enhance5"),
        help="Which enhanced Python package to drive.",
    )
    p.add_argument(
        "--matlab-dir",
        type=Path,
        default=Path(r"C:\Users\yanglu\Dropbox\ME_FLP_STORAGE\YLExperiment_51_20260504T223305"),
    )
    p.add_argument("--parity-out-loop", type=int, default=3)
    p.add_argument("--parity-in-loop", type=int, default=50)
    args = p.parse_args()

    config_mod = importlib.import_module(f"{args.package}.config")
    parity_mod = importlib.import_module(f"{args.package}.parity_check")
    solver_mod = importlib.import_module(f"{args.package}.solver")
    load_run_config = config_mod.load_run_config
    check_against_matlab = parity_mod.check_against_matlab
    SolverOptions = solver_mod.SolverOptions
    run_model = solver_mod.run_model

    print(f"[driver] package={args.package}")
    cfg = load_run_config(args.workbook, run_no=args.run_no)
    cfg = replace(cfg, out_loop_max=args.out_loop_max, in_loop_max=args.in_loop_max)
    print(
        f"config: run_no={cfg.run_no} n_squig={cfg.n_squig} n_rho={cfg.n_rho} "
        f"n_delta={cfg.n_delta} n_eps={cfg.n_eps} "
        f"out_loop_max={cfg.out_loop_max} in_loop_max={cfg.in_loop_max} svopt={cfg.svopt}"
    )

    options = SolverOptions(make_plots=False, output_root=args.output_root, workbook_path=args.workbook)

    t0 = time.perf_counter()
    res = run_model(cfg, options)
    elapsed = time.perf_counter() - t0

    print(f"\n[run_model] elapsed: {elapsed:.3f} s ({elapsed / 60.0:.2f} min)")
    print(f"[run_model] run_dir: {res.run_dir}")
    print(f"[run_model] boundary_events={len(res.boundary_events)}, multi_max_events={len(res.multi_max_events)}")

    print(
        f"\n[parity] comparing against MATLAB MPE{args.parity_out_loop}W{args.parity_in_loop}.mat in:"
    )
    print(f"  {args.matlab_dir}")
    summary = check_against_matlab(
        result=res,
        matlab_dir=args.matlab_dir,
        out_loop=args.parity_out_loop,
        in_loop=args.parity_in_loop,
    )
    print(f"\n[parity] compared_files={summary.compared_files}")
    width = max(len(k) for k in summary.metrics)
    for k, v in summary.metrics.items():
        print(f"  {k.ljust(width)} : {v:.6e}")


if __name__ == "__main__":
    main()
