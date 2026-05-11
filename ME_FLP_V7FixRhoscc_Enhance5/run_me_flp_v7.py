"""CLI entrypoint for reusable ME_FLP_V7FixRhoscc Python run."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_run_config
from .parity_check import check_against_matlab
from .solver import SolverOptions, run_model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run ME_FLP_V7FixRhoscc Python solver")
    p.add_argument("--workbook", type=Path, default=Path("KLexperiments_ME_FLP_PD.xlsx"))
    p.add_argument("--run-no", type=int, default=61)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--plots", action="store_true", help="Enable diagnostic plotting")
    p.add_argument("--parity-matlab-dir", type=Path, default=None)
    p.add_argument("--parity-out-loop", type=int, default=None)
    p.add_argument("--parity-in-loop", type=int, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_run_config(args.workbook, run_no=args.run_no)
    res = run_model(
        cfg,
        SolverOptions(
            make_plots=args.plots,
            output_root=args.output_root,
            workbook_path=args.workbook,
        ),
    )
    print(f"Completed run_no={cfg.run_no}")
    if res.run_dir is not None:
        print(f"Run directory: {res.run_dir}")
    print(f"Boundary events: {len(res.boundary_events)}")
    print(f"Multiple-maxima events: {len(res.multi_max_events)}")

    if args.parity_matlab_dir is not None:
        if args.parity_out_loop is None or args.parity_in_loop is None:
            raise ValueError("parity requires --parity-out-loop and --parity-in-loop")
        summary = check_against_matlab(
            result=res,
            matlab_dir=args.parity_matlab_dir,
            out_loop=args.parity_out_loop,
            in_loop=args.parity_in_loop,
        )
        print("Parity metrics:")
        for k, v in summary.metrics.items():
            print(f"  {k}: {v:.6e}")


if __name__ == "__main__":
    main()
