"""Profile a chosen package (ref or enhance) with out_loop_max=1, in_loop_max=1.

Disables .mat saving (svopt=0) so the profile reflects pure compute.
Writes pstats text and a .prof binary into ME_FLP_V6scc_Enhance/.
"""

from __future__ import annotations

import argparse
import cProfile
import importlib
import io
import pstats
import sys
import time
from dataclasses import replace
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="Profile minimal run (1 outer, 1 inner)")
    p.add_argument("--workbook", type=Path, default=WORKSPACE_ROOT / "KLexperiments_ME_FLP_PD.xlsx")
    p.add_argument("--run-no", type=int, default=51)
    p.add_argument(
        "--package",
        type=str,
        default="ME_FLP_V6scc",
        choices=("ME_FLP_V6scc", "ME_FLP_V6scc_Enhance", "ME_FLP_V6scc_Enhance2", "ME_FLP_V6scc_Enhance3", "ME_FLP_V6scc_Enhance4"),
        help="Which package to profile.",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag appended to output artifact filenames (e.g. 'phase1').",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=WORKSPACE_ROOT / "ME_FLP_V6scc_Enhance",
    )
    p.add_argument("--out-loop-max", type=int, default=1)
    p.add_argument("--in-loop-max", type=int, default=1)
    args = p.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)

    config_mod = importlib.import_module(f"{args.package}.config")
    solver_mod = importlib.import_module(f"{args.package}.solver")
    load_run_config = config_mod.load_run_config
    SolverOptions = solver_mod.SolverOptions
    run_model = solver_mod.run_model

    cfg = load_run_config(args.workbook, run_no=args.run_no)
    cfg = replace(
        cfg,
        out_loop_max=args.out_loop_max,
        in_loop_max=args.in_loop_max,
        svopt=0,
    )

    print("config grid sizes:")
    print(
        f"  n_squig={cfg.n_squig} n_rho={cfg.n_rho} "
        f"n_delta={cfg.n_delta} n_eps={cfg.n_eps} n_mup={cfg.n_mup}"
    )
    print(
        f"  out_loop_max={cfg.out_loop_max} in_loop_max={cfg.in_loop_max} "
        f"go={cfg.go} svopt={cfg.svopt}"
    )

    options = SolverOptions(make_plots=False)

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    res = run_model(cfg, options)
    profiler.disable()
    elapsed = time.perf_counter() - t0
    print(f"run_model elapsed: {elapsed:.3f} s")
    print(f"boundary events={len(res.boundary_events)}, multi_max events={len(res.multi_max_events)}")

    suffix = f"_{args.tag}" if args.tag else ""
    base = f"profile_run{args.run_no}_o{args.out_loop_max}i{args.in_loop_max}{suffix}"
    prof_path = args.report_dir / f"{base}.prof"
    profiler.dump_stats(str(prof_path))
    print(f"wrote {prof_path}")

    for sort_key, n_lines, label in [
        ("cumulative", 40, "cumulative"),
        ("tottime", 40, "tottime"),
    ]:
        buf = io.StringIO()
        ps = pstats.Stats(profiler, stream=buf).strip_dirs().sort_stats(sort_key)
        ps.print_stats(n_lines)
        out_path = args.report_dir / f"{base}_{label}.txt"
        out_path.write_text(buf.getvalue(), encoding="utf-8")
        print(f"wrote {out_path}")

    summary_path = args.report_dir / f"{base}_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"package={args.package}\n")
        f.write(f"run_no={cfg.run_no}\n")
        f.write(f"out_loop_max={cfg.out_loop_max}\n")
        f.write(f"in_loop_max={cfg.in_loop_max}\n")
        f.write(f"n_squig={cfg.n_squig}\n")
        f.write(f"n_rho={cfg.n_rho}\n")
        f.write(f"n_delta={cfg.n_delta}\n")
        f.write(f"n_eps={cfg.n_eps}\n")
        f.write(f"n_mup={cfg.n_mup}\n")
        f.write(f"go={cfg.go}\n")
        f.write(f"svopt={cfg.svopt}\n")
        f.write(f"elapsed_s={elapsed:.6f}\n")
        f.write(f"boundary_events={len(res.boundary_events)}\n")
        f.write(f"multi_max_events={len(res.multi_max_events)}\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
