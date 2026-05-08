"""Side-by-side parity gate: ME_FLP_V6scc_Enhance3 (Phase 3) vs ME_FLP_V6scc_Enhance4 (Phase 4)."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import ME_FLP_V6scc_Enhance3 as ref_pkg
import ME_FLP_V6scc_Enhance4 as new_pkg

from ME_FLP_V6scc_Enhance3.config import load_run_config
from ME_FLP_V6scc_Enhance3.solver import SolverOptions as RefSolverOptions
from ME_FLP_V6scc_Enhance4.solver import SolverOptions as NewSolverOptions


COMPARED_ARRAYS = (
    "a_dr",
    "alpha_dr",
    "einfl_dr",
    "muprime_dr",
    "Unew",
    "z_dr",
    "Zupdates",
    "Zchange",
)


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return float(np.max(np.abs(a - b)))


def main() -> None:
    p = argparse.ArgumentParser(description="Phase3 vs Phase4 parity gate (in-process).")
    p.add_argument("--workbook", type=Path, default=WORKSPACE_ROOT / "KLexperiments_ME_FLP_PD.xlsx")
    p.add_argument("--run-no", type=int, default=51)
    p.add_argument("--out-loop-max", type=int, default=1)
    p.add_argument("--in-loop-max", type=int, default=1)
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Hard pass/fail tolerance on max-abs diffs. 0.0 means exact bit-equality.",
    )
    args = p.parse_args()

    cfg = load_run_config(args.workbook, run_no=args.run_no)
    cfg = replace(
        cfg,
        out_loop_max=args.out_loop_max,
        in_loop_max=args.in_loop_max,
        svopt=0,
    )
    print(
        f"config: run_no={cfg.run_no} n_squig={cfg.n_squig} n_rho={cfg.n_rho} "
        f"n_delta={cfg.n_delta} n_eps={cfg.n_eps} "
        f"out_loop_max={cfg.out_loop_max} in_loop_max={cfg.in_loop_max}"
    )

    print("\n[ref] running ME_FLP_V6scc_Enhance3.run_model (Phase 3) ...")
    t0 = time.perf_counter()
    ref_res = ref_pkg.run_model(cfg, RefSolverOptions(make_plots=False))
    ref_elapsed = time.perf_counter() - t0
    print(f"[ref] elapsed: {ref_elapsed:.3f} s")

    print("\n[new] running ME_FLP_V6scc_Enhance4.run_model (Phase 4) ...")
    t0 = time.perf_counter()
    new_res = new_pkg.run_model(cfg, NewSolverOptions(make_plots=False))
    new_elapsed = time.perf_counter() - t0
    print(f"[new] elapsed: {new_elapsed:.3f} s")

    diffs: dict[str, float] = {}
    for name in COMPARED_ARRAYS:
        diffs[name] = _max_abs_diff(getattr(ref_res, name), getattr(new_res, name))

    print("\nmax-abs diffs (phase4 vs phase3):")
    width = max(len(k) for k in diffs)
    for k, v in diffs.items():
        print(f"  {k.ljust(width)} : {v:.6e}")

    speedup = ref_elapsed / new_elapsed if new_elapsed > 0 else float("inf")
    print(f"\nspeedup (phase3 / phase4): {speedup:.3f}x")

    worst = max(diffs.values())
    if worst > args.tolerance:
        print(f"\nFAIL: worst diff {worst:.6e} exceeds tolerance {args.tolerance:.6e}")
        sys.exit(1)
    print(f"\nPASS: worst diff {worst:.6e} <= tolerance {args.tolerance:.6e}")


if __name__ == "__main__":
    main()
