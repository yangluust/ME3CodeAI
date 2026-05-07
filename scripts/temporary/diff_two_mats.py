"""Compare two .mat files element-wise across all common keys."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    args = p.parse_args()

    A = loadmat(str(args.a))
    B = loadmat(str(args.b))

    keys_a = {k for k in A.keys() if not k.startswith("__")}
    keys_b = {k for k in B.keys() if not k.startswith("__")}
    common = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    print(f"a: {args.a}")
    print(f"b: {args.b}")
    if only_a:
        print(f"only in a: {only_a}")
    if only_b:
        print(f"only in b: {only_b}")

    width = max(len(k) for k in common)
    worst = 0.0
    for k in common:
        va = np.asarray(A[k])
        vb = np.asarray(B[k])
        if va.shape != vb.shape:
            print(f"  {k.ljust(width)} : SHAPE MISMATCH {va.shape} vs {vb.shape}")
            continue
        if not np.issubdtype(va.dtype, np.number) or not np.issubdtype(vb.dtype, np.number):
            print(f"  {k.ljust(width)} : non-numeric dtype")
            continue
        diff = float(np.max(np.abs(va.astype(float) - vb.astype(float))))
        worst = max(worst, diff)
        print(f"  {k.ljust(width)} : max-abs diff = {diff:.6e}")

    print(f"\nworst max-abs diff: {worst:.6e}")
    sys.exit(0 if worst == 0.0 else 0)


if __name__ == "__main__":
    main()
