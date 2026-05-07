"""Plot a_dr / alpha_dr / einfl_dr comparison: new (Phase 1) vs MATLAB reference.

Loads MPE<out>W<in>.mat from two run directories and produces a 3x3 panel grid
per array (rows = squig index, cols = selected mu index) with rho index on
the x-axis. Mirrors the existing
`scripts/temporary/a_dr_rho_compare_MPE3W50_*.png` style.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def _stem_from_dir(path: Path) -> str:
    name = path.name
    if "_" in name:
        return name.rsplit("_", 1)[-1].split("T")[0]
    return name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--new-dir",
        type=Path,
        default=Path(r"C:\Users\yanglu\Dropbox\ME_FLP_STORAGE\YL\Experiment_51_20260507T121043"),
    )
    p.add_argument(
        "--ref-dir",
        type=Path,
        default=Path(r"C:\Users\yanglu\Dropbox\ME_FLP_STORAGE\YLExperiment_51_20260504T223305"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("scripts/temporary"))
    p.add_argument("--out-loop", type=int, default=3)
    p.add_argument("--in-loop", type=int, default=50)
    p.add_argument("--new-tag", type=str, default=None)
    p.add_argument("--ref-tag", type=str, default=None)
    p.add_argument("--mu-cols", type=int, nargs="+", default=[1, 85, 168])
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = f"MPE{args.out_loop}W{args.in_loop}.mat"
    new_path = args.new_dir / base
    ref_path = args.ref_dir / base
    if not new_path.exists():
        raise FileNotFoundError(f"Missing new artifact: {new_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference artifact: {ref_path}")

    new = loadmat(str(new_path))
    ref = loadmat(str(ref_path))

    new_tag = args.new_tag or _stem_from_dir(args.new_dir)
    ref_tag = args.ref_tag or _stem_from_dir(args.ref_dir)
    new_label = f"MPE{args.out_loop}W{args.in_loop} ({new_tag})"
    ref_label = f"MPE{args.out_loop}W{args.in_loop} ({ref_tag})"

    arrays = ("a_dr", "alpha_dr", "einfl_dr")
    n_squig = int(np.asarray(new["a_dr"]).shape[0])
    n_rho = int(np.asarray(new["a_dr"]).shape[1])
    rho_index = np.arange(1, n_rho + 1)

    for name in arrays:
        new_arr = np.asarray(new[name])
        ref_arr = np.asarray(ref[name])
        if new_arr.shape != ref_arr.shape:
            raise ValueError(f"{name} shape mismatch: {new_arr.shape} vs {ref_arr.shape}")
        fig, axes = plt.subplots(n_squig, len(args.mu_cols), figsize=(12, 9))
        for r, isq in enumerate(range(n_squig)):
            for c, mu_col in enumerate(args.mu_cols):
                ax = axes[r, c] if n_squig > 1 else axes[c]
                m_idx = mu_col - 1
                ax.plot(rho_index, new_arr[isq, :, m_idx], label=new_label, color="tab:blue")
                ax.plot(rho_index, ref_arr[isq, :, m_idx], label=ref_label, color="tab:orange")
                ax.set_title(f"squig={isq + 1}, mu_idx={mu_col}")
                ax.set_xlabel("rho index")
                ax.set_ylabel(name)
                if r == 0 and c == 0:
                    ax.legend()
        fig.tight_layout()
        out_path = args.out_dir / (
            f"{name}_rho_compare_MPE{args.out_loop}W{args.in_loop}_"
            f"{new_tag}_vs_{ref_tag}.png"
        )
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
