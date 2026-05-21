"""Plot latest MPE3W50 policy slices in the existing rho-index style."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def plot_array(name: str, arr: np.ndarray, out_dir: Path, source_tag: str, mu_cols: list[int]) -> Path:
    n_squig, n_rho, n_mu = arr.shape
    rho_index = np.arange(1, n_rho + 1)

    for mu_col in mu_cols:
        if mu_col < 1 or mu_col > n_mu:
            raise ValueError(f"mu_idx={mu_col} out of bounds for {name}; valid range is 1..{n_mu}")

    fig, axes = plt.subplots(n_squig, len(mu_cols), figsize=(12, 9))
    for r in range(n_squig):
        for c, mu_col in enumerate(mu_cols):
            ax = axes[r, c] if n_squig > 1 else axes[c]
            m_idx = mu_col - 1
            ax.plot(rho_index, arr[r, :, m_idx], label=f"MPE3W50 ({source_tag})", color="tab:blue")
            ax.set_title(f"squig={r + 1}, mu_idx={mu_col}")
            ax.set_xlabel("rho index")
            ax.set_ylabel(name)
            if r == 0 and c == 0:
                ax.legend()

    fig.tight_layout()
    out_path = out_dir / f"{name}_rho_MPE3W50_{source_tag}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mat",
        type=Path,
        default=Path("outputs") / "Experiment_51_20260521T223150" / "MPE3W50.mat",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("scripts") / "temporary")
    parser.add_argument("--source-tag", type=str, default="run51_20260521")
    parser.add_argument("--mu-cols", type=int, nargs="+", default=[1, 85, 168])
    args = parser.parse_args()

    if not args.mat.exists():
        raise FileNotFoundError(f"Missing MPE file: {args.mat}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = loadmat(str(args.mat))
    for name in ("a_dr", "alpha_dr", "einfl_dr"):
        if name not in data:
            raise KeyError(f"{args.mat} does not contain {name}")
        arr = np.asarray(data[name])
        if arr.ndim != 3:
            raise ValueError(f"{name} must be 3D; got shape {arr.shape}")
        out_path = plot_array(
            name=name,
            arr=arr,
            out_dir=args.out_dir,
            source_tag=args.source_tag,
            mu_cols=args.mu_cols,
        )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
