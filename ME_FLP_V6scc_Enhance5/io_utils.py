"""I/O helpers for ME_FLP_V6scc Python port."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping

from scipy.io import loadmat, savemat

from .config import RunConfig


def default_output_root(config: RunConfig) -> Path:
    """Match MATLAB savefolder_new2 behavior for known user/platform pairs."""
    user = config.user
    platform = config.platform
    if user == "RK" and platform == "cluster":
        return Path("/project/rgkgrp/RK_ME_FL_predexp_results/")
    if user == "RK" and platform == "desktop":
        return Path(r"C:\Users\Robert\Dropbox\ME_FLP_STORAGE\RK")
    if user == "YL" and platform == "cluster":
        return Path("/project/rgkgrp/YL_ME_FL_predexp_results/")
    if user == "YL" and platform == "desktop":
        return Path(r"C:\Users\yanglu\Dropbox\ME_FLP_STORAGE\YL")
    raise ValueError(f"Unsupported user/platform combination: {user}/{platform}")


def create_run_dir(config: RunConfig, output_root: Path | None = None) -> tuple[Path, str]:
    """Create run output directory and return (run_dir, run_name)."""
    root = output_root if output_root is not None else default_output_root(config)
    run_name = f"Experiment_{config.run_no}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_name


def write_log_line(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def save_mat(path: Path, payload: Mapping[str, object], append: bool = False) -> None:
    """Save MATLAB-compatible .mat file.

    Note: scipy.io.savemat does not support append=True; fail explicitly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        existing = loadmat(str(path))
        merged: Dict[str, object] = {k: v for k, v in existing.items() if not k.startswith("__")}
        merged.update(dict(payload))
        savemat(str(path), merged, do_compression=False)
        return
    savemat(str(path), dict(payload), do_compression=False)


def save_iteration_mat(
    *,
    run_dir: Path,
    out_loop: int,
    in_loop: int,
    arrays: Dict[str, object],
) -> Path:
    mat_path = run_dir / f"MPE{out_loop}W{in_loop}.mat"
    save_mat(mat_path, arrays, append=False)
    return mat_path
