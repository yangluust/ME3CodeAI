# Deterministic Parameter Sweep Guide

This guide explains how to run a deterministic, user-controlled parameter sweep using:

- `scripts/robustness/sweep_param_a_minus_alpha.py`

The sweep varies exactly one `RunConfig` parameter over explicit values that you provide, then outputs:

- `a_dr - alpha_dr` curves as a function of `rho`
- one combined `.csv`
- one combined `.png`

## 1) Run from workspace root

```powershell
cd C:\ME3CodeAI
```

## 2) Basic usage

```powershell
python .\scripts\robustness\sweep_param_a_minus_alpha.py `
  --workbook ".\KLexperiments_ME_FLP_PD.xlsx" `
  --run-no 51 `
  --package "ME_FLP_V6scc_Enhance5" `
  --param "qq" `
  --values 0.94 0.95 0.96
```

This example runs three deterministic evaluations in the exact order listed in `--values`.

## 3) Important controls

- `--param`: the exact `RunConfig` field to vary (example: `qq`, `kappa`, `vtheta_pi1`, `xstar1`)
- `--values`: explicit values for that field (no random search is used)
- `--out-loop-max` and `--in-loop-max`: run-depth control (defaults: `1` and `5`)
- `--svopt {0,1}`: file-saving switch inside solver (default `0` for faster sweeps)
- `--squig-rows`: 1-based squig indices to include in output plots (default `1 2 3`)
- `--mu-cols`: 1-based mu indices to include in output plots (default `1 85 168`)
- `--out-dir`: output directory (default `scripts/robustness`)
- `--tag`: filename suffix for your sweep batch

## 4) Full-fidelity run example

If you want full loop depth (similar to your `MPE3W50` workflow), set loops explicitly:

```powershell
python .\scripts\robustness\sweep_param_a_minus_alpha.py `
  --workbook ".\KLexperiments_ME_FLP_PD.xlsx" `
  --run-no 51 `
  --package "ME_FLP_V6scc_Enhance5" `
  --param "qq" `
  --values 0.94 0.95 0.96 `
  --out-loop-max 3 `
  --in-loop-max 50 `
  --svopt 0 `
  --tag "qq_o3i50"
```

## 5) Output files

For the whole sweep:

- `sweep_<param>_a_minus_alpha_vs_rho_<tag>.csv`
  - long-format rows for selected squig/mu slices
  - columns:
    - `param_name`, `param_value`
    - `squig_idx`, `mu_idx`
    - `rho_idx`, `rho_value`
    - `a_minus_alpha`

- `sweep_<param>_a_minus_alpha_vs_rho_<tag>.png`
  - subplot grid:
    - rows = selected `squig` indices
    - columns = selected `mu` indices
  - each subplot overlays all parameter values vs `rho`

## 6) How to view outputs quickly

- Open the generated `.png` in Cursor image preview.
- Open the generated `.csv` in the editor and filter by:
  - `squig_idx`
  - `mu_idx`
  - `param_value`
- Use `rho_value` and `rho_idx` columns in `.csv` for programmatic post-processing.
