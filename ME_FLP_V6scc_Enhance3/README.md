# ME_FLP_V6scc Python Port

This folder contains a reusable Python conversion of `ME_FLP_V6scc.m` based on `ME_FLP_V6scc_to_Python.md`.

## Files

- `config.py`: workbook-driven config loading (replacement for `LoadModel.m` data loading behavior)
- `core_functions.py`: ports of MATLAB helper functions
- `rho_one.py`: analytical `rho=1` block (`rhoone_new` port)
- `mu_bounds.py`: `mu_bound` loop port
- `diagnostics.py`: boundary and multiple-maxima diagnostics
- `io_utils.py`: run directory creation and `.mat` writing
- `solver.py`: main solver loop port
- `parity_check.py`: lightweight comparison against MATLAB `.mat` artifacts
- `run_me_flp_v6.py`: CLI entrypoint

## Run

From workspace root:

```bash
python -m ME_FLP_V6scc.run_me_flp_v6 --workbook "KLexperiments_ME_FLP_PD.xlsx" --run-no 51
```

Optional output root override:

```bash
python -m ME_FLP_V6scc.run_me_flp_v6 --workbook "KLexperiments_ME_FLP_PD.xlsx" --run-no 51 --output-root "D:/ME3codeAI/output"
```

## Parity Check

If MATLAB produced `MPE<out_loop>W<in_loop>.mat` for run 51, compare key arrays:

```bash
python -m ME_FLP_V6scc.run_me_flp_v6 \
  --workbook "KLexperiments_ME_FLP_PD.xlsx" \
  --run-no 51 \
  --parity-matlab-dir "D:/path/to/matlab/run/folder" \
  --parity-out-loop 10 \
  --parity-in-loop 20
```

## Notes

- This port keeps MATLAB-like array semantics (`ndgrid`-style indexing, Fortran-order reshape where needed).
- Fail-fast behavior is used for missing inputs or unsupported run settings.
- GPU mode is not implemented in this first behavior-preserving Python pass.
