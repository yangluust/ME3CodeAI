# ME_FLP_V7FixRhoscc_Enhance5

Phase 5-style Python port of `ME_FLP_V7FixRhoscc.m` (constant reputation,
rho' = rho). Built directly on top of `ME_FLP_V6scc_Enhance5`'s architecture:
no incremental phases were needed because V7 is a strict simplification of
V6's body block.

## What changed vs `ME_FLP_V6scc_Enhance5`

V7 fixes rho' = rho (no Bayesian updating of reputation), so the per-state
expected-inflation calculation in the body block reduces to direct array
slicing:

```
einfl_1d[j] = bet * rho_st * Mhat1[i_squig, i_rhost, j]
            + bet * (1.0 - rho_st) * Mhat2[i_squig, i_rhost, j]
einfl[d, j] = einfl_1d[j]                # constant in delta
Uhat_1d[j]  = Uhat[i_squig, i_rhost, j]
```

V6 Phase 5 instead built a (rho', mu) bilinear interpolation table and
called `ii_fhat_factored_kernel` per state. V7 drops:

- `rhoprime1`, `rhoprime2` precompute (V6 solver lines 334-356)
- `bilinear_axis_table(d_rho, ...)` rho-axis tables
- `bilinear_axis_table(d_mu, d_mup)` mu-axis tables
- `mu_idx_int`, `mu_w_f64`, `fdens_flat` arguments to the parallel worker
- `rho_idx_stack_*`, `rho_w_stack_*` arguments to the parallel worker
- `ii_fhat_factored_kernel` calls inside the parallel worker

The boundary block (`st < n_squig`) and the rho=1 block (`solve_rho_one`)
are reused unchanged from V6 Phase 5 because V6.m and V7.m are identical
there (the V7 [YANG FIX] markers added A1/B1/a0 to V6.m as well).

`solver.py` fails fast on `VARIANT='endog_rho'`: V7 only supports
`VARIANT='exo_rho'`. It also asserts `d_mup == d_mu` (V7.m line 178),
which the V7 inner loop relies on to index Mhat directly along the mu axis.

## Validation (run_no=61)

In-process 1x1 vs MATLAB `YLExperiment_61_20241210T091431`:

| Field        | max-abs diff |
|--------------|--------------|
| a_dr         | 8.67e-19 |
| alpha_dr     | 8.67e-19 |
| einfl_dr     | 4.34e-19 |
| muprime_dr   | 0        |
| Unew         | 5.08e-20 |

Full 3x50 production vs MATLAB `YLExperiment_61_20241210T091431`:

| Field        | max-abs diff |
|--------------|--------------|
| a_dr         | 9.15e-04 |
| alpha_dr     | 9.15e-04 |
| einfl_dr     | 9.74e-04 |
| muprime_dr   | 3.21e-03 |
| Unew         | 1.62e-04 |

These magnitudes are consistent with cross-process FP non-determinism
(NumPy-MKL vs MATLAB-MKL micro-kernel order) accumulating over inner-loop
iterations and sub-ULP differences flipping argmax tie-breaking on the
d_mup grid. See `PERF_ANALYSIS.md` for the full diagnostic.

## Cross-config parity floor (V6 / V7, runs 51 / 52 / 61 / 62)

Holding all other parameters fixed, the residual Python-vs-MATLAB drift
depends **only on `alternative`** (the body-block control choice):

| run | pkg | variant   | alternative | go | a_dr      | alpha_dr  | einfl_dr  | muprime_dr | Unew      | PTkeep mism |
|----:|-----|-----------|-------------|---:|----------:|----------:|----------:|-----------:|----------:|------------:|
|  51 | V6  | endog_rho | optimizing  |  1 |  3.50e-04 |  3.50e-04 |  3.72e-04 |   3.80e-03 |  2.11e-05 | ~48% states |
|  52 | V6  | endog_rho | mechanical  |  1 | **0**     | **0**     | **0**     | **0**      | **0**     | **0**       |
|  61 | V7  | exo_rho   | optimizing  |  0 |  8.83e-04 |  8.83e-04 |  9.05e-04 |   7.61e-03 |  5.41e-05 | ~50% states |
|  62 | V7  | exo_rho   | optimizing  |  1 |  8.83e-04 |  8.83e-04 |  9.05e-04 |   7.61e-03 |  5.41e-05 | ~50% states |

Run 51 vs 52 is the cleanest natural experiment: identical config except
`alternative=optimizing` vs `mechanical`, yet 51 drifts to ~1e-3 / 1e-4
while 52 is bit-exact. Run 61 / 62 confirm `variant` (endog_rho vs
exo_rho) and `go` (0 vs 1) are not material drivers.

### Mechanism (in one line)
`optimizing` makes `urhs(d, j)` near-flat across many `(d, j)` pairs ⇒
sub-ULP BLAS differences flip ~half the argmax indices ⇒ O(grid-spacing)
differences in `a_dr` / `alpha_dr` (≈ `d_delta` step) and `muprime_dr`
(≈ `d_mup` step) that feed back through `z_dr` → `Mhat` → `einfl`.
`mechanical` sets `alpha_ic = alpha0vec[squig]` (constant in delta), so
there are no near-ties to flip and parity is bit-exact.

### Recommended cross-platform tolerances

| alternative | a_dr / alpha_dr / einfl_dr | muprime_dr | Unew    |
|-------------|---------------------------:|-----------:|--------:|
| mechanical  |                          0 |          0 |       0 |
| optimizing  |                       1e-3 |       1e-2 |    1e-4 |

These are the floor: the Python port is numerically correct, and the
remaining cross-process gap is a fundamental NumPy-MKL vs MATLAB-MKL
micro-kernel difference, not a porting bug. In-process Python runs
remain bit-exact under all configurations.

Total runtime for 3x50 on Quadro P1000 + AMD Ryzen 5 5600X: **89.5 s
(1.49 min)**. About 25% faster than V6 Phase 5's 1.98 min on the same
problem because V7's body block is `O(n_st * n_mu)` per state instead of
V6 Phase 5's `O(n_st * (n_eps * n_delta * n_mup + n_rho * n_mup))`.

## Entrypoints

- `run_me_flp_v7.py` -- CLI runner, defaults `--run-no=61`.
- `parity_check.py` -- compare an in-memory `SolverResult` against a MATLAB
  `MPE<out>W<in>.mat` file.

## Comparison plots

`scripts/temporary/plot_compare_3x50_vs_matlab.py` already supports
arbitrary `--new-dir` / `--ref-dir`. The 3x50 V7 vs MATLAB plots live at
`scripts/temporary/{a_dr,alpha_dr,einfl_dr}_rho_compare_MPE3W50_v7_run61_py_20260508_vs_v7_run61_matlab_20241210.png`.
