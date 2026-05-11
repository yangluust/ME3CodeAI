# ME_FLP_V7FixRhoscc_Enhance5 -- Performance & Parity Notes

Phase 5-style Python port of `ME_FLP_V7FixRhoscc.m` built directly on the
`ME_FLP_V6scc_Enhance5` architecture (numba prange parallel state worker,
JIT'd argmax+tie kernel). No intermediate phases were implemented because
V7 is a strict simplification of V6's body block, not an independent
algorithmic family.

## Algorithmic delta vs V6 Phase 5

V7 fixes rho' = rho (constant reputation, no Bayesian update). Combined
with V7.m line 178 (`d_mup = d_mu`), this collapses V6's bilinear
(rho', mu) interpolation of `Mhat1`, `Mhat2`, `Uhat` to direct 1-D slicing:

```python
einfl_1d[j] = bet * rho_st * Mhat1[i_squig, i_rhost, j] \
            + bet * (1.0 - rho_st) * Mhat2[i_squig, i_rhost, j]
Uhat_1d[j]  = Uhat[i_squig, i_rhost, j]
```

`einfl_1d` and `Uhat_1d` are then broadcast across the delta axis to form
the (n_delta, n_mup) tensors used by `ufcn_yl` and the F-flat ravel.

The boundary block (`st < n_squig`) and the rho=1 block (`solve_rho_one`)
are reused unchanged from V6 Phase 5 -- V6.m and V7.m are identical there
(the YANG FIX markers added A1/B1/a0 to V6.m as well, so V6 Phase 5 already
matches V7's boundary-block semantics).

## Implementation summary

- **Removed from V6 Phase 5 solver**: `rhoprime1`/`rhoprime2` precompute,
  `bilinear_axis_table` calls, `rho_idx_stack_*`/`rho_w_stack_*`/`mu_idx`/
  `mu_w`/`fdens_flat` plumbing, `ii_fhat_factored` post-pass calls.
- **Removed from V6 Phase 5 worker** (`process_body_states_parallel`):
  `ii_fhat_factored_kernel(Mhat1)` / `ii_fhat_factored_kernel(Mhat2)` /
  `ii_fhat_factored_kernel(Uhat)` calls, replaced with direct 1-D slicing
  + delta-axis broadcast in two tight `for j` loops.
- **Kept identical**: ufcn_yl FP-evaluation order (the V6 Phase 5 fixes
  `c1*(a*a)`, `(e+ivec)+kxv`, and `(1-rho)*alpha/rho` all carry over).
- **Fail-fast**: `VARIANT='endog_rho'` raises `ValueError`; `d_mup != d_mu`
  raises `ValueError` (V7 worker assumes `n_mup == n_mu` and identical
  values).

## Parity (run_no=61, MATLAB ref `YLExperiment_61_20241210T091431`)

In-process 1x1 (1 outer, 1 inner iteration):

| Field        | max-abs diff |
|--------------|--------------|
| a_dr         | 8.67e-19 |
| alpha_dr     | 8.67e-19 |
| einfl_dr     | 4.34e-19 |
| muprime_dr   | 0        |
| Unew         | 5.08e-20 |

Full 3x50 production:

| Field        | max-abs diff |
|--------------|--------------|
| a_dr         | 9.15e-04 |
| alpha_dr     | 9.15e-04 |
| einfl_dr     | 9.74e-04 |
| muprime_dr   | 3.21e-03 |
| Unew         | 1.62e-04 |

The 3x50 magnitudes match the cross-process FP non-determinism band we
already documented for V6 Phase 5 vs V6 MATLAB. They are at least 20x
smaller than the V6 Python vs V7 MATLAB comparison we previously ran
(~2.3e-2), confirming the V7 algorithmic shape is correctly reproduced.

## Cross-config diagnostic: where does the ~1e-3 floor come from?

We ran a controlled sweep across four configurations to isolate the
driver of cross-process FP drift. All other parameters held constant.

| run | pkg | variant   | alternative | go | a_dr      | alpha_dr  | einfl_dr  | muprime_dr | Unew      | PTkeep mism (states) |
|----:|-----|-----------|-------------|---:|----------:|----------:|----------:|-----------:|----------:|---------------------:|
|  51 | V6  | endog_rho | optimizing  |  1 |  3.50e-04 |  3.50e-04 |  3.72e-04 |   3.80e-03 |  2.11e-05 |        5052 / 10584  |
|  52 | V6  | endog_rho | mechanical  |  1 |     **0** |     **0** |     **0** |      **0** |     **0** |               **0**  |
|  61 | V7  | exo_rho   | optimizing  |  0 |  8.83e-04 |  8.83e-04 |  9.05e-04 |   7.61e-03 |  5.41e-05 |              ~ 50 %  |
|  62 | V7  | exo_rho   | optimizing  |  1 |  8.83e-04 |  8.83e-04 |  9.05e-04 |   7.61e-03 |  5.41e-05 |              ~ 50 %  |

References
- run 51 Py: `C:\Users\yanglu\Dropbox\ME_FLP_STORAGE\YL\Experiment_51_20260508T102519`
  vs MATLAB `YLExperiment_51_20260504T223305`
- run 52 Py: `YLExperiment_52_20260508T112642`
  vs same-dir MATLAB output
- run 61/62 Py / MATLAB: see `README.md` and earlier sections.

### What this proves

1. **`alternative` is the only differentiator.** Run 51 vs 52 is identical
   except `optimizing` vs `mechanical`; 51 drifts ~1e-3 / 1e-4, 52 is
   bit-exact. Run 51 also uses `endog_rho`, so endogenous-rho is not
   the source.
2. **`variant` and `go` do not matter.** V7 run 61 (exo_rho, go=0) and
   run 62 (exo_rho, go=1) match V6 run 51's drift pattern almost exactly,
   off by a constant factor (~2.5x) attributable to V7's broader argmax
   neighborhood under `rho'=rho`.
3. **PTkeep mismatch is the leading indicator.** Where parity is achieved
   PTkeep mismatch is 0 (run 52); where it is not, ~50% of states sit
   at near-ties on the body-block objective. This is the smoking gun for
   argmax tie-flipping caused by sub-ULP BLAS differences.

### Mechanism summary (already verified by C0 / C1 / C2')

`alternative=optimizing` sets `alpha_ic = A2*einfl + B2[squig]`, so for
fixed `einfl`, `urhs(d, j) ≈ urhs(d', j')` whenever `(d - d') ≈ -(A2/(1+A2))
* (einfl_j - einfl_{j'})`. Many `(d, j)` pairs land on this near-flat
manifold ⇒ sub-ULP differences in `einfl_j` (from NumPy-MKL vs MATLAB-MKL
in `squig_trans @ ...` and `kron`-style products inside `get_mhat`) flip
the argmax index by ±1 on either grid. Each flip yields an O(d_delta) ≈
1e-4 change in `a_dr` and an O(d_mup) ≈ 1e-3 change in `muprime_dr`,
which propagate through `z_dr → Mhat → einfl` and saturate at ~1e-3.

`alternative=mechanical` (run 52) bypasses the optimizing branch entirely:
`alpha_ic = alpha0vec[squig]` is constant in delta, so the objective is
monotone in the chosen direction and there are no near-ties to flip ⇒
bit-exact across platforms.

C0 (single-shot override at first divergence), C1 (single-threaded BLAS
via `MKL_NUM_THREADS=1`), and C2' (final-only override) all confirmed
that the drift is BLAS-micro-kernel order, not threading, and that it
saturates by mid-iteration regardless of when you intervene.

## C3: accept and document

We adopt **C3 (accept the cross-platform FP floor)** for any run with
`alternative=optimizing`. The Python port is numerically correct; the
residual gap is a fundamental NumPy-MKL vs MATLAB-MKL micro-kernel
difference amplified by argmax tie-breaking and iterative feedback.

### Recommended cross-platform parity tolerances

| alternative | a_dr / alpha_dr / einfl_dr | muprime_dr | Unew    |
|-------------|---------------------------:|-----------:|--------:|
| mechanical  |                          0 |          0 |       0 |
| optimizing  |                       1e-3 |       1e-2 |    1e-4 |

In-process Python runs remain bit-exact under all configurations
(identical seeds, identical BLAS, identical kernels), so the Python
package itself is deterministic. The cross-process bounds above are
only relevant when comparing Python output against a freshly generated
MATLAB run.

## Runtime

3x50 on AMD Ryzen 5 5600X (12 logical CPUs) + Quadro P1000:
- **V7 Phase-5-style**: 89.5 s (1.49 min), 5900 boundary events,
  0 multi_max_events.
- For comparison, V6 Phase 5 on the same machine, run_no=51, 3x50: 118 s
  (1.98 min). The ~25% V7 advantage comes from skipping the bilinear
  interp kernel (`O(n_eps * n_delta * n_mup + n_rho * n_mup)` per state
  in V6) in favor of a single `O(n_mu)` pass through Mhat1/Mhat2/Uhat
  per state.

## Driver scripts

- `python scripts/temporary/run_full_3x50.py --package ME_FLP_V7FixRhoscc_Enhance5
   --run-no 61 --out-loop-max 3 --in-loop-max 50
   --matlab-dir <V7_MATLAB_DIR> --parity-out-loop 3 --parity-in-loop 50`
- `python scripts/temporary/plot_compare_3x50_vs_matlab.py
   --new-dir <V7_PYTHON_RUN_DIR> --ref-dir <V7_MATLAB_DIR>
   --new-tag v7_run61_py_<DATE> --ref-tag v7_run61_matlab_<DATE>`

## Validated artifacts

- `scripts/temporary/v7_run61_3x50.log` -- full driver log.
- `scripts/temporary/{a_dr,alpha_dr,einfl_dr}_rho_compare_MPE3W50_v7_run61_py_20260508_vs_v7_run61_matlab_20241210.png`.
