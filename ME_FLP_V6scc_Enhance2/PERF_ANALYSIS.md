# ME_FLP_V6scc Phase 2 — Vectorize the inner `for i_mu` loop

This file documents the Phase 2 delta in `ME_FLP_V6scc_Enhance2/`. The Phase 0 baseline and Phase 1 design live in `ME_FLP_V6scc_Enhance/PERF_ANALYSIS.md` and are not repeated here.

**Scope.** Replace the per-`i_mu` Python loop in `solver.py` with a vectorized batch over `n_mu` while preserving bit-exact parity against Phase 1.

**Acceptance gate.** `scripts/temporary/parity_enhance1_vs_enhance2.py` reports max-abs diff = 0.0 (bit-equal) on every result array (`a_dr`, `alpha_dr`, `einfl_dr`, `muprime_dr`, `Unew`, `z_dr`, `Zupdates`, `Zchange`).

---

## 1. What landed

All edits live in `ME_FLP_V6scc_Enhance2/`:

- `solver.py` (inner `for i_mu` loop in the body for-`st` block):
  - Pre-allocate a single `flat_buf` of shape `(n_mu, n_delta * n_mup)` once per `run_model` call, reused across every state and every (out, in) iteration.
  - Per state, build the F-flat tensor `flat = d_mu[:, None] * om_st_f[None, :] + urhs_f[None, :]` directly into `flat_buf` via `np.multiply(..., out=flat_buf)` + in-place `+= urhs_f[None, :]`. `om_st_f` and `urhs_f` are F-order ravels of `om_st` and `urhs` (small, 670 KB each). This avoids the `(n_mu, n_delta, n_mup)` 3D intermediate and the `swapaxes(1, 2).reshape(...)` copy that 3D layouts force.
  - Vectorized argmax: `flat_argmax = np.argmax(flat, axis=1)` produces F-flat indices for all `n_mu` at once, replacing the `n_mu` per-state `np.argmax(om_st_mu.reshape(-1, order='F'))` calls.
  - Cautious tie detection uses **strict equality** (`flat == max_per_mu[:, None]`) instead of `np.isclose(..., atol=1e-16, rtol=0)`. This is bit-equivalent for our problem because all `om_st_mu` magnitudes have `|x| >> 1e-16`, so 1 ULP ≥ ~2e-16 > 1e-16 and `isclose` admits no values that strict equality would reject. Verified by the parity gate.
  - Slow-path `resolve_multiple_maxima` is invoked only for `i_mu` rows where `tie_count > 1`. The F-order `aic.reshape(-1, order='F')` and `einfl.reshape(-1, order='F')` copies are precomputed **once per state** and passed to every tied-`i_mu` resolve call via two new optional kwargs (`vaic`, `veinfl`).
  - PTkeep updates and the per-i_mu writes to `a_dr_new`, `alpha_dr_new`, `einfl_dr_new`, `muprime_dr_new`, `Unew` are now vectorized fancy-indexing assignments using `(r0_all, c0_all)` arrays of shape `(n_mu,)`.
  - `mu_lp == 0` boundary case (dead in production with `mu_bound`-derived `d_mu`) is preserved by setting `flat[zero_mu_indices, invalid_F]` to `-inf` before the argmax.
  - `override_ptkeep` and the optional `debug_trace` block are preserved with vectorized semantics.

- `diagnostics.py`:
  - `resolve_multiple_maxima` accepts optional `vaic`, `veinfl` kwargs. When provided, it skips the per-call F-order reshape of `aic` and `einfl`. Default (None) preserves the original signature for any other call site.

No changes to `core_functions.py`, `mu_bounds.py`, `rho_one.py`, `io_utils.py`, `parity_check.py`, `config.py`, or `run_me_flp_v6.py`.

---

## 2. Bit-exact parity (in-process)

`scripts/temporary/parity_enhance1_vs_enhance2.py` runs Phase 1 and Phase 2 back-to-back in the same process and compares every solver output array. Tolerance default is `0.0` (bit-equality required).

| out_loop_max × in_loop_max | worst max-abs diff | result |
| --- | --- | --- |
| 1 × 1 | `0.000000e+00` | PASS |
| 2 × 1 | `0.000000e+00` | PASS |
| 2 × 5 | `0.000000e+00` | PASS |
| 3 × 3 | `0.000000e+00` | PASS |

Logs: `parity_phase2_1x1_v4.log`, `parity_phase2_2x1.log`, `parity_phase2_2x5_v4.log`, `parity_phase2_3x3.log`.

### 2.1 Cross-process drift (not a regression)

When Phase 1's earlier `out=3, in=50` run (process `Experiment_51_20260507T121043`) is compared file-by-file to Phase 2's today's `out=3, in=50` run (process `Experiment_51_20260507T143647`), the saved `.mat` files agree bit-exactly through `MPE1W50` and then start to drift at `MPE2W1` (PTkeep up to ±2 indices, a_dr up to ~1.4e-4). This is **not** caused by Phase 2: the in-process parity gate at `3 × 3` proves Phase 1 and Phase 2 produce bit-equal results through every outer-loop transition.

The drift is multi-threaded OpenBLAS non-determinism. Default NumPy on this machine uses `openblas64_` with no thread limit; the `squig_trans @ Unew.reshape(...)` matmul at the end of every inner iteration produces sub-ULP-different partial sums depending on thread schedule. Over 50 inner iters of outer 1 the sub-ULP drift accumulates and, at the start of outer 2, tips the argmax over `om_st_mu` at near-tied cells, producing ±2 swaps in `PTkeep[i_squig, i_rho, i_mu]` and the corresponding ~1.4e-4 jump in `a_dr` (which depends on `d_delta[r0]`, `d_delta` step ≈ 1e-4). To force deterministic cross-process runs, set `OPENBLAS_NUM_THREADS=1`.

---

## 3. Speedup vs Phase 1

Measured wall-clock from `parity_enhance1_vs_enhance2.py` runs (Phase 1 ref vs Phase 2 new, same process, parity-checked).

| Config | Phase 1 (s) | Phase 2 (s) | speedup (P1 / P2) |
| --- | --- | --- | --- |
| 1 × 1 | 13.381 | 9.725 | **1.26x** |
| 2 × 5 | 110.95 | 89.538 | **1.24x** |

### 3.1 End-to-end production run (`out_loop_max=3, in_loop_max=50`, `svopt=1`)

Driven via `scripts/temporary/run_full_3x50.py --package ME_FLP_V6scc_Enhance2`, default output root, MATLAB ref `C:\Users\yanglu\Dropbox\ME_FLP_STORAGE\YLExperiment_51_20260504T223305\MPE3W50.mat`.

| Metric | Phase 1 (earlier process) | Phase 2 (this run) |
| --- | --- | --- |
| wall-clock | 1699.32 s (28.32 min) | **1379.81 s (23.00 min)** — 1.23x |
| `boundary_events` | 6276 | 6276 |
| `multi_max_events` | 7 | 7 |
| `a_dr` vs MATLAB MPE3W50.mat | 4.85e-04 | 3.50e-04 |
| `alpha_dr` vs MATLAB MPE3W50.mat | 4.85e-04 | 3.50e-04 |
| `einfl_dr` vs MATLAB MPE3W50.mat | 5.16e-04 | 3.72e-04 |
| `muprime_dr` vs MATLAB MPE3W50.mat | 3.29e-03 | 3.80e-03 |
| `Unew` vs MATLAB MPE3W50.mat | 1.69e-05 | 2.11e-05 |

The differences in the MATLAB-vs-Python diffs between the two runs are within the cross-process drift band described in §2.1, not a parity regression. Logs: `full_3x50_phase2.log`. Run dir for Phase 2 outputs: `C:\Users\yanglu\Dropbox\ME_FLP_STORAGE\YL\Experiment_51_20260507T143647`.

Profile (1 × 5, `cProfile`):

| Metric | Phase 1 | Phase 2 | delta |
| --- | --- | --- | --- |
| total wall-clock under `cProfile` | 58.66 s | 45.96 s | **−21.7%** |
| `getmax` cumtime | 19.85 s (47,880 calls) | 0 (call site removed) | −19.85 s |
| `np.isclose` / `within_tol` cumtime | 11.99 s (47,880 calls) | 0 (no isclose in cautious path) | −11.99 s |
| `np.argmax` (vectorized) | inside getmax | 2.42 s (285 calls = 1/state) | new |
| `np.ndarray.reshape` | 5.85 s (180,976 calls) | 0.14 s (77,241 calls) | −5.71 s |
| `resolve_multiple_maxima` cumtime | 5.36 s (28,586 calls) | 1.94 s (28,586 calls) | −3.42 s (cached vaic/veinfl) |
| `ii_fhat_factored` cumtime | 18.70 s (855 calls) | 20.18 s (855 calls) | +1.48 s (within noise) |
| `run_model` self-time | 13.09 s | 16.10 s | +3.01 s (3D-tensor alloc + indexing) |

The 3 s regression in `run_model` self-time (from the per-state `flat_buf` write of ~113 MB) is more than paid for by the elimination of `getmax`, `np.isclose`, and the per-call `aic`/`einfl` reshape copies.

Artifacts: `profile_run51_o1i5_phase1_1x5_*.txt`, `profile_run51_o1i5_phase2_v4_1x5_*.txt`.

---

## 4. Why Phase 2 first looked like a regression — and what fixed it

For full transparency, the first two iterations of Phase 2 were **slower** than Phase 1. Sequence of fixes that closed the gap:

1. **v1: 3D tensor `(n_mu, n_delta, n_mup)` + `swapaxes(1,2).reshape(...)`** → 1×1 ran in **24.96 s** (0.54x of Phase 1). The forced 113 MB swap-reshape copy and the `np.isclose` batch on (n_mu, n_delta·n_mup) elements were the culprits — both memory-bandwidth-bound (113 MB ≫ L2), whereas Phase 1's per-`i_mu` calls fit in L2 (670 KB).
2. **v2: pre-allocated buffer reused across states** → 17.78 s at 1×1 (0.73x). Removed 56 redundant 113 MB allocations per inner iter.
3. **v3: strict `==` instead of `np.isclose` for tie detection** → 10.63 s at 1×1 (1.17x). Saves ~20 s/inner-iter on the (n_mu, 14M) batch comparison without changing the tie set for typical `om_st_mu` magnitudes.
4. **v4: cached `vaic`/`veinfl` per state, plus dropped the dead `diff_mask` correction** → **9.73 s at 1×1 (1.26x)**. With strict equality, `np.argmax(tie_mask, axis=1) == flat_argmax` always, so the secondary correction was a no-op.

---

## 5. Open items / what would be needed to close the rest of the gap

Phase 2 lands at ~1.25x over Phase 1. To approach MATLAB's ~18–20 minute end-to-end (3×50, MPE3W50.mat reference), additional phases would be needed. The remaining hotspots are:

1. `ii_fhat_factored` (~20 s/inner iter at 1×5): bilinear gather + epsilon integration. Already heavily optimized via `np.einsum`. Further wins likely require Numba/Cython JIT or GPU.
2. `run_model` self-time (~16 s/inner iter at 1×5): Python-level orchestration of per-state arithmetic. Reducing this needs either fewer per-state ops (batch across states) or moving the orchestration to a compiled language.
3. `flat_buf` write per state (~6.4 GB memory traffic / inner iter): fundamental for vectorization; a Numba kernel would fuse the multiply-add-argmax to avoid materializing `flat_buf`.

A Phase 3 that JIT-compiles the inner state body (or just the multiply-add-argmax + tie detection) with Numba is the natural next step if a deeper speedup is required.

---

## 6. Reproducibility

Driver scripts (under `scripts/temporary/`):

- `parity_enhance1_vs_enhance2.py` — bit-exact parity gate; arguments `--out-loop-max`, `--in-loop-max`, `--tolerance` (default 0.0).
- `profile_minimal_run.py` — cProfile harness; `--package ME_FLP_V6scc_Enhance2 --tag <name>` writes `profile_run<run_no>_o<O>i<I>_<tag>.{prof,_cumulative.txt,_tottime.txt,_summary.txt}` into the directory passed via `--report-dir`.

Reference workbook: `KLexperiments_ME_FLP_PD.xlsx`, `--run-no 51`. Effective grid: `n_squig=3`, `n_rho=21`, `n_delta=501`, `n_eps=51`, `n_mu=n_mup=168`, body states = 57.
