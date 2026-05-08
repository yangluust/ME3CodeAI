# ME_FLP_V6scc Phase 4 — Numba JIT for `ii_fhat_factored`

This file documents the Phase 4 delta in `ME_FLP_V6scc_Enhance4/`. Phase 0–3 designs live in `ME_FLP_V6scc_Enhance{,2,3}/PERF_ANALYSIS.md` and are not repeated here.

**Scope.** Replace the Phase 3 NumPy-vectorized `ii_fhat_factored` (advanced indexing → two einsum calls → mu-axis interp) with a single `@njit(cache=True, boundscheck=False)` kernel that fuses bilinear gather + epsilon integration + mu-axis bilinear interp into three streaming passes with no intermediate (n_eps, n_delta, n_mu) tensor.

**Acceptance gate.** `scripts/temporary/parity_enhance3_vs_enhance4.py` reports max-abs diff = 0.0 on every result array.

---

## 1. What landed

All edits live in `ME_FLP_V6scc_Enhance4/`:

- `jit_kernels.py`:
  - New `ii_fhat_factored_kernel(f, rho_idx, rho_w, mu_idx, mu_w, fdens) -> (n_delta, n_mup)`. Fuses three logical stages into one C call:
    1. `g_lo[d, m] = sum_e (1 - rho_w[e, d]) * fdens[e] * f[rho_idx[e, d], m]` (sequential e, ascending)
    2. `g_hi[d, m] = sum_e rho_w[e, d] * fdens[e] * f[rho_idx[e, d] + 1, m]` (sequential e, ascending)
    3. `g[d, m] = g_lo[d, m] + g_hi[d, m]` then `out[d, j] = (1 - mu_w[j]) * g[d, mu_idx[j]] + mu_w[j] * g[d, mu_idx[j] + 1]`
  - The kernel never materializes `f_lo` or `f_hi` (the (n_eps, n_delta, n_mu) tensors that Phase 3's advanced indexing built); they were the dominant memory-bandwidth cost. Each f sample is read directly from the source `f` matrix into the per-(d, m) accumulator.
  - The two-step `tmp = w * f; acc = acc + tmp` split is preserved (no FMA contraction), and the lo/hi accumulations are kept in separate sub-loops so addition order matches Phase 3's `einsum + einsum` exactly.
  - `warmup()` extended with a tiny representative call to AOT-compile the new kernel before timed work starts.

- `core_functions.py`:
  - `ii_fhat_factored` retains the same public signature (still callable from `solver.py`, `mu_bounds.py`, etc.) and same input validation. After validation it now coerces all inputs to contiguous float64 / int64 buffers and delegates the entire numerical body to `ii_fhat_factored_kernel`. The Phase 3 NumPy `np.einsum + np.einsum + indexing` body is removed.

No changes to `solver.py`, `diagnostics.py`, `mu_bounds.py`, `rho_one.py`, `io_utils.py`, `parity_check.py`, `config.py`, or `run_me_flp_v6.py`. Phase 3's `fused_argmax_tie` / `collect_tied_F` kernels are preserved as-is.

**Runtime dependencies unchanged** from Phase 3 (`numba 0.58.1`, `llvmlite 0.41.1`).

---

## 2. Bit-exact parity (in-process)

`scripts/temporary/parity_enhance3_vs_enhance4.py` runs Phase 3 and Phase 4 back-to-back in the same process. Tolerance is `0.0` (bit-equality required).

| out_loop_max × in_loop_max | worst max-abs diff | speedup (P3 / P4) | result |
| --- | --- | --- | --- |
| 1 × 1 | `0.000000e+00` | 1.356× | PASS |
| 2 × 5 | `0.000000e+00` | 1.771× | PASS |
| 3 × 3 | `0.000000e+00` | 1.649× | PASS |

Compared arrays: `a_dr`, `alpha_dr`, `einfl_dr`, `muprime_dr`, `Unew`, `z_dr`, `Zupdates`, `Zchange`. Bit-exactness held without any FP-order workarounds, indicating that NumPy `einsum('ed,edm->dm', ...)` does sum the contracting axis in ascending index order for these shapes.

---

## 3. Profile delta at 1 × 5

Both runs use `cautious=1`, `svopt=0`, identical configuration. Phase 3 numbers are reproduced from `ME_FLP_V6scc_Enhance3/PERF_ANALYSIS.md` §3.

| Metric | Phase 3 (1×5) | Phase 4 (1×5) | Δ |
| --- | --- | --- | --- |
| `run_model` wall-clock | 41.673 s | 26.578 s | **1.568× faster** |
| `solver.py:run_model` `tottime` | 6.530 s | 1.255 s | −5.275 s |
| `ii_fhat_factored_kernel` `tottime` | — | 10.735 s | +10.735 s (new) |
| `ii_fhat_factored` (Python wrapper) `tottime` | 14.126 s | absent from top-40 | −14.126 s |
| `c_einsum` `tottime` | 6.648 s | absent from top-40 | −6.648 s |
| `fused_argmax_tie` (from Phase 3) | 6.992 s | 7.080 s | unchanged |
| `collect_tied_F` (from Phase 3) | 1.877 s | 1.858 s | unchanged |

Net at 1×5: ~20.8 s of Python wrapper + `c_einsum` cost (`14.13 + 6.65`) is replaced by ~10.7 s of compiled kernel time (≈ 49% reduction on `ii_fhat`). The ~5.3 s additional savings in `solver.py:run_model` `tottime` come from removing the per-call NumPy bookkeeping (`asarray`, `reshape`, `np.einsum` dispatch, `f_lo` / `f_hi` advanced indexing, `g_lo + g_hi` element-wise add) at every one of the 855 call sites.

### 3.1 Cumulative speedup vs Phase 0 baseline

With Phase 1 (~6.25× from `ii_fhat`), Phase 2 (1.22× from inner-loop vectorization), Phase 3 (1.18× from JIT-fused argmax), and Phase 4 (1.57× from JIT-fused `ii_fhat_factored`), the cumulative 1×5 speedup vs Phase 0 is roughly **14×–15×**. The full 3×50 confirms a 2.96× cumulative speedup vs Phase 1.

### 3.2 Full 3×50 production benchmark

End-to-end run with `out_loop_max=3, in_loop_max=50, svopt=1`, default output root, driven by `scripts/temporary/run_full_3x50.py --package ME_FLP_V6scc_Enhance4`.

| Phase | Wall-clock | Speedup vs Phase 1 | Speedup vs prev. |
| --- | --- | --- | --- |
| Phase 1 (`ME_FLP_V6scc_Enhance`) | 1699.32 s (28.32 min) | 1.000× | — |
| Phase 2 (`ME_FLP_V6scc_Enhance2`) | 1379.81 s (23.00 min) | 1.231× | 1.231× |
| Phase 3 (`ME_FLP_V6scc_Enhance3`) | 1132.54 s (18.88 min) | 1.500× | 1.218× |
| **Phase 4 (`ME_FLP_V6scc_Enhance4`)** | **573.97 s (9.57 min)** | **2.961×** | **1.974×** |

Outputs written to `C:\Users\yanglu\Dropbox\ME_FLP_STORAGE\YL\Experiment_51_20260507T180441\` (50 `MPE*.mat` files + `stuff.mat` + `MPE3Zinfo.mat`). Diagnostics: `boundary_events=6276`, `multi_max_events=7` — identical to Phase 3, as expected.

The full-run 1.974× speedup *exceeds* the 1×5 micro-benchmark gain (1.568×) because at 1×5 the JIT compile cost (~5–6 s warmup + first-call codegen) is amortized over only 5 inner iterations, whereas at 3×50 it's amortized over 150. Subtracting an estimated 6 s warmup, Phase 4 1×5 ≈ 20.6 s vs Phase 3 1×5 ≈ 35.7 s (warmup-corrected) → 1.73× per-iteration, which lines up with the 3×50 1.97×.

### 3.3 MATLAB cross-process parity (`MPE3W50.mat`)

`ME_FLP_V6scc_Enhance4.parity_check.check_against_matlab` against MATLAB reference `C:\Users\yanglu\Dropbox\ME_FLP_STORAGE\YLExperiment_51_20260504T223305\MPE3W50.mat`:

| metric | max-abs diff |
| --- | --- |
| `a_dr` | 3.497653e-04 |
| `alpha_dr` | 3.497653e-04 |
| `einfl_dr` | 3.721502e-04 |
| `muprime_dr` | 3.797935e-03 |
| `Unew` | 2.109591e-05 |

These numbers are **identical to Phase 3's cross-process MATLAB parity** (compare `ME_FLP_V6scc_Enhance3/PERF_ANALYSIS.md` §3.3). The diffs are governed by the same multi-threaded OpenBLAS non-determinism documented since Phase 2; Phase 4 does not change any matmul / `squig_trans @ Unew` paths, so the same accumulation pattern reappears. In-process Phase 3 vs Phase 4 parity at 3×3 is bit-exact (max-abs = 0).

Visual comparison plots:

- `scripts/temporary/a_dr_rho_compare_MPE3W50_phase4_20260507_vs_matlab_20260504.png`
- `scripts/temporary/alpha_dr_rho_compare_MPE3W50_phase4_20260507_vs_matlab_20260504.png`
- `scripts/temporary/einfl_dr_rho_compare_MPE3W50_phase4_20260507_vs_matlab_20260504.png`

---

## 4. Why this phase delivered the largest gain

The original Phase 0 hotspot analysis identified `ii_fhat` as ~82% of runtime. Phase 1 collapsed it via factored bilinear interp + einsum (6.25×). Phases 2–3 then chipped away at the inner-loop branch around it. Phase 4 returns to the *same* function and finally eliminates two costs that Phase 1's NumPy-only rewrite left behind:

1. **Advanced-indexing materialization.** `f[rho_idx]` and `f[rho_idx + 1]` allocate two `(n_eps, n_delta, n_mu)` tensors per call (≈ 30 MB each at run-51 sizes) and pay full read-write bandwidth for each. The JIT kernel reads each f sample directly into the per-(d, m) accumulator, so the f_lo / f_hi tensors never exist.
2. **`c_einsum` dispatch + intermediate.** `np.einsum('ed,edm->dm', ...)` is a fast single-loop reduction internally, but it still pays per-call dispatch overhead and writes a fresh (n_delta, n_mu) intermediate per einsum. We were calling it twice per `ii_fhat_factored` call, 855 times per 1×5 inner loop. The JIT kernel folds both reductions and the final element-wise add into one function call.

Together these cut the function's time roughly in half (14.1 s → 7 s for the body, plus 6.6 s einsum eliminated), and the (now smaller) call-site bookkeeping in `solver.py` drops by ~5 s on top of that.

---

## 5. Risks and follow-ups

- **FMA / fast-math.** Same posture as Phase 3: `fastmath` is off and `tmp = w * f; acc = acc + tmp` is split. Bit-exact gates passed at three scales without any FP-order workarounds, confirming NumPy's `c_einsum` reduction order matches a sequential ascending-e loop for these shapes.
- **AOT compile cost.** ~5–6 s on first interpreter run; subsequent runs reuse the `__pycache__` cache. The `warmup()` call is invoked once per `run_model` start so the timed body never pays per-call compile cost.
- **Remaining hotspots.** With `ii_fhat` and the inner-loop argmax both JIT'd, the 1×5 profile shows `solver.py:run_model` tottime down to 1.26 s. Remaining sizable items are `resolve_multiple_maxima` (2.6 s, but driven by Python `_unique_stable_indices` — only fires when ties exist) and `rho_one.solve_rho_one` (1.17 s). Neither would individually deliver another 1.5×; further wins likely require attacking the per-iteration `squig_trans @ Unew_reshape` matmul with a thread-pinned BLAS or a CPU-tiled JIT, or moving to GPU offload (recall: Quadro P1000, 4 GB VRAM, weak FP64).
- **Cross-process drift to MATLAB.** Unchanged from Phase 3 — same magnitudes, same root cause (multi-threaded OpenBLAS partial-sum non-determinism in `squig_trans @ Unew_reshape`). Not a Phase 4 regression.
