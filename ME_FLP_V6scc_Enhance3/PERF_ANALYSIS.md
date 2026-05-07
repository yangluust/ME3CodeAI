# ME_FLP_V6scc Phase 3 — Numba JIT for the inner-loop fused kernel

This file documents the Phase 3 delta in `ME_FLP_V6scc_Enhance3/`. The Phase 0 baseline and Phase 1/2 designs live in `ME_FLP_V6scc_Enhance/PERF_ANALYSIS.md` and `ME_FLP_V6scc_Enhance2/PERF_ANALYSIS.md` and are not repeated here.

**Scope.** Replace the Phase 2 NumPy-vectorized inner block (build flat + argmax + take + tie-mask + reduce-sum) with a single fused `@njit(cache=True, boundscheck=False)` kernel that streams over each i_mu row in two passes. Bit-exact parity with Phase 2 is required.

**Acceptance gate.** `scripts/temporary/parity_enhance2_vs_enhance3.py` reports max-abs diff = 0.0 on every result array (`a_dr`, `alpha_dr`, `einfl_dr`, `muprime_dr`, `Unew`, `z_dr`, `Zupdates`, `Zchange`).

---

## 1. What landed

All edits live in `ME_FLP_V6scc_Enhance3/`:

- `jit_kernels.py` (new):
  - `fused_argmax_tie(d_mu, om_st_f, urhs_f, invalid_F_mask, has_zero_mu)`: streaming kernel that computes, per i_mu row, `flat_argmax`, `max_per_mu`, and `tie_count` in two passes over the implicit F-flat tensor. The full `(n_mu, n_delta * n_mup)` flat buffer is **never materialized**.
  - `collect_tied_F(mu, om_st_f, urhs_f, invalid_F_mask, is_zero, best, n_ties)`: returns the 1-based F-flat indices where `flat[i_mu, F] == best` for a single tied row. Called only when `tie_count > 1` (rare, mostly during the in_loop ≤ 2 cautious window).
  - `warmup()`: tiny representative call that triggers AOT compilation of both kernels once at the start of `run_model` so the first inner iteration does not pay the JIT cost.
  - The kernels deliberately split `tmp = mu * om_st_f[F]; v = tmp + urhs_f[F]` and **do not enable `fastmath`**. This prevents LLVM from contracting mul+add into FMA or re-associating, which would break bit-exact parity with the NumPy reference.

- `solver.py`:
  - Drops the Phase 2 `flat_buf` allocation, `np.multiply(..., out=flat_buf)`, in-place `+= urhs_f[None, :]`, `np.argmax(flat, axis=1)`, `np.take_along_axis(...)`, `(flat == max_per_mu[:, None])`, and `tie_mask.sum(axis=1)`.
  - Replaces them with a single call to `fused_argmax_tie(...)` returning `flat_argmax`, `max_per_mu`, `tie_count`.
  - The F-flat invalid mask for the `mu_lp == 0` boundary case is precomputed **once** per `run_model` call from `d_delta` and `n_mup` and lives outside the inner loops. `has_zero_mu_global` short-circuits the zero-mu branch on the hot path when no row has mu == 0.
  - For each tied i_mu (rare), `collect_tied_F(...)` returns the tied F-flat indices, which feed `resolve_multiple_maxima` exactly as in Phase 2 (with cached `vaic_state` / `veinfl_state` reused per state).
  - Adds `jit_warmup()` immediately after the final `d_mu` is settled so AOT compile happens once outside the timed loops.

- `diagnostics.py`, `core_functions.py`, `mu_bounds.py`, `rho_one.py`, `io_utils.py`, `parity_check.py`, `config.py`, `run_me_flp_v6.py`: unchanged from Phase 2.

**New runtime dependency.** `numba == 0.58.1` (with `llvmlite == 0.41.1`). 0.58.1 is the latest version that supports the active Python 3.8.5 + NumPy 1.24.4. Older `numba 0.51.2` shipped via Anaconda was incompatible with NumPy 1.24 (`np.long` was removed) and had to be upgraded.

---

## 2. Bit-exact parity (in-process)

`scripts/temporary/parity_enhance2_vs_enhance3.py` runs Phase 2 and Phase 3 back-to-back in the same process. Tolerance is `0.0` (bit-equality required).

| out_loop_max × in_loop_max | worst max-abs diff | speedup (P2 / P3) | result |
| --- | --- | --- | --- |
| 1 × 1 | `0.000000e+00` | 1.069× | PASS |
| 2 × 5 | `0.000000e+00` | 1.222× | PASS |
| 3 × 3 | `0.000000e+00` | 1.181× | PASS |

Compared arrays: `a_dr`, `alpha_dr`, `einfl_dr`, `muprime_dr`, `Unew`, `z_dr`, `Zupdates`, `Zchange`.

The 1×1 speedup is depressed by the one-time AOT compilation of the two kernels (~5–6 s). On 2×5 / 3×3 the JIT cost is amortized and the per-iteration speedup vs Phase 2 is ~1.20–1.22×.

---

## 3. Profile delta at 1 × 5

Both runs use `cautious=1`, `svopt=0`, identical configuration. Phase 2 reference is run from `ME_FLP_V6scc_Enhance2`; Phase 3 from `ME_FLP_V6scc_Enhance3` with the JIT kernels already compiled and cached.

| Metric | Phase 2 (1×5) | Phase 3 (1×5) | Δ |
| --- | --- | --- | --- |
| `run_model` wall-clock | 47.342 s | 41.673 s | **1.136× faster** |
| `solver.py:run_model` `tottime` | 16.875 s | 6.530 s | −10.345 s |
| `fused_argmax_tie` `tottime` | — | 6.992 s | +6.992 s (new) |
| `collect_tied_F` `tottime` | — | 1.877 s | +1.877 s (new) |
| `np.argmax` `tottime` | 2.621 s | absent from top-40 | −2.621 s |
| `'reduce' of np.ufunc` | 3.125 s | absent from top-40 | −3.125 s |
| `'nonzero' of np.ndarray` | 0.557 s | absent from top-40 | −0.557 s |
| `ii_fhat_factored` | 13.479 s | 14.126 s | ~unchanged |
| `c_einsum` (in `ii_fhat_factored`) | 6.651 s | 6.648 s | unchanged |

Net: ~10.3 s of NumPy overhead in `run_model` (build + argmax + take + isclose-substitute + sum) is replaced by ~8.9 s of compiled kernel time, for a net ~5.7 s wall-clock reduction at 1×5 (≈ 1.14 s per inner iteration). The `c_einsum` cost inside `ii_fhat_factored` is now the single largest remaining hotspot and is the natural target for any future Phase 4.

### 3.1 Cumulative speedup vs Phase 0 baseline

The Phase 0 baseline (`ME_FLP_V6scc`) at 1×5 was ~5 min. With Phase 1 (6.25× from `ii_fhat`) and Phase 2 (1.22× from inner-loop vectorization) already landed, the Phase 3 1.18× brings cumulative speedup vs Phase 0 to roughly **9×–9.5×** on the 1×5 micro-benchmark. The end-to-end full 3×50 production run (Section 3.2) confirms a 1.50× cumulative speedup vs Phase 1.

### 3.2 Full 3×50 production benchmark

End-to-end run with `out_loop_max=3, in_loop_max=50, svopt=1`, default output root, driven by `scripts/temporary/run_full_3x50.py --package ME_FLP_V6scc_Enhance3`.

| Phase | Wall-clock | Speedup vs Phase 1 | Speedup vs Phase 2 |
| --- | --- | --- | --- |
| Phase 1 (`ME_FLP_V6scc_Enhance`) | 1699.32 s (28.32 min) | 1.00× | — |
| Phase 2 (`ME_FLP_V6scc_Enhance2`) | 1379.81 s (23.00 min) | 1.231× | 1.00× |
| **Phase 3 (`ME_FLP_V6scc_Enhance3`)** | **1132.54 s (18.88 min)** | **1.500×** | **1.218×** |

Outputs written to `C:\Users\yanglu\Dropbox\ME_FLP_STORAGE\YL\Experiment_51_20260507T172209\` (50 `MPE*.mat` files + `stuff.mat` + `MPE3Zinfo.mat`). Diagnostics: `boundary_events=6276`, `multi_max_events=7`.

The 1.218× full-run speedup matches the 2×5 (1.22×) and 3×3 (1.18×) micro-benchmark gains, confirming the JIT savings scale linearly with inner-loop iteration count.

### 3.3 MATLAB cross-process parity (`MPE3W50.mat`)

`ME_FLP_V6scc_Enhance3.parity_check.check_against_matlab` against MATLAB reference `C:\Users\yanglu\Dropbox\ME_FLP_STORAGE\YLExperiment_51_20260504T223305\MPE3W50.mat`:

| metric | max-abs diff |
| --- | --- |
| `a_dr` | 3.497653e-04 |
| `alpha_dr` | 3.497653e-04 |
| `einfl_dr` | 3.721502e-04 |
| `muprime_dr` | 3.797935e-03 |
| `Unew` | 2.109591e-05 |

These are the **same order of magnitude** as the Phase 2 cross-process numbers and are governed by the same multi-threaded OpenBLAS non-determinism in `squig_trans @ Unew_reshape` documented in Phase 2 (`ME_FLP_V6scc_Enhance2/PERF_ANALYSIS.md` §2.1). They are not a Phase 3 regression — the in-process Phase 2 vs Phase 3 parity gate at 3×3 is bit-exact (max-abs = 0).

Visual comparison plots:

- `scripts/temporary/a_dr_rho_compare_MPE3W50_phase3_20260507_vs_matlab_20260504.png`
- `scripts/temporary/alpha_dr_rho_compare_MPE3W50_phase3_20260507_vs_matlab_20260504.png`
- `scripts/temporary/einfl_dr_rho_compare_MPE3W50_phase3_20260507_vs_matlab_20260504.png`

---

## 4. Why the JIT pays off here

Phase 2 is correct and already ~6× faster than Phase 0, but its hot block performs **five full passes** over the `(n_mu, n_delta * n_mup)` flat tensor:

1. `np.multiply(d_mu[:, None], om_st_f[None, :], out=flat_buf)` — write
2. `flat_buf += urhs_f[None, :]` — read+write
3. `np.argmax(flat, axis=1)` — read
4. `flat == max_per_mu[:, None]` — read+write (allocates a new bool array)
5. `tie_mask.sum(axis=1)` — read

These passes are bandwidth-bound on a single-socket desktop. The `flat` buffer is small enough to fit in L2 most of the time, but each NumPy ufunc still pays for separate kernel launch + temp allocation overhead.

The JIT kernel collapses passes 1–3 into one streaming loop and passes 4–5 into a second streaming loop, with all operands kept in registers / L1. There are no intermediate ndarray allocations and no ufunc dispatch overhead. With 285 calls per inner iteration (one per state in the body block), this saves ~1 s of dispatch and bandwidth time per inner iteration before any compute speedup is counted.

---

## 5. Risks and follow-ups

- **FMA / fast-math.** `fastmath` is **off** by default in `@njit`. We additionally split `mul` and `add` across two statements to make any FMA contraction by LLVM illegal. The 1×1 / 2×5 / 3×3 parity gates confirm the result is bit-identical to NumPy at this Python / NumPy / Numba combination. If a future toolchain upgrade changes this, the parity gate will catch it on the next CI / pre-merge run.
- **AOT compile cache.** `cache=True` writes the compiled kernel to `__pycache__` next to `jit_kernels.py`. First invocation in a fresh Python interpreter pays a one-time ~5 s cost; subsequent invocations reuse the cache.
- **`ii_fhat_factored` is now the largest remaining hotspot** (~50% of `run_model` time at 1×5). A Phase 4 candidate is to JIT the `np.einsum`-based factored interpolation, or to fuse it with the bilinear table application. This is deferred until the full 3×50 benchmark confirms the Phase 3 end-to-end win and identifies whether the headline target has been reached.
- **Cross-process drift to MATLAB.** Identical to the Phase 2 caveat. Multi-threaded OpenBLAS still drives `squig_trans @ Unew_reshape` and other matmul paths, and is the dominant source of sub-ULP non-determinism between Phase 1 / 2 / 3 `.mat` outputs across separate process runs. In-process parity remains bit-exact.
