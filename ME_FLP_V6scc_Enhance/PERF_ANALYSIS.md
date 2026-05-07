# ME_FLP_V6scc Performance Analysis & Implementation Plan

**Goal:** at least 2x end-to-end runtime speedup on `--run-no 51`, with numerical parity preserved at the current `parity_check.py` tolerance, on a Quadro P1000 (4 GB VRAM, Pascal compute 6.1).

**Baseline measurement.** Profile collected with `out_loop_max=1`, `in_loop_max=1`, `svopt=0` (saving disabled to isolate compute) on the workbook `KLexperiments_ME_FLP_PD.xlsx`, run 51. Driver: `scripts/temporary/profile_minimal_run.py`. Raw artifacts: `profile_run51_o1i1.prof`, `profile_run51_o1i1_cumulative.txt`, `profile_run51_o1i1_tottime.txt`, `profile_run51_o1i1_summary.txt`.

**Effective grid for run 51** (observed at runtime):
- `n_squig=3`, `n_rho=21`, `n_st = n_squig*n_rho = 63`
- `n_mu = 168` (derived from `xgrid` over `[x_lower, xstar1]` with step `x_step`; `n_mu` in workbook is null and overridden inside `solver.py`)
- `n_delta=501`, `n_eps=51`, `n_mup=n_mu=168`
- Body-loop states (between the two analytical boundary blocks): 57

**Total wall-clock for one (out, in) iteration:** 78.88 s.
- `mu_bound` calibration (one-time per `run_model`): ~0.55 s — negligible at scale.
- Single-iteration body cost (what scales with `out_loop_max * in_loop_max`): ~78 s.

---

## 1. Hotspot Analysis

### 1.1 Top-level breakdown (cumulative time, 78.88 s total)

| Rank | Function | File:Line | Calls | Cum (s) | % of total | Class |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `ii_fhat` | `core_functions.py:107` | 171 | **64.75** | **82.1%** | CPU + memory |
| 2 | `scipy.interpolate.interpn` (inside `ii_fhat`) | `_rgi.py:516` | 171 | 45.62 | 57.8% | CPU |
| 2a | `_find_indices` (binary search inside scipy) | `_rgi_cython.find_indices` | 171 | 19.24 | 24.4% | CPU |
| 2b | `_prepare_xi` (allocation + validation inside scipy) | `_rgi.py:364` | 171 | 16.12 | 20.4% | memory + CPU |
| 3 | `ndarray.reshape` (mostly `order='F'`) | NumPy | 66,140 | 13.18 | 16.7% | memory |
| 4 | `ufunc.reduce` (min/max/sum) | NumPy | 32,953 | 10.98 | 13.9% | CPU |
| 5 | `getmax` | `core_functions.py:169` | 9,576 | 5.73 | 7.3% | CPU + memory |
| 6 | `np.isclose` (tie detection inside `getmax`) | NumPy | 9,576 | 3.89 | 4.9% | CPU |
| 7 | `column_stack` (query construction in `ii_fhat`) | `shape_base.py:612` | 173 | 3.84 | 4.9% | memory |
| 8 | `resolve_multiple_maxima` | `diagnostics.py:69` | 9,576 | 2.10 | 2.7% | CPU + Python |
| 9 | `compute_mu_bounds` (whole pre-loop) | `mu_bounds.py:22` | 1 | 0.55 | 0.7% | one-time |
| 10 | `solve_rho_one` | `rho_one.py:31` | 1,367 | 0.48 | 0.6% | CPU |

(Counts per single (out, in) iteration; `mu_bound` calls are once per `run_model`.)

### 1.2 Where the cycles go, by call-site

**(a) `ii_fhat` — 82% of runtime, 171 calls.** Per call: ~379 ms. Each call rebuilds:
- `points = (rh_grid=d_rho, mu_grid=d_mu)` — fixed across the entire (out_loop, in_loop) iteration;
- `query = column_stack((rhq_F, mupq_F))` — `rhq_F`, `mupq_F` are 4.29M-row vectors (`n_eps*n_delta*n_mup = 51*501*168`), and `column_stack` materializes a new 4.29M×2 array;
- `interpn(points, values, xi=query, method='linear', bounds_error=True)` — scipy re-does index search (`_find_indices`) for all 4.29M query points, even though the query coordinates are unchanged across the 3 calls per state and partially redundant across the 57 body states.

```107:131:ME_FLP_V6scc/core_functions.py
def ii_fhat(
    rh: np.ndarray,
    mu: np.ndarray,
    f: np.ndarray,
    rhq: np.ndarray,
    mupq: np.ndarray,
    fdens: np.ndarray,
    n_eps: int,
    n_delta: int,
    n_mup: int,
) -> np.ndarray:
    """Port of iiFhat.m interpolation+integration routine."""
    rh_grid = np.asarray(rh, dtype=float).reshape(-1)
    mu_grid = np.asarray(mu, dtype=float).reshape(-1)
    points = (rh_grid, mu_grid)
    vals = np.asarray(f, dtype=float)
    rhq_arr = np.asarray(rhq, dtype=float)
    mupq_arr = np.asarray(mupq, dtype=float)
    query = np.column_stack((rhq_arr.reshape(-1, order="F"), mupq_arr.reshape(-1, order="F")))
    interp_vals = interpn(points=points, values=vals, xi=query, method="linear", bounds_error=True)
    interp_cube = np.asarray(interp_vals, dtype=float).reshape((n_eps, n_delta, n_mup), order="F")
    dens = np.asarray(fdens, dtype=float).reshape(n_eps, 1, 1)
    return np.sum(dens * interp_cube, axis=0)
```

Key invariants the current code does **not** exploit:
- `(d_rho, d_mu)` is fixed within a (out, in) iteration.
- `mupq` is a constant `(n_eps*n_delta) × n_mup` array reused across **all** body states (it depends only on `d_mup`).
- `rhoq1[:,:,i_rhoi]` and `rhoq2[:,:,i_rhoi]` depend only on `i_rhoi`, not on `i_squig`, so they repeat `n_squig=3` times per `i_rhoi` across the body loop.
- The three `ii_fhat` calls per body state share the same query coordinates and differ only in `f` (Mhat1, Mhat2, or Uhat).

This means **the bilinear-interpolation index/weight tables can be computed once per `i_rhoi` and reused across 3 different `f` arrays and across the n_squig states sharing that `i_rhoi`**, eliminating the bulk of the `_find_indices` and `_prepare_xi` cost.

**(b) `getmax` + `resolve_multiple_maxima` — ~7.8 s, 13% of runtime.** Called 9,576 times = `57 body states × 168 mu values`. Inside the `for i_mu in range(n_mu)` loop:

```515:566:ME_FLP_V6scc/solver.py
                for i_mu in range(n_mu):
                    mu_lp = d_mu[i_mu]
                    om_st_mu = mu_lp * om_st + urhs

                    if opt.cautious == 0 and in_loop > 2:
                        loc = int(np.argmax(om_st_mu.reshape(-1, order="F"))) + 1
                        cloc = int(np.ceil(loc / config.n_delta))
                        rloc = int(loc - (cloc - 1) * config.n_delta)
                        PTkeep[st, :, i_mu] = [rloc, cloc]
                    else:
                        if mu_lp == 0:
                            valid = (d_delta.reshape(-1) <= 0.0)
                            _, loc_raw, rloc_arr_raw, cloc_arr = getmax(
                                om_st_mu[valid, :], atol=opt.getmax_atol, rtol=opt.getmax_rtol
                            )
                            ...
                        else:
                            _, loc, rloc_arr, cloc_arr = getmax(om_st_mu, atol=opt.getmax_atol, rtol=opt.getmax_rtol)
                        if len(rloc_arr) > 1:
                            bugcount, rloc, cloc, evt = resolve_multiple_maxima(...)
```

Issues:
- `om_st` and `urhs` are constant w.r.t. `mu_lp`, but the linear combo `mu_lp * om_st + urhs` is recomputed in pure-NumPy 168 times per state (vectorizable to a single tensor op).
- `getmax` reshapes with `order="F"` (forces a copy of an 84k-element array) and then runs `np.isclose(flat, maxval, atol=1e-16)` — the count of `resolve_multiple_maxima` calls (9,576 = 100%) shows that `np.isclose` flags ties for **every** `(st, i_mu)` pair under the default `atol=1e-16`, but `multi_max_events=0` shows that after the unique-stable filter all surviving ties collapse. The slow path is the default path.
- `resolve_multiple_maxima` runs in pure Python with `np.unique` calls and another `reshape(order='F')` per invocation.

**(c) `reshape(order='F')` copies — 13.2 s, 17% of runtime.** Many MATLAB-port idioms (`vec(x)`, Fortran-order reshapes inside `ii_fhat` and `getmax`) force a strided→contiguous copy each call. In particular `core_functions.py:127` and `core_functions.py:129` allocate two 4.29M-element copies per `ii_fhat` call (×171 = 1.5 GB of churned memory per iteration).

**(d) `mu_bound` — 0.55 s, fixed cost.** It runs `compute_mu_bounds` with 36 outer rounds × ~38 inner rounds = ~1,367 `solve_rho_one` calls. Already cheap; will not be a bottleneck at any realistic `out_loop_max × in_loop_max`.

### 1.3 Memory order issues

- `Unew.reshape(..., order='F')` → `squig_trans @ ...` → `.reshape(..., order='F')` at `solver.py:621-623` forces two physical copies per (out, in) iteration. The `squig_trans` matrix is (3,3) so the matmul is cheap; the copies dominate.
- `om_st_mu.reshape(-1, order='F')` inside `getmax` is the dominant copy in the inner loop because it executes 9,576 times.
- `np.repeat(...)` patterns in `rho_one.py:147-156` materialize copies but are called only ~1,367 times in `mu_bound` and 1× in the body — small at scale.

### 1.4 What is **not** a bottleneck

- `solve_rho_one` (analytical block): 1.4 s cumulative across 1,367 calls in `mu_bound` plus 1 in the body. Well-implemented dense linear algebra.
- `boundary_check` and event collection: negligible.
- Disk I/O (`save_iteration_mat`): not in this profile (svopt=0). At svopt=1 each iteration writes ~10 small `.mat` files; this is small relative to compute (a few hundred ms) but should be measured separately when scaling out.

---

## 2. Strategy Comparison Table

Effort: S=small (<1 day), M=medium (1–3 days), L=large (>3 days). Speedup is **expected over the current 78 s/iter baseline**; speedups stack roughly multiplicatively if they target disjoint hotspots, but the table reports each strategy's standalone effect.

| ID | Strategy | Targets | Expected speedup (standalone) | Effort | Parity risk | New deps | VRAM | Maintainability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **A1** | Cache `(d_rho, d_mu)` interp index/weight tables once per `i_rhoi`; replace scipy `interpn` with a hand-rolled bilinear gather inside `ii_fhat` | hotspot 1 (82%) | **3.0–4.5x** end-to-end | M | low if formula matches scipy linear; medium if we change boundary handling | none (NumPy only) | 0 | high |
| **A2** | Vectorize `for i_mu in range(n_mu)` into a single tensor op; vectorized argmax + lazy tie-cleanup | hotspots 5,6 (~13%) | 1.10–1.20x end-to-end | S | low (logic preserved; tie-cleanup path unchanged for ambiguous cases) | none | 0 | high |
| **A3** | Eliminate redundant `reshape(order='F')` copies (precompute Fortran-contiguous views; reuse `mupq`; reuse `rhoq1`/`rhoq2` per `i_rhoi` across `n_squig` states) | hotspots 3,7 (~22%) | 1.10–1.20x end-to-end | S | low | none | 0 | high |
| **B** | Numba `@njit` on the `(st, i_mu)` body loop and on a custom bilinear interp kernel | residuals after A | 1.20–1.50x on top of A | M | low (pure-float arithmetic; deterministic) | `numba` (Python 3.8 supported on numba ≤0.58) | 0 | medium (compile cache; debugger gets harder) |
| **C** | Joblib/loky parallel over the 57 body states inside each (out, in) iteration | the per-iter body | 1.5–3x on top of A on a typical 4–8 core CPU | M | low (deterministic; no shared mutable state if outputs assembled at the end) | `joblib` | 0 | medium (process startup; pickling overhead) |
| **D1** | CuPy port of `ii_fhat` only (FP64), keeping rest on CPU | hotspot 1 | 1.5–3x end-to-end at best on Quadro P1000 (FP64 ~150 GFLOPS, host↔device transfer dominates) | M | low if FP64 throughout | `cupy` for CUDA 11/12 | ~0.5 GB at peak per state, well under 4 GB | medium (CUDA toolkit dependency, fallback path) |
| **D2** | Full-iteration GPU residency: keep Mhat/Uhat/PTkeep on device; per-state ii_fhat batched; argmax on device | iteration body | 3–6x end-to-end at best | L | medium (FP32 mixed-precision risks parity; FP64 keeps parity but P1000 FP64 is weak) | `cupy` | up to 2.5 GB for batch-of-state working set (fits 4 GB) | medium |
| **D3** | PyTorch alternative to D2 (FP32 with TF-style autograd off) | iteration body | similar to D2 in FP32, **likely breaks parity** | L | high (FP32 only on Pascal CC 6.1; FP64 in PyTorch on CC 6.1 is ~1/32 throughput) | `torch` (CUDA build) | similar | medium |

**Why A is the headline strategy.** It attacks 82% of the runtime, requires zero new dependencies, has a tight numerical-parity guarantee (replicate scipy's `linear` formula bit-for-bit), and—per the profile—standalone is expected to clear the 2x bar. B/C/D are stack-on options to extend headroom.

**Why GPU is risky on the P1000.** Pascal compute capability 6.1:
- FP64 throughput is ~1/32 of FP32 → ~150 GFLOPS FP64. A modern desktop CPU using SIMD already reaches comparable FP64 throughput for the kinds of ops here. The win is in `interpn` index search (cache-friendly on GPU) but host↔device transfer of ~35 MB per state per call quickly dominates if not batched.
- FP32 would yield large speedup but `parity_check.py` currently uses `np.max(np.abs(.))` directly. Switching to FP32 will inflate diffs by ~1e-7, which the user has stated must not happen.
- 4 GB VRAM is enough for a single-iteration working set if we batch carefully, but precludes naive "everything on device".

---

## 3. Recommended Strategy

**Adopt A1 + A2 + A3 first.** This is a single coherent NumPy refactor that:
1. Lifts the interpolation-table construction out of `ii_fhat` into a pre-loop step keyed by `i_rhoi`.
2. Replaces the per-call `interpn` with a fancy-indexing bilinear gather (4 corner lookups + 4 weight multiplies, all vectorized).
3. Vectorizes the `for i_mu` inner loop into a single tensor op with vectorized argmax and a lazy slow-path for ties.
4. Removes redundant `reshape(order='F')` copies by pre-flattening once.

**Expected outcome:** 3–4x end-to-end on a single (out, in) iteration, comfortably exceeding the 2x target. Validation against MATLAB `MPE<o>W<i>.mat` artifacts via `parity_check.py` at the existing tolerance.

**If A still under-delivers** (e.g. <2x because of a different hot spot exposed once `ii_fhat` shrinks), proceed to B (Numba) before C/D. GPU is the last resort given the parity constraint and the P1000's weak FP64.

---

## 4. Phased Implementation Plan

Each phase ends with a **gate**: parity verified at the current tolerance and a measured speedup against the Phase 0 baseline. If a phase fails its gate, the cumulative state is rolled back via Git and we re-evaluate.

### Phase 0 — Profiling & baseline (DONE)
- Record baseline wall-clock and grid sizes for `--run-no 51`, `out_loop_max=in_loop_max=1`.
- Driver: `scripts/temporary/profile_minimal_run.py`.
- Artifacts: `profile_run51_o1i1.{prof,_cumulative.txt,_tottime.txt,_summary.txt}` in `ME_FLP_V6scc_Enhance/`.
- **Baseline:** 78.88 s/iter, 9,576 `getmax` calls, 171 `ii_fhat` calls, 64.75 s in `ii_fhat` cumulative.

### Phase 1 — `ii_fhat` precompute + manual bilinear (Strategy A1) — **DONE (2026-05-06)**
- **Objective:** drop `ii_fhat` cumulative time from 64.75 s to ≤15 s (≥4x on the kernel) without changing the numerical result.
- **What landed:**
  - `ME_FLP_V6scc_Enhance/core_functions.py`: added `bilinear_axis_table(grid, query)` and `ii_fhat_factored(f, rho_idx, rho_w, mu_idx, mu_w, fdens)`. Legacy `ii_fhat` left in place, no longer called.
  - `ME_FLP_V6scc_Enhance/solver.py`: precomputes `(mu_idx, mu_w)` once per `run_model` (`d_mu`, `d_mup` are fixed after `mu_bound`), and `(rho_idx, rho_w)` once per `i_rho` for both `rhoprime1` and `rhoprime2` (each constant for the entire run). Body loop's three `ii_fhat` call sites now use `ii_fhat_factored` with cached tables.
  - Removed dead `rhoq1 / rhoq2 / mupq` initial assignments outside the body loop and the per-state `rhoq1 / rhoq2` rebuild inside it.
- **Validation:**
  - Unit parity test `scripts/temporary/test_ii_fhat_factored.py`: 5 random trials at the run-51 grid sizes (n_rho=21, n_mu=168, n_eps=51, n_delta=501). **Worst max-abs diff vs legacy scipy `interpn`: 5.55e-16** (1 ULP). **Standalone kernel speedup: 15.45x** (2.205 s → 0.143 s for 5 calls including table builds).
  - End-to-end parity gate `scripts/temporary/parity_old_vs_new.py` at `out_loop=in_loop=1`. **Worst max-abs diff across all 8 result arrays: 1.30e-18** (sub-ULP — well below double-precision machine epsilon 2.22e-16).
- **Measured outcome (`profile_run51_o1i1_phase1_*` artifacts):**
  - **End-to-end wall-clock: 78.88 s → 12.31 s on a clean run, 13.87 s under cProfile. Speedup: 6.25x** (clean) / 5.69x (profiled). Far past the 2x stop condition.
  - `ii_fhat_factored`: 3.87 s cumulative (vs legacy 64.75 s) — **16.7x kernel-level reduction**, 23 ms/call vs 379 ms/call.
  - `bilinear_axis_table` precompute: 0.045 s total for 43 calls (one mu-axis table + 21 × 2 rho-axis tables). Negligible.
  - `mu_bound`: 0.93 s (was 0.55 s; small regression from extra strict-monotonicity validation in `bilinear_axis_table`; absolute cost still trivial).
- **Phase 1 acceptance gate: PASS.** Sub-ULP parity preserved, 6.25x clean-run speedup ≫ 1.8x rollback threshold.
- **Full-bench validation (2026-05-07).** Re-ran `parity_old_vs_new.py --out-loop-max 2 --in-loop-max 5` (10 body iterations) to confirm Phase 1 holds at scale. Log: `parity_o2i5.log`.
  - Reference wall-clock: 794.63 s. Phase 1 wall-clock: 115.16 s. **Speedup: 6.90x** — slightly better than the 1×1 result because the constant precompute (`mu_bound`, bilinear-table build) amortizes across more iterations.
  - Worst max-abs diff across all 8 result arrays: **1.39e-17** (still ~16x below double-precision machine epsilon). `Zupdates` and `z_dr` parity confirms the outer-loop update path (untouched by Phase 1) still produces equivalent results when fed the new inner-loop outputs.
  - Linear projection to a `out=10, in=50` run: ref ≈ 6.6 h, Phase 1 ≈ 58 min on this machine.
- **New post-Phase-1 hotspot ranking** (cumulative, 13.87 s total):

| Rank | Function | Cum (s) | % | Notes |
| --- | --- | --- | --- | --- |
| 1 | `getmax` | 4.17 | 30% | inner `i_mu` Python loop, 9,576 calls |
| 2 | `ii_fhat_factored` | 3.87 | 28% | now mostly `np.einsum` and fancy indexing |
| 3 | `run_model` body (Python orchestration) | 2.74 | 20% | self-time outside callees |
| 4 | `resolve_multiple_maxima` | 1.88 | 14% | per-`i_mu` tie cleanup |
| 5 | `reshape` (`order='F'` copies in getmax / resolve) | 1.55 | 11% | scoped to inner loop now |
| 6 | `mu_bound` | 0.93 | 7% | one-time |
| 7 | `solve_rho_one` | 0.82 | 6% | inside `mu_bound` |

### Phase 2 — Vectorized `(om_st_mu, getmax, resolve_multiple_maxima)` (Strategies A2 + A3)
- **Objective:** reduce `getmax` + tie-cleanup cumulative from ~7.8 s to ≤2 s and remove `reshape(order='F')` hot copies.
- **Files / functions touched:**
  - `solver.py:515-565`: replace `for i_mu` Python loop with a single tensor op
    - Build `om_st_mu_all = d_mu[:, None, None] * om_st[None, :, :] + urhs[None, :, :]` (shape `(n_mu, n_delta, n_mup)`).
    - Compute argmax over flattened `(n_delta * n_mup)` axis vectorised across `i_mu` (shape `(n_mu,)`).
    - For the `mu_lp == 0` case, mask `valid = d_delta <= 0` ahead of time and apply a 3D mask (only `i_mu == 0` slot in default ordering needs it).
    - Detect ties cheaply (`(om_st_mu_all == max[:, None, None]).sum(axis=(1,2)) > 1`) and dispatch ONLY tied `i_mu` indices to a Python-level tie-cleanup path that calls `resolve_multiple_maxima`.
  - `core_functions.py:169` (`getmax`): keep as-is for the slow-path tie callers; or split into `argmax_strict` (fast path) and `getmax_with_ties` (slow path).
- **Validation:**
  - Add `scripts/temporary/test_getmax_vectorized.py`: random `om_st`/`urhs` matrices with hand-crafted ties (including the `mu_lp==0` boundary), bit-equality against current implementation.
  - Re-run full `parity_check.py` end-to-end.
- **Stop / rollback criterion:** speedup contribution ≥1.10x; otherwise revert.

### Phase 3 — Iteration-body micro-optimizations (Strategy A3 finishing)
- **Objective:** absorb the residual MATLAB-ism overhead.
- **Files / functions touched:**
  - `core_functions.py:11-13` (`vec`): keep but mark internal call-sites where Fortran-contiguous semantics are not strictly needed and use C-order `ravel()` instead.
  - `solver.py:336-338` and `:459-460`: precompute `mupq` once before the outer loop (it never changes); reuse `rhoq1[:, :, i_rhoi]` and `rhoq2[:, :, i_rhoi]` flattened views across the n_squig states sharing `i_rhoi`.
  - `solver.py:621-623`: store `Uhat` already in C-order to drop one of the two `reshape(order='F')` copies (verify with parity test).
- **Validation:** end-to-end parity check.
- **Stop / rollback criterion:** measured speedup contribution ≥1.05x; otherwise leave individual hunks out.

### Phase 4 (optional) — Numba JIT on hot paths (Strategy B)
- **Objective:** if cumulative speedup at end of Phase 3 is <2x, JIT-compile the bilinear gather + the body loop.
- **Files / functions touched:** add `core_functions_jit.py` with `@njit(cache=True)` versions, gated by a try-import of `numba`.
- **Constraints:** Python 3.8 → use `numba ≥0.56, ≤0.58`.
- **Validation:** parity test + speedup measurement against Phase-3 baseline.
- **Stop / rollback criterion:** ≥1.2x or skip.

### Phase 5 (optional) — CPU parallelism (Strategy C)
- **Objective:** parallelize the 57 body states with `joblib` if Phase 4 still leaves headroom.
- **Constraints:** state writes are disjoint by `i_squig, i_rhoi` index, so it is safe; outputs must be assembled deterministically in `i_squig, i_rhoi` order. Process startup cost is paid once per process pool, amortized over many (out, in) iterations.

### Phase 6 (optional, last resort) — CuPy GPU offload (Strategy D1)
- **Objective:** move only `ii_fhat` to GPU (FP64), leaving everything else on CPU; keep parity by staying in FP64.
- **VRAM pre-check:** per-iteration peak working set on device ≤ 1.5 GB (fits 4 GB easily). Add an explicit pre-check at startup that aborts if `cupy.cuda.runtime.memGetInfo()` reports <2 GB free.
- **Fallback:** code path must auto-fallback to the CPU bilinear gather from Phase 1 when CuPy is unavailable or VRAM check fails (no silent fallback — emit a warning per fail-fast policy).
- **Stop / rollback criterion:** speedup ≥1.5x over Phase 5 result, parity preserved; otherwise drop.

### Stop condition (overall)
Stop adding phases as soon as **cumulative speedup ≥ 2x** against the Phase 0 baseline (78.88 s/iter) **and** `parity_check.py` reports metrics ≤ current MATLAB-compare values for `MPE<o>W<i>.mat` reference files. We expect to hit this after Phase 1 alone.

### Acceptance criteria
1. `parity_check.py` metrics for the agreed reference (e.g. `MPE3W50.mat`) are unchanged or smaller than the current control baseline in `scripts/temporary/control_runs_w1w5_atol1e16_iifhatfix/`.
2. CLI surface (`run_me_flp_v6.py` arguments) is unchanged.
3. Fail-fast errors are preserved; no silent fallbacks introduced.
4. End-to-end wall-clock for `--run-no 51 --out-loop-max 1 --in-loop-max 1` ≤ 39 s on the same machine (2x of 78.88 s).
5. New dependencies (if any) recorded in a `requirements.txt` (currently absent from the repo) and pinned.

---

## 5. Risks & Open Questions

### Risks
- **Bilinear formula equivalence (Phase 1).** scipy's `interpn(method='linear')` uses a tensor-product linear interpolation with a specific boundary-out-of-grid policy (`bounds_error=True` here). The hand-rolled gather must replicate the same out-of-grid behaviour, or we risk silently masking a query that scipy would have rejected. Mitigation: keep `bounds_error=True` semantics — assert all query points lie in `[d_rho.min(), d_rho.max()] × [d_mu.min(), d_mu.max()]` before dispatching.
- **Tie semantics (Phase 2).** Vectorized argmax via `np.argmax` returns the first occurrence on the flattened array, while `getmax` currently returns **all** ties. Phase 2 must keep dispatching to `resolve_multiple_maxima` for any `i_mu` where ties exist; the speedup comes from the fact that ties are rare in *post-cleanup* terms (`multi_max_events=0`) even though they are frequent in *pre-cleanup* (`isclose` with `atol=1e-16`) terms.
- **Memory peak (Phase 2).** A `(n_mu, n_delta, n_mup) = (168, 501, 168)` float64 tensor is ~108 MB. Materializing it per state is fine but should be allocated once and reused, not freshly allocated 57 times per iteration.
- **GPU strategy (Phase 6).** P1000's FP64 ceiling makes the speedup ceiling modest. If we go down this path we must measure carefully to ensure GPU FP64 is faster than CPU FP64 + AVX2 once host↔device overhead is included.

### Open questions
- **Q1.** Is there a tighter parity acceptance than "≤ current `parity_check.py` outputs"? `parity_check.py` itself does not assert any tolerance; it only reports `max_abs_diff` numbers. Should we adopt a hard pass/fail tolerance (e.g., `< 5e-13`) for CI?
- **Q2.** What is the typical full-run `out_loop_max × in_loop_max`? If it is 10×50 (per the parity-check examples in the README), single-iteration work scales to ~11 hours at baseline, ~3 hours at 4x. Worth noting because parallelism strategy (Phase 5) becomes more attractive at higher iteration counts.
- **Q3.** Does the project have a CI runner where we can lock in benchmark numbers, or should we add a lightweight `scripts/temporary/bench_run51_o1i1.py` driver to be re-run locally after each phase?
- **Q4.** `n_mu` is null in the workbook for run 51 (overridden by `solver.py:165-178`). Should we surface the runtime value in `RunConfig` for downstream tooling, or leave as is?
