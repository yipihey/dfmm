# Milestone-1 performance baseline

Per-step and per-phase timings collected through Milestone 1.
Baseline machine: Apple M-series (Tom's laptop), Julia 1.12.6, single-threaded.

The headline number is **full `Pkg.test()` suite runs in ~2m25s** at
1801 + 1 deferred tests on main HEAD (post Phase 7 + cross-phase
smoke). The split is ~70 s of compile/precompile, ~75 s of test
runtime.

## Per-phase wall times (from `Pkg.test()` Test Summary)

| Phase | Tests | Wall time | Per-test |
|---|---:|---:|---:|
| Phase 1 (Cholesky integrator) | 10 | 1.0 s | 0.10 s |
| Phase 2 (bulk + entropy) | 13 | 24 s | 1.85 s |
| ↳ Phase 2 acoustic (slow) | 4 | 14 s | 3.5 s |
| Phase 3 (cold limit B.2) | 162 | 30 s | 0.19 s |
| Phase 4 (energy drift B.1) | 5 | 8 s | 1.6 s |
| Phase 5 (Sod) | 9 | 12 s | 1.3 s |
| Phase 5b (tensor-q) | 72 | 9 s | 0.13 s |
| Phase 6 (cold sinusoid τ-scan) | 70 | 27 s | 0.39 s |
| Phase 7 (steady shock) | 84 | 2.0 s | 0.024 s |
| Phase 8 (stochastic injection) | 140 | 1.5 s | 0.011 s |
| Phase 11 (passive tracer) | 21 | 17 s | 0.81 s |
| Cross-phase smoke | 297 | 0.1 s | 0.0003 s |
| **All M1 phases** | **883** | **132 s** | 0.15 s |
| Track C (eos / diag / io / cal / plot) | 628 | 5.9 s | 0.0094 s |
| Track D (stochastic primitives) | 64 | 2.8 s | 0.044 s |
| Track B setups | 109 | 3.0 s | 0.028 s |
| Regression scaffold | 118 | 0.4 s | 0.0034 s |
| **Full suite total** | **1801** | **~145 s** | **0.080 s** |

The Phase 11 wall time (17 s) is dominated by 1000+ deterministic
steps in the bit-exactness check; the asserts themselves are cheap.

## Per-step Newton solver wall times

Implicit-midpoint Newton with `NonlinearSolve.NewtonRaphson(; autodiff = AutoForwardDiff())`.

| Phase | N segments | Jacobian | Iters/step | Wall/step |
|---|---:|---|---:|---:|
| Phase 2 (Phase 5b sparse path retroactive) | 32 | dense FD | 2 | ~5.8 ms |
| Phase 2 (sparse path) | 64 | sparse FD | 2 | ~16 ms |
| Phase 2 (sparse path) | 128 | sparse FD | 2 | ~63 ms |
| Phase 5 Sod (sparse) | 100 | sparse FD | 2-3 | ~25 ms |
| Phase 5 Sod | 200 | sparse FD | 2-3 | ~95 ms |
| Phase 7 steady shock | 80 | sparse FD | 2-3 | ~24 ms |
| Phase 8 wave-pool stochastic | 128 | sparse + post-step | 2-3 | ~0.5 s |

**Sparse-AD speedup.** Phase 5b's introduction of
`SparseConnectivityTracer` + `SparseMatrixColorings` + `AutoForwardDiff`
through `NonlinearSolve` brought a ~3× speedup on the full `Pkg.test()`
suite (from ~5 min to ~1m30s pre-cross-phase-smoke). The dense-AD
N²-chunk-build was the bottleneck at N≥64; the sparsity-detected
tri-band Jacobian (each segment couples only to immediate neighbors)
brings the build cost to O(N).

## Production experiment runtimes (`experiments/`)

| Experiment | N | Steps | Wall time |
|---|---:|---:|---:|
| `B1_energy_drift.jl` | 8 | 10⁵ | ~50 s |
| `B4_compression_bursts.jl` | 128 | 950 (capped) | ~8 min |
| `A1_sod.jl` (mirror-doubled) | 400 (=200*2) | ~7000 | ~5 min |
| `A1_sod_with_q.jl` | 200 | ~3500 | ~2 min |
| `A2_cold_sinusoid.jl` (τ-scan) | 128, 6 τ vals | 800 each | ~2 min total |
| `A3_steady_shock.jl` (Mach scan) | 80, 5 M_1 vals | varies | ~1 min total |
| `B5_passive_tracer.jl` | 200 (mirror) | 487 | ~30 s |

## Memory

No tracked allocations beyond the Manifest precompile cache (~2 GB at
~616 deps, dominated by CairoMakie + OrdinaryDiffEq transitive trees).
Per-step allocations in the Newton path are minimized by the sparse
preallocator path; `det_step!` typical allocation is ~10 KB per step
at N=128 (mostly the per-segment ForwardDiff Dual buffers).

## Bottlenecks

1. **Phase 2 acoustic test** (14 s on its own at N=64, 1500 steps).
   Could be dropped to N=32 + 500 steps for the regression test
   without losing the dispersion-rate measurement, saving ~10 s on
   the full suite.

2. **Stochastic injection wall time per step** (~0.5 s at N=128).
   Bulk of this is the post-Newton burst-stats accumulation, not the
   noise sample itself. Could be amortized by sampling stats every
   K steps rather than every step.

3. **Sparse Jacobian build** (~80% of Newton wall time in larger
   problems). Prebuilding the sparsity pattern once per IC and
   caching the colorings would shave ~30%; not done yet, would matter
   for production runs at N≥256.

## Implications for Milestone 3 (2D)

- **Sparse AD is already the default** — Phase 5b's path generalizes
  to 2D where the sparsity pattern is the triangle-adjacency graph
  (per methods paper §9.5). Same `SparseConnectivityTracer` flow.
- **Newton iter count (2-3) is structurally low** because the
  variational EL system has a strong contraction in smooth regions.
  Expect similar in 2D where the per-cell DOF goes from 4 to ~21
  (methods paper §9.2). The Newton system grows linearly with cell
  count; sparsity scales with adjacency, not 1D-vs-2D.
- **GPU porting**: per-step kernel work (residual eval, Jacobian
  build) is embarrassingly parallel over segments / triangles.
  KernelAbstractions.jl recommended in `specs/05` §5.1; the dense
  per-cell kernels are 10s of FLOPs each.
- **Headline 2D wall-time projection**: at N_cells = 10⁴ in 2D with
  4-iter Newton + sparse AD + GPU, expect ~10-100 ms per step
  (10-100× faster than the 1D dense reference). Production wave-pool
  spectra in 2D should fit a single overnight run.

## Citation-friendly summary

> dfmm-2d's Milestone-1 1D Julia variational integrator processes the
> implicit-midpoint discrete Euler-Lagrange system in 2-3 Newton
> iterations per step (5-200 ms wall time at N ∈ [32, 200] segments).
> The full 1801-test regression suite — covering Phases 1-11 of the
> milestone plan, including B.2 cold-limit unification, B.5
> passive-scalar exactness, and Tier-A.1/A.2/A.3 deterministic
> regression — completes in ~2m25s on a single Apple M-series core.
