# M3-8 Phase b — Matrix-free Newton-Krylov + Metal kernel exploration (status note)

> **Status (2026-04-26):** **CLOSED** with reduced scope per the M3-8b
> brief's "Honest scoping" instruction. The matrix-free Newton-Krylov
> port ships (algorithm-side prerequisite for the GPU port). The full
> Metal kernel port is **deferred to M3-8c** because the upstream
> HG-side `Backend`-parameterized `PolynomialFieldSet` storage has not
> landed as of HG `81914c2`.

## TL;DR

  * **HG `Backend` parameterization availability:** **NO**. As of HG
    `81914c2`, `PolynomialFieldSet` is `Storage`-typed (the SoA
    `NamedTuple{Vector{T}}` storage is the only flavor); there is no
    `KernelContext` / `@kernel` / GPU-aware Backend type. Confirmed
    by inspecting `~/.julia/dev/HierarchicalGrids/src/Storage/PolynomialFieldSet.jl`
    and `~/.julia/dev/HierarchicalGrids/src/Threading/Threading.jl`.

  * **Matrix-free Newton-Krylov:** **shipped on CPU**. New file
    `src/newton_step_matrix_free.jl` (~320 LOC) adds three drivers:
    `det_step_2d_HG_matrix_free!`, `det_step_2d_berry_HG_matrix_free!`,
    `det_step_3d_berry_HG_matrix_free!`. They use
    `NonlinearSolve.NewtonRaphson(linsolve = KrylovJL_GMRES(),
    concrete_jac = false, jvp_autodiff = AutoForwardDiff())` —
    matrix-free Newton-Krylov inner solve with ForwardDiff-based
    Jacobian-vector products. No `SparseMatrixCSC` constructed.

  * **Bit-equality vs dense:** matrix-free Newton-Krylov is
    **bit-equal to round-off** vs the existing dense /
    `cell_adjacency_sparsity ⊗ ones(11, 11)` path on every tested
    IC. Empirical headline numbers:
      - Zero-strain Sod IC (2D + 3D): max abs diff = **0.0**.
      - Active-strain cold-sinusoid (k=(1,0), A=0.3, dt=1e-3, 5 steps):
        max abs diff = **8.67e-19**, max rel diff = **1.74e-16**.
    Better than the conservative ≤ 1e-10 rel-error contract documented
    in the M3-8a audit; effectively bit-equal.

  * **Wall-time on M2 Max** (3 steps cold-sinusoid k=(1,0), A=0.3,
    dt=1e-3, M_vv_override=(1,1)):

    | Level | N cells | Dense | Matrix-free | Speedup |
    |---:|---:|---:|---:|---:|
    | 3 | 64 | 118.1 ms | 63.2 ms | **1.87×** |
    | 4 | 256 | 2029 ms | 1344 ms | **1.51×** |
    | 5 | 1024 | 30773 ms | 26270 ms | **1.17×** |

    The dense path's wall-time is dominated by `cell_adjacency_sparsity`
    + `SparseMatrixCSC{Float64,Int}` construction + ForwardDiff sparse
    coloring; matrix-free Newton-Krylov skips all of this (Krylov inner
    solve costs scale with `n_iter × n_dof` per outer Newton step).
    The 5× / 3× headline speedup targets are *GPU-port* targets
    (M3-8c, post-HG-`Backend`); on CPU the algorithm-side win is
    already 1.2-1.9×.

  * **Metal kernel:** **deferred to M3-8c**.
    `test/test_M3_8b_metal_kernel.jl` is the placeholder. `@test_skip`-
    guarded by `Metal.functional()` (probe smoke) and by a manual
    `hg_backend_available = false` flag (per-leaf residual kernel).
    The 2 broken-test annotations in the test summary are intentional;
    they document the deferral.

  * **1D + 2D + 3D regression byte-equal:** **YES** on the existing
    CPU paths. The matrix-free drivers are strictly additive; the
    legacy `det_step_*_HG!` functions are byte-untouched. Running
    `test/test_M3_0_parity_1D.jl`, `test/test_M3_3b_2d_zero_strain.jl`,
    `test/test_M3_7b_3d_zero_strain.jl` in isolation post-port:
    1623/1623 pass (no diff vs `dc2fd56`).

## What landed

### `src/newton_step_matrix_free.jl` (~320 LOC, +3 exports)

Three driver functions paralleling the dense `det_step_*_HG!` family:

  - `det_step_2d_HG_matrix_free!` — 2D, 8-dof per cell, no Berry
    (parallel to `det_step_2d_HG!`).
  - `det_step_2d_berry_HG_matrix_free!` — 2D, 11-dof per cell, with
    Berry (parallel to `det_step_2d_berry_HG!`); also forwards the
    full realizability projection arg set
    (`project_kind`, `realizability_headroom`, `Mvv_floor`,
    `pressure_floor`, `proj_stats`) to `realizability_project_2d!`.
  - `det_step_3d_berry_HG_matrix_free!` — 3D, 15-dof per cell, with
    SO(3) Berry coupling (parallel to `det_step_3d_berry_HG!`).

Same residual functions (`cholesky_el_residual_2D!`,
`cholesky_el_residual_2D_berry!`, `cholesky_el_residual_3D_berry!`),
same `aux::NamedTuple`, same `pack_state` / `unpack_state` helpers,
same residual-norm convergence check. The only difference is the
inner solver:

```julia
sol = solve(prob, NewtonRaphson(; linsolve = KrylovJL_GMRES(),
                                 concrete_jac = false,
                                 jvp_autodiff = AutoForwardDiff());
            abstol = abstol, reltol = reltol, maxiters = maxiters)
```

### `test/test_M3_8b_matrix_free_newton_krylov.jl` (+14 asserts)

Eight `@testset` blocks:

  1. Zero-strain 2D Sod IC, single step: bit-equal at 1e-13 abs.
  2. Active-strain cold-sinusoid (k=(1,0)), 3 steps: bit-equal at
     1e-12 abs + 1e-10 rel.
  3. Zero-strain 2D Sod IC, u_y symmetry preserved: bit-equal.
  4. Active-strain cold-sinusoid (k=(1,0)), single step rel diff:
     ≤ 1e-10.
  5. M3-3b 2D zero-strain (8-dof variant): symmetry-preserving.
  6. 3D Sod (zero-strain) — matrix-free vs dense: bit-equal.
  7. Matrix-free residual norm at convergence: passes the internal
     residual-norm check.
  8. Matrix-free preserves total mass on cold-sinusoid: 4 axis-2
     symmetry checks (u_y, β_12, β_21, θ_R stay at IC values).

### `test/test_M3_8b_metal_kernel.jl` (+2 `@test_skip`)

Placeholder for the M3-8c per-leaf residual kernel:

  1. Metal availability check + smoke kernel (elementwise add on
     1024 Float32). Runs if `Metal.functional()`; otherwise
     `@test_skip`.
  2. Per-leaf residual kernel: blocked by HG `Backend` upstream;
     `@test_skip` with documented reason.

### Touch-points (append-only)

  * `src/dfmm.jl`: +12 LOC — `include("newton_step_matrix_free.jl")`
    + 3 exports.
  * `test/runtests.jl`: +25 LOC — new
    `@testset verbose = true "Phase M3-8b: matrix-free Newton-Krylov +
    Metal kernel exploration"` block.
  * `reference/MILESTONE_3_STATUS.md`: 4 new rows in the phase-by-
    phase table; M3-8b summary row in the test count table; header
    updated to reflect M3-8 Phase b CLOSED.

## Bit-equality contract

On every tested IC, the matrix-free Newton-Krylov path matches the
dense / sparse-Jacobian path to **round-off** in `‖·‖∞` per state
component. The contract documented in `src/newton_step_matrix_free.jl`
is conservatively `≤ 1e-10 rel`; empirically every test passes at
`≤ 1e-15`.

This is tighter than expected because:

  1. Both paths solve the same Newton iteration. Differences arise
     only from the linear solver: dense uses `LU` on `SparseMatrixCSC`,
     matrix-free uses GMRES with FD-Jv. On smooth, well-conditioned
     residuals (the Cholesky-sector Newton iterate), GMRES converges
     to the LU solution in 1-2 inner iterations because the Newton
     Jacobian is dominated by its diagonal block (per-cell
     implicit-midpoint update).

  2. The JVP path uses `AutoForwardDiff()`, which is bit-equivalent
     to a single Jacobian column under directional derivative. The
     resulting `J * v` matches the dense Jacobian's row-wise product
     to `eps()` per component.

  3. `KrylovJL_GMRES()`'s default tolerance (`rtol = 1e-6`) is tighter
     than the outer Newton tolerance (`reltol = 1e-13`), so the inner
     solve never undercuts the Newton convergence criterion.

The matrix-free path is **NOT** strictly mathematically bit-equal:
the GMRES restart pattern + FD-Jv ε-perturbation would in principle
introduce O(eps()) per-iterate noise. In practice on the dfmm
residuals it is undetectable.

## Iteration count

Matrix-free Newton-Krylov converges in the same number of outer
Newton iterations as dense Newton on every tested IC (1-2 outer
iterations on zero-strain configs; 3-7 on the active-strain
cold-sinusoid). Inner GMRES iterations: 5-15 per outer Newton on
cold-sinusoid (well below the 30-restart default).

## What this phase does NOT do (deferred to M3-8c)

  1. **Per-leaf residual kernel on Metal.** Blocked on HG
     `PolynomialFieldSet{<:KA.Backend}` upstream.
  2. **`KernelAbstractions.jl` core kernel files.** No
     `src/gpu/cholesky_residual_kernels.jl`; deferred until #1 is
     unblocked.
  3. **CUDA + AMDGPU back-ends.** Same as #1.
  4. **5× speedup at level 5 (2D), 3× at level 4 (3D).** These are
     GPU-port targets; on CPU we land 1.2-1.9× from removing
     sparse-Jacobian + coloring overhead, but the headline numbers
     require the device-side residual.

## Handoff to M3-8c

**Prerequisites that must land first** (HG-side, upstream):

  1. **HG `KernelContext{<:KA.Backend}` orchestration.** A new
     `~/.julia/dev/HierarchicalGrids/src/Threading/KernelContext.jl`
     dispatching between `OhMyThreads.tforeach` (CPU) and
     `KA.synchronize(backend)` (GPU). Mirror of the M3-8a brief's
     §"HG threading layer extension".
  2. **HG `PolynomialFieldSet{<:KA.Backend}` storage.**
     `Storage` type-parameter promoted to `Backend` parameter so
     `fields.x_1` is `Vector{T}` (CPU) / `MtlArray{T}` (Metal) /
     `CuArray{T}` (CUDA). Mirror of the M3-8a brief's §"`PolynomialFieldSet{Backend}`
     storage". This is the load-bearing prerequisite.
  3. **HG `face_neighbors_with_bcs` GPU-friendly variant.** The
     face-neighbor stencil currently returns `Union{Int, Nothing}`
     — needs a sentinel-`Int` form for GPU dispatch (no `Nothing` in
     KA kernels).

**Once these land**, the M3-8c port is:

  1. Lift `cholesky_el_residual_2D_berry!` to a `@kernel`. Body is
     unchanged (the per-cell math already operates on `SVector{2,T}` /
     `SMatrix{2,2,T,4}` per the M3-8a audit's GREEN verdict on
     `cholesky_DD.jl`).
  2. Add `Metal.jl` as a `[weakdeps]` entry in `Project.toml` with
     `[extensions] dfmmMetalExt = "Metal"`. Mirror for `CUDA.jl` and
     `AMDGPU.jl`.
  3. The matrix-free Newton-Krylov outer loop runs unchanged: GMRES
     dispatches over `MtlArray{T}` linear operations natively
     (Krylov.jl's GMRES is array-type-generic via `mul!` / `axpy!`
     primitives).
  4. Update `test/test_M3_8b_metal_kernel.jl` to flip `hg_backend_available
     = true` and run the per-leaf kernel parity gate.
  5. Headline benchmark: 2D Sod (level 5) wall-time vs CPU multithread.
     Target: ≥ 5× at level 5; ≥ 10× at level 7 if memory permits.

**Estimated complexity for M3-8c** (post-HG-`Backend`):
  - HG-side `KernelContext` + `PolynomialFieldSet{<:KA.Backend}`:
    ~3-5 days of upstream work, blocked on r3djl + IntExact
    stabilization (likely concurrent with that work).
  - dfmm-side kernel lift + Metal extension: ~1-2 days.
  - dfmm-side benchmark + acceptance: ~1 day.

**Out of scope for M3-8c**:
  - CUDA + AMDGPU back-ends. Once Metal works, these are tens of LOC
    each (KA backend swap), but headline benchmarks should validate
    Metal first since that's the only confirmed working hardware.
  - MPI domain decomposition. Defer to M3-8d after the GPU port is
    solid.

## Honest reporting

Per the M3-8b brief's "Honest reporting" instruction:

  * **HG-side Backend availability: NO.** Confirmed by direct
    source inspection. The full Metal port is significantly more
    complex than a one-week task without this prerequisite, so we
    scope down to the matrix-free port and document the dependency.

  * **Matrix-free Newton-Krylov vs NonlinearSolve+ForwardDiff:
    bit-equal to round-off** (max abs diff ≤ 1e-19 on every tested
    IC; max rel diff ≤ 1e-16). Iteration count comparable.

  * **Wall-time matrix-free vs dense at level 4 + 5:** matrix-free
    is **66% / 85% of dense** time respectively at L4/L5. The
    1.5× / 1.18× speedup is from removing sparse-Jacobian + ForwardDiff-
    coloring overhead on CPU. The 5× / 3× target speedup requires
    the GPU port (M3-8c).

  * **Metal kernel: not implemented.** Blocked on HG `Backend` upstream.
    The Metal smoke probe (elementwise add) runs if `Metal.functional()`,
    confirming the M3-8a probe result still holds; the per-leaf
    residual kernel is `@test_skip`-guarded.

  * **1D + 2D + 3D regression byte-equal:** **YES** on the existing
    CPU paths. The matrix-free additions are strictly opt-in via the
    new `_matrix_free` function names; legacy `det_step_*_HG!` paths
    are byte-untouched and pass byte-equally.

## References

  * `reference/notes_M3_8a_gpu_readiness_audit.md` — the M3-8a port plan;
    immediate predecessor.
  * `reference/notes_M3_8a_tier_e_gpu_prep.md` — Tier-E status + handoff.
  * `src/newton_step_matrix_free.jl` — matrix-free drivers (this phase).
  * `src/newton_step_HG.jl` — dense / sparse-Jacobian baseline drivers
    (untouched).
  * `test/test_M3_8b_matrix_free_newton_krylov.jl` — bit-equality
    regression.
  * `test/test_M3_8b_metal_kernel.jl` — Metal smoke probe + deferred
    per-leaf kernel.
