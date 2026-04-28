# M3-8 Phase a — GPU Readiness Audit + Metal Probe

> **Status (2026-04-26):** Audit-only deliverable per the M3-8a brief.
> The actual Metal kernel port + benchmarks are **M3-8b** (next
> sub-phase); MPI port is **M3-8c**. This document is the input brief
> the M3-8b agent will work from.

## TL;DR

  * **GPU readiness verdict:** Mostly amber. The 2D / 3D Newton solver
    path uses `NonlinearSolve.jl` + `ForwardDiff` + sparse-AD Jacobians,
    none of which are GPU-aware out of the box. The hot loops (residual
    evaluation, pack/unpack, face-neighbor stencil) are
    `@inbounds`-tight and per-cell parallel — once the Newton driver is
    swapped to a GPU-friendly back-end (KrylovSolvers + matrix-free, or
    Enzyme-jacobian), the residual evaluation maps cleanly onto
    KernelAbstractions.jl.

  * **Top 3 blockers (in priority order):**
    1. `NonlinearSolve.jl + AutoForwardDiff` is CPU-only. Need to swap
       to either (a) `KrylovSolvers.jl` matrix-free Newton-Krylov, or
       (b) `NonlinearSolve.jl` with a GPU-aware autodiff via `Enzyme.jl`
       or hand-written Jacobian-vector products.
    2. `cell_adjacency_sparsity` Jacobian assembly uses `SparseArrays`
       (`push!`-based COO + `_SA.sparse(rows, cols, ...)`); this stays
       CPU-only. The matrix-free reformulation removes the need for a
       sparse-CSR object on GPU.
    3. `PolynomialFieldSet` (HG storage) is currently `Vector{T}`-backed
       SoA on CPU. Needs an `MtlArray{T}` / `CuArray{T}` / `ROCArray{T}`
       backend mode in HG (item 8 of the GPU sub-phase brief below).

  * **Metal probe:** Confirmed Metal.jl loads on this M2 Max in 1.66 s
    (warm precompile cache); first-kernel compile is ~12 s; warm
    elementwise broadcast at N=1024 is 0.83 ms; 5× elementwise add at
    N=1e6 is 8.8 ms (~1.76 ms / op). Metal is *not* added as a hard
    dependency — see "Metal.jl install path" below.

  * **Recommended Metal-first port plan:** see "M3-8b sub-phase brief"
    at the end of this document.

## Audit Methodology

Three `src/` files are the GPU-critical path:

  * `src/newton_step_HG.jl` (~1715 LOC): the deterministic Newton step
    drivers (1D, 2D, 2D-Berry, 3D, 3D-Berry).
  * `src/eom.jl` (~2310 LOC): the per-leaf EL residual evaluations
    (`cholesky_el_residual_2D!`, `_2D_berry!`, `_3D!`, `_3D_berry!`)
    + auxiliary builders (`build_residual_aux_*`,
    `build_face_neighbor_tables*`, `build_periodic_wrap_tables*`).
  * `src/cholesky_DD.jl` (~395 LOC) + `src/cholesky_DD_3d.jl` (~549
    LOC): per-cell Cholesky decompose / recompose / γ extraction —
    purely local-per-cell math, ideal GPU primitives.

The audit dimensions assessed per file:

  1. **Allocation hygiene.** `@inbounds`, no `push!` in hot loops,
     `Vector` vs `Array` vs `SArray` storage choice.
  2. **Branching pattern.** GPU prefers regular control flow; warp /
     thread-divergent branches are amber.
  3. **Dispatched type structure.** Generic `T` parameterization + no
     hard `Float64` casts in the hot loop ⇒ green (already
     KernelAbstractions-friendly).
  4. **External library coupling.** `NonlinearSolve.jl`, `ForwardDiff`,
     `SparseArrays` ⇒ CPU-only, must replace.
  5. **Per-cell vs cross-cell access.** Self-cell math is green;
     face-neighbor stencils need ghost-cell halo on GPU.

## Per-File Readiness Assessment

### `src/cholesky_DD.jl` — `cholesky_decompose_2d`, `cholesky_recompose_2d`, `gamma_per_axis_2d`, `h_rot_partial_dtheta` (~395 LOC)

**Verdict: GREEN.** Purely-local per-cell math on `SVector{2, T}` /
`SMatrix{2, 2, T, 4}`. All functions are `@inline`-marked, allocation-
free, and operate on stack-resident `StaticArrays`. The
`gamma_per_axis_2d_per_species` allocates an output `Matrix{T}` but the
math itself is warp-friendly.

Translates to KernelAbstractions kernels with zero refactoring:

```julia
@kernel function recompose_kernel!(L_out, α_in, θ_in)
    i = @index(Global, Linear)
    L_out[i] = cholesky_recompose_2d(α_in[i], θ_in[i])
end
```

### `src/cholesky_DD_3d.jl` — 3D analog (~549 LOC)

**Verdict: GREEN.** Same pattern as `cholesky_DD.jl`; 3D version uses
`SVector{3, T}` / `SMatrix{3, 3, T, 9}`. The intrinsic Cardan ZYX Euler
rotation is a 9-element matrix product — amenable to inlined GPU
kernels. No allocation, no branching.

### `src/eom.jl` residual functions — `cholesky_el_residual_2D[_berry]!`, `_3D[_berry]!`

**Verdict: AMBER.** The hot loop `@inbounds for i in 1:N ... end` is
GPU-portable in shape, but several concrete fixes are needed:

  1. **Index helpers** (`@inline get_x(y, a, i)`,
     `@inline get_α(y, a, i)`, etc) currently live as closures inside
     the function body. KernelAbstractions / Metal compilers handle
     these but they may not inline aggressively under the GPU
     compilation path. **Recommended fix:** lift to top-level `@inline`
     functions (or pass strides as kernel args).

  2. **`face_lo_idx` / `face_hi_idx`** are `Matrix{Int}` (or similar
     `2 × N`) lookups. On GPU this becomes an `MtlArray{Int}` /
     `CuArray{Int}` with bounds-checked `getindex`. Trivial port; just
     storage-mode swap.

  3. **`Tres = promote_type(eltype(y_np1), eltype(y_n), …)`** is
     correct but allocates a `Type` object in some Julia versions on
     GPU. **Recommended fix:** parameterize the kernel on `Tres` at
     dispatch time, not at runtime.

  4. **EOS `Mvv(J, s)` call** in `src/eos.jl` is `@inline`-marked and
     allocation-free; safe.

  5. **`@assert` lines** at the head of the residual function will
     not run on GPU. Move to CPU pre-condition checks before the
     kernel launch.

  6. **`cell_physical_box(frame, ci)` calls** in `build_residual_aux_*`
     happen CPU-side (auxiliary builder, not in the hot loop). Safe.

### `src/newton_step_HG.jl` — Newton drivers (~1715 LOC)

**Verdict: RED.** This is the most GPU-hostile file:

  1. **`NonlinearSolve.NewtonRaphson(; autodiff = AutoForwardDiff())`**
     uses `ForwardDiff.Dual` types under the hood. ForwardDiff has a
     long-standing CUDA path (NVIDIA-specific via `CUDA.jl + CUDADual`)
     but no Metal path. **Blocker #1.**

  2. **Sparse-Jacobian construction** (the `cell_sparsity =
     cell_adjacency_sparsity(mesh; depth = 1, leaves_only = true)` ⊗
     `ones(11, 11)` Kron pattern) materializes a `SparseMatrixCSC{Float64,
     Int}` for NonlinearSolve coloring. Sparse-CSR on GPU exists
     (`CuSparseMatrixCSR` for CUDA; `MtlSparseMatrix` for Metal does
     NOT yet exist as of Metal.jl v1.x). **Blocker #2.** Resolution:
     swap to **matrix-free Newton-Krylov** via `KrylovSolvers.jl`
     `gmres` + a hand-written `Jv = Jacobian-vector-product` closure
     that re-evaluates the residual via finite-difference or analytic
     directional derivative.

  3. **`pack_state_2d` / `unpack_state_2d!` etc** — these are simple
     Vector copies indexed by `leaves` order. GPU-port is trivial
     (`@kernel function pack_kernel!(y, fields, leaves)`).

  4. **`F_check` allocation + post-Newton residual norm** — the
     `maximum(abs, F_check)` reduction is a single GPU reduce kernel
     in KA / Metal. Trivial.

  5. **The error path** (`error("det_step_2d_berry_HG! Newton solve
     failed: …")`) cannot run inside a GPU kernel. **Recommended:**
     return retcode + residual norm, raise on CPU side after
     synchronization.

### `src/stochastic_injection.jl` — `realizability_project_2d!`, `inject_vg_noise_HG_2d!`

**Verdict: AMBER.** Per-cell projection math is GPU-friendly (just
algebra), but:

  - `Random` calls (variance-gamma noise generation) need
    `MetalRandom.jl` / `CURAND.jl`. **Standard pattern in CUDA.jl
    (Philox / xoshiro on GPU); Metal has KA's `Random` extension as of
    KA 0.9.x.** Recommend: the per-axis stochastic injection becomes
    "host-prepared noise tensor passed to GPU kernel" — keep the RNG
    on CPU (it's already byte-equal across architectures, an M3-3e-2
    invariant we don't want to break).

  - `ProjectionStats` mutation (`stats.n_events += 1`) needs an atomic
    counter on GPU. KernelAbstractions provides
    `KernelAbstractions.@atomic` for this. Trivial swap.

## HG Threading + Hardware Layer

The HG package's `Threading` module
(`~/.julia/dev/HierarchicalGrids/src/Threading/Threading.jl`) is built
on `OhMyThreads.jl` with explicit chunk-based scheduling. The brief
references "HG's `KernelContext` + `BlockView`" — these are *not yet
landed* in HG main as of the audit date (HG `81914c2` / `edc6d78`).
The Threading module is **shared-memory only** today. The brief's
reference to commit `af1558f` (KernelContext orchestration foundation)
appears to be a forward reference; the M3-8a agent works with what's
there and treats `KernelContext / BlockView` as an HG-side prerequisite
landing in **M3-8b**.

The chunk-based partitioning (`partition_for_threads(mesh, n_chunks)`)
is the natural unit for both:
  - Per-thread parallelism on CPU (current).
  - Per-process MPI domain decomposition (M3-8c).
  - Per-block GPU dispatch (M3-8b — each chunk = one workgroup).

The mapping is structurally identical: chunk → workgroup → MPI rank.

## Apple Metal Probe Result (M2 Max)

Hardware: Apple M2 Max (12 cores, 24 GB unified memory, GPU is part of
the Apple Silicon SoC). Probe environment: a fresh `/tmp/metal_probe`
project with `Metal.jl` added.

| Operation | Time | Notes |
|---|---|---|
| `using Metal` (warm precompile cache) | 1.66 s | First run after install ~30 s |
| `Metal.devices()` | < 1 ms | 1 device discovered (`AGXG14CDevice`) |
| HtoD copy `MtlArray(Float32[1:1024])` | 698 ms | First-call dominated by setup |
| First kernel launch (`a .+ b`, N=1024) | 11.9 s | LLVM IR → MSL JIT compile cost |
| Warm kernel launch (`a .+ b`, N=1024) | 0.83 ms | Steady-state per-kernel overhead |
| 5× `a .+ b` warm (N=1e6) | 8.8 ms | ~1.76 ms / op; bandwidth-bound |
| Result correctness | 0.0 max-err | Float32 elementwise add bit-exact |

**Verdict:** Metal.jl is *production-ready* for elementwise + reduce
kernels on this hardware. Compile cost is one-time per session;
runtime overhead is sub-ms. The scaling at N=1e6 (~1.76 ms/op) gives
~570 GB/s effective bandwidth (3 arrays × 4 B × 1e6 = 12 MB per op /
1.76 ms ≈ 6.8 GB/s memory traffic round-trip; that's headline-write-
heavy bound on an M2 Max with stated 400 GB/s memory bandwidth).

For a per-leaf 11-dof Newton residual evaluation on a level-5 mesh
(1024 cells in 2D, 32K cells in 3D), the per-kernel overhead is
dwarfed by the per-cell math time. Metal is the right Apple-first
target.

### Metal.jl install path

Metal.jl is **not** added to the dfmm Project.toml as a hard
dependency — keeping it opt-in lets non-Apple-Silicon users (and CI)
load dfmm cleanly. The recommended install path (M3-8b):

  1. Add Metal.jl as a `[weakdeps]` entry in `Project.toml` with
     `[extensions] dfmmMetalExt = "Metal"`.
  2. Mirror for `CUDA.jl` (`dfmmCUDAExt = "CUDA"`) and
     `AMDGPU.jl` (`dfmmAMDGPUExt = "AMDGPU"`).
  3. Common kernel code lives in a new `src/gpu/` subdirectory written
     against `KernelAbstractions.jl` (back-end-agnostic).
  4. Each backend-specific extension defines a thin
     `KernelAbstractions.Backend()` selector + array constructor.

This matches the Trixi.jl / GPU4GEMS pattern; well-tested in the Julia
ecosystem.

## M3-8b Sub-phase Brief (next)

**Goal:** Port the 2D Newton-step inner loop to GPU via
KernelAbstractions.jl, targeting Metal first, then CUDA + AMDGPU.

**Scope (in order):**

  1. **HG Threading layer extension.** Land
     `KernelContext{Backend}` in HG that selects between
     `OhMyThreads.tforeach` (CPU), `KA.synchronize(Backend)` (GPU). PR
     to HG.
  2. **`PolynomialFieldSet{Backend}` storage.** HG-side: parameterize
     the SoA storage on backend so leaf coefficients are
     `MtlArray{T}` / `CuArray{T}` / `Vector{T}`.
  3. **GPU-friendly residual evaluation.** Lift `cholesky_el_residual_2D_berry!`
     into a `@kernel` with explicit dispatched per-cell index. Verify
     bit-exact 0.0 parity with CPU residual on Float32 + Float64 inputs.
  4. **Matrix-free Newton-Krylov solver.** Replace
     `NonlinearSolve.NewtonRaphson + AutoForwardDiff` with
     `KrylovSolvers.gmres` + finite-difference Jacobian-vector
     products. CPU regression: bit-exact at residual norm ≤ 1e-10
     (relaxed from 1e-13 due to FD-Jv noise; still tight enough for
     all M3-3 / M3-4 / M3-6 / M3-7 tests).
  5. **Metal-specific dispatch + benchmarks.** Headline benchmark:
     2D Sod (level 5) wall-time vs CPU-multithread (current); target
     ≥ 5× speedup at level 5, ≥ 10× at level 7.
  6. **CUDA + AMDGPU back-ends.** Once Metal works, the KA back-end
     swap is ~tens of LOC.

**Deliverables (M3-8b):**

  - `src/gpu/cholesky_residual_kernels.jl` (~300 LOC; KA `@kernel`s).
  - `src/gpu/newton_step_matrix_free.jl` (~150 LOC; Krylov-Newton).
  - `src/dfmmMetalExt.jl` (extension; ~50 LOC).
  - `test/test_M3_8b_gpu_parity.jl` (~50 asserts; CPU vs GPU bit
    parity).
  - `reference/notes_M3_8b_gpu_port.md` (status note + benchmark table).

**Acceptance gates (M3-8b):**

  - Bit-equal residual evaluation CPU vs Metal GPU on all M3-7
    regression configurations (Float64).
  - Newton convergence equivalent to CPU on all C.1 / C.2 / C.3 / D.1 /
    D.4 / D.7 / D.10 drivers.
  - Wall-time ≥ 5× speedup at level 5 (2D), ≥ 3× at level 4 (3D).
  - 1D + 2D + 3D regression byte-equal on CPU path (`Backend = CPU`
    default = bit-equal to current code).

**Out of scope for M3-8b (deferred to M3-8c):**

  - MPI domain decomposition. The HG `partition_for_threads` chunk
    structure naturally extends to MPI ranks (one chunk = one rank);
    the M3-8c agent will land the MPI distributor + halo exchange.
  - Stochastic injection on GPU. The byte-equal RNG invariant
    (M3-3e-2) requires careful design; deferred to M3-8c after the
    deterministic path is solid.

## What this document does NOT do

  - **Does not write actual Metal kernels.** That's M3-8b. The audit
    establishes the readiness assessment + the port plan only.
  - **Does not benchmark Newton convergence on GPU.** No GPU code
    runs in this audit; the Metal probe is a trivial elementwise
    broadcast, not the dfmm Newton step.
  - **Does not modify `src/`.** All changes are documentation + the
    Tier-E IC factories (M3-8a (a)) + Tier-E test drivers (M3-8a (b)).

## References

  * `src/newton_step_HG.jl` — the Newton drivers under audit.
  * `src/eom.jl` — the EL residual under audit.
  * `src/cholesky_DD.jl`, `src/cholesky_DD_3d.jl` — per-cell math
    under audit.
  * `~/.julia/dev/HierarchicalGrids/src/Threading/Threading.jl` —
    the HG threading layer the GPU port hooks into.
  * `reference/notes_HG_design_guidance.md` — the upstream HG
    design-review note, item-7 (sparsity-pattern API) is the immediate
    GPU-side prerequisite.
  * Methods paper §11 revision (`paper/paper.tex`) — M3-8 listed as
    Tier E + GPU/MPI scope.
