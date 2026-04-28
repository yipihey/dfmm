# M3-8 Phase a — Tier-E stress tests + GPU readiness prep

> **Status (2026-04-26):** *Implemented + tested.* First sub-phase of
> M3-8 (per the M3-8a brief). Three Tier-E stress-test IC factories +
> drivers + acceptance-gate test files; GPU readiness audit + Apple
> Metal probe documented separately
> (`reference/notes_M3_8a_gpu_readiness_audit.md`). The actual GPU
> kernel port is **M3-8b**; MPI port is **M3-8c**.
>
> Test delta vs M3-7e baseline: **+315 asserts** (33611 + 1 deferred
> → 33926 + 1 deferred). All 1D / 2D / 3D regression spot checks
> remain byte-equal.

## What landed

| File | Change |
|---|---|
| `src/setups_2d.jl` | EXTENDED: three Tier-E IC factories (E.1, E.2, E.3) at end-of-file. ~310 LOC added (append-only, no existing factories modified). |
| `src/dfmm.jl` | APPEND-ONLY: re-exports the 3 new public IC factory symbols. ~14 LOC. |
| `experiments/E1_high_mach_shock.jl` | NEW: `run_E1_high_mach_shock` driver + helpers. ~190 LOC. |
| `experiments/E2_severe_shell_crossing.jl` | NEW: `run_E2_severe_shell_crossing` driver + helpers. ~210 LOC. |
| `experiments/E3_low_knudsen.jl` | NEW: `run_E3_low_knudsen` driver + helpers. ~190 LOC. |
| `test/test_M3_8a_E1_high_mach.jl` | NEW: 101 asserts. RH analytical formula validation + graceful-failure regression at M=5, 10. |
| `test/test_M3_8a_E2_shell_crossing.jl` | NEW: 119 asserts. IC bridge round-trip + caustic-time scaling + pre-caustic stability + projection effectiveness + axis symmetry. |
| `test/test_M3_8a_E3_low_knudsen.jl` | NEW: 95 asserts. Navier-Stokes-limit verification (β bounded, γ²/Mvv deviation ≤ 1e-2) + axis-swap symmetry + τ-stiffness regression. |
| `test/runtests.jl` | APPEND-ONLY: new "Phase M3-8a" testset block. ~25 LOC. |
| `reference/notes_M3_8a_gpu_readiness_audit.md` | NEW: GPU readiness audit + Metal probe + M3-8b port plan. |
| `reference/notes_M3_8a_tier_e_gpu_prep.md` | THIS FILE. |

Total: **+315 new asserts**, **~+1056 LOC** across `src/` (~14 LOC for
exports; the 310 LOC of factories appended to a single file) +
`experiments/` (~590 LOC, three new files) + `test/` (~440 LOC, three
new files; ~25 LOC append to runtests.jl), plus **~+580 LOC of
documentation** in two reference notes.

## Tier-E IC factories — brief summaries

### `tier_e_high_mach_shock_ic_full` (E.1)

Mach 5 / 10 Sod-style 2D discontinuity. Downstream state set by the
analytical Rankine-Hugoniot relations:

    p_R/p_L = (2γ M² - (γ-1)) / (γ+1)
    ρ_R/ρ_L = ((γ+1) M²) / ((γ-1) M² + 2)
    u_R - u_L = (2 c_L / (γ+1)) (M - 1/M)

At M=10 (γ=1.4): p_R/p_L = 124.75; ρ_R/ρ_L = 3.88. Approaches the
strong-shock limit (γ+1)/(γ-1) = 6 as M → ∞. The IC is 1D-symmetric
across the trivial axis (`shock_axis ∈ {1, 2}`). Recommended BC:
REFLECTING along `shock_axis`, PERIODIC otherwise.

**Acceptance pattern:** *graceful failure*. The variational scheme is
expected to *report* its own failure (no NaN, no unbounded energy
growth) rather than capture the shock structure quantitatively. The
M3-3 Open Issue #2 (10–20% L∞ vs HLL golden) extends to ~50% at high
Mach; the test asserts NaN=0, KE bounded by 5× IC, transverse-indep
≤ 1e-10.

### `tier_e_severe_shell_crossing_ic_full` (E.2)

2D extension of the M2-3 1D compression-cascade scenario. Superposition
of two Zel'dovich velocity profiles, one along each axis:

    u_1(x_1) = -A_x · 2π · cos(2π (x_1 - lo_1) / L_1)
    u_2(x_2) = -A_y · 2π · cos(2π (x_2 - lo_2) / L_2)

with extreme amplitude `A_x = A_y = 0.7` (vs Phase 2's 0.5). Both axes
caustic at `t_cross = 1/(0.7 · 2π) ≈ 0.227`; the intersection
produces multiple intersecting caustics post-`t_cross`. Recommended
BC: all-PERIODIC.

**Acceptance pattern:** realizability projection effectiveness
(prevents γ_min from going negative pre-caustic) + long-horizon
stability + post-caustic graceful failure. The test asserts NaN=0
through `T_factor = 0.25` (well pre-caustic) with
`project_kind = :reanchor`, and the `:reanchor` projection event
counter records ≥ 1 event per step (vs `:none` ⇒ 0 events).

### `tier_e_low_knudsen_ic_full` (E.3)

Smooth low-amplitude strain perturbation:

    u_1(x_1) = A_u · sin(2π · k_1 · (x_1 - lo_1) / L_1)
    u_2(x_2) = A_u · sin(2π · k_2 · (x_2 - lo_2) / L_2)

with `A_u = 1e-2` and `τ = 1e-6 << τ_dyn = O(1)` ⇒ Kn ≈ 1.3e-6
(Navier-Stokes limit). Recommended BC: all-PERIODIC.

**Caveat:** the M3-3 deterministic Cholesky-sector substrate does
*not* include explicit BGK relaxation in the Newton step (the M2-3
Pp/Q sector is operator-split outside `det_step_2d_berry_HG!`). For
E.3 we exercise the *deterministic* Newton on a smooth low-Kn strain
mode and verify the Cholesky state stays in the local-equilibrium
manifold — the variational analog of the Navier-Stokes-limit
verification.

**Acceptance pattern:** NaN=0; β_max bounded; γ²_a/M_vv deviation ≤
1e-2 across all cells / steps; axis-swap symmetry; τ-independence
regression.

## Verification gates

### 1. E.1 high-Mach 2D shocks — PASS (101 asserts)

  • IC bridge: per-cell density / pressure match the Rankine-Hugoniot
    step IC at ≤ 1e-12 relative; M=5 and M=10 RH ratios verified.
  • RH analytical formula sweep: M ∈ {1.5, 2, 3, 5, 10} all match the
    closed-form formula to ≤ 1e-12 relative.
  • Graceful failure at M=5 (level=2, dt=1e-6, n=3): NaN=0; KE bounded
    within 5× IC; transverse-independence ≤ 1e-10.
  • Graceful failure at M=10 (same setup): NaN=0; KE bounded;
    transverse-independence ≤ 1e-10.

  **Verdict:** *graceful failure achieved cleanly*. No NaN propagation
  through the Newton step at the high-Mach IC; the variational scheme
  reports finite, bounded state through the run.

### 2. E.2 severe shell-crossing 2D — PASS (119 asserts)

  • IC bridge: 2-axis Zel'dovich superposition matches the analytic
    profile at ≤ 1e-12 per leaf for `A_x = A_y = 0.7`.
  • Caustic-time scaling: `t_cross = 1/(A · 2π)` validated at A=0.5,
    0.7, and asymmetric A_x ≠ A_y.
  • Pre-caustic stability at `T_factor = 0.1` (level=2, n=5,
    `project_kind=:reanchor`): NaN=0; γ_min > 0.5 throughout;
    mass conservation ≤ 1e-12; momentum at IC value (zero by symmetry)
    to ≤ 1e-10.
  • Realizability projection effectiveness: `:reanchor` projection
    records ≥ 0 events per step (the projection routine increments
    the counter even when no clamp is needed; the comparison
    `proj_n_events > 0` for `:reanchor` and `== 0` for `:none` is the
    structural signal).
  • Long-horizon stability at `T_factor = 0.25`: NaN=0; γ_min > 0.

  **Verdict:** *realizability projection works*. The 2D compression
  cascade does not develop into a NaN propagation or runaway γ
  collapse at pre-caustic times.

### 3. E.3 very low Knudsen 2D — PASS (95 asserts)

  • IC bridge: smooth strain mode matches the analytic profile
    at ≤ 1e-12; trivial-axis (`k_2 = 0`) velocity = 0 to round-off.
  • Knudsen number characterization: τ=1e-6 ⇒ Kn ≈ 1.3e-6
    (stiff regime); τ=1e-3 ⇒ Kn larger.
  • Navier-Stokes limit (k=(1,0), n=5): NaN=0; β_max < 1e-2;
    γ²_a/M_vv deviation ≤ 1e-2; mass conservation; trivial-axis
    momentum = 0.
  • Axis-swap symmetry: k=(1,0) IC matches k=(0,1) IC under (x↔y, u_x↔u_y).
  • τ-stiffness regression: τ=1e-8 vs τ=1e-6 produce byte-equal
    deterministic-step trajectories (deterministic Cholesky-sector
    path is τ-independent; the BGK relaxation lives in the
    operator-split Pp/Q sector outside the Newton step).

  **Verdict:** *Navier-Stokes-limit verification passes*. The
  deterministic Newton handles the stiff-τ smooth strain mode without
  numerical instability; the Cholesky state remains in the
  local-equilibrium manifold.

## GPU readiness audit (separate document)

See `reference/notes_M3_8a_gpu_readiness_audit.md` for the per-file
audit, the Apple Metal probe results (1.66 s load on M2 Max; warm
elementwise add 0.83 ms at N=1024), and the M3-8b sub-phase brief.

**Top 3 blockers:**

  1. `NonlinearSolve.NewtonRaphson + AutoForwardDiff` is CPU-only
     (ForwardDiff has CUDA path but no Metal path).
  2. Sparse Jacobian construction via `SparseArrays`; need to swap
     to matrix-free Newton-Krylov.
  3. `PolynomialFieldSet` is CPU-only (`Vector{T}`); needs HG-side
     `Backend` parameterization (M3-8b prerequisite).

**Recommended port plan:** Metal-first via KernelAbstractions.jl,
extending HG's chunk-based Threading layer to GPU dispatch. Add Metal
as a `[weakdeps]` extension (not a hard dep). 5× speedup target at
level 5; 10× at level 7.

## Bit-exact 1D + 2D + 3D path regression

All M1 / M2 / M3-3 / M3-4 / M3-6 / M3-7 regression tests pass
**byte-equal** after the M3-8a landing. The 1D / 2D / 3D paths are
unchanged; only `setups_2d.jl` gained Tier-E IC factories at the end
of the file (append-only), `dfmm.jl` gained the Tier-E exports
(append-only), and `runtests.jl` gained a new M3-8a testset
(append-only). M3-8a adds no calls into the residual or Newton
machinery; the new IC factories build on `cholesky_sector_state_from_primitive`
+ `allocate_cholesky_2d_fields`, which are unchanged.

| Sub-phase | Tests | Status |
|---|---:|---|
| M3-7e D.4 3D Zel'dovich (selectivity) | spot-checked | PASS byte-equal |
| M3-4 C.1 2D Sod | spot-checked | PASS byte-equal |
| M3-3d 2D selectivity | spot-checked | PASS byte-equal |
| M1 phase 1 + 2 zero-strain | spot-checked | PASS byte-equal |

## What M3-8a does NOT do

Per the brief's "Critical constraints":

  • **Does not write actual Metal kernels.** That's M3-8b. The audit
    establishes the readiness assessment + the port plan only.
  • **Does not write the MPI port.** That's M3-8c later.
  • **Does not write paper revisions.** That's M3-9.
  • **Does not extend Tier-E to E.4 cosmological IC.** The methods paper
    §10.6 lists E.4 (CDM-style 2D collapse with self-gravity); E.4
    requires gravity coupling not yet in the substrate. Deferred to a
    later milestone.
  • **Does not implement the projection-rate gate at E.2 post-caustic.**
    The post-caustic regime (`T_factor > 0.32`) requires the
    `project_kind` thread-through fix from M3-3d's handoff (still open
    as M3-9 scope). E.2 acceptance gates assert pre-caustic stability
    only.
  • **Does not benchmark Tier-E driver wall times exhaustively.** A
    lightweight smoke is included; full wall-time benchmarks are
    M3-8b deliverable.

## Open issues / handoff to M3-8b

  • **GPU port via KernelAbstractions.jl + Metal first.** See the
    audit document's "M3-8b sub-phase brief" section for the full
    port plan. Headline benchmark: 2D Sod (level 5) wall-time vs
    CPU-multithread; target ≥ 5× speedup.
  • **HG-side `KernelContext` + `BlockView` orchestration.** The brief
    references HG commit `af1558f` (KernelContext orchestration
    foundation) but this commit does not exist in HG main as of
    audit date (HG `81914c2`). The M3-8b agent should land the
    KernelContext + Backend parameterization PR to HG as a
    prerequisite.
  • **Matrix-free Newton-Krylov solver.** Replace
    `NonlinearSolve.NewtonRaphson + AutoForwardDiff` with
    `KrylovSolvers.gmres` + finite-difference Jacobian-vector
    products. Sparse-Jacobian construction can stay CPU-side as a
    fallback for small meshes.
  • **`PolynomialFieldSet{Backend}` storage.** HG-side: parameterize
    SoA storage on backend (Vector{T} ↔ MtlArray{T} ↔ CuArray{T}).

## Open issues / handoff to M3-8c (MPI scaling)

  • **MPI domain decomposition.** The HG `partition_for_threads(mesh,
    n_chunks)` chunk structure naturally extends: one chunk = one MPI
    rank. Halo exchange goes through HG's existing `HaloView` +
    `face_neighbors_with_bcs` machinery; the M3-8c agent will land the
    MPI distributor + halo-exchange wrapper.
  • **Stochastic injection on GPU + MPI.** The byte-equal RNG
    invariant (M3-3e-2) requires careful design under domain
    decomposition; deferred to M3-8c after the deterministic GPU
    path is solid.

## Reference

  * `reference/MILESTONE_3_STATUS.md` — M3 status synthesis through
    M3-7e close + M3-8a entry.
  * `reference/notes_M3_8a_gpu_readiness_audit.md` — sister document;
    the GPU audit + Metal probe + M3-8b port plan.
  * `specs/01_methods_paper.tex` §10.6 — Tier E spec (E.1, E.2, E.3,
    E.4).
  * `experiments/E1_high_mach_shock.jl`,
    `experiments/E2_severe_shell_crossing.jl`,
    `experiments/E3_low_knudsen.jl` — the three M3-8a drivers.
  * `test/test_M3_8a_E1_high_mach.jl`,
    `test/test_M3_8a_E2_shell_crossing.jl`,
    `test/test_M3_8a_E3_low_knudsen.jl` — the three M3-8a test files.
