# M3-7 prep — 3D field set + per-axis Cholesky decomposition (scaffolding)

> **Status (2026-04-26):** *Implemented + tested*. Scaffolding pre-work
> for the M3-7 (3D extension) milestone (`reference/notes_M3_7_3d_extension.md`).
> Branch `m3-7-prep-3d-scaffolding`, single commit.
>
> Test delta vs M3-6 Phase 2 baseline: **+736 asserts** (90 raw `@test`
> macros expanded by loops). All M3-7a-e scientific phases (3D EL
> residual, Newton driver, Berry coupling integration, per-axis γ AMR,
> Tier-C/D drivers) are intentionally **not** in scope here — they are
> M3-7 proper.
>
> This sub-phase is **parallel-safe with M3-6 Phase 3** (which is
> extending `src/cholesky_DD.jl` in a sibling worktree): all new 3D
> primitives live in a new file `src/cholesky_DD_3d.jl`, and the only
> shared file (`src/types.jl`) is append-only.

## What landed

| File | Change |
|---|---|
| `src/cholesky_DD_3d.jl` | NEW (~370 LOC). Per-axis 3D Cholesky decomposition driver. Three primitives: `cholesky_decompose_3d`, `cholesky_recompose_3d`, `gamma_per_axis_3d` (matrix + diagonal forms), plus the `rotation_matrix_3d` helper. All allocation-free with `StaticArrays` inputs. SymPy-script-consistent SO(3) Euler-angle convention pinned in the top-of-file docstring (intrinsic Cardan ZYX). |
| `src/types.jl` | EXTENDED: appended `DetField3D{T}` working struct (~120 LOC of struct + docstrings + constructors). Carries the 13 Newton unknowns `(x_a, u_a, α_a, β_a)_{a=1,2,3} + (θ_12, θ_13, θ_23) + s` per leaf cell. Off-diagonal β + post-Newton (Pp/Q) sectors deferred per M3-3a Q3 default + M3-7 design note §4.4. The legacy 1D `DetField{T}` and 2D `DetField2D{T}` are untouched. |
| `Project.toml` | Added `LinearAlgebra` as a direct dependency (it was a transitive dep already; the new `cholesky_DD_3d.jl` uses `eigen`, `Symmetric`, `det` directly). |
| `src/dfmm.jl` | APPEND-ONLY: include the new file + export the new symbols. The 1D / 2D paths are unchanged. |
| `test/test_M3_7_prep_3d_scaffolding.jl` | NEW. 90 raw `@test` macros, expanded by loops to 736 asserts. Round-trip on 50 random samples in canonical hemisphere; iso-pullback gauge degeneracy + M-preservation; 2D reduction byte-equal on top-left block; per-axis γ on anisotropic / iso M_vv; allocation-free hot-path tests. |
| `test/runtests.jl` | APPEND-ONLY: new "Phase M3-7 prep" testset after M3-6 Phase 2. |
| `reference/notes_M3_7_prep_3d_scaffolding.md` | THIS FILE. |

## SO(3) Euler-angle convention (pinned)

The three Euler angles `(θ_12, θ_13, θ_23)` parameterize the principal-
axis rotation as the **intrinsic Cardan ZYX composition** of three
elementary Givens rotations in the (a, b)-coordinate planes:

    R(θ_12, θ_13, θ_23) = R_12(θ_12) · R_13(θ_13) · R_23(θ_23)

where

    R_12(θ) = [[ c, -s, 0],   R_13(θ) = [[ c, 0, -s],   R_23(θ) = [[1,  0,  0],
               [ s,  c, 0],              [ 0, 1,  0],              [0,  c, -s],
               [ 0,  0, 1]]              [ s, 0,  c]]              [0,  s,  c]]

This convention is consistent with the SymPy authority script
`scripts/verify_berry_connection_3D.py`, which treats `(θ_12, θ_13,
θ_23)` as three independent pair-rotation generator angles in the 9D
phase space (the script uses only infinitesimal forms via `sp.diff`,
so it does not constrain the finite-rotation composition order; we
adopt ZYX Cardan for the explicit recomposition). Properties verified
in `test_M3_7_prep_3d_scaffolding.jl`:

  * **2D reduction byte-equal** (300 asserts): at `θ_13 = θ_23 = 0`,
    `R = R_12(θ_12)`, and the top-left 2×2 block of
    `cholesky_recompose_3d` equals `cholesky_recompose_2d` byte-equally
    — the 3D 2D-symmetric ⊂ 2D dimension-lift slice
    (M3-7 design note §7.1b).
  * **R Rᵀ = I and det(R) = +1** at random angle samples (proper
    rotation in SO(3)).
  * **Round-trip** `decompose ∘ recompose ≡ id` to ≤ 5e-15 absolute
    on 50 random samples in the canonical hemisphere
    `θ_12, θ_13, θ_23 ∈ (-π/2, π/2)`.

The canonical hemisphere range is the 3D analog of the 2D
`θ_R ∈ (-π/2, π/2]` round-trip range (the `cos(θ_R) ≥ 0` condition,
generalized to `cos(θ_12) cos(θ_13) ≥ 0` and `cos(θ_13) cos(θ_23) ≥ 0`
for 3D — i.e., `R[1, 1] ≥ 0` and `R[3, 3] ≥ 0`). Inputs outside this
hemisphere still produce a valid Cholesky factor at recompose, but
`decompose ∘ recompose` returns the canonical gauge representative
(which differs from the input by gauge-equivalent angle shifts).

## Per-axis Cholesky decomposition algorithm (3D)

`cholesky_decompose_3d(L::SMatrix{3, 3, T, 9}) -> (α::SVector{3, T}, θ::SVector{3, T})`

Algorithm (per M3-7 design note §11 Q2 default — "use eigen for
correctness; profile-tune later"):

  1. Form `M = L · Lᵀ` (3×3 SPD), symmetrise to guard against round-off.
  2. `eigen(Symmetric(M))` → eigenvalues (ascending) + eigenvectors
     (`SMatrix{3, 3}` columns).
  3. Reverse to descending sort: `α_1 ≥ α_2 ≥ α_3 > 0`.
  4. Apply canonical gauge fix on the eigenvector matrix:
     - column 1 sign flip so `R[1, 1] ≥ 0`,
     - column 3 sign flip so `R[3, 3] ≥ 0`,
     - column 2 sign flip so `det(R) = +1` (proper rotation).
  5. Extract Cardan ZYX Euler angles from the canonical R:
     - `θ_13 = -asin(R[3, 1])` (clamped to `[-1, 1]` to handle
       round-off-driven values just outside the asin domain),
     - `θ_12 = atan2(R[2, 1], R[1, 1])`,
     - `θ_23 = atan2(R[3, 2], R[3, 3])`.

Allocation-free with `SMatrix{3, 3, Float64, 9}` inputs (verified by
`@allocated`-warmed closures: 0 bytes on the hot path).

## Verification gates (this commit)

| Gate | Magnitude | Pass |
|---|---:|---:|
| Round-trip on 50 random `(α, θ)` in canonical hemisphere | max_err = 4.5e-15 | ✓ (≤ 1e-12) |
| 2D reduction byte-equal (top-left 2×2 block at `θ_13 = θ_23 = 0`) | exact (`==`) | ✓ |
| 2D reduction angle recovery (`θ′[1] ≈ θ_R, θ′[2] = θ′[3] = 0`) | ≤ 1e-12 | ✓ |
| Iso-pullback (`α_1 = α_2 = α_3`): α-eigenvalues recovered | ≤ 1e-12 | ✓ |
| Iso-pullback: M = L Lᵀ preserved through `decompose ∘ recompose` | ≤ 1e-12 | ✓ |
| Iso-pullback: M = α² I structure on iso slice | ≤ 1e-12 | ✓ |
| Per-axis γ on anisotropic M_vv (3 entries by-hand, matrix form) | ≤ 1e-12 | ✓ |
| Per-axis γ M1-form (γ²_a = M_vv,aa − β_a²) by-hand | ≤ 1e-12 | ✓ |
| Per-axis γ realizability floor (γ ≥ 0 at `M_vv = 0`) | exact `== 0` | ✓ |
| Allocation: `cholesky_decompose_3d` on `SMatrix` input | 0 bytes | ✓ |
| Allocation: `cholesky_recompose_3d` on `SVector` inputs | 0 bytes | ✓ |
| Allocation: `gamma_per_axis_3d` (matrix + diagonal forms) | 0 bytes | ✓ |
| `rotation_matrix_3d`: orthogonal + det = +1 at random angles | ≤ 1e-12 | ✓ |
| `θ = (0, 0, 0)`: canonical L is diagonal with `L[a, a] = α_a` | exact (`==`) | ✓ |

## Iso-slice gauge note

On the iso slice `α_1 = α_2 = α_3` (so `M = α² I`), eigenvectors are
genuinely degenerate — every orthogonal R is a valid eigenvector basis.
The canonical gauge fix (`R[1, 1] ≥ 0`, `R[3, 3] ≥ 0`, `det(R) = +1`)
pins a deterministic representative, but it is **not** the input θ
(numerical eigen returns whatever the LAPACK driver computes — typically
some near-axis-aligned R that survives the gauge fix). The test
relaxes the iso-slice round-trip from "θ′ matches input θ" to "M is
preserved through decompose-then-recompose" — physically the only
gauge-invariant assertion. M3-7c's residual will need to handle this
gauge degeneracy; the standard answer is: ε-expansion off the iso
slice (M3-7 design note §7.3), where the Berry block contribution is
`O(ε²)` and the angle gauge has limited dynamical relevance.

## Q-resolution against the M3-7 design note's §11 open questions

| Q | Default | Resolution in M3-7 prep |
|---|---|---|
| Q1 — HaloView depth=2 in 3D | "stay at depth=1" (M3-3a pattern) | Deferred to M3-7a proper (this scaffolding does not touch HaloView). |
| Q2 — 3×3 eigendecomposition route | "use `LinearAlgebra.eigen`" | **Adopted**. `eigen(Symmetric(M))` on a 3×3 SMatrix; allocation-free when wrapped inline. |
| Q3 — off-diagonal β | "pin to zero" | **Adopted**. `DetField3D` has no off-diagonal β fields. M3-9 (3D D.1 KH) will add a parallel struct when needed. |
| Q4 — `Pp_a, Q_a` per-axis post-Newton | "stay at 13 dof per leaf for prep" | **Adopted**. M3-7c will extend with these when 3D D.7 / D.10 drivers need them. |
| Q5 — Euler-angle convention | "match SymPy script" | **Adopted intrinsic Cardan ZYX** (see top of `src/cholesky_DD_3d.jl` for the full pinning rationale). |

## 1D + 2D path bit-exact regression — confirmed

  * `test_phase1_zero_strain.jl` — 5 asserts pass.
  * `test_M3_3a_cholesky_DD.jl` (2D Cholesky driver) — 199 asserts pass.
  * `test_M3_3a_field_set_2d.jl` + `test_M3_3b_2d_zero_strain.jl` — 468
    asserts pass.

The new file `src/cholesky_DD_3d.jl` does not modify any 1D / 2D code
paths; the `dfmm.jl` change is append-only (new `include` + new
exports); the `types.jl` change is append-only. Project.toml gains
`LinearAlgebra` as a direct dependency (was already a transitive dep,
recorded in `Manifest.toml`); no version pin required since
`LinearAlgebra` is a Julia stdlib.

## M3-7a proper (post-Phase 3 M3-6) handoff items

When M3-6 Phase 3 closes and M3-7 proper launches, the launch agent
should:

  1. **Extend `src/cholesky_DD.jl`**: merge the 3D primitives from
     `src/cholesky_DD_3d.jl` into the canonical `cholesky_DD.jl`
     location (or keep them split — both work; the split was for
     M3-6 Phase 3 parallel-safety, not a permanent file structure).
     Decide based on M3-6 Phase 3's final shape.
  2. **Halo smoke test (3D)**: write `test/test_M3_7a_halo_smoke.jl`
     on a 4×4×4 balanced 3D `HierarchicalMesh{3}` (the 3D analog of
     `test/test_M3_3a_halo_smoke.jl`). Pin the `PolynomialView` halo
     contract at D=3.
  3. **Field set allocator** (`src/setups_2d.jl` → rename to
     `src/setups_multid.jl` or split into `setups_3d.jl`): add
     `allocate_cholesky_3d_fields(mesh::HierarchicalMesh{3})`
     producing a 16-named-field `PolynomialFieldSet` (13 Newton +
     `Pp, Q, s_post` post-Newton; M3-3a's 12-vs-14 split lifted to
     16 in 3D). The `read_detfield_3d` / `write_detfield_3d!` helpers
     follow the M3-3a pattern.
  4. **3D `cholesky_el_residual_3D!`** (M3-7b proper): per the M3-7
     design note §2.3 pseudo-code skeleton. The 3D analog of
     `cholesky_el_residual_2D!`. Dimension-lift gates §7.1a + §7.1b
     load-bearing.
  5. **3D Berry coupling** (M3-7c proper): consume `berry_partials_3d`
     from `src/berry.jl` directly (already verified at the stencil
     level, 797 asserts in `test_M3_prep_3D_berry_verification.jl`).
     Add the kernel-orthogonality residual on the θ_23 row per
     M3-7 design note §2.2 closed-form formula.
  6. **Per-axis γ AMR / realizability** (M3-7d proper): the per-axis
     γ functions in `src/cholesky_DD_3d.jl` are ready for consumption.
     The 3D `gamma_per_axis_3d_field` field-walking helper is M3-7d
     work (mirror M3-3d's 2D version in `src/diagnostics.jl`).
  7. **Gimbal-lock handling**: the canonical decompose currently
     handles `θ_13 = ±π/2` via the `clamp(R[3,1], -1, 1)` guard,
     but the resulting `(θ_12, θ_23)` pair is degenerate at the
     boundary. M3-7c's residual should guard against gimbal-lock
     ICs; the natural mitigation is to initialise `θ_13` away from
     `±π/2` in IC factories.
