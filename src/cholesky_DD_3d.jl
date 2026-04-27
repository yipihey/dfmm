# cholesky_DD_3d.jl
#
# Per-axis (principal-axis) Cholesky decomposition driver for the
# dimension-lifted 3D Cholesky-sector EL residual (Phase M3-7 prep).
#
# This file is the 3D analog of `src/cholesky_DD.jl` (which provides
# the 2D form `cholesky_decompose_2d` / `cholesky_recompose_2d` /
# `gamma_per_axis_2d`). It is intentionally factored into a separate
# file for parallel-safety: M3-6 Phase 3 has the 2D file open in a
# sibling worktree, so the M3-7 prep work goes here.
#
# The three primitives provided:
#
#   • `cholesky_decompose_3d(L) -> (α, θ)` — extract per-axis Cholesky
#     factors and three Euler angles from a 3×3 Cholesky factor (or
#     SPD moment matrix `M = L Lᵀ`).
#
#   • `cholesky_recompose_3d(α, θ) -> SMatrix{3, 3}` — inverse:
#     produce the canonical Cholesky factor `L = R(θ) · diag(α)` whose
#     square `L Lᵀ = R · diag(α²) · Rᵀ` is the diagonalised SPD moment
#     matrix.
#
#   • `gamma_per_axis_3d(β, M_vv_diag) -> SVector{3}` (M1 form) and
#     `gamma_per_axis_3d(α, M_vv) -> SVector{3}` (matrix form) — per-
#     axis γ diagnostic in the principal-axis frame.
#
# # SO(3) Euler-angle convention (intrinsic Cardan ZYX)
#
# **CRITICAL**: pin this convention before reading the code. The three
# Euler angles `(θ_12, θ_13, θ_23)` parameterize the rotation as the
# composition of three Givens rotations in the (a, b)-coordinate
# planes:
#
#     R(θ_12, θ_13, θ_23) = R_12(θ_12) · R_13(θ_13) · R_23(θ_23)
#
# where each `R_{ab}(θ)` is the elementary 3×3 rotation in the
# (a, b)-plane (cosines on the (a, a), (b, b) diagonal entries, sines
# off-diagonal with positive sin in the (b, a) entry, identity
# elsewhere):
#
#     R_12(θ) = [[ c, -s, 0],   R_13(θ) = [[ c, 0, -s],   R_23(θ) = [[1,  0,  0],
#                [ s,  c, 0],              [ 0, 1,  0],              [0,  c, -s],
#                [ 0,  0, 1]]              [ s, 0,  c]]              [0,  s,  c]]
#
# This is the **same convention** as the SymPy authority script
# `scripts/verify_berry_connection_3D.py` uses implicitly: the script
# treats `(θ_12, θ_13, θ_23)` as three independent pair-rotation
# generator angles in the 9D phase space, and the Berry stencils in
# `src/berry.jl::berry_F_3d` compute their per-pair Berry coefficients
# `F_{ab}` in this same generator basis. The choice of finite-rotation
# composition order (R_12 · R_13 · R_23, intrinsic) is the
# "Cardan ZYX intrinsic" convention (since R_12 rotates about axis 3
# / "Z", R_13 about axis 2 / "Y", R_23 about axis 1 / "X"). The SymPy
# script does not constrain the composition order at the symbolic
# level (it only uses infinitesimal forms via `sp.diff`), so we are
# free to pick a finite-rotation convention that is consistent at the
# infinitesimal level. The ZYX Cardan choice has the property that:
#
#   • At `θ_13 = θ_23 = 0` it reduces to `R = R_12(θ_12)` — the 2D
#     `R(θ_R)` from `cholesky_DD.jl::cholesky_recompose_2d` byte-
#     equally — matching the 3D 2D-symmetric ⊂ 2D dimension-lift gate
#     (CHECK 3b of `notes_M3_prep_3D_berry_verification.md`).
#   • At `θ_12 = θ_13 = 0` it reduces to `R = R_23(θ_23)` — a pure 2D
#     rotation in the (2, 3)-plane fixing axis 1. This is the
#     "swap-axis-roles" 2D dimension lift that M3-7b will exercise.
#   • At full isotropy `α_1 = α_2 = α_3` the three angles are gauge
#     under-determined (any rotation preserves M); we adopt the
#     convention `θ_12 = θ_13 = θ_23 = 0` (the Berry kinetic term
#     vanishes anyway on the iso slice — CHECK 3a of the verification
#     note).
#
# # Round-trip + sort convention
#
# The decomposition `cholesky_decompose_3d` returns:
#   • `α = (α_1, α_2, α_3)` sorted **descending** (largest first).
#   • Euler angles `(θ_12, θ_13, θ_23)` with `θ_13 ∈ (-π/2, π/2)` (no
#     gimbal-lock branch below the lift gates; M3-7 prep avoids the
#     boundary `θ_13 = ±π/2`), `θ_12, θ_23 ∈ (-π, π]` (atan2 range).
#
# The canonical recomposition `cholesky_recompose_3d(α, θ) = R · diag(α)`
# satisfies `decompose ∘ recompose ≡ id` modulo floating-point
# round-off (≤ 1e-12 absolute on 50 random samples; verified in
# `test/test_M3_7_prep_3d_scaffolding.jl`).
#
# # Algorithm: eigendecomposition route
#
# `cholesky_decompose_3d(L)` proceeds as follows:
#   1. Form the 3×3 SPD matrix `M = L Lᵀ`.
#   2. Compute `eigen(Symmetric(M))` (StaticArrays-aware: returns
#      `SVector{3}` eigenvalues and `SMatrix{3,3}` eigenvectors).
#      LinearAlgebra returns eigenvalues in ascending order; reverse
#      to get descending sort.
#   3. Reorder eigenvectors to match the descending sort. Adjust the
#      sign of column 3 so `det(R) = +1` (proper rotation).
#   4. Extract Euler angles from R via the closed-form Cardan ZYX
#      inversion:
#        θ_13 = -asin(R[3, 1])
#        θ_12 = atan2(R[2, 1], R[1, 1])
#        θ_23 = atan2(R[3, 2], R[3, 3])
#   5. Set `α = √eigenvalues` (descending).
#
# Per the M3-7 design note §11 Q2 default ("use `LinearAlgebra.eigen`
# for correctness; profile-tune later"). The wrapped inline path is
# allocation-free for `SMatrix{3, 3}` inputs.
#
# All routines are allocation-free given `StaticArrays` inputs.

using StaticArrays
using LinearAlgebra: eigen, Symmetric, det

# ──────────────────────────────────────────────────────────────────────
# Recomposition: (α, θ) → L
# ──────────────────────────────────────────────────────────────────────

"""
    rotation_matrix_3d(θ_12::Real, θ_13::Real, θ_23::Real) -> SMatrix{3, 3}

Build the SO(3) rotation matrix in the intrinsic Cardan ZYX convention:

    R = R_12(θ_12) · R_13(θ_13) · R_23(θ_23)

where `R_{ab}(θ)` is the elementary Givens rotation in the (a, b)-
coordinate plane (positive sin in the (b, a) entry).

Allocation-free with `Float64` arguments; the eltype follows the
promotion of the three angle arguments.

See the top-of-file docstring for the convention pinning relative to
the SymPy authority script `scripts/verify_berry_connection_3D.py`.
"""
@inline function rotation_matrix_3d(θ_12::Real, θ_13::Real, θ_23::Real)
    T = promote_type(typeof(float(θ_12)), typeof(float(θ_13)), typeof(float(θ_23)))
    c12, s12 = cos(T(θ_12)), sin(T(θ_12))
    c13, s13 = cos(T(θ_13)), sin(T(θ_13))
    c23, s23 = cos(T(θ_23)), sin(T(θ_23))

    # R = R_12 · R_13 · R_23 (intrinsic ZYX Cardan)
    # Closed-form expansion. Verified by hand and by the round-trip
    # test in test_M3_7_prep_3d_scaffolding.jl.
    R11 =  c12 * c13
    R12 =  c12 * s13 * s23 - s12 * c23
    R13 =  c12 * s13 * c23 + s12 * s23
    R21 =  s12 * c13
    R22 =  s12 * s13 * s23 + c12 * c23
    R23 =  s12 * s13 * c23 - c12 * s23
    R31 = -s13
    R32 =  c13 * s23
    R33 =  c13 * c23

    return SMatrix{3, 3, T, 9}(
        R11, R21, R31,    # column 1
        R12, R22, R32,    # column 2
        R13, R23, R33,    # column 3
    )
end

"""
    cholesky_recompose_3d(α::SVector{3, T}, θ::SVector{3, T}) -> SMatrix{3, 3, T, 9}

Build the canonical 3D Cholesky factor `L = R(θ) · diag(α)` from
principal-axis factors `α = (α_1, α_2, α_3)` (sorted descending by
convention) and the three Euler angles `θ = (θ_12, θ_13, θ_23)`. The
square `L Lᵀ = R · diag(α²) · Rᵀ` reproduces the symmetric positive-
definite moment matrix `M_xx` whose eigenvalues are `α_a²` and whose
eigenvectors are the columns of `R(θ)`.

This is the M3-7 storage convention for the per-cell Cholesky factor:
the Newton unknowns are `(α_1, α_2, α_3, θ_12, θ_13, θ_23)` rather
than the six independent entries of `L`, so `L` is reconstructed only
when needed (e.g. for the discrete pressure stencil or the per-axis
γ diagnostic).

# Example (round-trip)
```julia
α = SVector{3}(2.5, 1.2, 0.7)
θ = SVector{3}(0.31, -0.18, 0.42)
L = cholesky_recompose_3d(α, θ)
α′, θ′ = cholesky_decompose_3d(L)
@assert α ≈ α′ && θ ≈ θ′                # ≤ 1e-12 absolute
```

See the top-of-file docstring for the SO(3) Euler-angle convention.
"""
@inline function cholesky_recompose_3d(α::SVector{3, T},
                                       θ::SVector{3, T}) where {T<:Real}
    R = rotation_matrix_3d(θ[1], θ[2], θ[3])
    a1, a2, a3 = α[1], α[2], α[3]
    # L = R * diag(α). Build column-by-column.
    return SMatrix{3, 3, T, 9}(
        R[1, 1] * a1, R[2, 1] * a1, R[3, 1] * a1,    # column 1: R[:,1] * α_1
        R[1, 2] * a2, R[2, 2] * a2, R[3, 2] * a2,    # column 2: R[:,2] * α_2
        R[1, 3] * a3, R[2, 3] * a3, R[3, 3] * a3,    # column 3: R[:,3] * α_3
    )
end

# ──────────────────────────────────────────────────────────────────────
# Decomposition: L → (α, θ)
# ──────────────────────────────────────────────────────────────────────

"""
    cholesky_decompose_3d(L::SMatrix{3, 3, T, 9}) -> (α::SVector{3, T}, θ::SVector{3, T})

Extract principal-axis Cholesky factors `α = (α_1, α_2, α_3)` and three
Euler angles `θ = (θ_12, θ_13, θ_23)` from a 3×3 Cholesky factor `L`.
The decomposition follows the eigendecomposition of `M = L · Lᵀ`:

  • `α_a² = λ_a(M)` (the eigenvalues, sorted descending);
  • `R` is the rotation matrix whose columns are the corresponding
    (sorted, sign-corrected) eigenvectors;
  • `(θ_12, θ_13, θ_23)` are the intrinsic Cardan ZYX Euler angles
    of `R` (see top-of-file docstring for the convention).

# Sort / sign convention

We sort `α_1 ≥ α_2 ≥ α_3 > 0` (largest principal axis first). The
sign of column 3 of the eigenvector matrix is flipped if needed to
make `det(R) = +1` (proper rotation; reflections are not part of the
SO(3) parameterization).

# Iso slice (α_1 = α_2 = α_3)

The three angles are gauge under-determined on the iso slice (any
rotation preserves `M = α² I`). We do not special-case this; the
extraction returns whatever Euler angles result from the
eigendecomposition's sorted eigenvectors. The Berry kinetic term
vanishes on the iso slice (CHECK 3a of
`notes_M3_prep_3D_berry_verification.md`), so this is consistent with
the 3D iso-pullback ε-expansion (M3-7 design note §7.3).

# Closed-form Euler-angle extraction (ZYX Cardan inversion)

Given the rotation matrix R, the Cardan ZYX inversion is:

    θ_13 = -asin(R[3, 1])                    in (-π/2, π/2)
    θ_12 = atan2(R[2, 1], R[1, 1])           in (-π, π]
    θ_23 = atan2(R[3, 2], R[3, 3])           in (-π, π]

The gimbal-lock boundary `θ_13 = ±π/2` (where `R[3, 1] = ∓1`) makes
the (θ_12, θ_23) pair degenerate; in M3-7 prep we avoid this branch
(no test samples land near `θ_13 = ±π/2`). M3-7c will need a robust
gimbal-lock handler if 3D Tier-D ICs land near the boundary.

# Algorithm

  1. Form `M = L · Lᵀ` (3×3 symmetric SPD).
  2. `eigen(Symmetric(M))` → eigenvalues (ascending) + eigenvectors
     (`SMatrix{3, 3}` columns).
  3. Reverse to descending sort; permute eigenvector columns
     accordingly.
  4. If `det(R_sorted) < 0`, flip the sign of column 3 (this gives a
     proper rotation; the eigenvectors are determined up to sign, so
     this is a free gauge choice).
  5. Extract `(θ_12, θ_13, θ_23)` via the closed-form ZYX Cardan
     inversion above.

Per the M3-7 design note §11 Q2 default ("use eigen for correctness;
profile-tune later"). Allocation-free with `SMatrix{3, 3}` inputs
(verified by `@allocated` after warm-up; ≤ 0 bytes).
"""
@inline function cholesky_decompose_3d(L::SMatrix{3, 3, T, 9}) where {T<:Real}
    # 1. Form M = L * L'.
    M = L * transpose(L)
    # Symmetrize (guards against tiny round-off asymmetry that can
    # propagate through eigen).
    M_sym = (M + transpose(M)) / 2

    # 2. eigen(Symmetric(M)). LinearAlgebra returns eigenvalues
    # ascending and eigenvectors as columns of a matrix; for SMatrix
    # input the result fields are SVector / SMatrix.
    e = eigen(Symmetric(M_sym))
    λ_asc = SVector{3, T}(e.values)
    V_asc = SMatrix{3, 3, T, 9}(e.vectors)

    # 3. Descending sort: reverse the column order.
    λ = SVector{3, T}(λ_asc[3], λ_asc[2], λ_asc[1])
    R_pre = SMatrix{3, 3, T, 9}(
        V_asc[1, 3], V_asc[2, 3], V_asc[3, 3],   # column 1: largest eigenvector
        V_asc[1, 2], V_asc[2, 2], V_asc[3, 2],   # column 2: middle
        V_asc[1, 1], V_asc[2, 1], V_asc[3, 1],   # column 3: smallest
    )

    # 4. Canonical gauge fix.
    #
    # Eigenvectors are determined only up to sign per column. There
    # are 8 sign-equivalent versions of `R_pre`, of which 4 satisfy
    # `det(R) = +1` (proper SO(3) rotations); the other 4 are
    # improper (reflections). We pin the canonical SO(3)
    # representative by enforcing three constraints:
    #
    #   (a) `R[1, 1] ≥ 0` — fixes column 1 sign. In the ZYX Cardan
    #       parameterization `R[1, 1] = cos(θ_12) cos(θ_13)`. Combined
    #       with `θ_13 ∈ (-π/2, π/2)` (which makes `cos(θ_13) > 0`),
    #       this is equivalent to `θ_12 ∈ [-π/2, π/2]`. This is the
    #       3D analog of the 2D convention `θ_R ∈ (-π/2, π/2]`
    #       (where `cos(θ_R) ≥ 0`).
    #
    #   (b) `R[3, 3] ≥ 0` — fixes column 3 sign. `R[3, 3] =
    #       cos(θ_13) cos(θ_23)`, so this is equivalent to
    #       `θ_23 ∈ [-π/2, π/2]` under the same `θ_13` range
    #       restriction.
    #
    #   (c) `det(R) = +1` — fixes column 2 sign so that the result
    #       is a proper rotation. Combined with (a) and (b), the
    #       candidate is unique.
    #
    # **Round-trip implication.** With these gauges,
    # `cholesky_decompose_3d ∘ cholesky_recompose_3d ≡ id` provided
    # the input `(α, θ)` to recompose has:
    #   • `θ_13 ∈ (-π/2, π/2)` (gimbal-lock free; the natural domain
    #     for the ZYX Cardan parameterization),
    #   • `θ_12 ∈ (-π/2, π/2]` (canonical "near-identity" hemisphere),
    #   • `θ_23 ∈ (-π/2, π/2]` (canonical hemisphere).
    #
    # This is the natural 3D analog of the 2D round-trip range
    # `θ_R ∈ (-π/2, π/2]` from `cholesky_decompose_2d`. Inputs
    # outside this range still produce a valid Cholesky factor at
    # recompose, but `decompose ∘ recompose` returns the canonical
    # gauge representative (which differs from the input by a
    # combination of `(θ_12, θ_23) → (θ_12 + π, θ_23 + π)` or sign
    # flips that leave the underlying rotation invariant). Tests
    # should sample in the canonical hemisphere.
    s1 = ifelse(R_pre[1, 1] < zero(T), -one(T), one(T))
    R_c1 = SMatrix{3, 3, T, 9}(
        s1 * R_pre[1, 1], s1 * R_pre[2, 1], s1 * R_pre[3, 1],
        R_pre[1, 2],      R_pre[2, 2],      R_pre[3, 2],
        R_pre[1, 3],      R_pre[2, 3],      R_pre[3, 3],
    )
    s3 = ifelse(R_c1[3, 3] < zero(T), -one(T), one(T))
    R_c3 = SMatrix{3, 3, T, 9}(
        R_c1[1, 1],       R_c1[2, 1],       R_c1[3, 1],
        R_c1[1, 2],       R_c1[2, 2],       R_c1[3, 2],
        s3 * R_c1[1, 3],  s3 * R_c1[2, 3],  s3 * R_c1[3, 3],
    )
    s2 = ifelse(det(R_c3) < zero(T), -one(T), one(T))
    R = SMatrix{3, 3, T, 9}(
        R_c3[1, 1],       R_c3[2, 1],       R_c3[3, 1],
        s2 * R_c3[1, 2],  s2 * R_c3[2, 2],  s2 * R_c3[3, 2],
        R_c3[1, 3],       R_c3[2, 3],       R_c3[3, 3],
    )

    # 5. Extract Euler angles via ZYX Cardan inversion.
    # Clamp R[3, 1] to [-1, 1] (round-off can push it outside).
    R31_clamped = clamp(R[3, 1], -one(T), one(T))
    θ_13 = -asin(R31_clamped)
    θ_12 = atan(R[2, 1], R[1, 1])
    θ_23 = atan(R[3, 2], R[3, 3])

    α1 = sqrt(max(λ[1], zero(T)))
    α2 = sqrt(max(λ[2], zero(T)))
    α3 = sqrt(max(λ[3], zero(T)))

    return (SVector{3, T}(α1, α2, α3), SVector{3, T}(θ_12, θ_13, θ_23))
end

# ──────────────────────────────────────────────────────────────────────
# Per-axis γ diagnostic (3D)
# ──────────────────────────────────────────────────────────────────────

"""
    gamma_per_axis_3d(α::SVector{3, T}, M_vv::SMatrix{3, 3, T, 9}) -> SVector{3, T}

Per-axis γ diagnostic in the 3D principal-axis frame: returns

    γ_a = √max(M_vv[a, a] / α_a² , 0)         for a = 1, 2, 3.

This is the per-axis 3D analog of M1's `gamma_from_Mvv(β, M_vv)` with
the per-axis β contribution already subtracted by the caller (i.e.,
callers should pass `M_vv − β · βᵀ` as `M_vv` here when they want the
γ² = M_vv − β² convention). The helper enforces only the
realizability floor `γ_a² ≥ 0`.

# Why divide by `α_a²`?

In the 3D principal-axis frame, `M_xx = diag(α²)`, so the velocity
ellipsoid's per-axis variance scaled to the position ellipsoid is
`M_vv,aa / α_a²`. M1's 1D scalar reduces to `γ² = M_vv − β²` in the
mass-weighted frame where `α = 1`; the 3D generalization includes
the per-axis α normalization explicitly, mirroring `gamma_per_axis_2d`.

# Use cases (M3-7 design note §4.3)

- **AMR indicator selectivity** (per-axis rather than scalar). The
  3D analog of M3-3d's selectivity test fires only along axes where
  `γ_a → 0` (compressive directions).
- **Hessian-degeneracy regularization** (M1 carry-over). The per-axis
  γ-floor protects the per-axis Newton solve when one or two axes
  collapse while the third doesn't (3D Zel'dovich pancake).
- **Per-axis stochastic injection** (M2-3 carry-over). Injects only
  on the compressive axis (or axes).
"""
@inline function gamma_per_axis_3d(α::SVector{3, T},
                                   M_vv::SMatrix{3, 3, T, 9}) where {T<:Real}
    a1, a2, a3 = α[1], α[2], α[3]
    g1² = M_vv[1, 1] / (a1 * a1)
    g2² = M_vv[2, 2] / (a2 * a2)
    g3² = M_vv[3, 3] / (a3 * a3)
    γ1 = sqrt(max(g1², zero(T)))
    γ2 = sqrt(max(g2², zero(T)))
    γ3 = sqrt(max(g3², zero(T)))
    return SVector{3, T}(γ1, γ2, γ3)
end

"""
    gamma_per_axis_3d(β::SVector{3, T}, M_vv_diag::SVector{3, T}) -> SVector{3, T}

M1-style per-axis γ diagnostic in the convention `γ²_a = M_vv,aa − β_a²`
(the per-axis lift of M1's `gamma_from_state(M_vv, β) = √max(M_vv − β², 0)`).
Inputs are the per-axis β and the diagonal of `M_vv` in the 3D principal-
axis frame (i.e. `M_vv,aa` for `a = 1, 2, 3`). The realizability floor
`γ² ≥ 0` is enforced.

This is the form consumed by the M3-7d AMR + realizability-projection
per-axis wiring (M3-3d 2D pattern lifted to 3D). The principal-axis
M_vv per axis equals `Mvv(J, s)` for an isotropic EOS — the EOS does
not "see" the rotation, so `M_vv,aa = M_vv(J,s)` on all three axes —
but the per-axis form keeps the API ready for direction-dependent
extensions.

# Example
```julia
β = SVector(0.10, 0.05, 0.03)
Mvv_diag = SVector(0.40, 0.30, 0.20)
γ = gamma_per_axis_3d(β, Mvv_diag)
# γ_1 = √(0.40 - 0.01) = √0.39
# γ_2 = √(0.30 - 0.0025) = √0.2975
# γ_3 = √(0.20 - 0.0009) = √0.1991
```
"""
@inline function gamma_per_axis_3d(β::SVector{3, T},
                                   M_vv_diag::SVector{3, T}) where {T<:Real}
    g1² = M_vv_diag[1] - β[1] * β[1]
    g2² = M_vv_diag[2] - β[2] * β[2]
    g3² = M_vv_diag[3] - β[3] * β[3]
    γ1 = sqrt(max(g1², zero(T)))
    γ2 = sqrt(max(g2², zero(T)))
    γ3 = sqrt(max(g3², zero(T)))
    return SVector{3, T}(γ1, γ2, γ3)
end
