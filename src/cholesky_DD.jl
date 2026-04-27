# cholesky_DD.jl
#
# Per-axis (principal-axis) Cholesky decomposition driver for the
# dimension-lifted 2D Cholesky-sector EL residual (Phase M3-3a / M3-3b).
#
# In M1's 1D code, the per-cell phase-space Cholesky factor reduces to a
# scalar (α, β) pair (γ derived from the EOS). In 2D the corresponding
# 2×2 symmetric positive-definite second-moment matrix `M_xx`
# diagonalizes into a principal-axis frame parametrized by a rotation
# angle `θ_R` and a pair of principal-axis Cholesky factors
# `α = (α_1, α_2)`. The `(α_a, β_a, θ_R)` triplet is the M3-3 Newton
# unknown set; the Berry kinetic term
#
#     Θ_rot^(2D) = (1/3)(α_1³ β_2 − α_2³ β_1) · dθ_R
#
# couples this triplet across cells. See
# `reference/notes_M3_3_2d_cholesky_berry.md` §2 + §4 for the full
# derivation, and `src/berry.jl` for the closed-form Berry stencil.
#
# This module provides the **decomposition / recomposition driver**
# only — it does NOT yet write the EL residual (that is M3-3b) and
# does NOT yet wire θ_R as a Newton unknown (that is M3-3c).
#
# Three primitive operations:
#
#   • `cholesky_decompose_2d(L) -> (α, θ_R)`
#       Take a 2×2 Cholesky factor `L` of a symmetric positive-definite
#       moment matrix `M = L Lᵀ` and return the principal-axis Cholesky
#       factors α = √eigenvalues(M) and the rotation angle θ_R from the
#       lab to the principal-axis frame.
#
#   • `cholesky_recompose_2d(α, θ_R) -> L`
#       Inverse of `cholesky_decompose_2d`: produce the canonical
#       Cholesky factor `L = R(θ_R) · diag(α)` whose square is the
#       diagonalized M = R diag(α²) Rᵀ. The decomposition is a gauge
#       choice; this canonical L makes `decompose ∘ recompose ≡ id`
#       and is the form M3-3b stores per-cell.
#
#   • `gamma_per_axis_2d(α, M_vv) -> SVector{2}`
#       Per-axis γ diagnostic. With M_vv the 2×2 second velocity
#       moment in the principal-axis frame, returns
#       `γ_a = √max(M_vv[a,a]/α_a² − 0, 0)` per axis. (For the M1
#       per-axis form γ² = M_vv − β² generalized to 2D, callers should
#       pass M_vv − β·βᵀ as the M_vv argument; this helper enforces
#       only the realizability floor.) See §4.3 of the design note.
#
# All routines are allocation-free given `StaticArrays` inputs.

using StaticArrays

# ──────────────────────────────────────────────────────────────────────
# Recomposition: (α, θ_R) → L
# ──────────────────────────────────────────────────────────────────────

"""
    cholesky_recompose_2d(α::SVector{2,T}, θ_R::Real) -> SMatrix{2, 2, T, 4}

Build the canonical 2D Cholesky factor `L = R(θ_R) · diag(α)` from
principal-axis factors `α = (α_1, α_2)` and rotation angle `θ_R`. The
square `L Lᵀ = R(θ_R) · diag(α²) · R(θ_R)ᵀ` reproduces the
symmetric positive-definite moment matrix `M_xx` whose eigenvalues
are `α_a²` and whose eigenvectors are the columns of `R(θ_R)`.

This is the M3-3 storage convention for the per-cell Cholesky factor:
the Newton unknowns are `(α_1, α_2, θ_R)` rather than the three
independent entries of `L`, so `L` is reconstructed only when needed
(e.g. for the discrete pressure stencil or the off-diagonal-β
extension under D.1 KH; see `src/berry.jl::kinetic_offdiag_2d`).

# Example (round-trip)
```julia
α  = SVector{2}(2.5, 0.7)
θR = 0.31
L  = cholesky_recompose_2d(α, θR)         # 2×2 SMatrix
α′, θR′ = cholesky_decompose_2d(L)
@assert α ≈ α′ && θR ≈ θR′                # ≤ 1e-14 absolute
```
"""
@inline function cholesky_recompose_2d(α::SVector{2,T}, θ_R::Real) where {T<:Real}
    θ = T(θ_R)
    c = cos(θ)
    s = sin(θ)
    a1, a2 = α[1], α[2]
    # L = R(θ_R) * diag(α) — column-major SMatrix constructor.
    return SMatrix{2, 2, T, 4}(
        c * a1, s * a1,    # column 1: R[:,1] * α_1
       -s * a2, c * a2,    # column 2: R[:,2] * α_2
    )
end

# ──────────────────────────────────────────────────────────────────────
# Decomposition: L → (α, θ_R)
# ──────────────────────────────────────────────────────────────────────

"""
    cholesky_decompose_2d(L::SMatrix{2, 2, T, 4}) -> (α::SVector{2,T}, θ_R::T)

Extract principal-axis Cholesky factors `α = (α_1, α_2)` and rotation
angle `θ_R` from a 2×2 Cholesky factor `L`. The decomposition follows
the eigendecomposition of `M = L · Lᵀ`:

  • `α_a² = λ_a(M)` (the eigenvalues, sorted by the convention below);
  • `θ_R` is the rotation angle from the lab frame to the
    principal-axis (eigenvector) frame, in `(-π/2, π/2]`.

# Sign / sort convention

We sort `α_1 ≥ α_2 > 0` (largest principal axis first). With this
sort and the canonical recomposition `L_canonical = R(θ_R) · diag(α)`,
`cholesky_decompose_2d ∘ cholesky_recompose_2d ≡ id` modulo
floating-point round-off (verified to ≤ 1e-14 absolute by
`test_M3_3a_cholesky_DD.jl`).

The angle convention places `θ_R = 0` when `L` is already diagonal
with `L[1,1] ≥ L[2,2]`. For the iso case `α_1 = α_2`, the angle is
under-determined; we return `θ_R = 0`. (The Berry kinetic term
vanishes on the iso slice, so this is consistent with the
iso-pullback ε-expansion check of §6.3 of the design note.)

# Closed-form (2×2)

Let `M = L · Lᵀ` with entries `M = [[a, b], [b, c]]`. Then

    mean = (a + c) / 2
    diff = (a - c) / 2
    D    = √(diff² + b²)
    λ_1  = mean + D                      (largest)
    λ_2  = mean - D                      (smallest, ≥ 0 since M ⪰ 0)
    θ_R  = atan(2b, a - c) / 2

`α_a = √λ_a`. The `atan(2b, a-c) / 2` form gives `θ_R ∈ (-π/2, π/2]`
and is numerically stable across the iso boundary `a = c, b = 0`
(`atan(0, 0)` returns 0 in Julia).
"""
@inline function cholesky_decompose_2d(L::SMatrix{2, 2, T, 4}) where {T<:Real}
    # M = L * Lᵀ
    L11, L12 = L[1, 1], L[1, 2]
    L21, L22 = L[2, 1], L[2, 2]
    a = L11^2 + L12^2
    b = L11 * L21 + L12 * L22
    c = L21^2 + L22^2

    mean = (a + c) / 2
    diff = (a - c) / 2
    D = sqrt(diff^2 + b^2)
    λ1 = mean + D    # largest
    λ2 = mean - D    # smallest, ≥ 0 since M ⪰ 0; clamp for round-off

    α1 = sqrt(max(λ1, zero(T)))
    α2 = sqrt(max(λ2, zero(T)))

    # atan2(2b, a−c) / 2 places θ_R ∈ (-π/2, π/2].
    θ_R = atan(2 * b, a - c) / 2

    return (SVector{2, T}(α1, α2), θ_R)
end

# ──────────────────────────────────────────────────────────────────────
# Per-axis γ diagnostic
# ──────────────────────────────────────────────────────────────────────

"""
    gamma_per_axis_2d(α::SVector{2,T}, M_vv::SMatrix{2,2,T,4}) -> SVector{2,T}

Per-axis γ diagnostic in the principal-axis frame: returns

    γ_a = √max(M_vv[a, a] / α_a² , 0)         for a = 1, 2.

This is the per-axis 2D analog of M1's `gamma_from_Mvv(β, M_vv)` with
the per-axis β contribution already subtracted by the caller (i.e.,
callers should pass `M_vv − β · βᵀ` as `M_vv` here when they want the
γ² = M_vv − β² convention). The helper enforces only the
realizability floor `γ_a² ≥ 0`.

# Why divide by `α_a²`?

In the principal-axis frame, `M_xx = diag(α²)`, so the velocity
ellipsoid's per-axis variance scaled to the position ellipsoid is
`M_vv,aa / α_a²`. M1's 1D scalar reduces to `γ² = M_vv − β²` in the
mass-weighted frame where `α = 1`; the 2D generalization includes
the per-axis α normalization explicitly.

# Use cases (see §4.3 of the design note)

- AMR indicator selectivity: refine only along axes where
  `γ_a → 0` (the C.2 selectivity test).
- Hessian-degeneracy regularization (per-axis γ-floor protects the
  per-axis Newton solve when one axis collapses while another doesn't).
- Per-axis stochastic injection (M2-3 / Phase 8 carry-over for D.1 KH).

The argument is a 2×2 `SMatrix` so callers can pass either the
diagonal-only case (`SMatrix{2,2,T,4}(M11, 0, 0, M22)`) or the full
matrix; only the diagonal entries are read.
"""
@inline function gamma_per_axis_2d(α::SVector{2, T},
                                    M_vv::SMatrix{2, 2, T, 4}) where {T<:Real}
    a1, a2 = α[1], α[2]
    # γ² per axis = M_vv,aa / α_a²; floor at 0 for realizability.
    g1² = M_vv[1, 1] / (a1 * a1)
    g2² = M_vv[2, 2] / (a2 * a2)
    γ1 = sqrt(max(g1², zero(T)))
    γ2 = sqrt(max(g2², zero(T)))
    return SVector{2, T}(γ1, γ2)
end
