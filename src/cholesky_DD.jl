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

"""
    gamma_per_axis_2d(β::SVector{2, T}, M_vv_diag::SVector{2, T}) -> SVector{2, T}

M1-style per-axis γ diagnostic in the convention `γ²_a = M_vv,aa − β_a²`
(the per-axis lift of M1's `gamma_from_state(M_vv, β) = √max(M_vv − β², 0)`).
Inputs are the per-axis β and the diagonal of `M_vv` in the principal-axis
frame (i.e. `M_vv,aa` for `a = 1, 2`). The realizability floor `γ² ≥ 0` is
enforced.

This is the form consumed by the M3-3d AMR + realizability-projection per-
axis wiring. The principal-axis Mvv per axis equals `Mvv(J, s)` for an
isotropic EOS — the EOS does not "see" the rotation, so `M_vv,aa = M_vv(J,s)`
on both axes — but the per-axis form keeps the API ready for direction-
dependent extensions (off-diagonal β, anisotropic EOS, …).

# Example
```julia
β = SVector(0.10, 0.05)
Mvv_diag = SVector(0.40, 0.30)
γ = gamma_per_axis_2d(β, Mvv_diag)
# γ_1 = √(0.40 - 0.01) = √0.39
# γ_2 = √(0.30 - 0.0025) = √0.2975
```
"""
@inline function gamma_per_axis_2d(β::SVector{2, T},
                                    M_vv_diag::SVector{2, T}) where {T<:Real}
    g1² = M_vv_diag[1] - β[1] * β[1]
    g2² = M_vv_diag[2] - β[2] * β[2]
    γ1 = sqrt(max(g1², zero(T)))
    γ2 = sqrt(max(g2², zero(T)))
    return SVector{2, T}(γ1, γ2)
end


# Per-axis field-walking helper `gamma_per_axis_2d_field` lives in
# `src/diagnostics.jl` (M3-3d). It depends on `Mvv(J, s)` from `eos.jl`
# and the 2D Cholesky-sector field set allocated by
# `allocate_cholesky_2d_fields` in `src/setups_2d.jl`.

# ──────────────────────────────────────────────────────────────────────
# Per-species per-axis γ diagnostic — M3-6 Phase 3 (c)
# ──────────────────────────────────────────────────────────────────────

"""
    gamma_per_axis_2d_per_species(β::SVector{2, T},
                                    M_vv_diag_per_species::AbstractVector)
        -> Matrix{T}

Per-species per-axis γ math primitive. Generalises
`gamma_per_axis_2d(β, M_vv_diag)` to multiple species, each with its
own per-axis `M_vv_diag` 2-tuple. Returns a `(n_species, 2)` matrix
of γ values:

    γ[k, a] = √max(M_vv_diag_per_species[k][a] − β[a]², 0)

The shared per-axis `β` is the fluid-state Cholesky factor (one per
fluid cell, NOT per species — the variational scheme has a single
fluid β shared across all passive scalar species). Each species'
γ²_a = M_vv,aa(species k) − β_a² differs only via the species-
specific `M_vv,aa` (e.g. dust ⇒ `M_vv = 0` ⇒ γ = 0; gas ⇒
`M_vv = Mvv(J, s)` ⇒ γ as in the single-species form).

# Use cases

  • D.7 dust traps: `M_vv_dust = (0, 0)` (pressureless cold dust)
    and `M_vv_gas = (Mvv(J, s), Mvv(J, s))` (gas EOS) — the per-
    species γ tracks how the cold dust streams collapse independently
    of the gas equation of state.
  • D.10 ISM tracers: per-species `M_vv` may carry the species-
    dependent thermal velocity dispersion (warm/hot/cold ISM phases).

The single-species `n_species == 1` path reduces byte-equally to
`gamma_per_axis_2d(β, M_vv_diag_per_species[1])`.

# Math source

Per-axis γ²_a = M_vv,aa − β_a² lifts trivially to the multi-species
case because `β` is a fluid-state field (one β per fluid cell)
shared across passive species. The per-species `M_vv,aa(k)` then
parametrises the per-species γ.
"""
@inline function gamma_per_axis_2d_per_species(β::SVector{2, T},
                                                 M_vv_diag_per_species
                                                 ) where {T<:Real}
    K = length(M_vv_diag_per_species)
    out = zeros(T, K, 2)
    @inbounds for k in 1:K
        m = M_vv_diag_per_species[k]
        γ1² = T(m[1]) - β[1] * β[1]
        γ2² = T(m[2]) - β[2] * β[2]
        out[k, 1] = sqrt(max(γ1², zero(T)))
        out[k, 2] = sqrt(max(γ2², zero(T)))
    end
    return out
end

# ──────────────────────────────────────────────────────────────────────
# H_rot solvability constraint (M3-3c §6.4)
# ──────────────────────────────────────────────────────────────────────

"""
    h_rot_partial_dtheta(α::SVector{2,T}, β::SVector{2,T}, γ²::SVector{2,T})
        -> T

Closed-form value of `∂H_rot/∂θ_R` derived from the kernel-orthogonality
condition `dH · v_ker = 0` on the 5D Poisson manifold. With the kernel
direction
`v_ker = (-α_2³/(3α_1²), -β_2, α_1³/(3α_2²), β_1, 1)` in basis
`(α_1, β_1, α_2, β_2, θ_R)` (verified by `scripts/verify_berry_connection.py`
CHECK 7), the constraint `dH_Ch · v_ker + ∂H_rot/∂θ_R = 0` gives

    ∂H_rot/∂θ_R = − (γ_1² · α_2³)/(3·α_1) + (γ_2² · α_1³)/(3·α_2)
                   + (α_1² − α_2²) · β_1 · β_2.

This makes `H = H_Ch + H_rot(θ_R, …)` a valid Hamiltonian on the
rank-4 5D Poisson manifold.

# Sign convention

The closed form here has the **opposite overall sign** of the
narrative in `reference/notes_M3_phase0_berry_connection.md` §6.6
(which writes the magnitude only). The sign here matches the SymPy
verification output of `scripts/verify_berry_connection.py` CHECK 7:
solving `Ω · X = -dH` for the θ_R row yields
`h_rot = -[(γ_1² α_2³)/(3α_1) - (γ_2² α_1³)/(3α_2) - (α_1²-α_2²)β_1β_2]`.

# Algebraic guarantee in the discrete EL

The relation is **automatically satisfied by the per-axis residual
rows** of `cholesky_el_residual_2D_berry!`: substituting the Berry-
modified Hamilton equations (rows of Ω · X = -dH solved for α̇_a, β̇_a)
into the θ_R row identity makes the LHS equal to the closed form
above. The discrete residual encodes the per-axis rows directly,
so the kernel-orthogonality identity is structurally guaranteed at
every Newton iterate. This helper returns the closed form so the
M3-3c §6.4 verification gate can cross-check it numerically against
a kernel-orthogonality probe (`h_rot_kernel_orthogonality_residual`).

# Iso-slice behaviour

At `α_1 = α_2 = α_0`, `β_1 = β_2 = β_0`, the closed form simplifies to

    ∂H_rot/∂θ_R = -γ_1² α_0² / 3 + γ_2² α_0² / 3 + 0
                = -(γ_1² − γ_2²) · α_0² / 3.

When additionally `M_vv,1 = M_vv,2` (so `γ_1² = γ_2²`), this vanishes
identically — consistent with the iso-slice being a Lagrangian
submanifold of the 5D Poisson manifold.
"""
@inline function h_rot_partial_dtheta(α::SVector{2, T},
                                       β::SVector{2, T},
                                       γ²::SVector{2, T}) where {T<:Real}
    a1, a2 = α[1], α[2]
    b1, b2 = β[1], β[2]
    g1², g2² = γ²[1], γ²[2]
    return -(g1² * a2^3) / (3 * a1) +
            (g2² * a1^3) / (3 * a2) +
            (a1^2 - a2^2) * b1 * b2
end

"""
    h_rot_kernel_orthogonality_residual(α, β, γ², α̇, β̇)
        -> T

Numerically evaluate the kernel-orthogonality identity that `∂H_rot/∂θ_R`
is supposed to enforce. Specifically: when `α̇, β̇` solve the Berry-
modified per-axis Hamilton equations, the contraction

    R = -α_1²β_2·α̇_1 + (α_2³/3)·β̇_1 + α_2²β_1·α̇_2 - (α_1³/3)·β̇_2
        + ∂H_rot/∂θ_R

should evaluate to 0 (this is the θ_R row of `Ω · X = -dH`). This
helper computes `R` directly so the §6.4 verification test can
sample generic `(α, β, γ)` points, evolve via the boxed Berry-modified
equations to get `α̇, β̇`, and cross-check that the residual is at
machine precision.

Returns the residual as a scalar of type `T`.
"""
@inline function h_rot_kernel_orthogonality_residual(α::SVector{2, T},
                                                      β::SVector{2, T},
                                                      γ²::SVector{2, T},
                                                      α̇::SVector{2, T},
                                                      β̇::SVector{2, T}) where {T<:Real}
    a1, a2 = α[1], α[2]
    b1, b2 = β[1], β[2]
    R = -a1^2 * b2 * α̇[1] + (a2^3) * β̇[1] / 3 +
         a2^2 * b1 * α̇[2] - (a1^3) * β̇[2] / 3
    return R + h_rot_partial_dtheta(α, β, γ²)
end
