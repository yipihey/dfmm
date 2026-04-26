# berry.jl
#
# Berry-connection stencil pre-compute for the Cholesky-sector
# variational scheme (M3-prep, building blocks for M3-3).
#
# This module is a pure-functional collection of the per-cell Berry
# 1-form coefficients that arise when the Cholesky factor is rotated
# into its principal-axis frame in 2D / 3D. M3-3 (the upcoming
# Cholesky 2D phase) consumes these stencils inside its discrete
# Euler–Lagrange residual; this file does NOT yet wire them into the
# solver. It only provides:
#
#   • the evaluators `berry_term_2d`, `berry_term_3d`,
#   • the partial derivatives w.r.t. (α_a, β_a, θ_{ab}) used in the
#     EL residual,
#   • the off-diagonal-L₂ kinetic 1-form `kinetic_offdiag_2d`,
#   • thin pre-compute carriers `BerryStencil2D`, `BerryStencil3D`.
#
# The verified symbolic forms come from
#   `reference/notes_M3_phase0_berry_connection.md`           (2D)
#   `reference/notes_M3_phase0_berry_connection_3D.md`        (3D)
#   `reference/notes_M3_phase0_berry_connection.md` §7        (off-diag)
# and the SymPy verification scripts under `scripts/`.
#
# 2D principal-axis Berry 1-form (§4 of the 2D notes):
#
#     Θ_rot^(2D) = (1/3)(α_1³ β_2 − α_2³ β_1) · dθ_R
#
# 3D SO(3) extension (§8 of the 2D notes; full check in 3D notes):
#
#     Θ_rot^(3D) = (1/3) Σ_{a<b} (α_a³ β_b − α_b³ β_a) · dθ_{ab}
#
# Off-diagonal-L₂ kinetic 1-form (§7 of the 2D notes, corrected):
#
#     θ_offdiag = −(1/2)(α_1² α_2 dβ_{21} + α_1 α_2² dβ_{12})
#
# All routines are allocation-free given `StaticArrays` inputs.

using StaticArrays

# ──────────────────────────────────────────────────────────────────────
# 2D Berry-connection coefficient
# ──────────────────────────────────────────────────────────────────────

"""
    berry_F_2d(α::SVector{2,T}, β::SVector{2,T}) where {T<:Real}

Return the Berry function

    F^{(2D)}(α, β) = (1/3)(α_1³ β_2 − α_2³ β_1)

so that `Θ_rot^(2D) = F · dθ_R`. Unit-tested against
`scripts/verify_berry_connection.py` CHECK 5.
"""
@inline function berry_F_2d(α::SVector{2,T}, β::SVector{2,T}) where {T<:Real}
    return (α[1]^3 * β[2] - α[2]^3 * β[1]) / 3
end

"""
    berry_term_2d(α::SVector{2,T}, β::SVector{2,T}, dθ_R::Real)

Evaluate the 2D Berry 1-form on the velocity `dθ_R`:

    Θ_rot^(2D)(α, β) · dθ_R = (1/3)(α_1³ β_2 − α_2³ β_1) · dθ_R

This is the contribution to the per-cell discrete Lagrangian when the
principal-axis frame rotates by an increment `dθ_R` over a step.
"""
@inline function berry_term_2d(α::SVector{2,T},
                               β::SVector{2,T},
                               dθ_R::Real) where {T<:Real}
    return berry_F_2d(α, β) * dθ_R
end

"""
    berry_partials_2d(α::SVector{2,T}, β::SVector{2,T}, dθ_R::Real)

Return the partial derivatives of the 2D Berry contribution
`Θ_rot^(2D) = F · dθ_R` w.r.t. `(α_1, α_2, β_1, β_2, θ_R)` as an
`SVector{5,T}` in that order.

Closed-form (verified via the SymPy script):

    ∂Θ/∂α_1 =  α_1² β_2 · dθ_R
    ∂Θ/∂α_2 = -α_2² β_1 · dθ_R
    ∂Θ/∂β_1 = -(1/3) α_2³ · dθ_R
    ∂Θ/∂β_2 =  (1/3) α_1³ · dθ_R
    ∂Θ/∂θ_R =  F  (the coefficient of dθ_R itself)
"""
@inline function berry_partials_2d(α::SVector{2,T},
                                   β::SVector{2,T},
                                   dθ_R::Real) where {T<:Real}
    a1, a2 = α[1], α[2]
    b1, b2 = β[1], β[2]
    dθ = T(dθ_R)
    dF_dα1 =  a1^2 * b2 * dθ
    dF_dα2 = -a2^2 * b1 * dθ
    dF_dβ1 = -(a2^3) * dθ / 3
    dF_dβ2 =  (a1^3) * dθ / 3
    dF_dθ  = (a1^3 * b2 - a2^3 * b1) / 3
    return SVector{5,T}(dF_dα1, dF_dα2, dF_dβ1, dF_dβ2, dF_dθ)
end

# ──────────────────────────────────────────────────────────────────────
# 3D SO(3) Berry-connection coefficient
# ──────────────────────────────────────────────────────────────────────

"""
    berry_F_3d(α::SVector{3,T}, β::SVector{3,T})

Return the three Berry pair-functions `F_{ab}` for `(a,b) ∈ {(1,2),(1,3),(2,3)}`:

    F_{ab} = (1/3)(α_a³ β_b − α_b³ β_a)

as an `SVector{3,T}` in the order `(F_12, F_13, F_23)`.
"""
@inline function berry_F_3d(α::SVector{3,T}, β::SVector{3,T}) where {T<:Real}
    a1, a2, a3 = α[1], α[2], α[3]
    b1, b2, b3 = β[1], β[2], β[3]
    F12 = (a1^3 * b2 - a2^3 * b1) / 3
    F13 = (a1^3 * b3 - a3^3 * b1) / 3
    F23 = (a2^3 * b3 - a3^3 * b2) / 3
    return SVector{3,T}(F12, F13, F23)
end

"""
    berry_term_3d(α::SVector{3,T}, β::SVector{3,T}, dθ::SMatrix{3,3,T,9})

Evaluate the 3D SO(3) Berry 1-form on the antisymmetric Euler-angle
increment matrix `dθ` (only `dθ[a,b]` for `a<b` is read):

    Θ_rot^(3D) = Σ_{a<b} F_{ab} · dθ_{ab},
    F_{ab} = (1/3)(α_a³ β_b − α_b³ β_a).

`dθ` is taken as the antisymmetric "axis-pair" matrix with
`dθ[1,2] = dθ_12`, `dθ[1,3] = dθ_13`, `dθ[2,3] = dθ_23`. The lower
triangle is ignored (callers should supply an antisymmetric matrix
for clarity, but only the strictly upper triangle is used).
"""
@inline function berry_term_3d(α::SVector{3,T},
                               β::SVector{3,T},
                               dθ::SMatrix{3,3,T,9}) where {T<:Real}
    F = berry_F_3d(α, β)
    return F[1] * dθ[1, 2] + F[2] * dθ[1, 3] + F[3] * dθ[2, 3]
end

"""
    berry_partials_3d(α::SVector{3,T}, β::SVector{3,T}, dθ::SMatrix{3,3,T,9})

Per-axis partials of `Θ_rot^(3D) = Σ_{a<b} F_{ab} · dθ_{ab}`
w.r.t. `(α_1, α_2, α_3, β_1, β_2, β_3)` (returned as an
`SVector{6,T}` in that order), and the `dθ_{ab}`-partials
(returned as the three `F_{ab}` themselves, since `Θ` is linear
in each `dθ_{ab}` with coefficient `F_{ab}`).

Returns `(grad_αβ::SVector{6,T}, F::SVector{3,T})`.

Closed forms:

    ∂Θ/∂α_a = Σ_{b≠a} sign_{ab} · α_a² β_b · dθ_{ab*}
    ∂Θ/∂β_a = Σ_{b≠a} sign_{ab} · (-α_b³/3) · dθ_{ab*}

where `sign_{ab} = +1` if `a<b` (the term enters as `α_a³ β_b`) and
`-1` if `a>b` (the term enters as `−α_a³ β_b` from `F_{ba}`). See
the SymPy script `verify_berry_connection_3D.py` (`Omega[i, theta]`
entries).
"""
@inline function berry_partials_3d(α::SVector{3,T},
                                   β::SVector{3,T},
                                   dθ::SMatrix{3,3,T,9}) where {T<:Real}
    a1, a2, a3 = α[1], α[2], α[3]
    b1, b2, b3 = β[1], β[2], β[3]
    # dθ_{ab} for a<b
    d12 = dθ[1, 2]
    d13 = dθ[1, 3]
    d23 = dθ[2, 3]

    # ∂F_12/∂α_1 = a1² b2 ;   ∂F_13/∂α_1 = a1² b3
    # ∂F_12/∂α_2 = -a2² b1;  ∂F_23/∂α_2 = a2² b3
    # ∂F_13/∂α_3 = -a3² b1;  ∂F_23/∂α_3 = -a3² b2
    dα1 =  a1^2 * b2 * d12 +  a1^2 * b3 * d13
    dα2 = -a2^2 * b1 * d12 +  a2^2 * b3 * d23
    dα3 = -a3^2 * b1 * d13 + -a3^2 * b2 * d23

    # ∂F_12/∂β_1 = -a2³/3;  ∂F_13/∂β_1 = -a3³/3
    # ∂F_12/∂β_2 =  a1³/3;  ∂F_23/∂β_2 = -a3³/3
    # ∂F_13/∂β_3 =  a1³/3;  ∂F_23/∂β_3 =  a2³/3
    dβ1 = -(a2^3) * d12 / 3 + -(a3^3) * d13 / 3
    dβ2 =  (a1^3) * d12 / 3 + -(a3^3) * d23 / 3
    dβ3 =  (a1^3) * d13 / 3 +  (a2^3) * d23 / 3

    grad_αβ = SVector{6,T}(dα1, dα2, dα3, dβ1, dβ2, dβ3)
    F = berry_F_3d(α, β)
    return (grad_αβ, F)
end

# ──────────────────────────────────────────────────────────────────────
# Off-diagonal-L₂ kinetic 1-form
# ──────────────────────────────────────────────────────────────────────

"""
    kinetic_offdiag_coeffs_2d(α::SVector{2,T})

Return the two coefficients of the off-diagonal kinetic 1-form
`θ_offdiag = −(1/2)(α_1² α_2 · dβ_{21} + α_1 α_2² · dβ_{12})`
as an `SVector{2,T} = (c_β12, c_β21)`:

    c_β12 = ∂θ_offdiag / ∂(dβ_{12}) = -(1/2) α_1 α_2²
    c_β21 = ∂θ_offdiag / ∂(dβ_{21}) = -(1/2) α_1² α_2

Verified via `scripts/verify_berry_connection_offdiag.py` CHECK 2
(the symplectic-block entries `Ω[α_1, β_{12}]`, `Ω[α_2, β_{12}]`,
`Ω[α_1, β_{21}]`, `Ω[α_2, β_{21}]` come from differentiating
these coefficients).
"""
@inline function kinetic_offdiag_coeffs_2d(α::SVector{2,T}) where {T<:Real}
    a1, a2 = α[1], α[2]
    c_β12 = -(a1 * a2^2) / 2
    c_β21 = -(a1^2 * a2) / 2
    return SVector{2,T}(c_β12, c_β21)
end

"""
    kinetic_offdiag_2d(α::SVector{2,T}, β::SMatrix{2,2,T,4})

Evaluate the off-diagonal kinetic 1-form on the off-diagonal-velocity
matrix `β` whose entries are interpreted as the *increments*
`(dβ_{ij})`:

    θ_offdiag(α) · β = -(1/2)(α_1² α_2 · β[2,1] + α_1 α_2² · β[1,2]).

This is the symmetric-projection bilinear form used in the discrete
action (§7 of the 2D notes, corrected): under the symmetric extension
`β_sym = (β + βᵀ)/2`, the on-diagonal entries `β[1,1]`, `β[2,2]` are
ignored here (they live in the diagonal kinetic 1-form
`-(α_1³/3) dβ_1 − (α_2³/3) dβ_2`).

Verified against `scripts/verify_berry_connection_offdiag.py`.
"""
@inline function kinetic_offdiag_2d(α::SVector{2,T},
                                    β::SMatrix{2,2,T,4}) where {T<:Real}
    c = kinetic_offdiag_coeffs_2d(α)
    # c[1] = coeff of β[1,2], c[2] = coeff of β[2,1]
    return c[1] * β[1, 2] + c[2] * β[2, 1]
end

# ──────────────────────────────────────────────────────────────────────
# Pre-compute carriers (per-cell stencils for M3-3)
# ──────────────────────────────────────────────────────────────────────

"""
    BerryStencil2D{T}

Per-cell pre-computed 2D Berry-form coefficients. The fields
`(F, dF_dα, dF_dβ)` are the Berry function and its α/β-gradients at
the cell's `(α, β)`; the `dθ_R` factor multiplies them at use-time
because it is a per-step quantity (it depends on the rotation
between two consecutive states).

Construct via `BerryStencil2D(α, β)`. Allocation-free given `SVector`
inputs.

Fields:
- `F::T`         — Berry function `(α_1³ β_2 − α_2³ β_1)/3`.
- `dF_dα::SVector{2,T}` — `(α_1² β_2, -α_2² β_1)`.
- `dF_dβ::SVector{2,T}` — `(-α_2³/3, α_1³/3)`.

The full Berry term and its partials at a given `dθ_R` are:

    Θ_rot     = F * dθ_R
    ∂Θ/∂α_a   = dF_dα[a] * dθ_R
    ∂Θ/∂β_a   = dF_dβ[a] * dθ_R
    ∂Θ/∂θ_R   = F
"""
struct BerryStencil2D{T<:Real}
    F::T
    dF_dα::SVector{2,T}
    dF_dβ::SVector{2,T}
end

@inline function BerryStencil2D(α::SVector{2,T},
                                β::SVector{2,T}) where {T<:Real}
    a1, a2 = α[1], α[2]
    b1, b2 = β[1], β[2]
    F = (a1^3 * b2 - a2^3 * b1) / 3
    dF_dα = SVector{2,T}( a1^2 * b2, -a2^2 * b1)
    dF_dβ = SVector{2,T}(-(a2^3) / 3, (a1^3) / 3)
    return BerryStencil2D{T}(F, dF_dα, dF_dβ)
end

"""
    apply(stencil::BerryStencil2D, dθ_R)

Evaluate `Θ_rot^(2D) = F · dθ_R` from a pre-computed stencil.
"""
@inline apply(stencil::BerryStencil2D, dθ_R::Real) = stencil.F * dθ_R

"""
    BerryStencil3D{T}

Per-cell pre-computed 3D SO(3) Berry-form coefficients. The fields
`(F, dF_dα, dF_dβ)` carry the three pair-functions `F_{ab}` and
their α/β-gradients at the cell's `(α, β)`; the per-step
`dθ_{ab}` factor multiplies at use-time.

Fields:
- `F::SVector{3,T}` — `(F_12, F_13, F_23)`.
- `dF_dα::SMatrix{3,3,T,9}` — `dF_dα[i,k] = ∂F_{pair_i}/∂α_k`,
  with pair index `i ∈ {1=12, 2=13, 3=23}` and axis `k ∈ {1,2,3}`.
- `dF_dβ::SMatrix{3,3,T,9}` — same shape, w.r.t. β.

Construction: `BerryStencil3D(α, β)`. Allocation-free.
"""
struct BerryStencil3D{T<:Real}
    F::SVector{3,T}
    dF_dα::SMatrix{3,3,T,9}
    dF_dβ::SMatrix{3,3,T,9}
end

@inline function BerryStencil3D(α::SVector{3,T},
                                β::SVector{3,T}) where {T<:Real}
    a1, a2, a3 = α[1], α[2], α[3]
    b1, b2, b3 = β[1], β[2], β[3]
    F12 = (a1^3 * b2 - a2^3 * b1) / 3
    F13 = (a1^3 * b3 - a3^3 * b1) / 3
    F23 = (a2^3 * b3 - a3^3 * b2) / 3

    z = zero(T)
    # dF_dα[pair, axis]: row=pair (1=12, 2=13, 3=23), col=axis (1,2,3).
    # SMatrix constructor uses column-major argument order.
    dF_dα = SMatrix{3,3,T,9}(
        # column axis=1: (∂F_12, ∂F_13, ∂F_23)/∂α_1
        a1^2 * b2,   a1^2 * b3,   z,
        # column axis=2: (∂F_12, ∂F_13, ∂F_23)/∂α_2
        -a2^2 * b1,  z,           a2^2 * b3,
        # column axis=3: (∂F_12, ∂F_13, ∂F_23)/∂α_3
        z,          -a3^2 * b1,  -a3^2 * b2,
    )
    dF_dβ = SMatrix{3,3,T,9}(
        # column axis=1: (∂F_12, ∂F_13, ∂F_23)/∂β_1
        -(a2^3) / 3, -(a3^3) / 3, z,
        # column axis=2: (∂F_12, ∂F_13, ∂F_23)/∂β_2
         (a1^3) / 3,  z,          -(a3^3) / 3,
        # column axis=3: (∂F_12, ∂F_13, ∂F_23)/∂β_3
        z,           (a1^3) / 3,  (a2^3) / 3,
    )
    F = SVector{3,T}(F12, F13, F23)
    return BerryStencil3D{T}(F, dF_dα, dF_dβ)
end

"""
    apply(stencil::BerryStencil3D, dθ::SMatrix{3,3,T,9})

Evaluate `Θ_rot^(3D) = Σ_{a<b} F_{ab} · dθ_{ab}` from a pre-computed
stencil. Only the strict-upper triangle of `dθ` is read.
"""
@inline function apply(stencil::BerryStencil3D{T},
                       dθ::SMatrix{3,3,T,9}) where {T<:Real}
    return stencil.F[1] * dθ[1, 2] + stencil.F[2] * dθ[1, 3] +
           stencil.F[3] * dθ[2, 3]
end
