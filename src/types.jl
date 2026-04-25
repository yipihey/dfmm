# types.jl
#
# Field types for the dfmm-2d Julia package, Milestone 1 / Phase 1 scope.
#
# The 1D specialization of the 4×4 phase-space Cholesky factor reduces
# to scalar (α, β, γ) per Lagrangian cell (v2 §2.1; methods paper §2.4).
# In Phase 1 only the Cholesky-sector dynamics are exercised: α and β
# evolve under the boxed Hamilton equations
#
#     D_t^{(0)} α = β,   D_t^{(1)} β = γ²/α,
#
# and γ is supplied externally via the EOS-implied second moment M_vv
# (constant initial J, s in Phase 1, so M_vv is a fixed scalar).

"""
    ChField{T<:Real}

Per-cell Cholesky-sector state in the 1D specialization of the 4×4
phase-space Cholesky factor (v2 §2.1).

Fields:
- `α::T` — Cholesky 11-component (charge 0 under the strain group).
- `β::T` — Cholesky 21-component (charge 1).
- `γ::T` — Cholesky 22-component (charge 1); reconstructed from the
  EOS-implied `M_vv` via `γ² = M_vv − β²`. Stored explicitly for
  diagnostics but not an independent dynamical variable.

Realizability: `α > 0`, `γ² = M_vv − β² ≥ 0`. In Phase 1 the initial
conditions keep these well-clear of the boundary; cold-limit handling
is Phase 3.
"""
struct ChField{T<:Real}
    α::T
    β::T
    γ::T
end

"""
    ChField(α, β, γ)

Construct a `ChField` with promoted element type. Convenience constructor.
"""
ChField(α, β, γ) = ChField{promote_type(typeof(α), typeof(β), typeof(γ))}(
    promote(α, β, γ)...,
)

"""
    Mvv(field::ChField)

EOS-implied second velocity moment `M_vv = β² + γ²` for the cell.
In Phase 1, `M_vv` is supplied externally as a constant; this helper
exists for round-tripping a `ChField` back to `M_vv` for diagnostics.
"""
Mvv(field::ChField) = field.β^2 + field.γ^2

"""
    gamma_from_Mvv(β, M_vv)

Return `γ = √max(M_vv − β², 0)`. Phase 1 callers should never hit
the floor (Hessian-degeneracy regularization is Phase 3); the `max`
exists only as a guard against round-off in tests very near
`γ → 0`.
"""
function gamma_from_Mvv(β::Real, M_vv::Real)
    g2 = M_vv - β^2
    return sqrt(max(g2, zero(g2)))
end
