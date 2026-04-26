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

# ──────────────────────────────────────────────────────────────────────
# Phase 2: full deterministic per-segment / per-vertex state
# ──────────────────────────────────────────────────────────────────────
#
# Phase 2 lifts Phase 1 from a single autonomous (α, β) cell to a
# multi-segment Lagrangian mesh with bulk position x, velocity u, and
# specific entropy s. The deterministic Lagrangian density (v2 §3.2,
# methods paper §3.2 / §9.4) is
#
#     L_det = ½ ẋ² + L_Ch(α, β, β̇; γ),
#
# with γ derived from the EOS via γ² = M_vv(J, s) − β². Phase 2 does
# *not* yet include the deviatoric stress sector L_dev (Phase 5) or
# the heat-flux Lagrange multiplier λ_Q (Phase 7).
#
# Variable layout (see `reference/notes_phase2_discretization.md`):
#   - vertex variables:  x_i (positions), u_i (velocities) — charge 0
#   - cell-centered:     α_j, β_j, s_j — charges 0, 1, 0 respectively
#
# `DetField` below is a *segment-centered* slice of the dynamical
# state; `x` and `u` here represent the *left vertex* of the segment,
# packaged with its (α, β, s) for convenient mesh-bookkeeping. The
# `Mesh1D` constructor hides this packing detail so callers index by
# segment, not by vertex.

"""
    DetField{T<:Real}

Per-segment slice of the full deterministic state for the Phase-2
multi-segment integrator. Bundles the segment's left-vertex `x` and
`u` (charge 0) with the cell-centered Cholesky-sector `(α, β)`,
specific entropy `s`, and (Phase 5) the perpendicular pressure
`Pp = P_⊥`. γ is *derived*, not stored —
`γ = sqrt(max(M_vv(J, s) − β², 0))`.

Fields:
- `x::T` — position of the segment's left vertex (charge 0). For a
  periodic mesh, the right vertex of segment `j` is `x[j+1]`, with
  `x[N+1] ≡ x[1] + L_box`.
- `u::T` — velocity at the left vertex (charge 0).
- `α::T` — Cholesky 11-component, cell-centered (charge 0).
- `β::T` — Cholesky 21-component, cell-centered (charge 1).
- `s::T` — specific entropy in `c_v` units, cell-centered (charge 0).
- `Pp::T` — perpendicular pressure `P_⊥`, cell-centered. Carries
  charge 1 in the mass-density sense (`P_⊥/ρ` is conserved along
  Lagrangian trajectories during the hyperbolic step). Phase 5
  introduces it; Phase 1/2 callers that don't need it use the
  legacy 5-arg `DetField(x, u, α, β, s)` constructor below, which
  defaults `Pp = ρ · M_vv(J, s)` (the isotropic-Maxwellian initial
  condition that makes the Phase-1/2 tests insensitive to its
  presence).
- `Q::T` — heat flux (third central velocity moment, `Q = M_3 −
  ρ u^3 − 3 u P_xx`), cell-centered. Phase 7 introduces this as a
  Lagrangian post-Newton update mirroring the β / P_⊥ sectors:
  the variational integrator advances `(x, u, α, β)` implicitly,
  and after the Newton step a closed-form exponential BGK relaxes
  `Q → 0`. Phase 1/2/5 callers leave `Q = 0` (Maxwellian IC); the
  6-arg `DetField(x, u, α, β, s, Pp)` constructor defaults `Q = 0`
  and the integrator skips Q-related work when `Q = 0` everywhere.

Phase 1 compatibility: setting `x = m * J_0`, `u = 0`, and reading
`(α, β)` reproduces the Phase-1 single-cell state on a uniform mesh.
"""
struct DetField{T<:Real}
    x::T
    u::T
    α::T
    β::T
    s::T
    Pp::T
    Q::T
end

"""
    DetField(x, u, α, β, s, Pp, Q)

Convenience 7-arg constructor with element-type promotion. Phase 7
introduces the explicit `Q` (heat-flux) field; pass `Q = 0` for
Maxwellian initial conditions.
"""
DetField(x, u, α, β, s, Pp, Q) = DetField{promote_type(typeof(x), typeof(u),
                                                       typeof(α), typeof(β),
                                                       typeof(s), typeof(Pp),
                                                       typeof(Q))}(
    promote(x, u, α, β, s, Pp, Q)...,
)

"""
    DetField(x, u, α, β, s, Pp)

Phase-5 compatibility constructor. Defaults `Q = 0` (Maxwellian IC,
the value used by every Tier-A `setup_*` factory at t = 0). Phase 7+
callers that need explicit `Q` should use the 7-arg constructor.
"""
DetField(x, u, α, β, s, Pp) = DetField{promote_type(typeof(x), typeof(u),
                                                    typeof(α), typeof(β),
                                                    typeof(s), typeof(Pp))}(
    promote(x, u, α, β, s, Pp, zero(promote_type(typeof(x), typeof(u),
                                                  typeof(α), typeof(β),
                                                  typeof(s), typeof(Pp))))...,
)

"""
    DetField(x, u, α, β, s)

Phase-1/2 compatibility constructor. Leaves `Pp = NaN` as a
sentinel: callers that do not pass `Pp` are running pre-Phase-5
tests where `P_⊥` is not part of the state, and the integrator
will not read it. For Phase 5+ tests, pass `Pp` explicitly.
Defaults `Q = 0` for the same reason — pre-Phase-7 paths don't
read it.

Implementation note: `NaN` would propagate into any computation
that reads `Pp`, so this 5-arg constructor is *only* for legacy
Phase-1/2 paths. The Phase-5 driver
`Mesh1D(...; Pps = ...)` requires explicit `Pp` per segment.
"""
DetField(x, u, α, β, s) = DetField{promote_type(typeof(x), typeof(u),
                                                typeof(α), typeof(β),
                                                typeof(s))}(
    promote(x, u, α, β, s, NaN,
            zero(promote_type(typeof(x), typeof(u), typeof(α),
                              typeof(β), typeof(s))))...,
)
