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

# ──────────────────────────────────────────────────────────────────────
# Phase M3-0: dimension-generic Cholesky-sector field type.
# ──────────────────────────────────────────────────────────────────────
#
# The original `ChField{T}` (above) is a scalar (α, β, γ) triple
# specific to the 1D specialisation of the 4×4 phase-space Cholesky
# factor. As of M3-0 we introduce a dimension-generic `ChField{D, T}`
# that, in 1D, reduces to a single (α, β) pair (γ remains derived from
# the EOS); in 2D it becomes the principal-axis pair
# `(α_a, β_a)_{a=1,2}` plus a Berry rotation angle `θ_R`; in 3D it is
# `(α_a, β_a)_{a=1,2,3}` plus three angles `θ_{ab}`. The 2D and 3D
# constructors land in M3-3.
#
# In Phase M3-0 the generic type is *only* used for downstream APIs
# that need to accept a dimension parameter; the M1 `cholesky_step`
# kernel still operates on `SVector{2}` for the bit-equality contract.
# Storage of (α, β) per cell goes through HG's `PolynomialFieldSet`
# (see `src/newton_step_HG.jl`), which is dimension-generic by
# construction, so `ChField{D, T}` exists primarily as a
# documentation / tag type for now.
#
# DO NOT delete the legacy `ChField{T}` — it is referenced by Phase-1
# tests and downstream Phase-2/5 internals through the
# `Segment{T,ChField{T}}` path. The two coexist until M3-2 verifies
# full M1+M2 parity on the HG-based code path.

"""
    ChFieldND{D, T<:Real}

Dimension-generic Cholesky-sector state in the 1D / 2D / 3D
specialisation of the phase-space Cholesky factor.

In `D = 1` this carries a single `(α, β)` pair (γ remains derived
from the EOS via `γ² = M_vv − β²`). In `D = 2` it carries the
principal-axis pairs `(α_a, β_a)_{a=1,2}` and a Berry rotation
angle `θ_R` (M3-3 scope). In `D = 3` it carries three principal-axis
pairs and three angles `θ_{ab}` (M3-7 scope).

Phase M3-0 use: the type is a thin wrapper exposing
`alphas::NTuple{D, T}` and `betas::NTuple{D, T}` so dimension-generic
APIs can dispatch on `D` without committing to a concrete storage
layout. Per-cell storage of the polynomial expansion of these
fields lives in `HierarchicalGrids.PolynomialFieldSet`; this type is
mainly documentation + a dispatch tag.

Naming. The type name is `ChFieldND` (rather than `ChField{D, T}`)
to avoid colliding with the legacy 1D `ChField` until the legacy
path is retired in M3-2.
"""
struct ChFieldND{D, T<:Real}
    alphas::NTuple{D, T}
    betas::NTuple{D, T}
end

"""
    ChFieldND(α::T, β::T) where {T<:Real}

1D convenience constructor: build a `ChFieldND{1, T}` from a scalar
`(α, β)` pair. Mirrors the legacy `ChField(α, β, γ)` constructor's
calling convention (γ is derived externally and not stored here —
the M3-0 EOS coupling supplies it from `M_vv` per cell).
"""
ChFieldND(α::T, β::T) where {T<:Real} =
    ChFieldND{1, T}((α,), (β,))

"""
    spatial_dimension(::ChFieldND{D, T})

Spatial dimension `D` of the Cholesky-sector field. Mirrors the
HG-side `spatial_dimension(::SimplicialMesh{D, T})`.
"""
@inline spatial_dimension(::ChFieldND{D, T}) where {D, T} = D

# ──────────────────────────────────────────────────────────────────────
# Phase M3-1: dimension-generic full deterministic field type.
# ──────────────────────────────────────────────────────────────────────
#
# The original M1 `DetField{T}` (above) carries the full deterministic
# state per segment in 1D: `(x, u, α, β, s, Pp, Q)`. As of M3-1 we
# introduce a dimension-generic `DetFieldND{D, T}` that, in 1D, reduces
# to the 7-scalar M1 layout; in 2D it carries `(x_a, u_a, α_a, β_a,
# s, Pp, Q)` per axis with `θ_R` for the Berry rotation (M3-3 scope);
# in 3D it carries `(x_a, u_a, α_a, β_a, s, Pp, Q)` per axis plus
# three `θ_{ab}` (M3-7 scope).
#
# In Phase M3-1 the type is a documentation / dispatch tag; storage of
# `(x, u, α, β, s, Pp, Q)` per HG cell goes through HG's
# `PolynomialFieldSet` (see `src/newton_step_HG.jl`), which is
# dimension-generic by construction.
#
# DO NOT delete the legacy `DetField{T}` — it is referenced by Phase-2
# tests and downstream Phase-5/5b internals through the
# `Mesh1D{T,DetField{T}}` path. The two coexist until M3-2 verifies
# full M1+M2 parity on the HG-based code path.

"""
    DetFieldND{D, T<:Real}

Dimension-generic full deterministic state in the 1D / 2D / 3D
specialisation of the Phase-2/5 system.

In `D = 1` this carries a single `(x, u, α, β, s, Pp, Q)` tuple
matching M1's `DetField{T}`. In `D = 2` it carries per-axis
`(x_a, u_a, α_a, β_a)_{a=1,2}` plus a Berry rotation angle `θ_R` and
the scalar entropy `s`, perpendicular pressure `Pp`, heat flux `Q`
(M3-3 scope). In `D = 3` it carries three principal-axis tuples
plus three angles `θ_{ab}` (M3-7 scope).

Phase M3-1 use: a thin wrapper exposing scalar fields for the 1D path.
Per-cell storage of the polynomial expansion lives in
`HierarchicalGrids.PolynomialFieldSet`; this type is mainly
documentation + a dispatch tag.

Naming. The type name is `DetFieldND` (rather than
`DetField{D, T}`) to avoid colliding with the legacy 1D
`DetField` until the legacy path is retired in M3-2.
"""
struct DetFieldND{D, T<:Real}
    x::NTuple{D, T}
    u::NTuple{D, T}
    alphas::NTuple{D, T}
    betas::NTuple{D, T}
    s::T
    Pp::T
    Q::T
end

"""
    DetFieldND(x::T, u::T, α::T, β::T, s::T, Pp::T, Q::T) where {T<:Real}

1D convenience constructor: build a `DetFieldND{1, T}` from scalar
`(x, u, α, β, s, Pp, Q)` values. Mirrors the M1 `DetField{T}`'s
7-arg constructor.
"""
DetFieldND(x::T, u::T, α::T, β::T, s::T, Pp::T, Q::T) where {T<:Real} =
    DetFieldND{1, T}((x,), (u,), (α,), (β,), s, Pp, Q)

"""
    spatial_dimension(::DetFieldND{D, T})

Spatial dimension `D` of the deterministic field. Mirrors the HG-side
`spatial_dimension(::SimplicialMesh{D, T})`.
"""
@inline spatial_dimension(::DetFieldND{D, T}) where {D, T} = D

# ──────────────────────────────────────────────────────────────────────
# Phase M3-3a: working 2D Cholesky-sector field type.
# ──────────────────────────────────────────────────────────────────────
#
# `DetFieldND{2, T}` above is a documentation / dispatch tag. M3-3a
# promotes the 2D variant to a *working* struct carrying the full
# 10-dof Cholesky-sector unknown set used by the M3-3b discrete
# Euler–Lagrange residual:
#
#     (x_1, x_2, u_1, u_2, α_1, α_2, β_1, β_2, θ_R, s)
#                                                    ╰── 10 dof per leaf cell
#
# The off-diagonal Cholesky entries `β_{12}`, `β_{21}` are intentionally
# **omitted** per Q3 of `reference/notes_M3_3_2d_cholesky_berry.md`
# §10: they are pinned to zero in M3-3 and re-added by M3-6 when the
# D.1 KH falsifier activates. With `β_{12} = β_{21} = 0` the
# diagonal-sector Berry kinetic 1-form
#
#     Θ_rot^(2D) = (1/3)(α_1³ β_2 − α_2³ β_1) · dθ_R
#
# is the only off-axis coupling the Newton solver sees, exactly matching
# `src/berry.jl::berry_partials_2d`.
#
# The post-Newton sectors `Pp` (deviatoric / Phase 5) and `Q` (heat
# flux / Phase 7) are operator-split, NOT Newton unknowns; they ride
# along on the field-set storage but are advanced by the M1 + M2
# closed-form BGK steps applied per axis. We carry them on
# `DetField2D` so the 2D field-set allocation (`src/setups_2d.jl`)
# matches the 12-named-field layout of the M3-3b residual.

"""
    DetField2D{T<:Real}

Per-cell M3-3 deterministic state for the 2D Cholesky-sector
variational scheme. Carries the 10 Newton unknowns

    `x = (x_1, x_2)` — Lagrangian position (2 components, charge 0).
    `u = (u_1, u_2)` — Lagrangian velocity (2 components, charge 0).
    `alphas = (α_1, α_2)` — principal-axis Cholesky factors
                              (`α_a > 0`, charge 0).
    `betas  = (β_1, β_2)` — per-axis conjugate momenta (charge 1
                              under the per-axis strain group).
    `θ_R::T` — principal-axis rotation angle (Newton unknown; gates
               the Berry kinetic coupling).
    `s::T`   — specific entropy (charge 0).

plus two post-Newton sectors that ride along on the storage:

    `Pp::T` — deviatoric pressure (Phase 5; per-cell scalar in 2D
              for now — D.1 KH will lift this to per-axis).
    `Q::T`  — heat flux scalar (Phase 7).

The off-diagonal Cholesky factors `β_{12}, β_{21}` are **omitted**
(pinned to zero in M3-3); M3-6 will re-add them under the D.1 KH
falsifier (see `reference/notes_M3_3_2d_cholesky_berry.md` §4.4).

# Storage layout

The per-cell tuple has total scalar count

    2 + 2 + 2 + 2 + 1 + 1 + 1 + 1 = 12 scalars,

of which the **first 10 are Newton unknowns** and the last 2 (`Pp`,
`Q`) are post-Newton operator-split state. The HG-substrate
`PolynomialFieldSet` layout in `src/setups_2d.jl` reflects this with
12 named scalar fields (each at `MonomialBasis{2, 0}` for now;
M3-4 / M3-5 will move to higher-order Bernstein per methods paper
§9.2).

# Naming

`DetField2D` (rather than refactoring `DetFieldND` to carry `θ_R`)
keeps the 1D `DetFieldND{1, T}` doc-tag intact and avoids breaking
the M3-0/1/2 1D path. The two coexist; M3-7 will promote the 3D
variant similarly to `DetField3D` carrying three rotation angles.
"""
struct DetField2D{T<:Real}
    x::NTuple{2, T}
    u::NTuple{2, T}
    alphas::NTuple{2, T}
    betas::NTuple{2, T}
    θ_R::T
    s::T
    Pp::T
    Q::T
end

"""
    DetField2D(x, u, alphas, betas, θ_R, s, Pp, Q)

Convenience constructor with element-type promotion.
"""
function DetField2D(x::NTuple{2}, u::NTuple{2},
                    alphas::NTuple{2}, betas::NTuple{2},
                    θ_R, s, Pp, Q)
    T = promote_type(eltype(x), eltype(u), eltype(alphas), eltype(betas),
                     typeof(θ_R), typeof(s), typeof(Pp), typeof(Q))
    return DetField2D{T}(
        NTuple{2, T}(x), NTuple{2, T}(u),
        NTuple{2, T}(alphas), NTuple{2, T}(betas),
        T(θ_R), T(s), T(Pp), T(Q),
    )
end

"""
    DetField2D(x, u, alphas, betas, θ_R, s)

Phase M3-3a compatibility constructor: defaults `Pp = 0`, `Q = 0`
(Maxwellian post-Newton sectors). M3-3b's pure-Newton tests use this
form.
"""
function DetField2D(x::NTuple{2}, u::NTuple{2},
                    alphas::NTuple{2}, betas::NTuple{2},
                    θ_R, s)
    T = promote_type(eltype(x), eltype(u), eltype(alphas), eltype(betas),
                     typeof(θ_R), typeof(s))
    return DetField2D{T}(
        NTuple{2, T}(x), NTuple{2, T}(u),
        NTuple{2, T}(alphas), NTuple{2, T}(betas),
        T(θ_R), T(s), zero(T), zero(T),
    )
end

"""
    spatial_dimension(::DetField2D{T}) -> 2

Spatial dimension of the 2D Cholesky-sector deterministic field.
"""
@inline spatial_dimension(::DetField2D{T}) where {T} = 2

"""
    n_dof_newton(::DetField2D) -> 10

Number of per-cell Newton unknowns in the M3-3 2D Cholesky-sector
EL system: `(x_1, x_2, u_1, u_2, α_1, α_2, β_1, β_2, θ_R, s)`. The
post-Newton sectors `Pp`, `Q` are NOT counted.
"""
@inline n_dof_newton(::DetField2D) = 10
@inline n_dof_newton(::Type{<:DetField2D}) = 10

