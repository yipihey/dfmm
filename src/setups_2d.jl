"""
    setups_2d (dimension-generic Tier-C IC factories)

Tier-C initial-condition factories used by Milestone M3-4 (2D 1D-symmetric
consistency tests). They build a `PolynomialFieldSet` of cell-averages
on top of an HG `HierarchicalMesh{D}` + `EulerianFrame{D}`, so the same
factory works for `D = 1, 2, 3`. Because the body of each factory only
references `HierarchicalMesh{D}` / `EulerianFrame{D}` parametrically, no
explicit dimension-dispatch is needed; the dimension comes through the
mesh.

Tier-C tests covered (methods paper §10.4):

- C.1 1D-symmetric 2D Sod (`tier_c_sod_ic`) — Standard Sod shock tube
  but in 2D with a discontinuity at `x = x_split` along `shock_axis`.
  In 1D this collapses to the classical Sod IC.

- C.2 2D cold sinusoid (`tier_c_cold_sinusoid_ic`) — Velocity field
  `u_d(x_d) = A sin(2π k_d x_d)` (one component per axis), uniform
  density and pressure. The per-axis γ diagnostic on the resulting
  field set should identify any axis with `k_d = 0` as collapsing
  (`γ_d → small`); the other axes stay `O(1)`.

- C.3 Plane wave at angle θ (`tier_c_plane_wave_ic`) — Acoustic plane
  wave `δρ ∝ cos(2π k · x)` with `k = k_mag * (cos θ, sin θ)` (and a
  `k_z` extension for D=3); δu parallel to k with the linearised-EOS
  amplitude `δu = (c / ρ₀) δρ` so the IC sits on the `+`-going branch.
  At θ = 0 this bit-equals the corresponding 1D plane wave; under
  rotation by π/2 it produces the y-aligned variant.

# Storage layout

Each IC returns a `(mesh, frame, fields, params)` NamedTuple where:

- `mesh::HierarchicalMesh{D}` — the Eulerian quadtree/octree from
  `uniform_eulerian_mesh` (a leaf at every cell of an isotropic 2^level
  refinement). `enumerate_leaves(mesh)` is the canonical iteration
  order; the `PolynomialFieldSet` is allocated 1:1 against that order.
- `frame::EulerianFrame{D}` — physical bounding box of the root cell.
- `fields::PolynomialFieldSet` — order-0 cell-average storage with
  named scalar fields `(rho, ux, uy, uz, P)` (the velocity components
  scale with `D`; missing components are simply not present in the
  field set so the names never collide with M1's 1D `(x, u, α, β, ...)`
  layout).
- `params::NamedTuple` — the input parameters echoed back for
  bookkeeping (so downstream tests can re-derive analytical values).

The cell-average is computed via Gauss quadrature on the reference
domain (HG `gauss_quadrature_*`) of the appropriate dimension and an
affine map to the cell's physical AABB. With `quad_order = 3` the
factories integrate cubic polynomials exactly; the smoothness of the
ICs makes higher orders unnecessary in practice but the kwarg is
exposed for the convergence-order tests.

# HG API gap (per `reference/notes_HG_design_guidance.md` item #9)

There is currently no `HG.init_field_from!(field, mesh, f::Function)`
helper that handles the quadrature + projection in a single call. We
hand-roll the projection inline. When HG ships item #9 the helper
`_cell_average_*` blocks below should reduce to a single call.

# References

- methods paper §10.4 (Tier C list)
- `reference/MILESTONE_3_PLAN.md` Phase M3-4
- `reference/notes_HG_design_guidance.md` items #9
"""

using HierarchicalGrids: HierarchicalMesh, EulerianFrame,
    refine_cells!, enumerate_leaves, cell_physical_box,
    gauss_quadrature_interval, gauss_quadrature_quad, gauss_quadrature_cube,
    MonomialBasis, allocate_polynomial_fields, SoA, PolynomialFieldSet,
    n_quad_points

# ─────────────────────────────────────────────────────────────────────
# Mesh + quadrature plumbing (dimension-generic)
# ─────────────────────────────────────────────────────────────────────

"""
    uniform_eulerian_mesh(D::Integer, level::Integer; T::Type = Float64)

Build a `HierarchicalMesh{D}` whose leaves are the 2^(D*level) cells of
an isotropically refined unit grid. The mesh is initialized as a single
root cell and refined `level` times: at each round every leaf is split
into `2^D` children (the default fully-isotropic split mask). Returns
the mesh; pair it with an `EulerianFrame` for physical-coordinate
queries.

For `D = 2, level = 4` this yields a 16×16 grid of leaves (256 cells);
the default `level = 4` is the canonical Tier-C smoke-test resolution.
"""
function uniform_eulerian_mesh(D::Integer, level::Integer)
    D >= 1 || throw(ArgumentError("D must be ≥ 1"))
    level >= 0 || throw(ArgumentError("level must be ≥ 0"))
    mesh = HierarchicalMesh{Int(D)}()
    for _ in 1:level
        leaves = enumerate_leaves(mesh)
        refine_cells!(mesh, leaves)
    end
    return mesh
end

"""
    _quad_for_dim(D::Integer, order::Integer; T::Type = Float64)

Pick the right HG quadrature rule for an axis-aligned `D`-cube on the
reference domain `[0, 1]^D`. `order` is the number of points per axis
(tensor-product). Returns a `QuadRule{D, T}`.
"""
function _quad_for_dim(D::Integer, order::Integer; T::Type = Float64)
    if D == 1
        return gauss_quadrature_interval(Int(order); T = T)
    elseif D == 2
        return gauss_quadrature_quad(Int(order); T = T)
    elseif D == 3
        return gauss_quadrature_cube(Int(order); T = T)
    else
        throw(ArgumentError("dimension D=$D not supported by Tier-C IC factories"))
    end
end

"""
    _cell_average(f, lo::NTuple{D,T}, hi::NTuple{D,T}, quad)

Evaluate the cell average `(1/V) ∫_cell f(x) dx` for the axis-aligned
cell `[lo, hi]` ⊂ ℝ^D, using the HG reference-domain quadrature `quad`
and an affine pull-back. `f` must accept an `NTuple{D,T}` of physical
coordinates and return a scalar. Returns the average as a `Float64`.

The cell volume cancels between the Jacobian (= prod(extents)) and
the average's `1/V`, so the result is just the weighted sum of `f` at
the mapped quadrature points.
"""
@inline function _cell_average(f, lo::NTuple{D,T}, hi::NTuple{D,T},
                                quad) where {D, T}
    extents = ntuple(d -> hi[d] - lo[d], Val(D))
    s = zero(Float64)
    @inbounds for k in 1:n_quad_points(quad)
        ξ = quad.points[k]
        x = ntuple(d -> lo[d] + extents[d] * ξ[d], Val(D))
        s += quad.weights[k] * Float64(f(x))
    end
    return s
end

"""
    _allocate_tierc_fields(D::Integer, n_leaves::Integer; T::Type = Float64)

Allocate an order-0 `PolynomialFieldSet` over `n_leaves` cells with the
Tier-C field names: `:rho`, `:P`, plus a velocity component per
spatial axis (`:ux` for D≥1, `:uy` for D≥2, `:uz` for D=3). The
layout is HG `SoA` with `MonomialBasis{D, 0}` (one coefficient per
cell — the cell average).
"""
function _allocate_tierc_fields(D::Integer, n_leaves::Integer;
                                 T::Type = Float64)
    basis = MonomialBasis{Int(D), 0}()
    if D == 1
        return allocate_polynomial_fields(SoA(), basis, Int(n_leaves);
                                          rho = T, ux = T, P = T)
    elseif D == 2
        return allocate_polynomial_fields(SoA(), basis, Int(n_leaves);
                                          rho = T, ux = T, uy = T, P = T)
    elseif D == 3
        return allocate_polynomial_fields(SoA(), basis, Int(n_leaves);
                                          rho = T, ux = T, uy = T,
                                          uz = T, P = T)
    else
        throw(ArgumentError("dimension D=$D not supported"))
    end
end

# ─────────────────────────────────────────────────────────────────────
# C.1 — 1D-symmetric 2D Sod
# ─────────────────────────────────────────────────────────────────────

"""
    tier_c_sod_ic(; D = 2, level = 4, shock_axis = 1, x_split = 0.5,
                   ρL = 1.0, ρR = 0.125, pL = 1.0, pR = 0.1,
                   uL = 0.0, uR = 0.0, lo = nothing, hi = nothing,
                   quad_order = 3, T = Float64)

C.1 — 1D-symmetric 2D Sod shock tube (methods paper §10.4 C.1).

Builds a `HierarchicalMesh{D}` of `(2^level)^D` leaves on the box
`[lo, hi]` (defaults: `[0, 1] × [0, 0.2]` for D = 2, `[0, 1]` for
D = 1, `[0, 1] × [0, 0.2] × [0, 0.2]` for D = 3) and projects the
two-state Sod IC onto the leaves' cell-averages.

The discontinuity is along `shock_axis ∈ 1:D`: cells with center
coordinate `< x_split` along that axis carry the left state
`(ρL, uL, pL)`; cells with center `≥ x_split` carry the right state
`(ρR, uR, pR)`. The non-shock axes are y-periodic / z-periodic
(implicit; the IC itself is constant across them).

Returns a NamedTuple with fields `(name, mesh, frame, fields, params)`
matching the convention described in the module docstring.

# 1D parity

For `D = 1, shock_axis = 1` the cell-averages match `setup_sod`'s
two-state IC to round-off (provided `level` matches the
`N = 2^level`). The cells whose half straddles `x_split` get a
linearly-interpolated value (the cell-average of a step function);
this is the discrete analogue of M1's `ifelse.(x .< 0.5, ...)` after
one applies the cell-average operator.
"""
function tier_c_sod_ic(; D::Integer = 2,
                       level::Integer = 4,
                       shock_axis::Integer = 1,
                       x_split::Real = 0.5,
                       ρL::Real = 1.0, ρR::Real = 0.125,
                       pL::Real = 1.0, pR::Real = 0.1,
                       uL::Real = 0.0, uR::Real = 0.0,
                       lo = nothing, hi = nothing,
                       quad_order::Integer = 3,
                       T::Type = Float64)
    1 <= shock_axis <= D ||
        throw(ArgumentError("shock_axis $shock_axis out of range [1, $D]"))

    # Default physical box: x in [0,1], non-shock axes thin (length 0.2).
    lo_t, hi_t = _default_box(D, lo, hi, T;
                               default_lo = ntuple(d -> zero(T), D),
                               default_hi = ntuple(d -> d == 1 ? one(T)
                                                                : T(0.2), D))

    mesh = uniform_eulerian_mesh(D, level)
    frame = EulerianFrame(mesh, lo_t, hi_t)
    leaves = enumerate_leaves(mesh)
    n = length(leaves)
    fields = _allocate_tierc_fields(D, n; T = T)
    quad = _quad_for_dim(D, quad_order; T = T)

    # Per-cell average of the two-state primitives. The step is along
    # shock_axis; we map the "indicator(x_axis < x_split)" through the
    # quadrature so cells straddling the split receive the linearly
    # interpolated cell-average.
    rho_l = T(ρL); rho_r = T(ρR)
    p_l   = T(pL); p_r   = T(pR)
    u_l   = T(uL); u_r   = T(uR)
    xs    = T(x_split)
    sa    = Int(shock_axis)

    f_rho(x) = x[sa] < xs ? rho_l : rho_r
    f_P(x)   = x[sa] < xs ? p_l   : p_r
    f_us(x)  = x[sa] < xs ? u_l   : u_r   # velocity along shock axis

    @inbounds for (j, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        fields.rho[j] = (_cell_average(f_rho, lo_c, hi_c, quad),)
        fields.P[j]   = (_cell_average(f_P,   lo_c, hi_c, quad),)
        # Set every velocity component: along shock_axis use the
        # primitive jump; transverse axes are zero by construction.
        for d in 1:D
            v = d == sa ? _cell_average(f_us, lo_c, hi_c, quad) : zero(T)
            _set_velocity_component!(fields, j, d, v)
        end
    end

    return (
        name = "tier_c_sod",
        mesh = mesh, frame = frame, fields = fields,
        params = (D = D, level = level, shock_axis = shock_axis,
                  x_split = x_split,
                  ρL = ρL, ρR = ρR, pL = pL, pR = pR, uL = uL, uR = uR,
                  lo = lo_t, hi = hi_t, quad_order = quad_order),
    )
end

# ─────────────────────────────────────────────────────────────────────
# C.2 — 2D cold sinusoid
# ─────────────────────────────────────────────────────────────────────

"""
    tier_c_cold_sinusoid_ic(; D = 2, level = 4, A = 0.5,
                            k = nothing, ρ0 = 1.0, P0 = 1.0,
                            lo = nothing, hi = nothing,
                            quad_order = 3, T = Float64)

C.2 — 2D cold sinusoid (methods paper §10.4 C.2).

Velocity field per axis:

    u_d(x) = A * sin(2π k_d x_d / L_d)         (component d)

with uniform density `ρ0` and pressure `P0`. The wavevector `k` is a
`NTuple{D, Int}`; e.g. `k = (1, 0)` runs the 1D-symmetric variant
along x (the y-axis γ stays `O(1)` while the x-axis γ collapses), and
`k = (1, 1)` runs the genuinely 2D variant. The default `k` chooses
all axes active (`k = (1, 1, …)`).

The `L_d = hi[d] - lo[d]` factor in the argument means a single
period exactly fills the box along each axis, matching M1's
`setup_cold_sinusoid` convention (which uses `[0, 1]` periodic).

Returns the same NamedTuple structure as `tier_c_sod_ic`.

# Per-axis γ diagnostic

This factory does not compute the per-axis Cholesky γ; that is the
solver's job (Phase M3-3). The expected behaviour, asserted by the
M3-4 test, is that the analytical IC `u_d ∝ sin(2π k_d x_d)` produces
γ_d ≈ 1e-5 on axes with `k_d ≠ 0` (the "collapsing" axes; cf. the M1
Phase 3 cold-limit asymptote) and γ_d = O(1) on axes with `k_d = 0`
(the trivial axes).
"""
function tier_c_cold_sinusoid_ic(; D::Integer = 2,
                                 level::Integer = 4,
                                 A::Real = 0.5,
                                 k = nothing,
                                 ρ0::Real = 1.0,
                                 P0::Real = 1.0,
                                 lo = nothing, hi = nothing,
                                 quad_order::Integer = 3,
                                 T::Type = Float64)
    lo_t, hi_t = _default_box(D, lo, hi, T;
                               default_lo = ntuple(d -> zero(T), D),
                               default_hi = ntuple(d -> one(T), D))

    # Default: full 2D / 3D variant (all axes active).
    k_t::NTuple{D,Int} = if k === nothing
        ntuple(_ -> 1, D)
    else
        length(k) == D ||
            throw(ArgumentError("k length $(length(k)) does not match D=$D"))
        ntuple(d -> Int(k[d]), D)
    end

    mesh = uniform_eulerian_mesh(D, level)
    frame = EulerianFrame(mesh, lo_t, hi_t)
    leaves = enumerate_leaves(mesh)
    n = length(leaves)
    fields = _allocate_tierc_fields(D, n; T = T)
    quad = _quad_for_dim(D, quad_order; T = T)

    A_t  = T(A)
    ρ_t  = T(ρ0)
    P_t  = T(P0)
    L_d  = ntuple(d -> hi_t[d] - lo_t[d], D)
    lo_d = lo_t

    # u_d(x) = A * sin(2π k_d (x_d - lo_d) / L_d) when k_d != 0, else 0.
    @inline function u_axis(x, axis::Int)
        kd = k_t[axis]
        kd == 0 && return zero(T)
        return A_t * sin(T(2π) * T(kd) * (x[axis] - lo_d[axis]) / L_d[axis])
    end

    @inbounds for (j, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        fields.rho[j] = (ρ_t,)
        fields.P[j]   = (P_t,)
        for d in 1:D
            f_d(x) = u_axis(x, d)
            v = _cell_average(f_d, lo_c, hi_c, quad)
            _set_velocity_component!(fields, j, d, v)
        end
    end

    return (
        name = "tier_c_cold_sinusoid",
        mesh = mesh, frame = frame, fields = fields,
        params = (D = D, level = level, A = A, k = k_t,
                  ρ0 = ρ0, P0 = P0,
                  lo = lo_t, hi = hi_t, quad_order = quad_order),
    )
end

# ─────────────────────────────────────────────────────────────────────
# C.3 — Plane wave at arbitrary direction
# ─────────────────────────────────────────────────────────────────────

"""
    tier_c_plane_wave_ic(; D = 2, level = 4, A = 1e-3, k_mag = 1,
                         angle = 0.0, ρ0 = 1.0, P0 = 1.0, c = nothing,
                         lo = nothing, hi = nothing,
                         quad_order = 3, T = Float64)

C.3 — Acoustic plane wave at angle `angle` to the x-axis (methods
paper §10.4 C.3).

The wavevector is `k = k_mag * (cos θ, sin θ)` for D = 2 and
`(cos θ, sin θ, 0)` for D = 3 (2D-in-3D variant). The IC seeds a
single right-going acoustic mode of the linearised Euler equations
about the uniform background `(ρ0, 0, P0)`:

    δρ(x) = A * cos(2π k · x / L_box)
    δu_d(x) = (c / ρ0) * δρ(x) * k̂_d
    δp(x) = c² δρ(x)               (isothermal-equivalent for the IC)

where the sound speed `c = sqrt(γ p₀/ρ₀)` defaults to `sqrt(P0/ρ0)`
when `c` is not given (γ = 1; tests pass an explicit `c` when they
need a γ ≠ 1 EOS). The factor of `1/L_box` (taking `L_box[1]` as the
common reference length) ensures one full wavelength fits the box
when `k_mag = 1` along the x-axis at θ = 0; rotating θ doesn't
change the wavelength along k.

Returns the same NamedTuple structure as `tier_c_sod_ic`.

# Rotational invariance contract

For the same `k_mag`, two factories at θ = 0 and θ = π/2 should
produce field sets related by a 90° rotation of velocity components
and a 90° permutation of the spatial dependence. The M3-4 test asserts
this to plotting precision (1e-3) at level = 4 and tightens with
refinement.

# 1D parity

For `D = 1` the `angle` kwarg is ignored (`k = (k_mag,)`). The
resulting IC matches the M1 1D plane-wave configuration at θ = 0
modulo the cell-average vs point-sample distinction (which falls
off as `O(Δx²)` for cosines).
"""
function tier_c_plane_wave_ic(; D::Integer = 2,
                              level::Integer = 4,
                              A::Real = 1e-3,
                              k_mag::Real = 1,
                              angle::Real = 0.0,
                              ρ0::Real = 1.0,
                              P0::Real = 1.0,
                              c::Union{Real,Nothing} = nothing,
                              lo = nothing, hi = nothing,
                              quad_order::Integer = 3,
                              T::Type = Float64)
    lo_t, hi_t = _default_box(D, lo, hi, T;
                               default_lo = ntuple(d -> zero(T), D),
                               default_hi = ntuple(d -> one(T), D))

    mesh = uniform_eulerian_mesh(D, level)
    frame = EulerianFrame(mesh, lo_t, hi_t)
    leaves = enumerate_leaves(mesh)
    n = length(leaves)
    fields = _allocate_tierc_fields(D, n; T = T)
    quad = _quad_for_dim(D, quad_order; T = T)

    # Wavevector in physical units, scaled to "k_mag full wavelengths
    # per box side". Use lo_t[1]'s box length as the reference scale.
    L1 = hi_t[1] - lo_t[1]
    k_phys = if D == 1
        (T(k_mag) / L1,)
    elseif D == 2
        θ = T(angle)
        (T(k_mag) * cos(θ) / L1, T(k_mag) * sin(θ) / L1)
    elseif D == 3
        θ = T(angle)
        (T(k_mag) * cos(θ) / L1, T(k_mag) * sin(θ) / L1, zero(T))
    else
        throw(ArgumentError("dimension D=$D not supported"))
    end
    # Unit direction in the physical k vector (used for δu direction).
    kmag_phys = sqrt(sum(k_phys[d]^2 for d in 1:D))
    k̂ = ntuple(d -> kmag_phys > 0 ? k_phys[d] / kmag_phys : zero(T), D)

    A_t = T(A); ρ_t = T(ρ0); P_t = T(P0)
    c_eff = c === nothing ? sqrt(P_t / ρ_t) : T(c)

    @inline function δρ(x)
        # k · (x - lo) phase in cycles
        phase = zero(T)
        for d in 1:D
            phase += k_phys[d] * (x[d] - lo_t[d])
        end
        return A_t * cos(T(2π) * phase)
    end
    f_rho(x) = ρ_t + δρ(x)
    f_P(x)   = P_t + c_eff^2 * δρ(x)
    @inline function f_u(x, d::Int)
        return (c_eff / ρ_t) * δρ(x) * k̂[d]
    end

    @inbounds for (j, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        fields.rho[j] = (_cell_average(f_rho, lo_c, hi_c, quad),)
        fields.P[j]   = (_cell_average(f_P,   lo_c, hi_c, quad),)
        for d in 1:D
            f_d(x) = f_u(x, d)
            v = _cell_average(f_d, lo_c, hi_c, quad)
            _set_velocity_component!(fields, j, d, v)
        end
    end

    return (
        name = "tier_c_plane_wave",
        mesh = mesh, frame = frame, fields = fields,
        params = (D = D, level = level, A = A, k_mag = k_mag,
                  angle = angle, ρ0 = ρ0, P0 = P0, c = c_eff,
                  k = k_phys, k̂ = k̂,
                  lo = lo_t, hi = hi_t, quad_order = quad_order),
    )
end

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

"""
    _default_box(D, lo, hi, T; default_lo, default_hi)

Resolve user-supplied (or default) physical box bounds into matched
`NTuple{D, T}` `lo` and `hi`. Used by every Tier-C factory.
"""
function _default_box(D::Integer, lo, hi, T::Type;
                      default_lo, default_hi)
    lo_t = if lo === nothing
        NTuple{Int(D), T}(default_lo)
    else
        length(lo) == D ||
            throw(ArgumentError("lo has length $(length(lo)), expected D=$D"))
        ntuple(d -> T(lo[d]), Int(D))
    end
    hi_t = if hi === nothing
        NTuple{Int(D), T}(default_hi)
    else
        length(hi) == D ||
            throw(ArgumentError("hi has length $(length(hi)), expected D=$D"))
        ntuple(d -> T(hi[d]), Int(D))
    end
    return lo_t, hi_t
end

"""
    _set_velocity_component!(fields, j, axis, value)

Write a velocity-component cell-average into the Tier-C field set.
Dispatches on `axis ∈ 1:3` to the matching named field
(`:ux`, `:uy`, `:uz`). Used by all three factories.
"""
@inline function _set_velocity_component!(fields::PolynomialFieldSet,
                                           j::Integer, axis::Integer,
                                           value)
    if axis == 1
        fields.ux[j] = (value,)
    elseif axis == 2
        fields.uy[j] = (value,)
    elseif axis == 3
        fields.uz[j] = (value,)
    else
        throw(ArgumentError("axis $axis out of range [1, 3]"))
    end
    return value
end

"""
    tier_c_velocity_component(fields, j, axis)

Read the cell-average of the `axis`-th velocity component at leaf `j`.
Dispatches across `:ux`, `:uy`, `:uz` like `_set_velocity_component!`.
"""
@inline function tier_c_velocity_component(fields::PolynomialFieldSet,
                                            j::Integer, axis::Integer)
    if axis == 1
        return fields.ux[j][1]
    elseif axis == 2
        return fields.uy[j][1]
    elseif axis == 3
        return fields.uz[j][1]
    else
        throw(ArgumentError("axis $axis out of range [1, 3]"))
    end
end

"""
    tier_c_total_mass(ic::NamedTuple)

Compute the discrete total mass `Σ_j ρ_j * V_j` of an IC factory's
field set. The cell volume is read from
`cell_physical_box(ic.frame, leaf_index)`. Useful for the
mass-conservation assertion in the M3-4 unit tests.
"""
function tier_c_total_mass(ic::NamedTuple)
    leaves = enumerate_leaves(ic.mesh)
    M = 0.0
    @inbounds for (j, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(ic.frame, ci)
        V = 1.0
        for d in 1:length(lo_c)
            V *= hi_c[d] - lo_c[d]
        end
        M += V * ic.fields.rho[j][1]
    end
    return M
end

"""
    tier_c_cell_centers(ic::NamedTuple)

Return a `Vector{NTuple{D, T}}` of leaf-cell centers in the IC's
physical frame, one per cell in `enumerate_leaves(ic.mesh)` order.
Used by the IC sanity tests to look up primitive values at known
points.
"""
function tier_c_cell_centers(ic::NamedTuple)
    leaves = enumerate_leaves(ic.mesh)
    centers = Vector{NTuple{length(ic.params.lo), Float64}}(undef,
                                                              length(leaves))
    @inbounds for (j, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(ic.frame, ci)
        D = length(lo_c)
        centers[j] = ntuple(d -> 0.5 * (lo_c[d] + hi_c[d]), D)
    end
    return centers
end
