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

# L²-projection backing (M3-2b Swap 5)

Per-cell projection is delegated to HG's `init_field_from!`
(`HierarchicalGrids.Storage.Initialization`, commit 4b481da). For an
order-0 monomial basis the local mass matrix reduces to `M = [1]` and
the L²-projection collapses to the cell average

    c[1] = ∫_ref f(x(ξ)) dξ  ≈  Σ_k w_k * f(x_k),

reproducing the previous hand-rolled tensor-product GL quadrature
exactly for the same number of GL points per axis. The public
`quad_order` kwarg keeps its original semantics ("n GL points per
axis"); internally we translate to HG's `quadrature_order = 2n - 1`
which yields the same n-point tensor rule.

`init_field_from!` writes coefficients to *every* cell of the input
field set (including non-leaves of an AMR tree). The Tier-C public API
exposes leaf-indexed storage, so each factory allocates a temporary
n_cells-sized single-field PolynomialFieldSet per quantity, runs
`init_field_from!` on it, and copies leaf entries into the master
n_leaves-sized field set. This preserves bit-exact parity with the
previous hand-rolled version (the L²-projection at P = 0 is the
midpoint/GL cell average) while removing ~150 LOC of in-factory
quadrature plumbing.

# References

- methods paper §10.4 (Tier C list)
- `reference/MILESTONE_3_PLAN.md` Phase M3-4
- `reference/notes_M3_2b_swap5_init_field.md` (this swap's design note)
"""

using HierarchicalGrids: HierarchicalMesh, EulerianFrame,
    refine_cells!, enumerate_leaves, cell_physical_box,
    MonomialBasis, allocate_polynomial_fields, SoA, PolynomialFieldSet,
    init_field_from!

# ─────────────────────────────────────────────────────────────────────
# Mesh + field-set plumbing (dimension-generic)
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
    _hg_quadrature_order(quad_order_n::Integer) -> Int

Translate dfmm's "n GL points per axis" `quad_order` into HG's
`init_field_from!(...; quadrature_order)` argument. HG selects the
tensor-product rule via `n = ceil((order + 1) / 2)`, so
`quadrature_order = 2n - 1` reproduces the n-point Gauss-Legendre
tensor rule used by the legacy hand-rolled factory.
"""
@inline _hg_quadrature_order(n::Integer) = 2 * Int(n) - 1

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

"""
    _project_scalar_to_leaves!(target_view, frame, leaves, f, basis,
                                n_total_cells, hg_quad_order)

Project the scalar function `f` onto the order-0 polynomial basis on
every cell of the parent mesh via `init_field_from!`, then copy the
resulting leaf coefficients into `target_view` (a single named field
view of the public Tier-C field set). Returns nothing.

The temporary PolynomialFieldSet is allocated freshly per call; for
the order-0 monomial basis used by Tier-C, each cell holds one scalar
(the cell average), so this is a thin wrapper around HG's L²
projection helper.
"""
function _project_scalar_to_leaves!(target_view, frame::EulerianFrame{D, T},
                                     leaves::AbstractVector{<:Integer},
                                     f, basis, n_total_cells::Integer,
                                     hg_quad_order::Integer) where {D, T}
    tmp = allocate_polynomial_fields(SoA(), basis, Int(n_total_cells);
                                      v = T)
    init_field_from!(tmp, frame, f; quadrature_order = Int(hg_quad_order))
    @inbounds for (j, ci) in enumerate(leaves)
        target_view[j] = (tmp.v[ci][1],)
    end
    return nothing
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

    # HG's L² projection helper writes to every cell of its target field;
    # we project on the full n_cells(mesh) and copy leaves into `fields`.
    basis = MonomialBasis{Int(D), 0}()
    nc = HierarchicalGrids.n_cells(mesh)
    hg_qo = _hg_quadrature_order(quad_order)

    rho_l = T(ρL); rho_r = T(ρR)
    p_l   = T(pL); p_r   = T(pR)
    u_l   = T(uL); u_r   = T(uR)
    xs    = T(x_split)
    sa    = Int(shock_axis)

    # Two-state primitives. The step is along shock_axis; mapping the
    # indicator through the quadrature gives cells straddling the split
    # the linearly-interpolated cell-average.
    f_rho = x -> x[sa] < xs ? rho_l : rho_r
    f_P   = x -> x[sa] < xs ? p_l   : p_r
    f_us  = x -> x[sa] < xs ? u_l   : u_r   # velocity along shock axis
    f_zero = _ -> zero(T)

    _project_scalar_to_leaves!(fields.rho, frame, leaves, f_rho, basis, nc, hg_qo)
    _project_scalar_to_leaves!(fields.P,   frame, leaves, f_P,   basis, nc, hg_qo)
    for d in 1:D
        view_d = _velocity_view(fields, d)
        f_d = (d == sa) ? f_us : f_zero
        _project_scalar_to_leaves!(view_d, frame, leaves, f_d, basis, nc, hg_qo)
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

    basis = MonomialBasis{Int(D), 0}()
    nc = HierarchicalGrids.n_cells(mesh)
    hg_qo = _hg_quadrature_order(quad_order)

    A_t  = T(A)
    ρ_t  = T(ρ0)
    P_t  = T(P0)
    L_d  = ntuple(d -> hi_t[d] - lo_t[d], D)
    lo_d = lo_t

    # Constant fields: ρ0, P0.
    _project_scalar_to_leaves!(fields.rho, frame, leaves,
                                _ -> ρ_t, basis, nc, hg_qo)
    _project_scalar_to_leaves!(fields.P, frame, leaves,
                                _ -> P_t, basis, nc, hg_qo)

    # u_d(x) = A * sin(2π k_d (x_d - lo_d) / L_d) when k_d != 0, else 0.
    for d in 1:D
        kd = k_t[d]
        Ld = L_d[d]
        ld = lo_d[d]
        f_d = if kd == 0
            _ -> zero(T)
        else
            let A_t = A_t, kd = kd, Ld = Ld, ld = ld, axis = d
                x -> A_t * sin(T(2π) * T(kd) * (x[axis] - ld) / Ld)
            end
        end
        view_d = _velocity_view(fields, d)
        _project_scalar_to_leaves!(view_d, frame, leaves, f_d, basis, nc, hg_qo)
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

    basis = MonomialBasis{Int(D), 0}()
    nc = HierarchicalGrids.n_cells(mesh)
    hg_qo = _hg_quadrature_order(quad_order)

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

    # Closures pulled out to keep the projection calls flat.
    let A_t = A_t, ρ_t = ρ_t, P_t = P_t, c_eff = c_eff,
        k_phys = k_phys, k̂ = k̂, lo_t = lo_t, D = D
        @inline function δρ(x)
            phase = zero(T)
            for d in 1:D
                phase += k_phys[d] * (x[d] - lo_t[d])
            end
            return A_t * cos(T(2π) * phase)
        end
        f_rho = x -> ρ_t + δρ(x)
        f_P   = x -> P_t + c_eff^2 * δρ(x)

        _project_scalar_to_leaves!(fields.rho, frame, leaves, f_rho, basis, nc, hg_qo)
        _project_scalar_to_leaves!(fields.P,   frame, leaves, f_P,   basis, nc, hg_qo)
        for d in 1:D
            kd_hat = k̂[d]
            f_d = let kd_hat = kd_hat, c_eff = c_eff, ρ_t = ρ_t
                x -> (c_eff / ρ_t) * δρ(x) * kd_hat
            end
            view_d = _velocity_view(fields, d)
            _project_scalar_to_leaves!(view_d, frame, leaves, f_d, basis, nc, hg_qo)
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
    _velocity_view(fields, axis)

Return the named field view for the `axis`-th velocity component
(`:ux`, `:uy`, `:uz`) of a Tier-C `PolynomialFieldSet`. Used to plumb
per-axis projections into the master field set.
"""
@inline function _velocity_view(fields::PolynomialFieldSet, axis::Integer)
    if axis == 1
        return fields.ux
    elseif axis == 2
        return fields.uy
    elseif axis == 3
        return fields.uz
    else
        throw(ArgumentError("axis $axis out of range [1, 3]"))
    end
end

"""
    tier_c_velocity_component(fields, j, axis)

Read the cell-average of the `axis`-th velocity component at leaf `j`.
Dispatches across `:ux`, `:uy`, `:uz` (the inverse of the per-axis
projection writes performed by the factories).
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

# ─────────────────────────────────────────────────────────────────────
# M3-3a: 2D Cholesky-sector field-set allocation
# ─────────────────────────────────────────────────────────────────────

"""
    allocate_cholesky_2d_fields(mesh::HierarchicalMesh{2}; T::Type = Float64)
        -> PolynomialFieldSet

Allocate the per-cell `PolynomialFieldSet` for the M3-3 2D
Cholesky-sector EL system. The field set has 12 named scalar fields
laid out as

    Newton unknowns (10):
        :x_1, :x_2     — Lagrangian position components (charge 0)
        :u_1, :u_2     — Lagrangian velocity components (charge 0)
        :α_1, :α_2     — principal-axis Cholesky factors (charge 0)
        :β_1, :β_2     — per-axis conjugate momenta (charge 1)
        :θ_R           — principal-axis rotation angle (Berry coupling)
        :s             — specific entropy (charge 0)

    Post-Newton sectors (2):
        :Pp            — deviatoric pressure (Phase 5; per-cell scalar
                          for now, M3-6 will lift this to per-axis
                          under D.1 KH activation)
        :Q             — heat-flux scalar (Phase 7)

each at `MonomialBasis{2, 0}` (one coefficient per cell — the
order-0 cell-average storage that mirrors M3-2's 1D substrate).

# Sizing

The field set is sized to `n_cells(mesh)`, NOT `n_leaves(mesh)`, so
that HG's `halo_view`-based neighbor access (`hv[i, off]`) maps
1-to-1 against `mesh.cells[i]`. The leaf-only public API of M3-3b
will iterate via `enumerate_leaves(mesh)` and use those mesh-cell
indices directly into the field set.

# Off-diagonal β fields (M3-6 Phase 0)

The off-diagonal Cholesky factors `β_{12}, β_{21}` were omitted in
M3-3a (Q3 of `reference/notes_M3_3_2d_cholesky_berry.md` §10) and
**re-added in M3-6 Phase 0** as the prerequisite for the D.1 KH
falsifier (Phase 1). The allocator now produces 14 named scalar
fields: the 10 prior Newton fields, the new `:β_12, :β_21` pair,
plus the 2 post-Newton sectors `Pp, Q`. Off-diagonal β IC defaults
to zero in every IC factory so all pre-M3-6 regression configurations
remain byte-equal (the new residual rows trivialise at β_12=β_21=0).

The math source for the off-diag pair is §7 of
`reference/notes_M3_phase0_berry_connection.md` (axis-swap-
antisymmetric Berry extension) and `scripts/verify_berry_connection_offdiag.py`
(9 SymPy CHECKs reproduced numerically by `test_M3_6_phase0_offdiag_residual.jl`).
"""
function allocate_cholesky_2d_fields(mesh::HierarchicalMesh{2};
                                      T::Type = Float64)
    basis = MonomialBasis{2, 0}()
    nc = HierarchicalGrids.n_cells(mesh)
    return allocate_polynomial_fields(SoA(), basis, Int(nc);
                                       x_1 = T, x_2 = T,
                                       u_1 = T, u_2 = T,
                                       α_1 = T, α_2 = T,
                                       β_1 = T, β_2 = T,
                                       β_12 = T, β_21 = T,   # M3-6 Phase 0
                                       θ_R = T,
                                       s   = T,
                                       Pp  = T,
                                       Q   = T)
end

"""
    write_detfield_2d!(fields::PolynomialFieldSet, leaf_cell_idx::Integer,
                        v::DetField2D)

Write a `DetField2D` value into the 14-named-field 2D field set at
mesh-cell index `leaf_cell_idx`. The field set must be allocated by
`allocate_cholesky_2d_fields`. Writes 14 scalars total (M3-6 Phase 0:
the off-diag pair `β_12, β_21` is included).
"""
function write_detfield_2d!(fields::PolynomialFieldSet, leaf_cell_idx::Integer,
                             v::DetField2D)
    fields.x_1[leaf_cell_idx] = (v.x[1],)
    fields.x_2[leaf_cell_idx] = (v.x[2],)
    fields.u_1[leaf_cell_idx] = (v.u[1],)
    fields.u_2[leaf_cell_idx] = (v.u[2],)
    fields.α_1[leaf_cell_idx] = (v.alphas[1],)
    fields.α_2[leaf_cell_idx] = (v.alphas[2],)
    fields.β_1[leaf_cell_idx] = (v.betas[1],)
    fields.β_2[leaf_cell_idx] = (v.betas[2],)
    fields.β_12[leaf_cell_idx] = (v.betas_off[1],)   # M3-6 Phase 0
    fields.β_21[leaf_cell_idx] = (v.betas_off[2],)   # M3-6 Phase 0
    fields.θ_R[leaf_cell_idx] = (v.θ_R,)
    fields.s[leaf_cell_idx]   = (v.s,)
    fields.Pp[leaf_cell_idx]  = (v.Pp,)
    fields.Q[leaf_cell_idx]   = (v.Q,)
    return nothing
end

"""
    read_detfield_2d(fields::PolynomialFieldSet, leaf_cell_idx::Integer)
        -> DetField2D

Read a `DetField2D` value from the 14-named-field 2D field set at
mesh-cell index `leaf_cell_idx`. Inverse of `write_detfield_2d!`;
round-trip is bit-exact (M3-6 Phase 0: includes the off-diag pair).
"""
function read_detfield_2d(fields::PolynomialFieldSet, leaf_cell_idx::Integer)
    x  = (fields.x_1[leaf_cell_idx][1], fields.x_2[leaf_cell_idx][1])
    u  = (fields.u_1[leaf_cell_idx][1], fields.u_2[leaf_cell_idx][1])
    α  = (fields.α_1[leaf_cell_idx][1], fields.α_2[leaf_cell_idx][1])
    β  = (fields.β_1[leaf_cell_idx][1], fields.β_2[leaf_cell_idx][1])
    βoff = (fields.β_12[leaf_cell_idx][1], fields.β_21[leaf_cell_idx][1])  # M3-6 Phase 0
    θR = fields.θ_R[leaf_cell_idx][1]
    s  = fields.s[leaf_cell_idx][1]
    Pp = fields.Pp[leaf_cell_idx][1]
    Q  = fields.Q[leaf_cell_idx][1]
    return DetField2D(x, u, α, β, βoff, θR, s, Pp, Q)
end

# ─────────────────────────────────────────────────────────────────────
# M3-4 Phase 2: Tier-C IC bridge — primitive (ρ, u_x, u_y, P) → Cholesky
# ─────────────────────────────────────────────────────────────────────
#
# Bridges the existing primitive Tier-C IC factories
# (`tier_c_sod_ic`, `tier_c_cold_sinusoid_ic`, `tier_c_plane_wave_ic`)
# onto the M3-3 Cholesky-sector field set
# `(x_a, u_a, α_a, β_a, θ_R, s, Pp, Q)` consumed by
# `det_step_2d_berry_HG!`.
#
# Conventions (per the M3-4 handoff §"Pre-Tier-C handoff items"):
#   • α_a = 1.0 for both axes (cold-limit, isotropic IC).
#   • β_a = 0.0 for both axes.
#   • θ_R = 0.0 (no off-diagonal strain at IC).
#   • s solved from the EOS: Mvv(J = 1/ρ, s) = P/ρ
#       => log(P/ρ) = (1-Γ) log(1/ρ) + s/c_v
#       => s        = c_v · (log(P/ρ) - (1-Γ)·log(1/ρ))
#                   = c_v · (log(P/ρ) + (Γ-1)·log(1/ρ))
#   • Pp = 0, Q = 0 (no deviatoric / heat-flux at IC).
#   • (x_1, x_2) = cell centers from `cell_physical_box(frame, ci)`.
#   • (u_1, u_2) from the Tier-C velocity components.

"""
    s_from_pressure_density(ρ, P; Gamma=GAMMA_LAW_DEFAULT, cv=CV_DEFAULT)

Inverse of the EOS closure `M_vv(J, s) = J^(1-Γ) · exp(s/c_v) = P/ρ`
with `J = 1/ρ`. Solves for the specific entropy `s`:

    s = c_v · (log(P/ρ) + (Γ-1)·log(1/ρ))
      = c_v · (log(P/ρ) - (Γ-1)·log(ρ))

Round-trip: `Mvv(1/ρ, s_from_pressure_density(ρ, P)) ≈ P/ρ` to round-off.

Used by the Tier-C IC bridge to map primitive `(ρ, P)` to the
Cholesky-sector state's specific entropy field.
"""
@inline function s_from_pressure_density(ρ::Real, P::Real;
                                          Gamma::Real = GAMMA_LAW_DEFAULT,
                                          cv::Real = CV_DEFAULT)
    ρ > 0 && P > 0 ||
        throw(ArgumentError("ρ ($ρ) and P ($P) must be > 0"))
    return cv * (log(P / ρ) + (Gamma - 1) * log(1 / ρ))
end

"""
    cholesky_sector_state_from_primitive(ρ, u_x, u_y, P, x_center;
                                          Gamma=GAMMA_LAW_DEFAULT,
                                          cv=CV_DEFAULT)
        -> DetField2D{Float64}

Tier-C IC bridge: map a primitive `(ρ, u_x, u_y, P)` cell average plus
the cell center `x_center::NTuple{2}` onto the M3-3 Cholesky-sector
state `(x_a, u_a, α_a, β_a, θ_R, s, Pp, Q)`. Returns a `DetField2D`
ready to write into a `PolynomialFieldSet` allocated by
`allocate_cholesky_2d_fields`.

Cold-limit, isotropic IC convention (per M3-4 handoff):
  • α_a = 1.0, β_a = 0.0 for both axes.
  • θ_R = 0.0.
  • s from EOS: `s = c_v · (log(P/ρ) - (Γ-1)·log(ρ))`.
  • Pp = 0, Q = 0.

# Notes

  • The α=1, β=0 cold-limit choice puts γ²_a = M_vv − β² = P/ρ at IC.
    The Newton driver evolves α, β, s under the per-axis EL system
    starting from this initialization.
  • The pressure stored implicitly via `s` is recovered by
    `Mvv(1/ρ, s) = P/ρ`; downstream the Newton driver consumes
    `s` directly through `aux.s_vec`.
"""
function cholesky_sector_state_from_primitive(ρ::Real, u_x::Real, u_y::Real,
                                                P::Real,
                                                x_center::NTuple{2,<:Real};
                                                Gamma::Real = GAMMA_LAW_DEFAULT,
                                                cv::Real = CV_DEFAULT)
    s = s_from_pressure_density(ρ, P; Gamma = Gamma, cv = cv)
    # M3-6 Phase 0: explicit `betas_off = (0, 0)` (off-diag β default
    # at IC for all pre-M3-6 setups; M3-6 Phase 1 / D.1 KH will activate
    # non-zero IC for the falsifier driver).
    return DetField2D{Float64}(
        (Float64(x_center[1]), Float64(x_center[2])),
        (Float64(u_x), Float64(u_y)),
        (1.0, 1.0),
        (0.0, 0.0),
        (0.0, 0.0),     # betas_off — M3-6 Phase 0
        0.0,
        Float64(s),
        0.0, 0.0,
    )
end

"""
    primitive_recovery_2d(fields::PolynomialFieldSet, leaves, frame;
                          Gamma=GAMMA_LAW_DEFAULT, cv=CV_DEFAULT)
        -> NamedTuple{(:ρ, :u_x, :u_y, :P, :x, :y), NTuple{6, Vector{Float64}}}

Inverse of the Tier-C IC bridge: walk the leaves of a 2D Cholesky-
sector field set and recover primitive `(ρ, u_x, u_y, P)` per cell
(plus cell-center coordinates `x, y`) for diagnostics.

The recovery uses the geometric cell extent (from `cell_physical_box`)
and the per-cell `x_a` Newton-state values to estimate the cell's
specific volume per axis. For pure-translation flows where cell
boundaries follow `x_a` strictly, this gives the "physical" instantaneous
density per cell. For the M3-4 acceptance gates we treat the C.1 / C.2 /
C.3 ICs as small-amplitude or 1D-symmetric, so the per-cell density is
read as `ρ = ρ_ref / J_self` where `J_self = 1 / ρ_ref` at IC and
evolves only weakly under the Newton step.

This is the M3-4 first-cut recovery: the Newton driver does not yet
update cell boundaries (M3-5 / M3-6 will), so for the C.1 / C.2 / C.3
gates we recover ρ from the stored `s` and the IC-configured `ρ_ref`
rather than from a Lagrangian-volume-tracking calculation. Specifically:

    P/ρ = M_vv(J = 1/ρ, s) = ρ^(Γ-1) · exp(s/c_v)
    => assuming ρ ≈ ρ_ref (zero-strain limit), P = ρ · M_vv(1/ρ, s).

Returns a NamedTuple with vectors of length `length(leaves)` in leaf
order:
  • `ρ::Vector{Float64}` — per-cell density (currently uniform under the
    cold-limit IC bridge; future Lagrangian-volume tracking will make
    this time-varying).
  • `u_x, u_y::Vector{Float64}` — per-cell velocity components from
    the field set's `:u_1, :u_2` slots.
  • `P::Vector{Float64}` — per-cell pressure, P = ρ · M_vv(1/ρ, s).
  • `x, y::Vector{Float64}` — cell-center coordinates from
    `cell_physical_box(frame, ci)`. Useful for y-independence and
    1D-reduction-vs-golden gates.

# Use cases (M3-4 acceptance gates)

  • C.1 1D-symmetric Sod: extract a y=const slice → compare to
    `reference/golden/A1_sod.h5`.
  • C.1 y-independence diagnostic: assert ρ(x, y_1) ≈ ρ(x, y_2) for
    all y_1, y_2 in the mesh.
  • C.3 phase-velocity check: track perturbation amplitude over time
    along the `k`-direction.
"""
function primitive_recovery_2d(fields::PolynomialFieldSet,
                                leaves::AbstractVector{<:Integer},
                                frame::EulerianFrame{2, T};
                                ρ_ref::Real = 1.0,
                                Gamma::Real = GAMMA_LAW_DEFAULT,
                                cv::Real = CV_DEFAULT) where {T}
    N = length(leaves)
    ρ_out  = Vector{Float64}(undef, N)
    u_x    = Vector{Float64}(undef, N)
    u_y    = Vector{Float64}(undef, N)
    P_out  = Vector{Float64}(undef, N)
    x_out  = Vector{Float64}(undef, N)
    y_out  = Vector{Float64}(undef, N)
    @inbounds for (i, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        x_out[i] = 0.5 * (lo_c[1] + hi_c[1])
        y_out[i] = 0.5 * (lo_c[2] + hi_c[2])
        u_x[i]   = Float64(fields.u_1[ci][1])
        u_y[i]   = Float64(fields.u_2[ci][1])
        s_i      = Float64(fields.s[ci][1])
        # Cold-limit bridge IC: ρ ≈ ρ_ref. For the M3-4 zero-strain /
        # short-time gates this holds; the Newton driver evolves α and
        # β but does not modify the cell-tagged density. Future M3-5 /
        # M3-6 work tracks the Lagrangian J explicitly.
        ρ_out[i] = Float64(ρ_ref)
        P_out[i] = ρ_out[i] * Float64(Mvv(1.0 / ρ_out[i], s_i; Gamma = Gamma, cv = cv))
    end
    return (ρ = ρ_out, u_x = u_x, u_y = u_y, P = P_out, x = x_out, y = y_out)
end

"""
    primitive_recovery_2d_per_cell(fields, leaves, frame, ρ_per_cell;
                                    Gamma=GAMMA_LAW_DEFAULT, cv=CV_DEFAULT)

Variant of `primitive_recovery_2d` for non-uniform-density configurations
(e.g. C.1 Sod). Takes an explicit per-cell `ρ_per_cell::Vector{Float64}`
(length `N`, leaf-major) so the C.1 driver can pass the IC's two-state
density profile through to the recovery / pressure reconstruction.
"""
function primitive_recovery_2d_per_cell(fields::PolynomialFieldSet,
                                         leaves::AbstractVector{<:Integer},
                                         frame::EulerianFrame{2, T},
                                         ρ_per_cell::AbstractVector{<:Real};
                                         Gamma::Real = GAMMA_LAW_DEFAULT,
                                         cv::Real = CV_DEFAULT) where {T}
    N = length(leaves)
    @assert length(ρ_per_cell) == N "ρ_per_cell length $(length(ρ_per_cell)) ≠ N=$N"
    ρ_out  = Vector{Float64}(undef, N)
    u_x    = Vector{Float64}(undef, N)
    u_y    = Vector{Float64}(undef, N)
    P_out  = Vector{Float64}(undef, N)
    x_out  = Vector{Float64}(undef, N)
    y_out  = Vector{Float64}(undef, N)
    @inbounds for (i, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        x_out[i] = 0.5 * (lo_c[1] + hi_c[1])
        y_out[i] = 0.5 * (lo_c[2] + hi_c[2])
        u_x[i]   = Float64(fields.u_1[ci][1])
        u_y[i]   = Float64(fields.u_2[ci][1])
        s_i      = Float64(fields.s[ci][1])
        ρ_i      = Float64(ρ_per_cell[i])
        ρ_out[i] = ρ_i
        P_out[i] = ρ_i * Float64(Mvv(1.0 / ρ_i, s_i; Gamma = Gamma, cv = cv))
    end
    return (ρ = ρ_out, u_x = u_x, u_y = u_y, P = P_out, x = x_out, y = y_out)
end

# ─────────────────────────────────────────────────────────────────────
# M3-4 Phase 2: Full-IC factories — Cholesky-sector field set built
# from primitive Tier-C IC factories.
# ─────────────────────────────────────────────────────────────────────
#
# These extend the existing primitive `tier_c_*_ic` factories: the
# factories return primitive `(ρ, u_x, u_y, P)` field sets; the *_full_ic
# wrappers below allocate a Cholesky-sector field set, call the
# primitive factory under the hood, then apply the IC bridge per leaf.

"""
    tier_c_sod_full_ic(; level=4, shock_axis=1, x_split=0.5,
                       ρL=1.0, ρR=0.125, pL=1.0, pR=0.1,
                       uL=0.0, uR=0.0, lo=nothing, hi=nothing,
                       quad_order=3,
                       Gamma=GAMMA_LAW_DEFAULT, cv=CV_DEFAULT,
                       T=Float64)
        -> NamedTuple{(:name, :mesh, :frame, :leaves, :fields, :ρ_per_cell, :params)}

Allocate a 2D Cholesky-sector field set + balanced HG mesh and populate
it with the C.1 1D-symmetric 2D Sod IC. Returns:

  • `name = "tier_c_sod_full"`
  • `mesh::HierarchicalMesh{2}` — balanced quadtree of (2^level)² leaves.
  • `frame::EulerianFrame{2, T}`
  • `leaves::Vector{Int}` — leaf-iteration order.
  • `fields::PolynomialFieldSet` — 12-named-field 2D Cholesky-sector
    storage (allocated via `allocate_cholesky_2d_fields`), populated
    cell-by-cell via the IC bridge.
  • `ρ_per_cell::Vector{Float64}` — leaf-major per-cell density (Sod
    has a 8× jump along `shock_axis`); used by the C.1 driver as the
    per-cell density for the residual's pressure stencil and primitive
    recovery.
  • `params::NamedTuple` — IC parameters echo (for downstream tests).
"""
function tier_c_sod_full_ic(; level::Integer = 4,
                            shock_axis::Integer = 1,
                            x_split::Real = 0.5,
                            ρL::Real = 1.0, ρR::Real = 0.125,
                            pL::Real = 1.0, pR::Real = 0.1,
                            uL::Real = 0.0, uR::Real = 0.0,
                            lo = nothing, hi = nothing,
                            quad_order::Integer = 3,
                            Gamma::Real = GAMMA_LAW_DEFAULT,
                            cv::Real = CV_DEFAULT,
                            T::Type = Float64)
    # Step 1: build primitive Tier-C IC.
    prim_ic = tier_c_sod_ic(; D = 2, level = level, shock_axis = shock_axis,
                             x_split = x_split,
                             ρL = ρL, ρR = ρR, pL = pL, pR = pR,
                             uL = uL, uR = uR,
                             lo = lo, hi = hi,
                             quad_order = quad_order, T = T)

    # Step 2: rebuild a balanced HG mesh of the same shape (the
    # primitive IC uses a non-balanced default; the M3-3 Newton driver
    # asserts `mesh.balanced == true`).
    mesh = HierarchicalMesh{2}(; balanced = true)
    for _ in 1:level
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, prim_ic.params.lo, prim_ic.params.hi)
    fields = allocate_cholesky_2d_fields(mesh; T = T)

    # Step 3: per-leaf IC bridge.
    ρ_per_cell = Vector{Float64}(undef, length(leaves))
    @inbounds for (j, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        cx = 0.5 * (lo_c[1] + hi_c[1])
        cy = 0.5 * (lo_c[2] + hi_c[2])
        # Sod is a step along `shock_axis`; map cell-center.
        center = (cx, cy)
        is_left = center[shock_axis] < x_split
        ρ_i = Float64(is_left ? ρL : ρR)
        P_i = Float64(is_left ? pL : pR)
        u_axis = Float64(is_left ? uL : uR)
        u_x = shock_axis == 1 ? u_axis : 0.0
        u_y = shock_axis == 2 ? u_axis : 0.0
        v = cholesky_sector_state_from_primitive(ρ_i, u_x, u_y, P_i,
                                                   (cx, cy);
                                                   Gamma = Gamma, cv = cv)
        write_detfield_2d!(fields, ci, v)
        ρ_per_cell[j] = ρ_i
    end

    return (
        name = "tier_c_sod_full",
        mesh = mesh, frame = frame, leaves = leaves,
        fields = fields,
        ρ_per_cell = ρ_per_cell,
        params = (level = level, shock_axis = shock_axis, x_split = x_split,
                   ρL = ρL, ρR = ρR, pL = pL, pR = pR, uL = uL, uR = uR,
                   lo = prim_ic.params.lo, hi = prim_ic.params.hi,
                   quad_order = quad_order, Gamma = Gamma, cv = cv),
    )
end

"""
    tier_c_cold_sinusoid_full_ic(; level=4, A=0.5, k=(1, 0),
                                  ρ0=1.0, P0=1.0,
                                  lo=nothing, hi=nothing,
                                  quad_order=3,
                                  Gamma=GAMMA_LAW_DEFAULT, cv=CV_DEFAULT,
                                  T=Float64)
        -> NamedTuple

C.2 2D cold sinusoid full Cholesky-sector IC. Builds a balanced quadtree,
allocates the 12-field 2D Cholesky-sector field set, and populates it
per-leaf via the IC bridge. The velocity field is

    u_d(x) = A · sin(2π · k_d · (x_d − lo_d) / L_d)    for d ∈ {1, 2}

with uniform `(ρ0, P0)`. The density is uniform so `ρ_per_cell` is a
constant `fill(ρ0, N)`.

Returns the same NamedTuple shape as `tier_c_sod_full_ic`.
"""
function tier_c_cold_sinusoid_full_ic(; level::Integer = 4,
                                       A::Real = 0.5,
                                       k = (1, 0),
                                       ρ0::Real = 1.0, P0::Real = 1.0,
                                       lo = nothing, hi = nothing,
                                       quad_order::Integer = 3,
                                       Gamma::Real = GAMMA_LAW_DEFAULT,
                                       cv::Real = CV_DEFAULT,
                                       T::Type = Float64)
    lo_t, hi_t = _default_box(2, lo, hi, T;
                                default_lo = (zero(T), zero(T)),
                                default_hi = (one(T), one(T)))
    k_t = (Int(k[1]), Int(k[2]))

    mesh = HierarchicalMesh{2}(; balanced = true)
    for _ in 1:level
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, lo_t, hi_t)
    fields = allocate_cholesky_2d_fields(mesh; T = T)

    L1 = hi_t[1] - lo_t[1]
    L2 = hi_t[2] - lo_t[2]
    ρ_per_cell = fill(Float64(ρ0), length(leaves))
    @inbounds for (j, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        cx = 0.5 * (lo_c[1] + hi_c[1])
        cy = 0.5 * (lo_c[2] + hi_c[2])
        u_x = k_t[1] == 0 ? 0.0 : Float64(A) * sin(2π * k_t[1] * (cx - lo_t[1]) / L1)
        u_y = k_t[2] == 0 ? 0.0 : Float64(A) * sin(2π * k_t[2] * (cy - lo_t[2]) / L2)
        v = cholesky_sector_state_from_primitive(Float64(ρ0), u_x, u_y, Float64(P0),
                                                   (cx, cy);
                                                   Gamma = Gamma, cv = cv)
        write_detfield_2d!(fields, ci, v)
    end

    return (
        name = "tier_c_cold_sinusoid_full",
        mesh = mesh, frame = frame, leaves = leaves,
        fields = fields,
        ρ_per_cell = ρ_per_cell,
        params = (level = level, A = A, k = k_t, ρ0 = ρ0, P0 = P0,
                   lo = lo_t, hi = hi_t,
                   quad_order = quad_order, Gamma = Gamma, cv = cv),
    )
end

"""
    tier_c_plane_wave_full_ic(; level=4, A=1e-3, k_mag=1, angle=0.0,
                              ρ0=1.0, P0=1.0, c=nothing,
                              lo=nothing, hi=nothing,
                              quad_order=3,
                              Gamma=GAMMA_LAW_DEFAULT, cv=CV_DEFAULT,
                              T=Float64)
        -> NamedTuple

C.3 2D acoustic plane-wave full Cholesky-sector IC at angle `angle`
(radians) to the x-axis. Builds a balanced quadtree, allocates the
12-field 2D Cholesky-sector field set, and populates per-leaf via the
IC bridge. The IC seeds a single right-going acoustic mode about
`(ρ0, 0, P0)`:

    δρ(x) = A · cos(2π · k · x / L)
    δu_d(x) = (c_s / ρ0) · δρ(x) · k̂_d
    δp(x) = c_s² · δρ(x)

with `k = k_mag · (cos θ, sin θ)`. The sound-speed `c_s` defaults to
`sqrt(Γ · P0/ρ0)` (γ-law); pass an explicit `c` to override.

Returns the same NamedTuple shape as the other full-IC factories. The
`ρ_per_cell` is the cell-by-cell perturbed density `ρ0 + δρ(cell-center)`,
suitable for the residual's pressure stencil under the Newton driver.
"""
function tier_c_plane_wave_full_ic(; level::Integer = 4,
                                    A::Real = 1e-3,
                                    k_mag::Real = 1,
                                    angle::Real = 0.0,
                                    ρ0::Real = 1.0, P0::Real = 1.0,
                                    c::Union{Real,Nothing} = nothing,
                                    lo = nothing, hi = nothing,
                                    quad_order::Integer = 3,
                                    Gamma::Real = GAMMA_LAW_DEFAULT,
                                    cv::Real = CV_DEFAULT,
                                    T::Type = Float64)
    lo_t, hi_t = _default_box(2, lo, hi, T;
                                default_lo = (zero(T), zero(T)),
                                default_hi = (one(T), one(T)))

    mesh = HierarchicalMesh{2}(; balanced = true)
    for _ in 1:level
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, lo_t, hi_t)
    fields = allocate_cholesky_2d_fields(mesh; T = T)

    L1 = hi_t[1] - lo_t[1]
    θ = Float64(angle)
    k_phys = (Float64(k_mag) * cos(θ) / L1, Float64(k_mag) * sin(θ) / L1)
    kmag_phys = sqrt(k_phys[1]^2 + k_phys[2]^2)
    k̂ = kmag_phys > 0 ? (k_phys[1] / kmag_phys, k_phys[2] / kmag_phys) : (0.0, 0.0)

    c_eff = c === nothing ? sqrt(Float64(Gamma) * Float64(P0) / Float64(ρ0)) : Float64(c)

    ρ_per_cell = Vector{Float64}(undef, length(leaves))
    @inbounds for (j, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        cx = 0.5 * (lo_c[1] + hi_c[1])
        cy = 0.5 * (lo_c[2] + hi_c[2])
        phase = k_phys[1] * (cx - lo_t[1]) + k_phys[2] * (cy - lo_t[2])
        δρ = Float64(A) * cos(2π * phase)
        ρ_i = Float64(ρ0) + δρ
        P_i = Float64(P0) + c_eff^2 * δρ
        u_x = (c_eff / Float64(ρ0)) * δρ * k̂[1]
        u_y = (c_eff / Float64(ρ0)) * δρ * k̂[2]
        v = cholesky_sector_state_from_primitive(ρ_i, u_x, u_y, P_i,
                                                   (cx, cy);
                                                   Gamma = Gamma, cv = cv)
        write_detfield_2d!(fields, ci, v)
        ρ_per_cell[j] = ρ_i
    end

    return (
        name = "tier_c_plane_wave_full",
        mesh = mesh, frame = frame, leaves = leaves,
        fields = fields,
        ρ_per_cell = ρ_per_cell,
        params = (level = level, A = A, k_mag = k_mag, angle = angle,
                   ρ0 = ρ0, P0 = P0, c = c_eff,
                   k = k_phys, k̂ = k̂,
                   lo = lo_t, hi = hi_t,
                   quad_order = quad_order, Gamma = Gamma, cv = cv),
    )
end

# ─────────────────────────────────────────────────────────────────────
# M3-6 Phase 1b — D.1 Kelvin–Helmholtz IC factory
# ─────────────────────────────────────────────────────────────────────
#
# Tier-D D.1 Kelvin–Helmholtz initial condition. Sheared base flow
#
#     u_1(y) = U_jet · tanh((y − y_0) / w)        (axis-1 streamwise)
#     u_2(x, y) = 0                                (no transverse base flow)
#
# overlaid with a small-amplitude antisymmetric tilt-mode perturbation
# in the off-diagonal Cholesky factors:
#
#     δβ_12(x, y) = perturbation_amp · sin(2π k_x · (x − lo_x) / L_x)
#                                    · sech²((y − y_0) / w)
#     δβ_21(x, y) = − δβ_12(x, y)                  (antisymmetric)
#
# The antisymmetric tilt mode `(δβ_12 − δβ_21)` is the linearised
# eigenmode that the off-diagonal Berry sector drives via the Phase
# 1a strain coupling (`H_rot^off ∝ G̃_12 · (α_1·β_21 + α_2·β_12)/2`)
# — see CHECK 9 of `scripts/verify_berry_connection_offdiag.py` for the
# linearisation sketch and §7.6 of
# `reference/notes_M3_phase0_berry_connection.md` for the kinematic
# argument.
#
# Boundary conditions: PERIODIC along axis 1 (streamwise; the
# streamwise wavenumber `perturbation_k` selects the wavelength that
# fits the box) and REFLECTING along axis 2 (the shear layer is
# bounded between two walls; the velocity vanishes asymptotically as
# y → 0, 1 only when y_0 is centred and `jet_width ≪ 1`, but
# REFLECTING walls are still the correct closure for the bounded
# domain test).

"""
    tier_d_kh_ic(; level=4, U_jet=1.0, jet_width=0.1,
                 perturbation_amp=1e-3, perturbation_k=2,
                 y_0=nothing, ρ0=1.0, P0=1.0,
                 lo=nothing, hi=nothing,
                 quad_order=3, T=Float64)
        -> NamedTuple

D.1 Kelvin–Helmholtz primitive IC. Builds a balanced 2D mesh on
`[lo, hi]` (default `[0, 1]²`), evaluates the sheared base flow
`u_1(y) = U_jet · tanh((y − y_0) / jet_width)`, `u_2 = 0` at cell
centres (analytic, NOT projected — the IC is smooth so the cell
average and the centre value agree to `O((Δy / w)²)`; we use centre
sampling for reproducibility against the perturbation overlay), and
returns a primitive Tier-C-style field set `(rho, ux, uy, P)` plus
two extra named fields `(δβ_12, δβ_21)` carrying the antisymmetric
tilt-mode perturbation.

Returns a NamedTuple with `(name, mesh, frame, fields, δβ_12, δβ_21,
params)`:

  • `fields::PolynomialFieldSet` — primitive `(rho, ux, uy, P)` cell
    averages.
  • `δβ_12::Vector{Float64}`, `δβ_21::Vector{Float64}` — leaf-major
    antisymmetric off-diag β perturbation samples (cell-centre values).
  • `params::NamedTuple` — IC parameters echo for downstream tests.

# Boundary conditions

The KH IC is intended for a periodic / reflecting BC mix:

    bc_kh = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                (REFLECTING, REFLECTING)))

The streamwise wavelength `λ_x = L_x / perturbation_k` fits the box
exactly when `perturbation_k` is a positive integer; the resulting
sin-mode perturbation closes back on itself across the seam.

# 1D parity / dimension lift

The KH IC is intrinsically 2D; there is no 1D parity. At
`perturbation_amp = 0` the IC reduces to a steady sheared base flow
(the M3-3c regression configurations stay byte-equal — the sheared u_1
fires the strain stencil but with β_12 = β_21 = 0 IC, F^β_12 / F^β_21
drives are bounded by `(U_jet/w)/2` and Phase 1a's residual reduces to
M3-3c at the iso-rest slice).
"""
function tier_d_kh_ic(; level::Integer = 4,
                      U_jet::Real = 1.0,
                      jet_width::Real = 0.1,
                      perturbation_amp::Real = 1e-3,
                      perturbation_k::Integer = 2,
                      y_0::Union{Real,Nothing} = nothing,
                      ρ0::Real = 1.0, P0::Real = 1.0,
                      lo = nothing, hi = nothing,
                      quad_order::Integer = 3,
                      T::Type = Float64)
    lo_t, hi_t = _default_box(2, lo, hi, T;
                               default_lo = (zero(T), zero(T)),
                               default_hi = (one(T), one(T)))

    L1 = hi_t[1] - lo_t[1]
    L2 = hi_t[2] - lo_t[2]
    y0_t = y_0 === nothing ? T(0.5) * (lo_t[2] + hi_t[2]) : T(y_0)
    w_t  = T(jet_width)
    U_t  = T(U_jet)
    A_t  = T(perturbation_amp)
    k_x  = Int(perturbation_k)

    mesh = uniform_eulerian_mesh(2, level)
    frame = EulerianFrame(mesh, lo_t, hi_t)
    leaves = enumerate_leaves(mesh)
    n = length(leaves)
    fields = _allocate_tierc_fields(2, n; T = T)

    basis = MonomialBasis{2, 0}()
    nc = HierarchicalGrids.n_cells(mesh)
    hg_qo = _hg_quadrature_order(quad_order)

    ρ_const = T(ρ0)
    P_const = T(P0)

    # Project base-flow primitives onto leaves.
    _project_scalar_to_leaves!(fields.rho, frame, leaves,
                                _ -> ρ_const, basis, nc, hg_qo)
    _project_scalar_to_leaves!(fields.P, frame, leaves,
                                _ -> P_const, basis, nc, hg_qo)
    # u_1(y) = U_jet · tanh((y − y_0) / w); u_2 = 0.
    f_u1 = let U = U_t, y0 = y0_t, w = w_t
        x -> U * tanh((x[2] - y0) / w)
    end
    _project_scalar_to_leaves!(fields.ux, frame, leaves, f_u1, basis, nc, hg_qo)
    _project_scalar_to_leaves!(fields.uy, frame, leaves,
                                _ -> zero(T), basis, nc, hg_qo)

    # Per-leaf perturbation samples (centre-evaluated).
    δβ_12 = Vector{Float64}(undef, n)
    δβ_21 = Vector{Float64}(undef, n)
    @inbounds for (j, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        cx = 0.5 * (lo_c[1] + hi_c[1])
        cy = 0.5 * (lo_c[2] + hi_c[2])
        # δβ_12 = A · sin(2π k_x (x − lo_x) / L_x) · sech²((y − y_0) / w)
        sech_arg = (cy - Float64(y0_t)) / Float64(w_t)
        sech_val = 1.0 / cosh(sech_arg)
        sech2 = sech_val * sech_val
        sin_phase = sin(2π * Float64(k_x) * (cx - Float64(lo_t[1])) / Float64(L1))
        δβ_12[j] = Float64(A_t) * sin_phase * sech2
        δβ_21[j] = -δβ_12[j]   # antisymmetric tilt mode
    end

    return (
        name = "tier_d_kh",
        mesh = mesh, frame = frame, fields = fields,
        δβ_12 = δβ_12, δβ_21 = δβ_21,
        params = (level = level, U_jet = U_jet, jet_width = jet_width,
                   perturbation_amp = perturbation_amp,
                   perturbation_k = perturbation_k,
                   y_0 = Float64(y0_t),
                   ρ0 = ρ0, P0 = P0,
                   lo = lo_t, hi = hi_t,
                   L1 = Float64(L1), L2 = Float64(L2),
                   quad_order = quad_order),
    )
end

"""
    tier_d_kh_ic_full(; level=4, U_jet=1.0, jet_width=0.1,
                       perturbation_amp=1e-3, perturbation_k=2,
                       y_0=nothing, ρ0=1.0, P0=1.0,
                       lo=nothing, hi=nothing, quad_order=3,
                       Gamma=GAMMA_LAW_DEFAULT, cv=CV_DEFAULT,
                       T=Float64)
        -> NamedTuple

D.1 Kelvin–Helmholtz Cholesky-sector full IC. Allocates a 14-named-
field 2D Cholesky-sector field set on a balanced HG quadtree, applies
the IC bridge `cholesky_sector_state_from_primitive` per leaf, then
overlays the antisymmetric tilt-mode perturbation in the off-diagonal
β slots (`β_12 = δβ_12`, `β_21 = −δβ_12`).

Returns a NamedTuple shape matching `tier_c_*_full_ic`:

  • `name = "tier_d_kh_full"`
  • `mesh, frame, leaves, fields` — Cholesky-sector field set ready
    for `det_step_2d_berry_HG!`.
  • `ρ_per_cell::Vector{Float64}` — uniform `fill(ρ0, N)`.
  • `params::NamedTuple` — IC parameters echo plus `(L1, L2, y_0)`.

# Boundary conditions

The recommended BC mix is

    bc_kh = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                (REFLECTING, REFLECTING)))

(periodic streamwise, reflecting transverse). The factory does not
attach a BC; the caller selects it.

# 1D parity / dimension lift

At `perturbation_amp = 0`, the off-diag β slots are exactly zero per
leaf. The Phase 1a strain stencil fires off the sheared base flow
(non-zero G̃_12 in the shear layer), driving β_12 / β_21 over time, but
the IC itself is byte-equal in the off-diag slot to the M3-6 Phase 0
base case. With `perturbation_amp ≠ 0` the IC seeds the antisymmetric
tilt mode that the Phase 1c Drazin–Reid calibration measures.
"""
function tier_d_kh_ic_full(; level::Integer = 4,
                            U_jet::Real = 1.0,
                            jet_width::Real = 0.1,
                            perturbation_amp::Real = 1e-3,
                            perturbation_k::Integer = 2,
                            y_0::Union{Real,Nothing} = nothing,
                            ρ0::Real = 1.0, P0::Real = 1.0,
                            lo = nothing, hi = nothing,
                            quad_order::Integer = 3,
                            Gamma::Real = GAMMA_LAW_DEFAULT,
                            cv::Real = CV_DEFAULT,
                            T::Type = Float64)
    lo_t, hi_t = _default_box(2, lo, hi, T;
                               default_lo = (zero(T), zero(T)),
                               default_hi = (one(T), one(T)))
    L1 = hi_t[1] - lo_t[1]
    L2 = hi_t[2] - lo_t[2]
    y0_t = y_0 === nothing ? T(0.5) * (lo_t[2] + hi_t[2]) : T(y_0)
    w_t  = T(jet_width)
    U_t  = T(U_jet)
    A_t  = T(perturbation_amp)
    k_x  = Int(perturbation_k)

    mesh = HierarchicalMesh{2}(; balanced = true)
    for _ in 1:level
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, lo_t, hi_t)
    fields = allocate_cholesky_2d_fields(mesh; T = T)

    ρ_per_cell = fill(Float64(ρ0), length(leaves))
    @inbounds for (j, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        cx = 0.5 * (lo_c[1] + hi_c[1])
        cy = 0.5 * (lo_c[2] + hi_c[2])
        # Sheared base flow at cell centre.
        u1 = Float64(U_t) * tanh((cy - Float64(y0_t)) / Float64(w_t))
        u2 = 0.0
        # Apply the IC bridge to get the cold-limit Cholesky-sector
        # state with α = 1, β = 0, θ_R = 0, s = s(ρ, P).
        v_base = cholesky_sector_state_from_primitive(Float64(ρ0), u1, u2,
                                                        Float64(P0),
                                                        (cx, cy);
                                                        Gamma = Gamma, cv = cv)
        # Overlay the antisymmetric tilt-mode perturbation in
        # off-diagonal β. δβ_21 = −δβ_12.
        sech_arg = (cy - Float64(y0_t)) / Float64(w_t)
        sech_val = 1.0 / cosh(sech_arg)
        sech2 = sech_val * sech_val
        sin_phase = sin(2π * Float64(k_x) * (cx - Float64(lo_t[1])) / Float64(L1))
        δβ12 = Float64(A_t) * sin_phase * sech2
        v = DetField2D{Float64}(
            v_base.x, v_base.u, v_base.alphas, v_base.betas,
            (δβ12, -δβ12),
            v_base.θ_R, v_base.s, v_base.Pp, v_base.Q,
        )
        write_detfield_2d!(fields, ci, v)
    end

    return (
        name = "tier_d_kh_full",
        mesh = mesh, frame = frame, leaves = leaves,
        fields = fields,
        ρ_per_cell = ρ_per_cell,
        params = (level = level, U_jet = U_jet, jet_width = jet_width,
                   perturbation_amp = perturbation_amp,
                   perturbation_k = perturbation_k,
                   y_0 = Float64(y0_t),
                   ρ0 = ρ0, P0 = P0,
                   lo = lo_t, hi = hi_t,
                   L1 = Float64(L1), L2 = Float64(L2),
                   quad_order = quad_order, Gamma = Gamma, cv = cv),
    )
end

# ─────────────────────────────────────────────────────────────────────
# M3-6 Phase 2: D.4 Zel'dovich pancake IC factory
# ─────────────────────────────────────────────────────────────────────
#
# Cosmological reference test: a 1D-symmetric sinusoidal velocity
# perturbation along axis 1 (the collapsing axis); axis 2 is trivial.
# In the Eulerian frame, the cell-centred initial condition is
#
#     u_1(x) = -A · 2π · cos(2π (x_1 - lo_1) / L_1)
#     u_2 = 0
#
# In Lagrangian coordinates the trajectory is
#
#     x_1(m_1, t) = m_1 + A · sin(2π m_1) · (1 - t / t_cross)... + ...
#
# but in Eulerian-frame IC sampling we just store the cell-centred velocity.
# The cold-limit pressure floor is small but non-zero (`P0 ≈ 1e-6`) — the
# Cholesky-sector residual needs a positive `P` to set `s` via the EOS.
#
# Caustic time:  t_cross = 1 / (A · 2π) (1D Zel'dovich; pancake forms at
# `m_1 = 0` mod `L_1`).
#
# The methods paper §10.5 D.4 prediction: per-axis γ correctly identifies
# the collapsing axis. γ_1 develops spatial structure (drops sharply at
# the compressive trough as t → t_cross) while γ_2 stays uniform.

"""
    tier_d_zeldovich_pancake_ic(; level=5, A=0.5,
                                  ρ0=1.0, P0=1e-6,
                                  lo=nothing, hi=nothing,
                                  Gamma=GAMMA_LAW_DEFAULT, cv=CV_DEFAULT,
                                  T=Float64) -> NamedTuple

D.4 Zel'dovich pancake collapse Cholesky-sector full IC.

Builds a balanced HG quadtree on `[lo, hi]` (default `[0, 1]²`),
allocates the 14-named-field 2D Cholesky-sector field set, applies
`cholesky_sector_state_from_primitive` per leaf with the Zel'dovich
velocity profile

    u_1(x) = -A · 2π · cos(2π (x_1 - lo_1) / L_1)
    u_2    = 0

uniform `(ρ0, P0)`, cold-limit α = 1, β = 0, β_off = 0, θ_R = 0.

The caustic forms at `t_cross = 1 / (A · 2π)`; with `A = 0.5` we get
`t_cross ≈ 0.318`.

Returns a NamedTuple shape matching `tier_c_*_full_ic`:

  • `name = "tier_d_zeldovich_pancake"`
  • `mesh, frame, leaves, fields` — Cholesky-sector field set ready
    for `det_step_2d_berry_HG!`.
  • `ρ_per_cell::Vector{Float64}` — uniform `fill(ρ0, N)`.
  • `t_cross::Float64` — analytic caustic time.
  • `params::NamedTuple` — IC parameters echo plus `(L1, L2, t_cross)`.

# Boundary conditions

The recommended BC mix is

    bc_zel = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                  (REFLECTING, REFLECTING)))

(periodic along the collapsing axis, reflecting along the trivial axis).
The factory does not attach a BC; the caller selects it.

# 1D parity / dimension lift

The IC is 1D-symmetric: u_2 = 0 strictly, and all axis-2 cells share the
same axis-1-aligned velocity profile. The Phase 1a strain coupling stays
*inert* at IC (∂_2 u_1 = 0 and ∂_1 u_2 = 0 at every face), so the
off-diagonal β slots remain zero throughout the run. This is a clean
cross-check that Phase 1a's H_rot^off coupling does not fire on
axis-aligned ICs.
"""
function tier_d_zeldovich_pancake_ic(; level::Integer = 5,
                                       A::Real = 0.5,
                                       ρ0::Real = 1.0,
                                       P0::Real = 1e-6,
                                       lo = nothing, hi = nothing,
                                       Gamma::Real = GAMMA_LAW_DEFAULT,
                                       cv::Real = CV_DEFAULT,
                                       T::Type = Float64)
    lo_t, hi_t = _default_box(2, lo, hi, T;
                                default_lo = (zero(T), zero(T)),
                                default_hi = (one(T), one(T)))
    L1 = hi_t[1] - lo_t[1]
    L2 = hi_t[2] - lo_t[2]
    A_t = T(A)

    mesh = HierarchicalMesh{2}(; balanced = true)
    for _ in 1:level
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, lo_t, hi_t)
    fields = allocate_cholesky_2d_fields(mesh; T = T)

    ρ_per_cell = fill(Float64(ρ0), length(leaves))
    @inbounds for (j, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        cx = 0.5 * (lo_c[1] + hi_c[1])
        cy = 0.5 * (lo_c[2] + hi_c[2])
        # Zel'dovich velocity: -A·2π·cos(2π m_1) along axis 1.
        m1_norm = (cx - Float64(lo_t[1])) / Float64(L1)
        u1 = -Float64(A_t) * 2π * cos(2π * m1_norm)
        u2 = 0.0
        v = cholesky_sector_state_from_primitive(Float64(ρ0), u1, u2,
                                                   Float64(P0),
                                                   (cx, cy);
                                                   Gamma = Gamma, cv = cv)
        write_detfield_2d!(fields, ci, v)
    end

    t_cross = 1.0 / (Float64(A_t) * 2π)

    return (
        name = "tier_d_zeldovich_pancake",
        mesh = mesh, frame = frame, leaves = leaves,
        fields = fields,
        ρ_per_cell = ρ_per_cell,
        t_cross = t_cross,
        params = (level = level, A = Float64(A_t),
                   ρ0 = ρ0, P0 = P0,
                   lo = lo_t, hi = hi_t,
                   L1 = Float64(L1), L2 = Float64(L2),
                   t_cross = t_cross,
                   Gamma = Gamma, cv = cv),
    )
end

# ─────────────────────────────────────────────────────────────────────
# M3-6 Phase 4: D.7 dust-traps IC factory (Taylor-Green vortex + dust)
# ─────────────────────────────────────────────────────────────────────
#
# §10.5 D.7 dust-trapping in vortices. The methods-paper test calls
# for "KH instability with passive dust; vortex-center accumulation
# matches reference codes". We use the simpler, more robust
# Taylor-Green vortex IC (single periodic vortex array on `[0, 1]²`)
# rather than a sheared KH base flow:
#
#     u_1(x, y) = U0 · sin(2π m_1) · cos(2π m_2)
#     u_2(x, y) = -U0 · cos(2π m_1) · sin(2π m_2)
#
# which is divergence-free (∇·u = 0 at IC) and creates a 2×2 array of
# counter-rotating vortices on the unit square. Vortex centres sit at
# `(m_1, m_2) ∈ {(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)}`
# (4 cell-centre positions on the unit cube where `|u| = U0`); hyperbolic
# stagnation points sit at the midpoints of the cell edges
# `(m_1, m_2) ∈ {(0, 0), (0.5, 0.5), …}` (where `u = 0` and the strain
# is hyperbolic).
#
# The two-species substrate (Phase 3 `TracerMeshHG2D`) carries:
#  • Gas species (`:gas`, k = 1): uniform concentration `c_gas = 1.0`.
#    Tracks the gas phase. Ignored for the variational fluid state
#    (the fluid evolves via `det_step_2d_berry_HG!` independently of the
#    tracer matrix).
#  • Dust species (`:dust`, k = 2): IC seeded with a small spatially-
#    varying perturbation `c_dust(x, y) = 1 + ε · sin(2π m_1) sin(2π m_2)`.
#    Centred at `c_dust ≈ 1`, with peaks/troughs of amplitude `ε`
#    aligned with the vortex centres. Under pure-Lagrangian advection
#    in the dfmm 2D variational scheme, the dust concentration matrix
#    is byte-stable per `det_step_2d_berry_HG!` call (Phase 3 contract)
#    — total integrated dust mass conserved bit-exactly through any
#    number of steps.
#
# Per-species `M_vv` for the diagnostic γ:
#  • Gas: EOS-derived `Mvv(J, s)` — finite, isotropic, β-cone full-rank
#    in the limit β → 0.
#  • Dust: `M_vv = 0` for both axes (pressureless cold dust). γ_a = 0
#    everywhere (β > 0 ⇒ γ²_a < 0 ⇒ floored). Distinguishes the two
#    phases at the diagnostic level.
#
# The IC factory returns an `ic` NamedTuple compatible with the other
# factories (`fields`, `mesh`, `frame`, `leaves`, `ρ_per_cell`,
# `params`) plus a `tm::TracerMeshHG2D` field holding the per-species
# tracer matrix populated with the gas + dust IC concentrations.

"""
    tier_d_dust_trap_ic_full(; level=4, U0=1.0, ρ0=1.0, P0=1.0,
                              ε_dust=0.05,
                              lo=nothing, hi=nothing,
                              Gamma=GAMMA_LAW_DEFAULT, cv=CV_DEFAULT,
                              T=Float64) -> NamedTuple

D.7 dust-traps Cholesky-sector full IC (Taylor-Green vortex + 2-species
dust + gas tracer mesh).

Builds a balanced HG quadtree on `[lo, hi]` (default `[0, 1]²`),
allocates the 14-named-field 2D Cholesky-sector field set, applies
`cholesky_sector_state_from_primitive` per leaf with the Taylor-Green
velocity profile

    u_1(x, y) = U0 · sin(2π m_1) · cos(2π m_2)
    u_2(x, y) = -U0 · cos(2π m_1) · sin(2π m_2)

uniform `(ρ0, P0)`, cold-limit α = 1, β = 0, β_off = 0, θ_R = 0.

Also allocates a 2-species `TracerMeshHG2D` with:
  • Species 1 `:gas` — uniform `c_gas = 1.0` per leaf.
  • Species 2 `:dust` — `c_dust(x, y) = 1 + ε_dust · sin(2π m_1) ·
    sin(2π m_2)`. The amplitude `ε_dust` is small (default 5%) so
    vortex-center accumulation produces a measurable but non-trivial
    response under any sub-cell drift.

Returns a NamedTuple shape matching `tier_c_*_full_ic` plus the new
`tm::TracerMeshHG2D` field:

  • `name = "tier_d_dust_trap"`
  • `mesh, frame, leaves, fields` — Cholesky-sector field set ready
    for `det_step_2d_berry_HG!`.
  • `tm::TracerMeshHG2D{Float64}` — 2-species tracer mesh with
    `(:gas, :dust)` initialised at the cell-centred concentrations.
  • `ρ_per_cell::Vector{Float64}` — uniform `fill(ρ0, N)`.
  • `params::NamedTuple` — IC parameters echo plus `(L1, L2, t_eddy)`
    where `t_eddy = L1 / U0` is the vortex eddy turnover time.

# Boundary conditions

The recommended BC mix is

    bc_dust = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                   (PERIODIC, PERIODIC)))

(periodic on both axes — the Taylor-Green vortex is doubly periodic).

# Per-species M_vv for γ diagnostic

Use:

    M_vv_per_species = ((1.0, 1.0),    # gas: EOS reference (or :nothing)
                         (0.0, 0.0))    # dust: pressureless cold

Then `gamma_per_axis_2d_per_species_field(fields, leaves;
M_vv_override_per_species=M_vv_per_species, n_species=2)` distinguishes
the two phases (γ_dust → 0 everywhere; γ_gas finite).

# Honest scientific note

The dfmm 2D variational scheme under `det_step_2d_berry_HG!` evolves
the fluid state but `advect_tracers_HG_2d!` is a no-op (pure-Lagrangian
frame; the tracer matrix is byte-stable per step — Phase 3 contract).
This means the per-cell dust concentration *cannot* drift toward
vortex centres in the current substrate — sub-cell centrifugal-drift
physics is not captured. The Phase 4 acceptance gate is
correspondingly modest: we verify (i) total dust mass is conserved
bit-exactly, (ii) per-species γ correctly distinguishes phases, and
(iii) the 4-component cone is respected throughout. Dust accumulation
toward vortex centres is reported as a *diagnostic* — if measured
non-zero, that's the variational scheme's substrate response under
finite-volume effects; if measured zero, that's the structural
contract of the Phase 3 substrate.
"""
function tier_d_dust_trap_ic_full(; level::Integer = 4,
                                    U0::Real = 1.0,
                                    ρ0::Real = 1.0,
                                    P0::Real = 1.0,
                                    ε_dust::Real = 0.05,
                                    lo = nothing, hi = nothing,
                                    Gamma::Real = GAMMA_LAW_DEFAULT,
                                    cv::Real = CV_DEFAULT,
                                    T::Type = Float64)
    lo_t, hi_t = _default_box(2, lo, hi, T;
                                default_lo = (zero(T), zero(T)),
                                default_hi = (one(T), one(T)))
    L1 = hi_t[1] - lo_t[1]
    L2 = hi_t[2] - lo_t[2]
    U_t = T(U0)
    ε_t = T(ε_dust)

    mesh = HierarchicalMesh{2}(; balanced = true)
    for _ in 1:level
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, lo_t, hi_t)
    fields = allocate_cholesky_2d_fields(mesh; T = T)

    ρ_per_cell = fill(Float64(ρ0), length(leaves))
    # Pre-compute per-leaf cell centres + the IC concentration.
    c_gas_vec = fill(1.0, length(leaves))
    c_dust_vec = Vector{Float64}(undef, length(leaves))
    @inbounds for (j, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        cx = 0.5 * (lo_c[1] + hi_c[1])
        cy = 0.5 * (lo_c[2] + hi_c[2])
        m1 = (cx - Float64(lo_t[1])) / Float64(L1)
        m2 = (cy - Float64(lo_t[2])) / Float64(L2)
        # Taylor-Green vortex velocity.
        u1 = Float64(U_t) * sin(2π * m1) * cos(2π * m2)
        u2 = -Float64(U_t) * cos(2π * m1) * sin(2π * m2)
        v = cholesky_sector_state_from_primitive(Float64(ρ0), u1, u2,
                                                   Float64(P0),
                                                   (cx, cy);
                                                   Gamma = Gamma, cv = cv)
        write_detfield_2d!(fields, ci, v)
        # Dust concentration: 1 + ε · sin(2π m_1) · sin(2π m_2).
        # Peaks at vortex centres (1+ε at (0.25, 0.25), (0.75, 0.75));
        # troughs at the other vortex pair (1−ε at (0.25, 0.75),
        # (0.75, 0.25)). Hyperbolic stagnation points have c ≈ 1.
        c_dust_vec[j] = 1.0 + Float64(ε_t) * sin(2π * m1) * sin(2π * m2)
    end

    # Eddy turnover time t_eddy = L / U0. (For unit box, equals 1/U0.)
    t_eddy = Float64(L1) / max(Float64(U_t), 1e-300)

    # Allocate the 2-species TracerMeshHG2D and populate.
    tm = TracerMeshHG2D(fields, mesh; n_species = 2,
                         names = [:gas, :dust], T = Float64)
    set_species!(tm, :gas, c_gas_vec, leaves)
    set_species!(tm, :dust, c_dust_vec, leaves)

    return (
        name = "tier_d_dust_trap",
        mesh = mesh, frame = frame, leaves = leaves,
        fields = fields,
        tm = tm,
        ρ_per_cell = ρ_per_cell,
        t_eddy = t_eddy,
        params = (level = level, U0 = Float64(U_t),
                   ρ0 = ρ0, P0 = P0,
                   ε_dust = Float64(ε_t),
                   lo = lo_t, hi = hi_t,
                   L1 = Float64(L1), L2 = Float64(L2),
                   t_eddy = t_eddy,
                   Gamma = Gamma, cv = cv),
    )
end

# ─────────────────────────────────────────────────────────────────────
# M3-6 Phase 5: D.10 ISM-like 2D multi-tracer IC factory
# ─────────────────────────────────────────────────────────────────────
#
# Methods paper §10.5 D.10 community-impact test: "Multi-tracer 2D
# shocked turbulence; metallicity PDF and spatial structure consistent
# with Lagrangian methods (SPH, AREPO); significantly better than
# Eulerian methods." The dfmm 2D variational scheme tests this via a
# multi-phase ISM-like IC carrying N_species ≥ 3 passive scalars on a
# sheared KH base flow, then driven through the 2D Newton step
# (`det_step_2d_berry_HG!`) optionally combined with stochastic
# injection (`inject_vg_noise_HG_2d!`) for the "shocked turbulence"
# regime. The headline result is multi-tracer fidelity: the tracer
# matrix stays *byte-stable* through K stochastic-injection +
# deterministic-step iterations because (i) `advect_tracers_HG_2d!`
# is a no-op (Phase 3 design contract — pure-Lagrangian frame) and
# (ii) `inject_vg_noise_HG_2d!`'s write set is a strict subset of the
# fluid Cholesky-sector fields (excludes `tm.tracers` by inspection;
# 2D analog of the M2-2 1D structural argument).

"""
    tier_d_ism_tracers_ic_full(; level=4, n_species=3,
                                 U_jet=1.0, jet_width=0.1,
                                 perturbation_amp=1e-3, perturbation_k=2,
                                 y_0=nothing,
                                 ρ0=1.0, P0=1.0,
                                 lo=nothing, hi=nothing,
                                 species_names=nothing,
                                 species_widths=nothing,
                                 species_centres=nothing,
                                 ε_phase=0.02,
                                 Gamma=GAMMA_LAW_DEFAULT, cv=CV_DEFAULT,
                                 T=Float64) -> NamedTuple

D.10 ISM-like multi-tracer Cholesky-sector full IC.

Builds a balanced HG quadtree on `[lo, hi]` (default `[0, 1]²`),
allocates the 14-named-field 2D Cholesky-sector field set, applies
`cholesky_sector_state_from_primitive` per leaf with the KH-style
sheared base flow

    u_1(y) = U_jet · tanh((y - y_0) / w)
    u_2    = 0

uniform `(ρ0, P0)`, cold-limit `α = 1, β = 0, β_off = 0, θ_R = 0`,
plus an antisymmetric tilt-mode perturbation `δβ_12 = -δβ_21 = A ·
sin(2π k_x x) · sech²((y - y_0)/w)` (the M3-6 Phase 1b stencil).

Allocates an `n_species`-row `TracerMeshHG2D` carrying ISM-like
phase-tagged tracers. The default 3-species recipe places one phase
in each y-band:

  • Species 1 `:cold`: peaked at `y < y_lo + L2/3` (cold molecular
    phase analog).
  • Species 2 `:warm`: peaked in `[y_lo + L2/3, y_lo + 2L2/3]` (warm
    neutral phase analog).
  • Species 3 `:hot`: peaked at `y > y_lo + 2L2/3` (hot ionized
    phase analog).

The per-species concentration profile is a Gaussian-windowed
metallicity-tracer-like field

    c_k(x, y) = clamp(exp(-(y - y_centre[k])² / (2·width[k]²))
                       + ε_phase · cos(2π m_1) · cos(2π m_2),
                     0, ∞)

which sums to ≈ 1 along columns (so the multi-phase ISM mixing
signature is well-defined). The small `ε_phase` term tilts the
species spatially in `m_1` to break the 1D-symmetry and drive the
"multi-phase mixing" regime under stochastic injection.

Returns a NamedTuple shape matching `tier_c_*_full_ic` plus the new
`tm::TracerMeshHG2D` field:

  • `name = "tier_d_ism_tracers"`
  • `mesh, frame, leaves, fields` — Cholesky-sector field set ready
    for `det_step_2d_berry_HG!`.
  • `tm::TracerMeshHG2D{Float64}` — `n_species`-species tracer mesh
    populated at the cell-centred concentrations.
  • `ρ_per_cell::Vector{Float64}` — uniform `fill(ρ0, N)`.
  • `params::NamedTuple` — IC parameters echo plus `(L1, L2, t_KH)`.

The KH timescale `t_KH = L1 / U_jet` is the natural simulation
horizon for the shocked-turbulence regime (multiple shears across
the box).

# Boundary conditions

The recommended BC mix is

    bc_ism = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                  (REFLECTING, REFLECTING)))

(periodic along the streamwise / shearing axis; reflecting along the
phase-stratification axis).

# Per-species M_vv for γ diagnostic

The default per-species `M_vv` recipe carries a *species-dependent
thermal velocity dispersion* analogous to the warm/hot/cold ISM
phases:

    M_vv_per_species = ((1.0, 1.0),    # cold (small dispersion)
                         (2.0, 2.0),    # warm (intermediate)
                         (4.0, 4.0))    # hot (large)

The per-species γ diagnostic via
`gamma_per_axis_2d_per_species_field(...; M_vv_override_per_species)`
then has each species' γ trajectory distinct from the others — the
qualitative "per-species γ separation" test.

# Honest scientific note

In the dfmm 2D variational scheme `advect_tracers_HG_2d!` is a no-op
(Phase 3 design contract — pure-Lagrangian frame; the tracer matrix
is byte-stable per step), AND `inject_vg_noise_HG_2d!` writes only
fluid `(β_a, u_a, s, Pp)` fields (NOT `tm.tracers`). This means the
multi-tracer matrix stays bit-identical to its IC value through any
combination of `det_step_2d_berry_HG!` + `inject_vg_noise_HG_2d!`
calls — the 2D analog of the M2-2 structural bit-exactness argument.
The Phase 5 test exercises this contract as a *defensive regression
guard* against future refactors that might accidentally put
`tm.tracers` in the write set of either operator.
"""
function tier_d_ism_tracers_ic_full(; level::Integer = 4,
                                     n_species::Integer = 3,
                                     U_jet::Real = 1.0,
                                     jet_width::Real = 0.1,
                                     perturbation_amp::Real = 1e-3,
                                     perturbation_k::Integer = 2,
                                     y_0::Union{Real,Nothing} = nothing,
                                     ρ0::Real = 1.0, P0::Real = 1.0,
                                     lo = nothing, hi = nothing,
                                     species_names = nothing,
                                     species_centres = nothing,
                                     species_widths = nothing,
                                     ε_phase::Real = 0.02,
                                     Gamma::Real = GAMMA_LAW_DEFAULT,
                                     cv::Real = CV_DEFAULT,
                                     T::Type = Float64)
    @assert n_species ≥ 1 "tier_d_ism_tracers_ic_full requires ≥ 1 species"
    lo_t, hi_t = _default_box(2, lo, hi, T;
                                default_lo = (zero(T), zero(T)),
                                default_hi = (one(T), one(T)))
    L1 = hi_t[1] - lo_t[1]
    L2 = hi_t[2] - lo_t[2]
    y0_t = y_0 === nothing ? T(0.5) * (lo_t[2] + hi_t[2]) : T(y_0)
    w_t  = T(jet_width)
    U_t  = T(U_jet)
    A_t  = T(perturbation_amp)
    k_x  = Int(perturbation_k)
    ns   = Int(n_species)

    # Default species partitioning: evenly placed centres along axis 2
    # with characteristic width L2 / (3 · ns).
    if species_centres === nothing
        species_centres = ntuple(k -> Float64(lo_t[2]) +
                                  (Float64(k) - 0.5) * Float64(L2) /
                                  Float64(ns),
                                  ns)
    else
        @assert length(species_centres) == ns "species_centres length must == n_species"
    end
    if species_widths === nothing
        species_widths = ntuple(_ -> Float64(L2) / (3.0 * Float64(ns)), ns)
    else
        @assert length(species_widths) == ns "species_widths length must == n_species"
    end
    # Default species names: :cold, :warm, :hot for ns == 3; generic
    # `:species_k` otherwise.
    if species_names === nothing
        if ns == 3
            species_names = [:cold, :warm, :hot]
        else
            species_names = [Symbol("species_$(k)") for k in 1:ns]
        end
    else
        @assert length(species_names) == ns "species_names length must == n_species"
    end

    mesh = HierarchicalMesh{2}(; balanced = true)
    for _ in 1:level
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, lo_t, hi_t)
    fields = allocate_cholesky_2d_fields(mesh; T = T)

    ρ_per_cell = fill(Float64(ρ0), length(leaves))

    # Pre-compute per-leaf cell centres + per-species concentrations.
    # Layout: c_per_species[k] is a Vector{Float64} of length(leaves).
    c_per_species = [Vector{Float64}(undef, length(leaves)) for _ in 1:ns]
    @inbounds for (j, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        cx = 0.5 * (lo_c[1] + hi_c[1])
        cy = 0.5 * (lo_c[2] + hi_c[2])
        m1 = (cx - Float64(lo_t[1])) / Float64(L1)
        m2 = (cy - Float64(lo_t[2])) / Float64(L2)
        # KH base flow + cold-limit Cholesky state.
        u1 = Float64(U_t) * tanh((cy - Float64(y0_t)) / Float64(w_t))
        u2 = 0.0
        v_base = cholesky_sector_state_from_primitive(Float64(ρ0), u1, u2,
                                                        Float64(P0),
                                                        (cx, cy);
                                                        Gamma = Gamma, cv = cv)
        # Phase 1b antisymmetric tilt-mode β_off perturbation.
        sech_arg = (cy - Float64(y0_t)) / Float64(w_t)
        sech_val = 1.0 / cosh(sech_arg)
        sech2 = sech_val * sech_val
        sin_phase = sin(2π * Float64(k_x) * (cx - Float64(lo_t[1])) / Float64(L1))
        δβ12 = Float64(A_t) * sin_phase * sech2
        v = DetField2D{Float64}(
            v_base.x, v_base.u, v_base.alphas, v_base.betas,
            (δβ12, -δβ12),
            v_base.θ_R, v_base.s, v_base.Pp, v_base.Q,
        )
        write_detfield_2d!(fields, ci, v)
        # Per-species concentration: Gaussian-windowed in y +
        # ε_phase tilt term in (m_1, m_2).
        for k in 1:ns
            yc = Float64(species_centres[k])
            σk = Float64(species_widths[k])
            σ2 = max(σk * σk, 1e-300)
            gauss = exp(-((cy - yc)^2) / (2.0 * σ2))
            tilt = Float64(ε_phase) * cos(2π * m1) * cos(2π * m2)
            c_per_species[k][j] = max(gauss + tilt, 0.0)
        end
    end

    # KH simulation horizon: t_KH = L1 / U_jet.
    t_KH = Float64(L1) / max(Float64(U_t), 1e-300)

    # Allocate the n_species TracerMeshHG2D and populate.
    tm = TracerMeshHG2D(fields, mesh; n_species = ns,
                         names = collect(Symbol, species_names),
                         T = Float64)
    for k in 1:ns
        set_species!(tm, k, c_per_species[k], leaves)
    end

    return (
        name = "tier_d_ism_tracers",
        mesh = mesh, frame = frame, leaves = leaves,
        fields = fields,
        tm = tm,
        ρ_per_cell = ρ_per_cell,
        t_KH = t_KH,
        params = (level = level, n_species = ns,
                   U_jet = Float64(U_t), jet_width = Float64(w_t),
                   perturbation_amp = Float64(A_t),
                   perturbation_k = k_x,
                   y_0 = Float64(y0_t),
                   ρ0 = Float64(ρ0), P0 = Float64(P0),
                   ε_phase = Float64(ε_phase),
                   species_names = Tuple(species_names),
                   species_centres = Tuple(Float64.(species_centres)),
                   species_widths = Tuple(Float64.(species_widths)),
                   lo = lo_t, hi = hi_t,
                   L1 = Float64(L1), L2 = Float64(L2),
                   t_KH = t_KH,
                   Gamma = Gamma, cv = cv),
    )
end

# ─────────────────────────────────────────────────────────────────────
# M3-7a: 3D Cholesky-sector field-set allocation + read/write helpers
# ─────────────────────────────────────────────────────────────────────
#
# 3D analog of `allocate_cholesky_2d_fields` / `read_detfield_2d` /
# `write_detfield_2d!`. Mirrors the 2D pattern but at `MonomialBasis{3,
# 0}` (one coefficient per cell) and at the 13-Newton-unknowns + (no
# off-diagonal β + no post-Newton Pp/Q) layout per M3-7 design note §2
# and `reference/notes_M3_7_prep_3d_scaffolding.md`.
#
# Storage layout — 13 named scalar fields on the HG substrate:
#
#     :x_1, :x_2, :x_3        Lagrangian position (charge 0)
#     :u_1, :u_2, :u_3        Lagrangian velocity (charge 0)
#     :α_1, :α_2, :α_3        principal-axis Cholesky factors
#     :β_1, :β_2, :β_3        per-axis conjugate momenta (charge 1)
#     :θ_12, :θ_13, :θ_23     Berry rotation angles (Newton unknowns)
#     :s                      specific entropy (operator-split)
#
# That's 16 named fields total (13 Newton + 1 entropy). The "13 dof"
# label in the M3-7 design note counts entropy in for completeness but
# treats it as operator-split — Newton drives 15 fields per cell
# (positions + velocities + α + β + 3 angles), of which θ_12, θ_13,
# θ_23 + α + β are the *Cholesky-sector* unknowns (non-position).
#
# Off-diagonal β and Pp / Q are deferred — see `DetField3D` docstring
# for the rationale (M3-3a Q3 default + M3-7 design note §4.4).

"""
    allocate_cholesky_3d_fields(mesh::HierarchicalMesh{3}; T::Type = Float64)
        -> PolynomialFieldSet

Allocate the per-cell `PolynomialFieldSet` for the M3-7 3D Cholesky-
sector EL system (Phase M3-7a). The field set has 16 named scalar
fields laid out as

    Newton unknowns (13 + s = 16 named):
        :x_1, :x_2, :x_3      — Lagrangian position components (charge 0)
        :u_1, :u_2, :u_3      — Lagrangian velocity components (charge 0)
        :α_1, :α_2, :α_3      — principal-axis Cholesky factors
        :β_1, :β_2, :β_3      — per-axis conjugate momenta (charge 1)
        :θ_12, :θ_13, :θ_23   — Berry rotation angles (Cardan ZYX)
        :s                    — specific entropy (operator-split)

each at `MonomialBasis{3, 0}` (one coefficient per cell — the
order-0 cell-average storage that mirrors M3-3a's 2D substrate).

# Sizing

The field set is sized to `n_cells(mesh)`, NOT `n_leaves(mesh)`, so
that HG's `halo_view`-based neighbor access (`hv[i, off]`) maps
1-to-1 against `mesh.cells[i]`. The leaf-only public API of M3-7b
will iterate via `enumerate_leaves(mesh)` and use those mesh-cell
indices directly into the field set.

# Off-diagonal β and post-Newton Pp / Q — deferred

Per M3-7 design note §4.4 + M3-3a Q3 default, the 6 off-diagonal
`β_{ab}` entries (3 antisymmetric + 3 symmetric in 3D) and the per-
axis post-Newton `Pp_a`, `Q_a` (3 + 3 = 6 scalars) are NOT carried
on `DetField3D`. M3-9 (3D D.1 KH follow-up) and M3-7c (post-Newton
sector activation) will add parallel allocator constructors when
needed. The current allocator stops at the 13-Newton-unknown set
matching `n_dof_newton(::DetField3D) == 13`.

# Convention

The Euler-angle convention is the intrinsic Cardan ZYX composition
pinned in `src/cholesky_DD_3d.jl`:

    R(θ_12, θ_13, θ_23) = R_12(θ_12) · R_13(θ_13) · R_23(θ_23).

At `θ_13 = θ_23 = 0` the 3D rotation reduces to `R_12(θ_12)` byte-
equally, matching the 2D `R(θ_R)` convention from
`cholesky_recompose_2d` (the 3D 2D-symmetric ⊂ 2D dimension-lift
gate, M3-7 design note §7.1b).
"""
function allocate_cholesky_3d_fields(mesh::HierarchicalMesh{3};
                                      T::Type = Float64)
    basis = MonomialBasis{3, 0}()
    nc = HierarchicalGrids.n_cells(mesh)
    return allocate_polynomial_fields(SoA(), basis, Int(nc);
                                       x_1 = T, x_2 = T, x_3 = T,
                                       u_1 = T, u_2 = T, u_3 = T,
                                       α_1 = T, α_2 = T, α_3 = T,
                                       β_1 = T, β_2 = T, β_3 = T,
                                       θ_12 = T, θ_13 = T, θ_23 = T,
                                       s = T)
end

"""
    write_detfield_3d!(fields::PolynomialFieldSet, leaf_cell_idx::Integer,
                        v::DetField3D)

Write a `DetField3D` value into the 16-named-field 3D field set at
mesh-cell index `leaf_cell_idx`. The field set must be allocated by
`allocate_cholesky_3d_fields`. Writes 16 scalars total (13 Newton
unknowns + entropy `s`).

Inverse of `read_detfield_3d`; round-trip is bit-exact.
"""
function write_detfield_3d!(fields::PolynomialFieldSet, leaf_cell_idx::Integer,
                             v::DetField3D)
    fields.x_1[leaf_cell_idx] = (v.x[1],)
    fields.x_2[leaf_cell_idx] = (v.x[2],)
    fields.x_3[leaf_cell_idx] = (v.x[3],)
    fields.u_1[leaf_cell_idx] = (v.u[1],)
    fields.u_2[leaf_cell_idx] = (v.u[2],)
    fields.u_3[leaf_cell_idx] = (v.u[3],)
    fields.α_1[leaf_cell_idx] = (v.alphas[1],)
    fields.α_2[leaf_cell_idx] = (v.alphas[2],)
    fields.α_3[leaf_cell_idx] = (v.alphas[3],)
    fields.β_1[leaf_cell_idx] = (v.betas[1],)
    fields.β_2[leaf_cell_idx] = (v.betas[2],)
    fields.β_3[leaf_cell_idx] = (v.betas[3],)
    fields.θ_12[leaf_cell_idx] = (v.θ_12,)
    fields.θ_13[leaf_cell_idx] = (v.θ_13,)
    fields.θ_23[leaf_cell_idx] = (v.θ_23,)
    fields.s[leaf_cell_idx]   = (v.s,)
    return nothing
end

"""
    read_detfield_3d(fields::PolynomialFieldSet, leaf_cell_idx::Integer)
        -> DetField3D

Read a `DetField3D` value from the 16-named-field 3D field set at
mesh-cell index `leaf_cell_idx`. Inverse of `write_detfield_3d!`;
round-trip is bit-exact.
"""
function read_detfield_3d(fields::PolynomialFieldSet, leaf_cell_idx::Integer)
    x = (fields.x_1[leaf_cell_idx][1],
         fields.x_2[leaf_cell_idx][1],
         fields.x_3[leaf_cell_idx][1])
    u = (fields.u_1[leaf_cell_idx][1],
         fields.u_2[leaf_cell_idx][1],
         fields.u_3[leaf_cell_idx][1])
    α = (fields.α_1[leaf_cell_idx][1],
         fields.α_2[leaf_cell_idx][1],
         fields.α_3[leaf_cell_idx][1])
    β = (fields.β_1[leaf_cell_idx][1],
         fields.β_2[leaf_cell_idx][1],
         fields.β_3[leaf_cell_idx][1])
    θ12 = fields.θ_12[leaf_cell_idx][1]
    θ13 = fields.θ_13[leaf_cell_idx][1]
    θ23 = fields.θ_23[leaf_cell_idx][1]
    s   = fields.s[leaf_cell_idx][1]
    return DetField3D(x, u, α, β, θ12, θ13, θ23, s)
end

# ─────────────────────────────────────────────────────────────────────
# M3-7e: 3D Tier-C IC bridge — primitive (ρ, u_x, u_y, u_z, P) → Cholesky
# ─────────────────────────────────────────────────────────────────────
#
# 3D analog of `cholesky_sector_state_from_primitive` (the 2D bridge in
# M3-4 Phase 2). Maps a primitive `(ρ, u_x, u_y, u_z, P)` per-cell
# average plus the cell center `(x_1, x_2, x_3)` onto the M3-7 13-dof
# Cholesky-sector state. Cold-limit, isotropic IC convention:
#   • α_a = 1, β_a = 0 for all three axes.
#   • θ_12 = θ_13 = θ_23 = 0 (no off-diagonal Berry rotation).
#   • s solved from the EOS via `s_from_pressure_density`.
#
# Returns a `DetField3D` ready to write into a 3D field set allocated
# by `allocate_cholesky_3d_fields`.

"""
    cholesky_sector_state_from_primitive_3d(ρ, u_x, u_y, u_z, P, x_center;
                                             Gamma=GAMMA_LAW_DEFAULT,
                                             cv=CV_DEFAULT)
        -> DetField3D{Float64}

3D Tier-C IC bridge. Map `(ρ, u_x, u_y, u_z, P)` plus
`x_center::NTuple{3}` onto the M3-7 Cholesky-sector state. Returns a
`DetField3D` with `α_a = 1`, `β_a = 0`, `θ_ab = 0`, and `s` from EOS
inversion. Mirrors the 2D `cholesky_sector_state_from_primitive`.
"""
function cholesky_sector_state_from_primitive_3d(ρ::Real, u_x::Real, u_y::Real,
                                                   u_z::Real, P::Real,
                                                   x_center::NTuple{3,<:Real};
                                                   Gamma::Real = GAMMA_LAW_DEFAULT,
                                                   cv::Real = CV_DEFAULT)
    s = s_from_pressure_density(ρ, P; Gamma = Gamma, cv = cv)
    return DetField3D{Float64}(
        (Float64(x_center[1]), Float64(x_center[2]), Float64(x_center[3])),
        (Float64(u_x), Float64(u_y), Float64(u_z)),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
        0.0, 0.0, 0.0,
        Float64(s),
    )
end

"""
    primitive_recovery_3d(fields::PolynomialFieldSet, leaves, frame;
                          ρ_ref=1.0, Gamma=GAMMA_LAW_DEFAULT, cv=CV_DEFAULT)
        -> NamedTuple

3D analog of `primitive_recovery_2d`. Walks leaves of a 3D Cholesky-
sector field set and recovers `(ρ, u_x, u_y, u_z, P, x, y, z)` per cell
for diagnostics. Density is treated as `ρ_ref` (cold-limit IC bridge
convention; Lagrangian J-tracking is M3-9 work).
"""
function primitive_recovery_3d(fields::PolynomialFieldSet,
                                leaves::AbstractVector{<:Integer},
                                frame::EulerianFrame{3, T};
                                ρ_ref::Real = 1.0,
                                Gamma::Real = GAMMA_LAW_DEFAULT,
                                cv::Real = CV_DEFAULT) where {T}
    N = length(leaves)
    ρ_out  = Vector{Float64}(undef, N)
    u_x    = Vector{Float64}(undef, N)
    u_y    = Vector{Float64}(undef, N)
    u_z    = Vector{Float64}(undef, N)
    P_out  = Vector{Float64}(undef, N)
    x_out  = Vector{Float64}(undef, N)
    y_out  = Vector{Float64}(undef, N)
    z_out  = Vector{Float64}(undef, N)
    @inbounds for (i, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        x_out[i] = 0.5 * (lo_c[1] + hi_c[1])
        y_out[i] = 0.5 * (lo_c[2] + hi_c[2])
        z_out[i] = 0.5 * (lo_c[3] + hi_c[3])
        u_x[i]   = Float64(fields.u_1[ci][1])
        u_y[i]   = Float64(fields.u_2[ci][1])
        u_z[i]   = Float64(fields.u_3[ci][1])
        s_i      = Float64(fields.s[ci][1])
        ρ_out[i] = Float64(ρ_ref)
        P_out[i] = ρ_out[i] * Float64(Mvv(1.0 / ρ_out[i], s_i; Gamma = Gamma, cv = cv))
    end
    return (ρ = ρ_out, u_x = u_x, u_y = u_y, u_z = u_z, P = P_out,
            x = x_out, y = y_out, z = z_out)
end

"""
    primitive_recovery_3d_per_cell(fields, leaves, frame, ρ_per_cell; …)

Variant of `primitive_recovery_3d` for non-uniform density (e.g. C.1
Sod). Takes an explicit per-cell `ρ_per_cell` (length N, leaf-major).
"""
function primitive_recovery_3d_per_cell(fields::PolynomialFieldSet,
                                          leaves::AbstractVector{<:Integer},
                                          frame::EulerianFrame{3, T},
                                          ρ_per_cell::AbstractVector{<:Real};
                                          Gamma::Real = GAMMA_LAW_DEFAULT,
                                          cv::Real = CV_DEFAULT) where {T}
    N = length(leaves)
    @assert length(ρ_per_cell) == N "ρ_per_cell length $(length(ρ_per_cell)) ≠ N=$N"
    ρ_out  = Vector{Float64}(undef, N)
    u_x    = Vector{Float64}(undef, N)
    u_y    = Vector{Float64}(undef, N)
    u_z    = Vector{Float64}(undef, N)
    P_out  = Vector{Float64}(undef, N)
    x_out  = Vector{Float64}(undef, N)
    y_out  = Vector{Float64}(undef, N)
    z_out  = Vector{Float64}(undef, N)
    @inbounds for (i, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        x_out[i] = 0.5 * (lo_c[1] + hi_c[1])
        y_out[i] = 0.5 * (lo_c[2] + hi_c[2])
        z_out[i] = 0.5 * (lo_c[3] + hi_c[3])
        u_x[i]   = Float64(fields.u_1[ci][1])
        u_y[i]   = Float64(fields.u_2[ci][1])
        u_z[i]   = Float64(fields.u_3[ci][1])
        s_i      = Float64(fields.s[ci][1])
        ρ_i      = Float64(ρ_per_cell[i])
        ρ_out[i] = ρ_i
        P_out[i] = ρ_i * Float64(Mvv(1.0 / ρ_i, s_i; Gamma = Gamma, cv = cv))
    end
    return (ρ = ρ_out, u_x = u_x, u_y = u_y, u_z = u_z, P = P_out,
            x = x_out, y = y_out, z = z_out)
end

# ─────────────────────────────────────────────────────────────────────
# M3-7e: 3D Tier-C full IC factories — Cholesky-sector field set built
# from primitive 3D Tier-C ICs (3D analogs of `tier_c_*_full_ic`).
# ─────────────────────────────────────────────────────────────────────

"""
    tier_c_sod_3d_full_ic(; level=3, shock_axis=1, x_split=0.5,
                            ρL=1.0, ρR=0.125, pL=1.0, pR=0.1,
                            uL=0.0, uR=0.0, lo=nothing, hi=nothing,
                            Gamma=GAMMA_LAW_DEFAULT, cv=CV_DEFAULT,
                            T=Float64) -> NamedTuple

C.1 1D-symmetric 3D Sod IC. Builds a balanced octree (default
`level=3`, 8×8×8 = 512 leaves) on `[lo, hi]` (default `[0, 1]³`),
allocates the 16-named-field 3D Cholesky-sector field set, and
populates it per-leaf via `cholesky_sector_state_from_primitive_3d`.

The IC is a step discontinuity in `(ρ, P)` along `shock_axis ∈ {1, 2, 3}`:
left state `(ρL, pL, uL)` at axis-coordinate < `x_split`, right state
`(ρR, pR, uR)`. The step is trivial along the other two axes.

Returns the same NamedTuple shape as `tier_c_sod_full_ic` (2D):
  • `name = "tier_c_sod_3d_full"`
  • `mesh::HierarchicalMesh{3}, frame::EulerianFrame{3, T}, leaves`
  • `fields::PolynomialFieldSet` — 16-named 3D Cholesky-sector storage.
  • `ρ_per_cell::Vector{Float64}` — leaf-major density (Sod step).
  • `params::NamedTuple` — IC parameter echo.
"""
function tier_c_sod_3d_full_ic(; level::Integer = 3,
                                  shock_axis::Integer = 1,
                                  x_split::Real = 0.5,
                                  ρL::Real = 1.0, ρR::Real = 0.125,
                                  pL::Real = 1.0, pR::Real = 0.1,
                                  uL::Real = 0.0, uR::Real = 0.0,
                                  lo = nothing, hi = nothing,
                                  Gamma::Real = GAMMA_LAW_DEFAULT,
                                  cv::Real = CV_DEFAULT,
                                  T::Type = Float64)
    1 <= shock_axis <= 3 ||
        throw(ArgumentError("shock_axis must be ∈ {1, 2, 3}; got $shock_axis"))
    lo_t, hi_t = _default_box(3, lo, hi, T;
                                default_lo = (zero(T), zero(T), zero(T)),
                                default_hi = (one(T), one(T), one(T)))

    mesh = HierarchicalMesh{3}(; balanced = true)
    for _ in 1:level
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, lo_t, hi_t)
    fields = allocate_cholesky_3d_fields(mesh; T = T)

    ρ_per_cell = Vector{Float64}(undef, length(leaves))
    @inbounds for (j, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        cx = 0.5 * (lo_c[1] + hi_c[1])
        cy = 0.5 * (lo_c[2] + hi_c[2])
        cz = 0.5 * (lo_c[3] + hi_c[3])
        center = (cx, cy, cz)
        is_left = center[shock_axis] < x_split
        ρ_i = Float64(is_left ? ρL : ρR)
        P_i = Float64(is_left ? pL : pR)
        u_axis = Float64(is_left ? uL : uR)
        u_x = shock_axis == 1 ? u_axis : 0.0
        u_y = shock_axis == 2 ? u_axis : 0.0
        u_z = shock_axis == 3 ? u_axis : 0.0
        v = cholesky_sector_state_from_primitive_3d(ρ_i, u_x, u_y, u_z, P_i,
                                                      (cx, cy, cz);
                                                      Gamma = Gamma, cv = cv)
        write_detfield_3d!(fields, ci, v)
        ρ_per_cell[j] = ρ_i
    end

    return (
        name = "tier_c_sod_3d_full",
        mesh = mesh, frame = frame, leaves = leaves,
        fields = fields,
        ρ_per_cell = ρ_per_cell,
        params = (level = level, shock_axis = shock_axis, x_split = x_split,
                   ρL = ρL, ρR = ρR, pL = pL, pR = pR, uL = uL, uR = uR,
                   lo = lo_t, hi = hi_t,
                   Gamma = Gamma, cv = cv),
    )
end

"""
    tier_c_cold_sinusoid_3d_full_ic(; level=3, A=0.5, k=(1, 0, 0),
                                      ρ0=1.0, P0=1.0,
                                      lo=nothing, hi=nothing,
                                      Gamma=GAMMA_LAW_DEFAULT, cv=CV_DEFAULT,
                                      T=Float64) -> NamedTuple

C.2 3D cold sinusoid full IC. Velocity field is

    u_d(x) = A · sin(2π · k_d · (x_d - lo_d) / L_d)    for d ∈ {1, 2, 3}

(component-wise; `u_d = 0` if `k_d = 0`). Uniform `(ρ0, P0)`. Returns
the same NamedTuple shape as `tier_c_sod_3d_full_ic`.
"""
function tier_c_cold_sinusoid_3d_full_ic(; level::Integer = 3,
                                            A::Real = 0.5,
                                            k = (1, 0, 0),
                                            ρ0::Real = 1.0, P0::Real = 1.0,
                                            lo = nothing, hi = nothing,
                                            Gamma::Real = GAMMA_LAW_DEFAULT,
                                            cv::Real = CV_DEFAULT,
                                            T::Type = Float64)
    lo_t, hi_t = _default_box(3, lo, hi, T;
                                default_lo = (zero(T), zero(T), zero(T)),
                                default_hi = (one(T), one(T), one(T)))
    k_t = (Int(k[1]), Int(k[2]), Int(k[3]))

    mesh = HierarchicalMesh{3}(; balanced = true)
    for _ in 1:level
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, lo_t, hi_t)
    fields = allocate_cholesky_3d_fields(mesh; T = T)

    L1 = hi_t[1] - lo_t[1]
    L2 = hi_t[2] - lo_t[2]
    L3 = hi_t[3] - lo_t[3]
    ρ_per_cell = fill(Float64(ρ0), length(leaves))
    @inbounds for (j, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        cx = 0.5 * (lo_c[1] + hi_c[1])
        cy = 0.5 * (lo_c[2] + hi_c[2])
        cz = 0.5 * (lo_c[3] + hi_c[3])
        u_x = k_t[1] == 0 ? 0.0 : Float64(A) * sin(2π * k_t[1] * (cx - lo_t[1]) / L1)
        u_y = k_t[2] == 0 ? 0.0 : Float64(A) * sin(2π * k_t[2] * (cy - lo_t[2]) / L2)
        u_z = k_t[3] == 0 ? 0.0 : Float64(A) * sin(2π * k_t[3] * (cz - lo_t[3]) / L3)
        v = cholesky_sector_state_from_primitive_3d(Float64(ρ0), u_x, u_y, u_z,
                                                      Float64(P0),
                                                      (cx, cy, cz);
                                                      Gamma = Gamma, cv = cv)
        write_detfield_3d!(fields, ci, v)
    end

    return (
        name = "tier_c_cold_sinusoid_3d_full",
        mesh = mesh, frame = frame, leaves = leaves,
        fields = fields,
        ρ_per_cell = ρ_per_cell,
        params = (level = level, A = A, k = k_t, ρ0 = ρ0, P0 = P0,
                   lo = lo_t, hi = hi_t,
                   Gamma = Gamma, cv = cv),
    )
end

"""
    tier_c_plane_wave_3d_full_ic(; level=3, A=1e-3, k=(1, 0, 0),
                                   ρ0=1.0, P0=1.0, c=nothing,
                                   lo=nothing, hi=nothing,
                                   Gamma=GAMMA_LAW_DEFAULT, cv=CV_DEFAULT,
                                   T=Float64) -> NamedTuple

C.3 3D acoustic plane wave full IC. Builds a balanced octree, allocates
the 16-named-field 3D Cholesky-sector field set, and populates per-leaf.
The IC seeds a single right-going acoustic mode about `(ρ0, 0, 0, 0, P0)`:

    δρ(x) = A · cos(2π · k · x / L)
    δu_d(x) = (c_s / ρ0) · δρ(x) · k̂_d
    δp(x) = c_s² · δρ(x)

with `k = (k_x, k_y, k_z)` integer wave-numbers. The sound-speed `c_s`
defaults to `sqrt(Γ · P0/ρ0)`. Returns a NamedTuple shape matching the
other 3D full-IC factories.
"""
function tier_c_plane_wave_3d_full_ic(; level::Integer = 3,
                                         A::Real = 1e-3,
                                         k = (1, 0, 0),
                                         ρ0::Real = 1.0, P0::Real = 1.0,
                                         c::Union{Real,Nothing} = nothing,
                                         lo = nothing, hi = nothing,
                                         Gamma::Real = GAMMA_LAW_DEFAULT,
                                         cv::Real = CV_DEFAULT,
                                         T::Type = Float64)
    lo_t, hi_t = _default_box(3, lo, hi, T;
                                default_lo = (zero(T), zero(T), zero(T)),
                                default_hi = (one(T), one(T), one(T)))
    k_t = (Int(k[1]), Int(k[2]), Int(k[3]))

    mesh = HierarchicalMesh{3}(; balanced = true)
    for _ in 1:level
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, lo_t, hi_t)
    fields = allocate_cholesky_3d_fields(mesh; T = T)

    L1 = hi_t[1] - lo_t[1]
    L2 = hi_t[2] - lo_t[2]
    L3 = hi_t[3] - lo_t[3]
    k_phys = (Float64(k_t[1]) / Float64(L1),
              Float64(k_t[2]) / Float64(L2),
              Float64(k_t[3]) / Float64(L3))
    kmag_phys = sqrt(k_phys[1]^2 + k_phys[2]^2 + k_phys[3]^2)
    k̂ = kmag_phys > 0 ? (k_phys[1] / kmag_phys,
                          k_phys[2] / kmag_phys,
                          k_phys[3] / kmag_phys) : (0.0, 0.0, 0.0)

    c_eff = c === nothing ? sqrt(Float64(Gamma) * Float64(P0) / Float64(ρ0)) : Float64(c)

    ρ_per_cell = Vector{Float64}(undef, length(leaves))
    @inbounds for (j, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        cx = 0.5 * (lo_c[1] + hi_c[1])
        cy = 0.5 * (lo_c[2] + hi_c[2])
        cz = 0.5 * (lo_c[3] + hi_c[3])
        phase = k_phys[1] * (cx - lo_t[1]) + k_phys[2] * (cy - lo_t[2]) +
                k_phys[3] * (cz - lo_t[3])
        δρ = Float64(A) * cos(2π * phase)
        ρ_i = Float64(ρ0) + δρ
        P_i = Float64(P0) + c_eff^2 * δρ
        u_x = (c_eff / Float64(ρ0)) * δρ * k̂[1]
        u_y = (c_eff / Float64(ρ0)) * δρ * k̂[2]
        u_z = (c_eff / Float64(ρ0)) * δρ * k̂[3]
        v = cholesky_sector_state_from_primitive_3d(ρ_i, u_x, u_y, u_z, P_i,
                                                      (cx, cy, cz);
                                                      Gamma = Gamma, cv = cv)
        write_detfield_3d!(fields, ci, v)
        ρ_per_cell[j] = ρ_i
    end

    return (
        name = "tier_c_plane_wave_3d_full",
        mesh = mesh, frame = frame, leaves = leaves,
        fields = fields,
        ρ_per_cell = ρ_per_cell,
        params = (level = level, A = A, k = k_t,
                   ρ0 = ρ0, P0 = P0, c = c_eff,
                   k_phys = k_phys, k̂ = k̂,
                   lo = lo_t, hi = hi_t,
                   Gamma = Gamma, cv = cv),
    )
end

# ─────────────────────────────────────────────────────────────────────
# M3-7e: 3D Zel'dovich pancake — D.4 cosmological reference test in 3D
# ─────────────────────────────────────────────────────────────────────
#
# Methods paper §10.5 D.4 lifted to 3D. The Zel'dovich IC is intrinsically
# 1D-symmetric: position perturbation along axis 1 only; axes 2 and 3
# are trivial. The cosmological prediction is that γ_1 develops spatial
# structure (collapses) while γ_2 and γ_3 stay uniform (trivial axes
# preserve anisotropy).

"""
    tier_d_zeldovich_pancake_3d_ic_full(; level=3, A=0.5,
                                         ρ0=1.0, P0=1e-6,
                                         lo=nothing, hi=nothing,
                                         Gamma=GAMMA_LAW_DEFAULT,
                                         cv=CV_DEFAULT,
                                         T=Float64) -> NamedTuple

D.4 3D Zel'dovich pancake collapse Cholesky-sector full IC. The 3D
analog of `tier_d_zeldovich_pancake_ic` (2D). Builds a balanced octree
on `[lo, hi]` (default `[0, 1]³`), allocates the 16-named 3D field
set, and populates each leaf via `cholesky_sector_state_from_primitive_3d`
with the Zel'dovich velocity profile

    u_1(x) = -A · 2π · cos(2π (x_1 - lo_1) / L_1)
    u_2 = u_3 = 0

uniform `(ρ0, P0)`, cold-limit `α = 1, β = 0, θ_ab = 0`. The caustic
forms at `t_cross = 1 / (A · 2π)`; with `A = 0.5` we get
`t_cross ≈ 0.318`.

Returns a NamedTuple matching the 2D `tier_d_zeldovich_pancake_ic`
shape, with `t_cross` and `(L1, L2, L3)` in `params`. The recommended
BC mix is `PERIODIC` along axis 1 (collapsing) and `REFLECTING` along
axes 2 and 3 (trivial).
"""
function tier_d_zeldovich_pancake_3d_ic_full(; level::Integer = 3,
                                                A::Real = 0.5,
                                                ρ0::Real = 1.0,
                                                P0::Real = 1e-6,
                                                lo = nothing, hi = nothing,
                                                Gamma::Real = GAMMA_LAW_DEFAULT,
                                                cv::Real = CV_DEFAULT,
                                                T::Type = Float64)
    lo_t, hi_t = _default_box(3, lo, hi, T;
                                default_lo = (zero(T), zero(T), zero(T)),
                                default_hi = (one(T), one(T), one(T)))
    L1 = hi_t[1] - lo_t[1]
    L2 = hi_t[2] - lo_t[2]
    L3 = hi_t[3] - lo_t[3]
    A_t = T(A)

    mesh = HierarchicalMesh{3}(; balanced = true)
    for _ in 1:level
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, lo_t, hi_t)
    fields = allocate_cholesky_3d_fields(mesh; T = T)

    ρ_per_cell = fill(Float64(ρ0), length(leaves))
    @inbounds for (j, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        cx = 0.5 * (lo_c[1] + hi_c[1])
        cy = 0.5 * (lo_c[2] + hi_c[2])
        cz = 0.5 * (lo_c[3] + hi_c[3])
        # Zel'dovich velocity along axis 1 only.
        m1_norm = (cx - Float64(lo_t[1])) / Float64(L1)
        u1 = -Float64(A_t) * 2π * cos(2π * m1_norm)
        u2 = 0.0
        u3 = 0.0
        v = cholesky_sector_state_from_primitive_3d(Float64(ρ0), u1, u2, u3,
                                                      Float64(P0),
                                                      (cx, cy, cz);
                                                      Gamma = Gamma, cv = cv)
        write_detfield_3d!(fields, ci, v)
    end

    t_cross = 1.0 / (Float64(A_t) * 2π)

    return (
        name = "tier_d_zeldovich_pancake_3d",
        mesh = mesh, frame = frame, leaves = leaves,
        fields = fields,
        ρ_per_cell = ρ_per_cell,
        t_cross = t_cross,
        params = (level = level, A = Float64(A_t),
                   ρ0 = ρ0, P0 = P0,
                   lo = lo_t, hi = hi_t,
                   L1 = Float64(L1), L2 = Float64(L2), L3 = Float64(L3),
                   t_cross = t_cross,
                   Gamma = Gamma, cv = cv),
    )
end
