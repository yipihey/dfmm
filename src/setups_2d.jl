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
