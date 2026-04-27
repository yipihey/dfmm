# newton_step_HG.jl — HG-based Phase-1 Cholesky-sector Newton driver.
#
# This file is the Phase M3-0 counterpart to `newton_step.jl`'s
# Phase-1 driver `cholesky_step` / `cholesky_run`. Where M1's driver
# operates on a bare `SVector{2}` per cell, the HG driver operates on
# an HG `SimplicialMesh{1, T}` + `PolynomialFieldSet` pair, advancing
# every simplex in the mesh by one timestep `dt`.
#
# Bit-equality contract. Each per-simplex Newton step calls into
# M1's `cholesky_step`, which is the production-tested kernel
# `SimpleNewtonRaphson(autodiff = AutoForwardDiff())` solving the
# 2-DOF discrete EL residual `cholesky_el_residual`. With the same
# initial conditions, same `M_vv`, same `divu_half`, same `dt`, and
# the same Newton tolerances, the per-cell trajectory is byte-equal
# to M1's. The HG storage layer is just a packing/unpacking shim.
#
# Threading. Per-simplex iterations are independent (each simplex's
# Newton solve only reads/writes its own `(α, β)` pair) and so we use
# HG's `parallel_for_cells` to dispatch them. Threading is opt-in
# behind the `threaded` kwarg; in Phase M3-0 the parity tests run with
# `threaded = false` to remove threading-induced reordering as a
# variable. This matches the M1 convention.

using StaticArrays: SVector
using HierarchicalGrids: SimplicialMesh, PolynomialFieldSet,
    n_simplices, n_elements, parallel_for_cells,
    FrameBoundaries, BCKind, PERIODIC, INFLOW, OUTFLOW, REFLECTING, DIRICHLET,
    simplex_neighbor, HierarchicalMesh, EulerianFrame
using HierarchicalGrids.BoundaryConditions: is_periodic_axis
import SparseArrays as _SA
using SparseArrays: SparseMatrixCSC
using NonlinearSolve: NonlinearProblem, NonlinearFunction, solve,
                      NewtonRaphson, AutoForwardDiff, ReturnCode

"""
    cholesky_step_HG!(mesh::SimplicialMesh{1, T},
                      fields::PolynomialFieldSet,
                      M_vv, divu_half, dt;
                      threaded::Bool = false,
                      abstol::Real = 1e-13,
                      reltol::Real = 1e-13,
                      maxiters::Int = 50)

Advance every simplex in `mesh` by one timestep `dt`, mutating the
order-0 `(alpha, beta)` coefficients in `fields` in-place. The
per-simplex Newton step is M1's `cholesky_step`, called byte-identically.

Arguments:
- `mesh::SimplicialMesh{1, T}` — Lagrangian 1D simplex mesh. The
  positions are not read in M3-0 (Phase-1's autonomous Cholesky
  sector has no spatial coupling), but `n_simplices(mesh)` must
  equal `n_elements(fields)` so the field-set indexing aligns.
- `fields::PolynomialFieldSet` — order-0 field set with named
  fields `:alpha`, `:beta`.
- `M_vv`, `divu_half`, `dt` — global scalars (Phase-1 convention).

Keyword arguments:
- `threaded::Bool = false` — when `true`, use HG's
  `parallel_for_cells` for the per-simplex loop. Default `false`
  for bit-reproducibility against the M1 baseline.
- `abstol`, `reltol`, `maxiters` — forwarded to `cholesky_step`.

Returns `fields` for convenience.
"""
function cholesky_step_HG!(mesh::SimplicialMesh{1, T},
                            fields::PolynomialFieldSet,
                            M_vv::Real, divu_half::Real, dt::Real;
                            threaded::Bool = false,
                            abstol::Real = 1e-13,
                            reltol::Real = 1e-13,
                            maxiters::Int = 50) where {T<:Real}
    N = n_simplices(mesh)
    @assert N == n_elements(fields) "mesh and fields must have matching cell counts"

    if threaded
        # HG `parallel_for_cells` chunks the iteration space and
        # threads each chunk. Each simplex's Newton solve is fully
        # independent (no shared scratch in `cholesky_step`), so the
        # data race surface is empty.
        parallel_for_cells(N) do j
            q_n   = read_alphabeta(fields, j)
            q_np1 = cholesky_step(q_n, M_vv, divu_half, dt;
                                  abstol = abstol, reltol = reltol,
                                  maxiters = maxiters)
            write_alphabeta!(fields, j, q_np1)
        end
    else
        @inbounds for j in 1:N
            q_n   = read_alphabeta(fields, j)
            q_np1 = cholesky_step(q_n, M_vv, divu_half, dt;
                                  abstol = abstol, reltol = reltol,
                                  maxiters = maxiters)
            write_alphabeta!(fields, j, q_np1)
        end
    end
    return fields
end

"""
    cholesky_run_HG!(mesh, fields, M_vv, divu_half, dt, N_steps; kwargs...)

Convenience driver: run `N_steps` consecutive `cholesky_step_HG!`
calls on the same `(mesh, fields)` pair, mutating in-place. Returns
`fields`. Used by the M3-0 parity tests to evolve a per-simplex
trajectory and compare against M1's `cholesky_run`.
"""
function cholesky_run_HG!(mesh::SimplicialMesh{1, T},
                           fields::PolynomialFieldSet,
                           M_vv::Real, divu_half::Real,
                           dt::Real, N_steps::Integer;
                           kwargs...) where {T<:Real}
    for _ in 1:N_steps
        cholesky_step_HG!(mesh, fields, M_vv, divu_half, dt; kwargs...)
    end
    return fields
end

# ─────────────────────────────────────────────────────────────────────
# Convenience constructors
# ─────────────────────────────────────────────────────────────────────

"""
    single_cell_simplicial_mesh_1D(T = Float64)

Build a one-simplex `SimplicialMesh{1, T}` with vertices at 0 and 1.
Used by the M3-0 parity tests to mirror M1's single-cell Phase-1
configuration. Returns the mesh.
"""
function single_cell_simplicial_mesh_1D(::Type{T} = Float64) where {T<:Real}
    positions = [(zero(T),), (one(T),)]
    sv = reshape(Int32[1, 2], 2, 1)
    sn = zeros(Int32, 2, 1)
    return SimplicialMesh{1, T}(positions, sv, sn)
end

"""
    uniform_simplicial_mesh_1D(N::Integer, L::Real = 1.0; T = Float64)

Build a uniform-spacing `SimplicialMesh{1, T}` of `N` simplices on
`[0, L]`, with vertex `i` at `(i-1) * L / N`. Used to evolve
many independent Phase-1 trajectories in parallel for the
symplecticity test.
"""
function uniform_simplicial_mesh_1D(N::Integer, L::Real = 1.0;
                                     T::Type = Float64)
    @assert N >= 1
    Lt = T(L)
    positions = [(Lt * (i - 1) / N,) for i in 1:(N + 1)]
    sv = Matrix{Int32}(undef, 2, N)
    sn = zeros(Int32, 2, N)
    for j in 1:N
        sv[1, j] = j
        sv[2, j] = j + 1
        sn[1, j] = j > 1 ? Int32(j - 1) : Int32(0)   # face opposite vertex 1 is left neighbour
        sn[2, j] = j < N ? Int32(j + 1) : Int32(0)   # face opposite vertex 2 is right neighbour
    end
    return SimplicialMesh{1, T}(positions, sv, sn)
end

"""
    allocate_chfield_HG(N::Integer; T = Float64)

Allocate an HG order-0 `PolynomialFieldSet` for `N` cells with the
1D Cholesky-sector field names `(alpha, beta)`. Wraps
`HierarchicalGrids.allocate_polynomial_fields` with the M3-0
convention so callers don't have to spell out the basis.
"""
function allocate_chfield_HG(N::Integer; T::Type = Float64)
    basis = HierarchicalGrids.MonomialBasis{1, 0}()
    return HierarchicalGrids.allocate_polynomial_fields(
        HierarchicalGrids.SoA(), basis, Int(N);
        alpha = T, beta = T,
    )
end

# ─────────────────────────────────────────────────────────────────────
# Phase M3-1: Phase 2/5/5b on the HG substrate.
# ─────────────────────────────────────────────────────────────────────
#
# Phase 2/5/5b extend the autonomous Cholesky cell (M3-0) to a
# multi-segment Lagrangian-mass mesh with bulk fields `(x, u)` per
# vertex and `(α, β, s, Pp, Q)` per cell. The discrete EL system is
# 4N-DOF per timestep on a periodic mesh. The post-Newton operator-
# split steps (Phase 5 BGK relaxation of `(P_xx, P_⊥)`, Phase 5b
# entropy debit for the artificial-viscosity dissipation) ride on top
# of the Newton solve.
#
# Bit-equality contract (M3-1).
#
# To guarantee bit-exact parity with the M1 1D path, the HG-substrate
# driver `det_step_HG!` packs the HG state into a transient
# `Mesh1D{T,DetField{T}}`, calls M1's `det_step!` byte-identically,
# and unpacks the result back into the HG field set. The transient
# mesh is allocated once per `DetMeshHG` (cached in the wrapper) and
# reused across timesteps. The integrator algorithm — Newton solve,
# `det_el_residual`, BGK relaxation, q-dissipation, periodic-BC
# wrap — is exactly M1's. M3-2 / M3-3 will replace the transient
# `Mesh1D` with native HG iteration when the algorithm has to fan out
# to dimension-generic neighbour stencils.

using HierarchicalGrids: simplex_volume, simplex_vertex_indices,
    n_vertices, vertex_position

"""
    DetMeshHG{T}

Container bundling an HG `SimplicialMesh{1, T}` with a Phase-2/5
`PolynomialFieldSet` (named scalar fields `(alpha, beta, x, u, s, Pp,
Q)`), the per-segment Lagrangian masses `Δm`, the periodic-box length
`L_box`, the cyclic vertex half-step momenta `p_half`, and the
boundary-condition tag `bc`. The wrapper exists so the HG-side driver
`det_step_HG!` has cyclic-neighbour and BC information accessible
without re-deriving them from the simplicial mesh on every call.

Note: HG's `SimplicialMesh{1, T}` does not natively carry
periodic-wrap topology in its positions; we carry the period in
`L_box`. The simplex-neighbour matrix is wired up by HG's upstream
`HierarchicalGrids.periodic!` builder for `bc == :periodic`, and is
left with zero boundary entries for non-periodic BCs (M3-2b Swap 1).
"""
mutable struct DetMeshHG{T<:Real}
    mesh::SimplicialMesh{1, T}
    fields::PolynomialFieldSet
    Δm::Vector{T}
    L_box::T
    p_half::Vector{T}
    bc::Symbol
    # Cached transient mesh used as the byte-equal scratch space for
    # delegating to M1's `det_step!`. Re-used across timesteps to avoid
    # re-allocating the segment array; refreshed in-place from the HG
    # field set at the start of each step.
    cache_mesh::Mesh1D{T,DetField{T}}
    # M3-2b Swap 8: HG-native boundary-condition spec, mirroring `bc`
    # via the `BCKind` / `FrameBoundaries` framework. For 1D this is a
    # single `FrameBoundaries{1}` whose lone axis carries `(BCKind, BCKind)`
    # for the lo / hi sides. We retain `bc::Symbol` for legacy callers
    # and translate between the two representations at the boundaries
    # of `det_step_HG!`.
    bc_spec::FrameBoundaries{1}
    # M3-2b Swap 8: optional inflow / outflow state literals. Required
    # by Phase-7 steady-shock pinning when `bc_spec` declares an
    # `INFLOW` lo side and an `OUTFLOW` hi side. `BCKind` itself does
    # not carry the post-shock primitive vector (BCKind is a tag only),
    # so the literals live here on the wrapper. `n_pin` matches M1's
    # Phase-7 convention.
    inflow_state::Union{NamedTuple,Nothing}
    outflow_state::Union{NamedTuple,Nothing}
    n_pin::Int
end

"""
    n_cells(mesh::DetMeshHG)

Number of segments in the HG-side Phase-2 mesh. Equals
`n_simplices(mesh.mesh)`.
"""
n_cells(mesh::DetMeshHG) = HierarchicalGrids.n_simplices(mesh.mesh)

"""
    allocate_detfield_HG(N::Integer; T = Float64)

Allocate an HG order-0 `PolynomialFieldSet` for `N` cells holding the
Phase-2/5 deterministic state with named scalar fields
`(alpha, beta, x, u, s, Pp, Q)`. Wraps
`HierarchicalGrids.allocate_polynomial_fields`.
"""
function allocate_detfield_HG(N::Integer; T::Type = Float64)
    basis = HierarchicalGrids.MonomialBasis{1, 0}()
    return HierarchicalGrids.allocate_polynomial_fields(
        HierarchicalGrids.SoA(), basis, Int(N);
        alpha = T, beta = T, x = T, u = T, s = T, Pp = T, Q = T,
    )
end

"""
    DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                          Δm, L_box, Pps = nothing, Qs = nothing,
                          bc = :periodic, T = Float64)

Construct a `DetMeshHG{T}` Phase-2/5 mesh from M1-style flat arrays.
This is the HG-substrate counterpart to M1's
`Mesh1D(positions, velocities, αs, βs, ss; Δm, L_box, periodic, ...)`
constructor. The field set is allocated, populated from the arrays
using the same Pp/Q defaulting rules as `Mesh1D`, and bundled with
the simplicial mesh + cyclic-neighbour matrix into a `DetMeshHG`.

The transient `cache_mesh::Mesh1D` used for delegating to M1's
`det_step!` is built here once and reused on every call. Its segment
states are refreshed from the HG field set at the start of each
timestep.
"""
function DetMeshHG_from_arrays(
    positions::AbstractVector,
    velocities::AbstractVector,
    αs::AbstractVector,
    βs::AbstractVector,
    ss::AbstractVector;
    Δm::AbstractVector,
    L_box::Union{Real,Nothing} = nothing,
    Pps::Union{AbstractVector,Nothing} = nothing,
    Qs::Union{AbstractVector,Nothing}  = nothing,
    bc::Symbol = :periodic,
    bc_spec::Union{FrameBoundaries{1},Nothing} = nothing,
    inflow_state::Union{NamedTuple,Nothing} = nothing,
    outflow_state::Union{NamedTuple,Nothing} = nothing,
    n_pin::Integer = 2,
    T::Type = Float64,
)
    # Build the M1 Mesh1D first; this enforces the same sanity checks
    # M1 does and computes the same `Pp` defaults / `p_half` init so the
    # parity is automatic.
    periodic = bc == :periodic
    cache_mesh = Mesh1D(positions, velocities, αs, βs, ss;
                        Δm = Δm, Pps = Pps, Qs = Qs, L_box = L_box,
                        periodic = periodic, bc = bc)
    N = n_segments(cache_mesh)
    Lt = T(cache_mesh.L_box)

    # Build the periodic HG simplicial mesh on the Lagrangian-mass
    # coordinate. The mesh's vertex positions encode `m` (the
    # Lagrangian-mass coordinate); each simplex's `simplex_volume`
    # returns the per-segment Δm in this convention. Boundary face
    # neighbours are wired up via HG's upstream `periodic!` builder
    # (M3-2b Swap 1).
    cum_m = T(0)
    pos_m = Vector{NTuple{1, T}}(undef, N + 1)
    pos_m[1] = (T(0),)
    for j in 1:N
        cum_m += T(Δm[j])
        pos_m[j + 1] = (cum_m,)
    end
    sv = Matrix{Int32}(undef, 2, N)
    sn = zeros(Int32, 2, N)
    for j in 1:N
        sv[1, j] = j
        sv[2, j] = j + 1
        sn[1, j] = j < N ? Int32(j + 1) : Int32(0)   # interior; wrap left to periodic!
        sn[2, j] = j > 1 ? Int32(j - 1) : Int32(0)   # interior; wrap left to periodic!
    end
    hg_mesh = SimplicialMesh{1, T}(pos_m, sv, sn)
    # Wire periodic wrap-around on the mass axis. Iff `bc == :periodic`
    # we identify the lo-mass boundary face with the hi-mass one; for
    # other BCs the boundary entries stay 0 (HG convention).
    if periodic
        HierarchicalGrids.periodic!(hg_mesh, (true,), ((T(0), cum_m),))
    end

    # Allocate the HG field set, populate from the cache mesh.
    fields = allocate_detfield_HG(N; T = T)
    @inbounds for j in 1:N
        write_detfield!(fields, j, cache_mesh.segments[j].state)
    end

    Δm_vec = T[seg.Δm for seg in cache_mesh.segments]
    p_half = copy(cache_mesh.p_half)

    # M3-2b Swap 8: build a `FrameBoundaries{1}` reflecting the
    # symbolic `bc` if no explicit `bc_spec` was supplied. The mapping:
    #   :periodic         → (PERIODIC, PERIODIC)
    #   :inflow_outflow   → (INFLOW,   OUTFLOW)
    # If callers want a different combination they can pass `bc_spec`
    # directly (e.g. mixed BCs in 2D/3D phases that won't reach this
    # 1D constructor).
    bc_spec_resolved = if bc_spec !== nothing
        bc_spec
    else
        if bc == :periodic
            FrameBoundaries{1}(((PERIODIC, PERIODIC),))
        elseif bc == :inflow_outflow
            FrameBoundaries{1}(((INFLOW, OUTFLOW),))
        else
            # Conservative default for unrecognised `bc`: REFLECTING/REFLECTING.
            FrameBoundaries{1}(((REFLECTING, REFLECTING),))
        end
    end

    return DetMeshHG{T}(hg_mesh, fields, Δm_vec, Lt, p_half, bc, cache_mesh,
                         bc_spec_resolved, inflow_state, outflow_state, Int(n_pin))
end

"""
    sync_cache_from_HG!(mesh::DetMeshHG)

Refresh the cached `Mesh1D` from the HG field set so the next
`det_step!` sees the current state. Internal helper used by
`det_step_HG!`.
"""
@inline function sync_cache_from_HG!(mesh::DetMeshHG{T}) where {T<:Real}
    N = n_cells(mesh)
    fs = mesh.fields
    @inbounds for j in 1:N
        det = read_detfield(fs, j)
        mesh.cache_mesh.segments[j].state = det
    end
    # Refresh p_half from the cached mesh's vertex velocities + Δm so
    # M1's invariant `p_half[i] = m̄_i u_i` holds at the start of the
    # step.
    @inbounds for i in 1:N
        i_left = i == 1 ? N : i - 1
        m̄ = (mesh.cache_mesh.segments[i_left].Δm + mesh.cache_mesh.segments[i].Δm) / 2
        mesh.cache_mesh.p_half[i] = m̄ * mesh.cache_mesh.segments[i].state.u
    end
    return mesh
end

"""
    sync_HG_from_cache!(mesh::DetMeshHG)

Mirror of `sync_cache_from_HG!`: write the M1 cache mesh's segment
states back into the HG field set after a `det_step!` call. Also
copies the `p_half` cache vector.
"""
@inline function sync_HG_from_cache!(mesh::DetMeshHG{T}) where {T<:Real}
    N = n_cells(mesh)
    fs = mesh.fields
    @inbounds for j in 1:N
        write_detfield!(fs, j, mesh.cache_mesh.segments[j].state)
    end
    @inbounds for i in 1:N
        mesh.p_half[i] = mesh.cache_mesh.p_half[i]
    end
    return mesh
end

"""
    bc_symbol_from_spec(spec::FrameBoundaries{1}) -> Symbol

Translate a 1D `FrameBoundaries{1}` into the legacy dfmm `bc::Symbol`
tag consumed by M1's `det_step!`. Mapping:

- `(PERIODIC, PERIODIC)` → `:periodic`
- `(INFLOW,   OUTFLOW)`  → `:inflow_outflow`
- otherwise              → `:periodic` fallback (REFLECTING and
                            DIRICHLET don't have M1 1D handlers yet)

Used by `det_step_HG!` when the cache-mesh delegate still consumes
the legacy `bc` kwarg (this thin translation layer goes away when
M3-3 retires the cache mesh).
"""
@inline function bc_symbol_from_spec(spec::FrameBoundaries{1})
    lo, hi = spec.spec[1]
    if lo === PERIODIC && hi === PERIODIC
        return :periodic
    elseif lo === INFLOW && hi === OUTFLOW
        return :inflow_outflow
    else
        return :periodic
    end
end

"""
    det_step_HG!(mesh::DetMeshHG, dt; tau=nothing, sparse_jac=true,
                 q_kind=:none, c_q_quad=1.0, c_q_lin=0.5,
                 abstol=1e-13, reltol=1e-13, maxiters=50)

Advance the HG-substrate Phase-2/5/5b/7 mesh by one timestep `dt`.

**M3-3e-1: native HG-side Newton driver.** Pre-M3-3e-1, this driver
synced state into a cached `Mesh1D` shim and delegated to M1's
`det_step!`. Starting M3-3e-1 the deterministic Newton path runs
natively on the HG field set and per-wrapper bookkeeping (`mesh.Δm`,
`mesh.L_box`, `mesh.p_half`, `mesh.bc`/`bc_spec`, inflow/outflow
literals). The `cache_mesh` field is still allocated on the wrapper
because M3-3e-2 (stochastic injection), M3-3e-3 (AMR + tracers), and
M3-3e-4 (realizability projection) still consume it; M3-3e-5 will
drop the field once all paths are native.

Bit-exact contract. The native path produces byte-equal output to
the cache-mesh-driven path on the deterministic Newton tests
(Phase 2 / 5 / 5b / 7 + M3-2b Swap 8). The Newton solve itself
already calls the pure-flat-array `det_el_residual` (so per-Newton
arithmetic is unchanged); the bit-equality covers also the Phase-5
post-Newton BGK, Phase-5b q-dissipation entropy update, and Phase-7
inflow/outflow vertex-chain pinning, which are all reimplemented
against `fields::PolynomialFieldSet` here.

**M3-2b Swap 8:** boundary-condition data (`bc`, `inflow_state`,
`outflow_state`, `n_pin`) are read from the wrapper struct, not from
kwargs. Use `DetMeshHG_from_arrays(...; bc, inflow_state, outflow_state,
n_pin, bc_spec=...)` to attach them at construction. The 1D
`FrameBoundaries{1}` lives on `mesh.bc_spec`; we translate it to the
legacy symbol-tag (still consumed by `det_el_residual` and the
Phase-7 pinning branch) via `bc_symbol_from_spec`.

Mutates the HG field set in-place; returns `mesh`.
"""
function det_step_HG!(mesh::DetMeshHG{T}, dt::Real;
                       tau::Union{Real,Nothing} = nothing,
                       sparse_jac::Bool = true,
                       q_kind::Symbol = Q_KIND_NONE,
                       c_q_quad::Real = 1.0,
                       c_q_lin::Real  = 0.5,
                       abstol::Real = 1e-13, reltol::Real = 1e-13,
                       maxiters::Int = 50) where {T<:Real}
    return det_step_HG_native!(mesh, dt;
                                tau = tau,
                                sparse_jac = sparse_jac,
                                q_kind = q_kind,
                                c_q_quad = c_q_quad,
                                c_q_lin  = c_q_lin,
                                abstol = abstol, reltol = reltol,
                                maxiters = maxiters)
end

# ─────────────────────────────────────────────────────────────────────
# M3-3e-1: native deterministic Newton driver on the HG field set.
# ─────────────────────────────────────────────────────────────────────
#
# Mirrors M1's `det_step!` (`src/newton_step.jl`) line-for-line on the
# physics, but reads/writes the per-cell state through
# `fields::PolynomialFieldSet` (via `read_detfield` / `write_detfield!`)
# instead of `mesh.segments[j].state`. The `cache_mesh::Mesh1D` shim is
# not touched.
#
# Sub-blocks:
#   1. Pre-Newton bookkeeping  — pack flat `y_n` and `s_vec`; cache
#      pre-Newton `ρ_n_vec` and `Pp_n_vec` (Phase 5 transport step).
#   2. Newton solve           — `det_el_residual` (pure flat-array)
#      via NonlinearSolve, identical kwargs to M1.
#   3. Unpack `(x, u, α, β)`  — write back into `fields`; refresh
#      `mesh.p_half[i] = m̄_i u_i`.
#   4. Phase 5b post-Newton entropy update for the q-dissipation
#      (`q_kind = :vNR_linear_quadratic`).
#   5. Phase 5  post-Newton BGK joint relaxation of (P_xx, P_⊥) +
#      β realizability clip + Q · exp(-Δt/τ) (when `tau` is supplied).
#   6. Phase 7  inflow/outflow Dirichlet pinning of leftmost /
#      rightmost `n_pin` segments + cumulative-Δx vertex chain.
#
# All per-segment scalar arithmetic is byte-identical to M1's
# `det_step!`: the only change is the storage target.
"""
    det_step_HG_native!(mesh::DetMeshHG, dt; ...)

Native HG-side variant of `det_step_HG!` that does not touch
`mesh.cache_mesh`. Bit-exact equal to the M1 cache-mesh-driven path
on the deterministic Newton tests (M3-1 Phase 2/5/5b, M3-2 Phase 7,
M3-2b Swap 8). Internal helper; callers should use `det_step_HG!`.
"""
function det_step_HG_native!(mesh::DetMeshHG{T}, dt::Real;
                              tau::Union{Real,Nothing} = nothing,
                              sparse_jac::Bool = true,
                              q_kind::Symbol = Q_KIND_NONE,
                              c_q_quad::Real = 1.0,
                              c_q_lin::Real  = 0.5,
                              abstol::Real = 1e-13, reltol::Real = 1e-13,
                              maxiters::Int = 50) where {T<:Real}
    @assert q_kind_supported(q_kind) "Unsupported q_kind = $q_kind. " *
        "Use :none or :vNR_linear_quadratic."
    N = n_cells(mesh)
    fs = mesh.fields
    Δm_vec = copy(mesh.Δm)
    L_box  = mesh.L_box
    bc_legacy = bc_symbol_from_spec(mesh.bc_spec)

    # ── (1) Pre-Newton bookkeeping ───────────────────────────────────
    # Pack `y_n` and `s_vec` from the field set, cache pre-Newton ρ_n
    # (segment density) and Pp_n for the Phase-5 transport step.
    y_n      = Vector{T}(undef, 4N)
    s_vec    = Vector{T}(undef, N)
    ρ_n_vec  = Vector{T}(undef, N)
    Pp_n_vec = Vector{T}(undef, N)
    @inbounds for j in 1:N
        x_j  = T(fs.x[j][1])
        u_j  = T(fs.u[j][1])
        α_j  = T(fs.alpha[j][1])
        β_j  = T(fs.beta[j][1])
        s_j  = T(fs.s[j][1])
        Pp_j = T(fs.Pp[j][1])
        y_n[4*(j-1) + 1] = x_j
        y_n[4*(j-1) + 2] = u_j
        y_n[4*(j-1) + 3] = α_j
        y_n[4*(j-1) + 4] = β_j
        s_vec[j]  = s_j
        Pp_n_vec[j] = Pp_j
        # ρ_n = Δm_j / (x_{j+1} - x_j), with periodic wrap on j == N.
        if j < N
            x_right_j = T(fs.x[j+1][1])
        else
            x_right_j = T(fs.x[1][1]) + L_box
        end
        ρ_n_vec[j] = Δm_vec[j] / (x_right_j - x_j)
    end

    y0 = explicit_euler_guess(y_n, Δm_vec, s_vec, L_box, T(dt))

    # Phase 7: precompute inflow / outflow ghost data for the residual
    # so the cyclic stencil at the boundary is replaced by Dirichlet
    # ghosts. Mirrors M1's `det_step!` exactly.
    inflow_xun  = nothing
    outflow_xun = nothing
    inflow_Pq   = nothing
    outflow_Pq  = nothing
    if bc_legacy == :inflow_outflow && mesh.inflow_state !== nothing
        inflow_Pq = T(mesh.inflow_state.Pp)
        if mesh.outflow_state !== nothing
            outflow_xun = (
                x_n   = T(L_box),
                x_np1 = T(L_box),
                u_n   = T(mesh.outflow_state.u),
                u_np1 = T(mesh.outflow_state.u),
            )
            outflow_Pq = T(mesh.outflow_state.Pp)
        end
    end

    cqq = T(c_q_quad)
    cql = T(c_q_lin)

    # ── (2) Newton solve ────────────────────────────────────────────
    p = (y_n, Δm_vec, s_vec, L_box, T(dt))
    f = (u, p_in) -> begin
        y_n_in, Δm_in, s_in, L_box_in, dt_in = p_in
        return det_el_residual(u, y_n_in, Δm_in, s_in, L_box_in, dt_in;
                               q_kind = q_kind,
                               c_q_quad = cqq,
                               c_q_lin  = cql,
                               bc = bc_legacy,
                               inflow_xun = inflow_xun,
                               outflow_xun = outflow_xun,
                               inflow_Pq = inflow_Pq,
                               outflow_Pq = outflow_Pq)
    end

    if sparse_jac && N ≥ 16
        jac_proto = det_jac_sparsity(N)
        nf = NonlinearFunction(f; jac_prototype = jac_proto)
        prob = NonlinearProblem(nf, y0, p)
        sol = solve(
            prob,
            NewtonRaphson(; autodiff = AutoForwardDiff());
            abstol = abstol, reltol = reltol, maxiters = maxiters,
        )
    else
        prob = NonlinearProblem(f, y0, p)
        sol = solve(
            prob,
            NewtonRaphson(; autodiff = AutoForwardDiff());
            abstol = abstol, reltol = reltol, maxiters = maxiters,
        )
    end
    res = det_el_residual(sol.u, y_n, Δm_vec, s_vec, L_box, T(dt);
                          q_kind = q_kind,
                          c_q_quad = cqq,
                          c_q_lin  = cql,
                          bc = bc_legacy,
                          inflow_xun = inflow_xun,
                          outflow_xun = outflow_xun,
                          inflow_Pq = inflow_Pq,
                          outflow_Pq = outflow_Pq)
    res_norm = maximum(abs, res)
    if res_norm > 1e6 * abstol && sol.retcode != ReturnCode.Success
        error("det_step_HG_native! Newton solve failed: retcode = $(sol.retcode), " *
              "‖residual‖∞ = $res_norm")
    end

    # ── (3) Unpack `(x, u, α, β)` and refresh p_half ────────────────
    # Mirrors M1's `unpack_state!`: rewrites only x/u/α/β, leaves
    # s/Pp/Q untouched (entropy is frozen across the Newton step;
    # post-Newton operator splits below mutate them).
    @inbounds for j in 1:N
        x_new = T(sol.u[4*(j-1) + 1])
        u_new = T(sol.u[4*(j-1) + 2])
        α_new = T(sol.u[4*(j-1) + 3])
        β_new = T(sol.u[4*(j-1) + 4])
        fs.x[j]     = (x_new,)
        fs.u[j]     = (u_new,)
        fs.alpha[j] = (α_new,)
        fs.beta[j]  = (β_new,)
        i_left = j == 1 ? N : j - 1
        m̄ = (Δm_vec[i_left] + Δm_vec[j]) / 2
        mesh.p_half[j] = m̄ * u_new
    end

    # ── (4) Phase 5b post-Newton entropy update for q-dissipation ────
    # The artificial viscosity dissipates kinetic energy at rate
    # `(q/ρ) ∂_x u` (negative when ∂_x u < 0, i.e. compression heats).
    # For an ideal-gas EOS `M_vv = J^{1-Γ} exp(s/c_v)`, the change in
    # internal energy at fixed J converts to entropy via
    #   Δs/c_v = -(Γ-1) (q/P_xx) (∂_x u) Δt
    # which is non-negative in compression (q > 0, ∂_x u < 0). Skipped
    # at `q_kind = :none` so Phase 5 behaviour is bit-equal.
    if q_active(q_kind)
        Γ_q = T(GAMMA_LAW_DEFAULT)
        @inbounds for j in 1:N
            j_right = j == N ? 1 : j + 1
            wrap = (j == N) ? L_box : zero(T)
            x_left_n  = y_n[4*(j-1) + 1]
            x_right_n = y_n[4*(j_right-1) + 1] + wrap
            x_left_np1  = T(fs.x[j][1])
            x_right_np1 = T(fs.x[j_right][1]) + wrap
            u_left_n  = y_n[4*(j-1) + 2]
            u_right_n = y_n[4*(j_right-1) + 2]
            u_left_np1  = T(fs.u[j][1])
            u_right_np1 = T(fs.u[j_right][1])

            x̄_left  = (x_left_n  + x_left_np1)  / 2
            x̄_right = (x_right_n + x_right_np1) / 2
            ū_left  = (u_left_n  + u_left_np1)  / 2
            ū_right = (u_right_n + u_right_np1) / 2
            Δx̄ = x̄_right - x̄_left
            divu_j = (ū_right - ū_left) / Δx̄
            J̄_j = Δx̄ / Δm_vec[j]
            ρ̄_j = one(T) / J̄_j
            s_pre = T(fs.s[j][1])
            M̄vv_j = Mvv(J̄_j, s_pre)
            P̄xx_j = ρ̄_j * M̄vv_j

            c_s_bar = sqrt(max(Γ_q * M̄vv_j, zero(T)))
            q_j = compute_q_segment(divu_j, ρ̄_j, c_s_bar, Δx̄;
                                    c_q_quad = cqq, c_q_lin = cql)
            if q_j > 0 && P̄xx_j > 0
                Δs_q = -(Γ_q - one(T)) * (q_j / P̄xx_j) * divu_j * T(dt)
                fs.s[j] = (s_pre + Δs_q,)
            end
        end
    end

    # ── (5) Phase 5 post-Newton BGK joint relaxation ────────────────
    # Three substeps per segment (matches M1's `det_step!`):
    #   (1) Lagrangian transport of P_⊥: Pp_transport = Pp_n · ρ_np1/ρ_n.
    #   (2) Joint BGK relaxation of (P_xx, P_⊥) toward P_iso, with
    #       conservation of the trace P_xx + 2 P_⊥ to round-off.
    #   (3) Re-anchor entropy: s_new = s_old + log(P_xx_new / P_xx_pre).
    # Then β decays by `exp(-Δt/τ)` with realizability clip
    # `|β| ≤ 0.999 √M_vv_new`, and Q transports + decays the same way.
    if tau !== nothing
        τ = T(tau)
        decay = exp(-T(dt) / τ)
        rh = T(0.999)
        @inbounds for j in 1:N
            # ρ_np1 from the post-Newton x-chain (with periodic wrap).
            x_j    = T(fs.x[j][1])
            if j < N
                x_jp1 = T(fs.x[j+1][1])
            else
                x_jp1 = T(fs.x[1][1]) + L_box
            end
            ρ_np1 = Δm_vec[j] / (x_jp1 - x_j)
            J_np1 = one(T) / ρ_np1
            s_pre = T(fs.s[j][1])
            Mvv_pre = Mvv(J_np1, s_pre)
            Pxx_pre = ρ_np1 * Mvv_pre

            Pp_transport = Pp_n_vec[j] * ρ_np1 / ρ_n_vec[j]
            Pxx_new, Pp_new = bgk_relax_pressures(Pxx_pre, Pp_transport,
                                                   T(dt), τ)
            Mvv_new = Pxx_new / ρ_np1
            if Mvv_new > 0 && Mvv_pre > 0
                Δs_over_cv = log(Mvv_new / Mvv_pre)
                s_new = s_pre + Δs_over_cv
            else
                s_new = s_pre
                Mvv_new = Mvv_pre
            end
            β_pre = T(fs.beta[j][1])
            β_new = β_pre * decay
            β_max = rh * sqrt(max(Mvv_new, zero(T)))
            if β_new > β_max
                β_new = β_max
            elseif β_new < -β_max
                β_new = -β_max
            end
            Q_n = T(fs.Q[j][1])
            Q_transport = Q_n * ρ_np1 / ρ_n_vec[j]
            Q_new = Q_transport * decay

            fs.beta[j] = (β_new,)
            fs.s[j]    = (s_new,)
            fs.Pp[j]   = (Pp_new,)
            fs.Q[j]    = (Q_new,)
        end
    end

    # ── (6) Phase 7 inflow/outflow pinning ──────────────────────────
    # After the Newton solve + BGK + q-dissipation, pin the leftmost
    # `n_pin` segments to the upstream Dirichlet state and the rightmost
    # `n_pin` segments to the downstream state. The pinning preserves
    # `Δm` and only overwrites the segment's per-cell state and
    # vertex chain. Inflow segments march at `u_inflow` with vertex
    # spacing `Δx_j = Δm_j / ρ_inflow`; outflow segments anchor the
    # right boundary at `x = L_box` and chain leftward with
    # `Δx_j = Δm_j / ρ_outflow`.
    if bc_legacy == :inflow_outflow
        @assert mesh.inflow_state !== nothing "bc = :inflow_outflow requires inflow_state"
        n_pin_left  = mesh.n_pin
        n_pin_right = mesh.n_pin

        # Left (inflow) chain: anchor segment 1's left vertex at x = 0.
        ρ_in = T(mesh.inflow_state.rho)
        u_in   = T(mesh.inflow_state.u)
        α_in   = T(mesh.inflow_state.alpha)
        β_in   = T(mesh.inflow_state.beta)
        s_in   = T(mesh.inflow_state.s)
        Pp_in  = T(mesh.inflow_state.Pp)
        Q_in   = T(mesh.inflow_state.Q)
        cum_Δx_in = zero(T)
        @inbounds for j in 1:min(n_pin_left, N)
            x_left_pinned = (j == 1) ? zero(T) : cum_Δx_in
            cum_Δx_in += Δm_vec[j] / ρ_in
            fs.x[j]     = (x_left_pinned,)
            fs.u[j]     = (u_in,)
            fs.alpha[j] = (α_in,)
            fs.beta[j]  = (β_in,)
            fs.s[j]     = (s_in,)
            fs.Pp[j]    = (Pp_in,)
            fs.Q[j]     = (Q_in,)
            i_left = j == 1 ? N : j - 1
            m̄ = (Δm_vec[i_left] + Δm_vec[j]) / 2
            mesh.p_half[j] = m̄ * u_in
        end

        # Right (outflow) chain.
        if mesh.outflow_state !== nothing
            ρ_out  = T(mesh.outflow_state.rho)
            u_out  = T(mesh.outflow_state.u)
            α_out  = T(mesh.outflow_state.alpha)
            β_out  = T(mesh.outflow_state.beta)
            s_out  = T(mesh.outflow_state.s)
            Pp_out = T(mesh.outflow_state.Pp)
            Q_out  = T(mesh.outflow_state.Q)
            cum_Δx = zero(T)
            @inbounds for j in N:-1:(N - n_pin_right + 1)
                Δx_j = Δm_vec[j] / ρ_out
                cum_Δx += Δx_j
                x_left_pinned = T(L_box) - cum_Δx
                fs.x[j]     = (x_left_pinned,)
                fs.u[j]     = (u_out,)
                fs.alpha[j] = (α_out,)
                fs.beta[j]  = (β_out,)
                fs.s[j]     = (s_out,)
                fs.Pp[j]    = (Pp_out,)
                fs.Q[j]     = (Q_out,)
                i_left = j == 1 ? N : j - 1
                m̄ = (Δm_vec[i_left] + Δm_vec[j]) / 2
                mesh.p_half[j] = m̄ * u_out
            end
        else
            # Zero-gradient outflow: copy state of the first interior
            # segment into the trailing n_pin_right segments. Chain
            # rightward from segment (N - n_pin_right) using its
            # post-Newton density.
            j_src = max(1, N - n_pin_right)
            src_x = T(fs.x[j_src][1])
            src_u = T(fs.u[j_src][1])
            src_α = T(fs.alpha[j_src][1])
            src_β = T(fs.beta[j_src][1])
            src_s = T(fs.s[j_src][1])
            src_Pp = T(fs.Pp[j_src][1])
            src_Q  = T(fs.Q[j_src][1])
            # Density of the source segment (post-Newton).
            if j_src < N
                src_x_right = T(fs.x[j_src + 1][1])
            else
                src_x_right = T(fs.x[1][1]) + L_box
            end
            ρ_src = Δm_vec[j_src] / max(src_x_right - src_x, eps(T))
            @inbounds for j in (N - n_pin_right + 1):N
                if j > 1
                    prev_x  = T(fs.x[j-1][1])
                    x_left_pinned = prev_x + (Δm_vec[j-1] / ρ_src)
                else
                    x_left_pinned = T(fs.x[j][1])
                end
                fs.x[j]     = (x_left_pinned,)
                fs.u[j]     = (src_u,)
                fs.alpha[j] = (src_α,)
                fs.beta[j]  = (src_β,)
                fs.s[j]     = (src_s,)
                fs.Pp[j]    = (src_Pp,)
                fs.Q[j]     = (src_Q,)
                i_left = j == 1 ? N : j - 1
                m̄ = (Δm_vec[i_left] + Δm_vec[j]) / 2
                mesh.p_half[j] = m̄ * src_u
            end
        end
    end

    return mesh
end

"""
    det_run_HG!(mesh::DetMeshHG, dt, N_steps; kwargs...)

Convenience driver: run `N_steps` consecutive `det_step_HG!` calls
on `mesh`. Mutates `mesh` in place; returns `mesh`. Keyword
arguments (including the Phase-5 `tau` for BGK relaxation) are
forwarded to `det_step_HG!`.
"""
function det_run_HG!(mesh::DetMeshHG{T}, dt::Real, N_steps::Integer;
                      kwargs...) where {T<:Real}
    for _ in 1:N_steps
        det_step_HG!(mesh, dt; kwargs...)
    end
    return mesh
end

# ─────────────────────────────────────────────────────────────────────
# Diagnostics — HG-side mirrors of M1's `total_*` helpers.
# ─────────────────────────────────────────────────────────────────────

"""
    total_mass_HG(mesh::DetMeshHG)

Sum of `Δm_j` over the HG mesh. Bit-stable: per-segment `Δm` is a
fixed label on the wrapper, not a state variable.
"""
total_mass_HG(mesh::DetMeshHG) = sum(mesh.Δm)

"""
    total_momentum_HG(mesh::DetMeshHG)

Sum of vertex momenta `Σ_i m̄_i u_i` from the HG field set. Mirrors
M1's `total_momentum`.
"""
function total_momentum_HG(mesh::DetMeshHG{T}) where {T<:Real}
    sync_cache_from_HG!(mesh)
    return total_momentum(mesh.cache_mesh)
end

"""
    total_energy_HG(mesh::DetMeshHG)

Total deterministic energy (kinetic + Cholesky-Hamiltonian). Mirrors
M1's `total_energy`.
"""
function total_energy_HG(mesh::DetMeshHG{T}) where {T<:Real}
    sync_cache_from_HG!(mesh)
    return total_energy(mesh.cache_mesh)
end

"""
    segment_density_HG(mesh::DetMeshHG, j::Integer)

Eulerian density `ρ_j = Δm_j / (x_{j+1} − x_j)` of segment `j` on the
HG mesh, with periodic wrap on `j == N`.
"""
function segment_density_HG(mesh::DetMeshHG{T}, j::Integer) where {T<:Real}
    sync_cache_from_HG!(mesh)
    return segment_density(mesh.cache_mesh, j)
end

"""
    segment_length_HG(mesh::DetMeshHG, j::Integer)

Eulerian length of segment `j` on the HG mesh.
"""
function segment_length_HG(mesh::DetMeshHG{T}, j::Integer) where {T<:Real}
    sync_cache_from_HG!(mesh)
    return segment_length(mesh.cache_mesh, j)
end

# ─────────────────────────────────────────────────────────────────────
# M3-2b Swap 6: HG-aware Newton-Jacobian sparsity for the EL residual
# ─────────────────────────────────────────────────────────────────────
#
# `det_jac_sparsity_HG` produces the same `SparseMatrixCSC{Bool}` shape
# that the M1 hand-rolled `det_jac_sparsity(N)` builds — a 4N × 4N
# Boolean tri-block-banded sparsity matching the periodic-cyclic
# `(j-1, j, j+1)` connectivity per residual block.
#
# Two backends:
#
#  - For a periodic 1D `SimplicialMesh{1, T}` (the M3-1 / M3-2 case),
#    the HG-native mesh-level neighbor information is the simplex-
#    neighbor matrix `mesh.simplex_neighbors`: face `k` of simplex `s`
#    points to the simplex sharing that face (or 0 for a domain
#    boundary). On a periodic 1D mesh the lo / hi neighbours yield
#    the cyclic `(j-1, j, j+1)` triple and the sparsity matches M1
#    bit-exactly.
#
#  - The same helper accepts an explicit `n` and depth-1 connectivity
#    iterator, so future M3-3 callers can drive it from a
#    `cell_adjacency_sparsity` extracted off a `HierarchicalMesh{D}`
#    in 2D/3D with no algorithmic divergence.
#
# Why not call HG's `cell_adjacency_sparsity` directly? That API
# operates on `HierarchicalMesh{D, M}` (the cartesian-tree mesh), not
# on `SimplicialMesh{1, T}` (our Lagrangian-segment mesh). The 1D
# Lagrangian path therefore takes the `simplex_neighbors`-driven
# code path here. M3-3's native 2D Newton residual will use the
# `HierarchicalMesh`-side path on the Eulerian remap target.

"""
    det_jac_sparsity_HG(mesh::SimplicialMesh{1, T}; depth::Int = 1)
        -> SparseMatrixCSC{Bool, Int}

HG-aware Boolean sparsity prototype for the Phase-2 Euler-Lagrange
Jacobian. For each residual segment `i`, marks nonzero column entries
in segments `i_left`, `i`, `i_right`, where `(i_left, i_right) =
simplex_neighbor(mesh, i, k)` for `k = 1, 2`. The result is a
`4N × 4N` sparsity pattern, bit-equal to M1's hand-rolled
`det_jac_sparsity(N)` on a periodic 1D mesh. `depth` is forwarded to
match the HG `cell_adjacency_sparsity` API but only `depth = 1` is
exercised in M3-2b (the 1D Phase-2 EL stencil is tridiagonal-block).

This is a future hook for M3-3's native HG-side EL residual; in
M3-2b the cache-mesh-driven Newton solve continues to use the
M1 sparsity. Verified bit-exact equality on N = 16 periodic
`DetMeshHG` is the parity gate (see `test_M3_2b_swap6_sparsity_HG.jl`).
"""
function det_jac_sparsity_HG(mesh::SimplicialMesh{1, T};
                              depth::Int = 1) where {T<:Real}
    depth >= 1 || throw(ArgumentError("depth must be ≥ 1"))
    N = Int(n_simplices(mesh))
    rows = Int[]
    cols = Int[]
    @inbounds for i in 1:N
        # Per the post-Swap-1 layout in `DetMeshHG_from_arrays`:
        #   sv[1, j] = j      (lo-mass vertex)
        #   sv[2, j] = j+1    (hi-mass vertex)
        # ⇒ face 1 (opposite vertex 1) is the *hi*-side face (right
        #   neighbour), and face 2 (opposite vertex 2) is the *lo*-side
        #   face (left neighbour). HG's `periodic!` wires the boundary
        #   `0` entries through the wrap-around. The sparsity itself is
        #   symmetric in `(i_lo, i_hi)`, so the relabeling doesn't
        #   affect the resulting `(row, col)` pattern — but we honour
        #   the convention for clarity and to keep the M3-3 native
        #   path consistent.
        i_hi_raw = simplex_neighbor(mesh, i, 1)
        i_lo_raw = simplex_neighbor(mesh, i, 2)
        # Translate boundary `0` → cyclic wrap-around for parity with
        # M1's `det_jac_sparsity(N)`, which always wraps. M3-3 will
        # refine this for genuine non-periodic boundaries by consulting
        # `mesh.bc_spec` / `face_neighbors_with_bcs`.
        i_lo = i_lo_raw == 0 ? (i == 1 ? N : i - 1) : i_lo_raw
        i_hi = i_hi_raw == 0 ? (i == N ? 1 : i + 1) : i_hi_raw
        for k in 1:4
            r = 4 * (i - 1) + k
            for j in (i_lo, i, i_hi)
                for ℓ in 1:4
                    push!(rows, r)
                    push!(cols, 4 * (j - 1) + ℓ)
                end
            end
        end
    end
    # Use SparseArrays.sparse with combine=max to dedupe overlapping
    # (rows, cols) entries that arise when N=2 or i_lo == i_hi.
    return _SA.sparse(rows, cols, ones(Float64, length(rows)),
                      4N, 4N, max)
end

"""
    det_jac_sparsity_HG(mesh::DetMeshHG; depth::Int = 1)
        -> SparseMatrixCSC{Bool, Int}

Convenience overload that pulls the underlying `SimplicialMesh{1, T}`
out of a `DetMeshHG` wrapper.
"""
det_jac_sparsity_HG(mesh::DetMeshHG{T}; depth::Int = 1) where {T<:Real} =
    det_jac_sparsity_HG(mesh.mesh; depth = depth)

# Phase M3-2 wrappers (Phase 7/8/11 + M2-1 + M2-3) live in
# `src/newton_step_HG_M3_2.jl`, included from `dfmm.jl` after the
# corresponding M1/M2 files (`tracers.jl`, `stochastic_injection.jl`,
# `amr_1d.jl`) so the wrappers can reference their M1 counterparts.

# ─────────────────────────────────────────────────────────────────────
# Phase M3-3b: native HG-side 2D Newton step driver (no Berry; θ_R fixed)
# ─────────────────────────────────────────────────────────────────────
#
# Counterpart of M3-1's `det_step_HG!` for the 2D Cholesky-sector EL
# residual `cholesky_el_residual_2D!` (defined in `src/eom.jl`).
# Differences vs the 1D path:
#
#   • The 2D residual operates directly on a `HierarchicalMesh{2}` +
#     `PolynomialFieldSet` allocated by
#     `allocate_cholesky_2d_fields(mesh)` from `src/setups_2d.jl`.
#     There is NO `cache_mesh::Mesh1D` shim — this is the first native
#     HG-side EL residual in dfmm.
#   • Newton sparsity comes from `cell_adjacency_sparsity(mesh; depth=1)`
#     Kron-producted with a dense `8×8` block (per-cell coupling).
#   • Boundary handling is via M3-2b Swap 8's `FrameBoundaries{2}` +
#     `BCKind` framework. Periodic wrap is resolved upstream by
#     `face_neighbors_with_bcs`; closed boundaries (REFLECTING /
#     DIRICHLET) fall through to the residual's own ghost-cell synth
#     (mirror-self for REFLECTING in M3-3b).
#
# What M3-3b's driver does NOT do (per the M3-3 design note §9):
#
#   • Berry coupling — `θ_R` is read from the IC and held fixed across
#     the step; the Berry kinetic 1-form contribution to the residual
#     rows is omitted. M3-3c will plug it in.
#   • Off-diagonal `β_{12}, β_{21}` — pinned to zero (omitted from the
#     field set per Q3 of the design note's §10).
#   • Per-axis γ AMR / realizability — M3-3d / M3-3e scope.

using HierarchicalGrids: cell_adjacency_sparsity

"""
    det_step_2d_HG!(fields::PolynomialFieldSet,
                     mesh::HierarchicalMesh{2},
                     frame::EulerianFrame{2, T},
                     leaves::AbstractVector{<:Integer},
                     bc_spec::FrameBoundaries{2},
                     dt::Real;
                     M_vv_override = nothing,
                     ρ_ref::Real = 1.0,
                     sparse_jac::Bool = true,
                     abstol::Real = 1e-13, reltol::Real = 1e-13,
                     maxiters::Int = 50) where {T<:Real}

Advance the 2D Cholesky-sector field set by one timestep `dt` via an
implicit-midpoint Newton solve of `cholesky_el_residual_2D!`. Mutates
the 8 Newton-unknown slots `(x_a, u_a, α_a, β_a)` of `fields`
in-place; the `(θ_R, s, Pp, Q)` slots are left untouched (M3-3b holds
them fixed).

Arguments mirror the M3-3a allocation contract:

  • `fields` — built by `allocate_cholesky_2d_fields(mesh)`.
  • `mesh::HierarchicalMesh{2}` — must be balanced (Q4 of the design
    note's §10).
  • `frame::EulerianFrame{2, T}` — supplies per-cell physical extents
    via `cell_physical_box`.
  • `leaves` — leaf-iteration order (typically `enumerate_leaves(mesh)`).
  • `bc_spec` — 2D `FrameBoundaries{2}` (e.g. periodic-x, reflecting-y).

Keyword arguments:

  • `M_vv_override::Union{Nothing, NTuple{2, Real}}` — when not
    `nothing`, override the EOS `M_vv(J, s)` per axis with the
    supplied constants. M3-3b's zero-strain regression and
    dimension-lift gate use this to decouple from EOS specifics.
  • `ρ_ref::Real` — reference density for the per-axis Lagrangian
    mass step `Δm_a = ρ_ref · cell_extent_a`. Default 1.0.
  • `sparse_jac::Bool` — use HG's `cell_adjacency_sparsity` (Kron with
    8×8 dense block) as the Jacobian sparsity prototype.
  • `abstol`, `reltol`, `maxiters` — forwarded to NonlinearSolve.

Returns `fields` for convenience.

# Constraints

  • `mesh.balanced == true` (asserted).
  • The 2D residual is for the dimension-lifted Cholesky sector; per
    M3-3b's scope it is **not** physics-complete — Berry, off-diag β,
    per-axis γ AMR all land in M3-3c–M3-3e.
"""
function det_step_2d_HG!(fields::PolynomialFieldSet,
                          mesh::HierarchicalMesh{2},
                          frame::EulerianFrame{2, T},
                          leaves::AbstractVector{<:Integer},
                          bc_spec::FrameBoundaries{2},
                          dt::Real;
                          M_vv_override = nothing,
                          ρ_ref::Real = 1.0,
                          sparse_jac::Bool = true,
                          abstol::Real = 1e-13,
                          reltol::Real = 1e-13,
                          maxiters::Int = 50) where {T<:Real}
    @assert mesh.balanced == true "det_step_2d_HG! requires mesh.balanced == true"
    N = length(leaves)
    @assert N >= 1 "leaves vector must be non-empty"

    aux = build_residual_aux_2D(fields, mesh, frame, leaves, bc_spec;
                                 M_vv_override = M_vv_override,
                                 ρ_ref = ρ_ref)
    y_n = pack_state_2d(fields, leaves)

    # Initial guess: explicit Euler from the boxed equations evaluated
    # at q_n. For zero-strain configurations this is already a tight
    # initial guess; richer configurations (active strain) are M3-3c+.
    y0 = copy(y_n)

    # NonlinearProblem closure.
    f_in_place = (F, u, p_in) -> begin
        cholesky_el_residual_2D!(F, u, p_in.y_n, p_in.aux, p_in.dt)
        return nothing
    end
    p = (y_n = y_n, aux = aux, dt = T(dt))

    # Sparse-Jacobian prototype: cell_adjacency Kron with an 8×8 dense
    # block. M3-3b uses depth = 1 (per-axis pressure stencil reaches one
    # face-hop along each axis).
    if sparse_jac && N >= 4
        cell_sparsity = cell_adjacency_sparsity(mesh; depth = 1,
                                                  leaves_only = true)
        # Kron with a dense 8×8 Bool block. Materialize as
        # SparseMatrixCSC{Float64, Int} for NonlinearSolve coloring.
        rows = Int[]
        cols = Int[]
        cs_rows, cs_cols, _ = SparseArrays.findnz(cell_sparsity)
        sizehint!(rows, length(cs_rows) * 64)
        sizehint!(cols, length(cs_rows) * 64)
        @inbounds for k in eachindex(cs_rows)
            i = Int(cs_rows[k])
            j = Int(cs_cols[k])
            for r in 1:8, c in 1:8
                push!(rows, 8 * (i - 1) + r)
                push!(cols, 8 * (j - 1) + c)
            end
        end
        jac_proto = _SA.sparse(rows, cols, ones(Float64, length(rows)),
                                8 * N, 8 * N, max)
        nf = NonlinearFunction(f_in_place; jac_prototype = jac_proto)
        prob = NonlinearProblem(nf, y0, p)
        sol = solve(
            prob,
            NewtonRaphson(; autodiff = AutoForwardDiff());
            abstol = abstol, reltol = reltol, maxiters = maxiters,
        )
    else
        prob = NonlinearProblem(f_in_place, y0, p)
        sol = solve(
            prob,
            NewtonRaphson(; autodiff = AutoForwardDiff());
            abstol = abstol, reltol = reltol, maxiters = maxiters,
        )
    end

    # Verify residual norm (mirror the 1D `det_step!` pattern). Newton
    # may report `Stalled` / `MaxIters` even when the residual is at
    # round-off on smooth nearly-converged problems.
    F_check = similar(sol.u)
    cholesky_el_residual_2D!(F_check, sol.u, y_n, aux, T(dt))
    res_norm = maximum(abs, F_check)
    if res_norm > 10 * abstol && sol.retcode != ReturnCode.Success
        error("det_step_2d_HG! Newton solve failed: retcode = $(sol.retcode), " *
              "‖residual‖∞ = $res_norm")
    end

    unpack_state_2d!(fields, leaves, sol.u)
    return fields
end

# ─────────────────────────────────────────────────────────────────────
# Phase M3-3c: native HG-side 2D Newton step driver WITH Berry coupling
# and θ_R as a Newton unknown
# ─────────────────────────────────────────────────────────────────────
#
# Counterpart of M3-3b's `det_step_2d_HG!`. The residual is now
# `cholesky_el_residual_2D_berry!` (in `src/eom.jl`) with 9 dof per
# cell `(x_a, u_a, α_a, β_a, θ_R)`. The Newton sparsity grows from
# `cell_adjacency ⊗ 8×8` to `cell_adjacency ⊗ 9×9` (one row+column
# more per cell; ~17 nonzeros added per cell). The Berry coupling
# introduces θ_R-α / θ_R-β off-diagonals that the dense per-cell
# 9×9 sparsity covers.

"""
    det_step_2d_berry_HG!(fields::PolynomialFieldSet,
                           mesh::HierarchicalMesh{2},
                           frame::EulerianFrame{2, T},
                           leaves::AbstractVector{<:Integer},
                           bc_spec::FrameBoundaries{2},
                           dt::Real;
                           M_vv_override = nothing,
                           ρ_ref::Real = 1.0,
                           sparse_jac::Bool = true,
                           abstol::Real = 1e-13,
                           reltol::Real = 1e-13,
                           maxiters::Int = 50) where {T<:Real}

Advance the 2D Cholesky-sector field set by one timestep `dt` via an
implicit-midpoint Newton solve of `cholesky_el_residual_2D_berry!`
(M3-3c: Berry coupling + θ_R as Newton unknown). Mutates the 9 Newton-
unknown slots `(x_a, u_a, α_a, β_a, θ_R)` of `fields` in-place; the
`(s, Pp, Q)` slots are left untouched.

Arguments mirror `det_step_2d_HG!` (M3-3b). The only structural
differences are:

  • The Newton system is `9 N × 9 N` rather than `8 N × 8 N`.
  • The Jacobian sparsity prototype is `cell_adjacency_sparsity` ⊗
    a dense 9×9 block (vs M3-3b's 8×8).
  • The residual contains Berry α/β-modification terms and an explicit
    θ_R kinematic equation row.

Returns `fields` for convenience.

# Constraints

  • `mesh.balanced == true` (asserted).
  • The off-diagonal `β_{12}, β_{21}` are pinned to zero (omitted from
    the field set per Q3 of the design note).
  • The kinematic drive of `θ̇_R` is zero in M3-3c; off-diagonal
    velocity gradients (KH instability, D.1 falsifier) are M3-3d /
    M3-6 work.
"""
function det_step_2d_berry_HG!(fields::PolynomialFieldSet,
                                 mesh::HierarchicalMesh{2},
                                 frame::EulerianFrame{2, T},
                                 leaves::AbstractVector{<:Integer},
                                 bc_spec::FrameBoundaries{2},
                                 dt::Real;
                                 M_vv_override = nothing,
                                 ρ_ref::Real = 1.0,
                                 sparse_jac::Bool = true,
                                 abstol::Real = 1e-13,
                                 reltol::Real = 1e-13,
                                 maxiters::Int = 50,
                                 project_kind::Symbol = :none,
                                 realizability_headroom::Real = 1.05,
                                 Mvv_floor::Real = 1e-2,
                                 pressure_floor::Real = 1e-8,
                                 proj_stats = nothing) where {T<:Real}
    @assert mesh.balanced == true "det_step_2d_berry_HG! requires mesh.balanced == true"
    N = length(leaves)
    @assert N >= 1 "leaves vector must be non-empty"

    aux = build_residual_aux_2D(fields, mesh, frame, leaves, bc_spec;
                                 M_vv_override = M_vv_override,
                                 ρ_ref = ρ_ref)
    y_n = pack_state_2d_berry(fields, leaves)
    y0 = copy(y_n)

    f_in_place = (F, u, p_in) -> begin
        cholesky_el_residual_2D_berry!(F, u, p_in.y_n, p_in.aux, p_in.dt)
        return nothing
    end
    p = (y_n = y_n, aux = aux, dt = T(dt))

    if sparse_jac && N >= 4
        cell_sparsity = cell_adjacency_sparsity(mesh; depth = 1,
                                                  leaves_only = true)
        rows = Int[]
        cols = Int[]
        cs_rows, cs_cols, _ = SparseArrays.findnz(cell_sparsity)
        # Cap at 9×9 = 81 nonzeros per cell-cell adjacency entry.
        sizehint!(rows, length(cs_rows) * 81)
        sizehint!(cols, length(cs_rows) * 81)
        @inbounds for k in eachindex(cs_rows)
            i = Int(cs_rows[k])
            j = Int(cs_cols[k])
            for r in 1:9, c in 1:9
                push!(rows, 9 * (i - 1) + r)
                push!(cols, 9 * (j - 1) + c)
            end
        end
        jac_proto = _SA.sparse(rows, cols, ones(Float64, length(rows)),
                                9 * N, 9 * N, max)
        nf = NonlinearFunction(f_in_place; jac_prototype = jac_proto)
        prob = NonlinearProblem(nf, y0, p)
        sol = solve(
            prob,
            NewtonRaphson(; autodiff = AutoForwardDiff());
            abstol = abstol, reltol = reltol, maxiters = maxiters,
        )
    else
        prob = NonlinearProblem(f_in_place, y0, p)
        sol = solve(
            prob,
            NewtonRaphson(; autodiff = AutoForwardDiff());
            abstol = abstol, reltol = reltol, maxiters = maxiters,
        )
    end

    F_check = similar(sol.u)
    cholesky_el_residual_2D_berry!(F_check, sol.u, y_n, aux, T(dt))
    res_norm = maximum(abs, F_check)
    if res_norm > 10 * abstol && sol.retcode != ReturnCode.Success
        error("det_step_2d_berry_HG! Newton solve failed: retcode = $(sol.retcode), " *
              "‖residual‖∞ = $res_norm")
    end

    unpack_state_2d_berry!(fields, leaves, sol.u)

    # M3-3d post-Newton operator-split: per-axis realizability projection.
    # With `project_kind = :none` (default) this is a no-op and the driver
    # remains bit-for-bit identical to M3-3c. With `:reanchor`, raise `s`
    # so `M_vv ≥ headroom · max_a β_a²` per leaf — the per-axis 2D
    # generalisation of the M2-3 1D projection, mirroring how the 1D
    # `det_run_stochastic!` wraps `det_step!` with pre/post projections.
    realizability_project_2d!(fields, leaves;
                               project_kind = project_kind,
                               headroom = realizability_headroom,
                               Mvv_floor = Mvv_floor,
                               pressure_floor = pressure_floor,
                               ρ_ref = ρ_ref,
                               stats = proj_stats)

    return fields
end
