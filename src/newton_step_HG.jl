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
    n_simplices, n_elements, parallel_for_cells

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
periodic-wrap topology. We carry the wrap in `L_box` and in the
trailing-segment neighbour entries of the simplex-neighbour matrix
(see `periodic_simplicial_mesh_1D`).
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
    periodic_simplicial_mesh_1D(N::Integer, L::Real = 1.0; T = Float64)

Build a periodic-wrap-aware `SimplicialMesh{1, T}` of `N` simplices on
`[0, L]`. Vertex `i` sits at `(i-1) * L / N` for `i = 1, …, N+1`; the
last simplex's right-vertex (vertex `N+1`) closes the cycle by
linking back to vertex `1` through the simplex-neighbour matrix.

In M3-1 the HG mesh structure does not strictly require this
periodic linkage — the per-cell driver delegates to M1's
`Mesh1D{:periodic}` for the actual integration step — but the field
of trailing-edge neighbours is filled in for downstream M3-2/M3-3
phases that may iterate over neighbour stencils directly on the HG
mesh. The Lagrangian-mass coordinate is left for the caller to wire
through `Δm` (the HG simplex-volume API stays unaware of the mass
labelling).

Returns the mesh.
"""
function periodic_simplicial_mesh_1D(N::Integer, L::Real = 1.0;
                                      T::Type = Float64)
    @assert N >= 2 "periodic Phase-2 mesh requires N ≥ 2"
    Lt = T(L)
    positions = [(Lt * (i - 1) / N,) for i in 1:(N + 1)]
    sv = Matrix{Int32}(undef, 2, N)
    sn = zeros(Int32, 2, N)
    for j in 1:N
        sv[1, j] = j
        sv[2, j] = j + 1
        # Periodic neighbours: the face opposite vertex 1 is the left
        # neighbour, the face opposite vertex 2 is the right neighbour.
        sn[1, j] = j > 1 ? Int32(j - 1) : Int32(N)   # cyclic wrap
        sn[2, j] = j < N ? Int32(j + 1) : Int32(1)   # cyclic wrap
    end
    return SimplicialMesh{1, T}(positions, sv, sn)
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
    # returns the per-segment Δm in this convention.
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
        sn[1, j] = j > 1 ? Int32(j - 1) : Int32(N)
        sn[2, j] = j < N ? Int32(j + 1) : Int32(1)
    end
    hg_mesh = SimplicialMesh{1, T}(pos_m, sv, sn)

    # Allocate the HG field set, populate from the cache mesh.
    fields = allocate_detfield_HG(N; T = T)
    @inbounds for j in 1:N
        write_detfield!(fields, j, cache_mesh.segments[j].state)
    end

    Δm_vec = T[seg.Δm for seg in cache_mesh.segments]
    p_half = copy(cache_mesh.p_half)

    return DetMeshHG{T}(hg_mesh, fields, Δm_vec, Lt, p_half, bc, cache_mesh)
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
    det_step_HG!(mesh::DetMeshHG, dt; tau=nothing, sparse_jac=true,
                 q_kind=:none, c_q_quad=1.0, c_q_lin=0.5,
                 abstol=1e-13, reltol=1e-13, maxiters=50)

Advance the HG-substrate Phase-2/5/5b mesh by one timestep `dt`.
Bit-exact delegate to M1's `det_step!`: the cached `Mesh1D` is
refreshed from the HG field set, M1's `det_step!` is called with
identical kwargs, and the result is written back. The `tau` /
`q_kind` / etc. semantics are identical to M1.

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
    sync_cache_from_HG!(mesh)
    det_step!(mesh.cache_mesh, dt;
              tau = tau,
              sparse_jac = sparse_jac,
              q_kind = q_kind,
              c_q_quad = c_q_quad,
              c_q_lin  = c_q_lin,
              abstol = abstol, reltol = reltol,
              maxiters = maxiters)
    sync_HG_from_cache!(mesh)
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
