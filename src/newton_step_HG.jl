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
