# action_amr_helpers.jl — HG-side Phase M3-2 (M2-1) action-AMR shim.
#
# This file implements the conservative refine/coarsen + indicator
# bookkeeping for a `DetMeshHG{T}` wrapper, mirroring `src/amr_1d.jl`'s
# operations on `Mesh1D{T,DetField{T}}`. Per M3-2's brief, the
# implementation here goes through the cached `Mesh1D` shim — the
# canonical M1 refine/coarsen primitives are called on the cache mesh,
# and the HG wrapper's auxiliary state (the `SimplicialMesh{1, T}`
# vertex topology, the order-0 `PolynomialFieldSet`, `Δm`, `p_half`)
# is rebuilt from scratch from the cache mesh. The cost is one per-AMR-
# event re-allocation of the simplicial mesh; the bit-equality
# property is automatic (the AMR call path is byte-identical to M1's).
#
# When HG ships a `register_on_refine!` callback API
# (design-guidance item #5 in `notes_HG_design_guidance.md`), this
# rebuild step can be replaced by an in-place HG-side update. M3-2
# defers that to a future milestone (M3-3 or later) and stays inside
# the cache-mesh shim pattern.

using HierarchicalGrids: SimplicialMesh

"""
    rebuild_HG_from_cache!(mesh::DetMeshHG)

Rebuild the HG wrapper's `SimplicialMesh{1, T}`, `PolynomialFieldSet`,
`Δm`, `p_half`, and `L_box` fields from the current state of
`mesh.cache_mesh`. Used internally by the AMR helpers after a
refine/coarsen event mutates the cache mesh's segment vector.

The HG simplicial mesh is rebuilt with periodic-wrap topology (as
in `DetMeshHG_from_arrays`) — even on `bc = :inflow_outflow` paths
the simplicial mesh wiring is periodic; M1's `det_step!` handles
the BC via the `bc` kwarg / inflow-outflow ghost cells, not via
the simplex-neighbour matrix.
"""
function rebuild_HG_from_cache!(mesh::DetMeshHG{T}) where {T<:Real}
    cache = mesh.cache_mesh
    N = n_segments(cache)
    Lt = T(cache.L_box)

    # Rebuild the per-mass-coordinate vertex positions. Cumulative sum of
    # cache_mesh segment Δm.
    cum_m = T(0)
    pos_m = Vector{NTuple{1, T}}(undef, N + 1)
    pos_m[1] = (T(0),)
    @inbounds for j in 1:N
        cum_m += T(cache.segments[j].Δm)
        pos_m[j + 1] = (cum_m,)
    end
    sv = Matrix{Int32}(undef, 2, N)
    sn = zeros(Int32, 2, N)
    @inbounds for j in 1:N
        sv[1, j] = j
        sv[2, j] = j + 1
        sn[1, j] = j > 1 ? Int32(j - 1) : Int32(N)
        sn[2, j] = j < N ? Int32(j + 1) : Int32(1)
    end
    mesh.mesh = SimplicialMesh{1, T}(pos_m, sv, sn)

    # Reallocate the field set to match the new N and populate from the
    # cache mesh segments.
    mesh.fields = allocate_detfield_HG(N; T = T)
    @inbounds for j in 1:N
        write_detfield!(mesh.fields, j, cache.segments[j].state)
    end

    # Update Δm, p_half, L_box from the cache mesh.
    if length(mesh.Δm) != N
        resize!(mesh.Δm, N)
    end
    @inbounds for j in 1:N
        mesh.Δm[j] = T(cache.segments[j].Δm)
    end
    if length(mesh.p_half) != N
        resize!(mesh.p_half, N)
    end
    @inbounds for i in 1:N
        mesh.p_half[i] = T(cache.p_half[i])
    end
    mesh.L_box = Lt
    return mesh
end

"""
    refine_segment_HG!(mesh::DetMeshHG, j::Integer; tracers = nothing)
        -> DetMeshHG

Split simplex `j` of the HG-substrate mesh into two equal-mass
daughters. Mirrors `refine_segment!` on `Mesh1D`. Bit-exact-equivalent:
the cache mesh's `refine_segment!` is invoked, then the HG state is
rebuilt from the cache.

Optional `tracers::TracerMeshHG` is forwarded to the cache mesh's
embedded `TracerMesh` (which wraps the same cache mesh); after
the cache mesh refines, the HG-side tracer view sees the resized
matrix automatically because it shares storage with the cache mesh.
"""
function refine_segment_HG!(mesh::DetMeshHG{T}, j::Integer;
                             tracers = nothing) where {T<:Real}
    sync_cache_from_HG!(mesh)
    if tracers === nothing
        refine_segment!(mesh.cache_mesh, j)
    else
        refine_segment!(mesh.cache_mesh, j; tracers = tracers.tm)
    end
    rebuild_HG_from_cache!(mesh)
    return mesh
end

"""
    coarsen_segment_pair_HG!(mesh::DetMeshHG, j::Integer; tracers = nothing,
                             Gamma = GAMMA_LAW_DEFAULT) -> DetMeshHG

Merge simplices `j` and `j+1` of the HG-substrate mesh into one.
Mirrors `coarsen_segment_pair!` on `Mesh1D`. Bit-exact-equivalent.
"""
function coarsen_segment_pair_HG!(mesh::DetMeshHG{T}, j::Integer;
                                   tracers = nothing,
                                   Gamma::Real = GAMMA_LAW_DEFAULT) where {T<:Real}
    sync_cache_from_HG!(mesh)
    if tracers === nothing
        coarsen_segment_pair!(mesh.cache_mesh, j; Gamma = Gamma)
    else
        coarsen_segment_pair!(mesh.cache_mesh, j; tracers = tracers.tm,
                               Gamma = Gamma)
    end
    rebuild_HG_from_cache!(mesh)
    return mesh
end

"""
    action_error_indicator_HG(mesh::DetMeshHG; dt = nothing,
                              Gamma = GAMMA_LAW_DEFAULT) -> Vector

Per-segment ΔS_cell action-error indicator on the HG-substrate mesh.
Bit-equal-equivalent to `action_error_indicator(mesh_M1)` on the
underlying cache mesh.
"""
function action_error_indicator_HG(mesh::DetMeshHG{T};
                                    dt::Union{Real,Nothing} = nothing,
                                    Gamma::Real = GAMMA_LAW_DEFAULT) where {T<:Real}
    sync_cache_from_HG!(mesh)
    return action_error_indicator(mesh.cache_mesh; dt = dt, Gamma = Gamma)
end

"""
    gradient_indicator_HG(mesh::DetMeshHG; field = :rho,
                          Gamma = GAMMA_LAW_DEFAULT) -> Vector

Per-segment relative-gradient indicator on the HG-substrate mesh.
Bit-equal-equivalent to `gradient_indicator(mesh_M1; field = field)`.
"""
function gradient_indicator_HG(mesh::DetMeshHG{T};
                                field::Symbol = :rho,
                                Gamma::Real = GAMMA_LAW_DEFAULT) where {T<:Real}
    sync_cache_from_HG!(mesh)
    return gradient_indicator(mesh.cache_mesh; field = field, Gamma = Gamma)
end

"""
    amr_step_HG!(mesh::DetMeshHG, indicator, τ_refine,
                 τ_coarsen = τ_refine/4; tracers = nothing,
                 max_segments = 4096, min_segments = 4)

Apply one round of refine / coarsen events to a `DetMeshHG{T}`
mesh based on `indicator`. Mirrors `amr_step!` on `Mesh1D`.

Returns `(n_refined, n_coarsened)` for diagnostics.
"""
function amr_step_HG!(mesh::DetMeshHG{T},
                      indicator::AbstractVector{<:Real},
                      τ_refine::Real,
                      τ_coarsen::Real = τ_refine / 4;
                      tracers = nothing,
                      max_segments::Int = 4096,
                      min_segments::Int = 4) where {T<:Real}
    sync_cache_from_HG!(mesh)
    if tracers === nothing
        result = amr_step!(mesh.cache_mesh, indicator, τ_refine, τ_coarsen;
                           max_segments = max_segments,
                           min_segments = min_segments)
    else
        result = amr_step!(mesh.cache_mesh, indicator, τ_refine, τ_coarsen;
                           tracers = tracers.tm,
                           max_segments = max_segments,
                           min_segments = min_segments)
    end
    rebuild_HG_from_cache!(mesh)
    return result
end
