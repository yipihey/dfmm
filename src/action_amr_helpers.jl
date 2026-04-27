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

# ─────────────────────────────────────────────────────────────────────
# Phase M3-3d: per-axis action-AMR indicator on HierarchicalMesh{2}
# ─────────────────────────────────────────────────────────────────────
#
# The 2D Cholesky-sector field set carries 10 Newton-unknown scalars
# `(x_a, u_a, α_a, β_a)_{a=1,2} + θ_R + s` per leaf (12 with the post-
# Newton `Pp, Q` slots). The M2-1 action-AMR indicator generalises per-
# axis: we evaluate ΔS_cell separately along each axis using the
# axis-a quantities `(α_a, β_a, u_a)` and the per-axis γ collapse
# marker, then aggregate as `max_a` so a cell refines along the most
# action-energetic axis. This selects collapsing axes (γ → 0) without
# triggering on the trivial axis (γ = O(1)). See §4.3 + §6.5 of
# `reference/notes_M3_3_2d_cholesky_berry.md`.
#
# The indicator is evaluated using the per-axis 4-stencil `(j_lo, j,
# j_hi)` along axis `a`, where `(j_lo, j_hi)` come from the same
# face-neighbor table that the 2D residual consumes
# (`build_face_neighbor_tables`).

using HierarchicalGrids: HierarchicalMesh, EulerianFrame,
    register_refinement_listener!, unregister_refinement_listener!,
    enumerate_leaves, refine_by_indicator!,
    n_cells as hg_n_cells_amr, RefinementEvent, ListenerHandle
using HierarchicalGrids.AMR: step_with_amr!

"""
    action_error_indicator_2d_per_axis(fields, mesh, frame, leaves, bc_spec;
                                        ρ_ref=1.0, Gamma=GAMMA_LAW_DEFAULT,
                                        M_vv_override=nothing)
        -> Vector{Float64}

Per-leaf, per-axis-aggregated action-error indicator on a 2D
`HierarchicalMesh{2}`. The output `out[i]` is

    max_a [|d2(α_a)| + |d2(β_a)| + |d2(u_a)|/c_s + 0.01·γ_inv_marker_a]

evaluated as a 3-point second difference along axis `a`, where
`(j_lo_a, j_hi_a)` are pulled from `build_face_neighbor_tables(mesh,
leaves, bc_spec)` (so periodic wrap-around / reflecting BCs are
honoured upstream). The shared `s`-curvature term `|d2(s)|` is added
once per leaf (not per axis) since `s` is a scalar shared across axes.

The per-axis γ marker `γ_inv_marker_a = √Mvv / max(γ_a, ε) − 1` is the
same Hessian-degeneracy proxy used in the 1D `action_error_indicator`,
evaluated per axis. It is what gives the indicator its **per-axis
selectivity**: when `γ_a → 0` along the collapsing axis (`a = 1` for
the C.2 cold sinusoid with `k_y = 0`), the marker spikes along that
axis; the trivial axis stays `O(1)` so its marker stays near zero.
The `max_a` aggregation then picks the collapsing axis as the cell's
indicator value.

# Boundary handling

When `face_lo_idx[a][i] == 0` or `face_hi_idx[a][i] == 0` (out-of-domain
after BC processing), the indicator drops the corresponding axis-a
contribution to zero (one-sided second-difference is unreliable at
the indicator level — the 1D action-error indicator does the same on
non-periodic boundaries).

# Use cases

  • Drives `refine_by_indicator!` on `HierarchicalMesh{2}` via
    `step_with_amr_2d!` (this file).
  • The §6.5 selectivity test: with `k_y = 0` cold sinusoid and the
    indicator running per step, refinement events fire only on cells
    along the collapsing-x axis; the y-direction stays unrefined.
"""
function action_error_indicator_2d_per_axis(fields, mesh::HierarchicalMesh{2},
                                            frame::EulerianFrame{2, T},
                                            leaves::AbstractVector{<:Integer},
                                            bc_spec;
                                            ρ_ref::Real = 1.0,
                                            Gamma::Real = GAMMA_LAW_DEFAULT,
                                            M_vv_override = nothing) where {T}
    N = length(leaves)
    out = zeros(Float64, N)
    if N < 3
        return out
    end

    face_lo_idx, face_hi_idx = build_face_neighbor_tables(mesh, leaves, bc_spec)

    # Pre-compute per-leaf primitives.
    α1 = Vector{Float64}(undef, N); α2 = Vector{Float64}(undef, N)
    β1 = Vector{Float64}(undef, N); β2 = Vector{Float64}(undef, N)
    u1 = Vector{Float64}(undef, N); u2 = Vector{Float64}(undef, N)
    s_v = Vector{Float64}(undef, N)
    Mvv_v = Vector{Float64}(undef, N)
    cs_v = Vector{Float64}(undef, N)
    @inbounds for (i, ci) in enumerate(leaves)
        α1[i] = Float64(fields.α_1[ci][1])
        α2[i] = Float64(fields.α_2[ci][1])
        β1[i] = Float64(fields.β_1[ci][1])
        β2[i] = Float64(fields.β_2[ci][1])
        u1[i] = Float64(fields.u_1[ci][1])
        u2[i] = Float64(fields.u_2[ci][1])
        s_v[i] = Float64(fields.s[ci][1])
        if M_vv_override !== nothing
            Mvv_v[i] = max(Float64(M_vv_override[1]),
                            Float64(M_vv_override[2]))
        else
            Mvv_v[i] = Float64(Mvv(1.0 / Float64(ρ_ref), s_v[i];
                                    Gamma = Float64(Gamma)))
        end
        cs_v[i] = sqrt(max(Float64(Gamma) * Mvv_v[i], eps(Float64)))
    end

    @inbounds for i in 1:N
        # Shared s-curvature contribution (uses axis-1 stencil; the
        # 2D field set has a single scalar `s` per leaf).
        d2_s_axis1 = zero(Float64)
        ilo = face_lo_idx[1][i]
        ihi = face_hi_idx[1][i]
        if ilo > 0 && ihi > 0
            d2_s_axis1 = abs(s_v[ihi] - 2 * s_v[i] + s_v[ilo])
        end

        # Per-axis contributions.
        max_axis = zero(Float64)
        for a in 1:2
            ilo_a = face_lo_idx[a][i]
            ihi_a = face_hi_idx[a][i]
            if ilo_a == 0 || ihi_a == 0
                continue
            end
            if a == 1
                d2_α = abs(α1[ihi_a] - 2 * α1[i] + α1[ilo_a])
                d2_β = abs(β1[ihi_a] - 2 * β1[i] + β1[ilo_a])
                d2_u = abs(u1[ihi_a] - 2 * u1[i] + u1[ilo_a])
                β_self = β1[i]
            else
                d2_α = abs(α2[ihi_a] - 2 * α2[i] + α2[ilo_a])
                d2_β = abs(β2[ihi_a] - 2 * β2[i] + β2[ilo_a])
                d2_u = abs(u2[ihi_a] - 2 * u2[i] + u2[ilo_a])
                β_self = β2[i]
            end
            d2_u_norm = d2_u / max(cs_v[i], eps(Float64))
            γ²_a = max(Mvv_v[i] - β_self * β_self, 0.0)
            γ_inv = sqrt(max(Mvv_v[i], eps(Float64))) /
                    sqrt(max(γ²_a, eps(Float64)))
            γ_marker = γ_inv - 1.0
            axis_val = d2_α + d2_β + d2_u_norm + 0.01 * γ_marker
            max_axis = max(max_axis, axis_val)
        end
        out[i] = max_axis + d2_s_axis1
    end
    return out
end

"""
    register_field_set_on_refine!(fields, mesh::HierarchicalMesh{D};
                                   default_value=0.0)
        -> ListenerHandle

Register an HG `register_refinement_listener!` callback that keeps a
`PolynomialFieldSet` consistent across refine/coarsen events on
`mesh`. After each `refine_cells!` / `coarsen_cells!` (or
`refine_by_indicator!`) call, the listener:

  1. Resizes every named field's coefficient storage to
     `n_cells(mesh)` post-event (HG does NOT track field sets — the
     dfmm-side has to do it explicitly).
  2. For each refined parent → children mapping in
     `event.new_children`, copies the parent's order-0 cell-average
     into every child (piecewise-constant prolongation, the natural
     order-0 reconstruction).
  3. For each coarsened parent in `event.coarsened_parents`, averages
     the (now-removed) children's order-0 values into the parent
     coefficient (mass-conservative for `:rho`-like fields when the
     children are equal-volume, which is true for isotropic refinement).

The listener handle is returned so callers can `unregister_refinement_listener!`
when the field set's lifecycle ends. **The listener captures `fields`
by reference** — any subsequent reallocation of `fields` (which
shouldn't happen for `PolynomialFieldSet` SoA storage; the storage
arrays are mutated in place) would be a use-after-free.

# Limits

  • Order-0 piecewise-constant prolongation only. Higher-order
    Bernstein restriction/prolongation is M3-4 / M3-5 work.
  • The listener walks all named fields in the field set; the dfmm
    Cholesky-sector layout (12 named fields per the M3-3a allocator)
    is fully covered.
"""
function register_field_set_on_refine!(fields, mesh::HierarchicalMesh{D};
                                       default_value::Real = 0.0) where {D}
    # Names of the public named scalar fields. The HG `PolynomialFieldSet`
    # exposes the field names via `field_names(fields)`, with the SoA
    # storage backing accessible at `fields.storage` as a NamedTuple of
    # `Vector{T}` (one entry per coefficient × cell). For `MonomialBasis{D, 0}`
    # there is exactly one coefficient per cell, so the storage vector
    # length equals `n_cells(mesh)`.
    names = HierarchicalGrids.field_names(fields)
    T_default = Float64(default_value)
    cb = function(event::RefinementEvent)
        new_n = Int(HierarchicalGrids.n_cells(mesh))
        index_remap = event.index_remap
        old_n = length(index_remap)
        # Per name: snapshot old values, reallocate, prolongate into
        # children, average across coarsened groups.
        for name in names
            storage_arr = getproperty(fields.storage, name)
            old_vals = copy(storage_arr)::Vector{Float64}
            new_vals = fill(T_default, new_n)
            @inbounds for old_i in 1:old_n
                new_i = Int(index_remap[old_i])
                if new_i != 0
                    new_vals[new_i] = old_vals[old_i]
                end
            end
            # Prolongate refined parents into their children
            # (piecewise-constant order-0 prolongation).
            for k in eachindex(event.refined_parents)
                old_parent = Int(event.refined_parents[k])
                parent_val = old_vals[old_parent]
                child_range = event.new_children[k]
                for child in child_range
                    new_vals[Int(child)] = parent_val
                end
            end
            # Coarsened parents: HG's coarsen_cells! event populates
            # `removed_old_indices` in groups of 2^D per parent in
            # `coarsened_parents` order. Average each group.
            n_per_group = 1 << D
            for k in eachindex(event.coarsened_parents)
                new_parent = Int(event.coarsened_parents[k])
                child_acc = 0.0
                for offset in 0:(n_per_group - 1)
                    old_child = Int(event.removed_old_indices[
                        (k - 1) * n_per_group + offset + 1])
                    child_acc += old_vals[old_child]
                end
                new_vals[new_parent] = child_acc / n_per_group
            end
            # In-place resize + write-back.
            resize!(storage_arr, new_n)
            @inbounds for i in 1:new_n
                storage_arr[i] = new_vals[i]
            end
        end
        return nothing
    end
    return register_refinement_listener!(mesh, cb)
end

"""
    step_with_amr_2d!(fields, mesh::HierarchicalMesh{2},
                      frame::EulerianFrame{2, T},
                      bc_spec, dt, n_steps;
                      refine_threshold,
                      coarsen_threshold = refine_threshold / 4,
                      hysteresis_steps = 3,
                      max_level = typemax(Int),
                      isotropic = true,
                      M_vv_override = nothing,
                      ρ_ref = 1.0,
                      project_kind = :none,
                      realizability_headroom = 1.05,
                      Mvv_floor = 1e-2,
                      pressure_floor = 1e-8,
                      proj_stats = nothing,
                      newton_kwargs = (;)) where {T}

End-to-end M3-3d AMR-driven 2D run on a `HierarchicalMesh{2}`. Wraps
HG's `step_with_amr!` driver with:

  • A `step!` callback that (a) calls `det_step_2d_berry_HG!` per step
    on the current leaves, then (b) refreshes the leaves vector after
    each AMR cycle.
  • An `indicator(mesh)` callback that returns the per-axis-aggregated
    action-error indicator `action_error_indicator_2d_per_axis` mapped
    onto **mesh-cell indices** (length `n_cells(mesh)`; non-leaf cells
    get value 0 so they can never trigger refinement).
  • A refinement listener registered via `register_field_set_on_refine!`
    that prolongates / restricts the field set across each
    `refine_by_indicator!` event.

Returns `n_steps` (the number of physics steps taken). Mutates
`fields` and `mesh` in place. Closes M3-2b's deferred Swaps 2+3 for
the 2D scope (per the M3-3d brief): this is the first dfmm path that
uses HG's `register_refinement_listener!` + `step_with_amr!` natively.
"""
function step_with_amr_2d!(fields, mesh::HierarchicalMesh{2},
                            frame::EulerianFrame{2, T},
                            bc_spec, dt::Real, n_steps::Integer;
                            refine_threshold::Real,
                            coarsen_threshold::Real = refine_threshold / 4,
                            hysteresis_steps::Integer = 3,
                            max_level::Integer = typemax(Int),
                            isotropic::Bool = true,
                            M_vv_override = nothing,
                            ρ_ref::Real = 1.0,
                            project_kind::Symbol = :none,
                            realizability_headroom::Real = 1.05,
                            Mvv_floor::Real = 1e-2,
                            pressure_floor::Real = 1e-8,
                            proj_stats = nothing,
                            newton_kwargs::NamedTuple = NamedTuple()) where {T}
    @assert mesh.balanced "step_with_amr_2d! requires mesh.balanced == true"

    # State carried across the step!/indicator closures.
    state = Ref(enumerate_leaves(mesh))

    handle = register_field_set_on_refine!(fields, mesh)
    try
        step_fn = function(state_ref, frame_arg)
            leaves_now = state_ref[]
            # Refresh leaves if a refinement happened since last call
            # (the listener has updated `fields` by now).
            if length(leaves_now) != length(enumerate_leaves(frame_arg.mesh))
                leaves_now = enumerate_leaves(frame_arg.mesh)
                state_ref[] = leaves_now
            end
            det_step_2d_berry_HG!(fields, frame_arg.mesh, frame_arg, leaves_now,
                                   bc_spec, dt;
                                   M_vv_override = M_vv_override,
                                   ρ_ref = ρ_ref,
                                   project_kind = project_kind,
                                   realizability_headroom = realizability_headroom,
                                   Mvv_floor = Mvv_floor,
                                   pressure_floor = pressure_floor,
                                   proj_stats = proj_stats,
                                   newton_kwargs...)
            return nothing
        end

        ind_fn = function(mesh_arg)
            leaves_now = enumerate_leaves(mesh_arg)
            state[] = leaves_now
            ind_leaf = action_error_indicator_2d_per_axis(
                fields, mesh_arg, frame, leaves_now, bc_spec;
                ρ_ref = ρ_ref, M_vv_override = M_vv_override)
            # Map leaf-major indicator to mesh-cell indicator. Non-leaves
            # are not eligible for refine_by_indicator anyway, but HG
            # still calls the indicator across all cells; we return 0
            # for non-leaves so they never trigger.
            nc = HierarchicalGrids.n_cells(mesh_arg)
            full = zeros(Float64, nc)
            @inbounds for (k, ci) in enumerate(leaves_now)
                full[ci] = ind_leaf[k]
            end
            return full
        end

        return step_with_amr!(fields, frame, step_fn, ind_fn, n_steps;
                               refine_threshold = refine_threshold,
                               coarsen_threshold = coarsen_threshold,
                               hysteresis_steps = hysteresis_steps,
                               max_level = max_level,
                               isotropic = isotropic)
    finally
        unregister_refinement_listener!(mesh, handle)
    end
end
