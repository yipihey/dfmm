# newton_step_HG_M3_2.jl — HG-side wrappers for M1 Phase 7 / 8 / 11
# and M2-1 / M2-3 sub-phases (Phase M3-2).
#
# These wrappers all delegate to their M1 counterparts through the
# `DetMeshHG{T}` cache_mesh shim. The wrappers therefore inherit
# byte-identical behaviour from M1 and the bit-exact-parity test gate
# is automatic.
#
# This file is included by `src/dfmm.jl` after the M1/M2 files
# (`tracers.jl`, `stochastic_injection.jl`, `amr_1d.jl`) that define
# the M1-side primitives the wrappers delegate to.
#
# See `reference/notes_M3_2_phase7811_m2_port.md` for the design
# write-up and `reference/MILESTONE_3_PLAN.md` Phase M3-2 for the
# milestone plan.

# ─────────────────────────────────────────────────────────────────────
# Phase 8 — variance-gamma stochastic injection (M3-2 Block 2).
# ─────────────────────────────────────────────────────────────────────

"""
    inject_vg_noise_HG!(mesh::DetMeshHG, dt; params, rng,
                        diag = InjectionDiagnostics(...),
                        proj_stats = nothing)
        -> (mesh, diag)

HG-side wrapper around `inject_vg_noise!`. Refreshes the cached
`Mesh1D` from the HG field set, calls `inject_vg_noise!` on the
cache mesh with identical kwargs, and writes the result back. With
the same `rng` (same seed) and `params`, the per-cell
`(δ(ρu), ΔKE_vol, eta, …)` outputs are byte-identical to the M1
path.

Returns `(mesh, diag)` so the caller can inspect the per-cell
diagnostics. The `proj_stats` accumulator (if supplied) tracks
realizability-projection events on the cache mesh — same bit-exact
values as the M1 path.
"""
function inject_vg_noise_HG!(mesh::DetMeshHG{T}, dt::Real;
                              params::NoiseInjectionParams,
                              rng,
                              diag::InjectionDiagnostics =
                                  InjectionDiagnostics(n_cells(mesh)),
                              proj_stats::Union{ProjectionStats,Nothing} = nothing) where {T<:Real}
    sync_cache_from_HG!(mesh)
    inject_vg_noise!(mesh.cache_mesh, dt;
                     params = params, rng = rng, diag = diag,
                     proj_stats = proj_stats)
    sync_HG_from_cache!(mesh)
    return mesh, diag
end

"""
    det_run_stochastic_HG!(mesh::DetMeshHG, dt, n_steps;
                            params, rng, tau = nothing,
                            q_kind = :none, c_q_quad = 1.0,
                            c_q_lin = 0.5, accumulator = nothing,
                            monitor_every = 0, proj_stats = nothing,
                            kwargs...)
        -> mesh

HG-side wrapper around `det_run_stochastic!`. Drives the full
operator-split (deterministic Newton step + post-Newton VG noise
injection + realizability projection) on the cache mesh and writes
back to the HG field set. Same bit-exact contract as the M1 driver:
identical RNG seed, params, and `q_kind` produce identical per-cell
state.
"""
function det_run_stochastic_HG!(mesh::DetMeshHG{T}, dt::Real,
                                 n_steps::Integer;
                                 params::NoiseInjectionParams,
                                 rng,
                                 tau::Union{Real,Nothing} = nothing,
                                 q_kind::Symbol = Q_KIND_NONE,
                                 c_q_quad::Real = 1.0,
                                 c_q_lin::Real  = 0.5,
                                 accumulator::Union{BurstStatsAccumulator,Nothing} = nothing,
                                 monitor_every::Int = 0,
                                 proj_stats::Union{ProjectionStats,Nothing} = nothing,
                                 kwargs...) where {T<:Real}
    sync_cache_from_HG!(mesh)
    det_run_stochastic!(mesh.cache_mesh, dt, n_steps;
                        params = params, rng = rng,
                        tau = tau, q_kind = q_kind,
                        c_q_quad = c_q_quad, c_q_lin = c_q_lin,
                        accumulator = accumulator,
                        monitor_every = monitor_every,
                        proj_stats = proj_stats,
                        kwargs...)
    sync_HG_from_cache!(mesh)
    return mesh
end

# ─────────────────────────────────────────────────────────────────────
# Phase 11 — passive tracers (M3-2 Block 3).
# ─────────────────────────────────────────────────────────────────────

"""
    TracerMeshHG{T<:Real}

A passive-scalar field bundle attached to a `DetMeshHG{T}` fluid
mesh. The HG-side counterpart to M1's `TracerMesh{T, M<:Mesh1D}`.

Internally a `TracerMeshHG` wraps an M1 `TracerMesh` whose `fluid`
field is the HG wrapper's cached `Mesh1D`. The HG path therefore
gets the exact same matrix-of-tracer-rows storage layout, the same
`set_tracer!` / `add_tracer!` / `tracer_at_position` semantics, and
the same bit-exact preservation property — at zero algorithmic
divergence from the M1 path.

Construct via `TracerMeshHG(fluid_HG; n_tracers, names)`. The
`tracers(tm)` matrix is the live tracer state and is not mutated by
`advect_tracers_HG!` (which is a no-op in the pure-Lagrangian
frame); the user can mutate it directly via `set_tracer!` /
`add_tracer!`.
"""
mutable struct TracerMeshHG{T<:Real}
    fluid::DetMeshHG{T}
    tm::TracerMesh{T, Mesh1D{T, DetField{T}}}
end

"""
    TracerMeshHG(fluid::DetMeshHG{T}; n_tracers = 1, names = nothing)
        -> TracerMeshHG

Allocate a `TracerMeshHG` for `n_tracers` rows on `fluid`'s cache
mesh. Mirrors `TracerMesh(fluid::Mesh1D; n_tracers, names)`.
"""
function TracerMeshHG(fluid::DetMeshHG{T};
                       n_tracers::Integer = 1,
                       names::Union{Nothing,AbstractVector{Symbol}} = nothing) where {T<:Real}
    sync_cache_from_HG!(fluid)
    tm = TracerMesh(fluid.cache_mesh; n_tracers = n_tracers, names = names)
    return TracerMeshHG{T}(fluid, tm)
end

"""
    advect_tracers_HG!(tm::TracerMeshHG, dt) -> TracerMeshHG

No-op step (pure-Lagrangian frame): the tracer matrix is never
mutated. Provided for symmetry with `det_step_HG!` driver loops.
"""
@inline advect_tracers_HG!(tm::TracerMeshHG, dt::Real) =
    (advect_tracers!(tm.tm, dt); tm)

# Forward the per-tracer accessors so callers can use the same names
# they used for `TracerMesh`. Each one delegates to the inner `tm`.

"""
    n_tracer_fields(tm::TracerMeshHG) -> Int
"""
n_tracer_fields(tm::TracerMeshHG) = n_tracer_fields(tm.tm)

"""
    n_tracer_segments(tm::TracerMeshHG) -> Int
"""
n_tracer_segments(tm::TracerMeshHG) = n_tracer_segments(tm.tm)

"""
    tracer_index(tm::TracerMeshHG, name::Symbol) -> Int
"""
tracer_index(tm::TracerMeshHG, name::Symbol) = tracer_index(tm.tm, name)

"""
    set_tracer!(tm::TracerMeshHG, name_or_index, values_or_f) -> TracerMeshHG
"""
function set_tracer!(tm::TracerMeshHG, key, values_or_f)
    set_tracer!(tm.tm, key, values_or_f)
    return tm
end

"""
    add_tracer!(tm::TracerMeshHG, name::Symbol, values_or_f) -> TracerMeshHG
"""
function add_tracer!(tm::TracerMeshHG{T}, name::Symbol, values_or_f) where {T}
    new_tm = add_tracer!(tm.tm, name, values_or_f)
    return TracerMeshHG{T}(tm.fluid, new_tm)
end

"""
    tracer_at_position(tm::TracerMeshHG, x::Real, k::Integer = 1) -> Real
"""
tracer_at_position(tm::TracerMeshHG, x::Real, k::Integer = 1) =
    tracer_at_position(tm.tm, x, k)

# ─────────────────────────────────────────────────────────────────────
# M2-3 — realizability projection (M3-2 Block 4b).
# ─────────────────────────────────────────────────────────────────────

"""
    realizability_project_HG!(mesh::DetMeshHG; kind = :reanchor,
                              headroom = 1.05, Mvv_floor = 1e-2,
                              pressure_floor = 1e-8, stats = nothing)
        -> mesh

HG-side wrapper around `realizability_project!`. Refreshes the
cached `Mesh1D` from the HG field set, calls
`realizability_project!` on the cache mesh, writes back. Bit-exact
parity with the M1 path.
"""
function realizability_project_HG!(mesh::DetMeshHG{T};
                                    kind::Symbol = :reanchor,
                                    headroom::Real = 1.05,
                                    Mvv_floor::Real = 1e-2,
                                    pressure_floor::Real = 1e-8,
                                    stats::Union{ProjectionStats,Nothing} = nothing) where {T<:Real}
    sync_cache_from_HG!(mesh)
    realizability_project!(mesh.cache_mesh;
                           kind = kind,
                           headroom = headroom,
                           Mvv_floor = Mvv_floor,
                           pressure_floor = pressure_floor,
                           stats = stats)
    sync_HG_from_cache!(mesh)
    return mesh
end
