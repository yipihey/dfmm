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

HG-side native variance-gamma stochastic injection (M3-3e-2). Iterates
the HG `PolynomialFieldSet` directly — no `cache_mesh` round-trip. The
algorithm mirrors M1's `inject_vg_noise!` line-for-line on the physics:
RNG draws are taken in `1:N` segment order (byte-identical sequence);
per-cell drift / noise / amplitude-limiter / KE-debit / entropy
re-anchor; vertex-velocity update via mass-lumped half-each
distribution; post-injection `realizability_project_HG!` (still on
cache_mesh until M3-3e-4).

Returns `(mesh, diag)` so the caller can inspect the per-cell
diagnostics. The `proj_stats` accumulator (if supplied) tracks
realizability-projection events; identical bit-exact values to the M1
path.

Bit-equality contract: with the same `rng` (same seed) and `params`,
the per-cell `(divu, eta, δ(ρu), δ_drift, δ_noise, ΔKE_vol)` plus the
post-step `(x, u, α, β, s, Pp, Q)` are byte-identical to M1's
`inject_vg_noise!` running on a parallel `Mesh1D`.
"""
function inject_vg_noise_HG!(mesh::DetMeshHG{T}, dt::Real;
                              params::NoiseInjectionParams,
                              rng,
                              diag::InjectionDiagnostics =
                                  InjectionDiagnostics(n_cells(mesh)),
                              proj_stats::Union{ProjectionStats,Nothing} = nothing) where {T<:Real}
    return inject_vg_noise_HG_native!(mesh, dt;
                                       params = params, rng = rng,
                                       diag = diag,
                                       proj_stats = proj_stats)
end

"""
    inject_vg_noise_HG_native!(mesh::DetMeshHG, dt; ...) -> (mesh, diag)

Native HG-side stochastic injection: lifts M1's `inject_vg_noise!`
onto the `PolynomialFieldSet` SoA storage. Internal helper; callers
should use `inject_vg_noise_HG!`.

Per-step recipe (mirrors `inject_vg_noise!` exactly):
  1. Compute cell-centered `(∂_x u)_j` from the post-Newton vertex
     velocities stored in `fs.u` and `fs.x`.
  2. Draw `η_j ~ VG(λ, θ_factor)` for `j = 1:N` in linear order, then
     periodically smooth (3-point Gaussian) into `diag.eta`.
  3. Per cell `j`: drift, noise, amplitude limiter, KE-debit on
     `(P_xx, P_⊥)`, entropy re-anchor; mutate `fs.{s, Pp}[j]`.
  4. Distribute cell momentum half-each to the two adjacent vertices,
     mutating `fs.u[i]` and `mesh.p_half[i]`.
  5. Post-injection `realizability_project_HG!` (delegates through
     cache_mesh until M3-3e-4 lifts it).

The RNG sequencing is identical to M1's path: `rand_variance_gamma!`
draws N samples in the same `1:N` order, then per-cell scalar
arithmetic uses `diag.eta[j]` in the same order. With identical seed,
the byte sequence emitted by the rng matches exactly.
"""
function inject_vg_noise_HG_native!(mesh::DetMeshHG{T}, dt::Real;
                                     params::NoiseInjectionParams,
                                     rng,
                                     diag::InjectionDiagnostics =
                                         InjectionDiagnostics(n_cells(mesh)),
                                     proj_stats::Union{ProjectionStats,Nothing} = nothing) where {T<:Real}
    fs = mesh.fields
    N = n_cells(mesh)
    @assert length(diag.divu) == N "InjectionDiagnostics size mismatch ($(length(diag.divu)) vs $N)"

    Δt = Float64(dt)
    L_box = mesh.L_box
    Δm_vec = mesh.Δm

    # ── (1) Per-cell divu from post-Newton vertex (x, u). ─────────────
    @inbounds for j in 1:N
        j_right = j == N ? 1 : j + 1
        wrap = (j == N) ? L_box : zero(T)
        x_j  = T(fs.x[j][1])
        x_jr = T(fs.x[j_right][1]) + wrap
        u_j  = T(fs.u[j][1])
        u_jr = T(fs.u[j_right][1])
        Δx_j = x_jr - x_j
        Δu_j = u_jr - u_j
        diag.divu[j] = Float64(Δu_j / Δx_j)
    end

    # ── (2) Draw VG noise in 1:N order, then smooth periodically. ─────
    eta_white = Vector{Float64}(undef, N)
    rand_variance_gamma!(rng, eta_white, params.λ, params.θ_factor)
    if params.ell_corr > 0 && N >= 3
        smooth_periodic_3pt!(diag.eta, eta_white)
    else
        copyto!(diag.eta, eta_white)
    end

    # ── (3) Per-cell drift / noise / limiter / KE-debit / entropy. ────
    # Cell-momentum injections accumulated for the vertex update in (4).
    Δp_cell = Vector{Float64}(undef, N)

    @inbounds for j in 1:N
        # Cell-centered velocity = ½(u_j + u_{j+1}).
        j_right = j == N ? 1 : j + 1
        u_j  = T(fs.u[j][1])
        u_jr = T(fs.u[j_right][1])
        u_c  = Float64((u_j + u_jr) / 2)

        # Cell density ρ_j = Δm_j / (x_{j+1} - x_j).
        wrap = (j == N) ? L_box : zero(T)
        x_j  = T(fs.x[j][1])
        x_jr = T(fs.x[j_right][1]) + wrap
        Δx_j = x_jr - x_j
        ρ_j  = Float64(Δm_vec[j] / Δx_j)
        J_j  = 1.0 / ρ_j

        s_pre = Float64(fs.s[j][1])
        Mvv_pre = Float64(Mvv(J_j, s_pre))
        Pxx_old = ρ_j * Mvv_pre
        Pp_old  = Float64(fs.Pp[j][1])
        # Detect Pp sentinel (NaN from legacy 5-arg DetField). When Pp
        # is NaN, treat as isotropic Maxwellian so the debit can still
        # operate on the trace via entropy alone.
        if !isfinite(Pp_old)
            Pp_old = Pxx_old
        end

        divu_j = diag.divu[j]
        diag.compressive[j] = divu_j < 0

        drift_term = params.C_A * ρ_j * divu_j * Δt
        compression = max(-divu_j, 0.0)
        noise_amp = params.C_B * ρ_j * sqrt(compression * Δt)
        δ_drift = drift_term
        δ_noise = noise_amp * diag.eta[j]
        diag.delta_rhou_drift[j] = δ_drift
        diag.delta_rhou_noise[j] = δ_noise
        δ = δ_drift + δ_noise

        # Amplitude limiter: KE injection capped at
        # `ke_budget_fraction · IE_local`. Discriminant form matches
        # py-1d / M1 exactly.
        IE_local = 0.5 * Pxx_old + Pp_old
        KE_budget = params.ke_budget_fraction * IE_local
        abs_u = abs(u_c)
        disc = abs_u * abs_u + 2.0 * KE_budget / max(ρ_j, 1e-30)
        δ_max = (-abs_u + sqrt(max(disc, 0.0))) * ρ_j
        if δ > δ_max
            δ = δ_max
        elseif δ < -δ_max
            δ = -δ_max
        end
        diag.delta_rhou[j] = δ

        # KE-debit and pressure update.
        ΔKE_vol = u_c * δ + 0.5 * δ * δ / ρ_j
        diag.delta_KE_vol[j] = ΔKE_vol
        Pxx_new = Pxx_old - (2.0 / 3.0) * ΔKE_vol
        Pp_new  = Pp_old  - (2.0 / 3.0) * ΔKE_vol
        if Pxx_new < params.pressure_floor
            Pxx_new = params.pressure_floor
        end
        if Pp_new < params.pressure_floor
            Pp_new = params.pressure_floor
        end

        # Re-anchor entropy from P_xx.
        Mvv_new = Pxx_new / ρ_j
        if Mvv_new > 0 && Mvv_pre > 0
            s_new = s_pre + log(Mvv_new / Mvv_pre)
        else
            s_new = s_pre
        end

        # Mutate `(s, Pp)` in the field set; (x, u, α, β, Q) untouched
        # here (the vertex `u` update happens in step 4).
        fs.s[j]  = (T(s_new),)
        fs.Pp[j] = (T(Pp_new),)

        # Cell-momentum injection: ΔP_j = δ · Δx_j. Matches M1's
        # ordering exactly: Δx_j = J_j · Δm_j (= 1/ρ_j · Δm_j), not the
        # geometric `(x_{j+1} - x_j)`. The two are mathematically equal
        # but differ at the last bit due to FP roundoff in the J = 1/ρ
        # reciprocal. Bit-exact parity with M1's `inject_vg_noise!`
        # requires the J·Δm form.
        Δx_for_p = J_j * Float64(Δm_vec[j])
        Δp_cell[j] = δ * Δx_for_p
    end

    # ── (4) Distribute cell momentum to vertices (mass-lumped). ───────
    # Vertex i receives ΔP_{i-1}/2 from its left cell and ΔP_i/2 from
    # its right cell. Updated u_i = u_i^old + (ΔP_{i-1} + ΔP_i)/(2·m̄_i),
    # with `m̄_i = (Δm_{i-1} + Δm_i)/2` (cyclic in i).
    @inbounds for i in 1:N
        i_left = i == 1 ? N : i - 1
        m̄ = Float64((Δm_vec[i_left] + Δm_vec[i]) / 2)
        δu = (Δp_cell[i_left] + Δp_cell[i]) / (2.0 * m̄)
        u_old = Float64(fs.u[i][1])
        u_new = u_old + δu
        fs.u[i] = (T(u_new),)
        mesh.p_half[i] = T(m̄ * u_new)
    end

    # ── (5) Realizability projection (M2-3 / Phase 8.5). ──────────────
    # Still on cache_mesh until M3-3e-4 lifts it. Bit-exact equal to
    # M1's `realizability_project!` on a parallel `Mesh1D`.
    #
    # Fast-path: with `project_kind = :none` the projection itself is a
    # no-op (M1's `realizability_project!` short-circuits) and the only
    # observable mutation is `proj_stats.n_steps += 1`. Skip the
    # cache_mesh round-trip so the M3-3e-2 wall-time win is realized on
    # the most-common configuration. M3-3e-4 will retire this branch
    # by lifting the projection itself onto the field set.
    if params.project_kind === :none
        if proj_stats !== nothing
            proj_stats.n_steps += 1
        end
    else
        realizability_project_HG!(mesh;
                                  kind = params.project_kind,
                                  headroom = params.realizability_headroom,
                                  Mvv_floor = params.Mvv_floor,
                                  pressure_floor = params.pressure_floor,
                                  stats = proj_stats)
    end

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

HG-side native operator-split driver (M3-3e-2). Composes:

  1. Pre-Newton `realizability_project_HG!` (still on cache_mesh until
     M3-3e-4).
  2. `det_step_HG!` — the M3-3e-1 native deterministic Newton step.
  3. `inject_vg_noise_HG!` — the M3-3e-2 native VG injection (which
     itself calls a post-injection projection).
  4. Optional `BurstStatsAccumulator` recording.

Same bit-exact contract as `det_run_stochastic!`: identical RNG seed,
params, and `q_kind` produce byte-identical per-cell state to running
M1's `det_run_stochastic!` on a parallel `Mesh1D`.
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
    return det_run_stochastic_HG_native!(mesh, dt, n_steps;
                                          params = params, rng = rng,
                                          tau = tau, q_kind = q_kind,
                                          c_q_quad = c_q_quad,
                                          c_q_lin = c_q_lin,
                                          accumulator = accumulator,
                                          monitor_every = monitor_every,
                                          proj_stats = proj_stats,
                                          kwargs...)
end

"""
    det_run_stochastic_HG_native!(mesh::DetMeshHG, dt, n_steps; ...) -> mesh

Native HG-side composition driver. Internal helper; callers should use
`det_run_stochastic_HG!`. Mirrors M1's `det_run_stochastic!` per-step
loop ordering exactly:

```
for n in 1:n_steps
    realizability_project_HG!(mesh; ...)   # pre-Newton floor
    det_step_HG!(mesh, dt; ...)            # M3-3e-1 native Newton
    inject_vg_noise_HG!(mesh, dt; ...)     # native injection + post-projection
    record_step!(accumulator, diag, dt, params)  # optional
    self_consistency_check(accumulator)          # optional, every monitor_every
end
```

The pre-Newton projection is the M2-3 long-time stability fix (closes
M1 Open #4): it intercepts cells driven near the realizability boundary
by accumulated noise drift before the next Newton step sees them. With
`params.project_kind == :none` both projections are no-ops and the
driver reduces to M1 Phase 8 bit-for-bit.
"""
function det_run_stochastic_HG_native!(mesh::DetMeshHG{T}, dt::Real,
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
    N = n_cells(mesh)
    diag = InjectionDiagnostics(N)
    for n in 1:n_steps
        # Pre-Newton projection (M2-3 long-time stability fix).
        # Fast-path `:none` to skip the cache_mesh round-trip (the
        # projection itself is a no-op; only `n_steps` is incremented).
        if params.project_kind === :none
            if proj_stats !== nothing
                proj_stats.n_steps += 1
            end
        else
            realizability_project_HG!(mesh;
                                      kind = params.project_kind,
                                      headroom = params.realizability_headroom,
                                      Mvv_floor = params.Mvv_floor,
                                      pressure_floor = params.pressure_floor,
                                      stats = proj_stats)
        end
        # Deterministic Newton step (M3-3e-1 native).
        det_step_HG!(mesh, dt;
                     tau = tau, q_kind = q_kind,
                     c_q_quad = c_q_quad, c_q_lin = c_q_lin,
                     kwargs...)
        # Variance-gamma stochastic injection (M3-3e-2 native, includes
        # post-injection projection).
        inject_vg_noise_HG!(mesh, dt; params = params, rng = rng,
                             diag = diag, proj_stats = proj_stats)
        if accumulator !== nothing
            record_step!(accumulator, diag, dt, params)
            if monitor_every > 0 && (n % monitor_every == 0)
                # Side-effect-free monitor; result discarded (caller can
                # re-run `self_consistency_check` at end-of-run).
                _ = self_consistency_check(accumulator)
            end
        end
    end
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
