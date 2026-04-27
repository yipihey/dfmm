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
    _TracerStorageHG{T<:Real}

Internal mutable struct holding the per-cell tracer matrix and the
symbolic-label vector for `TracerMeshHG`. M3-3e-3 introduced this
storage type to retire the cache_mesh-shared M1 `TracerMesh` that
`TracerMeshHG` previously wrapped.

The shape of this struct preserves the test-side accessors
`tm.tm.tracers` (Matrix{T}) and `tm.tm.names` (Vector{Symbol}) used
by `test_M3_2_phase11_tracer_HG.jl` and
`test_M3_2_M2_2_multitracer_HG.jl`. Those tests compare HG and M1
matrices for bit-exact parity (`tm_M1.tracers == tm_HG.tm.tracers`),
which still works because both are plain `Matrix{Float64}` storage.
"""
mutable struct _TracerStorageHG{T<:Real}
    tracers::Matrix{T}
    names::Vector{Symbol}
end

"""
    TracerMeshHG{T<:Real}

A passive-scalar field bundle attached to a `DetMeshHG{T}` fluid
mesh. The HG-side counterpart to M1's `TracerMesh{T, M<:Mesh1D}`.

**M3-3e-3 (2026-04-26):** the storage is now native HG-side — a
freestanding `_TracerStorageHG{T}` whose `tracers::Matrix{T}` and
`names::Vector{Symbol}` are not aliased to the cache mesh's tracer
storage. Pre-M3-3e-3, `TracerMeshHG` wrapped an M1 `TracerMesh` whose
`fluid` field pointed at the HG wrapper's cached `Mesh1D` — that
path is gone. Mass-coordinate accessors (`set_tracer!(::Function)`)
now derive `m_j` from the cumulative `mesh.Δm` directly; Eulerian
position lookups (`tracer_at_position`) walk `mesh.fields.x[j]`.

Construct via `TracerMeshHG(fluid_HG; n_tracers, names)`. The
`tm.tracers` matrix is the live tracer state and is not mutated by
`advect_tracers_HG!` (which is a no-op in the pure-Lagrangian
frame); the user can mutate it directly via `set_tracer!` /
`add_tracer!`. AMR refine/coarsen mutations of the matrix go through
`_refine_tracer_HG!` / `_coarsen_tracer_HG!` (see
`src/action_amr_helpers.jl`), which mirror M1's
`refine_tracer!` / `coarsen_tracer!` byte-equally.
"""
mutable struct TracerMeshHG{T<:Real}
    fluid::DetMeshHG{T}
    tm::_TracerStorageHG{T}
end

"""
    TracerMeshHG(fluid::DetMeshHG{T}; n_tracers = 1, names = nothing)
        -> TracerMeshHG

Allocate a `TracerMeshHG` for `n_tracers` rows on `fluid`. Mirrors
`TracerMesh(fluid::Mesh1D; n_tracers, names)` for the matrix shape
and default names.
"""
function TracerMeshHG(fluid::DetMeshHG{T};
                       n_tracers::Integer = 1,
                       names::Union{Nothing,AbstractVector{Symbol}} = nothing) where {T<:Real}
    @assert n_tracers ≥ 1 "TracerMeshHG requires at least one tracer field"
    N = length(fluid.Δm)
    values = zeros(T, n_tracers, N)
    nms = if names === nothing
        [Symbol("tracer", k) for k in 1:n_tracers]
    else
        @assert length(names) == n_tracers "names length must match n_tracers"
        collect(Symbol, names)
    end
    storage = _TracerStorageHG{T}(values, nms)
    return TracerMeshHG{T}(fluid, storage)
end

"""
    advect_tracers_HG!(tm::TracerMeshHG, dt) -> TracerMeshHG

No-op step (pure-Lagrangian frame): the tracer matrix is never
mutated. Provided for symmetry with `det_step_HG!` driver loops.
M3-3e-3: the implementation is now a native no-op (no delegation to
M1's `advect_tracers!`); the per-cell matrix is untouched.
"""
@inline advect_tracers_HG!(tm::TracerMeshHG, dt::Real) = tm

# Forward the per-tracer accessors so callers can use the same names
# they used for `TracerMesh`. Each one operates on the native HG-side
# storage in `tm.tm`.

"""
    n_tracer_fields(tm::TracerMeshHG) -> Int
"""
n_tracer_fields(tm::TracerMeshHG) = size(tm.tm.tracers, 1)

"""
    n_tracer_segments(tm::TracerMeshHG) -> Int
"""
n_tracer_segments(tm::TracerMeshHG) = size(tm.tm.tracers, 2)

"""
    tracer_index(tm::TracerMeshHG, name::Symbol) -> Int

Look up a tracer's row index by its symbolic name. Raises an
`ArgumentError` if the name is not present.
"""
function tracer_index(tm::TracerMeshHG, name::Symbol)
    k = findfirst(==(name), tm.tm.names)
    k === nothing && throw(ArgumentError(
        "tracer name $name not found; have $(tm.tm.names)"))
    return k
end

"""
    set_tracer!(tm::TracerMeshHG, name_or_index, values_or_f) -> TracerMeshHG

Initialise a tracer field. Mirrors M1's
`set_tracer!(::TracerMesh, …)`. The vector form sets row `k` to
`values[1:N]`. The functional form evaluates `f(m_j)` at each
segment's Lagrangian mass-coordinate `m_j` (segment center, derived
from the cumulative `fluid.Δm` directly — no cache_mesh access).

Both raise if `name_or_index` is invalid or if the vector length
does not match `n_tracer_segments(tm)`.
"""
function set_tracer!(tm::TracerMeshHG, k::Integer, values::AbstractVector)
    N = n_tracer_segments(tm)
    @assert length(values) == N "values length $(length(values)) ≠ N=$N"
    @assert 1 ≤ k ≤ n_tracer_fields(tm) "tracer index $k out of range"
    @inbounds for j in 1:N
        tm.tm.tracers[k, j] = values[j]
    end
    return tm
end
set_tracer!(tm::TracerMeshHG, name::Symbol, values::AbstractVector) =
    set_tracer!(tm, tracer_index(tm, name), values)

function set_tracer!(tm::TracerMeshHG{T}, k::Integer, f::Function) where {T<:Real}
    N = n_tracer_segments(tm)
    @assert 1 ≤ k ≤ n_tracer_fields(tm) "tracer index $k out of range"
    Δm = tm.fluid.Δm
    cum = T(0)
    @inbounds for j in 1:N
        # Mass-coordinate cell center: cum_left + Δm_j/2.
        m_j = cum + Δm[j] / 2
        tm.tm.tracers[k, j] = f(m_j)
        cum += Δm[j]
    end
    return tm
end
set_tracer!(tm::TracerMeshHG, name::Symbol, f::Function) =
    set_tracer!(tm, tracer_index(tm, name), f)

"""
    add_tracer!(tm::TracerMeshHG, name::Symbol, values_or_f) -> TracerMeshHG

Allocate a new `TracerMeshHG` with one additional row whose values
come from `values_or_f`. Mirrors M1's
`add_tracer!(::TracerMesh, …)`. Existing rows are copied; the fluid
mesh reference is shared.
"""
function add_tracer!(tm::TracerMeshHG{T}, name::Symbol,
                      values::AbstractVector) where {T<:Real}
    @assert !(name in tm.tm.names) "tracer name $name already exists"
    N = n_tracer_segments(tm)
    @assert length(values) == N
    K = n_tracer_fields(tm) + 1
    new_mat = zeros(T, K, N)
    new_mat[1:end-1, :] = tm.tm.tracers
    @inbounds for j in 1:N
        new_mat[K, j] = T(values[j])
    end
    new_names = vcat(tm.tm.names, name)
    return TracerMeshHG{T}(tm.fluid, _TracerStorageHG{T}(new_mat, new_names))
end
function add_tracer!(tm::TracerMeshHG{T}, name::Symbol,
                      f::Function) where {T<:Real}
    N = n_tracer_segments(tm)
    Δm = tm.fluid.Δm
    cum = T(0)
    vals = Vector{T}(undef, N)
    @inbounds for j in 1:N
        m_j = cum + Δm[j] / 2
        vals[j] = T(f(m_j))
        cum += Δm[j]
    end
    return add_tracer!(tm, name, vals)
end

"""
    tracer_at_position(tm::TracerMeshHG, x::Real, k::Integer = 1) -> Real

Sample tracer `k` at Eulerian (lab-frame) position `x`. Walks the HG
field set's `x` coefficients cyclically until the segment containing
`x` is found and returns the segment-centred tracer value. Mirrors
M1's `tracer_at_position(::TracerMesh, …)` semantics.
"""
function tracer_at_position(tm::TracerMeshHG{T}, x::Real,
                             k::Integer = 1) where {T<:Real}
    @assert 1 ≤ k ≤ n_tracer_fields(tm) "tracer index $k out of range"
    fluid = tm.fluid
    fs = fluid.fields
    N = n_tracer_segments(tm)
    L = fluid.L_box
    x_left_1 = T(fs.x[1][1])
    xq = mod(x - x_left_1, L) + x_left_1
    @inbounds for j in 1:N
        x_l = T(fs.x[j][1])
        j_r = j == N ? 1 : j + 1
        wrap = (j == N) ? L : zero(T)
        x_r = T(fs.x[j_r][1]) + wrap
        if x_l ≤ xq < x_r
            return tm.tm.tracers[k, j]
        end
    end
    return tm.tm.tracers[k, N]
end

# ─────────────────────────────────────────────────────────────────────
# M2-3 — realizability projection (M3-2 Block 4b).
# ─────────────────────────────────────────────────────────────────────

"""
    realizability_project_HG!(mesh::DetMeshHG; kind = :reanchor,
                              headroom = 1.05, Mvv_floor = 1e-2,
                              pressure_floor = 1e-8, stats = nothing)
        -> mesh

HG-side native realizability projection (M3-3e-4). Iterates the HG
`PolynomialFieldSet` directly — no `cache_mesh` round-trip. Mirrors
M1's `realizability_project!` line-for-line on the physics: per-cell
projection that pushes the state back inside the realizability cone
`M_vv ≥ headroom · β²` (with `Mvv_floor` absolute safety floor),
debiting the extra internal energy from `P_⊥` first and admitting any
residual as a silent floor-gain.

The projection is per-cell with no inter-cell coupling, so the lift
is mechanical: read `(α, β, x, u, s, Pp, Q)` from `fields[j][1]`,
apply the unchanged per-cell math, write `(s, Pp)` back.

Bit-equality contract: with the same `(kind, headroom, Mvv_floor,
pressure_floor)` and IC, the per-cell `(s, Pp)` post-projection plus
the `ProjectionStats` accumulator are byte-identical to M1's
`realizability_project!` running on a parallel `Mesh1D`.
"""
function realizability_project_HG!(mesh::DetMeshHG{T};
                                    kind::Symbol = :reanchor,
                                    headroom::Real = 1.05,
                                    Mvv_floor::Real = 1e-2,
                                    pressure_floor::Real = 1e-8,
                                    stats::Union{ProjectionStats,Nothing} = nothing) where {T<:Real}
    if kind === :none
        if stats !== nothing
            stats.n_steps += 1
        end
        return mesh
    end
    @assert kind === :reanchor "realizability_project_HG!: unsupported kind=$(kind)"

    fs = mesh.fields
    N = n_cells(mesh)
    h = Float64(headroom)
    Mvv_floor_f = Float64(Mvv_floor)
    pf = Float64(pressure_floor)
    L_box = mesh.L_box
    Δm_vec = mesh.Δm

    if stats !== nothing
        stats.n_steps += 1
    end

    @inbounds for j in 1:N
        # Cell density ρ_j = Δm_j / (x_{j+1} - x_j) with periodic wrap.
        j_right = j == N ? 1 : j + 1
        wrap = (j == N) ? L_box : zero(T)
        x_j  = T(fs.x[j][1])
        x_jr = T(fs.x[j_right][1]) + wrap
        Δx_j = x_jr - x_j
        ρ_j  = Float64(Δm_vec[j] / Δx_j)
        J_j  = 1.0 / ρ_j

        s_pre  = Float64(fs.s[j][1])
        β_pre  = Float64(fs.beta[j][1])
        Mvv_pre = Float64(Mvv(J_j, s_pre))
        Pp_pre = Float64(fs.Pp[j][1])
        if !isfinite(Pp_pre)
            Pp_pre = ρ_j * Mvv_pre  # legacy NaN sentinel — treat isotropic
        end

        if stats !== nothing
            stats.Mvv_min_pre = min(stats.Mvv_min_pre, Mvv_pre)
        end

        # Realizability target. We require M_vv ≥ headroom · β² so the
        # next Newton step's γ² = M_vv − β² has a strict positive
        # margin; we *also* require M_vv ≥ Mvv_floor.
        Mvv_target_rel = h * β_pre * β_pre
        Mvv_target = max(Mvv_target_rel, Mvv_floor_f)

        if Mvv_pre >= Mvv_target
            if stats !== nothing
                stats.Mvv_min_post = min(stats.Mvv_min_post, Mvv_pre)
            end
            continue
        end

        # Projection event: raise s so M_vv = Mvv_target, take the
        # extra energy from P_⊥ first (stays ≥ pressure_floor), then
        # admit any residual as a silent floor-gain.
        if stats !== nothing
            stats.n_events += 1
            if Mvv_target > Mvv_target_rel
                stats.n_floor_events += 1
            end
        end

        Pxx_pre = ρ_j * Mvv_pre
        Pxx_new = ρ_j * Mvv_target
        ΔPxx = Pxx_new - Pxx_pre   # > 0 by construction

        # IE_per_volume = ½ Pxx + Pp. Debit ½ ΔPxx from P_⊥ (the
        # 1D-3D effective-dimensions split closing the paired-pressure
        # ledger; methods paper §9.6 / M2-3 notes §3).
        Pp_take_target = 0.5 * ΔPxx
        Pp_avail = max(Pp_pre - pf, 0.0)
        Pp_take = min(Pp_take_target, Pp_avail)
        Pp_new = Pp_pre - Pp_take

        # IE budget: with Pp_take = ½·ΔPxx the cell's IE is conserved.
        # Residual `0.5·ΔPxx - Pp_take ≥ 0` is admitted as a silent
        # floor-gain (mirrors py-1d's pressure_floor semantics). Use the
        # `J·Δm` ordering (matches M1's `J_j * seg.Δm` byte-for-byte —
        # the geometric `(x_{j+1} - x_j)` form differs at the last bit
        # due to the J = 1/ρ reciprocal).
        floor_gain = (0.5 * ΔPxx) - Pp_take
        if stats !== nothing
            Δx_for_E = J_j * Float64(Δm_vec[j])
            stats.total_dE_inj += floor_gain * Δx_for_E
        end

        # Re-anchor entropy from the new M_vv.
        s_new = s_pre + log(Mvv_target / max(Mvv_pre, 1e-300))

        # Mutate `(s, Pp)` only; (x, u, α, β, Q) untouched.
        fs.s[j]  = (T(s_new),)
        fs.Pp[j] = (T(Pp_new),)

        if stats !== nothing
            stats.Mvv_min_post = min(stats.Mvv_min_post, Mvv_target)
        end
    end
    return mesh
end
