# newton_step_HG_M3_2.jl — HG-side native implementations for Phase 7 / 8
# / 11 and M2-1 / M2-3 sub-phases (Phase M3-2).
#
# **M3-3e final state.** As of M3-3e-1/2/3/4/5 these implementations
# are native HG-side: stochastic VG injection, deterministic
# stochastic driver, passive tracers (`TracerMeshHG`), and
# realizability projection all operate directly on the
# `DetMeshHG.fields::PolynomialFieldSet` SoA storage with no parallel
# `cache_mesh::Mesh1D` snapshot.
#
# This file is included by `src/dfmm.jl` after the M1/M2 files
# (`tracers.jl`, `stochastic_injection.jl`, `amr_1d.jl`) that define
# the M1-side primitives whose physics formulae the native lifts
# reproduce byte-equally.
#
# See `reference/notes_M3_2_phase7811_m2_port.md` for the original
# design write-up and `reference/notes_M3_3e_5_cache_mesh_dropped.md`
# for the M3-3e retirement summary.

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
distribution; post-injection `realizability_project_HG!` (native
HG-side, M3-3e-4).

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
  5. Post-injection `realizability_project_HG!` (native HG-side as of
     M3-3e-4).

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
    # Native HG-side (M3-3e-4). Bit-exact equal to M1's
    # `realizability_project!` on a parallel `Mesh1D`.
    #
    # Fast-path: with `project_kind = :none` the projection itself is a
    # no-op (M1's `realizability_project!` short-circuits) and the only
    # observable mutation is `proj_stats.n_steps += 1`.
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

HG-side native operator-split driver. Composes:

  1. Pre-Newton `realizability_project_HG!` (native HG-side, M3-3e-4).
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
        # Fast-path `:none` skips the per-cell loop (the projection
        # itself is a no-op; only `n_steps` is incremented).
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

# ─────────────────────────────────────────────────────────────────────
# M3-6 Phase 3 (a): 2D passive tracer mesh.
# ─────────────────────────────────────────────────────────────────────
#
# Extends the 1D `TracerMeshHG` substrate to 2D `HierarchicalMesh{2}` +
# 2D `PolynomialFieldSet` (the 14-named-field set allocated by
# `allocate_cholesky_2d_fields`). The storage is per-species per-cell
# (NOT per-leaf) so that HG's `register_refinement_listener!` can
# resize the matrix uniformly with the field set.
#
# Mirrors M3-3e-3's 1D pattern (per-species concentration in a Matrix
# with rows = species, columns = cells), generalized to 2D Lagrangian
# topology. In pure-Lagrangian regions the matrix is never written,
# yielding bit-exact preservation by construction (Phase 11 + M2-2
# invariants on the 2D path).
#
# # Mesh-cell vs leaf-cell indexing
#
# The 2D field set in `setups_2d.jl::allocate_cholesky_2d_fields` is
# sized to `n_cells(mesh)` — including non-leaf parents — so that
# `halo_view`-based neighbor access stays 1-to-1 against
# `mesh.cells[i]`. The tracer matrix follows the same convention:
# `tracers[k, ci]` is the per-cell concentration of species `k` at
# mesh-cell index `ci`. Non-leaf entries are unused (post-refinement
# coarsening writes them via the listener; before any AMR they are
# at the IC default).
#
# # Refine/coarsen listener
#
# `register_tracers_on_refine_2d!` registers an HG refinement listener
# that mirrors `register_field_set_on_refine!`'s shape:
#  • Refine: parent's per-species concentration is copied piecewise-
#    constant into all 2^D children (concentration = mass/volume is
#    invariant under equal-volume splitting, so this is exactly
#    conservative for the per-cell mass `c·V`).
#  • Coarsen: children's per-species concentrations are averaged into
#    the parent (volume-weighted average; equal-volume children make
#    this a plain arithmetic mean, also exactly conservative for
#    per-cell mass).
# The listener handle is returned so callers can unregister at the
# field set's lifecycle end. Captures `tm` by reference (the matrix
# is mutated in place via `resize!`).

"""
    TracerMeshHG2D{T<:Real}

Per-species, per-cell passive-scalar field bundle attached to a 2D
HierarchicalGrids mesh + 2D `PolynomialFieldSet` (the 14-named-field
set allocated by `allocate_cholesky_2d_fields`). The 2D-counterpart
of `TracerMeshHG` (1D); decouples M3-6 Phase 3's 2D dust-trap (D.7)
and ISM-tracer (D.10) infrastructure from the 1D path's bit-equality
contract.

Storage:
- `mesh::HierarchicalMesh{2}` — the fluid mesh (shared, not owned).
- `fields::Any` — the 14-named-field 2D field set (shared, not owned).
- `tracers::Matrix{T}` — `(N_species, n_cells(mesh))` per-cell
  concentrations. Indexed by mesh-cell index `ci` (NOT leaf-major)
  so it tracks the field set's storage contract 1-to-1.
- `names::Vector{Symbol}` — species labels.

Construct via `TracerMeshHG2D(fields, mesh; n_species, names)`.

# Mass conservation / preservation contracts

  • **Pure-Lagrangian preservation (Phase 11 invariant lifted to 2D):**
    in the absence of refine/coarsen events, the matrix is never
    mutated by the dfmm path. Bit-exact preservation holds across
    arbitrary numbers of `det_step_2d_berry_HG!` calls.

  • **Refine/coarsen mass conservation (M2-2 invariant lifted to 2D):**
    via the refinement listener registered by
    `register_tracers_on_refine_2d!`. Each per-species per-cell
    concentration `c` represents a mass `c · V_cell`; under isotropic
    refinement the parent volume splits equally into `2^D = 4`
    children with `V_child = V_parent / 4`. Setting
    `c_child = c_parent` per child preserves total mass exactly
    (`Σ_children c_child · V_child = c_parent · V_parent`). Coarsen is
    the inverse: `c_parent = (Σ_children c_child · V_child) / V_parent
    = mean(c_children)`.

  • **Multi-species independence:** the listener walks each species
    row independently — refine/coarsen of species `k` does not touch
    species `k′ ≠ k` (no cross-contamination).
"""
mutable struct TracerMeshHG2D{T<:Real}
    mesh::Any
    fields::Any
    tracers::Matrix{T}
    names::Vector{Symbol}
end

"""
    TracerMeshHG2D(fields, mesh::HierarchicalMesh{2};
                    n_species = 1, names = nothing, T = Float64)
        -> TracerMeshHG2D{T}

Allocate a `TracerMeshHG2D` for `n_species` rows on `(mesh, fields)`.
The matrix is sized to `(n_species, n_cells(mesh))` — including non-
leaf cells — so it stays 1-to-1 with the field-set storage contract.
At allocation time, all entries are zero.
"""
function TracerMeshHG2D(fields, mesh;
                          n_species::Integer = 1,
                          names::Union{Nothing,AbstractVector{Symbol}} = nothing,
                          T::Type = Float64)
    @assert n_species ≥ 1 "TracerMeshHG2D requires at least one species"
    nc = HierarchicalGrids.n_cells(mesh)
    values = zeros(T, n_species, nc)
    nms = if names === nothing
        [Symbol("species", k) for k in 1:n_species]
    else
        @assert length(names) == n_species "names length must match n_species"
        collect(Symbol, names)
    end
    return TracerMeshHG2D{T}(mesh, fields, values, nms)
end

"""
    n_species(tm::TracerMeshHG2D) -> Int
"""
n_species(tm::TracerMeshHG2D) = size(tm.tracers, 1)

"""
    n_cells_2d(tm::TracerMeshHG2D) -> Int

Number of mesh cells in the underlying `HierarchicalMesh{2}` (matches
`HierarchicalGrids.n_cells(tm.mesh)`).
"""
n_cells_2d(tm::TracerMeshHG2D) = size(tm.tracers, 2)

"""
    species_index(tm::TracerMeshHG2D, name::Symbol) -> Int

Look up a species' row index by its symbolic name. Raises an
`ArgumentError` if the name is not present.
"""
function species_index(tm::TracerMeshHG2D, name::Symbol)
    k = findfirst(==(name), tm.names)
    k === nothing && throw(ArgumentError(
        "species name $name not found; have $(tm.names)"))
    return k
end

"""
    set_species!(tm::TracerMeshHG2D, name_or_index, values_or_f, leaves) -> tm

Initialise species `name_or_index` from per-leaf `values_or_f`.

  • Vector form `values::AbstractVector` of length `length(leaves)`:
    sets `tm.tracers[k, ci] = values[i]` for each `(i, ci)` in
    `enumerate(leaves)`.
  • Functional form `f::Function`: sets `tm.tracers[k, ci] = f(ci)`
    for each `ci` in `leaves`. Callers can compute cell centers via
    `cell_physical_box(frame, ci)` from inside `f`.

Non-leaf cells are left untouched (the field-set convention; refine
/ coarsen events propagate values into them via the listener).
"""
function set_species!(tm::TracerMeshHG2D, k::Integer,
                       values::AbstractVector,
                       leaves::AbstractVector{<:Integer})
    @assert 1 ≤ k ≤ n_species(tm) "species index $k out of range"
    @assert length(values) == length(leaves) "values length $(length(values)) ≠ length(leaves) $(length(leaves))"
    @inbounds for (i, ci) in enumerate(leaves)
        tm.tracers[k, ci] = values[i]
    end
    return tm
end
set_species!(tm::TracerMeshHG2D, name::Symbol, values::AbstractVector,
              leaves::AbstractVector{<:Integer}) =
    set_species!(tm, species_index(tm, name), values, leaves)

function set_species!(tm::TracerMeshHG2D, k::Integer, f::Function,
                       leaves::AbstractVector{<:Integer})
    @assert 1 ≤ k ≤ n_species(tm) "species index $k out of range"
    @inbounds for ci in leaves
        tm.tracers[k, ci] = f(ci)
    end
    return tm
end
set_species!(tm::TracerMeshHG2D, name::Symbol, f::Function,
              leaves::AbstractVector{<:Integer}) =
    set_species!(tm, species_index(tm, name), f, leaves)

"""
    advect_tracers_HG_2d!(tm::TracerMeshHG2D, dt) -> tm

No-op step (pure-Lagrangian frame, 2D path): the tracer matrix is
never mutated. Provided for symmetry with `det_step_2d_berry_HG!`
driver loops. Bit-exact preservation: any number of consecutive
calls leaves the matrix byte-identical.
"""
@inline advect_tracers_HG_2d!(tm::TracerMeshHG2D, dt::Real) = tm

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

# ─────────────────────────────────────────────────────────────────────
# M3-6 Phase 3 (b): 2D variance-gamma stochastic injection.
# ─────────────────────────────────────────────────────────────────────
#
# Per-axis variance-gamma stochastic injection on the 2D
# `PolynomialFieldSet` substrate. Axis-selective by design: callers
# pass `axes::NTuple{N,Int}` (a subset of `(1, 2)`) and only those
# axes receive injection. The RNG draws, divu evaluation, and
# per-axis state updates are independent across axes — verified by
# the M3-6 Phase 3 selectivity gate.
#
# # Per-axis recipe (per axis a in axes)
#
#  1. Compute axis-a divu per leaf from face-neighbor `u_a`
#     differences along axis `a` (using the precomputed
#     `face_lo_idx[a], face_hi_idx[a]` tables from
#     `build_face_neighbor_tables(mesh, leaves, bc_spec)`). For
#     boundary cells where `face_*_idx == 0`, divu_a is set to 0
#     (no injection contribution).
#  2. Draw VG noise `η_a[i]` for `i = 1:N` in linear leaf order, then
#     periodically smooth (3-point along axis a) into the per-axis
#     η buffer.
#  3. Per leaf: drift `C_A · ρ · divu_a · Δt`, noise `C_B · ρ ·
#     √(max(-divu_a, 0) · Δt) · η_a`, amplitude limiter, KE-debit
#     into `(s, Pp)` (entropy re-anchored from the new `M_vv`).
#  4. Vertex velocity update on axis `a` only: `u_a += δ_a / ρ`
#     (per-leaf — the 2D path stores cell-centered `u_a`, not
#     vertex velocities, so the update is applied directly).
#
# Other axes (not in `axes`) are NOT touched on this call — neither
# divu evaluation nor RNG draws nor field writes. This is the per-
# axis selectivity contract, the load-bearing structural property
# the M3-6 Phase 2 D.4 Zel'dovich pancake handoff item requires.
#
# # 4-component cone (M3-6 Phase 1b)
#
# After per-axis injection, the optional 2D realizability projection
# (`realizability_project_2d!`) enforces both the per-axis cone
# `M_vv,aa ≥ headroom · β_a²` and the 4-component cone
# `Q = β_1² + β_2² + 2(β_12² + β_21²) ≤ M_vv · headroom_offdiag`.
# When `params.project_kind == :none` the projection is skipped.

"""
    InjectionDiagnostics2D

Per-axis, per-cell stochastic injection diagnostics on a 2D
`PolynomialFieldSet`. The 2D analog of `InjectionDiagnostics`. Each
axis's vectors carry only the cells that received injection on this
call (set by the `axes` keyword to `inject_vg_noise_HG_2d!`).

Fields:
  • `divu::NTuple{2, Vector{Float64}}` — per-axis cell-centered divu_a.
  • `eta::NTuple{2, Vector{Float64}}`  — per-axis smoothed VG draws.
  • `delta_rhou::NTuple{2, Vector{Float64}}` — per-axis δ(ρu_a) after
                                                amplitude-limiting.
  • `delta_KE_vol::NTuple{2, Vector{Float64}}` — per-axis ΔKE injected.
  • `compressive::NTuple{2, BitVector}` — per-axis compressive mask.

For axes not in the `axes` set, the corresponding vectors stay at
their initial zero values (set by the `InjectionDiagnostics2D(N)`
constructor).
"""
struct InjectionDiagnostics2D
    divu::NTuple{2, Vector{Float64}}
    eta::NTuple{2, Vector{Float64}}
    delta_rhou::NTuple{2, Vector{Float64}}
    delta_KE_vol::NTuple{2, Vector{Float64}}
    compressive::NTuple{2, BitVector}
end

InjectionDiagnostics2D(N::Int) = InjectionDiagnostics2D(
    (zeros(Float64, N), zeros(Float64, N)),
    (zeros(Float64, N), zeros(Float64, N)),
    (zeros(Float64, N), zeros(Float64, N)),
    (zeros(Float64, N), zeros(Float64, N)),
    (falses(N), falses(N)),
)

"""
    inject_vg_noise_HG_2d!(fields, mesh::HierarchicalMesh{2},
                            frame::EulerianFrame{2, T},
                            leaves, bc_spec, dt;
                            params::NoiseInjectionParams,
                            rng,
                            axes::NTuple{N,Int} = (1, 2),
                            ρ_ref::Real = 1.0,
                            Gamma::Real = GAMMA_LAW_DEFAULT,
                            diag::InjectionDiagnostics2D =
                                InjectionDiagnostics2D(length(leaves)),
                            proj_stats::Union{ProjectionStats,Nothing} = nothing)
        -> (fields, diag)

Per-axis variance-gamma stochastic injection on the 2D Cholesky-
sector field set. Walks `leaves` in leaf-major order and applies
per-axis drift + VG noise to `(β_a, u_a, s, Pp)` for each axis
`a ∈ axes`. Axes NOT in `axes` are byte-untouched: neither divu
evaluation nor RNG draws nor field writes for those axes fire.

Returns `(fields, diag)`. The `diag::InjectionDiagnostics2D`
accumulator records per-axis per-cell statistics for downstream
self-consistency monitoring.

# Per-axis selectivity contract (load-bearing)

  • `axes = (1,)`: axis-1 fields `(β_1, u_1)` are mutated; axis-2
    fields `(β_2, u_2)` are byte-equal pre/post. RNG draws happen
    only for axis 1 (one buffer of length N). The s/Pp scalars
    can be touched (per-axis KE-debit), but the trivial-axis test
    case (axis-2-aligned IC with `divu_2 = 0` everywhere) leaves
    them invariant by zero noise amplitude.
  • `axes = (2,)`: symmetric (axis-1 fields untouched).
  • `axes = (1, 2)`: both axes injected; RNG advances first by one
    length-N buffer for axis 1, then by another for axis 2, in that
    order.

# 4-component realizability cone

After per-axis injection, the optional projection
`realizability_project_2d!(fields, leaves; project_kind =
params.project_kind, ...)` enforces both the per-axis 2-component
cone (`M_vv,aa ≥ headroom · β_a²`) and the 4-component cone
(`Q = β_1² + β_2² + 2(β_12² + β_21²) ≤ M_vv · headroom_offdiag`).
With `params.project_kind = :none` the projection is a no-op
(only `proj_stats.n_steps` is incremented).

# 1D parity gate (axis-aligned IC ⊂ 2D)

For an axis-1-aligned 2D IC (`u_2 = 0`, `β_2 = 0`, `M_vv,11 =
M_vv,22 = Mvv(J, s)` from the isotropic EOS), and `axes = (1,)`,
the axis-1 sub-recipe is byte-equal to the 1D `inject_vg_noise_HG!`
modulo the per-cell vs per-vertex `u` storage convention. This is
the M3-6 Phase 3 RNG-bit-equal cross-check (verified at the test
level).

# Math source

Per-axis injection on `(β_a, u_a)` follows the same boxed-Hamilton
ledger M1's 1D path uses (see `reference/notes_phase8_stochastic_injection.md`
§3.2): `δ(ρu_a) = drift_a + noise_a`, KE-debit
`ΔKE_vol = u_a · δ + ½ δ²/ρ` taken half-each from `(P_xx, P_⊥) =
(ρ M_vv, Pp)`, entropy re-anchored from the new `M_vv`. The 2D
extension simply runs this independently per axis with axis-a
divu and axis-a η.
"""
function inject_vg_noise_HG_2d!(fields, mesh,
                                  frame,
                                  leaves::AbstractVector{<:Integer},
                                  bc_spec, dt::Real;
                                  params::NoiseInjectionParams,
                                  rng,
                                  axes::Tuple = (1, 2),
                                  ρ_ref::Real = 1.0,
                                  Gamma::Real = GAMMA_LAW_DEFAULT,
                                  diag::InjectionDiagnostics2D =
                                      InjectionDiagnostics2D(length(leaves)),
                                  proj_stats::Union{ProjectionStats,Nothing} = nothing)
    N = length(leaves)
    Δt = Float64(dt)
    ρ_t = Float64(ρ_ref)
    Γ = Float64(Gamma)

    # Pre-compute face-neighbor tables (per-axis lo/hi neighbor indices
    # in leaf-major order). `face_*_idx[a][i] == 0` ⇒ boundary / out-of-
    # domain on axis `a` for leaf `i`; injection skipped for that cell.
    face_lo_idx, face_hi_idx = build_face_neighbor_tables(mesh, leaves, bc_spec)

    # Per-cell ρ, Mvv, s, Pp snapshot (used by all axes to apply KE-debit
    # / entropy re-anchor in a consistent order).
    ρ_v   = fill(ρ_t, N)         # zero-strain limit; M3-4 IC bridge
    s_v   = Vector{Float64}(undef, N)
    Pp_v  = Vector{Float64}(undef, N)
    @inbounds for (i, ci) in enumerate(leaves)
        s_v[i]  = Float64(fields.s[ci][1])
        Pp_v[i] = Float64(fields.Pp[ci][1])
        if !isfinite(Pp_v[i])
            Pp_v[i] = 0.0
        end
    end

    # Per-axis loop. Axes NOT in `axes` are skipped — no RNG draws,
    # no divu eval, no field writes ⇒ byte-equal selectivity.
    for a in axes
        @assert a == 1 || a == 2 "inject_vg_noise_HG_2d!: axis $a out of range"
        # ── (1) Per-axis divu_a from face-neighbor u_a differences. ──
        divu_a = diag.divu[a]
        face_lo = face_lo_idx[a]
        face_hi = face_hi_idx[a]
        @inbounds for (i, ci) in enumerate(leaves)
            ilo = face_lo[i]
            ihi = face_hi[i]
            if ilo == 0 || ihi == 0
                divu_a[i] = 0.0
                diag.compressive[a][i] = false
                continue
            end
            ci_lo = leaves[ilo]
            ci_hi = leaves[ihi]
            lo_self, hi_self = cell_physical_box(frame, ci)
            lo_lo, hi_lo     = cell_physical_box(frame, ci_lo)
            lo_hi, hi_hi     = cell_physical_box(frame, ci_hi)
            # Centered finite-difference: (u_a(hi) - u_a(lo)) / (x_a(hi) - x_a(lo))
            x_self = 0.5 * (lo_self[a] + hi_self[a])
            x_lo   = 0.5 * (lo_lo[a]   + hi_lo[a])
            x_hi   = 0.5 * (lo_hi[a]   + hi_hi[a])
            u_a_field = a == 1 ? fields.u_1 : fields.u_2
            u_lo = Float64(u_a_field[ci_lo][1])
            u_hi = Float64(u_a_field[ci_hi][1])
            Δx = x_hi - x_lo
            # Periodic-wrap: if neighbor's box is on opposite wall, add
            # the box length to keep Δx > 0. For non-wrap interior the
            # raw difference is already correct.
            L_a = Float64(frame.hi[a] - frame.lo[a])
            if Δx ≤ 0
                Δx += L_a
            end
            divu_a[i] = (u_hi - u_lo) / Δx
            diag.compressive[a][i] = divu_a[i] < 0
        end

        # ── (2) Draw VG noise per axis (length-N buffer) and smooth. ──
        eta_white = Vector{Float64}(undef, N)
        rand_variance_gamma!(rng, eta_white, params.λ, params.θ_factor)
        eta_a = diag.eta[a]
        if params.ell_corr > 0 && N >= 3
            smooth_periodic_3pt!(eta_a, eta_white)
        else
            copyto!(eta_a, eta_white)
        end

        # ── (3) Per-cell drift / noise / limiter / KE-debit / entropy. ─
        @inbounds for (i, ci) in enumerate(leaves)
            ρ_i = ρ_v[i]
            divu_i = divu_a[i]
            drift_term = params.C_A * ρ_i * divu_i * Δt
            compression = max(-divu_i, 0.0)
            noise_amp = params.C_B * ρ_i * sqrt(compression * Δt)
            δ_drift = drift_term
            δ_noise = noise_amp * eta_a[i]
            δ = δ_drift + δ_noise

            # Read per-axis u_a and current s, Pp.
            u_a_field = a == 1 ? fields.u_1 : fields.u_2
            β_a_field = a == 1 ? fields.β_1 : fields.β_2
            u_pre = Float64(u_a_field[ci][1])
            s_pre = s_v[i]
            Mvv_pre = Float64(Mvv(1.0 / ρ_i, s_pre; Gamma = Γ))
            Pxx_pre = ρ_i * Mvv_pre
            Pp_pre  = Pp_v[i]

            # Amplitude limiter against IE budget.
            IE_local = 0.5 * Pxx_pre + Pp_pre
            KE_budget = params.ke_budget_fraction * IE_local
            abs_u = abs(u_pre)
            disc = abs_u * abs_u + 2.0 * KE_budget / max(ρ_i, 1e-30)
            δ_max = (-abs_u + sqrt(max(disc, 0.0))) * ρ_i
            if δ > δ_max
                δ = δ_max
            elseif δ < -δ_max
                δ = -δ_max
            end
            diag.delta_rhou[a][i] = δ

            # KE-debit to (P_xx, P_⊥); re-anchor entropy.
            ΔKE_vol = u_pre * δ + 0.5 * δ * δ / ρ_i
            diag.delta_KE_vol[a][i] = ΔKE_vol
            Pxx_new = Pxx_pre - (2.0 / 3.0) * ΔKE_vol
            Pp_new  = Pp_pre  - (2.0 / 3.0) * ΔKE_vol
            # Apply pressure floor ONLY when a non-zero injection
            # actually perturbed the cell. This preserves the cold-
            # limit `Pp = 0` IC byte-exactly across no-op steps (uniform
            # IC, ΔKE_vol = 0). The 1D path's IC sets Pp to the
            # isotropic Maxwellian and never sits at 0; in 2D the IC
            # bridge sets Pp = 0 (cold limit) and we must not silently
            # raise it to the floor when no injection fired.
            if ΔKE_vol != 0.0
                if Pxx_new < params.pressure_floor
                    Pxx_new = params.pressure_floor
                end
                if Pp_new < params.pressure_floor
                    Pp_new = params.pressure_floor
                end
            end

            Mvv_new = Pxx_new / ρ_i
            if Mvv_new > 0 && Mvv_pre > 0
                s_new = s_pre + log(Mvv_new / Mvv_pre)
            else
                s_new = s_pre
            end

            # Mutate `(s, Pp, u_a)` only; β_a left untouched (the 2D
            # variational integrator's β_a is a Newton unknown driven
            # by the pressure stencil, not the post-Newton injection
            # — same ledger as the 1D path's `β` post-Newton update).
            #
            # u_a update: δ/ρ_i is the per-cell velocity increment.
            u_new = u_pre + δ / ρ_i
            u_a_field[ci] = (u_new,)

            # Stage `s, Pp` updates into local snapshot so sequential
            # axes (axis 2 after axis 1, when `axes = (1, 2)`) see the
            # updated thermodynamic state — exactly as the operator-
            # split per-step ledger requires.
            s_v[i]  = s_new
            Pp_v[i] = Pp_new
        end
    end

    # Write the staged s, Pp back to the field set.
    @inbounds for (i, ci) in enumerate(leaves)
        fields.s[ci]  = (s_v[i],)
        fields.Pp[ci] = (Pp_v[i],)
    end

    # ── (4) Realizability projection (per-axis + 4-comp cone). ────────
    if params.project_kind === :none
        if proj_stats !== nothing
            proj_stats.n_steps += 1
        end
    else
        realizability_project_2d!(fields, leaves;
                                   project_kind = params.project_kind,
                                   headroom = params.realizability_headroom,
                                   Mvv_floor = params.Mvv_floor,
                                   pressure_floor = params.pressure_floor,
                                   ρ_ref = ρ_ref,
                                   Gamma = Gamma,
                                   stats = proj_stats)
    end

    return fields, diag
end

# ─────────────────────────────────────────────────────────────────────
# M3-6 Phase 3 (c): per-species per-axis γ diagnostic field walker.
# ─────────────────────────────────────────────────────────────────────
#
# Wraps `gamma_per_axis_2d_field` (in `src/diagnostics.jl`) over
# multiple species. Each species can carry its own `M_vv` override
# (e.g. dust species are pressureless ⇒ M_vv ≡ 0; gas species use
# the EOS Mvv(J, s)).
#
# Returns a `Float64` 3-tensor of shape (N_species, 2, N_leaves) so
# downstream callers can index `out[k, a, i]` for species `k`, axis
# `a`, leaf `i`. (The brief mentions `SArray{N_species, 2, T}` but
# that's the per-leaf shape; the field walker returns a stacked
# Array{T,3} for memory efficiency over many leaves.)

"""
    gamma_per_axis_2d_per_species_field(fields, leaves;
                                          M_vv_override_per_species = nothing,
                                          ρ_ref = 1.0,
                                          Gamma = GAMMA_LAW_DEFAULT,
                                          n_species = 1)
        -> Array{Float64, 3}

Per-species per-axis γ diagnostic on the 2D Cholesky-sector field
set. Walks `leaves` and computes for each leaf, axis, and species

    γ[k, a, i] = √max(M_vv,aa[k] − β_a²[i], 0)         for a = 1, 2.

where `β_a` is read from `fields.β_a[ci]` (shared across species —
the per-axis β is a fluid-state Newton unknown, not a per-species
field) and `M_vv,aa[k]` comes from either a per-species override or
the EOS-derived `Mvv(J, s)`.

Returns an `Array{Float64, 3}` of shape `(n_species, 2,
length(leaves))`.

# `M_vv_override_per_species`

Either `nothing` (use EOS-derived `Mvv(J, s)` for every species,
isotropic) or a vector / tuple of length `n_species`, each entry
itself a 2-tuple `(M_vv,11_k, M_vv,22_k)`. The dust-trap (D.7) use
case sets `M_vv_k = (0, 0)` for the pressureless dust species
while the gas species use the EOS.

# Multi-species independence (M3-6 Phase 3 gate)

Each species' γ is computed independently — no cross-species reads
or writes. The single-species `n_species == 1` path reduces byte-
equally to `gamma_per_axis_2d_field(fields, leaves; M_vv_override =
M_vv_override_per_species[1])`.

# Wraps the math primitive

Internally calls `gamma_per_axis_2d_field` per species (or inlines
the equivalent loop). The math primitive is
`gamma_per_axis_2d(β, M_vv_diag) = SVector{2, T}(...)` in
`src/cholesky_DD.jl` §"Per-axis γ diagnostic"; this walker is the
multi-species field-set-layer wrapper.
"""
function gamma_per_axis_2d_per_species_field(fields, leaves;
                                                M_vv_override_per_species = nothing,
                                                ρ_ref::Real = 1.0,
                                                Gamma::Real = GAMMA_LAW_DEFAULT,
                                                n_species::Integer = 1)
    @assert n_species ≥ 1 "n_species must be ≥ 1"
    if M_vv_override_per_species !== nothing
        @assert length(M_vv_override_per_species) == n_species "M_vv_override_per_species length must match n_species"
    end
    N = length(leaves)
    out = zeros(Float64, n_species, 2, N)
    @inbounds for k in 1:n_species
        Mvv_k = M_vv_override_per_species === nothing ? nothing :
                M_vv_override_per_species[k]
        γ_mat = gamma_per_axis_2d_field(fields, leaves;
                                         M_vv_override = Mvv_k,
                                         ρ_ref = ρ_ref,
                                         Gamma = Gamma)
        for i in 1:N
            out[k, 1, i] = γ_mat[1, i]
            out[k, 2, i] = γ_mat[2, i]
        end
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────
# M4 Phase 3: Per-species momentum coupling (D.7 follow-up).
# ─────────────────────────────────────────────────────────────────────
#
# The M3-6 Phase 4 dust-trap finding (FALSIFIED on the literal claim,
# `reference/notes_M3_6_phase4_D7_dust_traps.md`) showed that the
# Phase 3 substrate's `advect_tracers_HG_2d!` is a no-op (pure-
# Lagrangian byte-stable contract) so passive-scalar tracers cannot
# accumulate at vortex centres. The methods paper §10.5 D.7 prediction
# requires a *physics extension*: per-species momentum + drag.
#
# This M4 Phase 3 implementation adds:
#
#   • `PerSpeciesMomentumHG2D{T}` — wraps a `TracerMeshHG2D` plus
#     per-species (u_x, u_y) fields per cell, per-species `ρ`
#     (mass per cell), per-species drag relaxation timescale
#     `τ_drag` (one per species), and per-species Lagrangian
#     position offsets `(x_k, y_k)` per cell.
#   • `drag_relax_per_species!(psm, dt)` — exponential relaxation
#     of each species' velocity toward gas: `u_k ← u_k +
#     (1 − exp(−dt/τ_k)) · (u_gas − u_k)`. For τ → 0 (passive-
#     scalar limit) the dust velocity tracks gas exactly per step;
#     for τ → ∞ (decoupled limit) the velocities are unchanged.
#   • `advance_positions_per_species!(psm, dt)` — kinematic update
#     `x_k ← x_k + dt · u_k` per species per axis. The position
#     offsets are stored *relative to the cell centre at IC* so a
#     species that moves with the gas has zero offset; a decoupled
#     species accumulates centrifugal drift relative to the gas
#     frame.
#   • `accumulate_dust_at_vortex!(psm, frame, leaves, lo, L1, L2)`
#     — remap dust mass from per-species position offsets back into
#     a per-cell concentration. This is the kinematic mechanism by
#     which decoupled dust accumulates at vortex centres: dust whose
#     Lagrangian path drifts toward a centre deposits its mass at
#     the cell containing the drifted position.
#
# Bit-exact opt-in contract: at construction with `enable=false` (or
# by simply not using `PerSpeciesMomentumHG2D`), nothing in the
# substrate path changes — `det_step_2d_berry_HG!` byte-equal,
# `advect_tracers_HG_2d!` byte-equal. The M3-6 Phase 4 / Phase 3
# regression suite is preserved unconditionally.

"""
    PerSpeciesMomentumHG2D{T<:Real}

M4 Phase 3 per-species momentum extension of `TracerMeshHG2D`.

Wraps a `TracerMeshHG2D` (which holds per-species per-cell
concentrations) and adds per-species per-cell velocity components
plus a Lagrangian position offset (relative to the cell centre at
IC). The drag kernel relaxes each species' velocity toward the gas
velocity stored in the underlying field set's `u_1, u_2` slots; the
position kernel advects the per-species position by its own velocity
(NOT the gas velocity), so a decoupled species drifts relative to
the Eulerian frame.

Storage:
- `tm::TracerMeshHG2D{T}` — base tracer mesh (concentrations + names).
- `u_per_species::Array{T, 3}` — `(N_species, 2, n_cells)` per-cell
  velocity components.
- `dx_per_species::Array{T, 3}` — `(N_species, 2, n_cells)` per-cell
  Lagrangian position offset relative to the cell centre at IC.
- `τ_drag_per_species::Vector{T}` — drag relaxation timescale per
  species (length N_species). `τ → 0` ⇒ tightly-coupled (passive-
  scalar limit); `τ → ∞` ⇒ decoupled (free-particles); intermediate
  ⇒ Stokes-drag relaxation.

# Bit-exact opt-in contract

The struct is purely additive — using `TracerMeshHG2D` directly (the
M3-6 Phase 3 / Phase 4 path) is byte-equal to the pre-M4-Phase-3
codebase. The momentum coupling is invoked explicitly via
`drag_relax_per_species!` and `advance_positions_per_species!`.

# Initial conditions

At construction the per-species velocity is initialised to the gas
velocity (read from `tm.fields.u_1, tm.fields.u_2`) so all species
are co-moving at t = 0. The position offsets are zero. Gas (species
1 by convention) keeps `τ_drag[1] = 0` so it byte-tracks the gas
field; dust species have `τ_drag[k] > 0`.
"""
mutable struct PerSpeciesMomentumHG2D{T<:Real}
    tm::TracerMeshHG2D{T}
    u_per_species::Array{T, 3}
    dx_per_species::Array{T, 3}
    τ_drag_per_species::Vector{T}
end

"""
    PerSpeciesMomentumHG2D(tm::TracerMeshHG2D{T};
                            τ_drag_per_species,
                            leaves) -> PerSpeciesMomentumHG2D{T}

Allocate a per-species momentum extension on top of `tm`. The
per-species velocity is initialised to the gas velocity stored in
`tm.fields.u_1, tm.fields.u_2` at every leaf; non-leaf cells are
left at zero (mirrors the field-set storage convention).

`τ_drag_per_species` must have length `n_species(tm)`; entry `k = 1`
(the gas species) should be `0.0` (tightly-coupled to itself) for
the canonical 2-species `[:gas, :dust]` configuration.
"""
function PerSpeciesMomentumHG2D(tm::TracerMeshHG2D{T};
                                  τ_drag_per_species::AbstractVector,
                                  leaves::AbstractVector{<:Integer}) where {T<:Real}
    K = n_species(tm)
    @assert length(τ_drag_per_species) == K "τ_drag_per_species length must equal n_species"
    nc = n_cells_2d(tm)
    u_arr = zeros(T, K, 2, nc)
    dx_arr = zeros(T, K, 2, nc)
    @inbounds for ci in leaves
        u1 = T(tm.fields.u_1[ci][1])
        u2 = T(tm.fields.u_2[ci][1])
        for k in 1:K
            u_arr[k, 1, ci] = u1
            u_arr[k, 2, ci] = u2
        end
    end
    τ = collect(T, τ_drag_per_species)
    return PerSpeciesMomentumHG2D{T}(tm, u_arr, dx_arr, τ)
end

"""
    n_species(psm::PerSpeciesMomentumHG2D) -> Int
"""
n_species(psm::PerSpeciesMomentumHG2D) = n_species(psm.tm)

"""
    species_index(psm::PerSpeciesMomentumHG2D, name::Symbol) -> Int
"""
species_index(psm::PerSpeciesMomentumHG2D, name::Symbol) =
    species_index(psm.tm, name)

"""
    drag_relax_per_species!(psm::PerSpeciesMomentumHG2D, dt;
                              leaves) -> psm

Stokes-drag exponential relaxation of each species' velocity toward
the gas velocity. For each leaf cell `ci` and each species `k`,
each axis `a`:

    u_k[a] ← u_k[a] + (1 − exp(−dt / τ_k)) · (u_gas[a] − u_k[a])

with `u_gas` read from `psm.tm.fields.u_1, .u_2` at cell `ci`. The
case `τ_k = 0` is treated as "infinitely tight coupling": `u_k ←
u_gas` (passive-scalar limit). The case `τ_k = Inf` is treated as
"decoupled": `u_k` unchanged. Intermediate τ produces Stokes-drag
relaxation toward gas.

Returns `psm` unchanged in identity (all writes in place).
"""
function drag_relax_per_species!(psm::PerSpeciesMomentumHG2D{T}, dt::Real;
                                   leaves::AbstractVector{<:Integer}) where {T<:Real}
    K = n_species(psm)
    dtT = T(dt)
    fields = psm.tm.fields
    @inbounds for ci in leaves
        ug1 = T(fields.u_1[ci][1])
        ug2 = T(fields.u_2[ci][1])
        for k in 1:K
            τk = psm.τ_drag_per_species[k]
            if τk == 0
                # Tightly-coupled limit — u_k = u_gas exactly.
                psm.u_per_species[k, 1, ci] = ug1
                psm.u_per_species[k, 2, ci] = ug2
            elseif !isfinite(τk)
                # Decoupled limit — leave u_k unchanged.
                continue
            else
                ratio = dtT / τk
                fac = ratio > 100 ? T(1) : T(1) - exp(-ratio)
                u1 = psm.u_per_species[k, 1, ci]
                u2 = psm.u_per_species[k, 2, ci]
                psm.u_per_species[k, 1, ci] = u1 + fac * (ug1 - u1)
                psm.u_per_species[k, 2, ci] = u2 + fac * (ug2 - u2)
            end
        end
    end
    return psm
end

"""
    advance_positions_per_species!(psm::PerSpeciesMomentumHG2D, dt;
                                     leaves) -> psm

Kinematic update of per-species Lagrangian position offset:

    dx_k[a] ← dx_k[a] + dt · u_k[a]

A species that tracks the gas velocity perfectly (`τ_k = 0`)
maintains `u_k = u_gas` so `dx_k` accumulates the integrated path
relative to the cell centre. A decoupled species (`τ_k = ∞`) keeps
its IC velocity, so its `dx_k` integrates that constant velocity.
Intermediate τ produces drift that lags the gas — the kinematic
mechanism for centrifugal accumulation in vortex flows.

Returns `psm` (writes in place).
"""
function advance_positions_per_species!(psm::PerSpeciesMomentumHG2D{T},
                                          dt::Real;
                                          leaves::AbstractVector{<:Integer}) where {T<:Real}
    K = n_species(psm)
    dtT = T(dt)
    @inbounds for ci in leaves
        for k in 1:K
            psm.dx_per_species[k, 1, ci] += dtT * psm.u_per_species[k, 1, ci]
            psm.dx_per_species[k, 2, ci] += dtT * psm.u_per_species[k, 2, ci]
        end
    end
    return psm
end

"""
    accumulate_species_to_cells!(psm::PerSpeciesMomentumHG2D, frame, leaves;
                                   k::Integer,
                                   lo::Tuple, hi::Tuple,
                                   wrap::Bool=true) -> Vector{T}

Remap species-`k`'s mass from its drifted Lagrangian positions back
into a per-cell concentration array. For each source cell `ci` with
concentration `c = psm.tm.tracers[k, ci]` and Lagrangian offset
`(dx, dy) = psm.dx_per_species[k, :, ci]`, the destination cell
centre `(cx + dx, cy + dy)` (modulo the box on each axis when
`wrap=true`) is computed and the cell containing that point gets
its concentration accumulator updated by `c` (volume-weighted —
since all leaves are equal-volume on a balanced quadtree this is
just an additive deposit divided by per-destination cell count).

Returns a `Vector{T}` of length `length(leaves)` giving the new
per-leaf concentration after remap. Total mass is conserved exactly
(deposits are bookkeeping moves of contributing concentrations).
"""
function accumulate_species_to_cells!(psm::PerSpeciesMomentumHG2D{T},
                                        frame, leaves::AbstractVector{<:Integer};
                                        k::Integer,
                                        lo::Tuple, hi::Tuple,
                                        wrap::Bool = true) where {T<:Real}
    n = length(leaves)
    @assert 1 ≤ k ≤ n_species(psm) "species index out of range"
    L1 = T(hi[1] - lo[1])
    L2 = T(hi[2] - lo[2])
    lo1 = T(lo[1]); lo2 = T(lo[2])
    # Cell centres + indices.
    cx_arr = Vector{T}(undef, n)
    cy_arr = Vector{T}(undef, n)
    @inbounds for (i, ci) in enumerate(leaves)
        lo_c, hi_c = HierarchicalGrids.cell_physical_box(frame, ci)
        cx_arr[i] = T(0.5 * (lo_c[1] + hi_c[1]))
        cy_arr[i] = T(0.5 * (lo_c[2] + hi_c[2]))
    end
    # Source mass: c * 1 (equal-volume cells; drop the constant factor).
    src_mass = Vector{T}(undef, n)
    @inbounds for (i, ci) in enumerate(leaves)
        src_mass[i] = T(psm.tm.tracers[k, ci])
    end
    # Compute destination cell index for each source.
    dest = Vector{Int}(undef, n)
    @inbounds for (i, ci) in enumerate(leaves)
        xt = cx_arr[i] + psm.dx_per_species[k, 1, ci]
        yt = cy_arr[i] + psm.dx_per_species[k, 2, ci]
        if wrap
            xt = lo1 + mod(xt - lo1, L1)
            yt = lo2 + mod(yt - lo2, L2)
        end
        # Find nearest cell-centre via brute-force (n is small at the
        # M3-6 Phase 4 acceptance window L ≤ 4).
        best = 1
        best_d = (xt - cx_arr[1])^2 + (yt - cy_arr[1])^2
        for j in 2:n
            d = (xt - cx_arr[j])^2 + (yt - cy_arr[j])^2
            if d < best_d
                best_d = d
                best = j
            end
        end
        dest[i] = best
    end
    # Accumulate destination mass.
    new_c = zeros(T, n)
    counts = zeros(Int, n)
    @inbounds for i in 1:n
        d = dest[i]
        new_c[d] += src_mass[i]
        counts[d] += 1
    end
    # Where multiple sources collapse onto the same dest, average
    # (preserves Σ c · A under equal-volume cells iff we record
    # density correctly: total mass = Σ_i c_i = Σ_i src_mass[i]; so
    # if cells `dest = j` accumulated count `n_j` with sum
    # `Σ_{i∈Sj} c_i`, the destination concentration `new_c[j] /
    # max(1, count_at_dest_j)` over-divides). We instead keep the
    # SUM at destinations that received traffic and 0 elsewhere; the
    # total-mass invariant is `sum(new_c) == sum(src_mass)`.
    return new_c
end

"""
    dust_peak_over_mean_remapped(psm, frame, leaves; lo, hi, wrap=true)
        -> NamedTuple

Compute the dust-species peak-over-mean ratio after remapping the
per-species Lagrangian positions back to cells. Returns:

  • `peak::Float64` — `max(new_c)` (most-piled-up cell concentration)
  • `mean::Float64` — `sum(new_c) / N` (preserved across remap)
  • `peak_over_mean::Float64` — `peak / mean` (ratio diagnostic)
  • `new_c::Vector` — per-leaf remapped concentration
  • `n_occupied::Int` — number of leaves that received any deposit
    (`< N` indicates dust collapse onto a smaller subset)
  • `collapse_fraction::Float64` — `1 − n_occupied / N` (0 = no
    collapse, 1 = all dust onto one cell)
"""
function dust_peak_over_mean_remapped(psm, frame,
                                        leaves::AbstractVector{<:Integer};
                                        lo::Tuple, hi::Tuple,
                                        wrap::Bool = true)
    k = species_index(psm.tm, :dust)
    new_c = accumulate_species_to_cells!(psm, frame, leaves;
                                          k = k, lo = lo, hi = hi,
                                          wrap = wrap)
    n = length(leaves)
    s = 0.0
    n_occ = 0
    @inbounds for i in 1:n
        v = Float64(new_c[i])
        s += v
        if v > 0
            n_occ += 1
        end
    end
    mean_c = s / n
    pk = Float64(maximum(new_c))
    pom = mean_c > 1e-300 ? pk / mean_c : NaN
    collapse_frac = 1.0 - n_occ / n
    return (peak = pk, mean = mean_c, peak_over_mean = pom,
            new_c = new_c, max_pile = pk,
            n_occupied = n_occ, collapse_fraction = collapse_frac)
end
