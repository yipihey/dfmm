# D7_dust_traps.jl
#
# §10.5 D.7 dust-trapping in vortices — M3-6 Phase 4
# (`reference/notes_M3_6_phase4_D7_dust_traps.md`).
#
# Methods paper §10.5 D.7: "KH instability with passive dust;
# vortex-center accumulation matches reference codes". The dfmm 2D
# variational scheme uses the simpler, more robust Taylor-Green
# vortex IC (single periodic vortex array on `[0, 1]²`) rather than
# a sheared KH base flow:
#
#     u_1(x, y) = U0 · sin(2π m_1) · cos(2π m_2)
#     u_2(x, y) = -U0 · cos(2π m_1) · sin(2π m_2)
#
# with dust as a passive scalar (TracerMeshHG2D 2-species: gas +
# dust; cf. `tier_d_dust_trap_ic_full`). The dust species is
# pressureless cold (M_vv = 0); per-species γ correctly identifies
# the dust phase (γ_dust = 0 everywhere) vs the gas phase (γ_gas
# finite).
#
# Drives the IC through `det_step_2d_berry_HG!` for several eddy
# turnover times, tracks per-step:
#   • per-species per-axis γ statistics (gas γ vs dust γ)
#   • dust mass conservation (Σ_leaves c_dust · A_cell)
#   • dust accumulation diagnostic: c_dust at vortex centres relative
#     to the spatial mean
#   • n_negative_jacobian (4-comp cone diagnostic)
#   • conservation invariants (M, Px, Py, KE)
#
# Saves a 4-panel CairoMakie headline plot showing:
#   (A) gas density (uniform — sanity)
#   (B) dust concentration map at end-time
#   (C) dust mass conservation over time
#   (D) per-species γ trajectories (gas vs dust)
#
# Usage (REPL):
#
#   julia> include("experiments/D7_dust_traps.jl")
#   julia> result = run_D7_dust_traps(; level=4)
#
# Or full sweep + plot:
#
#   julia> sweep = run_D7_dust_traps_sweep(; levels=(4, 5))
#   julia> plot_D7_dust_traps(sweep;
#              save_path="reference/figs/M3_6_phase4_D7_dust_traps.png")

using HierarchicalGrids: HierarchicalMesh, EulerianFrame,
    refine_cells!, enumerate_leaves, FrameBoundaries, REFLECTING, PERIODIC,
    cell_physical_box
using dfmm: tier_d_dust_trap_ic_full, allocate_cholesky_2d_fields,
    write_detfield_2d!, read_detfield_2d, DetField2D,
    det_step_2d_berry_HG!, gamma_per_axis_2d_field,
    gamma_per_axis_2d_per_species_field,
    advect_tracers_HG_2d!, n_species, species_index, n_cells_2d,
    ProjectionStats
using Statistics: mean, std

"""
    dust_trap_eddy_time(; U0=1.0, L=1.0) -> Float64

Eddy turnover time `t_eddy = L / U0` for the Taylor-Green vortex.
Used by the driver to set the simulation horizon.
"""
dust_trap_eddy_time(; U0::Real = 1.0, L::Real = 1.0) =
    Float64(L) / max(Float64(U0), 1e-300)

"""
    cell_areas_2d(frame, leaves) -> Vector{Float64}

Per-leaf cell area `(hi_x - lo_x)·(hi_y - lo_y)`.
"""
function cell_areas_2d(frame, leaves)
    A = Vector{Float64}(undef, length(leaves))
    @inbounds for (i, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        A[i] = (hi_c[1] - lo_c[1]) * (hi_c[2] - lo_c[2])
    end
    return A
end

"""
    dust_total_mass(tm, leaves, areas) -> Float64

Total integrated dust mass `Σ c_dust(ci) · A(ci)` over leaves.
Phase 3 contract: `advect_tracers_HG_2d!` is a no-op so the
underlying tracer matrix is byte-stable and this scalar is bit-
stable across `det_step_2d_berry_HG!` calls.
"""
function dust_total_mass(tm, leaves, areas)
    k = species_index(tm, :dust)
    s = 0.0
    @inbounds for (i, ci) in enumerate(leaves)
        s += tm.tracers[k, ci] * Float64(areas[i])
    end
    return s
end

"""
    gas_total_mass(tm, leaves, areas) -> Float64

Total integrated gas tracer mass `Σ c_gas(ci) · A(ci)`. By IC c_gas=1
so this equals the total integrated cell area.
"""
function gas_total_mass(tm, leaves, areas)
    k = species_index(tm, :gas)
    s = 0.0
    @inbounds for (i, ci) in enumerate(leaves)
        s += tm.tracers[k, ci] * Float64(areas[i])
    end
    return s
end

"""
    vortex_center_dust(tm, frame, leaves; lo, L1, L2) -> NamedTuple

Dust concentration evaluated at the four Taylor-Green vortex centre
positions `(m_1, m_2) ∈ {(0.25, 0.25), (0.25, 0.75), (0.75, 0.25),
(0.75, 0.75)}`. Returns the per-leaf-nearest-cell concentration at
each vortex centre + the spatial mean + the peak/mean ratio.

The IC's dust profile `c_dust = 1 + ε·sin(2π m_1)·sin(2π m_2)` has
`+ε` peaks at `(0.25, 0.25)` and `(0.75, 0.75)` (vortices A, D),
and `−ε` troughs at `(0.25, 0.75)` and `(0.75, 0.25)` (vortices B,
C). At t=0 the peak/mean ratio is `(1+ε)/1 = 1 + ε`. The Phase 4
acceptance gate is "dust at vortex centres > IC value × 1.1" but
honest reporting: under pure-Lagrangian advection (Phase 3
contract) the tracer matrix is byte-stable so this ratio is
constant — the 10% accumulation gate cannot fire on the current
substrate.
"""
function vortex_center_dust(tm, frame, leaves;
                              lo = (0.0, 0.0),
                              L1::Real = 1.0, L2::Real = 1.0)
    k = species_index(tm, :dust)
    targets = (((0.25, 0.25)), ((0.25, 0.75)),
               ((0.75, 0.25)), ((0.75, 0.75)))
    # Find nearest leaf cell to each vortex centre.
    n = length(leaves)
    best_dist = fill(Inf, length(targets))
    best_ci = fill(0, length(targets))
    centres = Vector{NTuple{2, Float64}}(undef, n)
    @inbounds for (i, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        cx = 0.5 * (lo_c[1] + hi_c[1])
        cy = 0.5 * (lo_c[2] + hi_c[2])
        centres[i] = (cx, cy)
        for (j, t) in enumerate(targets)
            tx = Float64(lo[1]) + Float64(t[1]) * Float64(L1)
            ty = Float64(lo[2]) + Float64(t[2]) * Float64(L2)
            d = (cx - tx)^2 + (cy - ty)^2
            if d < best_dist[j]
                best_dist[j] = d
                best_ci[j] = ci
            end
        end
    end
    c_at = Vector{Float64}(undef, length(targets))
    @inbounds for j in eachindex(targets)
        c_at[j] = tm.tracers[k, best_ci[j]]
    end
    # Spatial mean over leaves.
    s = 0.0
    @inbounds for ci in leaves
        s += tm.tracers[k, ci]
    end
    c_mean = s / n
    c_peak = maximum(c_at)
    c_min = minimum(c_at)
    peak_over_mean = c_mean > 1e-300 ? c_peak / c_mean : NaN
    return (
        c_at_vortex = c_at,
        c_mean = c_mean,
        c_peak = c_peak,
        c_min = c_min,
        peak_over_mean = peak_over_mean,
    )
end

"""
    dust_trap_conservation(fields, leaves, ρ_per_cell, areas) -> NamedTuple

Compute fluid conservation invariants `(M, Px, Py, KE)` and dust
mass `M_dust`. The fluid M, Px, Py, KE follow `det_step_2d_berry_HG!`
evolution; `M_dust` is the integrated dust concentration.

Returns `(M, Px, Py, KE)` (gas-fluid only; dust integrated separately).
"""
function dust_trap_conservation(fields, leaves, ρ_per_cell, areas)
    M = 0.0; Px = 0.0; Py = 0.0; KE = 0.0
    @inbounds for (i, ci) in enumerate(leaves)
        ρ_i = Float64(ρ_per_cell[i])
        A_i = Float64(areas[i])
        ux = Float64(fields.u_1[ci][1])
        uy = Float64(fields.u_2[ci][1])
        M += ρ_i * A_i
        Px += ρ_i * ux * A_i
        Py += ρ_i * uy * A_i
        KE += 0.5 * ρ_i * (ux * ux + uy * uy) * A_i
    end
    return (M = M, Px = Px, Py = Py, KE = KE)
end

"""
    negative_jacobian_count_dust_trap(fields, leaves; M_vv_override, ρ_ref) -> Int

Per-step safety: count cells with γ_a ≤ 1e-12 (γ-cone determinant
near zero). With the M3-6 Phase 1b 4-component projection this
should remain at 0 throughout the run on the gas (M_vv finite)
species. The dust species is pressureless (M_vv = 0) so γ_dust =
0 by construction — that's not a "negative Jacobian", it's the
pressureless cold-limit definition.
"""
function negative_jacobian_count_dust_trap(fields, leaves;
                                             M_vv_override = (1.0, 1.0),
                                             ρ_ref::Real = 1.0)
    γ = gamma_per_axis_2d_field(fields, leaves;
                                  M_vv_override = M_vv_override,
                                  ρ_ref = ρ_ref)
    n_neg = 0
    @inbounds for i in 1:size(γ, 2)
        if γ[1, i] ≤ 1e-12 || γ[2, i] ≤ 1e-12
            n_neg += 1
        end
    end
    return n_neg
end

"""
    per_species_gamma_stats(fields, leaves; n_species, M_vv_per_species, ρ_ref) -> NamedTuple

Compute per-species per-axis γ summary statistics. Returns a
NamedTuple with per-species per-axis (mean, std, min, max) plus
the species-separation ratio `gamma_separation = mean(γ_gas) /
max(γ_dust_max, 1e-300)` — measures how cleanly the per-species γ
diagnostic separates the two phases.
"""
function per_species_gamma_stats(fields, leaves;
                                   n_species_n::Integer = 2,
                                   M_vv_per_species = ((1.0, 1.0),
                                                        (0.0, 0.0)),
                                   ρ_ref::Real = 1.0)
    γ = gamma_per_axis_2d_per_species_field(fields, leaves;
                                              M_vv_override_per_species = M_vv_per_species,
                                              ρ_ref = ρ_ref,
                                              n_species = n_species_n)
    # Shape (n_species, 2, N).
    gas_g = γ[1, :, :]
    dust_g = γ[2, :, :]
    g1_mean = mean(gas_g)
    g1_std = std(gas_g)
    g1_min = minimum(gas_g)
    g1_max = maximum(gas_g)
    d1_mean = mean(dust_g)
    d1_std = std(dust_g)
    d1_min = minimum(dust_g)
    d1_max = maximum(dust_g)
    sep = g1_mean / max(d1_max, 1e-300)
    return (
        gas_mean = g1_mean, gas_std = g1_std,
        gas_min = g1_min, gas_max = g1_max,
        dust_mean = d1_mean, dust_std = d1_std,
        dust_min = d1_min, dust_max = d1_max,
        gamma_separation = sep,
    )
end

"""
    run_D7_dust_traps(; level=4, U0=1.0, ρ0=1.0, P0=1.0, ε_dust=0.05,
                       T_factor=2.0, dt=nothing,
                       project_kind=:reanchor,
                       realizability_headroom=1.05,
                       Mvv_floor=1e-2, pressure_floor=1e-8,
                       M_vv_override=(1.0, 1.0),
                       ρ_ref=1.0,
                       snapshots_at=(0.0, 0.5, 1.0),
                       verbose=false) -> NamedTuple

Drive a single mesh-level D.7 dust-traps trajectory.

  • Builds the IC via `tier_d_dust_trap_ic_full` at the requested
    level (resolution `2^level × 2^level`). The IC also allocates
    a 2-species `TracerMeshHG2D` with `(:gas, :dust)`.
  • Attaches the standard dust-trap BCs: PERIODIC on both axes
    (Taylor-Green vortex is doubly-periodic).
  • Runs `det_step_2d_berry_HG!` for `T_end ≈ T_factor · t_eddy`
    (default 2 eddy turnovers).
  • Calls `advect_tracers_HG_2d!` per step (no-op pure-Lagrangian).
  • Tracks per-step:
      - per-species γ stats via `per_species_gamma_stats`
      - dust mass conservation: `M_dust(t) - M_dust(0)`
      - vortex-center dust accumulation
      - n_negative_jacobian (gas only — dust γ = 0 by construction)
      - conservation invariants (M, Px, Py, KE)
  • Saves spatial profile snapshots at `snapshots_at` (fractions
    of `T_end`).

Returns a NamedTuple with trajectory arrays + diagnostics + IC handle.
"""
function run_D7_dust_traps(; level::Integer = 4,
                              U0::Real = 1.0,
                              ρ0::Real = 1.0,
                              P0::Real = 1.0,
                              ε_dust::Real = 0.05,
                              T_factor::Real = 2.0,
                              T_end::Union{Real,Nothing} = nothing,
                              dt::Union{Real,Nothing} = nothing,
                              project_kind::Symbol = :reanchor,
                              realizability_headroom::Real = 1.05,
                              Mvv_floor::Real = 1e-2,
                              pressure_floor::Real = 1e-8,
                              M_vv_override = (1.0, 1.0),
                              ρ_ref::Real = 1.0,
                              snapshots_at = (0.0, 0.5, 1.0),
                              verbose::Bool = false)
    t_eddy = dust_trap_eddy_time(; U0 = U0)
    T_end_val = T_end === nothing ? Float64(T_factor) * t_eddy : Float64(T_end)
    if dt === nothing
        # Mesh-scaled dt. CFL: dt < Δx / U0 / 2.
        Δx = 1.0 / (2^Int(level))
        dt_val = 0.25 * Δx / max(Float64(U0), 1e-300)
        # Cap so we always get at least 30 samples.
        dt_val = min(dt_val, T_end_val / 30.0)
    else
        dt_val = Float64(dt)
    end
    n_steps = Int(ceil(T_end_val / dt_val))
    dt_val = T_end_val / n_steps

    # Build IC.
    ic = tier_d_dust_trap_ic_full(; level = level, U0 = U0,
                                    ρ0 = ρ0, P0 = P0,
                                    ε_dust = ε_dust)
    bc_dust = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                   (PERIODIC, PERIODIC)))
    areas = cell_areas_2d(ic.frame, ic.leaves)
    L1 = ic.params.L1
    L2 = ic.params.L2
    lo_box = ic.params.lo

    M_vv_per_species = ((Float64(M_vv_override[1]), Float64(M_vv_override[2])),
                         (0.0, 0.0))

    # Pre-allocate trajectory arrays.
    N = n_steps + 1
    t = zeros(Float64, N)
    M_dust_traj = zeros(Float64, N)
    M_gas_traj = zeros(Float64, N)
    M_traj = zeros(Float64, N)
    Px_traj = zeros(Float64, N)
    Py_traj = zeros(Float64, N)
    KE_traj = zeros(Float64, N)
    n_neg_jac = zeros(Int, N)
    gas_gamma_mean = zeros(Float64, N)
    gas_gamma_std = zeros(Float64, N)
    gas_gamma_min = zeros(Float64, N)
    dust_gamma_max = zeros(Float64, N)
    gamma_separation = zeros(Float64, N)
    dust_peak = zeros(Float64, N)
    dust_mean_arr = zeros(Float64, N)
    peak_over_mean = zeros(Float64, N)

    # Snapshot bookkeeping.
    snap_times = collect(Float64, snapshots_at)
    snap_indices = Int[]
    for tf in snap_times
        target = tf * T_end_val
        idx = clamp(Int(round(target / dt_val)) + 1, 1, N)
        push!(snap_indices, idx)
    end
    snapshots = Vector{NamedTuple}(undef, length(snap_times))
    snap_taken = falses(length(snap_times))

    # Helper: capture a spatial snapshot (cell centres + dust map +
    # gas gamma map for the headline plot).
    function take_snapshot(t_now)
        k_dust = species_index(ic.tm, :dust)
        x_centres = Vector{Float64}(undef, length(ic.leaves))
        y_centres = Vector{Float64}(undef, length(ic.leaves))
        c_dust_arr = Vector{Float64}(undef, length(ic.leaves))
        u1_arr = Vector{Float64}(undef, length(ic.leaves))
        u2_arr = Vector{Float64}(undef, length(ic.leaves))
        for (i, ci) in enumerate(ic.leaves)
            lo_c, hi_c = cell_physical_box(ic.frame, ci)
            x_centres[i] = 0.5 * (lo_c[1] + hi_c[1])
            y_centres[i] = 0.5 * (lo_c[2] + hi_c[2])
            c_dust_arr[i] = ic.tm.tracers[k_dust, ci]
            u1_arr[i] = Float64(ic.fields.u_1[ci][1])
            u2_arr[i] = Float64(ic.fields.u_2[ci][1])
        end
        γ_gas = gamma_per_axis_2d_field(ic.fields, ic.leaves;
                                          M_vv_override = M_vv_override,
                                          ρ_ref = ρ_ref)
        γ_gas_1 = γ_gas[1, :]
        return (t = t_now, x_centres = x_centres, y_centres = y_centres,
                c_dust = c_dust_arr, u_1 = u1_arr, u_2 = u2_arr,
                γ_gas_1 = γ_gas_1)
    end

    # Initial diagnostics.
    M_dust_traj[1] = dust_total_mass(ic.tm, ic.leaves, areas)
    M_gas_traj[1] = gas_total_mass(ic.tm, ic.leaves, areas)
    cons0 = dust_trap_conservation(ic.fields, ic.leaves, ic.ρ_per_cell, areas)
    M_traj[1] = cons0.M; Px_traj[1] = cons0.Px
    Py_traj[1] = cons0.Py; KE_traj[1] = cons0.KE
    n_neg_jac[1] = negative_jacobian_count_dust_trap(ic.fields, ic.leaves;
                                                      M_vv_override = M_vv_override,
                                                      ρ_ref = ρ_ref)
    gstats0 = per_species_gamma_stats(ic.fields, ic.leaves;
                                        n_species_n = 2,
                                        M_vv_per_species = M_vv_per_species,
                                        ρ_ref = ρ_ref)
    gas_gamma_mean[1] = gstats0.gas_mean
    gas_gamma_std[1] = gstats0.gas_std
    gas_gamma_min[1] = gstats0.gas_min
    dust_gamma_max[1] = gstats0.dust_max
    gamma_separation[1] = gstats0.gamma_separation
    vc0 = vortex_center_dust(ic.tm, ic.frame, ic.leaves;
                              lo = lo_box, L1 = L1, L2 = L2)
    dust_peak[1] = vc0.c_peak
    dust_mean_arr[1] = vc0.c_mean
    peak_over_mean[1] = vc0.peak_over_mean
    if 1 in snap_indices
        for (k, idx) in enumerate(snap_indices)
            if idx == 1 && !snap_taken[k]
                snapshots[k] = take_snapshot(0.0)
                snap_taken[k] = true
            end
        end
    end

    proj_stats = ProjectionStats()
    nan_seen = false
    wall_t0 = time()
    for n in 1:n_steps
        try
            det_step_2d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                                    bc_dust, dt_val;
                                    M_vv_override = M_vv_override,
                                    ρ_ref = ρ_ref,
                                    project_kind = project_kind,
                                    realizability_headroom = realizability_headroom,
                                    Mvv_floor = Mvv_floor,
                                    pressure_floor = pressure_floor,
                                    proj_stats = proj_stats)
            advect_tracers_HG_2d!(ic.tm, dt_val)
        catch e
            if verbose
                @warn "Newton solve failed at step $n: $e"
            end
            nan_seen = true
            break
        end

        t[n + 1] = n * dt_val
        M_dust_traj[n + 1] = dust_total_mass(ic.tm, ic.leaves, areas)
        M_gas_traj[n + 1] = gas_total_mass(ic.tm, ic.leaves, areas)
        cons = dust_trap_conservation(ic.fields, ic.leaves, ic.ρ_per_cell, areas)
        M_traj[n + 1] = cons.M; Px_traj[n + 1] = cons.Px
        Py_traj[n + 1] = cons.Py; KE_traj[n + 1] = cons.KE
        n_neg_jac[n + 1] = negative_jacobian_count_dust_trap(ic.fields, ic.leaves;
                                                              M_vv_override = M_vv_override,
                                                              ρ_ref = ρ_ref)
        gstats = per_species_gamma_stats(ic.fields, ic.leaves;
                                           n_species_n = 2,
                                           M_vv_per_species = M_vv_per_species,
                                           ρ_ref = ρ_ref)
        gas_gamma_mean[n + 1] = gstats.gas_mean
        gas_gamma_std[n + 1] = gstats.gas_std
        gas_gamma_min[n + 1] = gstats.gas_min
        dust_gamma_max[n + 1] = gstats.dust_max
        gamma_separation[n + 1] = gstats.gamma_separation
        vc = vortex_center_dust(ic.tm, ic.frame, ic.leaves;
                                  lo = lo_box, L1 = L1, L2 = L2)
        dust_peak[n + 1] = vc.c_peak
        dust_mean_arr[n + 1] = vc.c_mean
        peak_over_mean[n + 1] = vc.peak_over_mean

        for (k, idx) in enumerate(snap_indices)
            if idx == n + 1 && !snap_taken[k]
                snapshots[k] = take_snapshot(t[n + 1])
                snap_taken[k] = true
            end
        end

        if !isfinite(gas_gamma_mean[n + 1]) || !isfinite(M_dust_traj[n + 1])
            nan_seen = true
            break
        end

        if verbose && (n % max(1, n_steps ÷ 10) == 0)
            @info "Step $n / $n_steps: t=$(round(t[n+1]; digits=4))," *
                  " M_dust err=$(round(M_dust_traj[n+1]-M_dust_traj[1]; sigdigits=3))," *
                  " peak/mean=$(round(peak_over_mean[n+1]; sigdigits=4))"
        end
    end
    wall_t1 = time()
    wall_time_per_step = (wall_t1 - wall_t0) / max(n_steps, 1)

    # Take any missing snapshots at the final step.
    for (k, taken) in enumerate(snap_taken)
        if !taken
            snapshots[k] = take_snapshot(t[end])
            snap_taken[k] = true
        end
    end

    # Conservation drift over the run.
    M_dust_err_max = maximum(abs.(M_dust_traj .- M_dust_traj[1]))
    M_gas_err_max = maximum(abs.(M_gas_traj .- M_gas_traj[1]))
    M_err_max = maximum(abs.(M_traj .- M_traj[1]))
    Px_err_max = maximum(abs.(Px_traj .- Px_traj[1]))
    Py_err_max = maximum(abs.(Py_traj .- Py_traj[1]))
    KE_err_max = maximum(abs.(KE_traj .- KE_traj[1]))

    return (
        t = t,
        M_dust_traj = M_dust_traj, M_gas_traj = M_gas_traj,
        M_traj = M_traj, Px_traj = Px_traj,
        Py_traj = Py_traj, KE_traj = KE_traj,
        M_dust_err_max = M_dust_err_max,
        M_gas_err_max = M_gas_err_max,
        M_err_max = M_err_max, Px_err_max = Px_err_max,
        Py_err_max = Py_err_max, KE_err_max = KE_err_max,
        n_negative_jacobian = n_neg_jac,
        gas_gamma_mean = gas_gamma_mean,
        gas_gamma_std = gas_gamma_std,
        gas_gamma_min = gas_gamma_min,
        dust_gamma_max = dust_gamma_max,
        gamma_separation = gamma_separation,
        dust_peak = dust_peak,
        dust_mean = dust_mean_arr,
        peak_over_mean = peak_over_mean,
        snapshot_times = snap_times,
        snapshot_indices = snap_indices,
        snapshots = snapshots,
        t_eddy = t_eddy,
        wall_time_per_step = wall_time_per_step,
        nan_seen = nan_seen,
        proj_stats_total = (
            n_steps = proj_stats.n_steps,
            n_events = proj_stats.n_events,
            n_floor_events = proj_stats.n_floor_events,
            n_offdiag_events = proj_stats.n_offdiag_events,
            total_dE_inj = proj_stats.total_dE_inj,
        ),
        ic = ic,
        params = (level = level, U0 = Float64(U0),
                   ρ0 = ρ0, P0 = P0, ε_dust = Float64(ε_dust),
                   dt = dt_val, n_steps = n_steps, T_end = T_end_val,
                   T_factor = T_factor,
                   project_kind = project_kind,
                   realizability_headroom = realizability_headroom,
                   Mvv_floor = Mvv_floor, pressure_floor = pressure_floor,
                   M_vv_override = M_vv_override, ρ_ref = ρ_ref,
                   snapshots_at = Tuple(snap_times)),
    )
end

"""
    run_D7_dust_traps_sweep(; levels=(4, 5), kwargs...) -> NamedTuple

Run `run_D7_dust_traps` at multiple refinement levels.

Returns:
  • `levels::Tuple`
  • `results::Vector{NamedTuple}` — per-level results
  • `t_eddy::Float64`
  • `peak_over_mean_final::Vector{Float64}` — final peak/mean per level
  • `M_dust_err_final::Vector{Float64}` — final dust mass error per level
"""
function run_D7_dust_traps_sweep(; levels = (4, 5),
                                    U0::Real = 1.0,
                                    T_factor::Real = 2.0,
                                    kwargs...)
    results = NamedTuple[]
    for L in levels
        push!(results, run_D7_dust_traps(; level = L, U0 = U0,
                                           T_factor = T_factor,
                                           kwargs...))
    end
    t_eddy = isempty(results) ? NaN : results[1].t_eddy
    pom_final = [r.peak_over_mean[end] for r in results]
    err_final = [r.M_dust_err_max for r in results]
    return (
        levels = Tuple(levels),
        results = results,
        t_eddy = t_eddy,
        peak_over_mean_final = pom_final,
        M_dust_err_final = err_final,
    )
end

"""
    save_D7_dust_traps_to_h5(sweep, save_path)

Write the mesh-sweep result to HDF5.
"""
function save_D7_dust_traps_to_h5(sweep, save_path::AbstractString)
    HDF5 = if isdefined(Main, :HDF5)
        getfield(Main, :HDF5)
    else
        Base.require(Main, :HDF5)
        getfield(Main, :HDF5)
    end
    mkpath(dirname(save_path))
    HDF5.h5open(save_path, "w") do f
        f["levels"] = collect(sweep.levels)
        f["t_eddy"] = sweep.t_eddy
        f["peak_over_mean_final"] = sweep.peak_over_mean_final
        f["M_dust_err_final"] = sweep.M_dust_err_final
        for (i, L) in enumerate(sweep.levels)
            r = sweep.results[i]
            grp = HDF5.create_group(f, "level_$(L)")
            grp["t"] = r.t
            grp["M_dust_traj"] = r.M_dust_traj
            grp["M_gas_traj"] = r.M_gas_traj
            grp["M_traj"] = r.M_traj
            grp["Px_traj"] = r.Px_traj
            grp["Py_traj"] = r.Py_traj
            grp["KE_traj"] = r.KE_traj
            grp["n_negative_jacobian"] = r.n_negative_jacobian
            grp["gas_gamma_mean"] = r.gas_gamma_mean
            grp["gas_gamma_std"] = r.gas_gamma_std
            grp["dust_gamma_max"] = r.dust_gamma_max
            grp["gamma_separation"] = r.gamma_separation
            grp["dust_peak"] = r.dust_peak
            grp["dust_mean"] = r.dust_mean
            grp["peak_over_mean"] = r.peak_over_mean
            grp["wall_time_per_step"] = r.wall_time_per_step
            for (k, snap) in enumerate(r.snapshots)
                sg = HDF5.create_group(grp, "snapshot_$(k)")
                sg["t"] = snap.t
                sg["x_centres"] = snap.x_centres
                sg["y_centres"] = snap.y_centres
                sg["c_dust"] = snap.c_dust
                sg["u_1"] = snap.u_1
                sg["u_2"] = snap.u_2
                sg["gamma_gas_1"] = snap.γ_gas_1
            end
        end
    end
    return save_path
end

"""
    plot_D7_dust_traps(sweep; save_path) -> save_path

4-panel CairoMakie headline figure showing the D.7 dust-traps result:

  • Panel A: gas density / fluid velocity field at end-time (vortex
             pattern visualisation via |u| heatmap).
  • Panel B: dust concentration heatmap at end-time.
  • Panel C: dust mass conservation `M_dust(t)` over time.
  • Panel D: per-species γ trajectories (gas vs dust).

Falls back to CSV if CairoMakie load fails.
"""
function plot_D7_dust_traps(sweep; save_path::AbstractString)
    try
        CM = if isdefined(Main, :CairoMakie)
            getfield(Main, :CairoMakie)
        else
            Base.require(Main, :CairoMakie)
            getfield(Main, :CairoMakie)
        end
        i_top = lastindex(sweep.results)
        r_top = sweep.results[i_top]
        L_top = sweep.levels[i_top]
        snap_late = r_top.snapshots[end]

        # Pick a regular (n × n) grid from the snapshot. Use unique
        # x and y centres.
        xs_unique = sort!(unique(round.(snap_late.x_centres; digits = 12)))
        ys_unique = sort!(unique(round.(snap_late.y_centres; digits = 12)))
        nx = length(xs_unique); ny = length(ys_unique)

        function reshape_to_grid(arr)
            G = fill(NaN, ny, nx)
            for k in eachindex(snap_late.x_centres)
                xi = searchsortedfirst(xs_unique,
                                        round(snap_late.x_centres[k]; digits = 12))
                yi = searchsortedfirst(ys_unique,
                                        round(snap_late.y_centres[k]; digits = 12))
                if 1 ≤ xi ≤ nx && 1 ≤ yi ≤ ny
                    G[yi, xi] = arr[k]
                end
            end
            return G
        end

        u_mag = sqrt.(snap_late.u_1.^2 .+ snap_late.u_2.^2)
        U_grid = reshape_to_grid(u_mag)
        D_grid = reshape_to_grid(snap_late.c_dust)

        fig = CM.Figure(size = (1100, 850))

        axA = CM.Axis(fig[1, 1];
            title = "A: |u| at t=$(round(snap_late.t; sigdigits=3)) (vortex map, L=$L_top)",
            xlabel = "x_1", ylabel = "x_2")
        CM.heatmap!(axA, xs_unique, ys_unique, transpose(U_grid))

        axB = CM.Axis(fig[1, 2];
            title = "B: dust concentration at t=$(round(snap_late.t; sigdigits=3))",
            xlabel = "x_1", ylabel = "x_2")
        CM.heatmap!(axB, xs_unique, ys_unique, transpose(D_grid))

        axC = CM.Axis(fig[2, 1];
            title = "C: dust mass conservation",
            xlabel = "t", ylabel = "M_dust(t) − M_dust(0)")
        for (i, L) in enumerate(sweep.levels)
            r = sweep.results[i]
            CM.lines!(axC, r.t, r.M_dust_traj .- r.M_dust_traj[1];
                       label = "L=$L")
        end
        CM.axislegend(axC; position = :rt)

        axD = CM.Axis(fig[2, 2];
            title = "D: per-species γ (gas mean vs dust max)",
            xlabel = "t", ylabel = "γ")
        CM.lines!(axD, r_top.t, r_top.gas_gamma_mean; label = "gas γ mean")
        CM.lines!(axD, r_top.t, r_top.dust_gamma_max; label = "dust γ max")
        CM.axislegend(axD; position = :rt)

        mkpath(dirname(save_path))
        CM.save(save_path, fig)
        return save_path
    catch e
        @warn "Plotting D.7 dust-traps figure failed: $(e). Saving CSV instead."
        mkpath(dirname(save_path))
        for (i, L) in enumerate(sweep.levels)
            r = sweep.results[i]
            csv = replace(save_path, ".png" => "_L$(L).csv")
            open(csv, "w") do f
                println(f, "t,M_dust_err,M_gas_err,M_err,Px_err,Py_err,KE,gas_gamma_mean,dust_gamma_max,peak_over_mean,n_neg_jac")
                for k in eachindex(r.t)
                    println(f,
                        "$(r.t[k]),$(r.M_dust_traj[k]-r.M_dust_traj[1])," *
                        "$(r.M_gas_traj[k]-r.M_gas_traj[1]),$(r.M_traj[k]-r.M_traj[1])," *
                        "$(r.Px_traj[k]-r.Px_traj[1]),$(r.Py_traj[k]-r.Py_traj[1])," *
                        "$(r.KE_traj[k]),$(r.gas_gamma_mean[k]),$(r.dust_gamma_max[k])," *
                        "$(r.peak_over_mean[k]),$(r.n_negative_jacobian[k])")
                end
            end
        end
        return save_path
    end
end
