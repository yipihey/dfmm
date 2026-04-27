# D4_zeldovich_pancake.jl
#
# §10.5 D.4 Zel'dovich pancake collapse driver — M3-6 Phase 2
# (`reference/notes_M3_6_phase2_D4_zeldovich.md`).
#
# The central novel cosmological reference test of the methods paper
# §10.5 D.4. Drives the M3-6 Phase 2 `tier_d_zeldovich_pancake_ic`
# IC through `det_step_2d_berry_HG!` (with M3-6 Phase 1a's strain
# coupling active and M3-6 Phase 1b's 4-component realizability cone
# wired in) at multiple refinement levels. The Zel'dovich IC is
# 1D-symmetric: u_2 = 0 at IC and ∂_2 u_1 = 0 at every face — Phase 1a
# stays inert and the off-diagonal β slots remain zero throughout.
# This is a clean cross-check.
#
# The headline scientific gate is **per-axis γ selectivity**:
#
#   • γ_1 (collapsing axis) develops spatial structure as t → t_cross;
#     drops sharply at the compressive trough.
#   • γ_2 (trivial axis) stays uniform across cells (std → 0 to
#     round-off).
#   • Spatial std ratio std(γ_1) / std(γ_2) → very large at near-caustic.
#
# Caustic time: t_cross = 1 / (A · 2π).
#
# Saves time series + per-axis γ + conservation invariants to HDF5 +
# 4-panel CairoMakie headline plot.
#
# Usage (REPL):
#
#   julia> include("experiments/D4_zeldovich_pancake.jl")
#   julia> result = run_D4_zeldovich_pancake(; level=5, A=0.5)
#
# Or full mesh-refinement battery + headline plot:
#
#   julia> sweep = run_D4_zeldovich_mesh_sweep(; levels=(4, 5))
#   julia> plot_D4_zeldovich_pancake(sweep;
#              save_path="reference/figs/M3_6_phase2_D4_zeldovich.png")

using HierarchicalGrids: HierarchicalMesh, EulerianFrame,
    refine_cells!, enumerate_leaves, FrameBoundaries, REFLECTING, PERIODIC,
    cell_physical_box
using dfmm: tier_d_zeldovich_pancake_ic, allocate_cholesky_2d_fields,
    write_detfield_2d!, read_detfield_2d, DetField2D,
    det_step_2d_berry_HG!, gamma_per_axis_2d_field,
    ProjectionStats
using Statistics: mean, std

"""
    zeldovich_caustic_time(; A=0.5) -> Float64

Analytic caustic time for the Zel'dovich pancake: `t_cross = 1 / (A · 2π)`.
"""
zeldovich_caustic_time(; A::Real = 0.5) = 1.0 / (Float64(A) * 2π)

"""
    zeldovich_velocity_analytic(x; A=0.5, lo=0.0, L=1.0) -> Float64

Analytic Zel'dovich velocity at Eulerian position `x`:

    u_1(x) = -A · 2π · cos(2π (x - lo) / L)

Used as a verification reference for the IC.
"""
function zeldovich_velocity_analytic(x::Real;
                                       A::Real = 0.5,
                                       lo::Real = 0.0,
                                       L::Real = 1.0)
    return -Float64(A) * 2π * cos(2π * (Float64(x) - Float64(lo)) / Float64(L))
end

"""
    pancake_axis_2_uniformity(fields, leaves; M_vv_override, ρ_ref) -> NamedTuple

Per-step diagnostic: compute γ_2 (trivial axis) statistics — mean, std,
range — across all leaves. The headline gate is `std(γ_2) / mean(γ_2)
≤ 0.01` (uniform along the trivial axis).

Returns `(γ2_mean, γ2_std, γ2_range, std_over_mean, γ1_mean, γ1_std, γ1_range,
selectivity_ratio)`.
"""
function pancake_axis_2_uniformity(fields, leaves;
                                     M_vv_override = (1.0, 1.0),
                                     ρ_ref::Real = 1.0)
    γ = gamma_per_axis_2d_field(fields, leaves;
                                  M_vv_override = M_vv_override,
                                  ρ_ref = ρ_ref)
    γ1_arr = γ[1, :]
    γ2_arr = γ[2, :]
    γ1_mean = mean(γ1_arr)
    γ1_std = std(γ1_arr)
    γ1_min = minimum(γ1_arr)
    γ1_max = maximum(γ1_arr)
    γ2_mean = mean(γ2_arr)
    γ2_std = std(γ2_arr)
    γ2_min = minimum(γ2_arr)
    γ2_max = maximum(γ2_arr)
    γ1_range = γ1_max - γ1_min
    γ2_range = γ2_max - γ2_min
    sel = γ2_std > 1e-300 ? γ1_std / γ2_std : Inf
    sm = γ2_mean > 1e-300 ? γ2_std / γ2_mean : NaN
    return (γ1_mean = γ1_mean, γ1_std = γ1_std,
            γ1_min = γ1_min, γ1_max = γ1_max,
            γ1_range = γ1_range,
            γ2_mean = γ2_mean, γ2_std = γ2_std,
            γ2_min = γ2_min, γ2_max = γ2_max,
            γ2_range = γ2_range,
            std_over_mean = sm,
            selectivity_ratio = sel)
end

"""
    negative_jacobian_count_pancake(fields, leaves; M_vv_override, ρ_ref) -> Int

Per-step safety: count cells with γ_a ≤ 1e-12 (Cholesky-cone determinant
near zero or negative). With the M3-6 Phase 1b 4-component projection
this should remain at 0 throughout the run.
"""
function negative_jacobian_count_pancake(fields, leaves;
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
    offdiag_beta_max(fields, leaves) -> Float64

Per-step safety: max |β_12|, |β_21| across all leaves. The Phase 1a
strain coupling stays inert on the Zel'dovich IC (∂_2 u_1 = u_2 = 0
at every face), so this should be `0.0` to round-off throughout.
"""
function offdiag_beta_max(fields, leaves)
    m = 0.0
    @inbounds for ci in leaves
        v = read_detfield_2d(fields, ci)
        m = max(m, abs(v.betas_off[1]), abs(v.betas_off[2]))
    end
    return m
end

"""
    cell_areas(frame, leaves) -> Vector{Float64}

Per-leaf cell area `(hi_x - lo_x)·(hi_y - lo_y)`.
"""
function cell_areas(frame, leaves)
    A = Vector{Float64}(undef, length(leaves))
    @inbounds for (i, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        A[i] = (hi_c[1] - lo_c[1]) * (hi_c[2] - lo_c[2])
    end
    return A
end

"""
    conservation_invariants(fields, leaves, ρ_per_cell, areas) -> NamedTuple

Compute (M, Px, Py, KE) for the cell field. Mass is `Σ ρ·A`,
momentum is `Σ ρ·u·A`, kinetic energy is `Σ 0.5·ρ·|u|²·A`.

Returns `(M, Px, Py, KE)`.
"""
function conservation_invariants(fields, leaves, ρ_per_cell, areas)
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
    run_D4_zeldovich_pancake(; level=5, A=0.5, ρ0=1.0, P0=1e-6,
                              dt=nothing, T_end=nothing,
                              T_factor=0.5, t_factor_caps=true,
                              project_kind=:reanchor,
                              realizability_headroom=1.05,
                              Mvv_floor=1e-2, pressure_floor=1e-8,
                              M_vv_override=(1.0, 1.0),
                              ρ_ref=1.0,
                              snapshots_at=(0.0, 0.5, 0.9),
                              verbose=false) -> NamedTuple

Drive a single mesh-level D.4 Zel'dovich pancake trajectory.

  • Builds the IC via `tier_d_zeldovich_pancake_ic` at the requested
    level (resolution `2^level × 2^level`).
  • Attaches the standard pancake BCs: PERIODIC along axis 1
    (collapsing), REFLECTING along axis 2 (trivial).
  • Runs `det_step_2d_berry_HG!` for `T_end ≈ T_factor · t_cross`
    (default `0.5 · t_cross`, well pre-caustic).
  • Tracks per-step conservation (M, Px, Py, KE), per-axis γ stats,
    n_negative_jacobian, max |β_off|.
  • Saves spatial profile snapshots at `snapshots_at` (fractions of
    `T_end`).

By default `M_vv_override = (1.0, 1.0)` — the cold-limit constant `M_vv`
override that decouples the per-axis γ diagnostic from the EOS branch.
This matches the M3-3d cold-sinusoid driver's convention.

Returns a NamedTuple with:
  • `t::Vector{Float64}` (per-step time)
  • `γ1_max, γ1_min, γ1_std::Vector{Float64}`
  • `γ2_max, γ2_min, γ2_std::Vector{Float64}`
  • `selectivity_ratio::Vector{Float64}` — `std(γ_1) / std(γ_2)`
  • `n_negative_jacobian::Vector{Int}`
  • `max_abs_beta_off::Vector{Float64}` (Phase 1a inertness check)
  • `M_traj, Px_traj, Py_traj, KE_traj::Vector{Float64}`
  • `M_err_max, Px_err_max, Py_err_max, KE_err_max::Float64`
  • `snapshot_times, snapshot_indices::Vector` (for plotting)
  • `snapshots::Vector{NamedTuple}` — per-snapshot profile data
    `(t, γ1, γ2, x_dev, x_centers, y_centers)` for plotting
  • `t_cross::Float64`
  • `wall_time_per_step::Float64`
  • `nan_seen::Bool`
  • `params::NamedTuple`
"""
function run_D4_zeldovich_pancake(; level::Integer = 5,
                                    A::Real = 0.5,
                                    ρ0::Real = 1.0,
                                    P0::Real = 1e-6,
                                    dt::Union{Real, Nothing} = nothing,
                                    T_end::Union{Real, Nothing} = nothing,
                                    T_factor::Real = 0.5,
                                    project_kind::Symbol = :reanchor,
                                    realizability_headroom::Real = 1.05,
                                    Mvv_floor::Real = 1e-2,
                                    pressure_floor::Real = 1e-8,
                                    M_vv_override = (1.0, 1.0),
                                    ρ_ref::Real = 1.0,
                                    snapshots_at = (0.0, 0.5, 0.9),
                                    verbose::Bool = false)
    t_cross = zeldovich_caustic_time(; A = A)
    T_end_val = T_end === nothing ? Float64(T_factor) * t_cross : Float64(T_end)
    if dt === nothing
        # Mesh-scaled dt. The Zel'dovich velocity peak is `A·2π`; CFL ~
        # u_max·dt/Δx → dt ~ Δx/(A·2π). Use 0.25 fraction.
        Δx = 1.0 / (2^Int(level))
        dt_val = 0.25 * Δx / (Float64(A) * 2π)
        # Cap so we always get at least 30 samples.
        dt_val = min(dt_val, T_end_val / 30.0)
    else
        dt_val = Float64(dt)
    end
    n_steps = Int(ceil(T_end_val / dt_val))
    dt_val = T_end_val / n_steps

    # Build IC.
    ic = tier_d_zeldovich_pancake_ic(; level = level, A = A,
                                       ρ0 = ρ0, P0 = P0)
    bc_zel = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                  (REFLECTING, REFLECTING)))
    areas = cell_areas(ic.frame, ic.leaves)

    # Pre-allocate trajectory arrays.
    N = n_steps + 1
    t = zeros(Float64, N)
    γ1_max = zeros(Float64, N)
    γ1_min = zeros(Float64, N)
    γ1_std_arr = zeros(Float64, N)
    γ2_max = zeros(Float64, N)
    γ2_min = zeros(Float64, N)
    γ2_std_arr = zeros(Float64, N)
    γ1_mean_arr = zeros(Float64, N)
    γ2_mean_arr = zeros(Float64, N)
    selectivity_ratio = zeros(Float64, N)
    n_negative_jacobian = zeros(Int, N)
    max_abs_beta_off = zeros(Float64, N)
    M_traj = zeros(Float64, N)
    Px_traj = zeros(Float64, N)
    Py_traj = zeros(Float64, N)
    KE_traj = zeros(Float64, N)

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

    # Helper: take a spatial snapshot of γ_1, γ_2, and x-deviation per cell.
    function take_snapshot(t_now)
        γ_now = gamma_per_axis_2d_field(ic.fields, ic.leaves;
                                          M_vv_override = M_vv_override,
                                          ρ_ref = ρ_ref)
        γ1_arr = γ_now[1, :]
        γ2_arr = γ_now[2, :]
        x_centers = Vector{Float64}(undef, length(ic.leaves))
        y_centers = Vector{Float64}(undef, length(ic.leaves))
        x_dev = Vector{Float64}(undef, length(ic.leaves))
        for (i, ci) in enumerate(ic.leaves)
            lo_c, hi_c = cell_physical_box(ic.frame, ci)
            cx = 0.5 * (lo_c[1] + hi_c[1])
            cy = 0.5 * (lo_c[2] + hi_c[2])
            x_centers[i] = cx
            y_centers[i] = cy
            # Deviation of stored x_1 from cell center (the Lagrangian
            # position drift). The IC stores x_1 = cell-center; under
            # the Newton evolution `x_a` advects ballistically.
            x_dev[i] = Float64(ic.fields.x_1[ci][1]) - cx
        end
        return (t = t_now, γ1 = γ1_arr, γ2 = γ2_arr,
                x_dev = x_dev,
                x_centers = x_centers, y_centers = y_centers)
    end

    # Initial diagnostics + IC snapshot bookkeeping.
    diag0 = pancake_axis_2_uniformity(ic.fields, ic.leaves;
                                        M_vv_override = M_vv_override,
                                        ρ_ref = ρ_ref)
    γ1_max[1] = diag0.γ1_max; γ1_min[1] = diag0.γ1_min
    γ1_std_arr[1] = diag0.γ1_std; γ1_mean_arr[1] = diag0.γ1_mean
    γ2_max[1] = diag0.γ2_max; γ2_min[1] = diag0.γ2_min
    γ2_std_arr[1] = diag0.γ2_std; γ2_mean_arr[1] = diag0.γ2_mean
    selectivity_ratio[1] = diag0.selectivity_ratio
    n_negative_jacobian[1] = negative_jacobian_count_pancake(ic.fields, ic.leaves;
                                                               M_vv_override = M_vv_override,
                                                               ρ_ref = ρ_ref)
    max_abs_beta_off[1] = offdiag_beta_max(ic.fields, ic.leaves)
    cons0 = conservation_invariants(ic.fields, ic.leaves, ic.ρ_per_cell, areas)
    M_traj[1] = cons0.M; Px_traj[1] = cons0.Px
    Py_traj[1] = cons0.Py; KE_traj[1] = cons0.KE
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
                                    bc_zel, dt_val;
                                    M_vv_override = M_vv_override,
                                    ρ_ref = ρ_ref,
                                    project_kind = project_kind,
                                    realizability_headroom = realizability_headroom,
                                    Mvv_floor = Mvv_floor,
                                    pressure_floor = pressure_floor,
                                    proj_stats = proj_stats)
        catch e
            if verbose
                @warn "Newton solve failed at step $n: $e"
            end
            nan_seen = true
            break
        end

        t[n + 1] = n * dt_val
        d = pancake_axis_2_uniformity(ic.fields, ic.leaves;
                                        M_vv_override = M_vv_override,
                                        ρ_ref = ρ_ref)
        γ1_max[n + 1] = d.γ1_max; γ1_min[n + 1] = d.γ1_min
        γ1_std_arr[n + 1] = d.γ1_std; γ1_mean_arr[n + 1] = d.γ1_mean
        γ2_max[n + 1] = d.γ2_max; γ2_min[n + 1] = d.γ2_min
        γ2_std_arr[n + 1] = d.γ2_std; γ2_mean_arr[n + 1] = d.γ2_mean
        selectivity_ratio[n + 1] = d.selectivity_ratio
        n_negative_jacobian[n + 1] = negative_jacobian_count_pancake(ic.fields, ic.leaves;
                                                                       M_vv_override = M_vv_override,
                                                                       ρ_ref = ρ_ref)
        max_abs_beta_off[n + 1] = offdiag_beta_max(ic.fields, ic.leaves)
        cons = conservation_invariants(ic.fields, ic.leaves, ic.ρ_per_cell, areas)
        M_traj[n + 1] = cons.M; Px_traj[n + 1] = cons.Px
        Py_traj[n + 1] = cons.Py; KE_traj[n + 1] = cons.KE

        # Snapshot capture if at desired index.
        for (k, idx) in enumerate(snap_indices)
            if idx == n + 1 && !snap_taken[k]
                snapshots[k] = take_snapshot(t[n + 1])
                snap_taken[k] = true
            end
        end

        if !isfinite(d.γ1_max) || !isfinite(d.γ2_max)
            nan_seen = true
            break
        end

        if verbose && (n % max(1, n_steps ÷ 10) == 0)
            @info "Step $n / $n_steps: t=$(round(t[n+1]; digits=4))," *
                  " γ1_min=$(round(d.γ1_min; sigdigits=4))," *
                  " γ2_std=$(round(d.γ2_std; sigdigits=3))," *
                  " sel=$(round(d.selectivity_ratio; sigdigits=4))"
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
    M_err_max = maximum(abs.(M_traj .- M_traj[1]))
    Px_err_max = maximum(abs.(Px_traj .- Px_traj[1]))
    Py_err_max = maximum(abs.(Py_traj .- Py_traj[1]))
    KE_err_max = maximum(abs.(KE_traj .- KE_traj[1]))

    return (
        t = t,
        γ1_max = γ1_max, γ1_min = γ1_min, γ1_std = γ1_std_arr,
        γ1_mean = γ1_mean_arr,
        γ2_max = γ2_max, γ2_min = γ2_min, γ2_std = γ2_std_arr,
        γ2_mean = γ2_mean_arr,
        selectivity_ratio = selectivity_ratio,
        n_negative_jacobian = n_negative_jacobian,
        max_abs_beta_off = max_abs_beta_off,
        M_traj = M_traj, Px_traj = Px_traj,
        Py_traj = Py_traj, KE_traj = KE_traj,
        M_err_max = M_err_max, Px_err_max = Px_err_max,
        Py_err_max = Py_err_max, KE_err_max = KE_err_max,
        snapshot_times = snap_times,
        snapshot_indices = snap_indices,
        snapshots = snapshots,
        t_cross = t_cross,
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
        params = (level = level, A = Float64(A),
                   ρ0 = ρ0, P0 = P0,
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
    run_D4_zeldovich_mesh_sweep(; levels=(4, 5), kwargs...) -> NamedTuple

Run `run_D4_zeldovich_pancake` at multiple refinement levels.

Returns:
  • `levels::Tuple`
  • `results::Vector{NamedTuple}` — per-level results
  • `t_cross::Float64`
  • `selectivity_final::Vector{Float64}` — final-step selectivity per level
  • `γ1_dynamic_range::Vector{Float64}` — γ1_max/γ1_min per level at end
"""
function run_D4_zeldovich_mesh_sweep(; levels = (4, 5),
                                       A::Real = 0.5,
                                       T_factor::Real = 0.5,
                                       kwargs...)
    results = NamedTuple[]
    for L in levels
        push!(results, run_D4_zeldovich_pancake(; level = L, A = A,
                                                  T_factor = T_factor,
                                                  kwargs...))
    end
    t_cross = isempty(results) ? NaN : results[1].t_cross
    sel_final = [r.selectivity_ratio[end] for r in results]
    dyn_range = [r.γ1_max[end] / max(r.γ1_min[end], 1e-300) for r in results]
    return (
        levels = Tuple(levels),
        results = results,
        t_cross = t_cross,
        selectivity_final = sel_final,
        γ1_dynamic_range = dyn_range,
    )
end

"""
    save_D4_zeldovich_to_h5(sweep, save_path)

Write the mesh-sweep result to HDF5.
"""
function save_D4_zeldovich_to_h5(sweep, save_path::AbstractString)
    HDF5 = if isdefined(Main, :HDF5)
        getfield(Main, :HDF5)
    else
        Base.require(Main, :HDF5)
        getfield(Main, :HDF5)
    end
    mkpath(dirname(save_path))
    HDF5.h5open(save_path, "w") do f
        f["levels"] = collect(sweep.levels)
        f["t_cross"] = sweep.t_cross
        f["selectivity_final"] = sweep.selectivity_final
        f["gamma1_dynamic_range"] = sweep.γ1_dynamic_range
        for (i, L) in enumerate(sweep.levels)
            r = sweep.results[i]
            grp = HDF5.create_group(f, "level_$(L)")
            grp["t"] = r.t
            grp["gamma1_max"] = r.γ1_max
            grp["gamma1_min"] = r.γ1_min
            grp["gamma1_std"] = r.γ1_std
            grp["gamma2_max"] = r.γ2_max
            grp["gamma2_min"] = r.γ2_min
            grp["gamma2_std"] = r.γ2_std
            grp["selectivity_ratio"] = r.selectivity_ratio
            grp["n_negative_jacobian"] = r.n_negative_jacobian
            grp["max_abs_beta_off"] = r.max_abs_beta_off
            grp["M_traj"] = r.M_traj
            grp["Px_traj"] = r.Px_traj
            grp["Py_traj"] = r.Py_traj
            grp["KE_traj"] = r.KE_traj
            grp["M_err_max"] = r.M_err_max
            grp["Px_err_max"] = r.Px_err_max
            grp["Py_err_max"] = r.Py_err_max
            grp["KE_err_max"] = r.KE_err_max
            grp["wall_time_per_step"] = r.wall_time_per_step
            for (k, snap) in enumerate(r.snapshots)
                sg = HDF5.create_group(grp, "snapshot_$(k)")
                sg["t"] = snap.t
                sg["gamma1"] = snap.γ1
                sg["gamma2"] = snap.γ2
                sg["x_dev"] = snap.x_dev
                sg["x_centers"] = snap.x_centers
                sg["y_centers"] = snap.y_centers
            end
        end
    end
    return save_path
end

"""
    plot_D4_zeldovich_pancake(sweep; save_path) -> save_path

4-panel CairoMakie headline plot showing per-axis γ selectivity:

  • Panel A: γ_1 spatial profile at 3 time slices (along axis-1 cells
             on a y=const slice, sorted by x).
  • Panel B: γ_2 spatial profile at the same time slices (uniform).
  • Panel C: |x_1(t) - cell_center| spatial profile at the same times.
  • Panel D: log10(γ_1) along x at near-caustic (collapse signature).

Falls back to CSV if CairoMakie load fails.
"""
function plot_D4_zeldovich_pancake(sweep; save_path::AbstractString)
    try
        CM = if isdefined(Main, :CairoMakie)
            getfield(Main, :CairoMakie)
        else
            Base.require(Main, :CairoMakie)
            getfield(Main, :CairoMakie)
        end
        # Plot the highest-level result to maximise spatial detail.
        i_top = lastindex(sweep.results)
        r_top = sweep.results[i_top]
        L_top = sweep.levels[i_top]
        # For each snapshot, pick the y=const slice closest to the
        # mesh's mid-y for clarity.
        function slice_along_x(snap)
            y_unique = sort!(unique(round.(snap.y_centers; digits = 12)))
            y_mid = y_unique[max(1, length(y_unique) ÷ 2)]
            keep = findall(y -> isapprox(y, y_mid; atol = 1e-12),
                            snap.y_centers)
            ord = sortperm(snap.x_centers[keep])
            kk = keep[ord]
            return snap.x_centers[kk], snap.γ1[kk], snap.γ2[kk], snap.x_dev[kk]
        end

        fig = CM.Figure(size = (1100, 850))

        axA = CM.Axis(fig[1, 1];
            title = "A: γ_1 spatial profile (collapsing axis)",
            xlabel = "x_1", ylabel = "γ_1")
        for (k, snap) in enumerate(r_top.snapshots)
            xs, g1, _, _ = slice_along_x(snap)
            CM.lines!(axA, xs, g1;
                      label = "t = $(round(snap.t; sigdigits=3))")
        end
        CM.axislegend(axA; position = :rb)

        axB = CM.Axis(fig[1, 2];
            title = "B: γ_2 spatial profile (trivial axis)",
            xlabel = "x_1", ylabel = "γ_2")
        for (k, snap) in enumerate(r_top.snapshots)
            xs, _, g2, _ = slice_along_x(snap)
            CM.lines!(axB, xs, g2;
                      label = "t = $(round(snap.t; sigdigits=3))")
        end
        CM.axislegend(axB; position = :rb)

        axC = CM.Axis(fig[2, 1];
            title = "C: |x_1(t) − cell center| (Lagrangian deviation)",
            xlabel = "x_1", ylabel = "|x_1 − x_center|")
        for snap in r_top.snapshots
            xs, _, _, xd = slice_along_x(snap)
            CM.lines!(axC, xs, abs.(xd);
                      label = "t = $(round(snap.t; sigdigits=3))")
        end
        CM.axislegend(axC; position = :rt)

        # Panel D: log10(γ_1) at the latest snapshot (near-caustic).
        snap_late = r_top.snapshots[end]
        xs_l, g1_l, _, _ = slice_along_x(snap_late)
        axD = CM.Axis(fig[2, 2];
            title = "D: log10(γ_1) at t=$(round(snap_late.t; sigdigits=3)) (near-caustic)",
            xlabel = "x_1", ylabel = "log10(γ_1)")
        CM.lines!(axD, xs_l, log10.(max.(g1_l, 1e-300)))

        mkpath(dirname(save_path))
        CM.save(save_path, fig)
        return save_path
    catch e
        @warn "Plotting D.4 Zel'dovich figure failed: $(e). Saving CSV instead."
        mkpath(dirname(save_path))
        for (i, L) in enumerate(sweep.levels)
            r = sweep.results[i]
            csv = replace(save_path, ".png" => "_L$(L).csv")
            open(csv, "w") do f
                println(f, "t,γ1_min,γ1_max,γ1_std,γ2_min,γ2_max,γ2_std," *
                            "selectivity,n_neg_jac,max_abs_β_off,KE")
                for k in eachindex(r.t)
                    println(f, "$(r.t[k]),$(r.γ1_min[k]),$(r.γ1_max[k])," *
                              "$(r.γ1_std[k]),$(r.γ2_min[k]),$(r.γ2_max[k])," *
                              "$(r.γ2_std[k]),$(r.selectivity_ratio[k])," *
                              "$(r.n_negative_jacobian[k])," *
                              "$(r.max_abs_beta_off[k]),$(r.KE_traj[k])")
                end
            end
        end
        return save_path
    end
end
