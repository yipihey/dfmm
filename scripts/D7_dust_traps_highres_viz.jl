# D7_dust_traps_highres_viz.jl
#
# High-resolution D.7 dust-trap run + Lagrangian-grid visualisation.
#
# Builds on `experiments/D7_dust_traps.jl::run_D7_dust_traps_per_species`
# (M4 Phase 3 driver, dust accumulates under intermediate-τ drag) but
# captures the per-cell Lagrangian positions `(fields.x_1, fields.x_2)`
# at each snapshot so we can draw the deformed Lagrangian mesh on top
# of the dust concentration.
#
# Usage (from repo root):
#
#   julia --project=. scripts/D7_dust_traps_highres_viz.jl
#
# Defaults: level=6 (64×64 = 4096 cells), τ_drag = 0.1·t_eddy,
# T_factor=1.0 (one eddy turnover). Snapshots at t/T = 0, 0.25,
# 0.5, 1.0.

using Printf

include(joinpath(@__DIR__, "..", "experiments", "D7_dust_traps.jl"))

using HierarchicalGrids: cell_physical_box, enumerate_leaves
using CairoMakie
import CairoMakie as CM

# -----------------------------------------------------------------
# Snapshot extension: capture Lagrangian positions (fields.x_1/x_2)
# alongside Eulerian cell centres and dust concentration. The
# existing `take_snapshot` in D7_dust_traps.jl uses cell-box centres
# for `x_centres`; we want the *true* Lagrangian marker position
# `fields.x_1[ci][1], fields.x_2[ci][1]` which evolves with the flow.
# -----------------------------------------------------------------

"""
    capture_lagrangian_snapshot(ic, t_now) -> NamedTuple

Per-cell snapshot containing:
  • `t::Float64`
  • `x_lagr, y_lagr::Vector{Float64}` — current Lagrangian marker.
  • `x_init, y_init::Vector{Float64}` — Eulerian-frame cell centres
     (initial Lagrangian positions; never change because the frame
     is fixed in the HG substrate).
  • `c_dust::Vector{Float64}` — per-cell dust concentration.
  • `c_dust_remapped::Vector{Float64}` — Eulerian-remap of per-species
     dust positions back to cell concentration (M4 Phase 3 diagnostic).
  • `u_1, u_2::Vector{Float64}` — gas velocity.
  • `i_grid, j_grid::Vector{Int}` — original Cartesian grid indices
     (for connecting Lagrangian neighbours in the deformed-mesh plot).
"""
function capture_lagrangian_snapshot(ic, t_now)
    leaves = ic.leaves
    n = length(leaves)
    x_lagr = Vector{Float64}(undef, n)
    y_lagr = Vector{Float64}(undef, n)
    x_init = Vector{Float64}(undef, n)
    y_init = Vector{Float64}(undef, n)
    c_dust = Vector{Float64}(undef, n)
    u_1    = Vector{Float64}(undef, n)
    u_2    = Vector{Float64}(undef, n)
    k_dust = species_index(ic.tm, :dust)
    for (i, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(ic.frame, ci)
        x_init[i] = 0.5 * (lo_c[1] + hi_c[1])
        y_init[i] = 0.5 * (lo_c[2] + hi_c[2])
        x_lagr[i] = Float64(ic.fields.x_1[ci][1])
        y_lagr[i] = Float64(ic.fields.x_2[ci][1])
        c_dust[i] = ic.tm.tracers[k_dust, ci]
        u_1[i]    = Float64(ic.fields.u_1[ci][1])
        u_2[i]    = Float64(ic.fields.u_2[ci][1])
    end

    # Remapped dust (M4 Phase 3): per-species position → per-cell c.
    lo_box = ic.params.lo
    L1 = ic.params.L1; L2 = ic.params.L2
    hi_box = (Float64(lo_box[1]) + L1, Float64(lo_box[2]) + L2)
    remap = dust_peak_over_mean_remapped(ic.psm, ic.frame, ic.leaves;
                                          lo = lo_box, hi = hi_box)
    c_dust_remapped = Float64.(remap.new_c)

    # Recover original Cartesian (i, j) from the initial cell centres.
    # IC layout: 2^L × 2^L uniform Cartesian on [lo, hi]^2.
    N = Int(round(sqrt(n)))
    @assert N * N == n "Expected square 2^L × 2^L grid; got $n cells"
    Δ = L1 / N
    i_grid = Vector{Int}(undef, n)
    j_grid = Vector{Int}(undef, n)
    for k in 1:n
        i_grid[k] = Int(round((x_init[k] - Float64(lo_box[1]) - 0.5 * Δ) / Δ)) + 1
        j_grid[k] = Int(round((y_init[k] - Float64(lo_box[2]) - 0.5 * Δ) / Δ)) + 1
    end

    return (t = t_now,
            x_lagr = x_lagr, y_lagr = y_lagr,
            x_init = x_init, y_init = y_init,
            c_dust = c_dust, c_dust_remapped = c_dust_remapped,
            u_1 = u_1, u_2 = u_2,
            i_grid = i_grid, j_grid = j_grid,
            N = N, L1 = L1, L2 = L2,
            lo = (Float64(lo_box[1]), Float64(lo_box[2])))
end

# -----------------------------------------------------------------
# High-res driver: replicates run_D7_dust_traps_per_species but with
# our own snapshot capture and adjustable level / snapshot times.
# -----------------------------------------------------------------

function run_highres(; level::Integer = 6,
                       τ_drag_dust::Real = 0.1,
                       T_factor::Real = 1.0,
                       snapshot_fracs = (0.0, 0.25, 0.5, 1.0),
                       U0::Real = 1.0, ρ0::Real = 1.0, P0::Real = 1.0,
                       ε_dust::Real = 0.10,
                       newton_abstol::Real = 1e-13,
                       newton_reltol::Real = 1e-13,
                       newton_maxiters::Int = 80,
                       cfl::Real = 0.125,
                       verbose::Bool = true)
    t_eddy = dust_trap_eddy_time(; U0 = U0)
    T_end = T_factor * t_eddy
    Δx = 1.0 / (2^Int(level))
    dt = Float64(cfl) * Δx / max(Float64(U0), 1e-300)
    n_steps = Int(ceil(T_end / dt))
    dt = T_end / n_steps

    τ_real = Float64(τ_drag_dust) * t_eddy
    ic = tier_d_dust_trap_per_species_ic_full(;
        level = level, U0 = U0, ρ0 = ρ0, P0 = P0, ε_dust = ε_dust,
        τ_drag_per_species = (0.0, τ_real))
    bc = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                              (PERIODIC, PERIODIC)))

    M_vv_override = (1.0, 1.0)
    ρ_ref = 1.0

    snap_steps = Set{Int}()
    for f in snapshot_fracs
        push!(snap_steps, clamp(Int(round(f * n_steps)), 0, n_steps))
    end

    snapshots = NamedTuple[]
    if 0 in snap_steps
        push!(snapshots, capture_lagrangian_snapshot(ic, 0.0))
    end

    proj_stats = ProjectionStats()
    wall_t0 = time()
    if verbose
        @info "Starting high-res D.7 run" level n_cells = ic.leaves |> length n_steps dt T_end t_eddy
    end

    stall_step = 0
    for n in 1:n_steps
        try
            det_step_2d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                                    bc, dt;
                                    M_vv_override = M_vv_override,
                                    ρ_ref = ρ_ref,
                                    abstol = newton_abstol,
                                    reltol = newton_reltol,
                                    maxiters = newton_maxiters,
                                    project_kind = :reanchor,
                                    realizability_headroom = 1.05,
                                    Mvv_floor = 1e-2,
                                    pressure_floor = 1e-8,
                                    proj_stats = proj_stats)
            advect_tracers_HG_2d!(ic.tm, dt)
            drag_relax_per_species!(ic.psm, dt; leaves = ic.leaves)
            advance_positions_per_species!(ic.psm, dt; leaves = ic.leaves)
        catch e
            stall_step = n
            if verbose
                @warn "Newton stall at step $n (t=$(round(n * dt; digits=4))); " *
                      "stopping evolution and using last successful state. " *
                      "Reason: $(sprint(showerror, e))"
            end
            # Take a final snapshot at the last successful time.
            push!(snapshots, capture_lagrangian_snapshot(ic, (n - 1) * dt))
            break
        end

        if n in snap_steps
            push!(snapshots, capture_lagrangian_snapshot(ic, n * dt))
        end

        if verbose && (n % max(1, n_steps ÷ 10) == 0)
            wt = time() - wall_t0
            @info @sprintf("Step %d/%d (t=%.4f, wall=%.1fs)",
                            n, n_steps, n * dt, wt)
        end
    end
    wall = time() - wall_t0
    if verbose
        @info @sprintf("Run complete: %.1fs total, %.3fs/step",
                        wall, wall / max(n_steps, 1))
    end
    return (snapshots = snapshots, ic = ic,
            params = (level = level, n_steps = n_steps, dt = dt,
                       T_end = T_end, t_eddy = t_eddy,
                       τ_drag_real = τ_real, τ_drag_eddy = τ_drag_dust,
                       ε_dust = ε_dust),
            wall = wall)
end

# -----------------------------------------------------------------
# Visualisation: deformed Lagrangian grid + dust + velocity.
# -----------------------------------------------------------------

"""
    plot_lagrangian_grid!(ax, snap; stride=1, alpha=0.5, color=:black, linewidth=0.5)

Overlay the deformed Lagrangian quadrilateral mesh on `ax`. Lines
connect Lagrangian markers that were originally cardinal neighbours
on the 2^L × 2^L Cartesian IC.
"""
function plot_lagrangian_grid!(ax, snap; stride::Integer = 1,
                                  color = :white, alpha::Real = 0.4,
                                  linewidth::Real = 0.4)
    N = snap.N
    L1, L2 = snap.L1, snap.L2
    # Build (i, j) → cell-index lookup.
    idx = fill(0, N, N)
    for k in eachindex(snap.x_lagr)
        idx[snap.i_grid[k], snap.j_grid[k]] = k
    end

    # Periodic-safe polyline: insert NaN whenever consecutive vertices
    # jump by more than L/2 in either axis (i.e. wrap across boundary).
    function safe_push!(xs, ys, x, y)
        if !isempty(xs)
            dx = x - xs[end]; dy = y - ys[end]
            if abs(dx) > 0.5 * L1 || abs(dy) > 0.5 * L2
                push!(xs, NaN); push!(ys, NaN)
            end
        end
        push!(xs, x); push!(ys, y)
    end

    # Horizontal lines: fix j, walk i.
    for j in 1:stride:N
        xs = Float64[]; ys = Float64[]
        for i in 1:N
            k = idx[i, j]
            if k > 0
                safe_push!(xs, ys, snap.x_lagr[k], snap.y_lagr[k])
            end
        end
        if !isempty(xs)
            CM.lines!(ax, xs, ys; color = (color, alpha),
                       linewidth = linewidth)
        end
    end
    # Vertical lines: fix i, walk j.
    for i in 1:stride:N
        xs = Float64[]; ys = Float64[]
        for j in 1:N
            k = idx[i, j]
            if k > 0
                safe_push!(xs, ys, snap.x_lagr[k], snap.y_lagr[k])
            end
        end
        if !isempty(xs)
            CM.lines!(ax, xs, ys; color = (color, alpha),
                       linewidth = linewidth)
        end
    end
    return ax
end

"""
    plot_dust_heatmap!(ax, snap; field=:c_dust, colormap=:inferno)

Reshape a per-cell field (length N²) to an N×N grid using i_grid/j_grid
and draw a heatmap at the original (Eulerian-frame) cell centres.
"""
function plot_dust_heatmap!(ax, snap; field::Symbol = :c_dust,
                              colormap = :inferno,
                              colorrange = nothing)
    N = snap.N
    G = fill(NaN, N, N)
    f = getfield(snap, field)
    for k in eachindex(f)
        G[snap.i_grid[k], snap.j_grid[k]] = f[k]
    end
    xs = snap.lo[1] .+ (snap.L1 / N) .* (0.5 .+ (0:N-1))
    ys = snap.lo[2] .+ (snap.L2 / N) .* (0.5 .+ (0:N-1))
    hm = if colorrange === nothing
        CM.heatmap!(ax, xs, ys, G; colormap = colormap)
    else
        CM.heatmap!(ax, xs, ys, G; colormap = colormap, colorrange = colorrange)
    end
    return hm
end

"""
    plot_velocity_quiver!(ax, snap; stride=4, color=:white, alpha=0.6)

Decimate the velocity field and draw arrows showing the Taylor-Green
vortex pattern.
"""
function plot_velocity_quiver!(ax, snap; stride::Integer = 4,
                                  arrowsize::Real = 8,
                                  color = :white, alpha::Real = 0.7,
                                  lengthscale::Real = 0.06)
    pts = CM.Point2f[]
    dirs = CM.Vec2f[]
    for k in eachindex(snap.u_1)
        if snap.i_grid[k] % stride == 0 && snap.j_grid[k] % stride == 0
            push!(pts, CM.Point2f(snap.x_init[k], snap.y_init[k]))
            push!(dirs, CM.Vec2f(snap.u_1[k] * lengthscale,
                                   snap.u_2[k] * lengthscale))
        end
    end
    CM.arrows!(ax, pts, dirs; color = (color, alpha),
                arrowsize = arrowsize, linewidth = 1.0)
    return ax
end

# -----------------------------------------------------------------
# Figure 1: Lagrangian grid evolution. 2×2 panels, one per snapshot.
# -----------------------------------------------------------------

function figure_lagrangian_grid(result; save_path::AbstractString)
    snaps = result.snapshots
    n_panels = length(snaps)
    nc = min(n_panels, 5)
    nr = Int(ceil(n_panels / nc))
    fig = CM.Figure(size = (380 * nc, 400 * nr + 80))

    dust_min = minimum(minimum.(getfield.(snaps, :c_dust)))
    dust_max = maximum(maximum.(getfield.(snaps, :c_dust)))

    for (p, snap) in enumerate(snaps)
        r = Int(ceil(p / nc)); c = mod1(p, nc)
        ax = CM.Axis(fig[r, c];
            title = @sprintf("t = %.2f  (%.2f t_eddy)", snap.t, snap.t / result.params.t_eddy),
            xlabel = "x₁", ylabel = "x₂", aspect = CM.DataAspect())
        hm = plot_dust_heatmap!(ax, snap; field = :c_dust,
                                  colormap = :inferno,
                                  colorrange = (dust_min, dust_max))
        plot_lagrangian_grid!(ax, snap;
            stride = max(1, snap.N ÷ 32),
            color = :white, alpha = 0.35, linewidth = 0.4)
        if p == n_panels
            CM.Colorbar(fig[r, c + 1], hm; label = "dust c")
        end
    end

    CM.Label(fig[0, :],
              @sprintf("D.7 dust-trap, level=%d (%d cells), τ_drag=%.2f t_eddy",
                        result.params.level,
                        result.snapshots[1].N^2,
                        result.params.τ_drag_eddy),
              fontsize = 18, font = :bold)

    mkpath(dirname(save_path))
    CM.save(save_path, fig)
    return save_path
end

# -----------------------------------------------------------------
# Figure 2: late-time close-up. Dust heatmap + Lagrangian mesh +
# velocity quiver in three side-by-side panels:
#   (A) initial state
#   (B) deformed grid (no fill)
#   (C) final dust + grid
# -----------------------------------------------------------------

function figure_grid_only(result; save_path::AbstractString)
    snaps = result.snapshots
    snap0 = first(snaps)
    snap_end = last(snaps)

    fig = CM.Figure(size = (1500, 540))

    ax1 = CM.Axis(fig[1, 1];
        title = "Initial Lagrangian grid + velocity",
        xlabel = "x₁", ylabel = "x₂", aspect = CM.DataAspect())
    plot_dust_heatmap!(ax1, snap0; field = :c_dust,
                          colormap = :viridis)
    plot_lagrangian_grid!(ax1, snap0;
        stride = max(1, snap0.N ÷ 32),
        color = :white, alpha = 0.6, linewidth = 0.5)
    plot_velocity_quiver!(ax1, snap0; stride = max(1, snap0.N ÷ 12))

    ax2 = CM.Axis(fig[1, 2];
        title = @sprintf("Deformed Lagrangian grid at t=%.2f t_eddy",
                          snap_end.t / result.params.t_eddy),
        xlabel = "x₁", ylabel = "x₂", aspect = CM.DataAspect())
    plot_lagrangian_grid!(ax2, snap_end;
        stride = max(1, snap_end.N ÷ 64),
        color = :black, alpha = 0.6, linewidth = 0.3)

    ax3 = CM.Axis(fig[1, 3];
        title = @sprintf("Final dust + Lagrangian mesh at t=%.2f t_eddy",
                          snap_end.t / result.params.t_eddy),
        xlabel = "x₁", ylabel = "x₂", aspect = CM.DataAspect())
    hm = plot_dust_heatmap!(ax3, snap_end; field = :c_dust,
                              colormap = :inferno)
    plot_lagrangian_grid!(ax3, snap_end;
        stride = max(1, snap_end.N ÷ 32),
        color = :white, alpha = 0.4, linewidth = 0.3)
    CM.Colorbar(fig[1, 4], hm; label = "dust c (Lagrangian)")

    mkpath(dirname(save_path))
    CM.save(save_path, fig)
    return save_path
end

# -----------------------------------------------------------------
# Figure 3: remapped (Eulerian) dust accumulation map. Side-by-side
# comparison of Lagrangian-frame c_dust (unchanged by pure-Lagrangian
# advection) vs Eulerian-remap c_dust (shows accumulation).
# -----------------------------------------------------------------

function figure_dust_comparison(result; save_path::AbstractString)
    snap_end = last(result.snapshots)
    fig = CM.Figure(size = (1200, 500))

    ax1 = CM.Axis(fig[1, 1];
        title = "Lagrangian-frame dust c (advected with parcels)",
        xlabel = "x₁", ylabel = "x₂", aspect = CM.DataAspect())
    hm1 = plot_dust_heatmap!(ax1, snap_end; field = :c_dust,
                                colormap = :inferno)
    CM.Colorbar(fig[1, 2], hm1)

    ax2 = CM.Axis(fig[1, 3];
        title = @sprintf("Eulerian-remapped dust c (drift-induced accumulation), τ=%.2f t_eddy",
                          result.params.τ_drag_eddy),
        xlabel = "x₁", ylabel = "x₂", aspect = CM.DataAspect())
    hm2 = plot_dust_heatmap!(ax2, snap_end; field = :c_dust_remapped,
                                colormap = :inferno)
    CM.Colorbar(fig[1, 4], hm2)

    mkpath(dirname(save_path))
    CM.save(save_path, fig)
    return save_path
end

# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    level_env = get(ENV, "DFMM_LEVEL", "5")
    level = parse(Int, level_env)
    T_factor_env = parse(Float64, get(ENV, "DFMM_TFACTOR", "1.0"))
    U0_env = parse(Float64, get(ENV, "DFMM_U0", "0.3"))
    cfl_env = parse(Float64, get(ENV, "DFMM_CFL", "0.04"))
    # Reduced vortex amplitude U0=0.3 keeps strain mild enough for the
    # Newton solver to survive a full eddy turnover. T_factor=1.0
    # corresponds to L/U0 = 1/0.3 ≈ 3.33 time units.
    # Snapshots dense in time to reveal deformation progression.
    result = run_highres(; level = level, T_factor = T_factor_env,
                            U0 = U0_env,
                            cfl = cfl_env,
                            τ_drag_dust = 0.1,
                            snapshot_fracs = (0.0, 0.25, 0.5, 0.75, 1.0),
                            ε_dust = 0.10,
                            verbose = true)

    out_dir = joinpath(@__DIR__, "..", "reference", "figs")
    p1 = figure_lagrangian_grid(result;
        save_path = joinpath(out_dir, "D7_dust_highres_lagrangian_grid.png"))
    p2 = figure_grid_only(result;
        save_path = joinpath(out_dir, "D7_dust_highres_grid_deformation.png"))
    p3 = figure_dust_comparison(result;
        save_path = joinpath(out_dir, "D7_dust_highres_dust_comparison.png"))

    @info "Saved:" p1 p2 p3
end
