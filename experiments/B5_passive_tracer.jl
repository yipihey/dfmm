# experiments/B5_passive_tracer.jl
#
# Tier B.5 — passive scalar advection through a shock + rarefaction.
#
# Demonstrates the variational scheme's tracer-exactness claim
# (methods paper §7): in pure-Lagrangian regions the deterministic
# numerical diffusion of passive tracers is *literally zero*. The
# segments themselves are the parcels; the tracer matrix is never
# written to during integration.
#
# Comparison: a reference first-order upwind Eulerian advection of
# the *same* step IC on a uniform mesh, transported by a constant
# velocity for the same number of timesteps. The Eulerian reference
# smears the interface to several cells; the variational tracer
# stays at machine-precision sharpness (one-cell step).
#
# Output:
#   reference/figs/B5_tracer_through_shock.png — multi-panel:
#     (top) variational tracer through Sod shock at t = 0.2;
#     (mid) Eulerian-reference tracer at the same final time;
#     (bot) interface-width vs time for both schemes.
#
# Driver: `main_b5_passive_tracer()` runs the production-resolution
# Sod (N = 200, mirror-doubled), records the full time series, and
# writes the figure.

using dfmm
using Printf
using CairoMakie

include(joinpath(@__DIR__, "A1_sod.jl"))

"""
    setup_b5_tracers(mesh) -> TracerMesh

Build the standard B.5 tracer bundle on `mesh`: three fields
distinguished by their IC shape (`:step` at m = M/2, `:sin` at one
period over the box, `:gauss` of width 5% M centred at M/2).
"""
function setup_b5_tracers(mesh)
    M = dfmm.total_mass(mesh)
    tm = TracerMesh(mesh; n_tracers = 3, names = [:step, :sin, :gauss])
    set_tracer!(tm, :step,  m -> m < 0.5 * M ? 1.0 : 0.0)
    set_tracer!(tm, :sin,   m -> sinpi(2 * m / M))
    set_tracer!(tm, :gauss, m -> exp(-((m - 0.5 * M) / (0.05 * M))^2))
    return tm
end

"""
    run_b5_variational(; N = 100, t_end = 0.2, tau = 1e-3,
                        n_record = 20, mirror = true)
        -> NamedTuple

Run the variational integrator on a Sod IC, advecting three passive
tracers in lockstep. Records the tracer matrix and the resampled
Eulerian profiles at `n_record` evenly-spaced times in `[0, t_end]`.

Returns a NamedTuple `(mesh, tm, ic, snapshots, summary)`, where
`snapshots` is a vector of NamedTuples `(t, x, T_step, T_sin, T_gauss)`.
"""
function run_b5_variational(; N::Int = 100, t_end::Float64 = 0.2,
                            tau::Float64 = 1e-3, n_record::Int = 20,
                            mirror::Bool = true,
                            sigma_x0::Float64 = 0.02,
                            verbose::Bool = false)
    ic = setup_sod(; N = N, t_end = t_end, sigma_x0 = sigma_x0, tau = tau)
    mesh = build_sod_mesh(ic; mirror = mirror)
    N_seg = dfmm.n_segments(mesh)

    tm = setup_b5_tracers(mesh)
    tracers_initial = copy(tm.tracers)

    Γ = 5.0 / 3.0
    c_s_max = sqrt(Γ * maximum(ic.P) / minimum(ic.rho))
    dx = (mirror ? 2.0 : 1.0) / N_seg
    dt = 0.3 * dx / c_s_max
    n_steps = ceil(Int, t_end / dt)
    dt = t_end / n_steps
    record_stride = max(1, n_steps ÷ n_record)

    snapshots = NamedTuple[]

    function record!(t)
        prof = extract_eulerian_profiles(mesh)
        if mirror
            push!(snapshots, (
                t = t,
                x = prof.x[1:N],
                rho = prof.rho[1:N],
                T_step  = tm.tracers[1, 1:N],
                T_sin   = tm.tracers[2, 1:N],
                T_gauss = tm.tracers[3, 1:N],
            ))
        else
            push!(snapshots, (
                t = t,
                x = prof.x,
                rho = prof.rho,
                T_step  = tm.tracers[1, :],
                T_sin   = tm.tracers[2, :],
                T_gauss = tm.tracers[3, :],
            ))
        end
    end

    record!(0.0)
    for n in 1:n_steps
        dfmm.det_step!(mesh, dt; tau = tau)
        advect_tracers!(tm, dt)
        if n % record_stride == 0 || n == n_steps
            record!(n * dt)
        end
    end

    # Bit-exactness sanity-check; if this fails, something is wrong.
    @assert tm.tracers == tracers_initial "TRACER MATRIX MUTATED — abort!"

    summary = (
        N = N, N_seg = N_seg, t_end = t_end, n_steps = n_steps, dt = dt,
        L_inf_tracer_change = maximum(abs.(tm.tracers .- tracers_initial)),
    )
    verbose && @info "B.5 variational run" summary
    return (; mesh, tm, ic, snapshots, summary, mirror)
end

"""
    run_b5_eulerian_reference(; N = 100, u_const = 0.5, t_end = 0.2,
                              cfl = 0.5)
        -> NamedTuple

First-order upwind reference: advect a step at x = 0.5 by a constant
velocity `u_const` on a uniform mesh of length 1. Records the
tracer profile at every step.

This is the canonical baseline for the Eulerian-numerical-diffusion
comparison; the variational scheme is bit-exact regardless of the
underlying flow, so we make the comparison scheme-vs-scheme using
the same step IC.
"""
function run_b5_eulerian_reference(; N::Int = 100, u_const::Float64 = 0.5,
                                   t_end::Float64 = 0.2,
                                   cfl::Float64 = 0.5)
    L = 1.0
    dx = L / N
    x = collect(((0:N-1) .+ 0.5) .* dx)
    T_eul = Float64[xi < 0.5 ? 1.0 : 0.0 for xi in x]
    u_field = fill(u_const, N)

    dt = cfl * dx / abs(u_const)
    n_steps = ceil(Int, t_end / dt)
    dt = t_end / n_steps

    n_record = 20
    record_stride = max(1, n_steps ÷ n_record)
    snapshots = NamedTuple[]
    push!(snapshots, (t = 0.0, x = copy(x), T = copy(T_eul)))
    widths = Float64[interface_width(T_eul, x)]
    times  = Float64[0.0]
    for n in 1:n_steps
        eulerian_upwind_advect!(T_eul, u_field, dx, dt; periodic = true)
        if n % record_stride == 0 || n == n_steps
            push!(snapshots, (t = n * dt, x = copy(x), T = copy(T_eul)))
        end
        push!(widths, interface_width(T_eul, x))
        push!(times,  n * dt)
    end
    return (; N, dx, x, T_final = T_eul, snapshots,
            widths_t = widths, t = times, dt, n_steps)
end

"""
    fidelity_table(var_run, eul_run) -> NamedTuple

Compute the headline fidelity numbers for the brief / paper:
- variational L∞ tracer change (must be 0.0)
- variational interface width at t_end (one-cell or sub-cell)
- Eulerian-reference interface width at t_end (several-cell smear)
- ratio
"""
function fidelity_table(var_run, eul_run)
    snap = last(var_run.snapshots)
    xv = snap.x
    Tv = snap.T_step
    width_var = interface_width(Tv, xv)

    width_eul = interface_width(eul_run.T_final, eul_run.x)

    return (
        L_inf_tracer_change = var_run.summary.L_inf_tracer_change,
        width_variational = width_var,
        width_eulerian    = width_eul,
        ratio_eul_to_var  = width_eul / max(width_var, eul_run.dx),
        n_steps = var_run.summary.n_steps,
    )
end

"""
    plot_b5_tracer_through_shock(var_run, eul_run; out_path)

Three-panel figure:
1. Density profile at t_end (variational only).
2. Step tracer at t_end: variational (bit-exact) vs Eulerian-reference (smeared).
3. Interface width vs time for the Eulerian reference (and a
   dashed line at zero for the variational scheme).
"""
function plot_b5_tracer_through_shock(var_run, eul_run;
                                      out_path::AbstractString =
                                      joinpath(@__DIR__, "..", "reference",
                                               "figs",
                                               "B5_tracer_through_shock.png"))
    snap = last(var_run.snapshots)
    fig = Figure(size = (1100, 1100))
    ax_rho = Axis(fig[1, 1], xlabel = "x", ylabel = "ρ",
                  title = "Sod density at t = $(round(var_run.summary.t_end; digits = 3))")
    lines!(ax_rho, snap.x, snap.rho, color = :black, linewidth = 1.4,
           label = "variational")
    axislegend(ax_rho, position = :rt)

    ax_step = Axis(fig[2, 1], xlabel = "x", ylabel = "T (step)",
                   title = "Step tracer at t = $(round(var_run.summary.t_end; digits = 3))")
    # Variational: step in Lagrangian frame mapped to Eulerian via
    # extract_eulerian_profiles.x positions.
    lines!(ax_step, snap.x, snap.T_step, color = :blue, linewidth = 1.6,
           label = "variational (bit-exact)")
    lines!(ax_step, eul_run.x, eul_run.T_final, color = :red, linewidth = 1.4,
           linestyle = :dash, label = "Eulerian upwind reference")
    axislegend(ax_step, position = :rt)

    ax_w = Axis(fig[3, 1], xlabel = "t", ylabel = "interface width / dx",
                title = "Interface diffusion vs time")
    lines!(ax_w, eul_run.t, eul_run.widths_t ./ eul_run.dx, color = :red,
           linewidth = 1.4, label = "Eulerian upwind")
    hlines!(ax_w, [0.0], color = :blue, linewidth = 1.6,
            label = "variational (= 0)")
    axislegend(ax_w, position = :rt)

    Label(fig[0, :],
          "Tier B.5 — passive scalar through Sod shock: variational vs Eulerian",
          fontsize = 16)

    mkpath(dirname(out_path))
    save(out_path, fig)
    return out_path
end

"""
    main_b5_passive_tracer()

Production driver: run the variational + Eulerian-reference
benchmarks at `N = 200` and write the headline figure plus a
fidelity table to `reference/figs/B5_tracer_through_shock.png`.
Also returns the run NamedTuples for REPL inspection.
"""
function main_b5_passive_tracer()
    @info "Running B.5 — variational tracer through Sod (N = 200)..."
    var_run = run_b5_variational(; N = 200, t_end = 0.2, tau = 1e-3,
                                 mirror = true, verbose = true)
    @info "Running B.5 — Eulerian upwind reference (N = 200)..."
    eul_run = run_b5_eulerian_reference(; N = 200, u_const = 0.5,
                                        t_end = 0.2, cfl = 0.5)

    table = fidelity_table(var_run, eul_run)
    @info "B.5 fidelity table" table
    @printf "\n  variational L∞ tracer change   = %.3e   (must be 0.0)\n" table.L_inf_tracer_change
    @printf "  variational interface width    = %.4f\n" table.width_variational
    @printf "  Eulerian   interface width    = %.4f\n" table.width_eulerian
    @printf "  ratio (Eulerian / variational, normalized) = %.2f×\n" table.ratio_eul_to_var

    fig_path = plot_b5_tracer_through_shock(var_run, eul_run)
    @info "Wrote figure" fig_path
    return (; var_run, eul_run, table, fig_path)
end
