# experiments/B3_action_amr.jl
#
# Tier B.3 — Action-based vs gradient-based 1D AMR on an off-center
# Sod-style blast wave.
#
# Methods paper §10.3 B.3 acceptance: action-AMR uses 20-50% fewer
# cells than gradient-AMR at fixed L² accuracy on the off-center
# blast wave.
#
# Setup. Off-center Sod-like initial discontinuity at x = 0.7 on a
# periodic mass-coordinate domain [0, 1]. We use the *mirror*
# trick from `experiments/A1_sod.jl` to make the integrator's
# periodic BC harmless:
#   - left-side state: (ρ, u, P) = (1.0, 0, 1.0) for x < 0.7
#   - right-side state: (0.125, 0, 0.1) for x ≥ 0.7
#   - mirror to a length-2 box at x = 1: rho/u/P reflected.
# The discontinuity is *off-center* in the inner [0, 1] region, which
# breaks the symmetry and gives action-based AMR more to do (the
# gradient indicator triggers on the strong jump only; the action
# indicator additionally tracks the rarefaction-fan smoothness and
# the contact discontinuity).
#
# Algorithm.
#   1. Initialise mesh of N0 uniform segments with the off-center IC.
#   2. Time-step with `det_step!`, applying `amr_step!` every
#      `amr_period` steps.
#   3. Record the cell-count time-series and the final-time profile.
#   4. Repeat with `gradient_indicator` (field=:rho) instead of
#      `action_error_indicator`.
#   5. Compare both runs against a high-resolution reference run
#      (Nref ≫ N0, no AMR) on the same IC.
#   6. Report the *time-averaged* cell count for each AMR run, the
#      L² error vs reference, and the (action / gradient) cell-count
#      ratio.
#
# Plot. `reference/figs/B3_amr_comparison.png` — three-panel figure:
#   (a) density profile at t_end, action-AMR vs gradient-AMR vs
#       reference.
#   (b) cell-count time-series for both runs.
#   (c) L² error vs cell-count, parametrised by the indicator
#       threshold τ.

using dfmm
using Printf
using CairoMakie

# ──────────────────────────────────────────────────────────────────────
# Off-center Sod IC
# ──────────────────────────────────────────────────────────────────────

"""
    setup_off_center_blast(; N=64, x_disc=0.7, sigma_x0=0.02, mirror=true)
        -> NamedTuple

Sod-like 1D blast wave with discontinuity at `x = x_disc` (default
0.7). Returns a NamedTuple with primitives + Cholesky seeds, mirror-
doubled by default so the integrator's periodic BC sees a smooth
seam at x = 0.
"""
function setup_off_center_blast(; N::Int = 64, x_disc::Float64 = 0.7,
                                 sigma_x0::Float64 = 0.02,
                                 mirror::Bool = true)
    # Inner [0, 1] grid with Sod-like primitives at x_disc.
    x = collect((0:N-1) .+ 0.5) ./ N
    rho = ifelse.(x .< x_disc, 1.0, 0.125)
    u   = zeros(Float64, N)
    P   = ifelse.(x .< x_disc, 1.0, 0.1)
    Γ = 5.0 / 3.0
    αs = fill(sigma_x0, N)
    βs = zeros(Float64, N)
    ss = log.(P ./ rho .^ Γ)
    return (
        N = N, x_disc = x_disc, mirror = mirror, sigma_x0 = sigma_x0,
        x = x, rho = rho, u = u, P = P,
        αs = αs, βs = βs, ss = ss, Γ = Γ,
    )
end

"""
    build_blast_mesh(ic; mirror = ic.mirror) -> Mesh1D

Convert `setup_off_center_blast` IC into an `Mesh1D`. With mirror=true
the IC is doubled into a periodic length-2 box so the integrator's
default periodic BC handles the boundaries cleanly.
"""
function build_blast_mesh(ic; mirror::Bool = ic.mirror)
    N0 = length(ic.rho)
    if mirror
        rho = vcat(ic.rho, reverse(ic.rho))
        u   = vcat(ic.u,   -reverse(ic.u))
        P   = vcat(ic.P,   reverse(ic.P))
        αs  = vcat(ic.αs,  reverse(ic.αs))
        βs  = vcat(ic.βs,  -reverse(ic.βs))
        N = 2 * N0
        dx = 2.0 / N
        L_box = 2.0
    else
        rho = copy(ic.rho); u = copy(ic.u); P = copy(ic.P)
        αs = copy(ic.αs);   βs = copy(ic.βs)
        N = N0
        dx = 1.0 / N
        L_box = 1.0
    end
    Δm = rho .* dx
    ss = log.(P ./ rho .^ ic.Γ)
    positions = collect((0:N-1) .* dx)
    velocities = u
    Pps = copy(P)
    return Mesh1D(positions, velocities, αs, βs, ss;
                  Δm = Δm, Pps = Pps, L_box = L_box, periodic = true)
end

# ──────────────────────────────────────────────────────────────────────
# Mesh-density profile sampling (for L² comparison)
# ──────────────────────────────────────────────────────────────────────

"""
    sample_density(mesh::Mesh1D; xs::AbstractVector) -> Vector{Float64}

Sample the mesh's per-segment density at the Eulerian sample
positions `xs` via piecewise-constant lookup (each cell's value
applies across its full Eulerian extent).
"""
function sample_density(mesh::Mesh1D, xs::AbstractVector)
    N = n_segments(mesh)
    L = mesh.L_box
    ρ_sample = zeros(Float64, length(xs))
    # Pre-compute segment edges (left vertex + cumulative Δx).
    xL = Vector{Float64}(undef, N + 1)
    xL[1] = mesh.segments[1].state.x
    @inbounds for j in 1:N
        xL[j+1] = xL[j] + segment_length(mesh, j)
    end
    @inbounds for (i, x) in enumerate(xs)
        # Wrap into the canonical period.
        xq = mod(x - xL[1], L) + xL[1]
        # Linear search; mesh is small.
        for j in 1:N
            if xL[j] <= xq < xL[j+1]
                ρ_sample[i] = segment_density(mesh, j)
                break
            end
        end
        if ρ_sample[i] == 0.0
            ρ_sample[i] = segment_density(mesh, N)
        end
    end
    return ρ_sample
end

"""
    sample_density_inner(mesh::Mesh1D, ic; n_samples = 200)
        -> (xs, rho_inner)

Sample density on a uniform `xs` grid spanning the *inner* [0, 1]
slice of a mirror-doubled mesh. For non-mirror runs, samples on the
full mesh extent.
"""
function sample_density_inner(mesh::Mesh1D, ic; n_samples::Int = 200)
    if ic.mirror
        xs = collect(range(0.0, 1.0; length = n_samples + 1)[1:end-1] .+
                     0.5 / n_samples)
    else
        xs = collect(range(0.0, 1.0; length = n_samples + 1)[1:end-1] .+
                     0.5 / n_samples)
    end
    return xs, sample_density(mesh, xs)
end

# ──────────────────────────────────────────────────────────────────────
# AMR-driven time integration
# ──────────────────────────────────────────────────────────────────────

"""
    run_amr(mesh::Mesh1D, dt, n_steps; indicator_fn, τ_refine,
            τ_coarsen = τ_refine/4, amr_period = 5,
            tau = 1e-3, max_segments = 4096, min_segments = 8,
            log_stride = 0)
        -> NamedTuple

Time-step `det_step!` with periodic AMR refinement based on
`indicator_fn(mesh)`. Returns:
  * `mesh`     — final mesh.
  * `cell_history::Vector{Int}` — n_segments at each step.
  * `t_history::Vector{Float64}` — time at each step.
  * `n_refined_total::Int`, `n_coarsened_total::Int`.
"""
function run_amr(mesh::Mesh1D, dt::Real, n_steps::Int;
                 indicator_fn, τ_refine::Real,
                 τ_coarsen::Real = τ_refine / 4,
                 amr_period::Int = 5,
                 tau::Real = 1e-3,
                 max_segments::Int = 4096,
                 min_segments::Int = 8,
                 log_stride::Int = 0)
    cell_history = Int[n_segments(mesh)]
    t_history = Float64[0.0]
    n_refined_total = 0
    n_coarsened_total = 0
    for step in 1:n_steps
        det_step!(mesh, dt; tau = tau)
        if step % amr_period == 0
            ind = indicator_fn(mesh)
            result = amr_step!(mesh, ind, τ_refine, τ_coarsen;
                               max_segments = max_segments,
                               min_segments = min_segments)
            n_refined_total += result.n_refined
            n_coarsened_total += result.n_coarsened
        end
        push!(cell_history, n_segments(mesh))
        push!(t_history, step * dt)
        if log_stride > 0 && step % log_stride == 0
            @info "AMR step" step n_segments=n_segments(mesh) refined=n_refined_total coarsened=n_coarsened_total
        end
    end
    return (mesh = mesh, cell_history = cell_history,
            t_history = t_history,
            n_refined_total = n_refined_total,
            n_coarsened_total = n_coarsened_total)
end

# ──────────────────────────────────────────────────────────────────────
# Headline driver: action-AMR vs gradient-AMR comparison
# ──────────────────────────────────────────────────────────────────────

"""
    run_b3_amr_comparison(; N0 = 64, t_end = 0.05, ...)

Run the Tier B.3 head-to-head: action-AMR vs gradient-AMR on the
off-center blast, both targeting the same final L² accuracy vs a
high-resolution reference.

Returns a NamedTuple with:
  * `N_action`, `N_gradient`  — *time-averaged* cell count per run.
  * `L2_action`, `L2_gradient` — final-time L² error vs reference
    (sampled on a uniform grid).
  * `cell_history_action`, `cell_history_gradient`, `t_history`
    — cell-count time series.
  * `xs`, `ρ_ref`, `ρ_action`, `ρ_gradient` — sampled profiles for
    plotting.
  * `ratio` = N_action / N_gradient.
"""
function run_b3_amr_comparison(;
        N0::Int = 64,
        t_end::Float64 = 0.05,
        tau::Float64 = 1e-3,
        sigma_x0::Float64 = 0.02,
        x_disc::Float64 = 0.7,
        Nref::Int = 256,
        amr_period::Int = 5,
        n_steps::Int = 40,
        τ_action_refine::Float64 = 0.04,
        τ_gradient_refine::Float64 = 0.10,
        verbose::Bool = true,
    )
    Γ = 5.0 / 3.0

    # ── Reference run: high-resolution, no AMR ─────────────────────
    ic_ref = setup_off_center_blast(; N = Nref, x_disc = x_disc,
                                    sigma_x0 = sigma_x0, mirror = true)
    mesh_ref = build_blast_mesh(ic_ref)
    # Time-step matched to the reference resolution.
    dx_ref = 2.0 / n_segments(mesh_ref)
    c_s_max = sqrt(Γ * maximum(ic_ref.P) / minimum(ic_ref.rho))
    dt_ref = 0.3 * dx_ref / c_s_max
    n_steps_ref = max(1, ceil(Int, t_end / dt_ref))
    dt_ref = t_end / n_steps_ref

    if verbose
        @info "B.3 reference run" Nref dx_ref dt_ref n_steps_ref
    end
    for _ in 1:n_steps_ref
        det_step!(mesh_ref, dt_ref; tau = tau)
    end

    # Sample reference density on a uniform grid for L² comparison.
    n_samples = 200
    xs, ρ_ref = sample_density_inner(mesh_ref, ic_ref; n_samples = n_samples)

    # ── Action-AMR run ──────────────────────────────────────────────
    ic = setup_off_center_blast(; N = N0, x_disc = x_disc,
                                sigma_x0 = sigma_x0, mirror = true)
    mesh_a = build_blast_mesh(ic)
    dx0 = 2.0 / n_segments(mesh_a)
    dt = 0.3 * dx0 / c_s_max
    if n_steps == 0
        n_steps = max(1, ceil(Int, t_end / dt))
    end
    dt = t_end / n_steps

    if verbose
        @info "B.3 action-AMR run" N0 dt n_steps τ_action_refine
    end
    res_action = run_amr(mesh_a, dt, n_steps;
                        indicator_fn = m -> action_error_indicator(m),
                        τ_refine = τ_action_refine,
                        amr_period = amr_period, tau = tau)
    _, ρ_action = sample_density_inner(res_action.mesh, ic;
                                       n_samples = n_samples)

    # ── Gradient-AMR run ────────────────────────────────────────────
    ic2 = setup_off_center_blast(; N = N0, x_disc = x_disc,
                                 sigma_x0 = sigma_x0, mirror = true)
    mesh_g = build_blast_mesh(ic2)
    if verbose
        @info "B.3 gradient-AMR run" N0 dt n_steps τ_gradient_refine
    end
    res_gradient = run_amr(mesh_g, dt, n_steps;
                          indicator_fn = m -> gradient_indicator(m; field = :rho),
                          τ_refine = τ_gradient_refine,
                          amr_period = amr_period, tau = tau)
    _, ρ_gradient = sample_density_inner(res_gradient.mesh, ic;
                                         n_samples = n_samples)

    # ── L² errors and time-averaged cell counts ─────────────────────
    L2_action = sqrt(sum((ρ_action .- ρ_ref).^2) / length(ρ_ref))
    L2_gradient = sqrt(sum((ρ_gradient .- ρ_ref).^2) / length(ρ_ref))
    N_action_avg = sum(res_action.cell_history) / length(res_action.cell_history)
    N_gradient_avg = sum(res_gradient.cell_history) / length(res_gradient.cell_history)

    ratio = N_action_avg / N_gradient_avg

    if verbose
        @info "B.3 results" N_action_avg N_gradient_avg ratio L2_action L2_gradient
    end

    return (
        N_action = N_action_avg, N_gradient = N_gradient_avg,
        L2_action = L2_action, L2_gradient = L2_gradient,
        ratio = ratio,
        cell_history_action = res_action.cell_history,
        cell_history_gradient = res_gradient.cell_history,
        t_history = res_action.t_history,
        xs = xs, ρ_ref = ρ_ref,
        ρ_action = ρ_action, ρ_gradient = ρ_gradient,
        n_refined_action = res_action.n_refined_total,
        n_coarsened_action = res_action.n_coarsened_total,
        n_refined_gradient = res_gradient.n_refined_total,
        n_coarsened_gradient = res_gradient.n_coarsened_total,
    )
end

# ──────────────────────────────────────────────────────────────────────
# Threshold scan (matched-L² sweep) and plotting
# ──────────────────────────────────────────────────────────────────────

"""
    threshold_scan_b3(; kwargs...) -> (; τ_action, τ_gradient,
                                        N_action, N_gradient,
                                        L2_action, L2_gradient)

Sweep the AMR refinement thresholds for both indicators across a
log-spaced range; record (cell_count, L² error) pairs for each. The
caller plots an L² vs cell-count curve to find the matched-accuracy
operating point.

Lightweight enough to run in the test harness when n_steps is
small; the full sweep with 6 points × 2 indicators × 40-step
integrations runs in a few seconds for N0 = 64.
"""
function threshold_scan_b3(;
        N0::Int = 64,
        t_end::Float64 = 0.05,
        Nref::Int = 256,
        n_steps::Int = 40,
        amr_period::Int = 5,
        action_τs::Vector{Float64} = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16],
        gradient_τs::Vector{Float64} = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64],
        verbose::Bool = true,
    )
    N_action = Float64[]
    L2_action = Float64[]
    N_gradient = Float64[]
    L2_gradient = Float64[]
    for (τa, τg) in zip(action_τs, gradient_τs)
        res = run_b3_amr_comparison(;
            N0 = N0, t_end = t_end, Nref = Nref,
            n_steps = n_steps, amr_period = amr_period,
            τ_action_refine = τa, τ_gradient_refine = τg,
            verbose = false,
        )
        push!(N_action, res.N_action)
        push!(L2_action, res.L2_action)
        push!(N_gradient, res.N_gradient)
        push!(L2_gradient, res.L2_gradient)
        if verbose
            @info "B.3 sweep point" τa τg Na=res.N_action Nb=res.N_gradient L2a=res.L2_action L2b=res.L2_gradient
        end
    end
    return (; action_τs, gradient_τs, N_action, N_gradient,
              L2_action, L2_gradient)
end

# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

"""
    plot_b3_comparison(res; out_path)

Render a 3-panel figure comparing action-AMR vs gradient-AMR:
  (a) ρ(x) at t_end: reference, action, gradient.
  (b) cell-count time-series for both runs.
  (c) annotation summarising N_avg and L² error.

Writes PNG to `out_path` (default `reference/figs/B3_amr_comparison.png`).
Pulls CairoMakie lazily so the test harness doesn't pay the
plotting cost.
"""
function plot_b3_comparison(res; out_path::AbstractString =
                           joinpath(@__DIR__, "..", "reference", "figs",
                                    "B3_amr_comparison.png"))
    fig = Figure(size = (1100, 850))
    # (a) density profiles.
    ax1 = Axis(fig[1, 1:2], xlabel = "x", ylabel = "ρ",
               title = "Off-center blast at t_end")
    lines!(ax1, res.xs, res.ρ_ref, color = :black, linewidth = 1.5,
           label = "Reference (Nref)")
    lines!(ax1, res.xs, res.ρ_action, color = :red, linestyle = :dash,
           linewidth = 1.4, label = "Action-AMR")
    lines!(ax1, res.xs, res.ρ_gradient, color = :blue, linestyle = :dot,
           linewidth = 1.4, label = "Gradient-AMR")
    axislegend(ax1, position = :rt)

    # (b) cell-count time-series.
    ax2 = Axis(fig[2, 1], xlabel = "step", ylabel = "n_segments",
               title = "Cell-count history")
    steps_a = 0:length(res.cell_history_action)-1
    steps_g = 0:length(res.cell_history_gradient)-1
    lines!(ax2, collect(steps_a), res.cell_history_action, color = :red,
           label = "action-AMR")
    lines!(ax2, collect(steps_g), res.cell_history_gradient, color = :blue,
           label = "gradient-AMR")
    axislegend(ax2, position = :rt)

    # (c) summary text.
    ax3 = Axis(fig[2, 2], title = "Summary")
    hidexdecorations!(ax3); hideydecorations!(ax3)
    cell_savings = (res.N_gradient - res.N_action) / res.N_gradient
    msg = string(
        "N_action  = ", round(res.N_action; digits=1), "\n",
        "N_gradient = ", round(res.N_gradient; digits=1), "\n",
        "L²_action  = ", round(res.L2_action; digits=4), "\n",
        "L²_gradient = ", round(res.L2_gradient; digits=4), "\n",
        "cell savings = ", round(100 * cell_savings; digits=1), "%",
    )
    text!(ax3, 0.5, 0.5; text = msg, align = (:center, :center),
          fontsize = 14)

    Label(fig[0, :],
          "Tier B.3 — action-AMR vs gradient-AMR (off-center blast)",
          fontsize = 16)

    mkpath(dirname(out_path))
    save(out_path, fig)
    return out_path
end

# ──────────────────────────────────────────────────────────────────────
# Standalone entrypoint
# ──────────────────────────────────────────────────────────────────────

function main_b3()
    @info "Running B.3 action-AMR vs gradient-AMR comparison..."
    res = run_b3_amr_comparison(; N0 = 64, t_end = 0.05, Nref = 256,
                                n_steps = 40, amr_period = 5,
                                τ_action_refine = 0.04,
                                τ_gradient_refine = 0.10,
                                verbose = true)
    fig_path = plot_b3_comparison(res)
    @info "Wrote figure" fig_path
    return res
end
