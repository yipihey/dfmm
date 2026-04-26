# experiments/M2_3_long_time_stochastic.jl
#
# M2-3 long-time stochastic-realizability stability driver. Closes
# Milestone-1 Open #4 — the wave-pool blow-up at ~950 steps under
# production calibration when a cell crosses `M_vv < β²` and Newton
# fails.
#
# Acceptance gates (see HANDOFF, MILESTONE_1_STATUS Open #4, M2-3 brief):
#   1. Wave-pool runs 10⁴+ steps without Newton failure under
#      `load_noise_model()` defaults.
#   2. Energy drift remains bounded (no new secular leak introduced
#      by the projection beyond the documented Phase-4 t¹ trend).
#   3. Mass conservation exact across projection events.
#   4. Projection-event rate documented (small at low calibration; at
#      production calibration the projection becomes the dominant
#      closure for cells driven into extreme compression — that is
#      the trade-off for stability, not a defect).
#   5. Bit-equality with M1 Phase 8 when `project_kind = :none`
#      (verified by `test/test_phase_M2_3_realizability.jl §3`).
#
# Output:
#   * `reference/figs/M2_3_stability_comparison.png` — wave-pool
#     E, M_vv-min, ρ-max trace with vs without projection.
#   * `reference/figs/M2_3_stability_comparison.h5` — raw history.
#   * Console summary including projection-event statistics.
#
# Wall time: ~3-5 min on a 128-segment wave-pool run with 12 000 steps.
#
# References:
#   * `reference/notes_M2_3_realizability.md` — design + variant notes.
#   * `reference/notes_phase8_stochastic_injection.md` §7 — root-cause
#     diagnosis of the original instability.
#   * `reference/MILESTONE_1_STATUS.md` Open #4.

using dfmm
using HDF5
using Printf
using Random: MersenneTwister
using CairoMakie

# -----------------------------------------------------------------------------
# Wave-pool driver (with optional projection)
# -----------------------------------------------------------------------------

"""
    run_long_time_wavepool(; project_kind=:reanchor, N=128,
                            n_steps=12000, dt=5e-4, seed=2026,
                            tau=1e-2, save_every=100,
                            verbose=true)

Run the production-calibrated wave-pool for `n_steps` steps with
`project_kind ∈ (:none, :reanchor)`. Returns a NamedTuple with the
running diagnostics.
"""
function run_long_time_wavepool(;
    project_kind::Symbol = :reanchor,
    N::Int = 128,
    n_steps::Int = 12000,
    dt::Float64 = 5e-4,
    seed::Int = 2026,
    tau::Float64 = 1e-2,
    save_every::Int = 100,
    verbose::Bool = true,
    Mvv_floor::Float64 = 1e-2,
    realizability_headroom::Float64 = 1.05,
)
    setup = setup_kmles_wavepool(N = N, t_end = n_steps * dt,
                                 u0 = 0.3, P0 = 1.0, K_max = 8,
                                 seed = seed, tau = tau,
                                 n_snaps = max(2, n_steps ÷ 100))
    L = 1.0
    Δx = L / N
    Δm_vec = fill(setup.params.rho0 * Δx, N)
    positions = collect(0.0:Δx:(L - Δx))
    s0 = log.(setup.P ./ setup.rho) .+ (2.0/3.0) .* log.(1.0 ./ setup.rho)
    mesh = Mesh1D(positions, setup.u, setup.alpha_init, setup.beta_init, s0;
                  Δm = Δm_vec, Pps = setup.Pp, L_box = L, periodic = true)

    nm = load_noise_model()
    params = NoiseInjectionParams(
        C_A = nm.C_A, C_B = nm.C_B,
        λ = 6.667, θ_factor = 0.15,
        ke_budget_fraction = 0.25, ell_corr = 2.0,
        pressure_floor = 1e-8,
        project_kind = project_kind,
        realizability_headroom = realizability_headroom,
        Mvv_floor = Mvv_floor,
    )

    rng = MersenneTwister(seed)
    diag = InjectionDiagnostics(N)
    proj_stats = ProjectionStats()

    t_history = Float64[]
    E_history = Float64[]
    M_history = Float64[]
    p_history = Float64[]
    Mvv_min_history = Float64[]
    Mvv_max_history = Float64[]
    rho_max_history = Float64[]
    n_events_history = Int[]

    n_done = 0
    M0 = total_mass(mesh)
    p0 = total_momentum(mesh)
    E0 = total_energy(mesh)

    if verbose
        @printf("[M2-3] project_kind=%s, N=%d, dt=%.1e, n_steps=%d\n",
                project_kind, N, dt, n_steps)
        @printf("[M2-3] initial E=%.4e, M=%.4e, p=%.4e\n", E0, M0, p0)
    end

    for n in 1:n_steps
        try
            # Pre-Newton projection.
            realizability_project!(mesh; kind = params.project_kind,
                                   headroom = params.realizability_headroom,
                                   Mvv_floor = params.Mvv_floor,
                                   pressure_floor = params.pressure_floor,
                                   stats = proj_stats)
            det_step!(mesh, dt; tau = tau,
                      q_kind = :vNR_linear_quadratic,
                      c_q_quad = 1.0, c_q_lin = 0.5)
            inject_vg_noise!(mesh, dt; params = params, rng = rng,
                             diag = diag, proj_stats = proj_stats)
            if !all(isfinite, diag.divu) || !isfinite(total_energy(mesh))
                if verbose
                    @printf("[M2-3] non-finite mesh at step %d → stopping\n", n)
                end
                break
            end
            n_done = n
            if n % save_every == 0 || n == 1
                push!(t_history, n * dt)
                push!(E_history, total_energy(mesh))
                push!(M_history, total_mass(mesh))
                push!(p_history, total_momentum(mesh))
                Mvv_min = Inf
                Mvv_max = -Inf
                rho_max = -Inf
                for j in 1:N
                    ρ = segment_density(mesh, j); J = 1/ρ
                    M = Mvv(J, mesh.segments[j].state.s)
                    if isfinite(M)
                        Mvv_min = min(Mvv_min, M)
                        Mvv_max = max(Mvv_max, M)
                    end
                    rho_max = max(rho_max, ρ)
                end
                push!(Mvv_min_history, Mvv_min)
                push!(Mvv_max_history, Mvv_max)
                push!(rho_max_history, rho_max)
                push!(n_events_history, proj_stats.n_events)
                if verbose && n % (save_every * 10) == 0
                    @printf("[M2-3] step %5d: E=%.4e ΔE_rel=%.2e Mvv_min=%.2e Mvv_max=%.2e rho_max=%.2e events=%d\n",
                            n, E_history[end], abs((E_history[end]-E0)/E0),
                            Mvv_min, Mvv_max, rho_max, proj_stats.n_events)
                end
            end
        catch err
            if verbose
                @printf("[M2-3] caught at step %d: %s → stopping\n",
                        n, sprint(showerror, err))
            end
            break
        end
    end

    if verbose
        @printf("[M2-3] reached %d / %d steps\n", n_done, n_steps)
        @printf("[M2-3] mass drift  = %.4e (exact = 0)\n",
                abs(total_mass(mesh) - M0))
        @printf("[M2-3] mom drift   = %.4e\n",
                abs(total_momentum(mesh) - p0))
        @printf("[M2-3] energy rel-drift = %.4e\n",
                abs((total_energy(mesh) - E0) / max(abs(E0), 1e-30)))
        @printf("[M2-3] total projection events = %d  (rate = %.4e / cell-step)\n",
                proj_stats.n_events,
                n_done > 0 ? proj_stats.n_events / (n_done * N) : 0.0)
        @printf("[M2-3] floor events            = %d\n", proj_stats.n_floor_events)
        @printf("[M2-3] silent dE_inj total     = %.4e\n", proj_stats.total_dE_inj)
    end

    return (
        project_kind = project_kind,
        n_done = n_done,
        n_steps_target = n_steps,
        t_history = t_history,
        E_history = E_history,
        M_history = M_history,
        p_history = p_history,
        Mvv_min_history = Mvv_min_history,
        Mvv_max_history = Mvv_max_history,
        rho_max_history = rho_max_history,
        n_events_history = n_events_history,
        proj_stats = proj_stats,
        E0 = E0, M0 = M0, p0 = p0,
        params = params,
    )
end

# -----------------------------------------------------------------------------
# Plot baseline-vs-fixed instability comparison
# -----------------------------------------------------------------------------

"""
    plot_stability_comparison(result_baseline, result_fixed; outpath)

Build the multi-panel headline figure: energy, M_vv-min, ρ_max traces
with vs without projection. Panels:
  * Top-left: total energy E(t) for both runs. Dashed vertical line
    marks the baseline failure step.
  * Top-right: M_vv-min(t) traces — the realizability-margin proxy.
  * Bottom-left: ρ_max(t) — the compression-cascade proxy.
  * Bottom-right: cumulative projection-event count vs time for the
    fixed run.
"""
function plot_stability_comparison(result_baseline, result_fixed;
                                   outpath::AbstractString)
    fig = Figure(size = (1200, 800))

    ax_E = Axis(fig[1, 1];
                title = "Total energy E(t)",
                xlabel = "t (sim units)",
                ylabel = "E")
    if !isempty(result_baseline.t_history)
        lab = string(":none baseline (n_done=", result_baseline.n_done, ")")
        lines!(ax_E, result_baseline.t_history, result_baseline.E_history;
               color = :crimson, linewidth = 2, label = lab)
    end
    if !isempty(result_fixed.t_history)
        lab = string(":reanchor fixed (n_done=", result_fixed.n_done, ")")
        lines!(ax_E, result_fixed.t_history, result_fixed.E_history;
               color = :steelblue, linewidth = 2, label = lab)
    end
    if result_baseline.n_done < result_baseline.n_steps_target
        t_fail = result_baseline.n_done * (result_fixed.t_history[end] /
                                           max(result_fixed.n_done, 1))
        vlines!(ax_E, [t_fail]; color = :crimson, linestyle = :dash,
                label = "baseline blow-up")
    end
    axislegend(ax_E; position = :rb)

    ax_M = Axis(fig[1, 2];
                title = "M_vv min over the mesh",
                xlabel = "t",
                ylabel = "M_vv_min",
                yscale = log10)
    if !isempty(result_baseline.t_history)
        Mvv = max.(result_baseline.Mvv_min_history, 1e-12)
        lines!(ax_M, result_baseline.t_history, Mvv;
               color = :crimson, linewidth = 2, label = ":none baseline")
    end
    if !isempty(result_fixed.t_history)
        Mvv = max.(result_fixed.Mvv_min_history, 1e-12)
        lines!(ax_M, result_fixed.t_history, Mvv;
               color = :steelblue, linewidth = 2, label = ":reanchor fixed")
    end
    hlines!(ax_M, [result_fixed.params.Mvv_floor];
            color = :gray, linestyle = :dash,
            label = string("Mvv_floor = ", result_fixed.params.Mvv_floor))
    axislegend(ax_M; position = :rb)

    ax_ρ = Axis(fig[2, 1];
                title = "ρ_max (compression cascade indicator)",
                xlabel = "t",
                ylabel = "ρ_max",
                yscale = log10)
    if !isempty(result_baseline.t_history)
        ρ = max.(result_baseline.rho_max_history, 1e-12)
        lines!(ax_ρ, result_baseline.t_history, ρ;
               color = :crimson, linewidth = 2, label = ":none baseline")
    end
    if !isempty(result_fixed.t_history)
        ρ = max.(result_fixed.rho_max_history, 1e-12)
        lines!(ax_ρ, result_fixed.t_history, ρ;
               color = :steelblue, linewidth = 2, label = ":reanchor fixed")
    end
    axislegend(ax_ρ; position = :rb)

    ax_ev = Axis(fig[2, 2];
                 title = "Cumulative projection events (:reanchor)",
                 xlabel = "t",
                 ylabel = "events")
    if !isempty(result_fixed.t_history)
        scatterlines!(ax_ev, result_fixed.t_history, result_fixed.n_events_history;
                      color = :purple, markersize = 4)
    end

    save(outpath, fig)
    return fig
end

# -----------------------------------------------------------------------------
# Save raw history to HDF5
# -----------------------------------------------------------------------------

function save_history(result_baseline, result_fixed; outpath::AbstractString)
    HDF5.h5open(outpath, "w") do f
        gb = HDF5.create_group(f, "baseline")
        gb["t"] = result_baseline.t_history
        gb["E"] = result_baseline.E_history
        gb["M"] = result_baseline.M_history
        gb["p"] = result_baseline.p_history
        gb["Mvv_min"] = result_baseline.Mvv_min_history
        gb["Mvv_max"] = result_baseline.Mvv_max_history
        gb["rho_max"] = result_baseline.rho_max_history
        gb["n_done"] = result_baseline.n_done
        gb["n_target"] = result_baseline.n_steps_target

        gf = HDF5.create_group(f, "fixed")
        gf["t"] = result_fixed.t_history
        gf["E"] = result_fixed.E_history
        gf["M"] = result_fixed.M_history
        gf["p"] = result_fixed.p_history
        gf["Mvv_min"] = result_fixed.Mvv_min_history
        gf["Mvv_max"] = result_fixed.Mvv_max_history
        gf["rho_max"] = result_fixed.rho_max_history
        gf["n_events"] = result_fixed.n_events_history
        gf["n_done"] = result_fixed.n_done
        gf["n_target"] = result_fixed.n_steps_target
        gf["total_events"] = result_fixed.proj_stats.n_events
        gf["floor_events"] = result_fixed.proj_stats.n_floor_events
        gf["total_dE_inj"] = result_fixed.proj_stats.total_dE_inj
    end
end

# -----------------------------------------------------------------------------
# Top-level
# -----------------------------------------------------------------------------

function main(; quick::Bool = false, n_steps_full::Int = 12000)
    n_steps = quick ? 2000 : n_steps_full
    N       = quick ? 64   : 128
    dt      = quick ? 1e-3 : 5e-4

    @printf("[M2-3] === BASELINE :none (M1 path) ===\n")
    result_baseline = run_long_time_wavepool(
        project_kind = :none, N = N, n_steps = n_steps, dt = dt,
        save_every = max(50, n_steps ÷ 100), verbose = true)

    @printf("\n[M2-3] === FIXED :reanchor (M2-3 path) ===\n")
    result_fixed = run_long_time_wavepool(
        project_kind = :reanchor, N = N, n_steps = n_steps, dt = dt,
        save_every = max(50, n_steps ÷ 100), verbose = true)

    @printf("\n[M2-3] === SUMMARY ===\n")
    @printf("[M2-3] baseline (:none): %d / %d steps before %s\n",
            result_baseline.n_done, n_steps,
            result_baseline.n_done < n_steps ? "BLOW-UP" : "completion")
    @printf("[M2-3] fixed (:reanchor): %d / %d steps; %d events (rate %.4e)\n",
            result_fixed.n_done, n_steps, result_fixed.proj_stats.n_events,
            result_fixed.n_done > 0 ?
                result_fixed.proj_stats.n_events / (result_fixed.n_done * N) :
                0.0)

    figs_dir = joinpath(@__DIR__, "..", "reference", "figs")
    mkpath(figs_dir)
    save_history(result_baseline, result_fixed;
                 outpath = joinpath(figs_dir, "M2_3_stability_comparison.h5"))
    plot_stability_comparison(result_baseline, result_fixed;
                              outpath = joinpath(figs_dir,
                                                 "M2_3_stability_comparison.png"))

    return (baseline = result_baseline, fixed = result_fixed)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
