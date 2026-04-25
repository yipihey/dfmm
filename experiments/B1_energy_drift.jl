# experiments/B1_energy_drift.jl
#
# Tier B.1 — long-time energy drift on a smooth periodic acoustic wave.
#
# Methods paper §10.3 B.1 acceptance: |ΔE|/|E_0| < 10⁻⁸ over 10⁵ timesteps
# on the variational integrator.
#
# Setup (per the Phase-4 brief):
# - Periodic 1D mesh on the Lagrangian-mass coordinate.
# - Uniform background ρ_0 = 1, M_vv,0 = 1 (so β_0 = 0, γ_0 = 1, α_0 = 1,
#   s_0 = 0 with cv = 1, Γ = 5/3 default).
# - Density perturbation δρ/ρ_0 = ε cos(2π m / M_total), ε = 1e-4.
#   (Velocity field zero at t = 0 — standing-wave IC.)
# - dt chosen for tight bounded oscillation (see "resolution choice" below).
#
# Output:
# - HDF5 history at `reference/figs/B1_energy_history.h5`
#   (all per-step E/momentum/mass arrays).
# - PNG plot at `reference/figs/B1_energy_drift.png`
#   showing (E(t)−E_0)/|E_0| vs. t on log-y, plus chunked envelope.
# - Console summary line.
#
# Resolution choice (Option A from the brief):
#
#   N = 8 segments, dt = 1e-6, 10⁵ steps  ⇒  t_end = 0.1 (well below
#   one acoustic period L/c_s ≈ 0.77 — a *long-timestep* test rather
#   than a long-physical-time test). Wall time ≈ 50 s for the full run.
#
# Why this resolution? Two competing effects in the Phase-2 setup:
#
# 1. The integrator is 2nd-order symplectic. Per-step truncation gives
#    bounded oscillation O(Δt² · |H|) ≈ 1e-12 at Δt = 1e-6.
#
# 2. The autonomous Cholesky-sector trajectory at fixed M_vv has β
#    monotonically saturating at √M_vv and α growing linearly (no
#    closed orbits — see derivation in
#    `reference/notes_phase4_energy_drift.md`). Numerical drift relative
#    to the level set α²(M_vv − β²) = 1 accumulates as O(Δt² · t).
#    To stay below 1e-8 we need t · Δt² × |H| < 1e-8, i.e. t < 2 × 10⁷ Δt²
#    (with |H| = 0.5). At Δt = 1e-6 ⇒ t < 20; at Δt = 1e-4 ⇒ t < 2.
#
# At Δt = 1e-6, t_end = 0.1: αmax stays ≈ 1.005, drift envelope
# < 1e-8 over the full 10⁵ steps. The trace shows a small monotone
# t¹ secular component (5e-9 over the run) plus bounded oscillation.
# See `reference/notes_phase4_energy_drift.md` "Findings" for the
# diagnosis and what would be needed to reduce the secular leak.
#
# The `quick = true` keyword runs only `n_steps ÷ 10` steps and projects
# the drift bound forward by ×10 (Option B from the brief; used by the
# in-suite regression test).

using dfmm
using HDF5
using Printf
using CairoMakie

"""
    setup_b1_acoustic(N; ε = 1e-4, dt = 1e-6)

Build the standing-wave acoustic-wave initial condition described in the
file header. Returns `(mesh, dt, c_s, ω_analytical, L_box, ε)`.

`dt` defaults to 1e-6 — the resolution that keeps the cumulative
(α, β)-sector drift bounded below 1e-8 over 10⁵ steps. See the
experiment-level docstring.
"""
function setup_b1_acoustic(N::Integer; ε::Real = 1e-4, dt::Real = 1e-6)
    L = 1.0
    ρ0 = 1.0
    s0 = 0.0       # M_vv,0 = J^(1-Γ) exp(s/cv) = 1 at J = 1
    α0 = 1.0
    β0 = 0.0
    Δm_val = ρ0 * (L / N)
    Δm = fill(Δm_val, N)
    M_total = sum(Δm)
    k_m = 2π / M_total

    # Density-perturbation IC on a Lagrangian mesh: place the N left-vertex
    # positions so that segment j has Eulerian density
    # ρ_j = ρ_0 (1 + ε cos(k_m · m_j_center)).
    positions = zeros(Float64, N)
    cum_x = 0.0
    for j in 1:N
        m_center = (j - 0.5) * Δm_val
        ρ_j = ρ0 * (1 + ε * cos(k_m * m_center))
        positions[j] = cum_x
        cum_x += Δm_val / ρ_j
    end
    L_box = cum_x

    velocities = zeros(Float64, N)
    αs = fill(α0, N)
    βs = fill(β0, N)
    ss = fill(s0, N)

    mesh = Mesh1D(positions, velocities, αs, βs, ss;
                  Δm = Δm, L_box = L_box, periodic = true)

    Γ = 5.0 / 3.0
    c_s = sqrt(Γ)
    ω_analytical = c_s * (2π / L_box)

    return (; mesh, dt, c_s, ω_analytical, L_box, ε)
end

"""
    run_b1_drift(; N = 8, n_steps = 100_000, dt = 1e-6, ε = 1e-4,
                  log_stride = 100,
                  out_h5 = nothing,
                  out_png = nothing,
                  quick = false)

Execute the long-time energy-drift experiment. Returns a NamedTuple with
the full per-step history (mass, momentum, energy) plus aggregate
statistics. If `out_h5` and/or `out_png` are paths, writes outputs.

Keyword arguments:
- `N`           — segment count (default 8; the brief notes "N ≥ 8 per
                  wavelength is fine for the lowest mode").
- `n_steps`     — total number of timesteps (default 10⁵).
- `dt`          — timestep (default 1e-6; see file header).
- `ε`           — fractional density-perturbation amplitude.
- `log_stride`  — log every k-th step to control HDF5 file size.
                  Default 100 ⇒ 1001 samples for n_steps = 100_000.
- `out_h5`      — HDF5 output path. `nothing` to skip.
- `out_png`     — PNG plot path. `nothing` to skip.
- `quick`       — if true, run only `n_steps ÷ 10` steps and project the
                  drift bound forward by ×10 (Option B from the brief).

Outputs to console: a one-line summary (max relE drift, end value, wall
time, αmax) plus chunk-mean trace.
"""
function run_b1_drift(;
    N::Integer = 8,
    n_steps::Integer = 100_000,
    dt::Real = 1e-6,
    ε::Real = 1e-4,
    log_stride::Integer = 100,
    out_h5::Union{Nothing,AbstractString} = nothing,
    out_png::Union{Nothing,AbstractString} = nothing,
    quick::Bool = false,
)
    if quick
        n_steps = max(1, div(n_steps, 10))
    end

    setup = setup_b1_acoustic(N; ε = ε, dt = dt)
    (; mesh, c_s, L_box) = setup

    E0 = total_energy(mesh)
    m0 = total_mass(mesh)
    p0 = total_momentum(mesh)

    n_log = div(n_steps, log_stride) + 1
    log_t = zeros(n_log)
    log_E = zeros(n_log)
    log_m = zeros(n_log)
    log_p = zeros(n_log)
    log_t[1] = 0.0
    log_E[1] = E0
    log_m[1] = m0
    log_p[1] = p0
    αmax_global = mesh.segments[1].state.α
    βmax_global = abs(mesh.segments[1].state.β)
    γ²min_global = Inf

    @info "B1 setup" N n_steps dt c_s ε E0

    t_start = time()
    for n in 1:n_steps
        det_step!(mesh, dt)
        for j in 1:n_segments(mesh)
            seg = mesh.segments[j].state
            αmax_global = max(αmax_global, seg.α)
            βmax_global = max(βmax_global, abs(seg.β))
            ρ_j = segment_density(mesh, j)
            Mvv_j = Mvv(1.0 / ρ_j, seg.s)
            γ²min_global = min(γ²min_global, Mvv_j - seg.β^2)
        end
        if n % log_stride == 0
            idx = div(n, log_stride) + 1
            log_t[idx] = n * dt
            log_E[idx] = total_energy(mesh)
            log_m[idx] = total_mass(mesh)
            log_p[idx] = total_momentum(mesh)
        end
    end
    wall = time() - t_start

    rel_E = (log_E .- E0) ./ abs(E0)
    rel_E_max = maximum(abs, rel_E)
    rel_E_end = rel_E[end]
    Δm_max = maximum(abs, log_m .- m0)
    Δp_max = maximum(abs, log_p .- p0)

    # If quick=true, project the bound forward (×10 in steps).
    projected_bound = quick ? 10.0 * rel_E_max : rel_E_max

    @printf("\n=== B.1 result ===\n")
    @printf("  N=%d, dt=%.0e, steps=%d (%squick), t_end=%.3f, wall=%.1fs\n",
            N, dt, n_steps, quick ? "" : "non-", n_steps * dt, wall)
    @printf("  E_0 = %.6e\n", E0)
    @printf("  max |ΔE|/|E_0|     = %.3e\n", rel_E_max)
    @printf("  ΔE_end/|E_0|       = %+.3e\n", rel_E_end)
    @printf("  max |Δm|           = %.3e (target: round-off)\n", Δm_max)
    @printf("  max |Δp|           = %.3e (target: 1e-12)\n", Δp_max)
    @printf("  αmax = %.3e   βmax = %.6f   γ²min = %.3e\n",
            αmax_global, βmax_global, γ²min_global)
    if quick
        @printf("  projected drift × 10 = %.3e (Option B)\n", projected_bound)
    end
    @printf("  acceptance |ΔE|/|E_0| < 1e-8: %s\n",
            projected_bound < 1e-8 ? "PASS" : "FAIL")

    # Chunk-mean trace: helps distinguish bounded oscillation from t¹ leak.
    chunks = 10
    cs = div(length(rel_E), chunks)
    chunk_mean = zeros(chunks)
    chunk_amp = zeros(chunks)
    chunk_tmid = zeros(chunks)
    for i in 1:chunks
        lo = (i-1) * cs + 1
        hi = i == chunks ? length(rel_E) : i * cs
        chunk = rel_E[lo:hi]
        chunk_mean[i] = sum(chunk) / length(chunk)
        chunk_amp[i] = (maximum(chunk) - minimum(chunk)) / 2
        chunk_tmid[i] = (log_t[lo] + log_t[hi]) / 2
    end

    # Optional outputs.
    if out_h5 !== nothing
        mkpath(dirname(out_h5))
        save_state(out_h5, (;
            t = log_t,
            energy = log_E,
            mass = log_m,
            momentum = log_p,
            rel_E = rel_E,
            chunk_t = chunk_tmid,
            chunk_mean = chunk_mean,
            chunk_amp = chunk_amp,
            E0 = E0,
            m0 = m0,
            p0 = p0,
            N = Int64(N),
            dt = dt,
            n_steps = Int64(n_steps),
            wall_seconds = wall,
            αmax = αmax_global,
            βmax = βmax_global,
            ε = ε,
            c_s = c_s,
            L_box = L_box,
            rel_E_max = rel_E_max,
            Δm_max = Δm_max,
            Δp_max = Δp_max,
        ); created_by = "experiments/B1_energy_drift.jl")
        @info "wrote HDF5" out_h5
    end

    if out_png !== nothing
        mkpath(dirname(out_png))
        plot_b1_drift(log_t, rel_E, chunk_tmid, chunk_mean, chunk_amp,
                      out_png; ε = ε, N = N, dt = dt, n_steps = n_steps,
                      rel_E_max = rel_E_max)
        @info "wrote PNG" out_png
    end

    return (;
        log_t, log_E, log_m, log_p, rel_E,
        E0, m0, p0,
        rel_E_max, rel_E_end, Δm_max, Δp_max,
        αmax = αmax_global, βmax = βmax_global, γ²min = γ²min_global,
        wall, dt, N, n_steps, ε,
        chunk_t = chunk_tmid, chunk_mean, chunk_amp,
        projected_bound,
    )
end

"""
    plot_b1_drift(t, rel_E, chunk_t, chunk_mean, chunk_amp, out_png; ...)

Two-panel figure: (top) signed `rel_E(t)` plus chunk-mean and ±chunk-amp
envelope; (bottom) `|rel_E(t)|` on log-y vs. the 1e-8 acceptance line.
"""
function plot_b1_drift(t, rel_E, chunk_t, chunk_mean, chunk_amp,
                       out_png::AbstractString;
                       ε, N, dt, n_steps, rel_E_max)
    fig = Figure(size = (900, 700))

    ax1 = Axis(fig[1, 1];
        xlabel = "t",
        ylabel = "(E(t) − E₀) / |E₀|",
        title  = "B.1 energy history (N=$N, Δt=$dt, $n_steps steps, ε=$ε)",
    )
    lines!(ax1, t, rel_E; color = :steelblue, label = "per-log-stride")
    upper = chunk_mean .+ chunk_amp
    lower = chunk_mean .- chunk_amp
    band!(ax1, chunk_t, lower, upper; color = (:darkorange, 0.25),
          label = "chunk envelope")
    lines!(ax1, chunk_t, chunk_mean; color = :darkorange, linewidth = 2,
           label = "chunk mean")
    hlines!(ax1, [1e-8, -1e-8]; color = :red, linestyle = :dash,
            label = "±10⁻⁸ acceptance")
    axislegend(ax1; position = :rt)

    ax2 = Axis(fig[2, 1];
        xlabel = "t",
        ylabel = "|(E(t) − E₀) / E₀|",
        yscale = log10,
        title  = "log-y absolute drift",
    )
    abs_rel = abs.(rel_E) .+ 1e-20  # avoid log(0)
    lines!(ax2, t, abs_rel; color = :steelblue)
    hlines!(ax2, [1e-8]; color = :red, linestyle = :dash,
            label = "10⁻⁸ acceptance")
    axislegend(ax2; position = :rt)

    save(out_png, fig)
    return out_png
end

# Entry point: when called as a script, run the full experiment with
# canonical paths.
if abspath(PROGRAM_FILE) == @__FILE__
    repo_root = joinpath(@__DIR__, "..")
    out_h5  = joinpath(repo_root, "reference", "figs", "B1_energy_history.h5")
    out_png = joinpath(repo_root, "reference", "figs", "B1_energy_drift.png")
    run_b1_drift(; N = 8, n_steps = 100_000, dt = 1e-6,
                 out_h5 = out_h5, out_png = out_png)
end
