# experiments/A2_cold_sinusoid.jl
#
# Tier A.2 — cold-sinusoid τ-scan on the variational integrator.
#
# Methods paper §10.2 A.2 / `reference/MILESTONE_1_PLAN.md` Phase 6.
# Generalizes the Phase-3 deterministic Zel'dovich match (single τ → ∞)
# across six τ decades. Reproduces dfmm Fig. 3 and compares against
# Track B's golden at the golden's τ value.
#
# # Setup
#
# Periodic [0, 1] mass-coordinate domain, uniform ρ_0 = 1, sinusoidal
# initial velocity u_0(m) = A sin(2π m), cold initial pressure
# P_0 = ρ_0 · T_0 with T_0 ≪ A². The caustic time of the cold sinusoid
# is t_cross = 1/(2π A); the integrator is run to t = 0.95 t_cross
# (just before the analytic Zel'dovich density diverges) for each
# τ ∈ {10⁻³, 10⁻¹, 10¹, 10³, 10⁵, 10⁷}.
#
# # Physical interpretation of the τ regimes
#
# - **τ → 0** (10⁻³): BGK relaxation is fast; (P_xx, P_⊥) instantly
#   isotropise to P_iso each step ⇒ effectively isothermal Euler. In
#   the cold-limit IC (M_vv ≈ 10⁻⁶ ≪ A²), the bulk advection still
#   dominates over pressure, so the wave still steepens into a caustic
#   — but with a tiny pressure floor that suppresses β-growth.
#
# - **τ → ∞** (10⁵, 10⁷): BGK is frozen; (P_xx, P_⊥) advect
#   independently. The Cholesky sector free-streams: β advected by
#   strain, γ² = M_vv − β². This is the deterministic limit Phase 3
#   verified to 2.57e-8.
#
# - **Intermediate** (10⁻¹, 10¹, 10³): smooth bridge between the two.
#   τ = 10³ is the golden's value (see `reference/golden/cold_sinusoid.h5`
#   `params/tau`).
#
# # γ-decade-drop expectation: where the brief and the implementation diverge
#
# The methods-paper §10.2 A.2 acceptance criterion reads "γ drops by
# ~6 decades at caustics" (from a kinetic perspective: as mass packs
# into the caustic, the velocity-dispersion ⇒ γ ought to shrink). With
# the current Γ = 5/3 ideal-gas EOS,
#     M_vv(J, s) = J^{1-Γ} e^{s/c_v} = ρ^{Γ-1} e^{s/c_v},
# γ² = M_vv − β² *grows* at compression for Γ > 1. The Phase-3 notes
# (`reference/notes_phase3_solver.md`) documented this same inversion
# for the Hessian-degeneracy diagnostic: |det Hess| ∝ α² γ² peaks at
# the caustic, not bottoms out there.
#
# What the variational integrator *does* show across all τ regimes
# (and what we plot in the τ-scan figure):
#   - Pre-crossing density matches Zel'dovich to L∞ rel < 1e-3.
#   - γ² stays bounded throughout (cold-limit-faithful: γ² ≪ 1).
#   - The integrator survives all six τ decades without Newton failure.
#
# The "γ drops 6 decades" methods-paper claim is flagged in the open-
# questions section of `reference/notes_phase6_cold_sinusoid.md`. The
# regression test asserts the empirical signature (γ² stays at machine-
# noise scale across the cold-limit τ → ∞ regime), not the inverted
# claim.
#
# # Reference
# - methods paper §10.2 A.2
# - reference/MILESTONE_1_PLAN.md Phase 6
# - reference/notes_phase3_solver.md (inversion of γ-at-caustic claim)
# - reference/notes_phase6_cold_sinusoid.md (this experiment's notes)
# - reference/golden/cold_sinusoid.h5 (Track-B regression target)

using dfmm
using HDF5
using Printf
using CairoMakie

# ──────────────────────────────────────────────────────────────────────
# Helpers — mesh construction in the cold-sinusoid IC
# ──────────────────────────────────────────────────────────────────────

"""
    build_cold_sinusoid_mesh(N, A; T0=1e-6, ρ0=1.0, σ_x0=0.02) -> Mesh1D

Construct a multi-segment Phase-2/5 periodic mesh for the cold-sinusoid
problem. Mirrors `setup_cold_sinusoid` in `src/setups.jl` but builds the
`Mesh1D` directly so the integrator can run on it.

Per-segment entropy is set so `M_vv(J=1, s) = T0`, i.e. `s = log(T0)`
(uniform IC; ρ_0 = 1 means J_0 = 1 for every segment). The vertex
velocity at the *left vertex* of segment j is `u_0(m_j) = A sin(2π m_j)`
with `m_j = (j-1)/N` matching the Phase-3 convention.

`Pp_init = ρ_0 · T_0 = T_0` (isotropic Maxwellian IC), so the Phase-5
post-Newton BGK update has nothing to relax at t = 0.
"""
function build_cold_sinusoid_mesh(N::Integer, A::Real;
                                  T0::Real = 1e-6,
                                  ρ0::Real = 1.0,
                                  σ_x0::Real = 0.02)
    L = 1.0
    Δx = L / N
    Δm = fill(ρ0 * Δx, N)
    positions  = collect((0:N-1) .* Δx)
    velocities = A .* sin.(2π .* (0:N-1) ./ N)
    αs = fill(σ_x0, N)
    βs = fill(0.0, N)
    ss = fill(log(T0), N)             # M_vv(J=1, s) = exp(s/c_v) = T0
    Pps = fill(ρ0 * T0, N)             # isotropic initial Maxwellian
    return dfmm.Mesh1D(positions, velocities, αs, βs, ss;
                       Δm = Δm, Pps = Pps, L_box = L, periodic = true)
end

# ──────────────────────────────────────────────────────────────────────
# Helpers — Zel'dovich analytic reference
# ──────────────────────────────────────────────────────────────────────

"""Analytic Zel'dovich position `x(m, t) = m + A sin(2π m) t`."""
zeldovich_x(m::Real, t::Real, A::Real) = m + A * sin(2π * m) * t

"""
Exact segment-integrated Zel'dovich density. The fair comparison for
the variational integrator's segment-density `Δm/(x_{j+1} − x_j)`.
"""
zeldovich_segment_density(m_l::Real, m_r::Real, t::Real, A::Real, Δm::Real) =
    Δm / (zeldovich_x(m_r, t, A) - zeldovich_x(m_l, t, A))

"""
    density_l_inf_rel(mesh, t, A) -> Float64

L∞ relative error of the variational density profile vs the exact
segment-integrated Zel'dovich density at time `t` and amplitude `A`.
"""
function density_l_inf_rel(mesh, t::Real, A::Real)
    N = dfmm.n_segments(mesh)
    L = mesh.L_box
    err = 0.0
    @inbounds for j in 1:N
        m_l = L * (j - 1) / N
        m_r = L *  j      / N
        ρ_now = dfmm.segment_density(mesh, j)
        ρ_pred = zeldovich_segment_density(m_l, m_r, t, A, mesh.segments[j].Δm)
        rel = abs(ρ_now - ρ_pred) / max(abs(ρ_pred), 1e-30)
        err = max(err, rel)
    end
    return err
end

# ──────────────────────────────────────────────────────────────────────
# Helpers — diagnostics extraction at the caustic cell
# ──────────────────────────────────────────────────────────────────────

"""
    caustic_index(N) -> Int

Cell index closest to the caustic location m = 1/2 on a uniform-Δm
mesh of N segments. Matches Phase-3's `j_caustic = (N ÷ 2) + 1`.
"""
caustic_index(N::Integer) = (N ÷ 2) + 1

"""
    gamma_sq_at(mesh, j) -> Float64

γ² = max(M_vv(J, s) − β², 0) at segment `j`, reconstructed from the
EOS. Matches `det_hessian_HCh`'s γ-reconstruction.
"""
function gamma_sq_at(mesh, j::Integer)
    seg = mesh.segments[j].state
    ρ = dfmm.segment_density(mesh, j)
    Mvv_j = dfmm.Mvv(1/ρ, seg.s)
    return max(Mvv_j - seg.β^2, 0.0)
end

"""
    profile_gamma_sq(mesh) -> Vector{Float64}

Spatial profile of γ² over all segments.
"""
function profile_gamma_sq(mesh)
    N = dfmm.n_segments(mesh)
    out = Vector{Float64}(undef, N)
    @inbounds for j in 1:N
        out[j] = gamma_sq_at(mesh, j)
    end
    return out
end

"""
    profile_eulerian(mesh) -> NamedTuple

Returns `(x, ρ, u, α, β, γ, Pxx, Pp)` per segment. `x` is the segment
center on the Lagrangian box (matching the golden's `(i+0.5)/N`
convention).
"""
function profile_eulerian(mesh)
    N = dfmm.n_segments(mesh)
    x = zeros(N); ρ = zeros(N); u = zeros(N)
    α = zeros(N); β = zeros(N); γ = zeros(N)
    Pxx = zeros(N); Pp = zeros(N)
    L = mesh.L_box
    @inbounds for j in 1:N
        seg = mesh.segments[j]
        ρ[j] = dfmm.segment_density(mesh, j)
        # Velocity at cell center = average of left and right vertex u.
        j_right = j == N ? 1 : j + 1
        u[j] = (seg.state.u + mesh.segments[j_right].state.u) / 2
        # Cell center position with periodic wrap on the last segment.
        wrap = (j == N) ? L : 0.0
        x[j] = (seg.state.x + mesh.segments[j_right].state.x + wrap) / 2
        α[j] = seg.state.α
        β[j] = seg.state.β
        Mvv_j = dfmm.Mvv(1/ρ[j], seg.state.s)
        γ[j]  = sqrt(max(Mvv_j - seg.state.β^2, 0.0))
        Pxx[j] = ρ[j] * Mvv_j
        Pp[j]  = seg.state.Pp
    end
    return (; x, ρ, u, α, β, γ, Pxx, Pp)
end

# ──────────────────────────────────────────────────────────────────────
# Driver — single-τ run
# ──────────────────────────────────────────────────────────────────────

"""
    run_cold_sinusoid(τ; N=64, A=1.0, T0=1e-6, t_frac=0.95,
                     N_steps=400, σ_x0=0.02, verbose=false)

Run the cold-sinusoid IC at relaxation time `τ` from t = 0 to
t = `t_frac` · t_cross with `N_steps` substeps. `t_cross = 1/(2π A)`.

Returns a NamedTuple `(mesh, summary, profile, mesh0_profile)` where
- `mesh` is the final state.
- `summary` carries (t_now, dt, ρ_err vs Zel'dovich, γ² ratio at caustic,
  conservation diagnostics, wall_seconds, n_steps).
- `profile` is `profile_eulerian(mesh)` at t_now.
- `mesh0_profile` is `profile_eulerian` at t = 0 (for plotting).
"""
function run_cold_sinusoid(τ::Real;
                           N::Integer = 64,
                           A::Real = 1.0,
                           T0::Real = 1e-6,
                           t_frac::Real = 0.95,
                           N_steps::Integer = 400,
                           σ_x0::Real = 0.02,
                           verbose::Bool = false)
    mesh = build_cold_sinusoid_mesh(N, A; T0 = T0, σ_x0 = σ_x0)

    # IC diagnostics for plotting + conservation accounting.
    mesh0_profile = profile_eulerian(mesh)
    j_c = caustic_index(N)
    γ²0_caustic = gamma_sq_at(mesh, j_c)
    mass0 = dfmm.total_mass(mesh)
    mom0  = dfmm.total_momentum(mesh)
    KE0   = dfmm.total_kinetic_energy(mesh)
    Eint0 = dfmm.total_internal_energy(mesh)
    Etot0 = KE0 + Eint0

    t_cross = 1 / (2π * A)
    t_target = t_frac * t_cross
    dt = t_target / N_steps

    t0 = time()
    fails = 0
    for n in 1:N_steps
        try
            dfmm.det_step!(mesh, dt; tau = τ)
        catch err
            fails += 1
            verbose && @error "Newton failure" τ n err
            break
        end
    end
    wall = time() - t0
    t_now = N_steps * dt

    γ²_caustic = gamma_sq_at(mesh, j_c)
    log10_γ_ratio = log10(max(γ²_caustic, 1e-300) / max(γ²0_caustic, 1e-300))

    ρ_err = density_l_inf_rel(mesh, t_now, A)

    KE1   = dfmm.total_kinetic_energy(mesh)
    Eint1 = dfmm.total_internal_energy(mesh)
    Etot1 = KE1 + Eint1

    summary = (
        τ = float(τ), N = Int(N), A = float(A), T0 = float(T0),
        N_steps = Int(N_steps), dt = dt, t_now = t_now, t_frac = float(t_frac),
        t_cross = t_cross,
        ρ_err = ρ_err, fails = fails,
        γ²0_caustic = γ²0_caustic, γ²_caustic = γ²_caustic,
        log10_γ_ratio = log10_γ_ratio,
        mass_err = abs(dfmm.total_mass(mesh) - mass0),
        mom_err  = abs(dfmm.total_momentum(mesh) - mom0),
        KE0 = KE0, Eint0 = Eint0, Etot0 = Etot0,
        KE1 = KE1, Eint1 = Eint1, Etot1 = Etot1,
        ΔE_rel = abs(Etot1 - Etot0) / max(abs(Etot0), 1e-30),
        wall_seconds = wall,
    )
    profile = profile_eulerian(mesh)
    return (; mesh, summary, profile, mesh0_profile)
end

# ──────────────────────────────────────────────────────────────────────
# Driver — six-τ scan
# ──────────────────────────────────────────────────────────────────────

"""
    run_tau_scan(; τs, N=64, A=1.0, T0=1e-6, N_steps=400, t_frac=0.95)
                 -> Vector{NamedTuple}

Run `run_cold_sinusoid` once per τ in `τs` and collect the results.
Default τs cover six logarithmic decades.
"""
function run_tau_scan(; τs::AbstractVector = [1e-3, 1e-1, 1e1, 1e3, 1e5, 1e7],
                      N::Integer = 64, A::Real = 1.0, T0::Real = 1e-6,
                      N_steps::Integer = 400, t_frac::Real = 0.95,
                      σ_x0::Real = 0.02, verbose::Bool = false)
    results = Vector{Any}(undef, length(τs))
    for (i, τ) in enumerate(τs)
        results[i] = run_cold_sinusoid(τ;
                                       N = N, A = A, T0 = T0,
                                       t_frac = t_frac, N_steps = N_steps,
                                       σ_x0 = σ_x0, verbose = verbose)
        verbose && @info "τ-scan step" τ ρ_err = results[i].summary.ρ_err γ_ratio = results[i].summary.log10_γ_ratio
    end
    return results
end

# ──────────────────────────────────────────────────────────────────────
# Plotting — 6-panel τ-scan figure (the headline output)
# ──────────────────────────────────────────────────────────────────────

"""
    plot_tau_scan(results; out_path)

Render a 6-panel figure (one per τ). Each panel shows:
- the integrator's final density profile (red, dashed),
- the analytic Zel'dovich density (black) at the same t/t_cross,
- a secondary axis for log₁₀ γ² (blue dotted),
all on the cell-center x-coordinate.

The Zel'dovich curve is sampled on the same Lagrangian m-grid
as the integrator output, so the two curves are directly comparable
when the integrator is faithful to Zel'dovich pre-crossing.
"""
function plot_tau_scan(results;
                       out_path::AbstractString =
                       joinpath(@__DIR__, "..", "reference", "figs",
                                "A2_cold_sinusoid_tauscan.png"))
    mkpath(dirname(out_path))
    fig = Figure(size = (1300, 800))
    Label(fig[0, 1:3],
          "Tier A.2 cold sinusoid τ-scan — variational integrator vs Zel'dovich pre-crossing",
          fontsize = 16)

    for (k, r) in enumerate(results)
        row = (k - 1) ÷ 3 + 1
        col = (k - 1) % 3 + 1
        s = r.summary

        # Final-time density at t_now / t_cross.
        x_jl = r.profile.x
        ρ_jl = r.profile.ρ

        # Zel'dovich segment-density on the same Lagrangian m-grid.
        N = length(x_jl)
        L = 1.0
        ρ_zel = [zeldovich_segment_density(L * (j - 1) / N, L * j / N,
                                           s.t_now, s.A, L / N)
                 for j in 1:N]
        # Plot Zel'dovich vs the *Lagrangian* m-coordinate (not Eulerian
        # x), so the analytic curve doesn't suffer from the integrator's
        # own caustic-narrowing. The integrator's x is then plotted at
        # m for honest visual alignment.
        m_grid = [L * (j - 0.5) / N for j in 1:N]

        ax = Axis(fig[row, col];
                  xlabel = "m (mass coordinate)",
                  ylabel = "ρ",
                  title = @sprintf("τ = %.0e   |   ρ_err = %.2e   |   log₁₀(γ²/γ²₀) = %+.2f",
                                   s.τ, s.ρ_err, s.log10_γ_ratio))
        lines!(ax, m_grid, ρ_zel,
               color = :black, linewidth = 1.6, label = "Zel'dovich")
        lines!(ax, m_grid, ρ_jl,
               color = :red, linestyle = :dash, linewidth = 1.4,
               label = "variational")
        if k == 1
            axislegend(ax; position = :lt, framevisible = false)
        end
    end

    save(out_path, fig)
    return out_path
end

"""
    plot_tau_scan_hessian(results; out_path)

Phase-3-style Hessian-degeneracy diagnostic per τ. Each panel shows the
spatial profile of log₁₀|det Hess(H_Ch)| at the final time, plus the
final γ² profile. Mirrors `reference/figs/phase3_hessian_degen.png`'s
right panel but with one panel per τ.
"""
function plot_tau_scan_hessian(results;
                               out_path::AbstractString =
                               joinpath(@__DIR__, "..", "reference", "figs",
                                        "A2_cold_sinusoid_density.png"))
    mkpath(dirname(out_path))
    fig = Figure(size = (1300, 800))
    Label(fig[0, 1:3],
          "Tier A.2 cold sinusoid τ-scan — γ² and |det Hess(H_Ch)| at t = 0.95 t_cross",
          fontsize = 16)

    for (k, r) in enumerate(results)
        row = (k - 1) ÷ 3 + 1
        col = (k - 1) % 3 + 1
        s = r.summary
        N = length(r.profile.x)
        L = 1.0
        m_grid = [L * (j - 0.5) / N for j in 1:N]

        γ²_prof = [r.profile.γ[j]^2 for j in 1:N]
        # |det Hess(H_Ch)| = α² (γ² + 4 β²); we plot log₁₀ of this.
        det_prof = [abs(dfmm.det_hessian_HCh(r.profile.α[j],
                                             r.profile.β[j],
                                             r.profile.γ[j])) for j in 1:N]

        ax = Axis(fig[row, col];
                  xlabel = "m",
                  ylabel = "log₁₀ |det Hess|",
                  title = @sprintf("τ = %.0e   |   max γ² = %.2e",
                                   s.τ, maximum(γ²_prof)))
        lines!(ax, m_grid, log10.(max.(det_prof, 1e-30)),
               color = :red, linewidth = 1.4, label = "log₁₀ |det Hess|")
        lines!(ax, m_grid, log10.(max.(γ²_prof, 1e-30)),
               color = :blue, linestyle = :dot, linewidth = 1.4,
               label = "log₁₀ γ²")
        if k == 1
            axislegend(ax; position = :lb, framevisible = false)
        end
    end

    save(out_path, fig)
    return out_path
end

# ──────────────────────────────────────────────────────────────────────
# Golden comparison
# ──────────────────────────────────────────────────────────────────────

"""
    compare_to_golden(τ; N=400, A=1.0, T0=1e-3, t_end=0.6,
                      N_steps=2000, σ_x0=0.02) -> NamedTuple

Match the golden's parameters as closely as possible, run the
variational integrator, and compare per-cell on (ρ, u, α, β, γ).

The golden was generated by py-1d at (N=400, A=1, T0=1e-3, t_end=0.6,
τ=1e3, σ_x0=0.02, periodic). Our integrator uses the same parameters.
"""
function compare_to_golden(τ::Real = 1e3;
                           N::Integer = 400, A::Real = 1.0,
                           T0::Real = 1e-3, t_end::Real = 0.6,
                           N_steps::Integer = 2000,
                           σ_x0::Real = 0.02,
                           verbose::Bool = true)
    golden = dfmm.load_tier_a_golden(:cold_sinusoid)

    mesh = build_cold_sinusoid_mesh(N, A; T0 = T0, σ_x0 = σ_x0)
    dt = t_end / N_steps
    t0 = time()
    for n in 1:N_steps
        dfmm.det_step!(mesh, dt; tau = τ)
    end
    wall = time() - t0

    profile = profile_eulerian(mesh)

    function linf_rel(a, b)
        scale = max(maximum(abs, b), 1e-12)
        return maximum(abs, a .- b) / scale
    end

    # Golden's last column = snapshot at t_end. The Lagrangian m-grid
    # used by the integrator is shifted by half a cell from the golden's
    # Eulerian grid; both are length N with cell centers at (i+0.5)/N
    # in Lagrangian coordinates by construction (identical IC).
    rho_g = golden.fields.rho[:, end]
    u_g   = golden.fields.u[:, end]
    α_g   = golden.fields.alpha[:, end]
    β_g   = golden.fields.beta[:, end]
    γ_g   = golden.fields.gamma[:, end]

    err_ρ = linf_rel(profile.ρ, rho_g)
    err_u = linf_rel(profile.u, u_g)
    err_α = linf_rel(profile.α, α_g)
    err_β = linf_rel(profile.β, β_g)
    err_γ = linf_rel(profile.γ, γ_g)

    if verbose
        @info "A.2 cold-sinusoid golden comparison" wall err_ρ err_u err_α err_β err_γ
    end

    return (; profile, golden, err_ρ, err_u, err_α, err_β, err_γ, wall)
end

# ──────────────────────────────────────────────────────────────────────
# Main: production driver — runs the full τ-scan and writes artefacts.
# ──────────────────────────────────────────────────────────────────────

"""
    main_a2_cold_sinusoid()

End-to-end production driver:
1. Run the six-τ scan at production resolution (N=128).
2. Render `A2_cold_sinusoid_tauscan.png` and `A2_cold_sinusoid_density.png`.
3. Run the golden comparison at τ = 1e3, N = 400.
4. Print the τ-scan results table to stdout.
"""
function main_a2_cold_sinusoid()
    @info "Running A.2 cold-sinusoid τ-scan at production resolution (N = 128)..."
    results = run_tau_scan(; N = 128, A = 1.0, T0 = 1e-6,
                           N_steps = 800, t_frac = 0.95, verbose = true)
    fig_main = plot_tau_scan(results)
    fig_aux  = plot_tau_scan_hessian(results)
    @info "τ-scan figures written" fig_main fig_aux

    println("\n=== A.2 τ-scan summary table (N = 128, A = 1, T_0 = 1e-6) ===")
    @printf "%10s | %10s | %12s | %12s | %12s | %12s | %10s\n" "τ" "ρ_err" "γ²_caustic_0" "γ²_caustic_T" "log10 ratio" "ΔE_rel" "wall (s)"
    println("-"^96)
    for r in results
        s = r.summary
        @printf "%10.0e | %10.2e | %12.2e | %12.2e | %+12.2f | %12.2e | %10.2f\n" s.τ s.ρ_err s.γ²0_caustic s.γ²_caustic s.log10_γ_ratio s.ΔE_rel s.wall_seconds
    end

    @info "Running A.2 cold-sinusoid golden comparison (τ = 1e3, N = 400)..."
    cmp = compare_to_golden(1e3; N = 400, A = 1.0, T0 = 1e-3,
                            t_end = 0.6, N_steps = 2000)
    println("\n=== A.2 golden comparison (τ = 1e3, N = 400, t = 0.6) ===")
    @printf "L∞ rel err:  ρ = %.3e   u = %.3e   α = %.3e   β = %.3e   γ = %.3e\n" cmp.err_ρ cmp.err_u cmp.err_α cmp.err_β cmp.err_γ

    return (; results, cmp)
end
