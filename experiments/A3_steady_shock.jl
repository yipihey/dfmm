# experiments/A3_steady_shock.jl
#
# Tier A.3 — Steady-shock Mach scan (Phase 7).
#
# Methods paper §10.2 A.3 / `reference/MILESTONE_1_PLAN.md` Phase 7.
# Mirrors `py-1d/experiments/03_steady_shock.py` and reproduces dfmm
# Fig. 4 across $M_1 ∈ \{1.5, 2, 3, 5, 10\}$. The integrator's
# inflow-outflow handling pins the leftmost `n_pin` segments to the
# upstream state and the rightmost `n_pin` segments to the downstream
# state after every step (`det_step!` `bc = :inflow_outflow` branch),
# matching py-1d's `step_inflow_outflow` Dirichlet ghost-cell scheme.
#
# Phase 7 deliverables:
#   - Steady-shock IC factory (`build_steady_shock_mesh`).
#   - Single-Mach driver (`run_steady_shock`).
#   - Mach scan (`mach_scan`) → comparison table & plot.
#   - Optional q-on / q-off comparison via `q_kind` kwarg.
#
# References:
#   methods paper §10.2 A.3
#   py-1d/dfmm/setups/shock.py (Rankine-Hugoniot, step_inflow_outflow)
#   py-1d/experiments/03_steady_shock.py (Mach scan, profile plot)

using dfmm
using Printf
using CairoMakie
using HDF5

const GAMMA_LAW = 5.0 / 3.0

"""
    rankine_hugoniot(rho1, u1, P1; gamma_eos = GAMMA_LAW)

Analytic post-shock state for upstream `(rho1, u1, P1)` and adiabatic
index `gamma_eos`. Returns `(rho2, u2, P2, M1)`. Mirrors
`py-1d/dfmm/setups/shock.py::rankine_hugoniot`.
"""
function rankine_hugoniot(rho1::Real, u1::Real, P1::Real;
                          gamma_eos::Real = GAMMA_LAW)
    c_s1 = sqrt(gamma_eos * P1 / rho1)
    M1 = u1 / c_s1
    rho_ratio = (gamma_eos + 1) * M1^2 / ((gamma_eos - 1) * M1^2 + 2)
    P_ratio   = (2 * gamma_eos * M1^2 - (gamma_eos - 1)) / (gamma_eos + 1)
    rho2 = rho1 * rho_ratio
    u2   = u1 / rho_ratio
    P2   = P1 * P_ratio
    return (rho2, u2, P2, M1)
end


"""
    primitives_to_inflow(rho, u, P; sigma_x0 = 0.02, gamma_eos = GAMMA_LAW)

Package primitives `(rho, u, P)` into a NamedTuple consumable by
`det_step!`'s `inflow_state` / `outflow_state` arguments. The Phase-7
mesh BC pinning rewrites the per-segment `DetField` from this
NamedTuple every step.

The Cholesky-sector seeds are taken from the upstream IC convention:
`alpha = sigma_x0`, `beta = 0`, `Pp = P` (isotropic Maxwellian),
`Q = 0` (zero heat flux for a Maxwellian distribution). The
EOS-anchoring entropy is `s = log(P/rho^Γ)` with `c_v = 1`.
"""
function primitives_to_inflow(rho::Real, u::Real, P::Real;
                              sigma_x0::Real = 0.02,
                              gamma_eos::Real = GAMMA_LAW)
    s = log(P / rho^gamma_eos)
    return (rho = rho, u = u, P = P, alpha = sigma_x0, beta = 0.0,
            s = s, Pp = P, Q = 0.0)
end


"""
    build_steady_shock_mesh(ic; gamma_eos = GAMMA_LAW)

Construct a `Mesh1D{Float64,DetField{Float64}}` initialized to the
two-state Rankine-Hugoniot IC at `x = 0.5`. Mirrors `setup_steady_shock`
plus the integrator-level packing details of `build_sod_mesh`. Returns
`(mesh, inflow, outflow)` where `inflow` and `outflow` are NamedTuples
ready for `det_step!`.

Lagrangian segment masses are chosen so that the Eulerian cell width
`Δx = (1/N)` is uniform at t = 0:
    Δm_j = ρ_j · Δx
The mesh internal topology is periodic; the Phase-7 driver pins the
boundary segments to inflow / outflow each step, so the periodic
seam at x = 0 = L_box (where ρ jumps from downstream to upstream)
never gets a chance to develop into a propagating disturbance.
"""
function build_steady_shock_mesh(ic; gamma_eos::Real = GAMMA_LAW)
    N = length(ic.rho)
    Γ = gamma_eos
    Δx = 1.0 / N  # py-1d uses linspace(0, 1, N) so dx = 1/(N-1); we use 1/N
    # so that the cumulative Δx = N · Δx = 1.0 = L_box. Tested: shifting
    # by half a cell in either direction has no measurable impact on
    # R-H plateau values.

    Δm = ic.rho .* Δx

    # Per-segment entropy: s_j = log(P_j/ρ_j^Γ).
    ss = log.(ic.P ./ ic.rho .^ Γ)

    # Vertex positions: cumulative sum of Δx (uniform mesh).
    positions = collect((0:N-1) .* Δx)
    velocities = copy(ic.u)
    αs = copy(ic.alpha_init)
    βs = copy(ic.beta_init)
    Pps = copy(ic.Pp)
    Qs = copy(ic.Q)

    mesh = dfmm.Mesh1D(positions, velocities, αs, βs, ss;
                       Δm = Δm, Pps = Pps, Qs = Qs,
                       L_box = 1.0, periodic = false,
                       bc = :inflow_outflow)

    # Inflow state (upstream, supersonic). Phase 7 pinning convention:
    # `Pp` = isotropic, `Q = 0`. Mirrors `setup_steady_shock`'s `inflow`.
    inflow = primitives_to_inflow(ic.rho1, ic.u1, ic.P1;
                                  sigma_x0 = αs[1], gamma_eos = Γ)
    # Outflow state (downstream, subsonic). Same packaging.
    outflow = primitives_to_inflow(ic.rho2, ic.u2, ic.P2;
                                   sigma_x0 = αs[1], gamma_eos = Γ)

    return (mesh, inflow, outflow)
end


"""
    extract_eulerian_profiles_ss(mesh) -> NamedTuple

Sample (rho, u, Pxx, Pp, Q, x) from the steady-shock mesh state.
Single-loop, returns Float64 column vectors.
"""
function extract_eulerian_profiles_ss(mesh)
    N = dfmm.n_segments(mesh)
    rho = zeros(N); u = zeros(N); Pxx = zeros(N); Pp = zeros(N)
    Qf = zeros(N); xc = zeros(N); α = zeros(N); β = zeros(N); γ = zeros(N)
    is_io = mesh.bc == :inflow_outflow
    @inbounds for j in 1:N
        seg = mesh.segments[j]
        # On inflow_outflow meshes the rightmost segment's "right
        # vertex" is the implicit Dirichlet plane at x = L_box, not
        # the cyclic image of vertex 1. Use the previous segment's
        # density (zero-gradient extrapolation) as a stand-in for ρ
        # at segment N when computing the cell-center diagnostic.
        if is_io && j == N
            # Outflow Dirichlet: ρ_N matches the pinned downstream
            # density. Use Δm_N / (L_box − x_N) as the segment density,
            # which equals the pinned ρ_outflow when the pinning is
            # consistent.
            Δx = mesh.L_box - seg.state.x
            if Δx > eps()
                ρ = seg.Δm / Δx
            else
                ρ = dfmm.segment_density(mesh, j-1)
            end
            J = 1 / ρ
            Mvv_j = dfmm.Mvv(J, seg.state.s)
            rho[j] = ρ
            u_left = seg.state.u
            u_right = u_left  # zero-gradient at outflow
            u[j] = (u_left + u_right) / 2
            Pxx[j] = ρ * Mvv_j
            Pp[j]  = seg.state.Pp
            Qf[j]  = seg.state.Q
            xc[j]  = (seg.state.x + mesh.L_box) / 2
            α[j]   = seg.state.α
            β[j]   = seg.state.β
            γ[j]   = sqrt(max(Mvv_j - seg.state.β^2, 0.0))
            continue
        end
        ρ = dfmm.segment_density(mesh, j)
        J = 1 / ρ
        Mvv_j = dfmm.Mvv(J, seg.state.s)
        rho[j] = ρ
        j_right = j == N ? 1 : j + 1
        u_left  = seg.state.u
        u_right = mesh.segments[j_right].state.u
        u[j]    = (u_left + u_right) / 2
        Pxx[j]  = ρ * Mvv_j
        Pp[j]   = seg.state.Pp
        Qf[j]   = seg.state.Q
        x_left = seg.state.x
        wrap = (j == N) ? mesh.L_box : 0.0
        x_right = mesh.segments[j_right].state.x + wrap
        xc[j]   = (x_left + x_right) / 2
        α[j]    = seg.state.α
        β[j]    = seg.state.β
        γ[j]    = sqrt(max(Mvv_j - seg.state.β^2, 0.0))
    end
    return (; rho, u, Pxx, Pp, Q = Qf, x = xc, α, β, γ)
end


"""
    run_steady_shock(; M1 = 3.0, N = 200, t_end = 3.0, tau = 1e-3,
                       cfl = 0.3, sigma_x0 = 0.02, n_pin = 2,
                       q_kind = :none, c_q_quad = 1.0, c_q_lin = 0.5,
                       verbose = false, log_stride = nothing)

Run the Phase-7 variational integrator on the steady-shock IC at
upstream Mach `M1`. Uses `setup_steady_shock(...)` for the IC, then
`det_step!(...; bc = :inflow_outflow, ...)` for each timestep with
inflow-outflow Dirichlet pinning of the boundary cells.

Keyword `q_kind ∈ {:none, :vNR_linear_quadratic}` opts in to the
Phase-5b artificial viscosity (see `src/artificial_viscosity.jl`).
With `q_kind = :none` (default) the integrator runs the bare
variational form — same regime as Phase 5 Sod, where the discrete
shock-jump conditions are smeared by a few cells. With `q_kind =
:vNR_linear_quadratic` the post-shock plateau converges much closer
to the analytical R-H values (see `notes_phase7_steady_shock.md`).

Returns NamedTuple `(mesh, profile, summary, ic, M1, n_steps, dt)`.
"""
function run_steady_shock(; M1::Float64 = 3.0, N::Int = 200,
                          t_end::Float64 = 3.0, tau::Float64 = 1e-3,
                          cfl::Float64 = 0.1, sigma_x0::Float64 = 0.02,
                          n_pin::Int = 2,
                          q_kind::Symbol = :none,
                          c_q_quad::Float64 = 1.0, c_q_lin::Float64 = 0.5,
                          verbose::Bool = false,
                          log_stride::Union{Int,Nothing} = nothing)
    ic = setup_steady_shock(; M1 = M1, N = N, t_end = t_end,
                            sigma_x0 = sigma_x0, tau = tau, cfl = cfl)
    mesh, inflow, outflow = build_steady_shock_mesh(ic)

    # CFL-bound dt: c_s_max from the IC, dx = 1/N (uniform).
    Γ = GAMMA_LAW
    c_s_max = sqrt(Γ * maximum(ic.P) / minimum(ic.rho))
    u_max   = maximum(abs.(ic.u))
    dx = 1.0 / N
    dt_cfl = cfl * dx / (c_s_max + u_max)
    n_steps = ceil(Int, t_end / dt_cfl)
    dt = t_end / n_steps

    if verbose
        @info "Steady-shock run setup" M1 N dt n_steps τ=tau
    end

    # Initial diagnostics.
    mass0 = dfmm.total_mass(mesh)
    mom0  = dfmm.total_momentum(mesh)

    # Integrate.
    t0 = time()
    if log_stride === nothing
        log_stride = max(1, n_steps ÷ 5)
    end
    for n in 1:n_steps
        dfmm.det_step!(mesh, dt;
                       tau = tau, bc = :inflow_outflow,
                       inflow_state = inflow, outflow_state = outflow,
                       n_pin = n_pin,
                       q_kind = q_kind,
                       c_q_quad = c_q_quad, c_q_lin = c_q_lin)
        if verbose && (n % log_stride == 0 || n == n_steps)
            prof_diag = extract_eulerian_profiles_ss(mesh)
            ρ_max = maximum(prof_diag.rho)
            u_min = minimum(prof_diag.u)
            @info "Steady-shock step" n n_steps t=n*dt ρ_max u_min
        end
    end
    wall = time() - t0

    summary = (
        t_end = t_end, n_steps = n_steps, dt = dt,
        mass_err = abs(dfmm.total_mass(mesh) - mass0) / mass0,
        wall_seconds = wall,
        M1 = M1, tau = tau, q_kind = q_kind, n_pin = n_pin,
    )
    profile = extract_eulerian_profiles_ss(mesh)
    return (; mesh, profile, summary, ic, M1, inflow, outflow,
            n_steps, dt)
end


"""
    measure_postshock_plateau(profile; left_frac = 0.6, right_frac = 0.85)

Average `(rho, u, P_iso)` over the post-shock plateau, defined as the
window `x ∈ [left_frac, right_frac]` (default `[0.6, 0.85]`). The
window excludes:
  - the shock front around x ≈ 0.5 (smeared by integrator),
  - the rightmost pinned segments where Dirichlet downstream is fixed,
  - the trailing reverse-shock region near x ≈ 1 from the periodic seam.

Returns `(rho_avg, u_avg, P_avg)` where `P = (Pxx + 2 Pp)/3` is the
isotropic component (matches py-1d's `extract_diagnostics` "P_iso").
"""
function measure_postshock_plateau(profile; left_frac = 0.6, right_frac = 0.85)
    mask = (profile.x .>= left_frac) .& (profile.x .<= right_frac)
    if !any(mask)
        return (NaN, NaN, NaN)
    end
    rho_avg = sum(profile.rho[mask]) / count(mask)
    u_avg   = sum(profile.u[mask])   / count(mask)
    P_iso   = (profile.Pxx .+ 2 .* profile.Pp) ./ 3
    P_avg   = sum(P_iso[mask])  / count(mask)
    return (rho_avg, u_avg, P_avg)
end


"""
    rh_residuals(profile, M1; gamma_eos = GAMMA_LAW)

Analytical R-H downstream values vs measured plateau averages.
Returns NamedTuple of (analytical, measured, residual_abs,
residual_rel) per primitive.
"""
function rh_residuals(profile, M1; gamma_eos::Real = GAMMA_LAW)
    rho1 = 1.0; P1 = 1.0
    c_s1 = sqrt(gamma_eos * P1 / rho1)
    u1 = M1 * c_s1
    rho2_an, u2_an, P2_an, _ = rankine_hugoniot(rho1, u1, P1; gamma_eos = gamma_eos)
    rho2_m, u2_m, P2_m = measure_postshock_plateau(profile)
    return (
        rho_an = rho2_an, u_an = u2_an, P_an = P2_an,
        rho_m  = rho2_m,  u_m  = u2_m,  P_m  = P2_m,
        rho_rel = abs(rho2_m - rho2_an) / rho2_an,
        u_rel   = abs(u2_m   - u2_an)   / u2_an,
        P_rel   = abs(P2_m   - P2_an)   / P2_an,
    )
end


"""
    mach_scan(M_list = [1.5, 2.0, 3.0, 5.0, 10.0]; N = 200, t_end = 3.0,
              tau = 1e-3, q_kind = :none, kwargs...)

Run the Phase-7 integrator at each `M1 ∈ M_list`, collect profiles +
R-H residuals, return a Vector of `(M1, profile, summary, residuals)`
NamedTuples.
"""
function mach_scan(M_list = [1.5, 2.0, 3.0, 5.0, 10.0]; N::Int = 200,
                   t_end::Float64 = 3.0, tau::Float64 = 1e-3,
                   q_kind::Symbol = :none, kwargs...)
    results = []
    for M1 in M_list
        result = run_steady_shock(; M1 = float(M1), N = N, t_end = t_end,
                                  tau = tau, q_kind = q_kind, kwargs...)
        residuals = rh_residuals(result.profile, float(M1))
        push!(results, (; M1 = float(M1), result.profile, result.summary,
                        residuals))
    end
    return results
end


"""
    plot_mach_scan(results, out_path; q_label = "q-off")

Render a 4-panel plot of the Mach scan: density, velocity, P_iso,
and Q vs x for each M1 in `results`. Mirrors the layout of
`py-1d/experiments/03_steady_shock.py` Fig. 1 (top-left density, etc).
"""
function plot_mach_scan(results; out_path::AbstractString =
                        joinpath(@__DIR__, "..", "reference", "figs",
                                 "A3_steady_shock_mach_scan.png"),
                        q_label::String = "q-off")
    fig = Figure(size = (1100, 800))
    ax_rho = Axis(fig[1, 1], xlabel = "x", ylabel = "ρ",
                  title = "Density")
    ax_u   = Axis(fig[1, 2], xlabel = "x", ylabel = "u",
                  title = "Velocity")
    ax_P   = Axis(fig[2, 1], xlabel = "x", ylabel = "P_iso = (Pxx+2Pp)/3",
                  title = "Pressure (isotropic)")
    ax_aniso = Axis(fig[2, 2], xlabel = "x",
                    ylabel = "(Pxx − Pp)/P_iso",
                    title = "Pressure anisotropy")

    colors = cgrad(:viridis, length(results); categorical = true)
    for (k, r) in enumerate(results)
        prof = r.profile
        Piso = (prof.Pxx .+ 2 .* prof.Pp) ./ 3
        aniso = (prof.Pxx .- prof.Pp) ./ max.(Piso, 1e-12)
        lines!(ax_rho, prof.x, prof.rho, color = colors[k],
               label = "M=$(r.M1)", linewidth = 1.2)
        lines!(ax_u,   prof.x, prof.u,   color = colors[k], linewidth = 1.2)
        lines!(ax_P,   prof.x, Piso,     color = colors[k], linewidth = 1.2)
        lines!(ax_aniso, prof.x, aniso,  color = colors[k], linewidth = 1.2)
    end
    axislegend(ax_rho, position = :rt, fontsize = 9)

    Label(fig[0, :],
          "Tier A.3 steady shock Mach scan ($q_label, τ = 1e-3, N = $(length(results[1].profile.x)))",
          fontsize = 16)

    mkpath(dirname(out_path))
    save(out_path, fig)
    return out_path
end


"""
    print_mach_scan_table(results)

Print a Markdown-style table of R-H residuals for each Mach number.
"""
function print_mach_scan_table(results)
    println("\n| M1   | rho_an | rho_m | u_an   | u_m    | P_an   | P_m    | rho_rel | u_rel  | P_rel  |")
    println("|------|--------|-------|--------|--------|--------|--------|---------|--------|--------|")
    for r in results
        rh = r.residuals
        @printf("| %4.1f | %6.3f | %5.3f | %6.3f | %6.3f | %6.3f | %6.3f | %.4f | %.4f | %.4f |\n",
                r.M1, rh.rho_an, rh.rho_m, rh.u_an, rh.u_m, rh.P_an, rh.P_m,
                rh.rho_rel, rh.u_rel, rh.P_rel)
    end
    println()
end


"""
    main_a3_steady_shock(; q_kind = :none)

Standalone driver: run Mach scan at production resolution, print
R-H table, save plot, return results.
"""
function main_a3_steady_shock(; q_kind::Symbol = :none, N::Int = 200,
                              t_end::Float64 = 3.0)
    @info "Running A.3 steady-shock Mach scan" q_kind N t_end
    results = mach_scan([1.5, 2.0, 3.0, 5.0, 10.0]; N = N, t_end = t_end,
                        q_kind = q_kind)
    print_mach_scan_table(results)
    fig_path = plot_mach_scan(results;
                              q_label = q_kind == :none ? "q-off" : "q-on")
    @info "Wrote Mach scan figure" fig_path
    return results
end
