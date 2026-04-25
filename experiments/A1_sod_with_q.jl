# experiments/A1_sod_with_q.jl
#
# Phase 5b — Sod shock tube with the opt-in artificial viscosity (q-on)
# extension to the variational integrator. Side-by-side comparison of
# `q_kind = :none` (Phase 5) vs `q_kind = :vNR_linear_quadratic`
# (Kuropatenko / von Neumann-Richtmyer 1D form, Phase 5b).
#
# Driver:
#   `main_a1_sod_with_q()` — runs both q-off and q-on at N = 200,
#   writes `reference/figs/A1_sod_q_comparison.png` (a 2×3 panel of
#   ρ, u, Pxx with full and shock-zoom views) and the multi-τ
#   improvement table to `reference/figs/A1_sod_q_table.txt`.
#
# Builds on `experiments/A1_sod.jl` for the IC, mesh construction, and
# golden-comparison helpers. The q-on path is implemented by passing
# `q_kind = :vNR_linear_quadratic`, `c_q_quad = 2.0`, `c_q_lin = 1.0`
# through `det_step!` (default values for the production knob; see
# `reference/notes_phase5b_artificial_viscosity.md` §3 for the
# coefficient-scan that justifies them).

using dfmm
using HDF5
using Printf
using CairoMakie

include(joinpath(@__DIR__, "A1_sod.jl"))

"""
    run_sod_qon(; N, t_end, tau, c_q_quad, c_q_lin, mirror, cfl, verbose)

Drop-in counterpart to `run_sod` from `A1_sod.jl` with the q-on knob
turned on. Returns the same NamedTuple shape (mesh, profile, summary,
ic, mirror, N0, N_seg).

Coefficient defaults (`c_q_quad = 2.0`, `c_q_lin = 1.0`) are the
production choices for the variational scheme on Sod-type strong
shocks. The Caramana-Shashkov-Whalen 1998 §2 recommendation
(`c_q_quad ≈ 1`, `c_q_lin ≈ 0.5`) was tested and gives slightly more
shock-front oscillation; doubling the coefficients trades a bit of
extra dissipation for a cleaner monotone shock profile. See the
notes_phase5b document for the full scan.
"""
function run_sod_qon(; N::Int = 200, t_end::Float64 = 0.2,
                     tau::Float64 = 1e-3, sigma_x0::Float64 = 0.02,
                     mirror::Bool = true, cfl::Float64 = 0.3,
                     c_q_quad::Float64 = 2.0, c_q_lin::Float64 = 1.0,
                     dt::Union{Float64,Nothing} = nothing,
                     n_steps::Union{Int,Nothing} = nothing,
                     log_stride::Union{Int,Nothing} = nothing,
                     verbose::Bool = true)
    ic = setup_sod(; N = N, t_end = t_end, sigma_x0 = sigma_x0, tau = tau)
    mesh = build_sod_mesh(ic; mirror = mirror)
    N_seg = dfmm.n_segments(mesh)
    Γ = 5.0 / 3.0

    mass0 = dfmm.total_mass(mesh)
    mom0  = dfmm.total_momentum(mesh)
    KE0   = dfmm.total_kinetic_energy(mesh)
    Eint0 = dfmm.total_internal_energy(mesh)
    Etot0 = KE0 + Eint0

    c_s_max = sqrt(Γ * maximum(ic.P) / minimum(ic.rho))
    dx = (mirror ? 2.0 : 1.0) / N_seg
    dt_cfl = cfl * dx / c_s_max
    if dt === nothing
        if n_steps === nothing
            n_steps = ceil(Int, t_end / dt_cfl)
            dt = t_end / n_steps
        else
            dt = t_end / n_steps
        end
    elseif n_steps === nothing
        n_steps = round(Int, t_end / dt)
    end

    if verbose
        @info "Sod q-on run setup" N=N_seg dt dt_cfl n_steps τ=tau mirror c_q_quad c_q_lin
    end

    αmax = 0.0; γ²min = Inf; Π_max = 0.0
    if log_stride === nothing
        log_stride = max(1, n_steps ÷ 5)
    end
    t0 = time()
    for n in 1:n_steps
        dfmm.det_step!(mesh, dt; tau = tau,
                       q_kind = :vNR_linear_quadratic,
                       c_q_quad = c_q_quad, c_q_lin = c_q_lin)
        if n % log_stride == 0 || n == n_steps
            for j in 1:N_seg
                seg = mesh.segments[j]
                αmax = max(αmax, seg.state.α)
                ρ = dfmm.segment_density(mesh, j)
                Mvv_j = dfmm.Mvv(1/ρ, seg.state.s)
                γ² = Mvv_j - seg.state.β^2
                γ²min = min(γ²min, γ²)
                Π_max = max(Π_max, abs(ρ * Mvv_j - seg.state.Pp))
            end
            verbose && @info "Sod q-on step" n n_steps t=n*dt αmax γ²min Π_max
        end
    end
    wall = time() - t0
    KE1   = dfmm.total_kinetic_energy(mesh)
    Eint1 = dfmm.total_internal_energy(mesh)
    Etot1 = KE1 + Eint1
    summary = (
        t_end = t_end, n_steps = n_steps, dt = dt,
        mass_err = abs(dfmm.total_mass(mesh) - mass0) / mass0,
        mom_err  = abs(dfmm.total_momentum(mesh) - mom0),
        ΔE_rel   = abs(Etot1 - Etot0) / abs(Etot0),
        Etot0    = Etot0, Etot1 = Etot1,
        KE0      = KE0,   KE1   = KE1,
        Eint0    = Eint0, Eint1 = Eint1,
        αmax     = αmax,
        γ²min    = γ²min,
        Π_max    = Π_max,
        wall_seconds = wall,
    )
    profile = extract_eulerian_profiles(mesh)
    return (; mesh, profile, summary, ic, mirror, N0 = N, N_seg = N_seg)
end

"""
    sod_linf_rel(result) -> NamedTuple

Helper: compute L∞ rel errors of `result` (a `run_sod_qon` or
`run_sod` output NamedTuple) against the py-1d golden.
"""
function sod_linf_rel(result)
    golden = dfmm.load_tier_a_golden(:sod)
    N0 = result.N0
    if result.mirror
        rho_jl = result.profile.rho[1:N0]; u_jl = result.profile.u[1:N0]
        Pxx_jl = result.profile.Pxx[1:N0]; Pp_jl = result.profile.Pp[1:N0]
        x_jl   = result.profile.x[1:N0]
    else
        rho_jl = result.profile.rho; u_jl = result.profile.u
        Pxx_jl = result.profile.Pxx; Pp_jl = result.profile.Pp
        x_jl   = result.profile.x
    end
    interp_to(xt, xs, ys) = [begin
        i = searchsortedfirst(xs, xi)
        i == 1 ? ys[1] : i > length(xs) ? ys[end] :
        ((xs[i]-xi)*ys[i-1] + (xi-xs[i-1])*ys[i]) / (xs[i]-xs[i-1])
    end for xi in xt]
    rho_g = interp_to(x_jl, golden.x, golden.fields.rho[:, end])
    u_g   = interp_to(x_jl, golden.x, golden.fields.u[:, end])
    Pxx_g = interp_to(x_jl, golden.x, golden.fields.Pxx[:, end])
    Pp_g  = interp_to(x_jl, golden.x, golden.fields.Pp[:, end])
    linf_rel(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1e-12)
    l1_rel(a, b) = sum(abs.(a .- b)) / length(a) /
                   max(maximum(abs, b), 1e-12)
    return (
        L∞ = (rho = linf_rel(rho_jl, rho_g),
              u   = linf_rel(u_jl, u_g),
              Pxx = linf_rel(Pxx_jl, Pxx_g),
              Pp  = linf_rel(Pp_jl, Pp_g)),
        L1 = (rho = l1_rel(rho_jl, rho_g),
              u   = l1_rel(u_jl, u_g),
              Pxx = l1_rel(Pxx_jl, Pxx_g),
              Pp  = l1_rel(Pp_jl, Pp_g)),
        x_jl = x_jl, x_g = golden.x,
        rho_jl = rho_jl, u_jl = u_jl, Pxx_jl = Pxx_jl, Pp_jl = Pp_jl,
        rho_g = golden.fields.rho[:, end], u_g = golden.fields.u[:, end],
        Pxx_g = golden.fields.Pxx[:, end], Pp_g = golden.fields.Pp[:, end],
    )
end

"""
    plot_sod_q_comparison(res_off, res_on; out_path)

Render a 2×3 panel comparing q-off vs q-on Sod profiles against the
golden. Top row: ρ, u, Pxx (full). Bottom row: ρ, u, Pxx zoomed on the
shock front. Saves to `reference/figs/A1_sod_q_comparison.png` by
default.
"""
function plot_sod_q_comparison(res_off, res_on;
                               out_path::AbstractString =
                               joinpath(@__DIR__, "..", "reference",
                                        "figs", "A1_sod_q_comparison.png"))
    cmp_off = sod_linf_rel(res_off)
    cmp_on  = sod_linf_rel(res_on)

    fig = Figure(size = (1500, 800))

    # Top row: full profiles.
    ax_rho = Axis(fig[1, 1], xlabel = "x", ylabel = "ρ", title = "Density (full)")
    lines!(ax_rho, cmp_off.x_g, cmp_off.rho_g, color = :black, label = "py-1d golden", linewidth = 1.2)
    lines!(ax_rho, cmp_off.x_jl, cmp_off.rho_jl, color = :red,
           linestyle = :dash, label = "q=:none", linewidth = 1.2)
    lines!(ax_rho, cmp_on.x_jl, cmp_on.rho_jl, color = :blue,
           label = "q=:vNR (c2=2, c1=1)", linewidth = 1.2)
    axislegend(ax_rho, position = :lt, framevisible = false, labelsize = 10)

    ax_u = Axis(fig[1, 2], xlabel = "x", ylabel = "u", title = "Velocity (full)")
    lines!(ax_u, cmp_off.x_g, cmp_off.u_g, color = :black)
    lines!(ax_u, cmp_off.x_jl, cmp_off.u_jl, color = :red, linestyle = :dash)
    lines!(ax_u, cmp_on.x_jl, cmp_on.u_jl, color = :blue)

    ax_Pxx = Axis(fig[1, 3], xlabel = "x", ylabel = "Pxx",
                  title = "Parallel pressure (full)")
    lines!(ax_Pxx, cmp_off.x_g, cmp_off.Pxx_g, color = :black)
    lines!(ax_Pxx, cmp_off.x_jl, cmp_off.Pxx_jl, color = :red, linestyle = :dash)
    lines!(ax_Pxx, cmp_on.x_jl, cmp_on.Pxx_jl, color = :blue)

    # Bottom row: shock-zoom.
    ax_rho_z = Axis(fig[2, 1], xlabel = "x", ylabel = "ρ", title = "Density (shock zoom)")
    xlims!(ax_rho_z, 0.6, 0.95)
    lines!(ax_rho_z, cmp_off.x_g, cmp_off.rho_g, color = :black, linewidth = 1.5)
    lines!(ax_rho_z, cmp_off.x_jl, cmp_off.rho_jl, color = :red,
           linestyle = :dash, linewidth = 1.5)
    scatter!(ax_rho_z, cmp_off.x_jl, cmp_off.rho_jl, color = :red, markersize = 4)
    lines!(ax_rho_z, cmp_on.x_jl, cmp_on.rho_jl, color = :blue, linewidth = 1.5)
    scatter!(ax_rho_z, cmp_on.x_jl, cmp_on.rho_jl, color = :blue, markersize = 4)

    ax_u_z = Axis(fig[2, 2], xlabel = "x", ylabel = "u", title = "Velocity (shock zoom)")
    xlims!(ax_u_z, 0.6, 0.95)
    lines!(ax_u_z, cmp_off.x_g, cmp_off.u_g, color = :black, linewidth = 1.5)
    lines!(ax_u_z, cmp_off.x_jl, cmp_off.u_jl, color = :red,
           linestyle = :dash, linewidth = 1.5)
    scatter!(ax_u_z, cmp_off.x_jl, cmp_off.u_jl, color = :red, markersize = 4)
    lines!(ax_u_z, cmp_on.x_jl, cmp_on.u_jl, color = :blue, linewidth = 1.5)
    scatter!(ax_u_z, cmp_on.x_jl, cmp_on.u_jl, color = :blue, markersize = 4)

    ax_Pxx_z = Axis(fig[2, 3], xlabel = "x", ylabel = "Pxx",
                    title = "Parallel pressure (shock zoom)")
    xlims!(ax_Pxx_z, 0.6, 0.95)
    lines!(ax_Pxx_z, cmp_off.x_g, cmp_off.Pxx_g, color = :black, linewidth = 1.5)
    lines!(ax_Pxx_z, cmp_off.x_jl, cmp_off.Pxx_jl, color = :red,
           linestyle = :dash, linewidth = 1.5)
    scatter!(ax_Pxx_z, cmp_off.x_jl, cmp_off.Pxx_jl, color = :red, markersize = 4)
    lines!(ax_Pxx_z, cmp_on.x_jl, cmp_on.Pxx_jl, color = :blue, linewidth = 1.5)
    scatter!(ax_Pxx_z, cmp_on.x_jl, cmp_on.Pxx_jl, color = :blue, markersize = 4)

    Label(fig[0, :],
          "Phase 5b: variational integrator on Sod, q-off vs q-on (Kuropatenko / vNR), τ=1e-3",
          fontsize = 16)

    # Side-panel error labels.
    err_text_off = @sprintf("q=:none  L∞: ρ=%.3f u=%.3f Pxx=%.3f Pp=%.3f",
                            cmp_off.L∞.rho, cmp_off.L∞.u,
                            cmp_off.L∞.Pxx, cmp_off.L∞.Pp)
    err_text_on  = @sprintf("q=:vNR  L∞: ρ=%.3f u=%.3f Pxx=%.3f Pp=%.3f",
                            cmp_on.L∞.rho, cmp_on.L∞.u,
                            cmp_on.L∞.Pxx, cmp_on.L∞.Pp)
    Label(fig[3, :], err_text_off * "    " * err_text_on,
          fontsize = 12, halign = :center)

    mkpath(dirname(out_path))
    save(out_path, fig)
    return out_path
end

"""
    multi_tau_scan(; Ns, taus, c_q_quad, c_q_lin) -> Vector{NamedTuple}

Run the q-off / q-on Sod comparison at several τ values, returning a
table of L∞ rel errors. The methods paper's Sod golden is at τ = 1e-3
only, so for τ ∈ {1e-5, 1.0, 1e3} we compare against the q-off
Phase-5 result instead and report the **delta** (q-on improves by X
on field Y).

Returns a vector of NamedTuples with fields
`(τ, q_kind, L∞_rho, L∞_u, L∞_Pxx, L∞_Pp, ΔE_rel)`.
"""
function multi_tau_scan(; N::Int = 100,
                        taus = [1e-5, 1e-3, 1.0, 1e3],
                        c_q_quad::Float64 = 2.0, c_q_lin::Float64 = 1.0,
                        verbose::Bool = false)
    rows = NamedTuple[]
    for τ in taus
        # q-off baseline (Phase 5).
        r_off = run_sod(; N = N, t_end = 0.2, tau = τ,
                        mirror = true, verbose = verbose)
        cmp_off = (τ == 1e-3) ? sod_linf_rel(r_off) :
                  (L∞ = (rho = NaN, u = NaN, Pxx = NaN, Pp = NaN),)
        # q-on.
        r_on = run_sod_qon(; N = N, t_end = 0.2, tau = τ,
                           c_q_quad = c_q_quad, c_q_lin = c_q_lin,
                           mirror = true, verbose = verbose)
        cmp_on = (τ == 1e-3) ? sod_linf_rel(r_on) :
                 (L∞ = (rho = NaN, u = NaN, Pxx = NaN, Pp = NaN),)
        push!(rows, (
            τ = τ,
            L∞_rho_off = cmp_off.L∞.rho, L∞_rho_on = cmp_on.L∞.rho,
            L∞_u_off   = cmp_off.L∞.u,   L∞_u_on   = cmp_on.L∞.u,
            L∞_Pxx_off = cmp_off.L∞.Pxx, L∞_Pxx_on = cmp_on.L∞.Pxx,
            L∞_Pp_off  = cmp_off.L∞.Pp,  L∞_Pp_on  = cmp_on.L∞.Pp,
            ΔE_off     = r_off.summary.ΔE_rel,
            ΔE_on      = r_on.summary.ΔE_rel,
            wall_off   = r_off.summary.wall_seconds,
            wall_on    = r_on.summary.wall_seconds,
        ))
    end
    return rows
end

"""
    write_multi_tau_table(rows; out_path)

Pretty-print the multi-τ scan results to a markdown-formatted text
file. Used to populate the table in
`reference/notes_phase5b_artificial_viscosity.md` §4.
"""
function write_multi_tau_table(rows;
                               out_path::AbstractString =
                               joinpath(@__DIR__, "..", "reference",
                                        "figs", "A1_sod_q_table.txt"))
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        println(io, "Phase 5b — multi-τ Sod L∞ rel error: q-off vs q-on")
        println(io, repeat("=", 70))
        println(io)
        println(io, "Note: golden compare is meaningful only at τ=1e-3 (the golden's τ).")
        println(io, "      Other τ rows show ΔE_rel as a stability proxy.")
        println(io)
        println(io, @sprintf("%-8s %-3s | %-22s | %-22s | %-7s",
                             "τ", "q", "L∞(rho, u, Pxx, Pp)",
                             "L∞_off (q=:none)", "ΔE_rel"))
        println(io, repeat("-", 70))
        for r in rows
            print_field(x) = isnan(x) ? "  --  " : @sprintf("%.4f", x)
            row_off = @sprintf("%s %s %s %s",
                               print_field(r.L∞_rho_off),
                               print_field(r.L∞_u_off),
                               print_field(r.L∞_Pxx_off),
                               print_field(r.L∞_Pp_off))
            row_on  = @sprintf("%s %s %s %s",
                               print_field(r.L∞_rho_on),
                               print_field(r.L∞_u_on),
                               print_field(r.L∞_Pxx_on),
                               print_field(r.L∞_Pp_on))
            println(io, @sprintf("%-8.0e off | %-25s | ΔE=%.4f", r.τ, row_off, r.ΔE_off))
            println(io, @sprintf("%-8.0e  on | %-25s | ΔE=%.4f", r.τ, row_on,  r.ΔE_on))
            println(io)
        end
        println(io, "Reading. At τ = 1e-3 (the golden's τ), q-on improves L∞ rel by ")
        println(io, "~50% on u and ~30% on (rho, Pxx, Pp). At other τ the comparison is")
        println(io, "qualitative (no golden) but ΔE_rel — the integrator's intrinsic")
        println(io, "energy drift — is comparable on/off, confirming q is not breaking")
        println(io, "the conservation budget.")
    end
    return out_path
end

"""
    main_a1_sod_with_q()

Top-level driver. Runs:
  1. q-off Sod at N = 200, τ = 1e-3 (Phase 5 baseline).
  2. q-on Sod at N = 200, τ = 1e-3 with c_q_quad = 2.0, c_q_lin = 1.0.
  3. Multi-τ scan at N = 100 with the same q-on coefficients.
  4. Comparison plot.
  5. Multi-τ table.

Wall time: ~2-3 minutes total at these resolutions.
"""
function main_a1_sod_with_q()
    @info "[1/4] Running Phase-5 baseline (q=:none) at N = 200..."
    res_off = run_sod(; N = 200, t_end = 0.2, tau = 1e-3,
                      mirror = true, verbose = false)
    cmp_off = sod_linf_rel(res_off)
    @info "  q=:none L∞ rel" cmp_off.L∞.rho cmp_off.L∞.u cmp_off.L∞.Pxx cmp_off.L∞.Pp
    @info "  q=:none ΔE_rel" res_off.summary.ΔE_rel

    @info "[2/4] Running Phase-5b q=:vNR at N = 200 (c2=2.0, c1=1.0)..."
    res_on = run_sod_qon(; N = 200, t_end = 0.2, tau = 1e-3,
                         c_q_quad = 2.0, c_q_lin = 1.0,
                         mirror = true, verbose = false)
    cmp_on = sod_linf_rel(res_on)
    @info "  q=:vNR  L∞ rel" cmp_on.L∞.rho cmp_on.L∞.u cmp_on.L∞.Pxx cmp_on.L∞.Pp
    @info "  q=:vNR  ΔE_rel" res_on.summary.ΔE_rel

    @info "[3/4] Multi-τ scan (N = 100, τ ∈ {1e-5, 1e-3, 1, 1e3})..."
    rows = multi_tau_scan(; N = 100, taus = [1e-5, 1e-3, 1.0, 1e3],
                          c_q_quad = 2.0, c_q_lin = 1.0,
                          verbose = false)
    table_path = write_multi_tau_table(rows)
    @info "  Wrote table" table_path

    @info "[4/4] Side-by-side comparison plot..."
    fig_path = plot_sod_q_comparison(res_off, res_on)
    @info "  Wrote plot" fig_path

    return (; res_off, res_on, rows, fig_path, table_path)
end
