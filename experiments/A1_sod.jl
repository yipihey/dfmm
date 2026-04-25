# experiments/A1_sod.jl
#
# Tier A.1 — Sod shock tube on the variational integrator.
#
# Methods paper §10.2 A.1 / `reference/MILESTONE_1_PLAN.md` Phase 5.
# Compares the Julia variational integrator against py-1d's HDF5 golden
# (`reference/golden/sod.h5`, default τ = 1e-3, intermediate "warm"
# regime) at t_end = 0.2.
#
# Setup mapping. py-1d's Sod uses a *transmissive* boundary on
# `[0, 1]` with cell centers at `(i + 0.5)/N`. The variational
# integrator currently only supports periodic BC (Phase 2). We
# bridge the gap by **mirror-doubling** the IC onto a periodic
# Lagrangian box of length 2:
#
#     m = 0           1.0           2.0           3.0           4.0
#     x = 0           0.5           1.0           1.5           2.0
#     | rho=1.0     | rho=0.125 |  rho=0.125 |  rho=1.0    |
#     |  Sod L      |  Sod R    |  mirror R  |  mirror L   |
#
# By the mirror symmetry around `x = 1`, the two Sod problems run
# back-to-back and the periodic boundary at `x = 0 = 2` is locally
# smooth (rho = 1 on both sides, u = 0). Waves never "leak" between
# the two Sods if t · c_s < 0.5 — at t_end = 0.2 with c_s ≈ 1.3, the
# wave reach is ≈ 0.26 ≪ 0.5. We extract the comparison profile
# from the *first* Sod (cells with `0 ≤ x < 1`).
#
# The mirror-doubled mesh has **2N segments** for an N-cell golden
# comparison; the regression test asserts L∞ rel error on the first
# half against `golden.fields.rho[:, end]` (the snapshot at t_end).
#
# Three τ regimes (Option B per the Phase-5 brief):
# - τ = 1e-3 (intermediate, default; matches the golden).
# - τ = 1.0 and τ = 1e3 (collisionless): supplementary table in
#   `reference/notes_phase5_sod.md`. No committed regression.

using dfmm
using HDF5
using Printf
using CairoMakie

"""
    build_sod_mesh(ic; mirror = true) -> Mesh1D

Convert the `setup_sod()` IC NamedTuple into a `Mesh1D` ready for
`det_run!`. With `mirror = true`, doubles the IC into a periodic
length-2 box per the comments in this file's header. Returns the
mesh.

The Lagrangian segment masses are equal to the Eulerian dx times the
local density (so `Δm_j = ρ_j · dx`). Per-segment entropy `s_j` is
chosen so that `M_vv(J_j, s_j) = P_j/ρ_j` at the IC, which makes
`P_xx,init = ρ_j · M_vv = P_j` consistent with the IC. With
`Γ = 5/3, c_v = 1`:
    s_j = log(P_j / ρ_j^Γ) = log(P_j) - Γ log(ρ_j).

Initial `P_⊥ = P_j` (isotropic Maxwellian IC), `α_j = sigma_x0`,
`β_j = 0` (matches `setup_sod`'s defaults).
"""
function build_sod_mesh(ic; mirror::Bool = true)
    N0 = length(ic.rho)
    Γ = 5.0 / 3.0  # GAMMA_LAW_DEFAULT
    if mirror
        # Concatenate IC and its mirror image. The mirror runs from
        # x = 1 to x = 2 with rho/u/P/Pp reflected so that x = 1 and
        # x = 2 = 0 (periodic) are smooth.
        rho   = vcat(ic.rho,   reverse(ic.rho))
        u     = vcat(ic.u,     -reverse(ic.u))    # u flips on mirror
        P     = vcat(ic.P,     reverse(ic.P))
        Pp    = vcat(ic.Pp,    reverse(ic.Pp))
        αs    = vcat(ic.alpha_init, reverse(ic.alpha_init))
        βs    = vcat(ic.beta_init,  -reverse(ic.beta_init))  # β charge-1
        N = 2 * N0
        # Box length 2; cell width dx = 2/N = 1/N0. Cell centers:
        # x = (i + 0.5)/N0 for i = 0..N-1, but on a length-2 box.
        dx = 2.0 / N
        L_box = 2.0
    else
        # Single-Sod periodic on [0, 1]; introduces a spurious
        # boundary discontinuity at x = 0 = 1 (rho jumps from 0.125 to
        # 1). Useful as a sanity check against the mirror configuration.
        rho = copy(ic.rho); u = copy(ic.u); P = copy(ic.P); Pp = copy(ic.Pp)
        αs = copy(ic.alpha_init); βs = copy(ic.beta_init)
        N = N0
        dx = 1.0 / N
        L_box = 1.0
    end

    # Lagrangian masses: each cell holds Δm_j = ρ_j · dx.
    Δm = rho .* dx

    # Per-segment entropy: s_j = log(P_j/ρ_j^Γ).
    ss = log.(P ./ rho .^ Γ)

    # Vertex positions on the Lagrangian frame: x_j = sum_{k<j} Δx_k =
    # sum_{k<j} dx (since dx_k = dx is uniform here).
    positions = collect((0:N-1) .* dx)
    velocities = u

    return dfmm.Mesh1D(positions, velocities, αs, βs, ss;
                       Δm = Δm, Pps = Pp, L_box = L_box, periodic = true)
end

"""
    extract_eulerian_profiles(mesh; n_eulerian = nothing) -> NamedTuple

Sample the (segment-centered) primitives `(rho, u, Pxx, Pp, P_iso, x)`
from the mesh's current state. Returns Julia column vectors of length
`n_segments(mesh)`. The `x` field is the segment center (midpoint
between the segment's left and right vertex), useful for matching the
golden's `(i+0.5)/N` cell centers.

For mirror-doubled meshes the caller can take the first-N0 slice
to compare against the original golden grid:
    `prof = extract_eulerian_profiles(mesh)`
    `rho_first_half = prof.rho[1:N0]`
"""
function extract_eulerian_profiles(mesh)
    N = dfmm.n_segments(mesh)
    rho   = zeros(Float64, N)
    u     = zeros(Float64, N)
    Pxx   = zeros(Float64, N)
    Pp    = zeros(Float64, N)
    Piso  = zeros(Float64, N)
    xc    = zeros(Float64, N)
    α_arr = zeros(Float64, N)
    β_arr = zeros(Float64, N)
    γ_arr = zeros(Float64, N)
    @inbounds for j in 1:N
        seg = mesh.segments[j]
        ρ = dfmm.segment_density(mesh, j)
        J = 1 / ρ
        Mvv_j = dfmm.Mvv(J, seg.state.s)
        rho[j] = ρ
        # Velocity at cell center = average of left/right vertex velocities.
        j_right = j == N ? 1 : j + 1
        u_left  = seg.state.u
        u_right = mesh.segments[j_right].state.u
        u[j]   = (u_left + u_right) / 2
        Pxx[j] = ρ * Mvv_j
        Pp[j]  = seg.state.Pp
        Piso[j] = (Pxx[j] + 2 * Pp[j]) / 3
        # Cell center: average of left/right vertex positions, with
        # periodic wrap on the last segment.
        x_left = seg.state.x
        wrap = (j == N) ? mesh.L_box : 0.0
        x_right = mesh.segments[j_right].state.x + wrap
        xc[j] = (x_left + x_right) / 2
        α_arr[j] = seg.state.α
        β_arr[j] = seg.state.β
        γ_arr[j] = sqrt(max(Mvv_j - seg.state.β^2, 0.0))
    end
    return (; rho, u, Pxx, Pp, Piso, x = xc, α = α_arr, β = β_arr, γ = γ_arr)
end


"""
    run_sod(; N=400, t_end=0.2, tau=1e-3, sigma_x0=0.02, mirror=true,
              cfl=0.3, n_steps=nothing, dt=nothing, log_stride=nothing,
              verbose=true)

Run the variational integrator on the Sod IC.

Either `dt` (with `n_steps`) or just `t_end` (with internal CFL choice)
must be supplied. By default we choose `dt` so that the run reaches
`t_end` in an integer number of steps with `dt ≤ cfl · dx / c_s_max`,
matching py-1d's CFL convention.

Returns a NamedTuple `(mesh, profile, summary)` where:
- `mesh` is the final Mesh1D state.
- `profile` is `extract_eulerian_profiles(mesh)`.
- `summary` carries (t_end, n_steps, dt, mass_err, mom_err, energy_change,
  wall_seconds, alphamax, gammamin, Pi_max).

If `mirror = true`, the run is on the doubled mesh; the user should
slice `profile.rho[1:N]` etc. to compare against the golden.
"""
function run_sod(; N::Int = 400, t_end::Float64 = 0.2,
                 tau::Float64 = 1e-3, sigma_x0::Float64 = 0.02,
                 mirror::Bool = true, cfl::Float64 = 0.3,
                 dt::Union{Float64,Nothing} = nothing,
                 n_steps::Union{Int,Nothing} = nothing,
                 log_stride::Union{Int,Nothing} = nothing,
                 verbose::Bool = true)
    ic = setup_sod(; N = N, t_end = t_end, sigma_x0 = sigma_x0, tau = tau)
    mesh = build_sod_mesh(ic; mirror = mirror)
    N_seg = dfmm.n_segments(mesh)
    Γ = 5.0 / 3.0

    # Initial diagnostics.
    mass0 = dfmm.total_mass(mesh)
    mom0  = dfmm.total_momentum(mesh)
    KE0   = dfmm.total_kinetic_energy(mesh)
    Eint0 = dfmm.total_internal_energy(mesh)
    Etot0 = KE0 + Eint0

    # CFL-bound dt: c_s_max from the IC, dx = (mirror ? 2 : 1)/N_seg.
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
        @info "Sod run setup" N=N_seg dt dt_cfl n_steps τ=tau mirror
    end

    # Integrate.
    t0 = time()
    αmax = 0.0; γ²min = Inf; Π_max = 0.0
    if log_stride === nothing
        log_stride = max(1, n_steps ÷ 5)
    end
    for n in 1:n_steps
        dfmm.det_step!(mesh, dt; tau = tau)
        # Per-step diagnostics (cheap; collect occasionally).
        if n % log_stride == 0 || n == n_steps
            for j in 1:N_seg
                seg = mesh.segments[j]
                αmax = max(αmax, seg.state.α)
                ρ = dfmm.segment_density(mesh, j)
                J_loc = 1 / ρ
                Mvv_j = dfmm.Mvv(J_loc, seg.state.s)
                γ² = Mvv_j - seg.state.β^2
                γ²min = min(γ²min, γ²)
                Π = ρ * Mvv_j - seg.state.Pp
                Π_max = max(Π_max, abs(Π))
            end
            verbose && @info "Sod step" n n_steps t=n*dt αmax=αmax γ²min=γ²min Π_max=Π_max
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
    compare_sod_to_golden(result; verbose = true) -> NamedTuple

Compute L∞ relative errors of the Julia integrator's profile against
the py-1d golden at `t_end`. Returns a NamedTuple of per-field
errors and the golden's data for plotting.

Sliced to the first half of the mesh if `mirror` was true.
"""
function compare_sod_to_golden(result; verbose::Bool = true)
    golden = dfmm.load_tier_a_golden(:sod)
    N0 = result.N0
    # Profile from the first Sod copy on the doubled mesh.
    if result.mirror
        rho_jl = result.profile.rho[1:N0]
        u_jl   = result.profile.u[1:N0]
        Pxx_jl = result.profile.Pxx[1:N0]
        Pp_jl  = result.profile.Pp[1:N0]
        x_jl   = result.profile.x[1:N0]
    else
        rho_jl = result.profile.rho
        u_jl   = result.profile.u
        Pxx_jl = result.profile.Pxx
        Pp_jl  = result.profile.Pp
        x_jl   = result.profile.x
    end

    # Golden snapshot at t_end is the last column.
    rho_g = golden.fields.rho[:, end]
    u_g   = golden.fields.u[:, end]
    Pxx_g = golden.fields.Pxx[:, end]
    Pp_g  = golden.fields.Pp[:, end]
    x_g   = golden.x

    # L∞ relative errors, normalized by max of the golden field's
    # absolute value to avoid 0-division near zero crossings of u.
    function linf_rel(a, b)
        scale = max(maximum(abs, b), 1e-12)
        return maximum(abs, a .- b) / scale
    end

    err_rho = linf_rel(rho_jl, rho_g)
    err_u   = linf_rel(u_jl,   u_g)
    err_Pxx = linf_rel(Pxx_jl, Pxx_g)
    err_Pp  = linf_rel(Pp_jl,  Pp_g)

    if verbose
        @info "Sod vs golden L∞ rel errors" err_rho err_u err_Pxx err_Pp
    end

    return (; err_rho, err_u, err_Pxx, err_Pp,
            x_jl, rho_jl, u_jl, Pxx_jl, Pp_jl,
            x_g, rho_g, u_g, Pxx_g, Pp_g)
end


"""
    plot_sod_comparison(cmp; out_path)

Render a 2x2 comparison plot of (ρ, u, Pxx, Pp) Julia vs golden at
t_end. Writes PNG to `out_path` (default `reference/figs/A1_sod_profiles.png`).
"""
function plot_sod_comparison(cmp; out_path::AbstractString =
                             joinpath(@__DIR__, "..", "reference", "figs",
                                      "A1_sod_profiles.png"))
    fig = Figure(size = (1100, 800))
    ax_rho = Axis(fig[1, 1], xlabel = "x", ylabel = "ρ", title = "Density")
    lines!(ax_rho, cmp.x_g, cmp.rho_g, color = :black, label = "py-1d golden", linewidth = 1.2)
    lines!(ax_rho, cmp.x_jl, cmp.rho_jl, color = :red, label = "Julia variational", linestyle = :dash, linewidth = 1.2)
    axislegend(ax_rho, position = :rt)

    ax_u = Axis(fig[1, 2], xlabel = "x", ylabel = "u", title = "Velocity")
    lines!(ax_u, cmp.x_g, cmp.u_g, color = :black, linewidth = 1.2)
    lines!(ax_u, cmp.x_jl, cmp.u_jl, color = :red, linestyle = :dash, linewidth = 1.2)

    ax_Pxx = Axis(fig[2, 1], xlabel = "x", ylabel = "Pxx", title = "Parallel pressure")
    lines!(ax_Pxx, cmp.x_g, cmp.Pxx_g, color = :black, linewidth = 1.2)
    lines!(ax_Pxx, cmp.x_jl, cmp.Pxx_jl, color = :red, linestyle = :dash, linewidth = 1.2)

    ax_Pp = Axis(fig[2, 2], xlabel = "x", ylabel = "Pp", title = "Perpendicular pressure")
    lines!(ax_Pp, cmp.x_g, cmp.Pp_g, color = :black, linewidth = 1.2)
    lines!(ax_Pp, cmp.x_jl, cmp.Pp_jl, color = :red, linestyle = :dash, linewidth = 1.2)

    Label(fig[0, :], "Tier A.1 Sod: variational integrator vs py-1d golden (t = 0.2, τ = 1e-3)",
          fontsize = 16)

    mkpath(dirname(out_path))
    save(out_path, fig)
    return out_path
end


"""
    save_sod_h5(result; out_path)

Write the integrator's final mesh state as an HDF5 snapshot for later
inspection. Schema mirrors the golden's `/fields/*` for ease of
diffing.
"""
function save_sod_h5(result; out_path::AbstractString =
                     joinpath(@__DIR__, "..", "reference", "figs",
                              "A1_sod_julia.h5"))
    mkpath(dirname(out_path))
    h5open(out_path, "w") do f
        attrs(f)["t_end"]  = result.summary.t_end
        attrs(f)["n_steps"] = result.summary.n_steps
        attrs(f)["dt"]     = result.summary.dt
        attrs(f)["N0"]     = result.N0
        attrs(f)["mirror"] = result.mirror ? 1 : 0
        f["x"]   = result.profile.x
        f["rho"] = result.profile.rho
        f["u"]   = result.profile.u
        f["Pxx"] = result.profile.Pxx
        f["Pp"]  = result.profile.Pp
        f["alpha"] = result.profile.α
        f["beta"]  = result.profile.β
        f["gamma"] = result.profile.γ
    end
    return out_path
end


# Standalone driver: run at production resolution and write artefacts.
function main_a1_sod()
    @info "Running A.1 Sod (warm) at production resolution..."
    result = run_sod(; N = 400, t_end = 0.2, tau = 1e-3,
                     mirror = true, verbose = true)
    @info "Run summary" result.summary
    cmp = compare_sod_to_golden(result; verbose = true)
    @info "Comparison" cmp.err_rho cmp.err_u cmp.err_Pxx cmp.err_Pp
    fig_path = plot_sod_comparison(cmp)
    h5_path  = save_sod_h5(result)
    @info "Wrote artefacts" fig_path h5_path
    return (; result, cmp)
end
