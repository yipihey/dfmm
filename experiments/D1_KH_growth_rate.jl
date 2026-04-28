# D1_KH_growth_rate.jl
#
# §10.5 D.1 Kelvin-Helmholtz falsifier driver — M3-6 Phase 1c
# (`reference/notes_M3_6_phase1c_D1_kh_falsifier.md`).
#
# Drives the M3-6 Phase 1b `tier_d_kh_ic_full` IC through
# `det_step_2d_berry_HG!` (with M3-6 Phase 1a's strain coupling
# active and M3-6 Phase 1b's 4-component realizability cone wired
# in) at three refinement levels, fits the linear growth rate
# γ_measured to the antisymmetric tilt-mode amplitude
# `δβ_12(t)`, and compares to the classical Drazin-Reid prediction
# γ_DR = U / (2 w):
#
#   • Linear-regime fit: log|⟨|δβ_12(t)|⟩| ~ γ_measured · t + const,
#     extracted by least-squares over the linear window
#     [t_window_lo, t_window_hi].
#   • c_off² calibration: c_off = γ_measured / γ_DR; c_off² is the
#     methods paper §10.5 D.1 prediction value.
#   • Per-axis γ diagnostic: γ_1 (unstable) develops spatial
#     structure; γ_2 (transverse) stays roughly uniform.
#   • 4-component realizability stats: n_offdiag_events,
#     n_negative_jacobian (per-leaf det Hess sign over time).
#
# Saves time series + growth-rate fit + 4-comp realizability
# diagnostics to HDF5 at
# `reference/figs/M3_6_phase1_D1_kh_growth_rate.h5` (caller-controlled).
#
# Usage (REPL):
#
#   julia> include("experiments/D1_KH_growth_rate.jl")
#   julia> result = run_D1_KH_growth_rate(; level=5)
#
# Or full mesh-refinement battery + headline plot:
#
#   julia> include("experiments/D1_KH_growth_rate.jl")
#   julia> sweep = run_D1_KH_mesh_sweep(; levels=(4, 5, 6))
#   julia> plot_D1_KH_growth_rate(sweep;
#              save_path="reference/figs/M3_6_phase1_D1_kh_growth_rate.png")
#
# Returns a NamedTuple with the trajectory + γ fit + per-axis γ
# diagnostic + ProjectionStats / negative-Jacobian diagnostic for
# use by `test/test_M3_6_phase1c_D1_kh_growth_rate.jl`.

using HierarchicalGrids: HierarchicalMesh, EulerianFrame,
    refine_cells!, enumerate_leaves, FrameBoundaries, REFLECTING, PERIODIC,
    cell_physical_box
using dfmm: tier_d_kh_ic_full, allocate_cholesky_2d_fields,
    write_detfield_2d!, read_detfield_2d, DetField2D,
    det_step_2d_berry_HG!, gamma_per_axis_2d_field,
    ProjectionStats
using Statistics: mean, std

"""
    drazin_reid_gamma(; U_jet=1.0, jet_width=0.15) -> Float64

Classical Drazin-Reid linear growth rate for the tanh shear layer:

    γ_DR = U_jet / (2 · jet_width).

This is the dominant-mode growth rate for the inviscid 2D
Kelvin-Helmholtz instability; the methods paper §10.5 D.1
predicts the dfmm scheme reproduces this rate up to an O(1)
correction `c_off`.
"""
drazin_reid_gamma(; U_jet::Real = 1.0, jet_width::Real = 0.15) =
    Float64(U_jet) / (2.0 * Float64(jet_width))

"""
    fit_linear_growth_rate(t::AbstractVector, A::AbstractVector;
                           t_window_lo, t_window_hi) -> NamedTuple

Least-squares fit `log|A(t)| = γ · t + b` over the linear window
`[t_window_lo, t_window_hi]`. Returns `(γ, b, n_pts, t_lo, t_hi,
log_A_residuals)`.

Robust against `A == 0` cells (filtered in the log) and against
negative `A` (we fit `log|A|`, treating amplitude magnitude as
positive). The window is the *linear* phase before saturation:
typically `[0.2 / γ_DR, 0.8 / γ_DR]` is a good default but the
caller picks based on the trajectory shape.
"""
function fit_linear_growth_rate(t::AbstractVector, A::AbstractVector;
                                t_window_lo::Real, t_window_hi::Real)
    @assert length(t) == length(A) "t and A must be same length"
    log_A = Float64[]
    t_w = Float64[]
    for k in eachindex(t)
        if t[k] >= t_window_lo && t[k] <= t_window_hi
            ak = abs(Float64(A[k]))
            if ak > 0.0 && isfinite(ak)
                push!(t_w, Float64(t[k]))
                push!(log_A, log(ak))
            end
        end
    end
    n = length(t_w)
    if n < 2
        return (γ = NaN, b = NaN, n_pts = n,
                t_lo = Float64(t_window_lo), t_hi = Float64(t_window_hi),
                log_A_residuals = Float64[])
    end
    t̄ = mean(t_w)
    L̄ = mean(log_A)
    num = 0.0
    den = 0.0
    @inbounds for k in eachindex(t_w)
        num += (t_w[k] - t̄) * (log_A[k] - L̄)
        den += (t_w[k] - t̄)^2
    end
    γ = den > 0 ? num / den : NaN
    b = L̄ - γ * t̄
    resid = [log_A[k] - (γ * t_w[k] + b) for k in eachindex(t_w)]
    return (γ = γ, b = b, n_pts = n,
            t_lo = Float64(t_window_lo), t_hi = Float64(t_window_hi),
            log_A_residuals = resid)
end

"""
    perturbation_amplitude(fields, leaves) -> Float64

Per-step diagnostic: the spatial RMS of `δβ_12(x, y, t)` across
all leaves. The antisymmetric tilt mode `δβ_12 = -δβ_21` is
seeded as a sin-mode at IC; under linear evolution it grows
exponentially while remaining sinusoidal in space. Tracking the
RMS amplitude (rather than a single-cell sample) is robust
against mesh-position aliasing.
"""
function perturbation_amplitude(fields, leaves)
    s = 0.0
    n = length(leaves)
    @inbounds for ci in leaves
        v = read_detfield_2d(fields, ci)
        s += v.betas_off[1]^2
    end
    return sqrt(s / max(n, 1))
end

"""
    negative_jacobian_count(fields, leaves; M_vv_override, ρ_ref) -> Int

Per-step diagnostic: the count of leaves where the per-axis
γ_a = sqrt(M_vv,aa - β_a²) is below a tiny floor (≤ 1e-12),
which would indicate the Cholesky-cone determinant has gone
non-positive. With the M3-6 Phase 1b 4-component realizability
projection active this should always remain at 0; tracking it
is the safety belt.
"""
function negative_jacobian_count(fields, leaves;
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
    run_D1_KH_growth_rate(; level=5, U_jet=1.0, jet_width=0.15,
                           perturbation_amp=1e-3, perturbation_k=2,
                           dt=nothing, T_end=nothing,
                           T_factor=1.0,
                           project_kind=:reanchor,
                           realizability_headroom=1.05,
                           Mvv_floor=1e-2, pressure_floor=1e-8,
                           M_vv_override=nothing,
                           ρ_ref=1.0,
                           t_window_factor=(0.2, 0.8))
        -> NamedTuple

Drive a single mesh-level D.1 KH falsifier trajectory.

  • Builds the IC via `tier_d_kh_ic_full` at the requested level
    (resolution `2^level × 2^level`).
  • Attaches the standard KH BCs: PERIODIC along axis 1 (x),
    REFLECTING along axis 2 (y).
  • Runs `det_step_2d_berry_HG!` for `T_end ≈ T_factor / γ_DR`
    (default 1 e-folding time).
  • Tracks `RMS(δβ_12(t))` per step + per-axis γ per step.
  • Fits γ_measured by least-squares on `log RMS` over the
    linear window `[t_window_factor[1] / γ_DR,
    t_window_factor[2] / γ_DR]`.
  • Reports `c_off = γ_measured / γ_DR`, the methods paper's
    §10.5 D.1 calibration value.

Returns a NamedTuple with:
  • `t::Vector{Float64}` (per-step time, length = n_steps + 1)
  • `A_rms::Vector{Float64}` (RMS |δβ_12|(t))
  • `γ1_max, γ1_min, γ2_max, γ2_min, γ1_std, γ2_std::Vector{Float64}`
    (per-axis γ statistics over time)
  • `n_negative_jacobian::Vector{Int}` (per-step count of leaves
    with γ_a ≤ 1e-12)
  • `n_offdiag_events::Vector{Int}` (per-step ProjectionStats
    increments)
  • `γ_DR, γ_measured, c_off::Float64`
  • `wall_time_per_step::Float64` (mean over the run)
  • `params::NamedTuple` (driver parameters echo)
"""
function run_D1_KH_growth_rate(; level::Integer = 5,
                                U_jet::Real = 1.0,
                                jet_width::Real = 0.15,
                                perturbation_amp::Real = 1e-3,
                                perturbation_k::Integer = 2,
                                dt::Union{Real, Nothing} = nothing,
                                T_end::Union{Real, Nothing} = nothing,
                                T_factor::Real = 1.0,
                                project_kind::Symbol = :reanchor,
                                realizability_headroom::Real = 1.05,
                                Mvv_floor::Real = 1e-2,
                                pressure_floor::Real = 1e-8,
                                M_vv_override = nothing,
                                ρ_ref::Real = 1.0,
                                t_window_factor::Tuple{<:Real, <:Real} = (0.5, 1.0),
                                c_back::Real = 1.0,
                                verbose::Bool = false)
    γ_DR = drazin_reid_gamma(; U_jet = U_jet, jet_width = jet_width)
    T_KH = 1.0 / γ_DR
    T_end_val = T_end === nothing ? Float64(T_factor) * T_KH : Float64(T_end)
    # Default dt: ~T_KH / 50 → ~50 steps per e-folding. Coarser than
    # M3-3d's dt=2e-3 because the KH driver lives in a quasi-linear
    # regime and dt is constrained by Newton stability rather than
    # accuracy on the linear mode amplitude.
    if dt === nothing
        # Mesh-scaled dt: at level L the cell size is 1/2^L. CFL ~
        # U/Δx implies dt_cfl ~ Δx/U ~ 1/(2^L · U). Use a fraction of
        # this to keep Newton iteration counts moderate.
        Δx = 1.0 / (2^Int(level))
        dt_val = 0.25 * Δx / Float64(U_jet)
        # Cap at T_end / 30 so we always get at least 30 samples.
        dt_val = min(dt_val, T_end_val / 30.0)
    else
        dt_val = Float64(dt)
    end
    n_steps = Int(ceil(T_end_val / dt_val))
    # Tighten n_steps so the last step lands within (1+1/n) · T_end.
    dt_val = T_end_val / n_steps

    # Build IC.
    ic = tier_d_kh_ic_full(; level = level,
                           U_jet = U_jet, jet_width = jet_width,
                           perturbation_amp = perturbation_amp,
                           perturbation_k = perturbation_k)
    bc_kh = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                  (REFLECTING, REFLECTING)))

    # Pre-allocate trajectory arrays.
    t = zeros(Float64, n_steps + 1)
    A_rms = zeros(Float64, n_steps + 1)
    γ1_max = zeros(Float64, n_steps + 1)
    γ1_min = zeros(Float64, n_steps + 1)
    γ2_max = zeros(Float64, n_steps + 1)
    γ2_min = zeros(Float64, n_steps + 1)
    γ1_std = zeros(Float64, n_steps + 1)
    γ2_std = zeros(Float64, n_steps + 1)
    n_negative_jacobian = zeros(Int, n_steps + 1)
    n_offdiag_events = zeros(Int, n_steps + 1)
    n_proj_events = zeros(Int, n_steps + 1)

    # Initial diagnostics.
    A_rms[1] = perturbation_amplitude(ic.fields, ic.leaves)
    γ_init = gamma_per_axis_2d_field(ic.fields, ic.leaves;
                                       M_vv_override = M_vv_override,
                                       ρ_ref = ρ_ref)
    γ1_max[1] = maximum(γ_init[1, :])
    γ1_min[1] = minimum(γ_init[1, :])
    γ2_max[1] = maximum(γ_init[2, :])
    γ2_min[1] = minimum(γ_init[2, :])
    γ1_std[1] = std(γ_init[1, :])
    γ2_std[1] = std(γ_init[2, :])
    n_negative_jacobian[1] = negative_jacobian_count(ic.fields, ic.leaves;
                                                       M_vv_override = M_vv_override === nothing ? (1.0, 1.0) : M_vv_override,
                                                       ρ_ref = ρ_ref)

    # Per-step ProjectionStats (we accumulate the running totals
    # at each step and expose the per-step delta).
    proj_stats = ProjectionStats()
    nan_seen = false

    wall_t0 = time()
    for n in 1:n_steps
        # Capture pre-step ProjectionStats counters.
        n_offdiag_pre = proj_stats.n_offdiag_events
        n_events_pre = proj_stats.n_events

        try
            det_step_2d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                                    bc_kh, dt_val;
                                    M_vv_override = M_vv_override,
                                    ρ_ref = ρ_ref,
                                    project_kind = project_kind,
                                    realizability_headroom = realizability_headroom,
                                    Mvv_floor = Mvv_floor,
                                    pressure_floor = pressure_floor,
                                    proj_stats = proj_stats,
                                    c_back = c_back)
        catch e
            if verbose
                @warn "Newton solve failed at step $n: $e"
            end
            nan_seen = true
            break
        end

        t[n + 1] = n * dt_val
        A_rms[n + 1] = perturbation_amplitude(ic.fields, ic.leaves)
        γ_now = gamma_per_axis_2d_field(ic.fields, ic.leaves;
                                          M_vv_override = M_vv_override,
                                          ρ_ref = ρ_ref)
        γ1_max[n + 1] = maximum(γ_now[1, :])
        γ1_min[n + 1] = minimum(γ_now[1, :])
        γ2_max[n + 1] = maximum(γ_now[2, :])
        γ2_min[n + 1] = minimum(γ_now[2, :])
        γ1_std[n + 1] = std(γ_now[1, :])
        γ2_std[n + 1] = std(γ_now[2, :])
        n_negative_jacobian[n + 1] = negative_jacobian_count(ic.fields, ic.leaves;
                                                               M_vv_override = M_vv_override === nothing ? (1.0, 1.0) : M_vv_override,
                                                               ρ_ref = ρ_ref)
        n_offdiag_events[n + 1] = proj_stats.n_offdiag_events - n_offdiag_pre
        n_proj_events[n + 1] = proj_stats.n_events - n_events_pre

        if !isfinite(A_rms[n + 1]) || isnan(A_rms[n + 1])
            nan_seen = true
            break
        end

        if verbose && (n % max(1, n_steps ÷ 10) == 0)
            @info "Step $n / $n_steps: t = $(round(t[n+1]; digits=4))," *
                  " A_rms = $(round(A_rms[n+1]; sigdigits=4))," *
                  " γ1_max = $(round(γ1_max[n+1]; sigdigits=4))"
        end
    end
    wall_t1 = time()
    wall_time_per_step = (wall_t1 - wall_t0) / max(n_steps, 1)

    # Linear-window growth-rate fit.
    t_lo = Float64(t_window_factor[1]) * T_KH
    t_hi = Float64(t_window_factor[2]) * T_KH
    fit = fit_linear_growth_rate(t, A_rms; t_window_lo = t_lo, t_window_hi = t_hi)
    γ_measured = fit.γ
    c_off = isnan(γ_measured) ? NaN : γ_measured / γ_DR

    return (
        t = t, A_rms = A_rms,
        γ1_max = γ1_max, γ1_min = γ1_min,
        γ2_max = γ2_max, γ2_min = γ2_min,
        γ1_std = γ1_std, γ2_std = γ2_std,
        n_negative_jacobian = n_negative_jacobian,
        n_offdiag_events = n_offdiag_events,
        n_proj_events = n_proj_events,
        γ_DR = γ_DR, T_KH = T_KH,
        γ_measured = γ_measured, c_off = c_off,
        fit = fit,
        wall_time_per_step = wall_time_per_step,
        nan_seen = nan_seen,
        proj_stats_total = (
            n_steps = proj_stats.n_steps,
            n_events = proj_stats.n_events,
            n_floor_events = proj_stats.n_floor_events,
            n_offdiag_events = proj_stats.n_offdiag_events,
            total_dE_inj = proj_stats.total_dE_inj,
            Mvv_min_pre = proj_stats.Mvv_min_pre,
            Mvv_min_post = proj_stats.Mvv_min_post,
        ),
        ic = ic,
        params = (level = level, U_jet = U_jet, jet_width = jet_width,
                  perturbation_amp = perturbation_amp,
                  perturbation_k = perturbation_k,
                  dt = dt_val, n_steps = n_steps, T_end = T_end_val,
                  T_factor = T_factor,
                  project_kind = project_kind,
                  realizability_headroom = realizability_headroom,
                  Mvv_floor = Mvv_floor,
                  pressure_floor = pressure_floor,
                  M_vv_override = M_vv_override,
                  ρ_ref = ρ_ref,
                  t_window_factor = t_window_factor),
    )
end

"""
    run_D1_KH_mesh_sweep(; levels=(4, 5, 6), kwargs...) -> NamedTuple

Run `run_D1_KH_growth_rate` at multiple refinement levels and
package the results for mesh-refinement convergence + headline
plot. `kwargs` are forwarded per level.

Returns:
  • `levels::Tuple{Int,Int,Int}`
  • `results::Vector{NamedTuple}` (one per level)
  • `γ_DR::Float64` (consistent across levels)
  • `γ_measured::Vector{Float64}`
  • `c_off::Vector{Float64}`
  • `convergence_rate_45::Float64` —
    `|γ(L=5) - γ(L=4)| / |γ(L=4)|`
  • `convergence_rate_56::Float64` —
    `|γ(L=6) - γ(L=5)| / |γ(L=5)|`
"""
function run_D1_KH_mesh_sweep(; levels = (4, 5, 6),
                              U_jet::Real = 1.0,
                              jet_width::Real = 0.15,
                              perturbation_amp::Real = 1e-3,
                              perturbation_k::Integer = 2,
                              T_factor::Real = 1.0,
                              t_window_factor = (0.5, 1.0),
                              kwargs...)
    results = NamedTuple[]
    for L in levels
        push!(results, run_D1_KH_growth_rate(; level = L,
                                              U_jet = U_jet,
                                              jet_width = jet_width,
                                              perturbation_amp = perturbation_amp,
                                              perturbation_k = perturbation_k,
                                              T_factor = T_factor,
                                              t_window_factor = t_window_factor,
                                              kwargs...))
    end
    γ_DR = results[1].γ_DR
    γ_measured = [r.γ_measured for r in results]
    c_off = [r.c_off for r in results]
    n = length(results)
    rate_45 = n >= 2 ? abs(γ_measured[2] - γ_measured[1]) / max(abs(γ_measured[1]), 1e-12) : NaN
    rate_56 = n >= 3 ? abs(γ_measured[3] - γ_measured[2]) / max(abs(γ_measured[2]), 1e-12) : NaN
    return (
        levels = Tuple(levels),
        results = results,
        γ_DR = γ_DR,
        γ_measured = γ_measured,
        c_off = c_off,
        convergence_rate_45 = rate_45,
        convergence_rate_56 = rate_56,
    )
end

"""
    save_D1_KH_to_h5(sweep, save_path)

Write the mesh-sweep result to HDF5. The file layout is:

  /levels                    Vector{Int}
  /γ_DR                       Float64
  /γ_measured                 Vector{Float64}
  /c_off                      Vector{Float64}
  /convergence_rate_45        Float64
  /convergence_rate_56        Float64
  /level_<L>/t                Vector{Float64}
  /level_<L>/A_rms            Vector{Float64}
  /level_<L>/γ1_max           Vector{Float64}
  /level_<L>/γ1_min           Vector{Float64}
  /level_<L>/γ2_max           Vector{Float64}
  /level_<L>/γ2_min           Vector{Float64}
  /level_<L>/γ1_std           Vector{Float64}
  /level_<L>/γ2_std           Vector{Float64}
  /level_<L>/n_negative_jacobian   Vector{Int}
  /level_<L>/n_offdiag_events      Vector{Int}
  /level_<L>/wall_time_per_step    Float64
"""
function save_D1_KH_to_h5(sweep, save_path::AbstractString)
    HDF5 = if isdefined(Main, :HDF5)
        getfield(Main, :HDF5)
    else
        Base.require(Main, :HDF5)
        getfield(Main, :HDF5)
    end
    mkpath(dirname(save_path))
    HDF5.h5open(save_path, "w") do f
        f["levels"] = collect(sweep.levels)
        f["gamma_DR"] = sweep.γ_DR
        f["gamma_measured"] = sweep.γ_measured
        f["c_off"] = sweep.c_off
        f["convergence_rate_45"] = sweep.convergence_rate_45
        f["convergence_rate_56"] = sweep.convergence_rate_56
        for (i, L) in enumerate(sweep.levels)
            grp = HDF5.create_group(f, "level_$(L)")
            r = sweep.results[i]
            grp["t"] = r.t
            grp["A_rms"] = r.A_rms
            grp["gamma1_max"] = r.γ1_max
            grp["gamma1_min"] = r.γ1_min
            grp["gamma2_max"] = r.γ2_max
            grp["gamma2_min"] = r.γ2_min
            grp["gamma1_std"] = r.γ1_std
            grp["gamma2_std"] = r.γ2_std
            grp["n_negative_jacobian"] = r.n_negative_jacobian
            grp["n_offdiag_events"] = r.n_offdiag_events
            grp["n_proj_events"] = r.n_proj_events
            grp["wall_time_per_step"] = r.wall_time_per_step
            grp["gamma_measured"] = r.γ_measured
            grp["c_off"] = r.c_off
        end
    end
    return save_path
end

"""
    plot_D1_KH_growth_rate(sweep; save_path) -> save_path

4-panel CairoMakie headline plot:
  • Panel A: log|A_rms(t)| at refinement levels with γ_DR · t overlay.
  • Panel B: γ_measured vs γ_DR ratio (= c_off) per level with
             c_off² annotation.
  • Panel C: per-axis γ_1, γ_2 trajectory at level 5.
  • Panel D: 4-comp realizability events
             (n_offdiag_events, n_negative_jacobian) over time.

Falls back to CSV if CairoMakie load fails.
"""
function plot_D1_KH_growth_rate(sweep; save_path::AbstractString)
    try
        # Prefer the already-loaded CairoMakie if available; else try
        # `Base.require` to load it. The double-handshake mirrors the
        # M3-3d driver's pattern: callers `using CairoMakie` first.
        CM = if isdefined(Main, :CairoMakie)
            getfield(Main, :CairoMakie)
        else
            Base.require(Main, :CairoMakie)
            getfield(Main, :CairoMakie)
        end
        fig = CM.Figure(size = (1100, 850))
        γ_DR = sweep.γ_DR

        # Panel A: log|A_rms(t)| at all levels with theory overlay.
        axA = CM.Axis(fig[1, 1];
            title = "A: log|⟨|δβ_12|⟩|(t) — KH linear growth",
            xlabel = "t", ylabel = "log|A_rms|")
        for (i, L) in enumerate(sweep.levels)
            r = sweep.results[i]
            keep = [k for k in eachindex(r.t) if r.A_rms[k] > 0]
            CM.lines!(axA, r.t[keep], log.(r.A_rms[keep]),
                      label = "L=$L (2^$L)²")
        end
        # Theory overlay anchored at t=0 with the first level's IC amplitude.
        if !isempty(sweep.results)
            r0 = sweep.results[1]
            t_th = collect(0.0:r0.params.dt:r0.params.T_end)
            log_A_th = log(r0.A_rms[1]) .+ γ_DR .* t_th
            CM.lines!(axA, t_th, log_A_th;
                      linestyle = :dash, label = "γ_DR·t")
        end
        CM.axislegend(axA; position = :rb)

        # Panel B: γ_measured / γ_DR ratio per level.
        axB = CM.Axis(fig[1, 2];
            title = "B: γ_measured / γ_DR (c_off calibration)",
            xlabel = "level", ylabel = "γ_measured / γ_DR")
        Lx = collect(Int, sweep.levels)
        c_off_arr = sweep.c_off
        CM.scatter!(axB, Lx, c_off_arr, markersize = 16)
        CM.lines!(axB, [extrema(Lx)...], [1.0, 1.0];
                  linestyle = :dash, color = :gray)
        if length(c_off_arr) >= 2
            μc = mean(filter(isfinite, c_off_arr))
            CM.text!(axB, 0.5, 0.05;
                space = :relative,
                text = "c_off ≈ $(round(μc; sigdigits = 3)),\n" *
                       "c_off² ≈ $(round(μc^2; sigdigits = 3))")
        end

        # Panel C: per-axis γ at level 5 (or the middle level).
        mid_idx = max(1, min(length(sweep.results), 2))
        rmid = sweep.results[mid_idx]
        Lmid = sweep.levels[mid_idx]
        axC = CM.Axis(fig[2, 1];
            title = "C: per-axis γ at L=$Lmid (max/min)",
            xlabel = "t", ylabel = "γ_a")
        CM.lines!(axC, rmid.t, rmid.γ1_max; label = "γ_1 max")
        CM.lines!(axC, rmid.t, rmid.γ1_min; label = "γ_1 min")
        CM.lines!(axC, rmid.t, rmid.γ2_max; label = "γ_2 max")
        CM.lines!(axC, rmid.t, rmid.γ2_min; label = "γ_2 min")
        CM.axislegend(axC; position = :rb)

        # Panel D: realizability events.
        axD = CM.Axis(fig[2, 2];
            title = "D: realizability cone events at L=$Lmid",
            xlabel = "t", ylabel = "events / step")
        CM.lines!(axD, rmid.t, rmid.n_offdiag_events; label = "n_offdiag")
        CM.lines!(axD, rmid.t, rmid.n_negative_jacobian; label = "n_neg_jac")
        CM.lines!(axD, rmid.t, rmid.n_proj_events; label = "n_proj (s-raise)")
        CM.axislegend(axD; position = :rt)

        mkpath(dirname(save_path))
        CM.save(save_path, fig)
        return save_path
    catch e
        @warn "Plotting D.1 KH figure failed: $(e). Saving CSV trajectory instead."
        mkpath(dirname(save_path))
        for (i, L) in enumerate(sweep.levels)
            r = sweep.results[i]
            csv = replace(save_path, ".png" => "_L$L.csv")
            open(csv, "w") do f
                println(f, "t,A_rms,γ1_max,γ1_min,γ2_max,γ2_min," *
                            "n_offdiag,n_neg_jac")
                for k in eachindex(r.t)
                    println(f, "$(r.t[k]),$(r.A_rms[k])," *
                              "$(r.γ1_max[k]),$(r.γ1_min[k])," *
                              "$(r.γ2_max[k]),$(r.γ2_min[k])," *
                              "$(r.n_offdiag_events[k]),$(r.n_negative_jacobian[k])")
                end
            end
        end
        return save_path
    end
end
