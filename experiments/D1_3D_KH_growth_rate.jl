# D1_3D_KH_growth_rate.jl
#
# §10.5 D.1 3D Kelvin-Helmholtz falsifier driver — M4 Phase 2.
# 3D analog of `experiments/D1_KH_growth_rate.jl` (M3-6 Phase 1c / M4
# Phase 1).
#
# Drives the M4 Phase 2 `tier_d_kh_3d_ic_full` IC through
# `det_step_3d_berry_kh_HG!` (M4 Phase 2 21-dof residual with three
# off-diagonal Cholesky pairs and closed-loop H_back per pair) at
# refinement level 3 (8³ = 512 cells; level-budget cap), fits the
# linear growth rate γ_measured to the antisymmetric tilt-mode
# amplitude `δβ_12(t)`, and compares to the Drazin-Reid prediction
# γ_DR = U / (2 w):
#
#   • Linear-regime fit: log|⟨|δβ_12(t)|⟩| ~ γ_measured · t + const.
#   • c_off² calibration: c_off = γ_measured / γ_DR.
#   • Per-axis γ diagnostic: γ_1, γ_2, γ_3 (three axes, vs 2D's two).
#   • 6-component β-cone realizability stats: n_negative_jacobian.
#
# Saves time series to HDF5 at
# `reference/figs/M4_phase2_3d_kh_falsifier.h5` (caller-controlled).
#
# Usage (REPL):
#
#   julia> include("experiments/D1_3D_KH_growth_rate.jl")
#   julia> result = run_D1_3D_KH_growth_rate(; level=3)
#
# Returns a NamedTuple with the trajectory + γ fit + per-axis γ
# diagnostic + per-step β-cone realizability stats for use by
# `test/test_M4_phase2_3d_kh_falsifier.jl`.

using HierarchicalGrids: HierarchicalMesh, EulerianFrame,
    refine_cells!, enumerate_leaves, FrameBoundaries, REFLECTING, PERIODIC,
    cell_physical_box
using dfmm: tier_d_kh_3d_ic_full,
    det_step_3d_berry_kh_HG!,
    allocate_cholesky_3d_kh_fields
using Statistics: mean, std

"""
    drazin_reid_gamma_3d(; U_jet=1.0, jet_width=0.15) -> Float64

Classical Drazin-Reid linear growth rate for the tanh shear layer:

    γ_DR = U_jet / (2 · jet_width).

In 3D the dominant 2D-symmetric mode along x has the same growth rate
as the 2D problem; this is the dimension-lift property exploited by the
M4 Phase 2 acceptance gate.
"""
drazin_reid_gamma_3d(; U_jet::Real = 1.0, jet_width::Real = 0.15) =
    Float64(U_jet) / (2.0 * Float64(jet_width))

"""
    perturbation_amplitude_3d(fields, leaves) -> Float64

Per-step diagnostic: the spatial RMS of `δβ_12(x, y, z, t)` across all
leaves of the 3D KH-active field set.
"""
function perturbation_amplitude_3d(fields, leaves)
    s = 0.0
    n = length(leaves)
    @inbounds for ci in leaves
        s += fields.β_12[ci][1]^2
    end
    return sqrt(s / max(n, 1))
end

"""
    gamma_per_axis_3d_kh(fields, leaves; M_vv_override, ρ_ref) -> Matrix{Float64}

Per-axis γ_a = sqrt(M_vv,aa - β_a²) for the 3D KH-active field set.
Returns a 3 × N matrix.
"""
function gamma_per_axis_3d_kh(fields, leaves;
                                M_vv_override = (1.0, 1.0, 1.0),
                                ρ_ref::Real = 1.0)
    N = length(leaves)
    γ = Array{Float64}(undef, 3, N)
    @inbounds for (i, ci) in enumerate(leaves)
        β1 = fields.β_1[ci][1]
        β2 = fields.β_2[ci][1]
        β3 = fields.β_3[ci][1]
        γ[1, i] = sqrt(max(0.0, M_vv_override[1] - β1^2))
        γ[2, i] = sqrt(max(0.0, M_vv_override[2] - β2^2))
        γ[3, i] = sqrt(max(0.0, M_vv_override[3] - β3^2))
    end
    return γ
end

"""
    negative_jacobian_count_3d(fields, leaves; M_vv_override, ρ_ref) -> Int

Per-step diagnostic: count of leaves where any of the three γ_a is
below 1e-12 (i.e., where the Cholesky-cone determinant has gone non-
positive). With the M4 Phase 2 IC (β_a = 0, off-diag β small) this
should remain 0 throughout the linear regime.
"""
function negative_jacobian_count_3d(fields, leaves;
                                      M_vv_override = (1.0, 1.0, 1.0),
                                      ρ_ref::Real = 1.0)
    γ = gamma_per_axis_3d_kh(fields, leaves;
                              M_vv_override = M_vv_override, ρ_ref = ρ_ref)
    n_neg = 0
    @inbounds for i in 1:size(γ, 2)
        if γ[1, i] ≤ 1e-12 || γ[2, i] ≤ 1e-12 || γ[3, i] ≤ 1e-12
            n_neg += 1
        end
    end
    return n_neg
end

"""
    six_component_offdiag_max(fields, leaves) -> NTuple{6, Float64}

Per-step diagnostic: per-pair maximum |β_{ab}| across leaves.
Order: (β_12, β_21, β_13, β_31, β_23, β_32).
"""
function six_component_offdiag_max(fields, leaves)
    m12 = 0.0; m21 = 0.0; m13 = 0.0; m31 = 0.0; m23 = 0.0; m32 = 0.0
    @inbounds for ci in leaves
        m12 = max(m12, abs(fields.β_12[ci][1]))
        m21 = max(m21, abs(fields.β_21[ci][1]))
        m13 = max(m13, abs(fields.β_13[ci][1]))
        m31 = max(m31, abs(fields.β_31[ci][1]))
        m23 = max(m23, abs(fields.β_23[ci][1]))
        m32 = max(m32, abs(fields.β_32[ci][1]))
    end
    return (m12, m21, m13, m31, m23, m32)
end

"""
    fit_linear_growth_rate_3d(t, A; t_window_lo, t_window_hi) -> NamedTuple

Least-squares fit `log|A(t)| = γ · t + b` over the linear window
`[t_window_lo, t_window_hi]`. Mirrors `fit_linear_growth_rate` in the
2D D.1 driver.
"""
function fit_linear_growth_rate_3d(t::AbstractVector, A::AbstractVector;
                                     t_window_lo::Real, t_window_hi::Real)
    @assert length(t) == length(A)
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
    num = 0.0; den = 0.0
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
    fit_linear_vs_exp_3d(t, A) -> NamedTuple

Two-model fit: `A ≈ a0 + b·t` (linear-in-t) vs `A ≈ exp(c + γ·t)`
(exp-in-t). Returns both fits' residual sum of squares plus
`lin_better::Bool` flag (the honest verdict — whichever model fits
better).
"""
function fit_linear_vs_exp_3d(t::AbstractVector, A::AbstractVector)
    n = length(t)
    @assert n == length(A)
    t̄ = mean(t); Ā = mean(A)
    den = sum((t .- t̄).^2)
    b_lin = den > 0 ? sum((t .- t̄) .* (A .- Ā)) / den : 0.0
    a0_lin = Ā - b_lin * t̄
    A_pred_lin = a0_lin .+ b_lin .* t
    ssr_lin = sum((A .- A_pred_lin).^2)

    mask = A .> 1e-14
    if sum(mask) < 2
        return (b_lin = b_lin, a0_lin = a0_lin, ssr_lin = ssr_lin,
                γ_exp = NaN, c_exp = NaN, ssr_exp = Inf,
                lin_better = true)
    end
    logA = log.(A[mask])
    t_e = t[mask]
    t̄_e = mean(t_e); L̄_e = mean(logA)
    den_e = sum((t_e .- t̄_e).^2)
    γ_exp = den_e > 0 ? sum((t_e .- t̄_e) .* (logA .- L̄_e)) / den_e : 0.0
    c_exp = L̄_e - γ_exp * t̄_e
    A_pred_exp = exp.(c_exp .+ γ_exp .* t)
    ssr_exp = sum((A .- A_pred_exp).^2)

    return (b_lin = b_lin, a0_lin = a0_lin, ssr_lin = ssr_lin,
            γ_exp = γ_exp, c_exp = c_exp, ssr_exp = ssr_exp,
            lin_better = ssr_lin < ssr_exp)
end

"""
    run_D1_3D_KH_growth_rate(; level=3, U_jet=1.0, jet_width=0.15,
                              perturbation_amp=1e-3, perturbation_k=2,
                              dt=nothing, T_end=nothing, T_factor=1.0,
                              M_vv_override=(1.0, 1.0, 1.0), ρ_ref=1.0,
                              t_window_factor=(0.5, 1.0),
                              c_back=1.0, verbose=false)
        -> NamedTuple

Drive a single mesh-level 3D D.1 KH falsifier trajectory.

  • Builds the IC via `tier_d_kh_3d_ic_full` at the requested level
    (resolution `2^level × 2^level × 2^level`).
  • Attaches the standard 3D KH BCs: PERIODIC along axis 1 (x),
    REFLECTING along axis 2 (y), PERIODIC along axis 3 (z).
  • Runs `det_step_3d_berry_kh_HG!` for `T_end ≈ T_factor / γ_DR`
    (default 1 e-folding time).
  • Tracks `RMS(δβ_12(t))` per step + per-axis γ + 6-component β
    cone realizability stats.

Returns a NamedTuple with:
  • `t::Vector{Float64}` (per-step time, length n_steps + 1)
  • `A_rms::Vector{Float64}` (RMS |δβ_12|(t))
  • `γ_a_max, γ_a_min::NTuple{3, Vector{Float64}}` (per-axis stats)
  • `n_negative_jacobian::Vector{Int}`
  • `offdiag_max::NTuple{6, Vector{Float64}}` (per-pair max amplitude)
  • `γ_DR, γ_measured, c_off::Float64`
  • `wall_time_per_step::Float64`
  • `params::NamedTuple`
"""
function run_D1_3D_KH_growth_rate(; level::Integer = 3,
                                    U_jet::Real = 1.0,
                                    jet_width::Real = 0.15,
                                    perturbation_amp::Real = 1e-3,
                                    perturbation_k::Integer = 2,
                                    dt::Union{Real,Nothing} = nothing,
                                    T_end::Union{Real,Nothing} = nothing,
                                    T_factor::Real = 1.0,
                                    M_vv_override = (1.0, 1.0, 1.0),
                                    ρ_ref::Real = 1.0,
                                    t_window_factor::Tuple{<:Real,<:Real} = (0.5, 1.0),
                                    c_back::Real = 1.0,
                                    verbose::Bool = false)
    γ_DR = drazin_reid_gamma_3d(; U_jet = U_jet, jet_width = jet_width)
    T_KH = 1.0 / γ_DR
    T_end_val = T_end === nothing ? Float64(T_factor) * T_KH : Float64(T_end)
    if dt === nothing
        Δx = 1.0 / (2^Int(level))
        dt_val = 0.25 * Δx / Float64(U_jet)
        dt_val = min(dt_val, T_end_val / 30.0)
    else
        dt_val = Float64(dt)
    end
    n_steps = Int(ceil(T_end_val / dt_val))
    dt_val = T_end_val / n_steps

    # Build IC.
    ic = tier_d_kh_3d_ic_full(; level = level,
                                U_jet = U_jet, jet_width = jet_width,
                                perturbation_amp = perturbation_amp,
                                perturbation_k = perturbation_k)
    bc_kh = FrameBoundaries{3}(((PERIODIC, PERIODIC),
                                  (REFLECTING, REFLECTING),
                                  (PERIODIC, PERIODIC)))

    # Pre-allocate trajectory arrays.
    t = zeros(Float64, n_steps + 1)
    A_rms = zeros(Float64, n_steps + 1)
    γ1_max = zeros(Float64, n_steps + 1); γ1_min = zeros(Float64, n_steps + 1)
    γ2_max = zeros(Float64, n_steps + 1); γ2_min = zeros(Float64, n_steps + 1)
    γ3_max = zeros(Float64, n_steps + 1); γ3_min = zeros(Float64, n_steps + 1)
    n_negative_jacobian = zeros(Int, n_steps + 1)
    off12_max = zeros(Float64, n_steps + 1)
    off21_max = zeros(Float64, n_steps + 1)
    off13_max = zeros(Float64, n_steps + 1)
    off31_max = zeros(Float64, n_steps + 1)
    off23_max = zeros(Float64, n_steps + 1)
    off32_max = zeros(Float64, n_steps + 1)

    # Initial diagnostics.
    A_rms[1] = perturbation_amplitude_3d(ic.fields, ic.leaves)
    γ_init = gamma_per_axis_3d_kh(ic.fields, ic.leaves;
                                    M_vv_override = M_vv_override, ρ_ref = ρ_ref)
    γ1_max[1] = maximum(γ_init[1, :]); γ1_min[1] = minimum(γ_init[1, :])
    γ2_max[1] = maximum(γ_init[2, :]); γ2_min[1] = minimum(γ_init[2, :])
    γ3_max[1] = maximum(γ_init[3, :]); γ3_min[1] = minimum(γ_init[3, :])
    n_negative_jacobian[1] = negative_jacobian_count_3d(ic.fields, ic.leaves;
                                                          M_vv_override = M_vv_override,
                                                          ρ_ref = ρ_ref)
    o0 = six_component_offdiag_max(ic.fields, ic.leaves)
    off12_max[1] = o0[1]; off21_max[1] = o0[2]
    off13_max[1] = o0[3]; off31_max[1] = o0[4]
    off23_max[1] = o0[5]; off32_max[1] = o0[6]

    nan_seen = false
    wall_t0 = time()
    for n in 1:n_steps
        try
            det_step_3d_berry_kh_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                                      bc_kh, dt_val;
                                      M_vv_override = M_vv_override,
                                      ρ_ref = ρ_ref, c_back = c_back)
        catch e
            if verbose
                @warn "Newton solve failed at step $n: $e"
            end
            nan_seen = true
            break
        end

        t[n + 1] = n * dt_val
        A_rms[n + 1] = perturbation_amplitude_3d(ic.fields, ic.leaves)
        γ_now = gamma_per_axis_3d_kh(ic.fields, ic.leaves;
                                      M_vv_override = M_vv_override, ρ_ref = ρ_ref)
        γ1_max[n + 1] = maximum(γ_now[1, :]); γ1_min[n + 1] = minimum(γ_now[1, :])
        γ2_max[n + 1] = maximum(γ_now[2, :]); γ2_min[n + 1] = minimum(γ_now[2, :])
        γ3_max[n + 1] = maximum(γ_now[3, :]); γ3_min[n + 1] = minimum(γ_now[3, :])
        n_negative_jacobian[n + 1] = negative_jacobian_count_3d(ic.fields, ic.leaves;
                                                                  M_vv_override = M_vv_override,
                                                                  ρ_ref = ρ_ref)
        on = six_component_offdiag_max(ic.fields, ic.leaves)
        off12_max[n + 1] = on[1]; off21_max[n + 1] = on[2]
        off13_max[n + 1] = on[3]; off31_max[n + 1] = on[4]
        off23_max[n + 1] = on[5]; off32_max[n + 1] = on[6]

        if !isfinite(A_rms[n + 1]) || isnan(A_rms[n + 1])
            nan_seen = true
            break
        end

        if verbose && (n % max(1, n_steps ÷ 10) == 0)
            @info "Step $n / $n_steps: t=$(round(t[n+1]; digits=4)), " *
                  "A_rms=$(round(A_rms[n+1]; sigdigits=4))"
        end
    end
    wall_t1 = time()
    wall_time_per_step = (wall_t1 - wall_t0) / max(n_steps, 1)

    t_lo = Float64(t_window_factor[1]) * T_KH
    t_hi = Float64(t_window_factor[2]) * T_KH
    fit = fit_linear_growth_rate_3d(t, A_rms;
                                     t_window_lo = t_lo, t_window_hi = t_hi)
    γ_measured = fit.γ
    c_off = isnan(γ_measured) ? NaN : γ_measured / γ_DR

    return (
        t = t, A_rms = A_rms,
        γ1_max = γ1_max, γ1_min = γ1_min,
        γ2_max = γ2_max, γ2_min = γ2_min,
        γ3_max = γ3_max, γ3_min = γ3_min,
        n_negative_jacobian = n_negative_jacobian,
        off12_max = off12_max, off21_max = off21_max,
        off13_max = off13_max, off31_max = off31_max,
        off23_max = off23_max, off32_max = off32_max,
        γ_DR = γ_DR, T_KH = T_KH,
        γ_measured = γ_measured, c_off = c_off,
        fit = fit,
        wall_time_per_step = wall_time_per_step,
        nan_seen = nan_seen,
        ic = ic,
        params = (level = level, U_jet = U_jet, jet_width = jet_width,
                   perturbation_amp = perturbation_amp,
                   perturbation_k = perturbation_k,
                   dt = dt_val, n_steps = n_steps, T_end = T_end_val,
                   T_factor = T_factor,
                   M_vv_override = M_vv_override, ρ_ref = ρ_ref,
                   t_window_factor = t_window_factor,
                   c_back = c_back),
    )
end

"""
    plot_D1_3D_KH_growth_rate(result; save_path) -> save_path

4-panel headline plot for the M4 Phase 2 3D KH falsifier.

  • Panel A: log|A_rms(t)| with linear-in-t and exp-in-t fit overlays
    plus γ_DR · t reference.
  • Panel B: 3D ⊂ 2D dimension-lift (γ_measured 3D vs typical 2D
    measurement, indicated by reference dotted line at c_off ≈ 1.26).
  • Panel C: 6-component β cone diagnostics (max amplitude per pair
    over time).
  • Panel D: per-axis γ_1, γ_2, γ_3 trajectories (max/min).

Falls back to CSV if CairoMakie load fails.
"""
function plot_D1_3D_KH_growth_rate(result; save_path::AbstractString)
    try
        CM = if isdefined(Main, :CairoMakie)
            getfield(Main, :CairoMakie)
        else
            Base.require(Main, :CairoMakie)
            getfield(Main, :CairoMakie)
        end
        fig = CM.Figure(size = (1200, 900))
        γ_DR = result.γ_DR
        # Panel A
        axA = CM.Axis(fig[1, 1];
            title = "A: 3D KH log|A_rms(t)| with linear & exp fits",
            xlabel = "t", ylabel = "log|A_rms|")
        keep = [k for k in eachindex(result.t) if result.A_rms[k] > 0]
        CM.lines!(axA, result.t[keep], log.(result.A_rms[keep]);
                  label = "L=$(result.params.level)")
        # Linear-in-t overlay
        b_lin = result.fit.γ
        if !isnan(b_lin) && !isempty(keep)
            log_A_lin = result.fit.b .+ b_lin .* result.t[keep]
            CM.lines!(axA, result.t[keep], log_A_lin;
                      linestyle = :dot, label = "log-fit (γ=$(round(b_lin; sigdigits=3)))")
        end
        t_th = collect(0.0:result.params.dt:result.params.T_end)
        log_A_th = log(result.A_rms[1]) .+ γ_DR .* t_th
        CM.lines!(axA, t_th, log_A_th;
                  linestyle = :dash, label = "γ_DR·t")
        CM.axislegend(axA; position = :rb)

        # Panel B: 3D ⊂ 2D dimension-lift
        axB = CM.Axis(fig[1, 2];
            title = "B: 3D γ_measured/γ_DR vs 2D reference",
            xlabel = "phase", ylabel = "γ_measured / γ_DR")
        c_off_3d = result.c_off
        # 2D Phase 1 reference: c_off ≈ 1.26 at level 4
        CM.scatter!(axB, [1.0], [c_off_3d];
                    markersize = 18, label = "3D L=$(result.params.level)")
        CM.scatter!(axB, [2.0], [1.26];
                    markersize = 18, color = :red, label = "2D L=4 (M4 Phase 1)")
        CM.lines!(axB, [0.5, 2.5], [1.0, 1.0];
                  linestyle = :dash, color = :gray)
        CM.lines!(axB, [0.5, 2.5], [0.5, 0.5];
                  linestyle = :dot, color = :gray)
        CM.lines!(axB, [0.5, 2.5], [2.0, 2.0];
                  linestyle = :dot, color = :gray)
        CM.axislegend(axB; position = :lt)

        # Panel C: 6-component β cone
        axC = CM.Axis(fig[2, 1];
            title = "C: 6-component β cone — max|β_{ab}|(t)",
            xlabel = "t", ylabel = "max|β|")
        CM.lines!(axC, result.t, result.off12_max; label = "β_12")
        CM.lines!(axC, result.t, result.off21_max; label = "β_21")
        CM.lines!(axC, result.t, result.off13_max; label = "β_13")
        CM.lines!(axC, result.t, result.off31_max; label = "β_31")
        CM.lines!(axC, result.t, result.off23_max; label = "β_23")
        CM.lines!(axC, result.t, result.off32_max; label = "β_32")
        CM.axislegend(axC; position = :rb)

        # Panel D: per-axis γ
        axD = CM.Axis(fig[2, 2];
            title = "D: per-axis γ_a (max/min) trajectories",
            xlabel = "t", ylabel = "γ_a")
        CM.lines!(axD, result.t, result.γ1_max; label = "γ_1 max")
        CM.lines!(axD, result.t, result.γ1_min; label = "γ_1 min")
        CM.lines!(axD, result.t, result.γ2_max; label = "γ_2 max")
        CM.lines!(axD, result.t, result.γ2_min; label = "γ_2 min")
        CM.lines!(axD, result.t, result.γ3_max; label = "γ_3 max")
        CM.lines!(axD, result.t, result.γ3_min; label = "γ_3 min")
        CM.axislegend(axD; position = :rb)

        mkpath(dirname(save_path))
        CM.save(save_path, fig)
        return save_path
    catch e
        @warn "Plotting M4 Phase 2 3D KH figure failed: $(e). Saving CSV instead."
        mkpath(dirname(save_path))
        csv = replace(save_path, ".png" => ".csv")
        open(csv, "w") do f
            println(f, "t,A_rms,gamma1_max,gamma1_min,gamma2_max,gamma2_min,gamma3_max,gamma3_min," *
                       "off12,off21,off13,off31,off23,off32,n_neg_jac")
            for k in eachindex(result.t)
                println(f, "$(result.t[k]),$(result.A_rms[k])," *
                          "$(result.γ1_max[k]),$(result.γ1_min[k])," *
                          "$(result.γ2_max[k]),$(result.γ2_min[k])," *
                          "$(result.γ3_max[k]),$(result.γ3_min[k])," *
                          "$(result.off12_max[k]),$(result.off21_max[k])," *
                          "$(result.off13_max[k]),$(result.off31_max[k])," *
                          "$(result.off23_max[k]),$(result.off32_max[k])," *
                          "$(result.n_negative_jacobian[k])")
            end
        end
        return save_path
    end
end
