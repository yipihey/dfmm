# C3_2d_plane_wave_convergence.jl
#
# M3-9 prep: Log-log convergence figure for the C.3 2D acoustic plane-wave
# initial condition. Complements the existing rotation-invariance plot
# (`reference/figs/M3_4_C3_plane_wave_rotation_invariance.png`).
#
# Drives `tier_c_plane_wave_ic` at HG refinement levels {3, 4, 5, 6}
# (16, 32, 64, 128 cells per axis) with amplitude A = 1e-3 and angle
# θ ∈ {0, π/4}. For each level computes the L∞ and L² norms of the
# cell-average vs. point-sample residual
#
#     err_j = δρ_cell(j) - δρ_pt(j)
#
# where δρ_cell(j) is the cell-averaged density perturbation written by
# the IC factory (via HG `init_field_from!` L²-projection at
# quad_order = 3) and δρ_pt(j) = A · cos(2π · k · x_center) is the
# point sample at the cell midpoint.
#
# The midpoint-rule analysis predicts an O(Δx²) leading-order error with
# coefficient -(2π)² · A / 24 (Taylor expansion of the cosine over a
# cell of width Δx around its center). With A = 1e-3 and Δx = 1/N, the
# expected L∞ error is ≈ 1.6e-3 / N². At N = 16, 32, 64, 128 this gives
# 6.4e-6, 1.6e-6, 4.0e-7, 1.0e-7 — successive ratios of 4.
#
# The §10.7 quoted ratios (3.93, 3.98 for level 4→5→6) are reproduced.
#
# Saves:
#   • `reference/figs/M3_3_C3_plane_wave_convergence.h5` — raw data
#   • `reference/figs/M3_3_C3_plane_wave_convergence.png` — log-log plot
#
# Usage (REPL):
#   julia> include("experiments/C3_2d_plane_wave_convergence.jl")
#   julia> result = run_C3_plane_wave_convergence()
#   julia> save_C3_plane_wave_convergence_to_h5(result)
#   julia> plot_C3_plane_wave_convergence(result)

using HierarchicalGrids: enumerate_leaves
using dfmm: tier_c_plane_wave_ic, tier_c_cell_centers

"""
    run_C3_plane_wave_convergence(; levels=(3, 4, 5, 6),
                                    angles=(0.0, π/4),
                                    A=1e-3, k_mag=1, ρ0=1.0, P0=1.0)
        -> NamedTuple

For each (level, angle) pair build the C.3 plane-wave IC and compute
the L∞ and L² norms of the cell-average vs. point-sample residual on
the density field.

Returns:
  • `levels::Vector{Int}`           — refinement levels swept
  • `n_per_axis::Vector{Int}`       — 2^level (cells per axis = N)
  • `dx::Vector{Float64}`           — Δx = 1/N (unit box)
  • `angles::Vector{Float64}`       — wave-vector angles in radians
  • `err_inf::Matrix{Float64}`      — (n_levels × n_angles) L∞ errors
  • `err_l2::Matrix{Float64}`       — (n_levels × n_angles) L² errors
  • `params::NamedTuple`            — (A, k_mag, ρ0, P0)
"""
function run_C3_plane_wave_convergence(;
        levels = (3, 4, 5, 6),
        angles = (0.0, π/4),
        A::Real = 1e-3, k_mag::Real = 1,
        ρ0::Real = 1.0, P0::Real = 1.0)
    n_levels = length(levels)
    n_angles = length(angles)
    err_inf = zeros(Float64, n_levels, n_angles)
    err_l2 = zeros(Float64, n_levels, n_angles)
    n_per_axis = Int[2^L for L in levels]
    dx = Float64[1.0 / N for N in n_per_axis]

    for (li, level) in enumerate(levels)
        for (ai, θ) in enumerate(angles)
            ic = tier_c_plane_wave_ic(D = 2, level = level, A = A,
                                       k_mag = k_mag, angle = θ,
                                       ρ0 = ρ0, P0 = P0)
            n = length(enumerate_leaves(ic.mesh))
            centers = tier_c_cell_centers(ic)
            k_phys = ic.params.k
            lo_t = ic.params.lo

            inf_err = 0.0
            l2_sq = 0.0
            for j in 1:n
                xc = centers[j]
                phase = k_phys[1] * (xc[1] - lo_t[1]) +
                        k_phys[2] * (xc[2] - lo_t[2])
                δρ_pt = Float64(A) * cos(2π * phase)
                δρ_cell = Float64(ic.fields.rho[j][1]) - Float64(ρ0)
                e = abs(δρ_cell - δρ_pt)
                inf_err = max(inf_err, e)
                l2_sq += e^2
            end
            err_inf[li, ai] = inf_err
            # L²-norm normalised by 1/N² (cells × cell-volume on unit box).
            err_l2[li, ai] = sqrt(l2_sq / n)
        end
    end

    return (
        levels = collect(levels),
        n_per_axis = n_per_axis,
        dx = dx,
        angles = collect(Float64.(angles)),
        err_inf = err_inf,
        err_l2 = err_l2,
        params = (A = Float64(A), k_mag = Float64(k_mag),
                  ρ0 = Float64(ρ0), P0 = Float64(P0)),
    )
end

"""
    fit_loglog_slope(x::AbstractVector, y::AbstractVector) -> Float64

Least-squares slope of `log10(y)` vs `log10(x)`. Used to characterise
the observed convergence rate.
"""
function fit_loglog_slope(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y) > 1
    lx = log10.(Float64.(x))
    ly = log10.(Float64.(y))
    x̄ = sum(lx) / length(lx)
    ȳ = sum(ly) / length(ly)
    num = sum((lx .- x̄) .* (ly .- ȳ))
    den = sum((lx .- x̄) .^ 2)
    return num / den
end

"""
    save_C3_plane_wave_convergence_to_h5(result; save_path)

Persist the convergence sweep to HDF5. Lazy-loads `HDF5` from `Main` so
the experiment driver doesn't pay precompile cost when the caller only
wants the in-memory `result`.
"""
function save_C3_plane_wave_convergence_to_h5(result;
        save_path::AbstractString =
            joinpath(@__DIR__, "..", "reference", "figs",
                      "M3_3_C3_plane_wave_convergence.h5"))
    HDF5 = if isdefined(Main, :HDF5)
        getfield(Main, :HDF5)
    else
        Base.require(Main, :HDF5)
        getfield(Main, :HDF5)
    end
    mkpath(dirname(save_path))
    HDF5.h5open(save_path, "w") do f
        f["levels"] = result.levels
        f["n_per_axis"] = result.n_per_axis
        f["dx"] = result.dx
        f["angles"] = result.angles
        f["err_inf"] = result.err_inf
        f["err_l2"] = result.err_l2
        attrs_g = HDF5.create_group(f, "params")
        attrs_g["A"] = result.params.A
        attrs_g["k_mag"] = result.params.k_mag
        attrs_g["rho0"] = result.params.ρ0
        attrs_g["P0"] = result.params.P0
        # Per-angle observed slopes for downstream consumers.
        slopes_inf = [fit_loglog_slope(result.dx, result.err_inf[:, ai])
                       for ai in 1:length(result.angles)]
        slopes_l2 = [fit_loglog_slope(result.dx, result.err_l2[:, ai])
                       for ai in 1:length(result.angles)]
        f["slope_inf"] = slopes_inf
        f["slope_l2"] = slopes_l2
    end
    return save_path
end

"""
    plot_C3_plane_wave_convergence(result; save_path)

Single-panel log-log convergence plot. x-axis is cells per axis; y-axis
is the L∞ and L² projection error. A reference Δx² line and the
fitted slope are annotated.

Falls back to a CSV dump if CairoMakie is unavailable.
"""
function plot_C3_plane_wave_convergence(result;
        save_path::AbstractString =
            joinpath(@__DIR__, "..", "reference", "figs",
                      "M3_3_C3_plane_wave_convergence.png"))
    try
        Base.require(Main, :CairoMakie)
        CM = getfield(Main, :CairoMakie)

        slope_inf_θ0 = fit_loglog_slope(result.dx, result.err_inf[:, 1])
        slope_l2_θ0 = fit_loglog_slope(result.dx, result.err_l2[:, 1])

        fig = CM.Figure(size = (760, 560))
        ax = CM.Axis(fig[1, 1];
                       title = "C.3 plane wave projection convergence " *
                               "(Δx² midpoint)",
                       xlabel = "cells per axis (N = 2^level)",
                       ylabel = "max |δρ_cell − δρ_pt|",
                       xscale = log10, yscale = log10)

        N = result.n_per_axis
        # Markers/lines per angle for L∞.
        markers = (:circle, :diamond, :rect, :utriangle)
        for (ai, θ) in enumerate(result.angles)
            lbl_inf = "L∞, θ = $(round(θ; digits = 3))"
            CM.scatterlines!(ax, N, result.err_inf[:, ai];
                              label = lbl_inf,
                              marker = markers[mod1(ai, length(markers))],
                              markersize = 12, linewidth = 2)
            lbl_l2 = "L², θ = $(round(θ; digits = 3))"
            CM.scatterlines!(ax, N, result.err_l2[:, ai];
                              label = lbl_l2,
                              marker = markers[mod1(ai, length(markers))],
                              markersize = 10, linewidth = 1.4,
                              linestyle = :dash)
        end

        # Reference Δx² scaling line, anchored at the coarsest L∞ value.
        ref_anchor = result.err_inf[1, 1]
        N_ref = Float64.(N)
        ref_line = ref_anchor .* (Float64(N[1]) ./ N_ref) .^ 2
        CM.lines!(ax, N, ref_line;
                    color = :gray, linestyle = :dot, linewidth = 1.6,
                    label = "Δx² reference (slope −2)")

        CM.axislegend(ax; position = :lb, framevisible = true,
                       labelsize = 11)

        # Annotation: fitted slopes at θ = 0.
        ann_text = "fitted slope (θ=0):\n" *
                   "  L∞ = $(round(slope_inf_θ0; digits = 3))\n" *
                   "  L²  = $(round(slope_l2_θ0; digits = 3))\n" *
                   "theory: −2.0"
        CM.text!(ax, 0.62, 0.92;
                   text = ann_text, space = :relative,
                   align = (:left, :top),
                   fontsize = 11)

        mkpath(dirname(save_path))
        CM.save(save_path, fig)
        return save_path
    catch e
        @warn "Plotting C.3 convergence figure failed: $(e). Saving CSV."
        mkpath(dirname(save_path))
        csv_path = replace(save_path, ".png" => ".csv")
        open(csv_path, "w") do f
            print(f, "level,N,dx")
            for θ in result.angles
                print(f, ",err_inf_theta=$(θ),err_l2_theta=$(θ)")
            end
            println(f)
            for li in eachindex(result.levels)
                print(f, "$(result.levels[li]),$(result.n_per_axis[li]),"
                          * "$(result.dx[li])")
                for ai in eachindex(result.angles)
                    print(f, ",$(result.err_inf[li, ai]),"
                              * "$(result.err_l2[li, ai])")
                end
                println(f)
            end
        end
        return csv_path
    end
end
