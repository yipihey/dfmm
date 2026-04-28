# D4_zeldovich_3d.jl
#
# §10.5 D.4 3D Zel'dovich pancake collapse driver — M3-7e (the 3D
# analog of `experiments/D4_zeldovich_pancake.jl`).
#
# The headline 3D scientific test of M3-7. Drives the M3-7e
# `tier_d_zeldovich_pancake_3d_ic_full` IC (1D-symmetric pancake along
# axis 1; trivial axes 2 and 3) through `det_step_3d_berry_HG!`. The
# central scientific gate is per-axis γ selectivity in 3D:
#
#   • γ_1 (collapsing axis) develops spatial structure as t → t_cross.
#   • γ_2, γ_3 (trivial axes) stay uniform across leaves (std → 0 to
#     round-off).
#   • Spatial std ratio std(γ_1) / (std(γ_2) + std(γ_3) + eps) → very
#     large at near-caustic.
#
# Caustic time: t_cross = 1 / (A · 2π) ≈ 0.318 at A = 0.5.
#
# Usage:
#   julia> include("experiments/D4_zeldovich_3d.jl")
#   julia> result = run_D4_zeldovich_pancake_3d(; level=2, A=0.5)

using HierarchicalGrids: HierarchicalMesh, EulerianFrame,
    refine_cells!, enumerate_leaves, FrameBoundaries, REFLECTING, PERIODIC,
    cell_physical_box
using dfmm: tier_d_zeldovich_pancake_3d_ic_full,
    det_step_3d_berry_HG!, gamma_per_axis_3d_field,
    read_detfield_3d, DetField3D
using Statistics: mean, std

"""
    zeldovich_caustic_time_3d(; A=0.5) -> Float64

Analytic caustic time for the 3D Zel'dovich pancake (same as 1D / 2D).
"""
zeldovich_caustic_time_3d(; A::Real = 0.5) = 1.0 / (Float64(A) * 2π)

"""
    pancake_3d_per_axis_uniformity(fields, leaves; M_vv_override, ρ_ref)
        -> NamedTuple

Per-step diagnostic: per-axis γ stats and the 3D selectivity ratio
`std(γ_1) / (std(γ_2) + std(γ_3) + eps)`. The pancake direction is
axis 1; axes 2 and 3 are trivial.
"""
function pancake_3d_per_axis_uniformity(fields, leaves;
                                          M_vv_override = (1.0, 1.0, 1.0),
                                          ρ_ref::Real = 1.0)
    γ = gamma_per_axis_3d_field(fields, leaves;
                                  M_vv_override = M_vv_override,
                                  ρ_ref = ρ_ref)
    γ1_arr = γ[1, :]; γ2_arr = γ[2, :]; γ3_arr = γ[3, :]
    γ1_mean = mean(γ1_arr); γ1_std = std(γ1_arr)
    γ2_mean = mean(γ2_arr); γ2_std = std(γ2_arr)
    γ3_mean = mean(γ3_arr); γ3_std = std(γ3_arr)
    γ1_min = minimum(γ1_arr); γ1_max = maximum(γ1_arr)
    γ2_min = minimum(γ2_arr); γ2_max = maximum(γ2_arr)
    γ3_min = minimum(γ3_arr); γ3_max = maximum(γ3_arr)
    sel = γ1_std / (γ2_std + γ3_std + eps(Float64))
    return (γ1_mean = γ1_mean, γ1_std = γ1_std,
            γ1_min = γ1_min, γ1_max = γ1_max,
            γ2_mean = γ2_mean, γ2_std = γ2_std,
            γ2_min = γ2_min, γ2_max = γ2_max,
            γ3_mean = γ3_mean, γ3_std = γ3_std,
            γ3_min = γ3_min, γ3_max = γ3_max,
            selectivity_ratio = sel)
end

"""
    cell_volumes_3d(frame, leaves) -> Vector{Float64}

Per-leaf 3D cell volume `(hi_x - lo_x)·(hi_y - lo_y)·(hi_z - lo_z)`.
"""
function cell_volumes_3d(frame, leaves)
    V = Vector{Float64}(undef, length(leaves))
    @inbounds for (i, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        V[i] = (hi_c[1] - lo_c[1]) * (hi_c[2] - lo_c[2]) * (hi_c[3] - lo_c[3])
    end
    return V
end

"""
    conservation_invariants_3d(fields, leaves, ρ_per_cell, vols) -> NamedTuple

Compute (M, Px, Py, Pz, KE) from leaf field set + cell volumes.
"""
function conservation_invariants_3d(fields, leaves, ρ_per_cell, vols)
    M = 0.0; Px = 0.0; Py = 0.0; Pz = 0.0; KE = 0.0
    @inbounds for (i, ci) in enumerate(leaves)
        ρ_i = Float64(ρ_per_cell[i])
        V_i = Float64(vols[i])
        ux = Float64(fields.u_1[ci][1])
        uy = Float64(fields.u_2[ci][1])
        uz = Float64(fields.u_3[ci][1])
        M += ρ_i * V_i
        Px += ρ_i * ux * V_i
        Py += ρ_i * uy * V_i
        Pz += ρ_i * uz * V_i
        KE += 0.5 * ρ_i * (ux * ux + uy * uy + uz * uz) * V_i
    end
    return (M = M, Px = Px, Py = Py, Pz = Pz, KE = KE)
end

"""
    run_D4_zeldovich_pancake_3d(; level=2, A=0.5, ρ0=1.0, P0=1e-6,
                                  dt=nothing, T_factor=0.5,
                                  M_vv_override=(1.0, 1.0, 1.0), ρ_ref=1.0,
                                  verbose=false) -> NamedTuple

Drive a single 3D Zel'dovich pancake trajectory. Builds the IC via
`tier_d_zeldovich_pancake_3d_ic_full(level=level, A=A)`. Runs to
`T_end = T_factor · t_cross` (default `0.5 · t_cross`, well pre-caustic).

Records per-step per-axis γ stats, conservation invariants, and a few
spatial snapshots. Returns a NamedTuple.

Default `M_vv_override = (1, 1, 1)` decouples per-axis γ from the EOS
J-tracking (matches the M3-7d cold-sinusoid driver convention).
"""
function run_D4_zeldovich_pancake_3d(; level::Integer = 2,
                                        A::Real = 0.5,
                                        ρ0::Real = 1.0,
                                        P0::Real = 1e-6,
                                        dt::Union{Real, Nothing} = nothing,
                                        T_factor::Real = 0.5,
                                        M_vv_override = (1.0, 1.0, 1.0),
                                        ρ_ref::Real = 1.0,
                                        verbose::Bool = false)
    t_cross = zeldovich_caustic_time_3d(; A = A)
    T_end = Float64(T_factor) * t_cross
    if dt === nothing
        Δx = 1.0 / (2^Int(level))
        dt_val = 0.25 * Δx / (Float64(A) * 2π)
        # Cap so we always get at least 10 samples even at low resolution.
        dt_val = min(dt_val, T_end / 10.0)
    else
        dt_val = Float64(dt)
    end
    n_steps = max(1, Int(ceil(T_end / dt_val)))
    dt_val = T_end / n_steps

    ic = tier_d_zeldovich_pancake_3d_ic_full(; level = level, A = A,
                                                ρ0 = ρ0, P0 = P0)
    # Pancake BCs: PERIODIC along axis 1 (collapsing); REFLECTING along
    # axes 2, 3 (trivial).
    bc_spec = FrameBoundaries{3}(((PERIODIC, PERIODIC),
                                    (REFLECTING, REFLECTING),
                                    (REFLECTING, REFLECTING)))
    vols = cell_volumes_3d(ic.frame, ic.leaves)

    N = n_steps + 1
    t = zeros(Float64, N)
    γ1_max = zeros(Float64, N); γ1_min = zeros(Float64, N); γ1_std = zeros(Float64, N)
    γ2_max = zeros(Float64, N); γ2_min = zeros(Float64, N); γ2_std = zeros(Float64, N)
    γ3_max = zeros(Float64, N); γ3_min = zeros(Float64, N); γ3_std = zeros(Float64, N)
    selectivity_ratio = zeros(Float64, N)
    M_traj = zeros(Float64, N)
    Px_traj = zeros(Float64, N); Py_traj = zeros(Float64, N); Pz_traj = zeros(Float64, N)
    KE_traj = zeros(Float64, N)

    diag0 = pancake_3d_per_axis_uniformity(ic.fields, ic.leaves;
                                             M_vv_override = M_vv_override,
                                             ρ_ref = ρ_ref)
    γ1_max[1] = diag0.γ1_max; γ1_min[1] = diag0.γ1_min; γ1_std[1] = diag0.γ1_std
    γ2_max[1] = diag0.γ2_max; γ2_min[1] = diag0.γ2_min; γ2_std[1] = diag0.γ2_std
    γ3_max[1] = diag0.γ3_max; γ3_min[1] = diag0.γ3_min; γ3_std[1] = diag0.γ3_std
    selectivity_ratio[1] = diag0.selectivity_ratio
    cons0 = conservation_invariants_3d(ic.fields, ic.leaves, ic.ρ_per_cell, vols)
    M_traj[1] = cons0.M; Px_traj[1] = cons0.Px
    Py_traj[1] = cons0.Py; Pz_traj[1] = cons0.Pz
    KE_traj[1] = cons0.KE

    nan_seen = false
    wall_t0 = time()
    for n in 1:n_steps
        try
            det_step_3d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                                    bc_spec, dt_val;
                                    M_vv_override = M_vv_override,
                                    ρ_ref = ρ_ref)
        catch e
            if verbose
                @warn "Newton solve failed at step $n: $e"
            end
            nan_seen = true
            break
        end
        t[n + 1] = n * dt_val
        d = pancake_3d_per_axis_uniformity(ic.fields, ic.leaves;
                                             M_vv_override = M_vv_override,
                                             ρ_ref = ρ_ref)
        γ1_max[n + 1] = d.γ1_max; γ1_min[n + 1] = d.γ1_min; γ1_std[n + 1] = d.γ1_std
        γ2_max[n + 1] = d.γ2_max; γ2_min[n + 1] = d.γ2_min; γ2_std[n + 1] = d.γ2_std
        γ3_max[n + 1] = d.γ3_max; γ3_min[n + 1] = d.γ3_min; γ3_std[n + 1] = d.γ3_std
        selectivity_ratio[n + 1] = d.selectivity_ratio
        cons = conservation_invariants_3d(ic.fields, ic.leaves, ic.ρ_per_cell, vols)
        M_traj[n + 1] = cons.M; Px_traj[n + 1] = cons.Px
        Py_traj[n + 1] = cons.Py; Pz_traj[n + 1] = cons.Pz
        KE_traj[n + 1] = cons.KE
        if !isfinite(d.γ1_max)
            nan_seen = true
            break
        end
    end
    wall_t1 = time()
    wall_time_per_step = (wall_t1 - wall_t0) / max(n_steps, 1)

    M_err_max = maximum(abs.(M_traj .- M_traj[1]))
    Px_err_max = maximum(abs.(Px_traj .- Px_traj[1]))
    Py_err_max = maximum(abs.(Py_traj .- Py_traj[1]))
    Pz_err_max = maximum(abs.(Pz_traj .- Pz_traj[1]))
    KE_err_max = maximum(abs.(KE_traj .- KE_traj[1]))

    # Final γ field for spatial inspection.
    γ_final = gamma_per_axis_3d_field(ic.fields, ic.leaves;
                                        M_vv_override = M_vv_override,
                                        ρ_ref = ρ_ref)
    # Cell-center coordinates + Lagrangian deviation in axis 1.
    x_centers = Vector{Float64}(undef, length(ic.leaves))
    y_centers = Vector{Float64}(undef, length(ic.leaves))
    z_centers = Vector{Float64}(undef, length(ic.leaves))
    x_dev_axis1 = Vector{Float64}(undef, length(ic.leaves))
    @inbounds for (i, ci) in enumerate(ic.leaves)
        lo_c, hi_c = cell_physical_box(ic.frame, ci)
        cx = 0.5 * (lo_c[1] + hi_c[1])
        cy = 0.5 * (lo_c[2] + hi_c[2])
        cz = 0.5 * (lo_c[3] + hi_c[3])
        x_centers[i] = cx; y_centers[i] = cy; z_centers[i] = cz
        x_dev_axis1[i] = Float64(ic.fields.x_1[ci][1]) - cx
    end

    return (
        t = t,
        γ1_max = γ1_max, γ1_min = γ1_min, γ1_std = γ1_std,
        γ2_max = γ2_max, γ2_min = γ2_min, γ2_std = γ2_std,
        γ3_max = γ3_max, γ3_min = γ3_min, γ3_std = γ3_std,
        selectivity_ratio = selectivity_ratio,
        M_traj = M_traj, Px_traj = Px_traj, Py_traj = Py_traj, Pz_traj = Pz_traj,
        KE_traj = KE_traj,
        M_err_max = M_err_max,
        Px_err_max = Px_err_max, Py_err_max = Py_err_max, Pz_err_max = Pz_err_max,
        KE_err_max = KE_err_max,
        γ_final = γ_final,
        x_centers = x_centers, y_centers = y_centers, z_centers = z_centers,
        x_dev_axis1 = x_dev_axis1,
        t_cross = t_cross,
        wall_time_per_step = wall_time_per_step,
        nan_seen = nan_seen,
        ic = ic,
        params = (level = level, A = Float64(A),
                   ρ0 = ρ0, P0 = P0,
                   dt = dt_val, n_steps = n_steps, T_end = T_end,
                   T_factor = T_factor,
                   M_vv_override = M_vv_override, ρ_ref = ρ_ref),
    )
end

"""
    plot_D4_zeldovich_3d(result; save_path) -> save_path

4-panel CairoMakie headline figure for the 3D Zel'dovich pancake:
  • Panel A: γ_1 spatial profile along axis 1 at the final time
             (slice through mid-y, mid-z).
  • Panel B: γ_2 and γ_3 std vs time (uniformity check).
  • Panel C: |x_1 - cell_center| spatial profile (Lagrangian
             deviation along axis 1).
  • Panel D: log10(γ_1) along axis 1 at near-caustic.

Falls back to CSV if CairoMakie is not available.
"""
function plot_D4_zeldovich_3d(result; save_path::AbstractString)
    try
        CM = if isdefined(Main, :CairoMakie)
            getfield(Main, :CairoMakie)
        else
            Base.require(Main, :CairoMakie)
            getfield(Main, :CairoMakie)
        end
        # Slice along axis 1 at the cell column closest to (mid-y, mid-z).
        y_unique = sort!(unique(round.(result.y_centers; digits = 12)))
        z_unique = sort!(unique(round.(result.z_centers; digits = 12)))
        y_mid = y_unique[max(1, length(y_unique) ÷ 2)]
        z_mid = z_unique[max(1, length(z_unique) ÷ 2)]
        keep = findall(i -> isapprox(result.y_centers[i], y_mid; atol = 1e-12) &&
                              isapprox(result.z_centers[i], z_mid; atol = 1e-12),
                       eachindex(result.y_centers))
        ord = sortperm(result.x_centers[keep])
        kk = keep[ord]
        xs = result.x_centers[kk]
        γ1_slice = result.γ_final[1, kk]
        x_dev_slice = result.x_dev_axis1[kk]

        fig = CM.Figure(size = (1100, 850))

        axA = CM.Axis(fig[1, 1];
            title = "A: γ_1 spatial profile (axis 1, mid y, mid z)",
            xlabel = "x_1", ylabel = "γ_1")
        CM.lines!(axA, xs, γ1_slice;
                  label = "t = $(round(result.t[end]; sigdigits=3))")
        CM.axislegend(axA; position = :rb)

        axB = CM.Axis(fig[1, 2];
            title = "B: std(γ_2), std(γ_3) vs time (trivial axes)",
            xlabel = "t", ylabel = "std")
        CM.lines!(axB, result.t, result.γ2_std; label = "std(γ_2)")
        CM.lines!(axB, result.t, result.γ3_std; label = "std(γ_3)")
        CM.axislegend(axB; position = :rt)

        axC = CM.Axis(fig[2, 1];
            title = "C: |x_1 − cell_center| (Lagrangian deviation, axis 1)",
            xlabel = "x_1", ylabel = "|x_1 − x_center|")
        CM.lines!(axC, xs, abs.(x_dev_slice))

        axD = CM.Axis(fig[2, 2];
            title = "D: log10(γ_1) at t=$(round(result.t[end]; sigdigits=3))",
            xlabel = "x_1", ylabel = "log10(γ_1)")
        CM.lines!(axD, xs, log10.(max.(γ1_slice, 1e-300)))

        mkpath(dirname(save_path))
        CM.save(save_path, fig)
        return save_path
    catch e
        @warn "Plotting D.4 3D Zel'dovich failed: $(e). Saving CSV instead."
        mkpath(dirname(save_path))
        csv = replace(save_path, ".png" => ".csv")
        open(csv, "w") do f
            println(f, "t,γ1_min,γ1_max,γ1_std,γ2_std,γ3_std,sel,KE")
            for k in eachindex(result.t)
                println(f, "$(result.t[k]),$(result.γ1_min[k]),$(result.γ1_max[k])," *
                          "$(result.γ1_std[k]),$(result.γ2_std[k])," *
                          "$(result.γ3_std[k]),$(result.selectivity_ratio[k])," *
                          "$(result.KE_traj[k])")
            end
        end
        return save_path
    end
end
