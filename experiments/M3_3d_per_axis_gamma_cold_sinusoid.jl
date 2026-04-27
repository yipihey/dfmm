# M3_3d_per_axis_gamma_cold_sinusoid.jl
#
# §6.5 Per-axis γ selectivity test (M3-3 design note `reference/notes_M3_3_2d_cholesky_berry.md`):
#
# Cold sinusoid IC with k_x = 1, k_y = 0:
#   ρ uniform, u = (A sin(2π x), 0), A = 0.5.
# Run on a 2D Cholesky-sector field set for a short window (pre-shell-
# crossing). Verify that the per-axis γ diagnostic correctly identifies
# the collapsing axis:
#
#   γ_1 (along x — collapsing) → small
#   γ_2 (along y — trivial)    → O(1)
#   γ_2 / γ_1 → grows large    (selectivity ratio)
#
# Saves a 4-panel plot to `reference/figs/M3_3d_per_axis_gamma_selectivity.png`
# showing γ_1, γ_2, |s|, and det(Hess) per cell at the final step.
#
# Usage (REPL):
#   julia> include("experiments/M3_3d_per_axis_gamma_cold_sinusoid.jl")
#   julia> result = run_M3_3d_per_axis_gamma_selectivity()
#
# Returns a NamedTuple with the trajectory + final γ ratio for use by
# `test/test_M3_3d_selectivity.jl`.

using StaticArrays
using HierarchicalGrids: HierarchicalMesh, EulerianFrame,
    refine_cells!, enumerate_leaves, FrameBoundaries, REFLECTING, PERIODIC,
    cell_physical_box
using dfmm: allocate_cholesky_2d_fields, write_detfield_2d!,
            read_detfield_2d, DetField2D,
            det_step_2d_berry_HG!, gamma_per_axis_2d_field

"""
    init_cold_sinusoid_2d!(fields, mesh, frame, leaves;
                            A=0.5, kx=1, ky=0,
                            α0=1.0, β0=0.0, s0=0.0)

Initialize a 2D Cholesky-sector field set with the C.2 cold-sinusoid IC:

    ρ uniform (per cell), u_1(x) = A sin(2π kx (x_1 - lo_1) / L_1),
    u_2(y) = A sin(2π ky (x_2 - lo_2) / L_2)         (= 0 if ky == 0).

α_a = α0 (constant), β_a = β0 (constant — cold limit), θ_R = 0,
s = s0 (uniform). The post-Newton sectors `Pp = 0`, `Q = 0`.
"""
function init_cold_sinusoid_2d!(fields, mesh, frame, leaves;
                                  A::Real = 0.5,
                                  kx::Integer = 1, ky::Integer = 0,
                                  α0::Real = 1.0, β0::Real = 0.0,
                                  s0::Real = 0.0)
    lo_box = (frame.lo[1], frame.lo[2])
    hi_box = (frame.hi[1], frame.hi[2])
    L1 = hi_box[1] - lo_box[1]
    L2 = hi_box[2] - lo_box[2]
    for ci in leaves
        lo, hi = cell_physical_box(frame, ci)
        cx = 0.5 * (lo[1] + hi[1])
        cy = 0.5 * (lo[2] + hi[2])
        u1 = kx == 0 ? 0.0 : A * sin(2π * kx * (cx - lo_box[1]) / L1)
        u2 = ky == 0 ? 0.0 : A * sin(2π * ky * (cy - lo_box[2]) / L2)
        v = DetField2D((cx, cy), (u1, u2),
                        (α0, α0), (β0, β0),
                        0.0, s0, 0.0, 0.0)
        write_detfield_2d!(fields, ci, v)
    end
    return fields
end

"""
    run_M3_3d_per_axis_gamma_selectivity(; level=4, A=0.5, kx=1, ky=0,
                                            dt=2e-3, n_steps=80,
                                            M_vv_override=(1.0, 1.0),
                                            ρ_ref=1.0,
                                            save_path=nothing) -> NamedTuple

Drive the §6.5 per-axis γ selectivity test. Builds a 2D `HierarchicalMesh{2}`
of `(2^level)^2` leaves, initializes the cold sinusoid IC with `(kx, ky)`,
and runs `n_steps` of `det_step_2d_berry_HG!`. At each step records
the per-axis γ diagnostic (mean across cells along each axis) and
returns a trajectory.

Returns:
  • `t::Vector{Float64}`             — per-step time
  • `γ1_max::Vector{Float64}`        — max_cell γ_1(t)
  • `γ1_min::Vector{Float64}`        — min_cell γ_1(t)
  • `γ2_max::Vector{Float64}`        — max_cell γ_2(t)
  • `γ2_min::Vector{Float64}`        — min_cell γ_2(t)
  • `γ_final::Matrix{Float64}`       — 2 × N final per-axis γ
  • `params::NamedTuple`             — IC parameters
"""
function run_M3_3d_per_axis_gamma_selectivity(; level::Integer = 4,
                                                A::Real = 0.5,
                                                kx::Integer = 1, ky::Integer = 0,
                                                dt::Real = 2e-3,
                                                n_steps::Integer = 80,
                                                M_vv_override = (1.0, 1.0),
                                                ρ_ref::Real = 1.0)
    mesh = HierarchicalMesh{2}(; balanced = true)
    for _ in 1:level
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
    fields = allocate_cholesky_2d_fields(mesh)
    init_cold_sinusoid_2d!(fields, mesh, frame, leaves;
                             A = A, kx = kx, ky = ky)

    # Reflecting BCs sidestep the periodic-x coordinate-wrap issue
    # documented in the M3-3c handoff (the residual treats x_a as
    # per-cell scalar without periodic wrap-around; periodic-x only
    # works for cold-limit / zero-strain ICs). For the cold sinusoid
    # the IC has u(0) = u(1) = 0 so reflecting BCs are physically
    # consistent over short integration windows.
    bc_spec = FrameBoundaries{2}(((REFLECTING, REFLECTING),
                                    (REFLECTING, REFLECTING)))

    t = zeros(Float64, n_steps + 1)
    γ1_max = zeros(Float64, n_steps + 1)
    γ1_min = zeros(Float64, n_steps + 1)
    γ2_max = zeros(Float64, n_steps + 1)
    γ2_min = zeros(Float64, n_steps + 1)
    γ_init = gamma_per_axis_2d_field(fields, leaves;
                                       M_vv_override = M_vv_override,
                                       ρ_ref = ρ_ref)
    γ1_max[1] = maximum(γ_init[1, :]); γ1_min[1] = minimum(γ_init[1, :])
    γ2_max[1] = maximum(γ_init[2, :]); γ2_min[1] = minimum(γ_init[2, :])

    for n in 1:n_steps
        det_step_2d_berry_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                                M_vv_override = M_vv_override,
                                ρ_ref = ρ_ref)
        t[n + 1] = n * dt
        γ_now = gamma_per_axis_2d_field(fields, leaves;
                                          M_vv_override = M_vv_override,
                                          ρ_ref = ρ_ref)
        γ1_max[n + 1] = maximum(γ_now[1, :])
        γ1_min[n + 1] = minimum(γ_now[1, :])
        γ2_max[n + 1] = maximum(γ_now[2, :])
        γ2_min[n + 1] = minimum(γ_now[2, :])
    end

    γ_final = gamma_per_axis_2d_field(fields, leaves;
                                        M_vv_override = M_vv_override,
                                        ρ_ref = ρ_ref)

    return (
        t = t, γ1_max = γ1_max, γ1_min = γ1_min,
        γ2_max = γ2_max, γ2_min = γ2_min,
        γ_final = γ_final,
        mesh = mesh, frame = frame, leaves = leaves,
        fields = fields,
        params = (level = level, A = A, kx = kx, ky = ky,
                  dt = dt, n_steps = n_steps,
                  M_vv_override = M_vv_override, ρ_ref = ρ_ref),
    )
end

"""
    plot_M3_3d_per_axis_gamma(result; save_path)

4-panel headline plot from a `run_M3_3d_per_axis_gamma_selectivity`
result: γ_1 trajectory, γ_2 trajectory, |s| histogram at final step,
det(Hess) histogram at final step.

Saves PNG to `save_path`. Falls back to a no-op if `CairoMakie` is not
available in the calling Project.
"""
function plot_M3_3d_per_axis_gamma(result;
                                    save_path::AbstractString)
    try
        # Lazy load CairoMakie so the experiment driver doesn't pay the
        # ~5 s precompile every time it's `include`d. The Project's deps
        # already pin CairoMakie; we just need the symbol bound here.
        Base.require(Main, :CairoMakie)
        CM = getfield(Main, :CairoMakie)
        fig = CM.Figure(size = (900, 700))
        ax1 = CM.Axis(fig[1, 1];
            title = "γ_1 (collapsing x-axis)",
            xlabel = "t", ylabel = "γ_1")
        CM.lines!(ax1, result.t, result.γ1_min, label = "min")
        CM.lines!(ax1, result.t, result.γ1_max, label = "max")
        CM.axislegend(ax1; position = :rb)

        ax2 = CM.Axis(fig[1, 2];
            title = "γ_2 (trivial y-axis)",
            xlabel = "t", ylabel = "γ_2")
        CM.lines!(ax2, result.t, result.γ2_min, label = "min")
        CM.lines!(ax2, result.t, result.γ2_max, label = "max")
        CM.axislegend(ax2; position = :rb)

        # Final |s| (entropy) snapshot per cell.
        s_final = [Float64(result.fields.s[ci][1]) for ci in result.leaves]
        ax3 = CM.Axis(fig[2, 1];
            title = "|s| final", xlabel = "cell index", ylabel = "|s|")
        CM.scatter!(ax3, abs.(s_final))

        # Final det(Hess) per cell, computed per-axis as
        # det = -α² γ² - 4 α² β² (M1 form, axis-1 only here).
        det_h1 = Float64[]
        for ci in result.leaves
            v = read_detfield_2d(result.fields, ci)
            α1 = v.alphas[1]; β1 = v.betas[1]
            γ1 = sqrt(max(1.0 - β1^2, 0.0))
            push!(det_h1, -α1^2 * γ1^2 - 4 * α1^2 * β1^2)
        end
        ax4 = CM.Axis(fig[2, 2];
            title = "det Hess (axis 1) final",
            xlabel = "cell index", ylabel = "det")
        CM.scatter!(ax4, det_h1)

        mkpath(dirname(save_path))
        CM.save(save_path, fig)
        return save_path
    catch e
        @warn "Plotting M3-3d figure failed: $(e). Saving CSV trajectory instead."
        # Fallback: save raw trajectory as CSV.
        mkpath(dirname(save_path))
        open(replace(save_path, ".png" => ".csv"), "w") do f
            println(f, "t,γ1_min,γ1_max,γ2_min,γ2_max")
            for k in eachindex(result.t)
                println(f, "$(result.t[k]),$(result.γ1_min[k]),"
                          * "$(result.γ1_max[k]),$(result.γ2_min[k]),"
                          * "$(result.γ2_max[k])")
            end
        end
        return save_path
    end
end
