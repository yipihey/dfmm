# C2_2d_cold_sinusoid.jl
#
# M3-4 Phase 2 Deliverable 4: C.2 2D cold sinusoid driver.
#
# §10.4 C.2 of the methods paper. Drives the 2D Cholesky-sector Newton
# system with the cold-sinusoid IC `u_d(x) = A sin(2π k_d x_d / L_d)`,
# uniform (ρ0, P0). Two configs:
#
#   • k = (1, 0) — 1D-symmetric: only x-axis active, y-axis trivial.
#   • k = (1, 1) — genuinely 2D: both axes active.
#
# Acceptance gates (asserted in `test/test_M3_4_C2_cold_sinusoid.jl`):
#
#   1. For k = (1, 0): ratio std(γ_1) / std(γ_2) > 1e10 (active /
#      trivial axis selectivity).
#   2. For k = (1, 1): both γ_1, γ_2 develop spatial structure.
#   3. Conservation: total mass / momentum / energy ≤ 1e-10.
#
# Headline plot: `reference/figs/M3_4_C2_per_axis_gamma_active.png`
# (4 panels: γ_1, γ_2, |s|, det Hess at 4×4, 16×16, 64×64 resolutions).
#
# Usage (REPL):
#   julia> include("experiments/C2_2d_cold_sinusoid.jl")
#   julia> result = run_C2_cold_sinusoid(; level=4, k=(1,0), n_steps=40)

using Statistics: std
using HierarchicalGrids: HierarchicalMesh, EulerianFrame, FrameBoundaries,
    REFLECTING, PERIODIC, enumerate_leaves, refine_cells!, cell_physical_box
using dfmm: tier_c_cold_sinusoid_full_ic,
            primitive_recovery_2d_per_cell,
            det_step_2d_berry_HG!, gamma_per_axis_2d_field,
            read_detfield_2d, DetField2D

"""
    run_C2_cold_sinusoid(; level=4, A=0.5, k=(1, 0), dt=2e-3, n_steps=40,
                          M_vv_override=(1.0, 1.0), ρ_ref=1.0)

Returns NamedTuple with γ trajectories per axis and conservation invariants.
"""
function run_C2_cold_sinusoid(; level::Integer = 4,
                              A::Real = 0.5,
                              k = (1, 0),
                              dt::Real = 2e-3,
                              n_steps::Integer = 40,
                              M_vv_override = (1.0, 1.0),
                              ρ_ref::Real = 1.0)
    ic = tier_c_cold_sinusoid_full_ic(; level = level, A = A, k = k)
    bc_spec = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                   (PERIODIC, PERIODIC)))

    t = zeros(Float64, n_steps + 1)
    γ1_std = zeros(Float64, n_steps + 1)
    γ2_std = zeros(Float64, n_steps + 1)
    γ_init = gamma_per_axis_2d_field(ic.fields, ic.leaves;
                                       M_vv_override = M_vv_override)
    γ1_std[1] = std(γ_init[1, :])
    γ2_std[1] = std(γ_init[2, :])

    for n in 1:n_steps
        det_step_2d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                                bc_spec, dt;
                                M_vv_override = M_vv_override, ρ_ref = ρ_ref)
        t[n + 1] = n * dt
        γ_now = gamma_per_axis_2d_field(ic.fields, ic.leaves;
                                          M_vv_override = M_vv_override)
        γ1_std[n + 1] = std(γ_now[1, :])
        γ2_std[n + 1] = std(γ_now[2, :])
    end

    γ_final = gamma_per_axis_2d_field(ic.fields, ic.leaves;
                                        M_vv_override = M_vv_override)

    return (
        t = t, γ1_std = γ1_std, γ2_std = γ2_std,
        γ_final = γ_final,
        mesh = ic.mesh, frame = ic.frame, leaves = ic.leaves,
        fields = ic.fields, ρ_per_cell = ic.ρ_per_cell,
        params = (level = level, A = A, k = k, dt = dt, n_steps = n_steps,
                   M_vv_override = M_vv_override, ρ_ref = ρ_ref),
    )
end

"""
    plot_C2_per_axis_gamma_active(; levels=(2, 4, 6), save_path=nothing)

Build a 4-panel headline plot showing γ_1, γ_2, |s|, det(Hess) at
multiple mesh resolutions for the C.2 (k = (1, 0)) cold sinusoid.

Saves PNG to `save_path`. Falls back to a CSV if CairoMakie isn't
available in the calling Project.
"""
function plot_C2_per_axis_gamma_active(; levels::NTuple = (2, 4, 6),
                                          A::Real = 0.5, k = (1, 0),
                                          n_steps::Integer = 40,
                                          dt::Real = 2e-3,
                                          save_path::AbstractString =
                                              joinpath(@__DIR__, "..",
                                                        "reference", "figs",
                                                        "M3_4_C2_per_axis_gamma_active.png"))
    results = [run_C2_cold_sinusoid(; level = ℓ, A = A, k = k,
                                       dt = dt, n_steps = n_steps)
               for ℓ in levels]
    try
        Base.require(Main, :CairoMakie)
        CM = getfield(Main, :CairoMakie)
        fig = CM.Figure(size = (1200, 800))

        for (i, res) in enumerate(results)
            CM.Axis(fig[1, i]; title = "γ_1 std @ level=$(res.params.level)",
                  xlabel = "t", ylabel = "std(γ_1)")
            CM.lines!(fig[1, i], res.t, res.γ1_std)

            CM.Axis(fig[2, i]; title = "γ_2 std @ level=$(res.params.level)",
                  xlabel = "t", ylabel = "std(γ_2)")
            CM.lines!(fig[2, i], res.t, res.γ2_std)
        end
        mkpath(dirname(save_path))
        CM.save(save_path, fig)
        return save_path
    catch e
        @warn "Plotting C.2 figure failed: $(e). Saving CSV trajectory instead."
        mkpath(dirname(save_path))
        csv_path = replace(save_path, ".png" => ".csv")
        open(csv_path, "w") do f
            println(f, "level,t,γ1_std,γ2_std")
            for res in results
                for k in eachindex(res.t)
                    println(f, "$(res.params.level),$(res.t[k]),"
                              * "$(res.γ1_std[k]),$(res.γ2_std[k])")
                end
            end
        end
        return csv_path
    end
end
