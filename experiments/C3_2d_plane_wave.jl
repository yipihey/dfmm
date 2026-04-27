# C3_2d_plane_wave.jl
#
# M3-4 Phase 2 Deliverable 5: C.3 2D plane-wave driver.
#
# §10.4 C.3 of the methods paper. Drives the 2D Cholesky-sector Newton
# system with an acoustic plane wave IC at angles θ ∈ {0, π/8, π/4, π/2}.
# Uses `tier_c_plane_wave_full_ic` (Phase 2a IC bridge) to populate the
# 12-field 2D field set.
#
# Acceptance gates (asserted in `test/test_M3_4_C3_plane_wave.jl`):
#
#   1. θ = 0 IC reduces to the 1D plane wave (u_y = 0; ρ deviation only
#      along x).
#   2. u parallel to k̂ at IC across all angles (right-going acoustic
#      mode).
#   3. Rotational invariance at θ = π/2: rotated IC matches the
#      θ = 0 IC with (u_x, u_y) ↔ (u_y, u_x), to mesh-discretization
#      tolerance.
#   4. Linear-acoustic mode is stable under implicit-midpoint Newton:
#      |u|_∞ stays bounded across short integration windows.
#   5. Conservation gates per angle: mass exact (ρ_per_cell fixed by
#      bridge); momentum ≤ 1e-9.
#
# Headline plot: `reference/figs/M3_4_C3_plane_wave_rotation_invariance.png`.
#
# Usage (REPL):
#   julia> include("experiments/C3_2d_plane_wave.jl")
#   julia> result = run_C3_plane_wave(; level=4, angle=π/4, n_steps=10)

using HierarchicalGrids: HierarchicalMesh, EulerianFrame, FrameBoundaries,
    REFLECTING, PERIODIC, enumerate_leaves, refine_cells!, cell_physical_box
using dfmm: tier_c_plane_wave_full_ic, primitive_recovery_2d_per_cell,
            det_step_2d_berry_HG!, read_detfield_2d, DetField2D

"""
    run_C3_plane_wave(; level=4, A=1e-3, k_mag=1, angle=0.0,
                       dt=5e-4, n_steps=10, ρ0=1.0, P0=1.0,
                       M_vv_override=(1.0, 1.0), ρ_ref=1.0)

Drive the C.3 plane-wave IC for `n_steps` Newton steps. Returns a
NamedTuple with the trajectory and the IC bridge's k̂ direction.
"""
function run_C3_plane_wave(; level::Integer = 4,
                            A::Real = 1e-3, k_mag::Real = 1,
                            angle::Real = 0.0,
                            dt::Real = 5e-4, n_steps::Integer = 10,
                            ρ0::Real = 1.0, P0::Real = 1.0,
                            M_vv_override = (1.0, 1.0),
                            ρ_ref::Real = 1.0)
    ic = tier_c_plane_wave_full_ic(; level = level, A = A, k_mag = k_mag,
                                     angle = angle, ρ0 = ρ0, P0 = P0)
    bc_spec = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                   (PERIODIC, PERIODIC)))

    t = zeros(Float64, n_steps + 1)
    u_max = zeros(Float64, n_steps + 1)
    u_max[1] = _u_max_norm(ic.fields, ic.leaves)

    for n in 1:n_steps
        det_step_2d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                                bc_spec, dt;
                                M_vv_override = M_vv_override, ρ_ref = ρ_ref)
        t[n + 1] = n * dt
        u_max[n + 1] = _u_max_norm(ic.fields, ic.leaves)
    end

    return (
        t = t, u_max = u_max,
        mesh = ic.mesh, frame = ic.frame, leaves = ic.leaves,
        fields = ic.fields, ρ_per_cell = ic.ρ_per_cell,
        params = ic.params,
    )
end

function _u_max_norm(fields, leaves)
    m = 0.0
    for ci in leaves
        u1 = Float64(fields.u_1[ci][1])
        u2 = Float64(fields.u_2[ci][1])
        m = max(m, sqrt(u1^2 + u2^2))
    end
    return m
end

"""
    plot_C3_rotation_invariance(; angles=(0.0, π/8, π/4, π/2),
                                  level=4, A=1e-3, save_path=nothing)

Build a panel plot showing the |u| trajectory for each angle and the
sample acoustic-mode profile at θ = π/4 (showing the rotated wave on
the 2D mesh).
"""
function plot_C3_rotation_invariance(; angles = (0.0, π/8, π/4, π/2),
                                       level::Integer = 4, A::Real = 1e-3,
                                       n_steps::Integer = 10,
                                       dt::Real = 5e-4,
                                       save_path::AbstractString =
                                          joinpath(@__DIR__, "..",
                                                    "reference", "figs",
                                                    "M3_4_C3_plane_wave_rotation_invariance.png"))
    results = [run_C3_plane_wave(; level = level, A = A, angle = θ,
                                   n_steps = n_steps, dt = dt)
               for θ in angles]
    try
        Base.require(Main, :CairoMakie)
        CM = getfield(Main, :CairoMakie)
        fig = CM.Figure(size = (1100, 500))
        ax = CM.Axis(fig[1, 1]; title = "|u|_∞ vs t per angle",
                       xlabel = "t", ylabel = "max |u|")
        for (θ, res) in zip(angles, results)
            CM.lines!(ax, res.t, res.u_max, label = "θ=$(round(θ; digits = 3))")
        end
        CM.axislegend(ax; position = :rt)

        # Sample θ = π/4 ρ field.
        θ_pick_idx = argmin(abs.([θ - π/4 for θ in angles]))
        res_pick = results[θ_pick_idx]
        rec = primitive_recovery_2d_per_cell(res_pick.fields, res_pick.leaves,
                                               res_pick.frame, res_pick.ρ_per_cell)
        ax2 = CM.Axis(fig[1, 2]; title = "ρ field at θ = π/4",
                       xlabel = "x", ylabel = "y")
        CM.scatter!(ax2, rec.x, rec.y; color = rec.ρ, markersize = 8)

        mkpath(dirname(save_path))
        CM.save(save_path, fig)
        return save_path
    catch e
        @warn "Plotting C.3 figure failed: $(e). Saving CSV trajectory instead."
        mkpath(dirname(save_path))
        csv_path = replace(save_path, ".png" => ".csv")
        open(csv_path, "w") do f
            println(f, "angle,t,u_max")
            for (θ, res) in zip(angles, results)
                for k in eachindex(res.t)
                    println(f, "$θ,$(res.t[k]),$(res.u_max[k])")
                end
            end
        end
        return csv_path
    end
end
