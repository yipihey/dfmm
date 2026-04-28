# C2_3d_cold_sinusoid.jl
#
# M3-7e: C.2 3D cold sinusoid driver. The 3D analog of
# `experiments/C2_2d_cold_sinusoid.jl`.
#
# Drives the 3D Cholesky-sector Newton system (`det_step_3d_berry_HG!`)
# with the C.2 cold sinusoid IC built via `tier_c_cold_sinusoid_3d_full_ic`
# (M3-7e Phase a). The IC seeds a per-axis sinusoidal velocity:
#
#   u_d(x) = A · sin(2π · k_d · (x_d - lo_d) / L_d)    for d ∈ {1, 2, 3}
#
# Active axes are those with `k_d ≠ 0`. The headline gate (asserted in
# `test/test_M3_7e_C2_3d_cold_sinusoid.jl`) is the §7.5 per-axis γ
# selectivity reproduction: for `k = (1, 0, 0)` the spatial std ratio
# `std(γ_1) / (std(γ_2) + std(γ_3) + eps)` exceeds the M3-7d gate of 1e6.

using HierarchicalGrids: HierarchicalMesh, EulerianFrame, FrameBoundaries,
    REFLECTING, PERIODIC, enumerate_leaves, refine_cells!, cell_physical_box
using dfmm: tier_c_cold_sinusoid_3d_full_ic,
            det_step_3d_berry_HG!, gamma_per_axis_3d_field
using Statistics: std

"""
    run_C2_3d_cold_sinusoid(; level=2, A=0.3, k=(1, 0, 0),
                              dt=2e-3, n_steps=20,
                              M_vv_override=(1.0, 1.0, 1.0), ρ_ref=1.0)

Drive the C.2 3D cold sinusoid IC. Returns per-axis γ trajectories
(min/max/std) plus the final field state for inspection.
"""
function run_C2_3d_cold_sinusoid(; level::Integer = 2,
                                    A::Real = 0.3,
                                    k = (1, 0, 0),
                                    dt::Real = 2e-3,
                                    n_steps::Integer = 20,
                                    M_vv_override = (1.0, 1.0, 1.0),
                                    ρ_ref::Real = 1.0)
    ic = tier_c_cold_sinusoid_3d_full_ic(; level = level, A = A, k = k)
    # REFLECTING all three axes (cold sinusoid with integer k has
    # u(0) = u(L) = 0, naturally compatible).
    bc_spec = FrameBoundaries{3}(((REFLECTING, REFLECTING),
                                    (REFLECTING, REFLECTING),
                                    (REFLECTING, REFLECTING)))

    N_step = n_steps + 1
    t = zeros(Float64, N_step)
    γ1_max = zeros(Float64, N_step); γ1_min = zeros(Float64, N_step)
    γ2_max = zeros(Float64, N_step); γ2_min = zeros(Float64, N_step)
    γ3_max = zeros(Float64, N_step); γ3_min = zeros(Float64, N_step)
    γ1_std = zeros(Float64, N_step)
    γ2_std = zeros(Float64, N_step)
    γ3_std = zeros(Float64, N_step)

    γ_init = gamma_per_axis_3d_field(ic.fields, ic.leaves;
                                       M_vv_override = M_vv_override,
                                       ρ_ref = ρ_ref)
    γ1_max[1] = maximum(γ_init[1, :]); γ1_min[1] = minimum(γ_init[1, :])
    γ2_max[1] = maximum(γ_init[2, :]); γ2_min[1] = minimum(γ_init[2, :])
    γ3_max[1] = maximum(γ_init[3, :]); γ3_min[1] = minimum(γ_init[3, :])
    γ1_std[1] = std(γ_init[1, :])
    γ2_std[1] = std(γ_init[2, :])
    γ3_std[1] = std(γ_init[3, :])

    for n in 1:n_steps
        det_step_3d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                               bc_spec, dt;
                               M_vv_override = M_vv_override, ρ_ref = ρ_ref)
        t[n + 1] = n * dt
        γ_now = gamma_per_axis_3d_field(ic.fields, ic.leaves;
                                          M_vv_override = M_vv_override,
                                          ρ_ref = ρ_ref)
        γ1_max[n + 1] = maximum(γ_now[1, :]); γ1_min[n + 1] = minimum(γ_now[1, :])
        γ2_max[n + 1] = maximum(γ_now[2, :]); γ2_min[n + 1] = minimum(γ_now[2, :])
        γ3_max[n + 1] = maximum(γ_now[3, :]); γ3_min[n + 1] = minimum(γ_now[3, :])
        γ1_std[n + 1] = std(γ_now[1, :])
        γ2_std[n + 1] = std(γ_now[2, :])
        γ3_std[n + 1] = std(γ_now[3, :])
    end

    γ_final = gamma_per_axis_3d_field(ic.fields, ic.leaves;
                                        M_vv_override = M_vv_override,
                                        ρ_ref = ρ_ref)

    # Selectivity ratio over the run (min/max/final).
    selectivity_ratio = γ1_std ./ (γ2_std .+ γ3_std .+ eps(Float64))

    return (
        t = t,
        γ1_max = γ1_max, γ1_min = γ1_min, γ1_std = γ1_std,
        γ2_max = γ2_max, γ2_min = γ2_min, γ2_std = γ2_std,
        γ3_max = γ3_max, γ3_min = γ3_min, γ3_std = γ3_std,
        γ_final = γ_final,
        selectivity_ratio = selectivity_ratio,
        mesh = ic.mesh, frame = ic.frame, leaves = ic.leaves,
        fields = ic.fields, ρ_per_cell = ic.ρ_per_cell,
        params = ic.params,
    )
end
