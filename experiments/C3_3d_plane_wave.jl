# C3_3d_plane_wave.jl
#
# M3-7e: C.3 3D plane wave driver. The 3D analog of
# `experiments/C3_2d_plane_wave.jl`.
#
# Drives the 3D Cholesky-sector Newton system (`det_step_3d_berry_HG!`)
# with the C.3 acoustic plane wave IC at integer wave-vector
# `k = (kx, ky, kz)` built via `tier_c_plane_wave_3d_full_ic`.
# Acceptance gates assert that the perturbation amplitude scales like
# Δx² across refinement levels (the variational scheme is second-order
# in space) and that velocity stays bounded under the implicit-midpoint
# Newton.

using HierarchicalGrids: HierarchicalMesh, EulerianFrame, FrameBoundaries,
    REFLECTING, PERIODIC, enumerate_leaves, refine_cells!, cell_physical_box
using dfmm: tier_c_plane_wave_3d_full_ic,
            det_step_3d_berry_HG!, primitive_recovery_3d_per_cell,
            read_detfield_3d, DetField3D

"""
    run_C3_3d_plane_wave(; level=2, A=1e-3, k=(1, 0, 0),
                          dt=1e-4, n_steps=5,
                          M_vv_override=(1.0, 1.0, 1.0), ρ_ref=1.0)

Drive the C.3 3D plane wave IC at the given wave-vector for `n_steps`
Newton steps. Returns trajectory amplitudes + per-axis velocity stats
+ a per-step amplitude metric (max(|u_par|) along the propagation
direction).
"""
function run_C3_3d_plane_wave(; level::Integer = 2,
                                 A::Real = 1e-3,
                                 k = (1, 0, 0),
                                 dt::Real = 1e-4,
                                 n_steps::Integer = 5,
                                 M_vv_override = (1.0, 1.0, 1.0),
                                 ρ_ref::Real = 1.0,
                                 abstol::Real = 1e-12,
                                 reltol::Real = 1e-12)
    ic = tier_c_plane_wave_3d_full_ic(; level = level, A = A, k = k)
    # All-PERIODIC BCs (matches the 2D `experiments/C3_2d_plane_wave.jl`
    # convention). The propagation axes wrap correctly via the 3-axis
    # periodic-coordinate-wrap tables (M3-7b). On trivial axes (kₐ = 0)
    # PERIODIC is byte-equivalent to REFLECTING since u_d ≡ 0 there.
    bc_spec = FrameBoundaries{3}(((PERIODIC, PERIODIC),
                                   (PERIODIC, PERIODIC),
                                   (PERIODIC, PERIODIC)))

    N_step = n_steps + 1
    t = zeros(Float64, N_step)
    u_inf = zeros(Float64, N_step)   # max |velocity| across all leaves
    ux_max = zeros(Float64, N_step)
    uy_max = zeros(Float64, N_step)
    uz_max = zeros(Float64, N_step)

    function record(step_idx)
        ux_arr = [Float64(ic.fields.u_1[ci][1]) for ci in ic.leaves]
        uy_arr = [Float64(ic.fields.u_2[ci][1]) for ci in ic.leaves]
        uz_arr = [Float64(ic.fields.u_3[ci][1]) for ci in ic.leaves]
        ux_max[step_idx] = maximum(abs, ux_arr)
        uy_max[step_idx] = maximum(abs, uy_arr)
        uz_max[step_idx] = maximum(abs, uz_arr)
        u_inf[step_idx] = max(ux_max[step_idx], uy_max[step_idx], uz_max[step_idx])
        return nothing
    end
    record(1)

    for n in 1:n_steps
        det_step_3d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                               bc_spec, dt;
                               M_vv_override = M_vv_override, ρ_ref = ρ_ref,
                               abstol = abstol, reltol = reltol)
        t[n + 1] = n * dt
        record(n + 1)
    end

    return (
        t = t,
        u_inf = u_inf,
        ux_max = ux_max, uy_max = uy_max, uz_max = uz_max,
        mesh = ic.mesh, frame = ic.frame, leaves = ic.leaves,
        fields = ic.fields, ρ_per_cell = ic.ρ_per_cell,
        params = ic.params,
    )
end

"""
    run_C3_3d_plane_wave_convergence(; levels=(2, 3), kwargs...) -> NamedTuple

Run `run_C3_3d_plane_wave` at multiple refinement levels and report the
mesh-spacing-scaled amplitude (the variational scheme is second-order
in space; on a smooth IC the per-step max amplitude scales like Δx²
relative to the analytical mode amplitude `A`).
"""
function run_C3_3d_plane_wave_convergence(; levels = (2, 3), kwargs...)
    results = NamedTuple[]
    for L in levels
        push!(results, run_C3_3d_plane_wave(; level = L, kwargs...))
    end
    Δx = [1.0 / (2^Int(L)) for L in levels]
    final_amplitude = [r.u_inf[end] for r in results]
    return (
        levels = Tuple(levels),
        Δx = Δx,
        final_amplitude = final_amplitude,
        results = results,
    )
end
