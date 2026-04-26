# test_M3_1_phase2_free_streaming_HG.jl
#
# Phase M3-1 verification gate (Phase-2 sub-phase): cold-limit free
# streaming on the HG-substrate driver. With γ → 0 (`s` very negative
# ⇒ M_vv ≈ 1e-12) the bulk EL reduces to ẍ = 0, giving exactly
# ballistic vertex trajectories.

using Test
using dfmm

@testset "M3-1 Phase-2 (HG): free-streaming (cold limit, pre-crossing)" begin
    N = 16
    L = 1.0
    ρ0 = 1.0
    α0 = 1.0
    β0 = 0.0
    s0 = log(1e-12)
    Δx = L / N

    A = 0.01
    Δm = fill(ρ0 * Δx, N)
    positions = collect((0:N-1) .* Δx)
    velocities = A .* sin.(2π .* (0:N-1) ./ N)
    αs = fill(α0, N)
    βs = fill(β0, N)
    ss = fill(s0, N)

    mesh_HG = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                     Δm = Δm, L_box = L, bc = :periodic)

    x0 = [read_detfield(mesh_HG.fields, j).x for j in 1:N]
    u0 = [read_detfield(mesh_HG.fields, j).u for j in 1:N]

    dt = 1e-3
    N_steps = 100
    T_end = N_steps * dt
    @test T_end < L / (2π * A) / 10

    for n in 1:N_steps
        det_step_HG!(mesh_HG, dt)
    end

    max_err = 0.0
    for i in 1:N
        x_pred = x0[i] + u0[i] * T_end
        x_now = read_detfield(mesh_HG.fields, i).x
        max_err = max(max_err, abs(x_now - x_pred))
    end
    @test max_err < 1e-10

    max_du = 0.0
    for i in 1:N
        u_now = read_detfield(mesh_HG.fields, i).u
        max_du = max(max_du, abs(u_now - u0[i]))
    end
    @test max_du < 1e-10
end
