# test_M3_1_phase2_momentum_HG.jl
#
# Phase M3-1 verification gate (Phase-2 sub-phase): momentum conservation
# on the HG-substrate driver, mirroring M1's `test_phase2_momentum.jl`.
# Periodic-BC translation invariance ⇒ Σ_i m̄_i u_i is exactly conserved.

using Test
using dfmm

@testset "M3-1 Phase-2 (HG): momentum conservation (zero-momentum perturbation)" begin
    N = 32
    L = 1.0
    ρ0 = 1.0
    s0 = 0.0
    α0 = 1.0
    β0 = 0.0
    Δx = L / N
    Δm = fill(ρ0 * Δx, N)
    positions = collect((0:N-1) .* Δx) .+ 1e-4 .* sin.(2π .* (0:N-1) ./ N) .* Δx
    velocities = 1e-3 .* cos.(2π .* (0:N-1) ./ N)
    αs = fill(α0, N)
    βs = fill(β0, N)
    ss = fill(s0, N)

    mesh_HG = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                     Δm = Δm, L_box = L, bc = :periodic)
    p0 = total_momentum_HG(mesh_HG)
    @test abs(p0) < 1e-12

    dt = 5e-3
    N_steps = 100
    p_history = Float64[]
    for n in 1:N_steps
        det_step_HG!(mesh_HG, dt)
        push!(p_history, total_momentum_HG(mesh_HG))
    end

    @test maximum(abs.(p_history)) < 1e-12
end

@testset "M3-1 Phase-2 (HG): momentum conservation (nonzero net momentum)" begin
    N = 32
    L = 1.0
    ρ0 = 1.0
    s0 = 0.0
    α0 = 1.0
    β0 = 0.0
    Δx = L / N
    Δm = fill(ρ0 * Δx, N)
    positions = collect((0:N-1) .* Δx) .+ 1e-4 .* sin.(2π .* (0:N-1) ./ N) .* Δx
    u_offset = 0.05
    velocities = u_offset .+ 1e-3 .* cos.(2π .* (0:N-1) ./ N)
    αs = fill(α0, N)
    βs = fill(β0, N)
    ss = fill(s0, N)

    mesh_HG = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                     Δm = Δm, L_box = L, bc = :periodic)
    p0 = total_momentum_HG(mesh_HG)

    dt = 5e-3
    N_steps = 100
    drift = 0.0
    for n in 1:N_steps
        det_step_HG!(mesh_HG, dt)
        drift = max(drift, abs(total_momentum_HG(mesh_HG) - p0) / abs(p0))
    end

    @test drift < 1e-10
end
