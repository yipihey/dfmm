# test_M3_1_phase2_mass_HG.jl
#
# Phase M3-1 verification gate (Phase-2 sub-phase): mass conservation
# on the HG-substrate Phase-2 multi-segment driver. Mirrors M1's
# `test_phase2_mass.jl` — per-segment Lagrangian mass `Δm_j` is a fixed
# label so `total_mass_HG` is invariant by construction. The test also
# verifies bit-exact parity against M1: running the same IC through
# both `det_step!` (M1) and `det_step_HG!` (HG) should yield identical
# segment states at every step.

using Test
using dfmm

@testset "M3-1 Phase-2 (HG): mass conservation" begin
    # Smooth periodic perturbation; non-trivial dynamics.
    N = 16
    L = 1.0
    ρ0 = 1.0
    s0 = 0.0
    α0 = 1.0
    β0 = 0.0
    Δx = L / N
    Δm = fill(ρ0 * Δx, N)
    positions = collect((0:N-1) .* Δx) .+ 0.001 .* sin.(2π .* (0:N-1) ./ N) .* Δx
    velocities = 0.01 .* cos.(2π .* (0:N-1) ./ N)
    αs = fill(α0, N)
    βs = fill(β0, N)
    ss = fill(s0, N)

    mesh_HG = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                     Δm = Δm, L_box = L, bc = :periodic)

    M0 = total_mass_HG(mesh_HG)
    @test M0 ≈ sum(Δm) atol = 100 * eps()

    dt = 1e-3
    N_steps = 50
    masses_seen = Float64[]
    for n in 1:N_steps
        det_step_HG!(mesh_HG, dt)
        push!(masses_seen, total_mass_HG(mesh_HG))
    end

    # Mass is a sum of fixed labels — bit-stable across steps.
    @test all(abs.(masses_seen .- M0) .<= 100 * eps(Float64))

    # Per-cell masses must be unchanged.
    @test mesh_HG.Δm == Δm
end

@testset "M3-1 Phase-2 (HG): bit-exact parity vs M1" begin
    N = 16
    L = 1.0
    ρ0 = 1.0
    s0 = 0.0
    α0 = 1.0
    β0 = 0.0
    Δx = L / N
    Δm = fill(ρ0 * Δx, N)
    positions = collect((0:N-1) .* Δx) .+ 0.001 .* sin.(2π .* (0:N-1) ./ N) .* Δx
    velocities = 0.01 .* cos.(2π .* (0:N-1) ./ N)
    αs = fill(α0, N)
    βs = fill(β0, N)
    ss = fill(s0, N)

    mesh_M1 = dfmm.Mesh1D(positions, velocities, αs, βs, ss;
                          Δm = Δm, L_box = L, periodic = true)
    mesh_HG = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                     Δm = Δm, L_box = L, bc = :periodic)

    dt = 1e-3
    N_steps = 50
    max_err = 0.0
    for n in 1:N_steps
        dfmm.det_step!(mesh_M1, dt)
        det_step_HG!(mesh_HG, dt)
        for j in 1:N
            sM = mesh_M1.segments[j].state
            sH = read_detfield(mesh_HG.fields, j)
            max_err = max(max_err,
                          abs(sM.x  - sH.x),
                          abs(sM.u  - sH.u),
                          abs(sM.α  - sH.α),
                          abs(sM.β  - sH.β),
                          abs(sM.s  - sH.s),
                          abs(sM.Pp - sH.Pp))
        end
    end
    # Bit-exact (delegate-to-M1 path); brief target is 5e-13.
    @test max_err < 5e-13
    @test max_err == 0.0
end
