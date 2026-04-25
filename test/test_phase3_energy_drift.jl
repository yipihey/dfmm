# test_phase3_energy_drift.jl
#
# Phase 3 — Pre-crossing conservation laws on the cold-limit Zel'dovich
# run. Per the Phase-3 brief acceptance criterion #4:
#
#   * Mass bit-stable (Δm_j are labels; total mass is the trivial sum).
#   * Total energy (kinetic + Cholesky H_Ch) drift < 1e-8 relative.
#   * Total momentum conserved to round-off (translation invariance).
#
# Different from Phase 2's `test_phase2_mass.jl` / `test_phase2_momentum.jl`
# in that we run the *cold* problem all the way to 0.9 t_cross — the
# physics of interest for the methods paper's central unification claim.
# Phase 4 (B.1) covers the long-time energy drift on a smooth acoustic
# wave; this test covers the cold-limit conservation budget.

using Test
using dfmm

const _phase3_helpers = joinpath(@__DIR__, "test_phase3_zeldovich.jl")
isdefined(@__MODULE__, :setup_zeldovich) || include(_phase3_helpers)

@testset "Phase 3: conservation laws pre-crossing (cold limit)" begin
    A = 0.01
    t_cross = 1 / (2π * A)
    dt = t_cross / 1000

    @testset "N = 128, t = 0.9 t_cross: mass bit-stable, energy drift < 1e-8" begin
        N = 128
        mesh = setup_zeldovich(N, A)
        E0 = total_energy(mesh)
        M0 = total_mass(mesh)
        P0 = total_momentum(mesh)
        N_steps = Int(round(0.9 * t_cross / dt))
        E_max = E0
        E_min = E0
        for _ in 1:N_steps
            det_step!(mesh, dt)
            E = total_energy(mesh)
            E_max = max(E_max, E)
            E_min = min(E_min, E)
        end
        Ef = total_energy(mesh)
        Mf = total_mass(mesh)
        Pf = total_momentum(mesh)
        ΔE_rel = (Ef - E0) / abs(E0)
        ΔE_swing = (E_max - E_min) / abs(E0)

        @info "Phase-3 conservation" N E0 Ef ΔE_rel ΔE_swing M0 Mf P0 Pf

        # Mass: bit-stable (per-segment Δm_j are labels, not state).
        @test Mf == M0
        # Energy: drift well below the brief's 1e-8 target. Empirically
        # |ΔE/E0| ≈ 4e-10 with the implicit-midpoint scheme.
        @test abs(ΔE_rel) < 1e-8
        @test ΔE_swing < 1e-8
        # Momentum: conserved to round-off by translation invariance
        # (proved in Phase 2). Initial momentum is essentially zero
        # for the symmetric sin perturbation; tolerance 1e-15 ⋅ N.
        @test abs(Pf - P0) < 1e-12
    end

    @testset "convergence study at multiple N: energy drift remains < 1e-8" begin
        for N in [64, 128, 256]
            mesh = setup_zeldovich(N, A)
            E0 = total_energy(mesh)
            M0 = total_mass(mesh)
            N_steps = Int(round(0.9 * t_cross / dt))
            for _ in 1:N_steps
                det_step!(mesh, dt)
            end
            Ef = total_energy(mesh)
            Mf = total_mass(mesh)
            ΔE_rel = (Ef - E0) / abs(E0)
            @info "Phase-3 conservation per N" N ΔE_rel
            @test Mf == M0
            @test abs(ΔE_rel) < 1e-8
        end
    end
end
