# test_M3_8a_E1_high_mach.jl
#
# M3-8 Phase a: E.1 high-Mach 2D shock acceptance gates
# (methods paper §10.6 E.1).
#
# Stress test: Mach 5 and Mach 10 Sod-style 2D discontinuities. The
# variational Cholesky-sector scheme is *expected to fail gracefully*
# (no NaN, no unbounded energy growth) rather than capture the shock
# structure quantitatively. This test file asserts the graceful-failure
# acceptance pattern — NOT a quantitative shock-capture pattern.
#
# References:
#   • methods paper §10.6 E.1 (Tier E.1 spec)
#   • reference/notes_M3_8a_tier_e_gpu_prep.md
#   • experiments/E1_high_mach_shock.jl

using Test
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, PERIODIC, cell_physical_box
using dfmm: tier_e_high_mach_shock_ic_full, primitive_recovery_2d_per_cell,
            det_step_2d_berry_HG!, GAMMA_LAW_DEFAULT

const M3_8A_E1_TRANSVERSE_TOL = 1.0e-10
const M3_8A_E1_KE_BOUND_FACTOR = 5.0   # KE must stay within 5× of IC across n_steps

include(joinpath(@__DIR__, "..", "experiments", "E1_high_mach_shock.jl"))

@testset "M3-8a E.1: high-Mach 2D shocks (graceful failure)" begin
    @testset "E.1 IC bridge round-trip at t=0 (M=5, level=2)" begin
        ic = tier_e_high_mach_shock_ic_full(; level = 2, mach = 5.0)
        # Rankine-Hugoniot analytical downstream (γ=1.4, M=5):
        # p_R/p_L = (2*1.4*25 - 0.4) / 2.4 = 69.6/2.4 = 29.0
        γ = Float64(GAMMA_LAW_DEFAULT)
        M = 5.0
        pR_analytic = (2*γ*M^2 - (γ-1)) / (γ+1)
        ρR_analytic = ((γ+1)*M^2) / ((γ-1)*M^2 + 2)
        @test abs(ic.params.pR / ic.params.pL - pR_analytic) ≤ 1e-12
        @test abs(ic.params.ρR / ic.params.ρL - ρR_analytic) ≤ 1e-12
        @test ic.params.ρL == 1.0
        @test ic.params.pL == 1.0

        # Per-leaf check: the IC has a step at x_split with the analytical
        # downstream state on the right.
        rec = primitive_recovery_2d_per_cell(ic.fields, ic.leaves, ic.frame,
                                                ic.ρ_per_cell)
        for i in eachindex(ic.leaves)
            ρ_expect = rec.x[i] < 0.5 ? 1.0 : ρR_analytic
            P_expect = rec.x[i] < 0.5 ? 1.0 : pR_analytic
            @test abs(rec.ρ[i] - ρ_expect) ≤ 1e-12
            @test abs(rec.P[i] - P_expect) / P_expect ≤ 1e-10
        end
    end

    @testset "E.1 IC bridge at M=10 (analytical RH downstream)" begin
        ic = tier_e_high_mach_shock_ic_full(; level = 2, mach = 10.0)
        γ = Float64(GAMMA_LAW_DEFAULT)
        M = 10.0
        pR_analytic = (2*γ*M^2 - (γ-1)) / (γ+1)
        ρR_analytic = ((γ+1)*M^2) / ((γ-1)*M^2 + 2)
        @test abs(ic.params.pR / ic.params.pL - pR_analytic) ≤ 1e-12
        @test abs(ic.params.ρR / ic.params.ρL - ρR_analytic) ≤ 1e-12
        # M=10 RH limits: pR/pL ≈ 124.75; ρR/ρL ≈ 3.88 (approaches (γ+1)/(γ-1)=6 as M→∞).
        @test ic.params.pR / ic.params.pL > 100.0
        @test ic.params.pR / ic.params.pL < 130.0
        @test ic.params.ρR / ic.params.ρL > 3.0
        @test ic.params.ρR / ic.params.ρL < 6.0
        # Sound speeds.
        @test ic.params.c_L ≈ sqrt(γ * 1.0 / 1.0) atol=1e-12
        @test ic.params.c_R ≈ sqrt(γ * pR_analytic / ρR_analytic) atol=1e-10
    end

    @testset "E.1 axis-swap symmetry (shock_axis ∈ {1, 2})" begin
        ic1 = tier_e_high_mach_shock_ic_full(; level = 2, shock_axis = 1, mach = 5.0)
        ic2 = tier_e_high_mach_shock_ic_full(; level = 2, shock_axis = 2, mach = 5.0)
        @test ic1.params.shock_axis == 1
        @test ic2.params.shock_axis == 2
        @test ic1.params.ρR ≈ ic2.params.ρR atol=1e-14
        @test ic1.params.pR ≈ ic2.params.pR atol=1e-14
        @test length(ic1.leaves) == length(ic2.leaves)
    end

    @testset "E.1 graceful failure: M=5, level=2, dt=1e-6, n=3 steps" begin
        # Use M_vv_override=(1.0, 1.0) to decouple from EOS specifics for
        # the graceful-failure regression. The shock IC is 1D-symmetric
        # so transverse (y) trivial axis stays trivial.
        result = run_E1_high_mach_shock(; level = 2, mach = 5.0, dt = 1e-6,
                                          n_steps = 3,
                                          M_vv_override = (1.0, 1.0))
        # No NaN propagation through the Newton step.
        for k in eachindex(result.nan_count)
            @test result.nan_count[k] == 0
        end
        # Total KE bounded: stays within 5× of IC value across the run.
        KE0 = result.KE[1]
        @test KE0 > 0.0
        for KE_t in result.KE
            @test KE_t ≥ 0.0
            @test KE_t ≤ M3_8A_E1_KE_BOUND_FACTOR * KE0
        end
        # Mass conservation (exact by ρ_per_cell convention; the field
        # set's α/β/u evolve but ρ_per_cell is a static driver input).
        @test result.mass[end] ≈ result.mass[1] atol=1e-12
        # 1D-symmetry: transverse independence preserved.
        for ydev in result.y_dev_max
            @test ydev ≤ M3_8A_E1_TRANSVERSE_TOL
        end
    end

    @testset "E.1 graceful failure: M=10, level=2, dt=1e-6, n=3 steps" begin
        result = run_E1_high_mach_shock(; level = 2, mach = 10.0, dt = 1e-6,
                                          n_steps = 3,
                                          M_vv_override = (1.0, 1.0))
        # No NaN propagation.
        for k in eachindex(result.nan_count)
            @test result.nan_count[k] == 0
        end
        # KE bounded.
        KE0 = result.KE[1]
        @test KE0 > 0.0
        for KE_t in result.KE
            @test KE_t ≥ 0.0
            @test KE_t ≤ M3_8A_E1_KE_BOUND_FACTOR * KE0
        end
        # Mass conservation.
        @test result.mass[end] ≈ result.mass[1] atol=1e-12
        # Transverse independence.
        for ydev in result.y_dev_max
            @test ydev ≤ M3_8A_E1_TRANSVERSE_TOL
        end
        # IC reports M=10 RH downstream.
        @test result.ic_params.mach == 10.0
        @test result.ic_params.pR / result.ic_params.pL > 100.0
    end

    @testset "E.1 RH analytical formula sanity (γ=1.4)" begin
        # Sweep Mach numbers and verify the IC's reported downstream state
        # matches the analytical RH formula at each.
        γ = Float64(GAMMA_LAW_DEFAULT)
        for M in (1.5, 2.0, 3.0, 5.0, 10.0)
            ic = tier_e_high_mach_shock_ic_full(; level = 2, mach = M)
            pR_analytic = (2*γ*M^2 - (γ-1)) / (γ+1)
            ρR_analytic = ((γ+1)*M^2) / ((γ-1)*M^2 + 2)
            @test abs(ic.params.pR / ic.params.pL - pR_analytic) ≤ 1e-12
            @test abs(ic.params.ρR / ic.params.ρL - ρR_analytic) ≤ 1e-12
            # Strong-shock limit check: as M → ∞, ρR/ρL → (γ+1)/(γ-1) = 6.
            ρ_ratio = ic.params.ρR / ic.params.ρL
            if M >= 5.0
                @test ρ_ratio < 6.0
                @test ρ_ratio > 3.0
            end
        end
    end
end
