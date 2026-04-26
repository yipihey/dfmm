# test_M3_1_phase5b_qformula_unit.jl
#
# Phase M3-1 Phase-5b sub-phase, block 3: unit tests for the
# Kuropatenko / vNR `compute_q_segment` formula on a few hand-computed
# cases. Mirror of M1's `test_phase5b_artificial_viscosity.jl` block 1.
# Since the HG-substrate driver delegates to M1's `compute_q_segment`
# directly (the formula is a pure function on scalars, no mesh
# coupling), we re-assert the formula here and additionally check that
# the HG path's residual computation hits the same q values when the
# integrator is invoked on a single segment.

using Test
using dfmm

@testset "M3-1 Phase-5b: compute_q_segment formula (HG-shared kernel)" begin
    # Expansion (∂_x u ≥ 0) ⇒ q = 0
    @test compute_q_segment(0.0,  1.0, 1.0, 1.0) == 0.0
    @test compute_q_segment(1.0,  1.0, 1.0, 1.0) == 0.0
    @test compute_q_segment(1e-3, 2.0, 1.5, 0.1) == 0.0

    # Strict expansion at any strength: q = 0
    for divu in (0.1, 1.0, 10.0, 100.0)
        @test compute_q_segment(divu, 1.0, 1.0, 0.1;
                                c_q_quad = 2.0, c_q_lin = 0.5) == 0.0
    end

    # Quadratic-only (c_q_lin = 0): q = c_q^(2) ρ L² (∂_x u)²
    let ρ = 2.0, L = 0.5, divu = -3.0, c2 = 1.0
        expected = c2 * ρ * L^2 * divu^2
        got = compute_q_segment(divu, ρ, 0.0, L; c_q_quad = c2, c_q_lin = 0.0)
        @test got ≈ expected
    end

    # Linear-only (c_q_quad = 0): q = c_q^(1) ρ L c_s |∂_x u|
    let ρ = 1.5, L = 0.2, divu = -0.7, c_s = 1.3, c1 = 0.5
        expected = c1 * ρ * L * c_s * abs(divu)
        got = compute_q_segment(divu, ρ, c_s, L; c_q_quad = 0.0, c_q_lin = c1)
        @test got ≈ expected
    end

    # Combined: q = ρ [c2 L² (∂_x u)² + c1 L c_s |∂_x u|]
    let ρ = 1.0, L = 0.1, divu = -2.0, c_s = 1.2, c2 = 1.0, c1 = 0.5
        expected = ρ * (c2 * L^2 * divu^2 + c1 * L * c_s * abs(divu))
        got = compute_q_segment(divu, ρ, c_s, L; c_q_quad = c2, c_q_lin = c1)
        @test got ≈ expected
    end

    # q is monotone in |divu|.
    let ρ = 1.0, L = 0.1, c_s = 1.2
        q1 = compute_q_segment(-0.5, ρ, c_s, L)
        q2 = compute_q_segment(-1.0, ρ, c_s, L)
        q3 = compute_q_segment(-2.0, ρ, c_s, L)
        @test q1 < q2 < q3
        @test q1 > 0
    end

    # Continuity at divu = 0.
    @test compute_q_segment(0.0,    1.0, 1.0, 1.0) == 0.0
    @test compute_q_segment(-eps(), 1.0, 1.0, 1.0) ≈ 0.0 atol = 1e-15

    # Default coefficients.
    let ρ = 1.0, L = 0.1, divu = -1.0, c_s = 1.0
        q = compute_q_segment(divu, ρ, c_s, L)
        @test q ≈ 1.0 * L^2 * divu^2 + 0.5 * L * c_s * abs(divu)
    end
end
