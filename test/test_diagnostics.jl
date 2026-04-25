using Test
using dfmm
using StaticArrays

# Avoid pulling in LinearAlgebra as a test extra; for a 2x2 matrix the
# determinant is `H[1,1]*H[2,2] - H[1,2]*H[2,1]`.
det2(M) = M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]

@testset "diagnostics: hessian_HCh matches v2 §3.5 closed form" begin
    # Spread of (alpha, beta, gamma) inputs.
    for alpha in (0.5, 1.0, 1.7, 3.3),
        beta  in (-1.4, -0.3, 0.0, 0.6, 2.1),
        gamma in (0.0, 0.05, 0.4, 1.2)

        H = dfmm.hessian_HCh(alpha, beta, gamma)
        @test H isa SMatrix{2, 2, Float64}
        @test H[1, 1] ≈ -gamma^2
        @test H[1, 2] ≈ 2 * alpha * beta
        @test H[2, 1] ≈ 2 * alpha * beta
        @test H[2, 2] ≈ alpha^2
        # Symmetric.
        @test H[1, 2] == H[2, 1]
    end
end

@testset "diagnostics: det_hessian_HCh closed form and limits" begin
    # Closed form vs LinearAlgebra det of the explicit matrix.
    for alpha in (0.7, 1.5),
        beta  in (-0.8, 0.0, 0.6),
        gamma in (0.0, 0.3, 1.0)

        d_closed = dfmm.det_hessian_HCh(alpha, beta, gamma)
        d_explicit = det2(dfmm.hessian_HCh(alpha, beta, gamma))
        @test d_closed ≈ d_explicit atol = 1.0e-12
        # Sign: -alpha^2 (gamma^2 + 4 beta^2) ≤ 0 always.
        @test d_closed ≤ 0
    end

    # gamma -> 0 ⇒ det -> -4 alpha^2 beta^2.
    for alpha in (0.5, 1.0, 2.0), beta in (0.3, 1.0)
        @test dfmm.det_hessian_HCh(alpha, beta, 0.0) ≈ -4 * alpha^2 * beta^2
    end

    # beta -> 0 ⇒ det -> -alpha^2 gamma^2.
    for alpha in (0.5, 1.0, 2.0), gamma in (0.3, 1.0)
        @test dfmm.det_hessian_HCh(alpha, 0.0, gamma) ≈ -alpha^2 * gamma^2
    end

    # Caustic: gamma=beta=0 ⇒ det = 0 (rank-1 Hessian).
    @test dfmm.det_hessian_HCh(1.7, 0.0, 0.0) == 0.0

    # Continuity in gamma at fixed (alpha, beta).
    for gamma in (1.0e-1, 1.0e-3, 1.0e-6, 1.0e-9)
        d = dfmm.det_hessian_HCh(1.0, 0.5, gamma)
        @test d ≈ -1.0 * (gamma^2 + 4 * 0.25)
    end
end

@testset "diagnostics: gamma_rank_indicator" begin
    # Pure cold limit (gamma=0): indicator = 0.
    @test dfmm.gamma_rank_indicator(1.0, 0.5, 0.0) == 0.0
    # Symmetric warm state (beta=0, gamma>0): indicator = 1.
    @test dfmm.gamma_rank_indicator(1.0, 0.0, 0.7) ≈ 1.0
    # 50/50 mix: gamma/sqrt(beta^2 + gamma^2) for gamma=beta is 1/sqrt(2).
    @test dfmm.gamma_rank_indicator(1.0, 0.4, 0.4) ≈ 1 / sqrt(2)
    # Edge: both zero, defined to be 0 (avoid 0/0 NaN).
    @test dfmm.gamma_rank_indicator(1.0, 0.0, 0.0) == 0.0
    # Range bounded by [0,1].
    for beta in (-0.7, 0.3), gamma in (0.0, 0.5, 2.0)
        v = dfmm.gamma_rank_indicator(1.0, beta, gamma)
        @test 0 ≤ v ≤ 1
    end
end

@testset "diagnostics: realizability_marker" begin
    # gamma >= 0 ⇒ marker = 0.
    @test dfmm.realizability_marker(1.0, 0.5, 0.0) == 0.0
    @test dfmm.realizability_marker(1.0, 0.5, 0.3) == 0.0
    # gamma < 0 (caller bypassed clamp) ⇒ marker = gamma^2 > 0.
    @test dfmm.realizability_marker(1.0, 0.0, -0.4) ≈ 0.16

    # The Mvv-form: violated when beta^2 > Mvv.
    @test dfmm.realizability_marker_from_Mvv(0.5, 1.0) == 0.0  # Mvv >= beta^2
    @test dfmm.realizability_marker_from_Mvv(0.5, 0.25) == 0.0 # boundary
    @test dfmm.realizability_marker_from_Mvv(0.5, 0.20) ≈ 0.05 # 0.25 - 0.20
end
