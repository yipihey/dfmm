# Self-consistency monitor tests.
#
# At a target shape λ = 2 (so excess kurtosis 3/λ = 1.5), draw 10^5
# variance-gamma samples and verify:
#   * `residual_kurtosis(samples)` returns ≈ 1.5 within sampling error;
#   * `gamma_shape_from_kurtosis(1.5) == 2.0` exactly.
#
# The monitor relation is the boxed equation in
# specs/01_methods_paper.tex §3.4:
#   λ_residual = (gamma-shape of compression burst durations).

using Test
using Random
using StatsBase: kurtosis
using dfmm

@testset "gamma_shape_from_kurtosis: closed-form" begin
    @test gamma_shape_from_kurtosis(1.5) == 2.0
    @test gamma_shape_from_kurtosis(3.0) == 1.0
    @test gamma_shape_from_kurtosis(0.5) == 6.0
    @test gamma_shape_from_kurtosis(0.0) == Inf
    @test gamma_shape_from_kurtosis(-0.1) == Inf  # platykurtic ⇒ Inf
end

@testset "residual_kurtosis: Gaussian baseline" begin
    rng = MersenneTwister(20260501)
    # 10^5 Gaussian draws; SE_excess_kurtosis ≈ √(24/N) ≈ 0.0155.
    ek = residual_kurtosis(randn(rng, 100_000))
    @test abs(ek) < 0.1  # ~6 SE upper bound; very forgiving
end

@testset "residual_kurtosis ↔ gamma_shape: VG self-consistency at λ=2" begin
    rng = MersenneTwister(20260502)
    N = 100_000
    λ_true = 2.0
    θ_true = 1.0
    samples = Vector{Float64}(undef, N)
    rand_variance_gamma!(rng, samples, λ_true, θ_true)

    ek = residual_kurtosis(samples)
    expected_ek = 3.0 / λ_true   # 1.5

    # Sample-kurtosis SE for VG(λ, θ): the variance of the sample
    # excess-kurtosis estimator is dominated by the 8th moment of the
    # parent, which for VG scales as θ^4 · O(λ^{-3}). Empirically at
    # λ = 2, N = 1e5 the SE is ~0.05–0.1; we use a generous 0.3 bound
    # (≈ 5 SE) to keep the test robust across seeds.
    @test isapprox(ek, expected_ek; atol = 0.3)

    λ_hat = gamma_shape_from_kurtosis(ek)
    @info "self-consistency at λ=2.0" ek λ_hat expected_ek
    @test isapprox(λ_hat, λ_true; atol = 0.4)
end

@testset "residual_kurtosis ↔ gamma_shape: VG self-consistency at λ=1 (Laplace)" begin
    rng = MersenneTwister(20260503)
    N = 100_000
    λ_true = 1.0
    θ_true = 1.0
    samples = Vector{Float64}(undef, N)
    rand_variance_gamma!(rng, samples, λ_true, θ_true)

    ek = residual_kurtosis(samples)  # expected 3.0 (Laplace excess kurt)
    @info "Laplace excess kurtosis" ek
    @test isapprox(ek, 3.0; atol = 0.6)
    λ_hat = gamma_shape_from_kurtosis(ek)
    @test isapprox(λ_hat, 1.0; atol = 0.3)
end
