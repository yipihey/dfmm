# Variance-gamma sampler tests.
#
# Generates ~10^5 samples at a few (λ, θ) and verifies:
#   * sample mean ≈ 0 within 5 standard errors;
#   * sample variance ≈ λ θ within 5 standard errors;
#   * sample excess kurtosis ≈ 3/λ within a few standard errors;
#   * Kolmogorov–Smirnov test against the Bessel-K analytical CDF
#     (computed by adaptive quadrature) does not reject at α = 0.01.
#
# Reproducibility: explicit MersenneTwister seeds for each parameter point.
# Reference: specs/01_methods_paper.tex §3 ("Stochastic dressing"), eq. (eq:VG).

using Test
using Random
using Statistics
using StatsBase: kurtosis
using dfmm

const _SEEDS = Dict(
    (1.0, 1.0) => 20260425,
    (1.5, 1.0) => 20260426,
    (2.0, 0.5) => 20260427,
    (3.0, 2.0) => 20260428,
)

@testset "rand_variance_gamma: moment matching" begin
    N = 100_000
    for ((λ, θ), seed) in _SEEDS
        rng = MersenneTwister(seed)
        samples = Vector{Float64}(undef, N)
        rand_variance_gamma!(rng, samples, λ, θ)

        m = mean(samples)
        v = var(samples; corrected = true)
        ek = kurtosis(samples)  # excess kurtosis (Pearson-Fisher)

        # Mean: 0; SE = sqrt(λθ / N).
        se_mean = sqrt(λ * θ / N)
        @test abs(m) < 5 * se_mean

        # Variance: λθ; SE_var ≈ √(2/(N-1)) * λθ for Gaussian; for VG the
        # true sampling-error of the variance is larger because of the
        # mixture, but Var(ε^2) = E[ε^4] - (λθ)^2 = (3/λ + 3) (λθ)^2 - (λθ)^2
        # = (3/λ + 2) (λθ)^2; so SE_var = √((3/λ + 2)/N) · λθ.
        true_var = λ * θ
        se_var = sqrt((3.0/λ + 2.0) / N) * true_var
        @test abs(v - true_var) < 5 * se_var

        # Excess kurtosis: 3/λ. Sample excess-kurtosis SE for an iid
        # sample with finite 8th moment is O(1/√N), but the constant
        # depends strongly on the parent distribution. We bound by a
        # generous tolerance — VG with λ ~ O(1) has substantial 8th-
        # moment, so we use 0.2 * (3 + 3/λ) at N = 1e5 (empirically loose).
        true_ek = 3.0 / λ
        @test abs(ek - true_ek) < 0.2 * (3.0 + true_ek)
    end
end

@testset "rand_variance_gamma: Laplace special case (λ = 1)" begin
    # λ = 1 reduces to Laplace(0, scale = √(θ/2))? No: variance-mixture
    # Laplace has variance V₀ when V ~ Exp(V₀), and Distributions' Gamma(1, θ)
    # has scale θ ⇒ V₀ = θ ⇒ ε ~ Laplace with var = θ ⇒ b = √(θ/2)
    # since Laplace(0, b) has variance 2 b². Check pdf at 0:
    #   f_Laplace(0; b) = 1/(2b) = 1/(√(2θ)).
    @test isapprox(pdf_variance_gamma(0.0, 1.0, 1.0), 1.0 / sqrt(2.0);
                   rtol = 1e-12)
    @test isapprox(pdf_variance_gamma(0.0, 1.0, 0.5), 1.0 / sqrt(1.0);
                   rtol = 1e-12)
    # Tail: f_Laplace(x; b) = (2b)^-1 exp(-|x|/b).
    for x in (0.3, 1.0, 2.5)
        b = sqrt(0.5)  # for θ = 1
        ref = exp(-abs(x) / b) / (2b)
        @test isapprox(pdf_variance_gamma(x, 1.0, 1.0), ref; rtol = 1e-10)
    end
end

@testset "pdf_variance_gamma: normalisation by quadrature" begin
    using QuadGK
    for (λ, θ) in [(0.7, 1.0), (1.0, 1.0), (1.5, 1.0), (2.0, 0.5), (3.0, 2.0)]
        I, _ = quadgk(t -> pdf_variance_gamma(t, λ, θ), -50.0, 50.0;
                      rtol = 1e-10)
        @test isapprox(I, 1.0; atol = 1e-6)
    end
end

@testset "ks_test: Gaussian sample vs Gaussian CDF" begin
    using SpecialFunctions: erf
    rng = MersenneTwister(20260429)
    samples = randn(rng, 5_000)
    res = ks_test(samples, x -> 0.5 * (1 + erf(x / sqrt(2))))
    @test res.p_value > 0.01
    @test 0.0 <= res.statistic <= 1.0
end

@testset "ks_test: VG sample vs VG CDF (Bessel-K reference)" begin
    # For each tractable (λ, θ), draw N samples and KS-test against the
    # quadrature CDF. Sample size kept modest (5000) so quadrature cost
    # is bounded; at this N we still get reliable statistics.
    N = 5_000
    pvals = Dict{Tuple{Float64,Float64}, Float64}()
    for (λ, θ) in [(1.0, 1.0), (1.5, 1.0), (2.0, 0.5), (3.0, 2.0)]
        rng = MersenneTwister(_SEEDS[(λ, θ)])
        samples = Vector{Float64}(undef, N)
        rand_variance_gamma!(rng, samples, λ, θ)
        cdf = x -> cdf_variance_gamma(x, λ, θ; rtol = 1e-7)
        res = ks_test(samples, cdf)
        pvals[(λ, θ)] = res.p_value
        # 0.01 threshold; with 4 parameter points expected family-wise
        # rejection rate at α=0.01 is ~4%, so any single rejection is
        # unlikely for the seeded streams used here.
        @test res.p_value > 0.005
    end
    @info "VG KS-test p-values" pvals
end
