# Burst-statistics tests on a synthetic divu stream.
#
# Synthetic-stream construction (reproducible; see seeds below):
#   1. Draw N_b = 1000 burst durations T_n ~ Γ(k = 2.5, θ_T = 1.0) and
#      intensities κ_n ~ Uniform(0.3, 1.0).
#   2. Place them sequentially with an inter-burst quiet gap of
#      g_n ~ Uniform(2, 6) samples (no overlap).
#   3. Each burst occupies ⌈T_n / dt⌉ samples (with dt = 1.0). During
#      a burst the divu signal is set to -κ_n + ξ where ξ is small
#      Gaussian sub-threshold noise to model finite-precision rounding
#      (it stays well below threshold = -1e-3).
#   4. Quiet gaps carry small positive ξ, also above threshold = -1e-3.
#
# We then verify:
#   * `burst_detect` recovers ≥ 95% of bursts with the right onset;
#   * `estimate_gamma_shape(durations)` returns k̂ within ±0.2 of 2.5.
#
# Reference: reference/MILESTONE_1_PLAN.md Phase 9 acceptance criteria;
# specs/01_methods_paper.tex §3.4 (self-consistency relation).

using Test
using Random
using Distributions
using dfmm

const _SEED = 20260430

"""
Construct one synthetic divu stream with embedded compressive bursts.
Returns (divu, true_bursts) where `true_bursts` is a Vector of
`(start_i, duration)` pairs in 1-based stream indices.
"""
function _make_synthetic_divu(rng::AbstractRNG;
                              N_bursts::Int = 1_000,
                              k::Float64 = 2.5,
                              θ_T::Float64 = 1.0,
                              dt::Float64 = 1.0,
                              quiet_min::Int = 2,
                              quiet_max::Int = 6,
                              κ_lo::Float64 = 0.3,
                              κ_hi::Float64 = 1.0,
                              noise_amp::Float64 = 1e-5)
    G = Gamma(k, θ_T)
    durations_real = rand(rng, G, N_bursts)
    intensities = κ_lo .+ (κ_hi - κ_lo) .* rand(rng, N_bursts)
    quiets = rand(rng, quiet_min:quiet_max, N_bursts + 1)

    # Convert real-valued durations to integer sample counts (≥ 1).
    durations_n = max.(1, ceil.(Int, durations_real ./ dt))

    total_n = sum(durations_n) + sum(quiets)
    divu = zeros(Float64, total_n)
    true_bursts = Vector{Tuple{Int,Int}}(undef, N_bursts)

    cursor = 1
    cursor += quiets[1]  # leading quiet gap
    for b in 1:N_bursts
        dur = durations_n[b]
        κ = intensities[b]
        for i in 0:(dur - 1)
            divu[cursor + i] = -κ + noise_amp * randn(rng)
        end
        true_bursts[b] = (cursor, dur)
        cursor += dur
        cursor += quiets[b + 1]
    end

    # Sub-threshold positive noise in quiet zones (overwrite zeros).
    for i in 1:total_n
        if divu[i] == 0.0
            divu[i] = noise_amp * (1.0 + abs(randn(rng)))  # strictly > 0
        end
    end

    return divu, true_bursts, durations_n
end

@testset "burst_detect: edge cases" begin
    # Empty input.
    @test isempty(burst_detect(Float64[]))

    # All non-compressive.
    @test isempty(burst_detect([0.1, 0.2, 0.3]))
    @test isempty(burst_detect(zeros(5)))  # threshold 0.0 ⇒ strict <

    # Single-sample burst at start.
    bs = burst_detect([-0.5, 0.1, 0.2])
    @test length(bs) == 1
    @test bs[1].start_i == 1 && bs[1].end_i == 1 && bs[1].duration == 1
    @test isapprox(bs[1].intensity, 0.5)

    # Single-sample burst at end.
    bs = burst_detect([0.1, 0.2, -0.7])
    @test length(bs) == 1
    @test bs[1].start_i == 3 && bs[1].end_i == 3
    @test isapprox(bs[1].intensity, 0.7)

    # Two adjacent bursts separated by exactly one non-compressive sample.
    bs = burst_detect([-0.1, -0.2, 0.0, -0.3, -0.4])
    @test length(bs) == 2
    @test bs[1].start_i == 1 && bs[1].duration == 2
    @test bs[2].start_i == 4 && bs[2].duration == 2
    @test isapprox(bs[1].intensity, 0.15)
    @test isapprox(bs[2].intensity, 0.35)

    # Threshold non-zero.
    bs = burst_detect([0.0, -0.05, -0.2, 0.0]; threshold = -0.1)
    @test length(bs) == 1
    @test bs[1].start_i == 3 && bs[1].duration == 1
end

@testset "burst_detect: synthetic-stream recovery" begin
    rng = MersenneTwister(_SEED)
    divu, true_bursts, true_durs = _make_synthetic_divu(rng)
    detected = burst_detect(divu; threshold = -1e-3)

    # Build a set of true onsets for matching.
    true_onsets = Set([b[1] for b in true_bursts])
    detected_onsets = Set([b.start_i for b in detected])

    matched = length(intersect(true_onsets, detected_onsets))
    recall = matched / length(true_onsets)
    @info "burst_detect recall" recall n_true=length(true_onsets) n_detected=length(detected)
    @test recall >= 0.95

    # Each detected burst should have a duration close to a true one.
    # We do a length sanity check: total "compressive samples" should
    # roughly match.
    total_true_dur = sum(true_durs)
    total_det_dur = sum(b.duration for b in detected)
    @test isapprox(total_det_dur, total_true_dur; rtol = 0.05)
end

@testset "estimate_gamma_shape: synthetic burst durations" begin
    # Use a modest sample (1000 bursts, per the brief). Convert to
    # continuous (real-valued) durations using dt to avoid the integer-
    # rounding bias inherent in sample-counted durations.
    rng = MersenneTwister(_SEED + 1)
    k_true = 2.5
    θ_true = 1.0
    durations = rand(rng, Gamma(k_true, θ_true), 1_000)

    k_hat, θ_hat = estimate_gamma_shape(durations; method = :mom)
    @info "MoM gamma fit" k_hat θ_hat
    @test abs(k_hat - k_true) < 0.2
    # Scale-parameter recovery is also bounded.
    @test isapprox(θ_hat, θ_true; atol = 0.15)

    # MLE should agree to within sampling error.
    k_mle, θ_mle = estimate_gamma_shape(durations; method = :mle)
    @info "MLE gamma fit" k_mle θ_mle
    @test abs(k_mle - k_true) < 0.2
    @test isapprox(θ_mle, θ_true; atol = 0.15)
end

@testset "estimate_gamma_shape: input validation" begin
    @test_throws ArgumentError estimate_gamma_shape(Float64[])
    @test_throws ArgumentError estimate_gamma_shape([1.0, -0.5, 2.0])
    @test_throws ArgumentError estimate_gamma_shape([1.0, 1.0, 1.0])  # zero var
    @test_throws ArgumentError estimate_gamma_shape([1.0, 2.0]; method = :foo)
end
