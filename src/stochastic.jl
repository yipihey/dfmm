module Stochastic

# Variance-gamma sampling, burst-statistics primitives, and self-consistency
# diagnostics for the stochastic dressing of the dfmm action.
#
# References:
#   * specs/01_methods_paper.tex §3 ("Stochastic dressing"):
#       - eq. (\ref{eq:VG}) for the Bessel-K marginal density;
#       - the self-consistency relation
#         lambda_residual = (gamma-shape of compression burst durations)
#         (boxed equation in §3.4).
#   * design/03_action_note_v2.tex §4 (Laplace-as-variance-mixed-Gaussian
#     theorem; lines ~293–300) — variance-gamma generalises this from
#     λ = 1 (Laplace) to λ free.
#   * design/04_action_note_v3_FINAL.pdf §1 (empirical findings:
#     gamma-distributed burst durations with shape k ~ 2.5–2.8) and §3
#     (variance-gamma marginal).
#
# This module is intentionally light-weight: it depends only on
# `Random`, `Distributions`, `SpecialFunctions`, `StatsBase`, and
# `QuadGK`. No coupling to the integrator (`src/cholesky_sector.jl` etc.).
# All primitives are unit-testable on synthetic data.

using Random
import Distributions
using Distributions: Gamma
using SpecialFunctions: besselk, gamma as _gamma
using StatsBase: mean, var, std, skewness, kurtosis
using QuadGK: quadgk

export rand_variance_gamma,
       rand_variance_gamma!,
       pdf_variance_gamma,
       cdf_variance_gamma,
       burst_detect,
       estimate_gamma_shape,
       residual_kurtosis,
       gamma_shape_from_kurtosis,
       ks_test

# -----------------------------------------------------------------------------
# Variance-gamma sampling
# -----------------------------------------------------------------------------

"""
    rand_variance_gamma(rng::AbstractRNG, λ::Real, θ::Real) -> Float64

Draw one sample from the variance-gamma distribution VG(λ, θ) by the
canonical variance-mixture construction (specs/01_methods_paper.tex §3.2):

    V ~ Γ(λ, θ);   ε | V ~ 𝒩(0, V).

Mean is 0; variance is `λ θ`; excess kurtosis is `3/λ` (so the kurtosis is
`3 + 3/λ`). At λ = 1 this reduces to a Laplace law.

Units: dimensionless (in the implementation `θ` carries whatever variance
units the caller chose).

```julia
rng = MersenneTwister(42)
ε = rand_variance_gamma(rng, 1.5, 1.0)
```
"""
function rand_variance_gamma(rng::AbstractRNG, λ::Real, θ::Real)::Float64
    λ > 0 || throw(ArgumentError("λ must be > 0 (got $λ)"))
    θ > 0 || throw(ArgumentError("θ must be > 0 (got $θ)"))
    V = rand(rng, Gamma(λ, θ))
    return sqrt(V) * randn(rng)
end

"""
    rand_variance_gamma!(rng::AbstractRNG, out::AbstractVector, λ::Real, θ::Real)

In-place vectorised sampler. Writes `length(out)` independent VG(λ, θ) draws
into `out` and returns `out`.

```julia
buf = Vector{Float64}(undef, 1024)
rand_variance_gamma!(MersenneTwister(0), buf, 2.0, 0.5)
```
"""
function rand_variance_gamma!(rng::AbstractRNG, out::AbstractVector,
                              λ::Real, θ::Real)
    λ > 0 || throw(ArgumentError("λ must be > 0 (got $λ)"))
    θ > 0 || throw(ArgumentError("θ must be > 0 (got $θ)"))
    G = Gamma(λ, θ)
    @inbounds for i in eachindex(out)
        V = rand(rng, G)
        out[i] = sqrt(V) * randn(rng)
    end
    return out
end

# -----------------------------------------------------------------------------
# Variance-gamma pdf and reference cdf
# -----------------------------------------------------------------------------

"""
    pdf_variance_gamma(ε::Real, λ::Real, θ::Real) -> Float64

Marginal density of VG(λ, θ) under the variance-mixture parametrisation
of `rand_variance_gamma` (i.e.\\ `V ~ Distributions.Gamma(λ, θ)` with
shape `λ` and scale `θ` in the Distributions convention `mean(V) = λθ`,
`var(V) = λθ²`). Direct evaluation of the marginal

    f(ε) = ∫₀^∞ 𝒩(ε; 0, V) · Γ_pdf(V; λ, θ) dV
         = √(2/π) · θ^{-λ} · (|ε| √(θ/2))^{λ-1/2} · K_{λ-1/2}(|ε|·√(2/θ)) / Γ(λ)

where `K_ν` is the modified Bessel function of the second kind
(`SpecialFunctions.besselk`). This is equivalent to specs/01_methods_paper.tex
eq. (eq:VG) up to the standard rescaling between the *generalised
hyperbolic* and *Gamma-mixture* parametrisations of the variance-gamma
density (the paper writes `K_{λ-1/2}(|ε|/√θ)` under the rescaling
`θ_paper = θ/2`; we use the Distributions-canonical Gamma scale here so
that the variance is exactly `λθ` as a Distributions user would expect).

Variance check: `var(VG(λ, θ)) = λ θ` regardless of parametrisation;
mean is 0; excess kurtosis is `3/λ`.

Numerical caveats:
  * For `λ < 1/2` the density has an integrable singularity at `ε = 0`
    (`K_ν(z)` diverges as `z → 0` for `ν > 0`, with `ν = 1/2 - λ` after
    the standard reflection `K_ν = K_{-ν}`). The implementation returns
    `+Inf` at exact `ε = 0` for `λ < 1/2`.
  * For `λ = 1/2` the density is the Gaussian limit `𝒩(0, θ)`.
  * For `λ > 1/2` the density is finite at the origin; closed-form
    limit is `f(0) = Γ(λ - 1/2) / ( 2 √π · Γ(λ) · √(θ/2) ) · √(2/π) · θ^{-1/2}`,
    which we evaluate via the limit-form `(|ε|√(θ/2))^{λ-1/2} K_{λ-1/2}(z)
    → (1/2) Γ(λ-1/2) (2/c)^{λ-1/2}` as `ε → 0⁺`.
  * For `λ = 1` this reproduces the Laplace density with scale
    `b = √(θ/2)`, i.e.\\ `f(ε) = (1/(2b)) exp(-|ε|/b)`. At `ε = 0` and
    `θ = 1`, `f(0) = 1/√2`.

```julia
pdf_variance_gamma(0.5, 1.5, 1.0)
```
"""
function pdf_variance_gamma(ε::Real, λ::Real, θ::Real)::Float64
    λ > 0 || throw(ArgumentError("λ must be > 0 (got $λ)"))
    θ > 0 || throw(ArgumentError("θ must be > 0 (got $θ)"))

    c = sqrt(2.0 / θ)        # argument scale: |ε| · c is the K-arg
    ν = λ - 0.5

    if ε == 0
        if λ < 0.5
            return Inf
        elseif λ == 0.5
            # Gaussian limit 𝒩(0, θ):
            return 1.0 / sqrt(2π * θ)
        else
            # f(0) = √(2/π) / Γ(λ) · θ^{-λ} · (1/c)^{λ-1/2} · (1/2) Γ(λ-1/2) · c^{λ-1/2}·...
            # Actually using the limit  z^ν K_ν(z) → 2^{ν-1} Γ(ν)  for z → 0⁺:
            # (|ε|·c)^ν K_ν(|ε|·c) → 2^{ν-1} Γ(ν), so
            # (|ε|/c)^{ν} = ((|ε|·c)/c²)^ν = (|ε|·c)^ν · c^{-2ν}, giving
            # f(0) = √(2/π)/Γ(λ) · θ^{-λ} · c^{-2ν} · 2^{ν-1} Γ(ν)
            #     = √(2/π)/Γ(λ) · θ^{-λ} · (θ/2)^{ν} · 2^{ν-1} Γ(ν)
            #     = Γ(ν)·√(2/π) · θ^{-λ} θ^{ν} 2^{-ν+ν-1} / Γ(λ)
            #     = Γ(ν)·√(2/π) · θ^{-1/2} / (2 Γ(λ))
            return _gamma(ν) * sqrt(2.0 / π) / (2 * _gamma(λ) * sqrt(θ))
        end
    end

    z = abs(ε) * c                        # K-argument
    arg = abs(ε) * sqrt(θ / 2.0)          # base raised to ν
    log_prefactor = 0.5 * log(2.0 / π) - λ * log(θ) - log(_gamma(λ))
    log_main = ν * log(arg) + log(besselk(ν, z))
    return exp(log_prefactor + log_main)
end

"""
    cdf_variance_gamma(x::Real, λ::Real, θ::Real;
                       atol = 1e-10, rtol = 1e-8) -> Float64

CDF of VG(λ, θ) computed by adaptive quadrature of `pdf_variance_gamma`
(via `QuadGK.quadgk`). The distribution is symmetric about 0, so we
evaluate `1/2 + sign(x) * (∫₀^{|x|} f) / 2` and avoid the origin
singularity for `λ < 1/2` by integrating a tiny analytical correction
free of the singular point (we still pass `0` as the lower limit;
QuadGK handles the integrable singularity by adaptive subdivision —
verified for λ ≥ 0.5 in the test suite).

This is the reference CDF used by the K–S test in the unit tests; for
λ ∈ [0.5, 5] it converges robustly. Outside that range tighten the
tolerances or switch to a series expansion.
"""
function cdf_variance_gamma(x::Real, λ::Real, θ::Real;
                            atol::Real = 1e-10, rtol::Real = 1e-8)::Float64
    if x == 0
        return 0.5
    end
    half_int, _ = quadgk(t -> pdf_variance_gamma(t, λ, θ), 0.0, abs(x);
                         atol = atol, rtol = rtol)
    return 0.5 + sign(x) * half_int
end

# -----------------------------------------------------------------------------
# Burst detection
# -----------------------------------------------------------------------------

"""
    burst_detect(divu::AbstractVector{<:Real}; threshold::Real = 0.0)

Detect contiguous compressive runs in a 1-D divergence-of-velocity stream
`divu`. A burst is a maximal run of indices with `divu[i] < threshold`.

Returns `Vector{NamedTuple{(:start_i, :end_i, :duration, :intensity), Tuple{Int, Int, Int, Float64}}}`
with one entry per detected burst, ordered by `start_i`:

  * `start_i`: 1-based first index of the run.
  * `end_i`:   1-based last index of the run (inclusive).
  * `duration`: `end_i - start_i + 1` (in samples; multiply by `dt`
    upstream to convert to time).
  * `intensity`: `mean(-divu[start_i:end_i])` — the average compressive
    rate over the run, sign-flipped so positive ⇔ compressive.

Edge cases handled:
  * Burst at array start or end.
  * Single-sample bursts.
  * All-noncompressive `divu` → empty vector.
  * Empty `divu` → empty vector.

```julia
bursts = burst_detect([0.1, -0.5, -0.3, 0.0, -0.1]; threshold = 0.0)
# 2 bursts: [(2,3,2,0.4), (5,5,1,0.1)]
```
"""
function burst_detect(divu::AbstractVector{<:Real}; threshold::Real = 0.0)
    NT = NamedTuple{(:start_i, :end_i, :duration, :intensity),
                    Tuple{Int, Int, Int, Float64}}
    out = NT[]
    isempty(divu) && return out

    n = length(divu)
    i = firstindex(divu)
    last_idx = lastindex(divu)

    # Convert to 1-based local indices for the report (matches docstring).
    # We iterate using the actual indices but record (i - first + 1).
    while i <= last_idx
        if divu[i] < threshold
            j = i
            while j + 1 <= last_idx && divu[j + 1] < threshold
                j += 1
            end
            s_local = i - firstindex(divu) + 1
            e_local = j - firstindex(divu) + 1
            mean_neg = -mean(@view divu[i:j])  # average of -divu over run
            push!(out, (start_i = s_local,
                        end_i = e_local,
                        duration = e_local - s_local + 1,
                        intensity = Float64(mean_neg)))
            i = j + 1
        else
            i += 1
        end
    end
    return out
end

# -----------------------------------------------------------------------------
# Gamma-shape estimation from a sample of durations
# -----------------------------------------------------------------------------

"""
    estimate_gamma_shape(durations::AbstractVector{<:Real}; method = :mom) ->
        (k_hat, θ_hat)

Estimate the shape `k` and scale `θ` of a Gamma distribution from a
sample. Two estimators are supported:

  * `method = :mom` (default) — method of moments:
        k̂ = mean² / var,   θ̂ = var / mean.
    Closed form, robust to small samples, and the recommended estimator
    for the Phase 9 burst-statistics test in
    `reference/MILESTONE_1_PLAN.md`.

  * `method = :mle` — maximum-likelihood, delegated to
    `Distributions.fit_mle(Gamma, durations)`. Slightly more efficient
    but requires positive samples.

`durations` must contain only positive entries (Gamma support).

```julia
k̂, θ̂ = estimate_gamma_shape(rand(Gamma(2.5, 1.0), 1_000))
```
"""
function estimate_gamma_shape(durations::AbstractVector{<:Real};
                              method::Symbol = :mom)
    isempty(durations) && throw(ArgumentError("durations is empty"))
    any(d -> d <= 0, durations) &&
        throw(ArgumentError("durations must be strictly positive"))

    if method === :mom
        m = mean(durations)
        v = var(durations; corrected = true)
        v > 0 || throw(ArgumentError("durations have zero variance"))
        k_hat = m^2 / v
        θ_hat = v / m
        return (Float64(k_hat), Float64(θ_hat))
    elseif method === :mle
        # Lazy import: only require the fit machinery if MLE is asked for.
        # Distributions.fit_mle returns Gamma(α, θ) with shape α, scale θ.
        d = Distributions.fit_mle(Distributions.Gamma,
                                   collect(Float64, durations))
        return (Float64(Distributions.shape(d)),
                Float64(Distributions.scale(d)))
    else
        throw(ArgumentError("method must be :mom or :mle (got $method)"))
    end
end

# -----------------------------------------------------------------------------
# Residual-kurtosis ↔ shape-parameter conversion (self-consistency monitor)
# -----------------------------------------------------------------------------

"""
    residual_kurtosis(samples::AbstractVector{<:Real}) -> Float64

Return the **excess** (Pearson-Fisher) kurtosis of `samples`, i.e.
`κ - 3`, so a Gaussian sample returns 0 (up to sampling noise) and a
Laplace sample returns 3.

This wraps `StatsBase.kurtosis`, which computes the bias-uncorrected
sample excess kurtosis. For finite-sample applications (the
self-consistency monitor at output cadence) the bias is subleading
relative to the sampling error.

```julia
ε = randn(MersenneTwister(0), 100_000)
abs(residual_kurtosis(ε)) < 0.05  # Gaussian baseline
```
"""
function residual_kurtosis(samples::AbstractVector{<:Real})::Float64
    return Float64(kurtosis(samples))
end

"""
    gamma_shape_from_kurtosis(excess_kurt::Real) -> Float64

Invert the variance-gamma kurtosis relation: VG(λ, θ) has excess
kurtosis `3/λ`, so `λ = 3 / excess_kurt`.

Used by the runtime self-consistency monitor (specs/01_methods_paper.tex
§3.4): the shape inferred from the residual kurtosis of the closure
error must match `k`, the shape of the burst-duration distribution.

Returns `Inf` for `excess_kurt ≤ 0` (a Gaussian-or-platykurtic sample
implies infinite λ in the VG sense).

```julia
gamma_shape_from_kurtosis(1.5) == 2.0   # exact
```
"""
function gamma_shape_from_kurtosis(excess_kurt::Real)::Float64
    excess_kurt > 0 || return Inf
    return 3.0 / excess_kurt
end

# -----------------------------------------------------------------------------
# Two-sided one-sample Kolmogorov–Smirnov test
# -----------------------------------------------------------------------------

"""
    ks_test(samples::AbstractVector{<:Real}, cdf::Function) ->
        (statistic, p_value)

Two-sided one-sample Kolmogorov–Smirnov test against the user-supplied
reference CDF `cdf(x) -> Float64`. Returns a NamedTuple `(statistic, p_value)`.

The statistic is

    D_n = sup_x | F_n(x) - F(x) |

computed at the sample points (the standard order-statistic form).

The p-value uses the Kolmogorov asymptotic series
    P(D_n > d) ≈ 2 ∑_{j=1}^∞ (-1)^{j-1} exp(-2 j² (√n d)²),
truncated when terms drop below 1e-12. This is the same series
`HypothesisTests.ApproximateOneSampleKSTest` reports; for `n ≥ 30` the
approximation is excellent.

```julia
samples = randn(MersenneTwister(0), 1_000)
res = ks_test(samples, x -> 0.5 * (1 + erf(x / √2)))
res.p_value > 0.05  # Gaussian sample shouldn't reject Gaussian CDF
```
"""
function ks_test(samples::AbstractVector{<:Real}, cdf::Function)
    n = length(samples)
    n > 0 || throw(ArgumentError("samples is empty"))

    sorted = sort(collect(Float64, samples))
    D = 0.0
    @inbounds for i in 1:n
        Fx = Float64(cdf(sorted[i]))
        F_lo = (i - 1) / n
        F_hi = i / n
        d_lo = abs(Fx - F_lo)
        d_hi = abs(Fx - F_hi)
        D = max(D, d_lo, d_hi)
    end

    # Kolmogorov asymptotic survival function
    λ = sqrt(n) * D
    p = _kolmogorov_sf(λ)
    return (statistic = D, p_value = p)
end

"""
    _kolmogorov_sf(λ::Real) -> Float64

Asymptotic survival function P(K > λ) of the Kolmogorov distribution:
2 ∑_{j=1}^∞ (-1)^{j-1} exp(-2 j² λ²). Series is monotone decreasing in
`j` for any `λ > 0`, so we truncate when the next term drops below
relative tolerance.
"""
function _kolmogorov_sf(λ::Real)::Float64
    λ > 0 || return 1.0
    s = 0.0
    sign_j = 1.0
    prev = Inf
    @inbounds for j in 1:1_000
        term = sign_j * exp(-2 * j * j * λ * λ)
        s += term
        if abs(term) < 1e-14 && abs(term) < abs(prev) * 1e-6
            break
        end
        prev = term
        sign_j = -sign_j
    end
    p = 2.0 * s
    return clamp(p, 0.0, 1.0)
end

end # module Stochastic
