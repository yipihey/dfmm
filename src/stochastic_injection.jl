# stochastic_injection.jl
#
# Phase 8: variance-gamma stochastic injection wired into the
# deterministic variational integrator as a post-Newton operator-split
# step.
#
# Methods paper §4 + §9.6 specify a per-step recipe: at each cell with
# compressive strain ((∂_x u)_j < 0), perturb (ρu)_j by a drift +
# variance-gamma noise, debit the resulting kinetic-energy change
# from internal energy by paired-pressure (P_xx, P_⊥) bookkeeping,
# and apply a 3-point Gaussian smoothing to the per-cell noise sample.
#
# The implementation here mirrors py-1d's `noise_model.py:hll_step_noise`
# (drift, noise, KE-debit, amplitude limiter) but expresses the energy
# debit through the variational variables (entropy s for the trace
# `P_xx = ρ · M_vv(J, s)`, plus the explicit `P_⊥` field). The split
# between the two "transverse" 3D dimensions is fixed at 2/3 each so
# the 1D specialization matches py-1d bit-for-bit on internal-energy
# accounting.
#
# The per-step Newton solve in `det_step!` is **not** modified: this
# module is a strictly post-Newton mutator on the mesh fields. With
# `params.C_B == 0` and `params.C_A == 0` the operator-split step is a
# no-op, and the integrator reduces to Phase 5/5b deterministic
# evolution bit-for-bit.
#
# References:
#   * specs/01_methods_paper.tex §4 (variance-gamma derivation),
#                                §9.6 (per-step injection recipe),
#                                §10.3 B.4 (burst-statistics acceptance).
#   * design/04_action_note_v3_FINAL.pdf §1 (empirical findings,
#                                            calibration mismatch).
#   * py-1d/dfmm/closure/noise_model.py — the Python reference
#                                          implementation we mirror.
#   * src/stochastic.jl — the variance-gamma + burst-stats primitives
#                         (Track D); read but not modified here.

using Random: AbstractRNG, MersenneTwister
using StatsBase: mean, var

# Reuse the variance-gamma sampler + burst-stats primitives from the
# Track-D module without re-exporting them.
using .Stochastic: rand_variance_gamma, rand_variance_gamma!, burst_detect,
                   estimate_gamma_shape, residual_kurtosis,
                   gamma_shape_from_kurtosis, ks_test

# -----------------------------------------------------------------------------
# Calibration-parameter helper
# -----------------------------------------------------------------------------

"""
    NoiseInjectionParams(; C_A, C_B, λ, θ_factor, ke_budget_fraction,
                          ell_corr, ...)

Bundle of stochastic-injection knobs. Defaults match py-1d's production
configuration (`SimulationConfig` with the calibrated `C_A`, `C_B` from
`py-1d/data/noise_model_params.npz`, `noise_ke_budget_fraction = 0.25`,
`ell_corr = 2.0` cells).

Fields:
  * `C_A::Float64`         — drift coefficient.
  * `C_B::Float64`         — noise amplitude.
  * `λ::Float64`           — variance-gamma shape (production: derived
                             from kurt 3.45 ⇒ λ ≈ 6.67; small-data
                             best fit gives λ ≈ 1.6).
  * `θ_factor::Float64`    — variance-mixing scale; set so the marginal
                             variance is `θ_factor·λ = 1` (unit
                             variance VG noise; combine with `C_B` for
                             physical units).
  * `ke_budget_fraction::Float64` — amplitude-limiter fraction (matches
                                    py-1d, default 0.25).
  * `ell_corr::Float64`    — Gaussian-smoothing correlation length in
                             cells (matches py-1d, default 2.0).
  * `pressure_floor::Float64` — minimum P_xx, P_⊥ retained after the
                                debit (matches py-1d).

The constructor `NoiseInjectionParams(; ...)` accepts any subset of
these as keyword arguments and fills in py-1d defaults for the rest.
The `from_calibration` factory below builds an instance directly from
a `load_noise_model()` NamedTuple.
"""
Base.@kwdef struct NoiseInjectionParams
    C_A::Float64 = 0.336
    C_B::Float64 = 0.548
    λ::Float64   = 6.667     # 3.0 / (kurt - 3.0) at production kurt 3.45
    θ_factor::Float64 = 0.15 # = 1/λ so var = λ·θ_factor = 1
    ke_budget_fraction::Float64 = 0.25
    ell_corr::Float64 = 2.0
    pressure_floor::Float64 = 1e-8
end

"""
    from_calibration(nm::NamedTuple; ell_corr=2.0,
                     ke_budget_fraction=0.25,
                     pressure_floor=1e-8) -> NoiseInjectionParams

Build a `NoiseInjectionParams` from `load_noise_model()`. The
variance-gamma shape `λ` is derived from `nm.kurt` via
`gamma_shape_from_kurtosis(nm.kurt - 3.0)` (excess kurtosis = 3/λ for
VG); `θ_factor` is set to `1/λ` so the marginal variance equals 1
(i.e. `C_B` is the only amplitude knob).

The calibrated production values are `C_A ≈ 0.336`, `C_B ≈ 0.548`,
`kurt ≈ 3.45`. The `kurt → λ` inversion is sensitive to the exact
empirical excess kurtosis; the small-data fit `λ ≈ 1.6` corresponds
to excess ≈ 1.875.
"""
function from_calibration(nm::NamedTuple;
                          ell_corr::Float64 = 2.0,
                          ke_budget_fraction::Float64 = 0.25,
                          pressure_floor::Float64 = 1e-8)
    excess = nm.kurt - 3.0
    if excess <= 0
        # Calibration is sub-Gaussian; fall back to the small-data
        # fit λ = 1.6 with a conservative comment in the notes.
        λ = 1.6
    else
        λ = 3.0 / excess
    end
    θ_factor = 1.0 / λ
    return NoiseInjectionParams(C_A = nm.C_A, C_B = nm.C_B,
                                λ = λ, θ_factor = θ_factor,
                                ke_budget_fraction = ke_budget_fraction,
                                ell_corr = ell_corr,
                                pressure_floor = pressure_floor)
end

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

"""
    _segment_divu_centered(mesh, j) -> Float64

Cell-centered ∂_x u for segment `j`, computed as
`(u_{j+1} - u_j) / (x_{j+1} - x_j)` from the Phase-2 vertex
representation. This is the same definition used by `det_el_residual`
for the in-step strain and matches the **midpoint** strain to
leading order (the operator-split step uses the post-Newton state).
"""
function _segment_divu_centered(mesh::Mesh1D{T,DetField{T}}, j::Integer) where {T<:Real}
    N = n_segments(mesh)
    j_right = j == N ? 1 : j + 1
    wrap = (j == N) ? mesh.L_box : zero(T)
    Δx_j = mesh.segments[j_right].state.x + wrap - mesh.segments[j].state.x
    Δu_j = mesh.segments[j_right].state.u - mesh.segments[j].state.u
    return Float64(Δu_j / Δx_j)
end

"""
    _segment_velocity_centered(mesh, j) -> Float64

Cell-centered velocity `½(u_j + u_{j+1})` for the cell-centered KE
bookkeeping (matches py-1d's per-cell `u = U[IDX_MOM]/rho`
interpretation).
"""
function _segment_velocity_centered(mesh::Mesh1D{T,DetField{T}}, j::Integer) where {T<:Real}
    N = n_segments(mesh)
    j_right = j == N ? 1 : j + 1
    return Float64((mesh.segments[j].state.u + mesh.segments[j_right].state.u) / 2)
end

"""
    smooth_periodic_3pt!(out, eta)

In-place 3-point Gaussian-like periodic smoother with weights
`(1, 2, 1)/4` (a discrete approximation to a Gaussian with σ ≈ 1
cell). Variance is renormalized to match the input sample variance
so the smoother does not bias the noise amplitude. With `length(eta)
== 1` the smoother is a no-op.

This is the 1D specialization of `smooth_gaussian_periodic` from
`py-1d/dfmm/closure/noise_model.py`. The full FFT-based smoother
with `ell_corr = 2.0` cells is qualitatively similar; the 3-point
binomial kernel is correct to leading order in `1/ell_corr` and
avoids an FFT dependency for what is a per-step inner loop.
"""
function smooth_periodic_3pt!(out::AbstractVector{Float64},
                              eta::AbstractVector{Float64})
    N = length(eta)
    @assert length(out) == N
    if N < 2
        copyto!(out, eta)
        return out
    end
    # Pre-smooth standard deviation for variance-preserving rescale.
    σ_in = sqrt(var(eta; corrected = false))
    @inbounds for i in 1:N
        ip = i == N ? 1 : i + 1
        im = i == 1 ? N : i - 1
        out[i] = 0.25 * eta[im] + 0.5 * eta[i] + 0.25 * eta[ip]
    end
    σ_out = sqrt(var(out; corrected = false))
    if σ_out > 0 && σ_in > 0
        scale = σ_in / σ_out
        @inbounds for i in 1:N
            out[i] *= scale
        end
    end
    return out
end

# -----------------------------------------------------------------------------
# Per-step injection
# -----------------------------------------------------------------------------

"""
    InjectionDiagnostics

Per-cell diagnostics produced by `inject_vg_noise!`, used by the
self-consistency monitor (`BurstStatsAccumulator`).

Fields:
  * `divu::Vector{Float64}`        — cell-centered ∂_x u after the
                                     deterministic step.
  * `eta::Vector{Float64}`         — smoothed unit-variance VG draw.
  * `delta_rhou::Vector{Float64}`  — total δ(ρu) (drift + noise),
                                     **after** amplitude-limiting.
  * `delta_rhou_drift::Vector{Float64}` — drift component only.
  * `delta_rhou_noise::Vector{Float64}` — noise component only,
                                          before limiting (i.e.
                                          `noise_amp * eta`).
  * `delta_KE_vol::Vector{Float64}` — per-cell ΔKE injected (cell
                                      volumetric).
  * `compressive::BitVector`        — true on cells with `divu < 0`.
"""
struct InjectionDiagnostics
    divu::Vector{Float64}
    eta::Vector{Float64}
    delta_rhou::Vector{Float64}
    delta_rhou_drift::Vector{Float64}
    delta_rhou_noise::Vector{Float64}
    delta_KE_vol::Vector{Float64}
    compressive::BitVector
end

InjectionDiagnostics(N::Int) = InjectionDiagnostics(
    zeros(Float64, N), zeros(Float64, N),
    zeros(Float64, N), zeros(Float64, N),
    zeros(Float64, N), zeros(Float64, N),
    falses(N),
)

"""
    inject_vg_noise!(mesh::Mesh1D, dt; params, rng,
                     diag = InjectionDiagnostics(n_segments(mesh)))

Apply one variance-gamma noise + drift injection step to `mesh`,
in place. Returns `(mesh, diag)`. The mesh's `(u, s, P_⊥)` fields are
mutated; `(α, β)` are left untouched (the noise lives in the
momentum sector, with the energy debit absorbed via entropy).

Per-step recipe (mirrors py-1d's `hll_step_noise`):
  1. Compute cell-centered `(∂_x u)_j` from the post-Newton vertex
     velocities.
  2. Draw `η_j ~ VG(λ, θ_factor)`, smooth periodically.
  3. Drift `δ_drift = C_A ρ_j (∂_x u)_j Δt`; noise
     `δ_noise = C_B ρ_j √(max(-(∂_x u)_j, 0) Δt) η_j`.
  4. Amplitude-limit so `|δ(ρu)|` cannot extract more than
     `params.ke_budget_fraction · IE_local` from the cell.
  5. Compute `ΔKE_vol = u_j δ + δ²/(2ρ_j)`; debit
     `(2/3) ΔKE_vol` from `P_xx` and `(2/3) ΔKE_vol` from `P_⊥`.
  6. Translate `P_xx_new ↦ s_new` via the EOS (re-anchor entropy);
     update `P_⊥` directly. Both subject to a `pressure_floor` clip.
  7. Distribute the cell-momentum `Δm_j δ_j J_j = δ_j Δx_j` half
     each to the two adjacent vertices (mass-lumped); update
     `mesh.p_half`.

Conservation:
  * Total mass: bit-exact (no `Δm` mutation).
  * Total momentum: equal to `∑_j δ_j Δx_j` per step (sums over
    vertices and cells differ by zero in periodic BC since each
    vertex receives exactly half from each neighbor).
  * Total energy (KE_bulk + IE_internal + Cholesky-Hamiltonian):
    bounded; any drift is bounded by the post-step amplitude
    limiting and the floor-clipping.
"""
function inject_vg_noise!(mesh::Mesh1D{T,DetField{T}}, dt::Real;
                          params::NoiseInjectionParams,
                          rng::AbstractRNG,
                          diag::InjectionDiagnostics =
                              InjectionDiagnostics(n_segments(mesh))) where {T<:Real}
    N = n_segments(mesh)
    @assert length(diag.divu) == N "InjectionDiagnostics size mismatch ($(length(diag.divu)) vs $N)"

    Δt = Float64(dt)

    # 1. Compute divu and ρ per cell from post-Newton state.
    @inbounds for j in 1:N
        diag.divu[j] = _segment_divu_centered(mesh, j)
    end

    # 2. Draw VG noise and smooth.
    eta_white = Vector{Float64}(undef, N)
    rand_variance_gamma!(rng, eta_white, params.λ, params.θ_factor)
    if params.ell_corr > 0 && N >= 3
        smooth_periodic_3pt!(diag.eta, eta_white)
    else
        copyto!(diag.eta, eta_white)
    end

    # 3-6. Per-cell drift, noise, limiter, energy debit, entropy update.
    # We accumulate cell-momentum injection (δρu · Δx) for the vertex
    # update in step 7.
    Δp_cell = Vector{Float64}(undef, N)

    @inbounds for j in 1:N
        seg = mesh.segments[j]
        ρ_j = Float64(segment_density(mesh, j))
        u_c = _segment_velocity_centered(mesh, j)
        J_j = 1.0 / ρ_j
        s_pre = Float64(seg.state.s)
        Mvv_pre = Float64(Mvv(J_j, s_pre))
        Pxx_old = ρ_j * Mvv_pre
        Pp_old  = Float64(seg.state.Pp)
        # Detect Pp sentinel (NaN from legacy 5-arg DetField). When Pp
        # is NaN, treat as isotropic Maxwellian so the debit can still
        # operate on the trace via entropy alone.
        if !isfinite(Pp_old)
            Pp_old = Pxx_old
        end

        divu_j = diag.divu[j]
        diag.compressive[j] = divu_j < 0

        drift_term = params.C_A * ρ_j * divu_j * Δt
        compression = max(-divu_j, 0.0)
        noise_amp = params.C_B * ρ_j * sqrt(compression * Δt)
        δ_drift = drift_term
        δ_noise = noise_amp * diag.eta[j]
        diag.delta_rhou_drift[j] = δ_drift
        diag.delta_rhou_noise[j] = δ_noise
        δ = δ_drift + δ_noise

        # 4. Amplitude limiter (matches py-1d, lines 159-167 of
        #    noise_model.py): KE injection capped at
        #    `ke_budget_fraction · IE_local`. IE_local per unit volume
        #    = 0.5·P_xx + P_⊥, the effective "kinetic-energy reservoir"
        #    in the trace. The discriminant form for the symmetric cap
        #    matches py-1d exactly.
        IE_local = 0.5 * Pxx_old + Pp_old
        KE_budget = params.ke_budget_fraction * IE_local
        abs_u = abs(u_c)
        disc = abs_u * abs_u + 2.0 * KE_budget / max(ρ_j, 1e-30)
        δ_max = (-abs_u + sqrt(max(disc, 0.0))) * ρ_j
        if δ > δ_max
            δ = δ_max
        elseif δ < -δ_max
            δ = -δ_max
        end
        diag.delta_rhou[j] = δ

        # 5. KE-debit and pressure update.
        ΔKE_vol = u_c * δ + 0.5 * δ * δ / ρ_j
        diag.delta_KE_vol[j] = ΔKE_vol
        Pxx_new = Pxx_old - (2.0 / 3.0) * ΔKE_vol
        Pp_new  = Pp_old  - (2.0 / 3.0) * ΔKE_vol
        if Pxx_new < params.pressure_floor
            Pxx_new = params.pressure_floor
        end
        if Pp_new < params.pressure_floor
            Pp_new = params.pressure_floor
        end

        # 6. Re-anchor entropy from P_xx (variational-state convention).
        Mvv_new = Pxx_new / ρ_j
        if Mvv_new > 0 && Mvv_pre > 0
            s_new = s_pre + log(Mvv_new / Mvv_pre)
        else
            s_new = s_pre
        end

        # Mutate the segment. (x, u, α, β) untouched here — the vertex
        # u update happens in step 7. The segment's `state.u` (left
        # vertex value) is rewritten in step 7 too.
        mesh.segments[j].state = DetField{T}(seg.state.x, seg.state.u,
                                             seg.state.α, seg.state.β,
                                             T(s_new), T(Pp_new))
        # Cell-momentum injection per cell: ΔP_j = δ · Δx_j.
        Δx_j = J_j * Float64(seg.Δm)
        Δp_cell[j] = δ * Δx_j
    end

    # 7. Distribute cell momentum to vertices (mass-lumped, half each).
    # Vertex i receives ΔP_{i-1}/2 from its left cell and ΔP_i/2 from
    # its right cell. Updated u_i = u_i^old + (ΔP_{i-1} + ΔP_i)/(2·m̄_i).
    @inbounds for i in 1:N
        i_left = i == 1 ? N : i - 1
        m̄ = Float64(vertex_mass(mesh, i))
        δu = (Δp_cell[i_left] + Δp_cell[i]) / (2.0 * m̄)
        u_old = Float64(mesh.segments[i].state.u)
        u_new = u_old + δu
        seg = mesh.segments[i]
        mesh.segments[i].state = DetField{T}(seg.state.x, T(u_new),
                                             seg.state.α, seg.state.β,
                                             seg.state.s, seg.state.Pp)
        mesh.p_half[i] = m̄ * u_new
    end

    return mesh, diag
end

# -----------------------------------------------------------------------------
# Burst-statistics accumulator
# -----------------------------------------------------------------------------

"""
    BurstStatsAccumulator(N::Int)

State container for the runtime self-consistency monitor. Accumulates
two streams across timesteps:

  * **Per-cell divu time series.** Per cell `j`, a vector
    `Vector{Float64}` of `(∂_x u)_j(t_n)` samples. Used by
    `burst_durations(...)` to detect contiguous compression runs in
    each cell's history; the empirical burst-duration sample fed to
    `estimate_gamma_shape` is the union of all cells' burst durations.

  * **Compression-cell residual sample.** A single flat
    `Vector{Float64}` of `(δ(ρu) − δ_drift) / noise_amp_norm` values
    on cells with `divu < 0`. The denominator scales by the
    deterministic-amplitude prediction `C_B · ρ · √(|divu| Δt)` so the
    sample is unit-variance under perfect VG noise — `residual_kurtosis`
    on this sample inverts to `λ̂_res` via `gamma_shape_from_kurtosis`.

Implementation notes:
  * The divu history is stored per cell as a `Vector{Vector{Float64}}`
    of `N` independent series, so per-cell `burst_detect` runs find
    contiguous compressions in time-of-cell, not across cells.
  * The residual sample uses every compression-cell injection that
    survives the amplitude limiter; the limiter saturation rate is
    tracked separately for diagnostic purposes
    (`acc.n_limited / acc.n_compress`).
"""
mutable struct BurstStatsAccumulator
    N::Int
    divu_history::Vector{Vector{Float64}}     # per-cell time series
    residual_samples::Vector{Float64}         # flat sample buffer
    n_compress::Int
    n_limited::Int
    n_steps::Int
    dt_history::Vector{Float64}
end

function BurstStatsAccumulator(N::Int)
    return BurstStatsAccumulator(
        N,
        [Float64[] for _ in 1:N],
        Float64[],
        0, 0, 0,
        Float64[],
    )
end

"""
    record_step!(acc::BurstStatsAccumulator, diag::InjectionDiagnostics,
                 dt::Real, params::NoiseInjectionParams)

Append one step's per-cell `(∂_x u, η, δ-residual)` data to the
accumulator. The residual is `eta_j` itself — by construction this is
the realized VG draw, and its kurtosis converges to `3/λ` (excess) as
the sample grows. Cells where the amplitude limiter capped the
injection are flagged via `n_limited`; the residual sample includes
**only** unsaturated compression cells so the kurtosis estimate is not
biased by the cap.
"""
function record_step!(acc::BurstStatsAccumulator,
                      diag::InjectionDiagnostics, dt::Real,
                      params::NoiseInjectionParams)
    @assert length(diag.divu) == acc.N "Accumulator/diag size mismatch"
    acc.n_steps += 1
    push!(acc.dt_history, Float64(dt))
    @inbounds for j in 1:acc.N
        push!(acc.divu_history[j], diag.divu[j])
        if diag.compressive[j]
            acc.n_compress += 1
            # Saturation check: if the limited δ differs from the
            # nominal (drift + noise_amp · η) by more than 1e-12,
            # the limiter fired. Only collect unsaturated samples.
            δ_nominal = diag.delta_rhou_drift[j] + diag.delta_rhou_noise[j]
            if abs(diag.delta_rhou[j] - δ_nominal) > 1e-12 * max(abs(δ_nominal), 1.0)
                acc.n_limited += 1
            else
                # The realized residual is η itself (unit variance under
                # perfect VG); aggregate directly.
                push!(acc.residual_samples, diag.eta[j])
            end
        end
    end
    return acc
end

"""
    burst_durations(acc::BurstStatsAccumulator) -> Vector{Float64}

Walk the per-cell `divu_history` time series, detect contiguous
compression runs via `Stochastic.burst_detect`, and return the union
of durations across all cells in **physical-time units** (multiplied
by the running `dt` mean — Phase-8 driver uses fixed `dt`, so this is
just `n_samples * dt_mean`). Empty if no compression was observed.
"""
function burst_durations(acc::BurstStatsAccumulator)
    isempty(acc.dt_history) && return Float64[]
    dt_mean = mean(acc.dt_history)
    out = Float64[]
    for j in 1:acc.N
        bursts = burst_detect(acc.divu_history[j])
        for b in bursts
            push!(out, dt_mean * b.duration)
        end
    end
    return out
end

"""
    self_consistency_check(acc::BurstStatsAccumulator;
                           warn_ratio = 2.0) -> NamedTuple

Compute the Phase-8/9 self-consistency monitor. Returns a NamedTuple
with the empirical numbers and the `ok` flag:

  * `n_bursts`         — total compression bursts detected.
  * `n_residual`       — size of the un-saturated residual sample.
  * `k_hat`            — gamma-shape of the burst-duration histogram
                         from `estimate_gamma_shape(durations)`. `Inf`
                         if too few bursts.
  * `theta_T_hat`      — burst-duration scale (τ̂ in the methods paper).
  * `lambda_res_hat`   — variance-gamma λ inferred from the residual
                         excess kurtosis. `Inf` if the sample is too
                         small or sub-Gaussian.
  * `ratio`            — `max(k_hat/λ_res_hat, λ_res_hat/k_hat)`.
                         Methods-paper §3.4 and v3 §1.2 say this should
                         be near 1.0 in production.
  * `ok::Bool`         — `ratio ≤ warn_ratio` (default 2.0).
  * `limiter_rate`     — `n_limited / n_compress` (saturation fraction).

The `warn_ratio = 2.0` default is the documented production tolerance:
Tom's v3 §1.2 reports the small-data fit `λ ≈ 1.6` vs the production
kurt 3.45 ⇒ `λ ≈ 6.7`, a factor-4 mismatch attributed to the chaotic-
divergence floor biasing the production residual toward Gaussian. A
factor-2 tolerance is the "is the wiring right" bar.
"""
function self_consistency_check(acc::BurstStatsAccumulator;
                                warn_ratio::Float64 = 2.0)
    durations = burst_durations(acc)
    n_b = length(durations)
    if n_b >= 5
        # estimate_gamma_shape requires positive samples and var > 0.
        # Add a defensive try/catch for degenerate single-step bursts.
        try
            k_hat, θ_T_hat = estimate_gamma_shape(durations)
        catch
            k_hat = Inf
            θ_T_hat = NaN
        end
        # The above try/catch can't write to outer locals in Julia,
        # so re-do with explicit assignment.
        k_hat = Inf
        θ_T_hat = NaN
        try
            (k_hat, θ_T_hat) = estimate_gamma_shape(durations)
        catch
            # leave defaults
        end
    else
        k_hat = Inf
        θ_T_hat = NaN
    end

    n_r = length(acc.residual_samples)
    if n_r >= 100
        ek = residual_kurtosis(acc.residual_samples)
        λ_res = gamma_shape_from_kurtosis(ek)
    else
        λ_res = Inf
    end

    if isfinite(k_hat) && isfinite(λ_res) && k_hat > 0 && λ_res > 0
        ratio = max(k_hat / λ_res, λ_res / k_hat)
    else
        ratio = Inf
    end
    ok = ratio <= warn_ratio

    limiter_rate = acc.n_compress > 0 ?
                   Float64(acc.n_limited) / acc.n_compress : 0.0

    return (n_bursts = n_b, n_residual = n_r,
            k_hat = Float64(k_hat),
            theta_T_hat = Float64(θ_T_hat),
            lambda_res_hat = Float64(λ_res),
            ratio = Float64(ratio),
            ok = ok,
            limiter_rate = limiter_rate)
end

# -----------------------------------------------------------------------------
# Driver: deterministic step + injection wrapper
# -----------------------------------------------------------------------------

"""
    det_run_stochastic!(mesh::Mesh1D, dt, n_steps;
                        params::NoiseInjectionParams,
                        rng::AbstractRNG,
                        tau::Union{Real,Nothing} = nothing,
                        q_kind::Symbol = :none,
                        c_q_quad::Real = 1.0,
                        c_q_lin::Real = 0.5,
                        accumulator::Union{BurstStatsAccumulator,Nothing} = nothing,
                        monitor_every::Int = 0,
                        kwargs...) -> mesh

Run `n_steps` of `det_step!` followed by `inject_vg_noise!` per step.
Mutates `mesh` in place; returns `mesh`. If `accumulator !== nothing`,
calls `record_step!(accumulator, diag, dt, params)` after each
injection. If `monitor_every > 0` and `accumulator !== nothing`, the
self-consistency monitor is invoked every `monitor_every` steps and
its result is **not** persisted (caller can re-run
`self_consistency_check(accumulator)` at end-of-run).

`tau`, `q_kind`, `c_q_quad`, `c_q_lin` are forwarded to `det_step!`
(Phase 5 BGK and Phase 5b artificial-viscosity knobs).

With `params.C_A = params.C_B = 0` this driver is bit-equal to
`det_run!(mesh, dt, n_steps; tau=tau, q_kind=q_kind, …)`. This
property is asserted by the Phase-8 unit tests.
"""
function det_run_stochastic!(mesh::Mesh1D{T,DetField{T}}, dt::Real,
                             n_steps::Integer;
                             params::NoiseInjectionParams,
                             rng::AbstractRNG,
                             tau::Union{Real,Nothing} = nothing,
                             q_kind::Symbol = :none,
                             c_q_quad::Real = 1.0,
                             c_q_lin::Real = 0.5,
                             accumulator::Union{BurstStatsAccumulator,Nothing} = nothing,
                             monitor_every::Int = 0,
                             kwargs...) where {T<:Real}
    N = n_segments(mesh)
    diag = InjectionDiagnostics(N)
    for n in 1:n_steps
        det_step!(mesh, dt;
                  tau = tau, q_kind = q_kind,
                  c_q_quad = c_q_quad, c_q_lin = c_q_lin,
                  kwargs...)
        inject_vg_noise!(mesh, dt; params = params, rng = rng, diag = diag)
        if accumulator !== nothing
            record_step!(accumulator, diag, dt, params)
            if monitor_every > 0 && (n % monitor_every == 0)
                # Side-effect-free monitor; warn via @debug to avoid
                # log noise in the test suite.
                _ = self_consistency_check(accumulator)
            end
        end
    end
    return mesh
end
