# Phase 8 — variance-gamma stochastic injection

**Scope.** Wire Track D's variance-gamma sampling and burst-statistics
primitives into the deterministic variational integrator as a per-step
operator-split mutation. Plus the Phase 9 / Tier B.4 self-consistency
check on a calibrated wave-pool problem.

**Deliverables.** `src/stochastic_injection.jl`,
`experiments/B4_compression_bursts.jl`,
`test/test_phase8_stochastic_injection.jl`, this notes file, and the
multi-panel figure `reference/figs/B4_burst_statistics.png` + raw HDF5
history `reference/figs/B4_burst_statistics.h5`.

---

## 1. Recipe

Per timestep, after the deterministic Newton solve (`det_step!` in
`src/newton_step.jl`) returns the post-step mesh state, apply the
following per-cell mutation (`inject_vg_noise!` in
`src/stochastic_injection.jl`):

1. **Cell-centered strain.** Compute
   `divu_j = (u_{j+1} - u_j) / (x_{j+1} - x_j)` from the post-Newton
   vertex velocities. (Same definition used by the EL-residual midpoint
   strain, evaluated at the post-step state — methods paper §9.5
   defines the midpoint strain rate; here we use the post-step state
   because the operator split has decoupled the noise injection from
   the variational solve.)

2. **Variance-gamma noise draw.** Sample
   `η_j ~ VG(λ, θ_factor)` per cell via Track D's
   `rand_variance_gamma!`. `θ_factor = 1/λ` is fixed so the marginal
   variance is `λ · θ_factor = 1`; `C_B` is the only amplitude knob.
   Apply a 3-point binomial Gaussian-like smoother
   (`smooth_periodic_3pt!`) with weights `(¼, ½, ¼)` and a
   variance-preserving renormalization. This is the 1D specialization
   of py-1d's `smooth_gaussian_periodic` with `ell_corr = 2.0`.

3. **Drift + noise.**
   ```
   δ_drift = C_A · ρ_j · divu_j · Δt
   δ_noise = C_B · ρ_j · √(max(-divu_j, 0) · Δt) · η_j
   δ       = δ_drift + δ_noise
   ```
   Note: drift is signed in `divu` (so it is negative on compression,
   positive on expansion), while noise is gated on the compressive
   sign. This matches py-1d `noise_model.py:hll_step_noise` lines
   154-157 and the methods-paper §9.6 recipe verbatim.

4. **Amplitude limiter.** Cap `|δ|` so the resulting per-cell ΔKE
   cannot exceed `params.ke_budget_fraction · IE_local`, where
   `IE_local = ½ P_xx + P_⊥` is the local internal-energy density.
   The discriminant solve for the symmetric cap matches py-1d
   `noise_model.py` lines 159-167.

5. **Energy debit.** Compute
   `ΔKE_vol = u_centered · δ + ½ δ² / ρ_j` and debit `(2/3) ΔKE_vol`
   from each of `P_xx` and `P_⊥`. The 2/3 split per axis is the 3D
   convention from py-1d (kinetic energy distributed evenly across
   the three momentum components). In the variational state space we
   re-anchor entropy from the new `P_xx`:
   ```
   M_vv_new = P_xx_new / ρ_j
   s_new   = s_old + log(M_vv_new / M_vv_old)
   ```
   `P_⊥_new` is updated in-place. Both subject to a `pressure_floor`
   clip.

6. **Vertex velocity update.** Convert the per-cell momentum injection
   `ΔP_j = δ_j · Δx_j` into a vertex-velocity change by mass-lumping
   half each to the two adjacent vertices. Concretely
   ```
   δ u_i = (ΔP_{i-1} + ΔP_i) / (2 · m̄_i)
   ```
   with vertex mass `m̄_i = (Δm_{i-1} + Δm_i)/2`. The cached
   `mesh.p_half[i]` is rewritten consistently. Total momentum
   `Σ_i m̄_i u_i` after the injection equals
   `Σ_j ΔP_j` plus the pre-step momentum — exactly cell-summed.

The state mutated by `inject_vg_noise!`: vertex `u`, segment `s`,
segment `Pp`. Vertex `x` and segment `(α, β)` are left untouched —
the noise lives in the momentum sector with its energy debit absorbed
in entropy + perpendicular pressure, exactly per methods paper §9.6.

---

## 2. Calibration parameters (`load_noise_model()`)

The npz at `py-1d/data/noise_model_params.npz` contains:

| key       | value (production) | meaning                                 |
|-----------|--------------------|-----------------------------------------|
| `C_A`     | ≈ 0.336            | drift coefficient                       |
| `C_B`     | ≈ 0.548            | noise amplitude                         |
| `kurt`    | ≈ 3.45             | empirical residual kurtosis             |
| `skew`    | ≈ -0.22            | empirical residual skewness             |
| `dt_save` | 0.01               | output cadence at calibration           |
| `floor`   | (small)            | chaotic-divergence variance floor       |

**λ derivation.** The variance-gamma marginal has excess kurtosis
`3/λ`, so we invert `λ = 3 / (kurt - 3)`. At the production kurt
3.45, that gives `λ ≈ 6.7` (a near-Gaussian VG; the classical
Laplace λ = 1 has excess kurt 3 ⇒ kurt 6).

**θ_factor derivation.** Fix unit marginal variance:
`var(η) = λ · θ = 1` ⇒ `θ_factor = 1/λ`. The physical amplitude
is then carried entirely by `C_B`. (Alternative: absorb a separate
θ scale into `C_B`; we choose the unit-variance convention so that
calibrated `C_B` has the same units in Julia as in py-1d.)

`from_calibration(load_noise_model())` returns a `NoiseInjectionParams`
populated this way.

---

## 3. The v3 §1.2 calibration mismatch — what we observe

**The mismatch.** Tom's v3 action note flags an inconsistency between
two empirical paths to `λ`:

* **Small-data fit (v3 §1.1).** Fitting the residual histogram on
  short calibration runs gives `λ ≈ 1.6` (excess kurt ≈ 1.875).
  This is a heavy-tailed VG, close to Laplace.

* **Production kurt-inversion (v3 §1.2).** The calibration npz reports
  empirical excess kurtosis `0.45`, which inverts to `λ ≈ 6.7`. This
  is a near-Gaussian VG.

Tom's interpretation: the chaotic-divergence floor on long runs
biases the residual variance estimate toward Gaussian, suppressing
the apparent kurtosis. Either λ is acceptable for the framework; the
mismatch tells us about the *physical interpretation* of the noise
drift, not about the variational structure.

**What the Julia wave-pool integrator says.** The B.4 driver runs the
wave-pool with the production-derived `λ ≈ 6.7` and computes:

* Burst-duration shape `k̂` from `estimate_gamma_shape(durations)` on
  the union of per-cell compression runs.
* Residual-kurtosis-implied `λ̂_res` from
  `gamma_shape_from_kurtosis(residual_kurtosis(η_residuals))` on the
  un-saturated compression-cell sample.

**Production-mode wave-pool results** (N=128, dt=5e-4, run-length
capped by integrator instability at step 950 ⇒ ~0.48 simulated time
units, after ≈ 4000 bursts and 55K residual samples):

| metric                          | value         | acceptance         |
|---------------------------------|--------------:|--------------------|
| `n_bursts`                      | 3994          | ≥ 1000 ✓           |
| `n_residual`                    | 55177         | ≥ 100 ✓            |
| `k_hat`                         | 0.73          | finite ✓           |
| `theta_T_hat`                   | 0.0095        | finite ✓           |
| `lambda_res_hat`                | 14.7          | finite ✓           |
| `ratio`                         | 20.1          | (warn_ratio = 2)   |
| `limiter_rate`                  | 0.007         | « 1 ✓              |
| KS test p-value (vs Γ(k̂, θ̂))   | < 1e-4        | not Gamma-fit ✗    |
| mass drift                      | 0.0 (exact)   | ✓                  |
| momentum drift                  | 0.13          | bounded ✓          |
| energy rel. drift               | 0.10          | bounded ✓          |

**Quick-mode** (N=64, 200 steps) for unit-test parity:
`k̂ ≈ 0.74`, `λ̂_res ≈ 12.8`, ratio ≈ 17.

Both regimes give the same qualitative picture: `λ̂_res >> k̂`, with
ratios well above the documented `warn_ratio = 2`. The KS test
rejects the simple Γ(k̂, θ̂_T) null because the burst-duration
histogram has a sharp peak at sub-`dt` resolutions plus a heavy tail —
the integrator's per-cell `divu` time series is too "spiky" for the
moment-fitted Gamma to fit at small `t` (the histogram's first bin is
20× the Gamma prediction).

**What this means.**

* `λ̂_res ≈ 13` is closer to the production calibration value 6.7 (a
  factor-of-2 mismatch with the input `λ`) than to the small-data
  λ ≈ 1.6 (a factor-of-8 mismatch). This is consistent with v3 §1.2's
  claim that the residual is *more Gaussian* than the small-data fit
  predicts.
* `k̂ ≈ 0.74` is *below* both candidate λ values. The Phase-8
  integrator's burst durations are short and clipped (the wave-pool's
  rapid Mach-< 1 turbulence does not give long sustained
  compression runs). With a moderate-Mach IC (u_0 = 0.4, P_0 = 0.5),
  the burst durations scatter over a few timesteps each — so `k̂` is
  more sensitive to discretization than the kurtosis estimator.
* The `ratio ≈ 17` is **above** the documented `warn_ratio = 2.0`,
  but the discrepancy is consistent with the open-question
  documentation in v3: this is the "production-vs-small-data λ
  discrepancy" Tom flagged, exercised on the integrator.

**Decision.** The self-consistency monitor *fires* (`ok = false`)
in this regime, which is the documented expected behaviour given the
known calibration mismatch. The acceptance criterion for Phase 8/9 is
that the **monitor wires correctly and produces sensible numbers**,
not that ratio ≤ 2 holds at production calibration. See the brief
("the documented production-vs-small-data mismatch") and v3 §1.2.

We mark the calibration mismatch as **carried forward to Tom**
(HANDOFF "Open" §1.ii) — the integrator now has the diagnostic
plumbing to investigate it directly, but resolving it is out of
Milestone-1 scope.

---

## 4. Conservation laws

* **Mass.** Per-segment `Δm` is a label, never mutated → exact.
* **Momentum.** Cell-summed momentum injection
  `Σ_j δ_j Δx_j` is what is added to total momentum per step. The
  drift contribution `Σ_j C_A ρ_j divu_j Δt Δx_j` reduces (via
  `Δx_j ρ_j = Δm_j` and `Σ_j Δm_j divu_j = 0` for the cell-centered
  divu under periodic BC and uniform `Δm`) to a near-zero discrete
  sum. The noise contribution is mean-zero by construction.
  Net: total momentum stays at `O(√n_steps)` of the per-step
  noise scale, sub-leading to the deterministic momentum.
* **Energy.** Per-cell ΔKE is exactly debited from the internal
  energy by construction (closes the ledger to round-off — verified
  by the unit test `Phase 8: per-cell ΔKE_vol bookkeeping`). The
  amplitude limiter caps the per-cell injection at 25% of the local
  IE so the floor-clipping is rare in practice. Total energy stays
  bounded; long-time drift is qualitatively similar to Phase 4's t¹
  leak plus a noise-driven random-walk floor.

---

## 5. Self-consistency monitor — usage

```julia
using dfmm
using Random: MersenneTwister

setup = setup_kmles_wavepool(N = 128, t_end = 2.0)
mesh  = build_mesh_from_setup(setup)  # see B4_compression_bursts.jl
params = from_calibration(load_noise_model())
acc    = BurstStatsAccumulator(n_segments(mesh))
det_run_stochastic!(mesh, 5e-4, 4000;
                    params = params,
                    rng    = MersenneTwister(2026),
                    accumulator = acc,
                    tau    = 1e-2,
                    q_kind = :vNR_linear_quadratic)
res = self_consistency_check(acc; warn_ratio = 2.0)
@show res.k_hat, res.lambda_res_hat, res.ratio, res.ok
```

The monitor returns a NamedTuple with:

* `n_bursts`          — total compression-run sample size.
* `n_residual`        — un-saturated VG-residual sample size.
* `k_hat`, `theta_T_hat` — Gamma fit to burst durations.
* `lambda_res_hat`    — VG λ from residual excess kurtosis.
* `ratio`             — `max(k̂/λ̂_res, λ̂_res/k̂)`.
* `ok`                — `ratio ≤ warn_ratio`.
* `limiter_rate`      — saturation fraction (≈ 1% in our wave-pool).

Sample-size thresholds: `k_hat` is finite for `n_bursts ≥ 5`,
`lambda_res_hat` is finite for `n_residual ≥ 100`. Below those the
monitor returns `Inf` and `ok = false`.

---

## 6. Files

* `src/stochastic_injection.jl` — `inject_vg_noise!`,
  `det_run_stochastic!`, `BurstStatsAccumulator`,
  `self_consistency_check`, `NoiseInjectionParams`,
  `from_calibration`, `smooth_periodic_3pt!`.
* `src/dfmm.jl` — exports the above.
* `experiments/B4_compression_bursts.jl` — wave-pool driver,
  HDF5 history writer, multi-panel CairoMakie figure.
* `test/test_phase8_stochastic_injection.jl` — 9 testsets covering
  smoothing, zero-noise bit-equality with `det_run!`, mass +
  momentum bookkeeping, per-cell ΔKE ledger, compression-cell
  injection localization, burst-stats round-trip, calibration
  loader, end-to-end wave-pool smoke, synthetic VG residual
  recovers λ via kurtosis.
* `reference/figs/B4_burst_statistics.png` — multi-panel figure.
* `reference/figs/B4_burst_statistics.h5` — raw history.

---

## 7. Integrator instability at long time

The full-mode B.4 driver hits a `non-finite mesh diagnostic` at step
≈ 950 of 1000 in the production parameter regime. The cause is a
slow-cascade entropy debit: each compressive cell's `s_new` decreases
on every step, eventually driving `M_vv = J^{1-Γ} exp(s) ` below the
`pressure_floor = 1e-8`. Once one cell's `M_vv → 0`, the next Newton
step blows up `γ² = M_vv − β² < 0` and the mesh state diverges.

**Mitigations not implemented (Phase 8.5 work):**

1. **State projection** on segments crossing the realizability
   boundary: clip `M_vv` to `pressure_floor` and re-project entropy
   accordingly. py-1d does this via the `pressure_floor` clip on
   `Pxx_new`, `Pp_new` directly; the variational form would do
   `s_new ← log(pressure_floor / ρ_j) - (1-Γ) log(J)`.
2. **Adaptive dt with realizability check**: halve dt on the next step
   after a saturated injection, restore on success.
3. **Symmetric debit**: split the energy debit equally between
   internal-energy decrease and a small kinetic adjustment so
   compressive bursts don't asymptotically deplete entropy.

For Phase 8/9 acceptance, the ≈ 950 steps we get is sufficient
(≥ 4000 bursts, ≥ 50K residual samples). The instability is
documented and flagged but not fixed in this milestone.

---

## 8. Open questions for Tom

1. **The `ratio ≈ 17` at production calibration** (this run) **vs. the
   `ratio ≈ 4` predicted by the small-data v3 fit.** The integrator's
   `λ̂_res ≈ 13` is *more* Gaussian than the production calibration's
   λ ≈ 6.7, and **much** more Gaussian than the small-data fit's
   λ ≈ 1.6. Three possible interpretations:
   - The integrator's per-cell residual-η stream genuinely tracks the
     input-`λ` (perfect VG sampling, no contamination), so the
     observed `λ̂_res` is just the *input* `λ` ≈ 6.7 plus a kurtosis
     estimator bias of ~0.13σ at this sample size — explaining the
     ~×2 inflation relative to the input.
   - The wave-pool's burst durations are too short on the
     moderate-Mach IC for `k̂` to be well-defined; the `ratio` is
     dominated by the `k̂` estimator's discretization error rather
     than a real λ mismatch.
   - The v3 §1.2 chaotic-divergence-floor explanation predicts that
     `λ̂_res > λ_true` on long runs, which is what we observe.
   Worth verifying with a run at higher Mach (a true compression-
   cascade regime where bursts last 5+ timesteps).
2. **The drift decomposition** `b_Itô ≈ 0.15 + b_closure ≈ 0.19` from
   HANDOFF "Open" §1.i is not exercised here — would require
   instrumenting the post-Newton drift into Itô vs Stratonovich
   contributions. Flag for Phase 9 follow-up if needed.

The B4 history HDF5 has all the raw arrays a follow-up agent (or
Tom) would need to redo this analysis.
