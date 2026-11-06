# Two-level AMR as an online closure laboratory (1D)

**Code:** `py-1d/dfmm/amr.py` · **Experiments:** `py-1d/experiments/19_amr_germano_online.py`
(~2 min), `py-1d/experiments/20_amr_refinement_nyquist.py` (~1 min); `QUICK=1` for smoke runs ·
**Tests:** `py-1d/tests/test_amr.py` (15 tests) ·
**Figures:** `py-1d/paper/figs/amr_germano_online.png`, `py-1d/paper/figs/amr_refinement_nyquist.png` ·
**Numbers:** `py-1d/data/amr_online_params.npz`

## Idea

In an Enzo-style hierarchy the coarse level is advanced first, the fine
level subcycles, and then projection and refluxing overwrite the coarse
solution. If the coarse level carries a subgrid model, the fine level is
best seen as a *correction* to it, and three things follow:

1. **Germano bookkeeping at projection.** Volume-averaging momentum removes
   K_res = ½(⟨ρu²⟩ − ⟨ρu⟩²/⟨ρ⟩) ≥ 0 of resolved kinetic energy. dfmm's
   conservative restriction already puts it into coarse P_xx, because
   sub-cell velocity variance *is* velocity dispersion. With a Semenov-style
   K field, set K_c ← ⟨K_f⟩ + K_res. The thermal pressure P_xx − 2K then
   restricts as a plain average.
2. **The defect is the closure error.** After each projection, U_c^n = R(U_f^n)
   on covered cells. So ε = U_c^{n+1} − R(U_f^{n+1}) is exactly experiment 11's
   "reset" closure error, with interval = one coarse step, at no extra cost.
3. **The K defect calibrates the model.** δK = K_c^pred − K_c^true at
   projection gives a dynamic (Germano-style) fit of C_ε, and optionally of
   the source fraction f_K. The same coefficients are used on both levels
   (scale similarity).

## Implementation

- `TwoLevelAMR`: static fine patch (or full coverage), Berger–Oliger
  subcycling with r fine steps per coarse step. Fine ghosts are the coarse
  state, linearly interpolated in time and prolonged by `prolong_cells`.
  Plain linear prolongation of E_xx and ρu can make P_xx negative (it
  reached −0.2 on the wave pool). `prolong_cells` therefore prolongs the
  thermal P_th with minmod and takes the slope-induced sub-cell KE out of K,
  which makes it positivity-preserving and conservative. First-order HLL or
  MUSCL-Hancock (`cfg.reconstruction`).
  Projection, then refluxing of the two uncovered neighbours. It wraps the
  unchanged `cholesky.hll_step` (BGK disabled inside, reapplied outside).
- K model (`TurbModel`). K is a *partition* of P_xx (P_xx = P_th,xx + 2K), not
  extra energy. Each level evolves it as:
  - upwind advection;
  - −2K∂ₓu compression (1D, γ=3);
  - source f_K·max(D_num, 0), where D_num is the numerical dissipation measured
    as actual minus entropy-advected internal energy (the Semenov 2022/2025
    construction);
  - exact decay K' = −C_ε ρσ_t³/Δ.

  BGK isotropizes only P_th, so K is an xx-only pressure until it decays.
  With K ≡ 0 the step equals `hll_step` to 1.6e-15.

## Results (N_c = 256, r = 8, wave pool of exps 10–14, seeds 42/77/101, t ≤ 1)

**Verification**

| check | result |
|---|---|
| composite mass / momentum / energy, half-domain patch, with refluxing | 0, 6e-17, 1e-16 |
| same, without refluxing | 2e-3, 2e-3, 3e-3 |
| full coverage: R(U_f) = U_c after projection | exact |

**Online Kramers–Moyal calibration** (estimator of experiment 11)

| protocol | C_A | C_B | excess kurtosis |
|---|---|---|---|
| online, per coarse step | 0.001 | 0.121 | 187 |
| online, shadow windows of 0.01 | 0.003 | 0.132 | 116 |
| offline experiment-11 protocol | 0.003 | 0.132 | 116 |
| online, half-domain patch, per step | −0.001 | 0.122 | 168 |

The online window estimate reproduces the offline protocol to three
digits: an AMR run *is* a calibration run.

**Caveat (pre-existing).** The unmodified `experiments/11_kmles_calibrate.py`
gives the same C_A = 0.0031, C_B = 0.132, kurtosis 116 in this environment.
That is about 100× below the stored production values in
`data/noise_model_params.npz` (C_A = 0.336, C_B = 0.548, kurtosis 3.45). The
experiment's docstring says "a factor of a few". This gap needs resolving
independently of the AMR work. The residuals here are much heavier-tailed
than Laplace.

**Semenov K with Germano bookkeeping**

| fit | C_ε | f_K |
|---|---|---|
| 1-parameter (f_K = 1: all numerical dissipation → K, as in Semenov 2025) | 36.5, resolution-dependent (18 / 24 / 32 at N_c = 64 / 128 / 256 for seed 42); clips at 50 at late times | 1 |
| 2-parameter dynamic fit | 3.7 (2.1 / 2.6 / 3.6 at N_c = 64 / 128 / 256) | 0.037 (0.03–0.04 at all three resolutions) |

- Only about 4% of the numerical dissipation in this 1D, shock-dominated flow
  behaves like unresolved bulk motion. The rest is genuine shock heating.
  Forcing f_K = 1 makes C_ε absorb the error. In 3D supersonic turbulence the
  fraction should be much larger. That is Semenov's regime, and the same
  measurement would quantify it there.
- Initial-guess independence: 1e-5 at N_c = 256, ~2% at 64, ~5% at 32 (the
  early C_ε shapes the K trajectory the fit then sees).
- **A posteriori, coarse-only runs vs filtered DNS.** Plain coarse runs
  underestimate the pressure anisotropy ⟨P_xx − P_⊥⟩: restriction deposits
  sub-cell bulk KE into P_xx, and coarse BGK isotropizes it as heat. Mean
  relative error over 0.02 < t < 0.5: coarse 0.36 → coarse+K 0.25
  (either fit).
- K tracks K_res to within ~20% until t ≈ 0.2. After that, both constant-
  coefficient runs keep too much K; after a dissipation burst near t ≈ 0.53
  they hold it about 2–3× too long. The dynamic run's C_ε rises late
  (4 → 6), so a coarse run using the time-dependent coefficients, as an AMR
  run would, should do better. Not yet tested.
- K does not reduce the resolved-KE deficit of the coarse run. That remains
  the job of the KM noise (backscatter). K and KM-LES are complementary: K
  stores forward-transferred energy, and noise could draw backscatter from K
  rather than from heat.

## Refinement events: K_c → resolved fluctuations at the new Nyquist scale

`refine_from_coarse` builds a new fine level. Modes:

- `heat`: smooth prolongation, K handed to P_th (a hierarchy without K).
- `keepK`: smooth prolongation, K_f = K_c − K_res.
- `nyquist`: random-phase k^-n velocity fluctuations on wavelengths
  2Δ_f…2Δ_c. They have zero mass-weighted mean per coarse cell (momentum
  exact), and their KE is debited from P_xx (that is, from K) so E_xx is
  exact. The total resolved sub-cell KE is set to φK_c with φ = 1 − r^{1−n}
  (0.875 for n=2, r=8). Afterwards K_f = K_c − K_res, so Germano continuity
  holds exactly.
- `phi_band_only=True`: inject φ·K_c *on top of* the slope-restored part.
- `shape='front'`: place the same energy in a tanh sub-grid shock in each
  compressive coarse cell.

Experiment 20 refines at t = 0.1 a coarse+K run (N_c = 256) by r = 8, then
compares the fine level with the same-scheme DNS. (C_ε, f_K, φ_band) for
each scheme come from a short online calibration run.

1. **Box filter vs spectral band.** Only 9% (first-order) or 20% (MUSCL-MC)
   of the box-filter K_res at N_c = 256 lies above k_c; the rest is sinc²
   leakage of resolved modes just below k_c. The share is resolution-dependent
   (first-order 26/17/9%, MUSCL 60/39/19% at N_c = 64/128/256). So
   φ = 1 − r^{1−n} overfills the band 6.5–8.3×. The online-measured φ_band
   (`band_fraction`) gives 2.3–3.1×. Most of the remainder is the coarse
   K_c itself being 1.4–1.7× too large (constant-coefficient coarse run).
   In a coarse-run probe at N_c = 128, MUSCL with φ = 0.875 hit 1.0× the DNS
   band power. The bookkeeping is right; the amplitude needs the measured
   band share.
2. **Injected band energy does not persist in 1D.** For every injection
   (random, steep spectrum, band-limited to k ≤ k_f/4, sub-grid shock
   fronts), the band power collapses within 0.0025–0.005 (2–4 coarse
   steps). From then on all modes follow the same curve: the band refills
   from the resolved cascade (first-order: 0.53 of DNS at 0.01, 0.82 at
   0.05; MUSCL: 0.65 at 0.01, 0.92 at 0.02). Estimate: the numerical
   viscosity ν ≈ ½cΔ_f damps the new band on 1/(νk_c²) ≈ 0.008 at k_c and
   ~1e-4 at the fine Nyquist. The eddy time Δ_c/σ_t ≈ 0.17. In 1D the
   sub-grid content is shocks, which thermalize KE on a crossing time, so
   shock-shaped fronts do no better. MUSCL holds the injection somewhat
   longer, as expected from its lower dissipation.
3. **What does persist is continuity of the total sub-coarse dispersion.**
   K_res + ⟨K_f⟩ (what a Semenov-type star-formation criterion sees) equals
   K_c at the refinement instant for keepK/nyquist/front, but drops to
   0.18–0.21 of the DNS value for `heat`. It recovers over ~0.005 (MUSCL) to
   ~0.02 (first-order). A virial-parameter criterion would see a spurious
   collapse of σ_t in exactly the cells that were just refined.

Expectation for 3D: solenoidal sub-grid motions at k ≳ k_c are not
thermalized on a crossing time the way 1D shocks are. Nyquist injection
should then persist for ~ min(eddy time at k_c, numerical damping time),
so it should matter more with low-dissipation schemes such as PPM.
Measuring φ_band online is dimension-agnostic and carries over directly.

## Band correction: deconvolving the scheme's numerical damping (prototype)

Motivation: numerical dissipation acts far above the grid scale. On this
problem (N_c = 256), the coarse/filtered-DNS spectrum ratio T(k) is:

| wavelength (coarse cells) | first-order T(k) | MUSCL-MC T(k) |
|---|---|---|
| 64–85 | 0.87–0.93 | 1.00 |
| 37–43 | 0.75 | 1.00 |
| 21–26 | 0.57 | 0.95 |
| 11–13 | 0.30 | 0.94 |
| 8–10 | 0.17 | 0.82 |
| 4–5 | 0.06 | 0.39 |

The per-step coarse–fine defect, however, sits mostly at the grid scale:
80–90% of its power is at wavelengths under 6 cells for first-order, and
the bands under 10 cells hold 91% of it for MUSCL. The large-scale
deficit is the accumulation of small systematic per-step errors.

`dfmm/band_correction.py` (ADM-style, after Stolz & Adams):
- `TransferMeter` / `measure_transfer`: measure the one-step damping rate
  ν(k) per band from the full-coverage AMR defect. Estimators: energy ratio
  (default) or regression; they agree to 5–20%. First-order: ν ∝ k²,
  measurable already at 40–85 cells.
- `BandCorrector`: multiplies the velocity spectrum by the truncated van
  Cittert factor Q_N(e^{−νΔt}) for wavelengths ≥ λ_min, optionally tapered
  to zero toward λ_min (the analogue of ADM's relaxation term). Momentum is
  exact. The added KE is funded from K first, then from a capped share
  `heat_fraction` of P_th.
- `calibrate_dynamic`: re-measures the residual damping with the correction
  active (via `TwoLevelAMR.coarse_hook`) and iterates ν ← ν + ν_res.
- `run_corrected`: coarse-only runs that raise `RunAborted` if the state
  becomes non-finite or the time step collapses.

Probe results (`experiments/21_band_correction_probe.py`; first-order HLL,
calibrated on seed 42, tested on held-out seeds 77/101; T for the
64–256 / 32–64 / 16–32 / 8–16-cell bands):

| variant | result |
|---|---|
| no correction | 0.90–0.95 / 0.62–0.83 / 0.69–0.74 / 0.26–0.28 |
| hard cutoff at λ ≥ 8, one-pass ν | 0.96–1.00 / 0.81–0.85 / 0.83–1.24 / **5.4–6.3** (pile-up at the cutoff) |
| taper 8→32, one-pass ν | 1.29–1.35 / 1.27–1.38 / 1.5–1.7 / 0.63–0.72 (large-scale over-correction) |
| dynamic ν, heat allowed | converges a priori (residual 12–26%), but **every** a-posteriori run blows up (t ≈ 0.02–0.09) |
| dynamic, taper, K-only funding | seed 101: **1.04 / 0.92 / 0.91 / 0.43** with ~0 heat drawn; seed 77: blows up at t = 0.13 |

Lessons:
1. ADM's two ingredients are both needed. Deconvolution without relaxation
   piles energy up at the cutoff.
2. Damping measured on the uncorrected scheme over-corrects once the band
   is restored, because part of it was transfer into the damped band.
   Self-consistent (dynamic) calibration is required, and the AMR coarse
   hook provides it.
3. A-priori convergence does not give a-posteriori stability. A global
   linear anti-diffusion with cell-wise energy caps is not robust near
   shocks: per-cell limiting makes the effective amplification
   discontinuous, which feeds grid-scale ringing.
4. Funding only from K (no heat) is the physically consistent version and
   gives the best result when it is stable. In 1D, however, the dissipation
   in the 10–60-cell band is genuinely shock heating, and K (f_K ≈ 0.05)
   holds little of it.

To make it robust: a local (compact-stencil) filter instead of the global
FFT; correction scale factors smoothed in space; route the in-band
numerical dissipation D_band to the reservoir; in 3D, act on the solenoidal
velocity only. The real target is 3D vortical flow, where under-resolved
swirls in the 10–60-cell band lose energy to numerical dissipation rather
than to shocks.

## Next steps

1. Dynamic regridding driven by the defect |ε| relative to the model's
   predicted variance, using `refine_from_coarse` at each new patch
   (refinement events are done; see above).
2. Coarse-only runs with time-dependent (C_ε, f_K) taken from an AMR run.
3. Draw the KM-noise energy debit from K instead of P_th (physical backscatter).
4. Resolve the experiment-11 vs production-coefficient discrepancy.
5. Carry the same bookkeeping into the Julia/HG code, where the tensor-q
   dissipation is explicit, so D_num needs no entropy trick.
