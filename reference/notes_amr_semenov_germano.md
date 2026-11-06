# Two-level AMR as an online closure laboratory (1D)

**Code:** `py-1d/dfmm/amr.py` · **Experiment:** `py-1d/experiments/19_amr_germano_online.py`
(~2 min; `QUICK=1` for ~30 s) · **Tests:** `py-1d/tests/test_amr.py` (7 tests) ·
**Figure:** `py-1d/paper/figs/amr_germano_online.png` · **Numbers:** `py-1d/data/amr_online_params.npz`

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
  state, linearly interpolated in time and prolonged with minmod slopes.
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

## Next steps

1. Dynamic regridding. Then the refinement event matters: prolongation can
   either keep K_f = K_c or seed resolved fluctuations from K_c (correlated
   over ~2 fine cells, as dfmm's ℓ_corr result suggests). Also use the
   defect |ε| relative to the model's predicted variance as the refinement
   criterion.
2. Coarse-only runs with time-dependent (C_ε, f_K) taken from an AMR run.
3. Draw the KM-noise energy debit from K instead of P_th (physical backscatter).
4. Resolve the experiment-11 vs production-coefficient discrepancy.
5. Carry the same bookkeeping into the Julia/HG code, where the tensor-q
   dissipation is explicit, so D_num needs no entropy trick.
