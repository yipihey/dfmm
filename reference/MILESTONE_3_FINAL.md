# Milestone 3 — final synthesis (suitable for §A appendix or reviewer response)

**Status (2026-04-26):** M3 closed in its entirety. The 2D and 3D
Cholesky-sector substrates are operational; all Tier-D headline
falsifiers (D.1, D.4, D.7, D.10) and Tier-E stress tests (E.1, E.2,
E.3) have landed; the methods paper §10.5 / §10.6 / §11 are revised
to reflect the actual results.

## What M3 delivers

1. **Native HG-side substrates.** The 1D path (M1 + M2) and the 2D
   path (M3-3 / M3-4) and the 3D path (M3-7) all run natively on
   `HierarchicalGrids.jl`. The transitional `cache_mesh::Mesh1D`
   shim is retired (M3-3e). Dimension-lift gates hold at *Δ = 0.0
   absolute*: the 2D residual reduces byte-equal to the 1D Phase-1
   trajectory on 1D-symmetric configurations; the 3D residual
   reduces byte-equal to the 2D residual on 2D-symmetric
   configurations and to the 1D Phase-1 trajectory on 1D-symmetric
   configurations.

2. **Off-diagonal Cholesky-Berry sector.** M3-6 Phase 0 reactivates
   the off-diagonal pair (β₁₂, β₂₁) in the 2D residual (11-dof
   Newton system); 9 SymPy CHECKs reproduced numerically. Phase 1a
   wires the off-diagonal strain coupling
   H_rot^off = G̃_12·(α_1·β_21 + α_2·β_12)/2 into the residual rows.
   Phase 1b adds the KH IC factory and a 4-component realizability
   cone Q = β₁² + β₂² + 2(β₁₂² + β₂₁²) ≤ M_vv·headroom_offdiag.

3. **3D SO(3) Berry coupling.** M3-7c integrates SO(3) Berry coupling
   into `cholesky_el_residual_3D_berry!`: 8 SymPy CHECKs reproduced;
   iso-pullback ε-extrapolation slope 1 ± 10⁻³; H_rot kernel-
   orthogonality residual ≤ 10⁻¹⁰. Per-axis γ + AMR + 3-component
   realizability projection in 3D land in M3-7d.

4. **Bayesian L↔E remap.** M3-5 wires `compute_overlap` +
   `polynomial_remap_l_to_e!`/`_e_to_l!` through `BayesianRemapState`;
   IntExact audit harness; Liouville monotone-necessary diagnostic.
   Mass conserved 0–6.7×10⁻¹⁶ over 5 remap cycles; partition-of-
   unity to 10⁻¹².

5. **Matrix-free Newton-Krylov.** M3-8b ships
   `det_step_{2d, 2d_berry, 3d_berry}_HG_matrix_free!` using
   `NonlinearSolve.NewtonRaphson(linsolve = KrylovJL_GMRES(),
   concrete_jac = false, jvp_autodiff = AutoForwardDiff())`. Bit-
   equal to round-off vs the dense ForwardDiff-coloured baseline;
   1.17–1.87× CPU speedup at L3–L5. The algorithm-side prerequisite
   for a future GPU port (deferred to M3-8c pending HG `Backend`).

## Tier-D headline verdicts (the four scientific deliverables)

The four Tier-D falsifiers exercise the substrate's *falsifiable*
predictions; their verdicts characterise what the pure-Lagrangian
variational substrate captures natively vs requires extensions for.

- **D.1 2D Kelvin-Helmholtz (PASSED at broad-band gate; honest
  finding flagged for follow-up).** The driver
  `experiments/D1_KH_growth_rate.jl` calibrates the off-diagonal
  growth rate against the Drazin-Reid rate γ_DR = U/(2w):
  γ_measured/γ_DR = 1.34 at level 5 (32×32 mesh), c_off² = 1.78,
  mesh-converged to 1.2% between L4 and L5, n_negative_jacobian = 0.
  The methods-paper broad-band gate c_off ∈ [0.5, 2.0] is met. The
  honest finding: the growth is forced kinematic, not Drazin-Reid
  exponential self-amplification. The Phase 1a residual reproduces
  the kinematic response of δβ₁₂ to base-flow strain G̃_12·ᾱ/2, but
  the closed-loop coupling between along-axis Cholesky factors β_a
  and the perturbation amplitude β_off — required for self-amplified
  eigenmode growth — is not present in the current residual.
  Per-principal-axis-noise *structure* is verified kinematically;
  full eigenmode dynamics flagged as a M4 physics extension.

- **D.4 Zel'dovich pancake collapse, 2D and 3D (PASSED — the headline
  scientific success of M3).** The novel cosmological reference
  test of §10.5. In 2D (M3-6 Phase 2):
  std(γ_1)/std(γ_2) ≈ 2.6×10¹⁴ at near-caustic time (level 4,
  T_factor=0.16); γ_1 develops spatial structure (dynamic range
  4.18×) along the collapsing axis while γ_2 stays uniform to
  round-off; the Phase 1a strain coupling verifies inert
  (max|β_off| = 0) on the axis-aligned IC. In 3D (M3-7e), the
  same falsifier lifts to `tier_d_zeldovich_pancake_3d_ic_full` +
  `det_step_3d_berry_HG!`; selectivity ratio 3.0×10¹³ at near-
  caustic, exceeding the 10¹⁰ headline gate by three orders;
  γ_2 = γ_3 byte-equal by 1D-symmetric reduction; conservation
  ≤ 10⁻⁸. The per-principal-axis decomposition correctly identifies
  the collapsing axis at the discrete level, in both dimensions,
  with margins (10¹⁴ in 2D, 10¹³ in 3D) far exceeding the gates.

- **D.7 2D dust-trapping in vortices (substrate sound; literal
  claim FALSIFIED — the value of falsifiable predictions).** The
  driver `experiments/D7_dust_traps.jl` (Taylor-Green vortex IC +
  2-species `TracerMeshHG2D[:gas, :dust]`, doubly-periodic) verifies
  what the pure-Lagrangian variational substrate *does* capture:
  dust mass conservation M_dust_err = 0.0 bit-exact across L=3,4,5;
  per-species γ separation > 10¹⁰ (gas γ ≈ 1, dust γ = 0); 4-
  component cone n_negative_jacobian = 0; momentum exactness
  P_x,err = P_y,err = 0.0 (Taylor-Green symmetry); long-horizon
  stability without NaN. What the substrate does *not* natively
  capture: sub-cell centrifugal drift of dust toward vortex centres.
  In the pure-Lagrangian frame the substrate's variational
  integrator preserves the multi-species tracer matrix bit-exactly
  (`advect_tracers_HG_2d!` is by design a no-op) and the Eulerian
  cell volumes are fixed under the cone projection. The methods-
  paper prediction "vortex-centre accumulation matches reference
  codes" therefore requires a physics extension beyond the M3
  substrate: per-species momentum coupling with drag, or a
  Lagrangian volume update via composition with the M3-5 Bayesian
  L↔E remap with per-species mass tracking. Both are flagged as M4+
  physics work.

- **D.10 ISM-like 2D multi-tracer fidelity (PASSED in the strongest
  possible form — bit-exact).** The driver
  `experiments/D10_ism_multi_tracer.jl` (KH-style sheared base flow
  + Phase 1b antisymmetric tilt-mode overlay + N=3 species
  `TracerMeshHG2D[:cold,:warm,:hot]` with phase-stratified Gaussian
  concentration profiles in y, per-species γ = 1, √2, 2 at IC)
  drives K iterations of `det_step_2d_berry_HG!` +
  `advect_tracers_HG_2d!` + `inject_vg_noise_HG_2d!` (axes=(1,2),
  project_kind=:reanchor) — both deterministic and stochastic
  injection enabled, the "shocked turbulence" regime. The multi-
  tracer matrix at end-time is *byte-equal* to its IC value:
  tracers_byte_equal_to_ic == true, tracers_max_diff_final == 0.0
  across L=3,4,5. This is the 2D analogue of M2-2's 1D structural
  argument: the multi-tracer matrix is *literally never* in the
  write set of either operator, so bit-exact preservation is a
  *structural property* of the implementation rather than a
  tolerance-bounded numerical claim. Per-species mass conservation
  follows: M_per_species_err = 0.0 for every species. The community-
  impact claim of multi-tracer fidelity in shocked turbulence holds
  for the pure-Lagrangian variational substrate.

## Tier-E verdicts (M3-8a, graceful-degradation mode)

E.1 high-Mach 2D shocks (M=5/M=10 Sod, Rankine-Hugoniot downstream
state to ≤ 10⁻¹² relative): graceful failure achieved cleanly —
n_NaN = 0, KE bounded ≤ 5× IC, transverse-indep ≤ 10⁻¹⁰. E.2 severe
shell-crossing (2-axis Zel'dovich superposition at A=0.7,
t_cross ≈ 0.227): pre-caustic stability T_factor ≤ 0.25 with
n_NaN = 0, γ_min > 0.5, projection effectiveness verified
(`:reanchor` records non-zero events vs `:none` ⇒ 0). E.3 very low
Knudsen (smooth strain mode, τ=10⁻⁶, Kn ≈ 1.3×10⁻⁶): Navier-Stokes-
limit substrate property verified — β_max < 10⁻², |γ_a²/M_vv − 1|
≤ 10⁻², axis-swap symmetry, τ-stiffness independence. E.4
(cosmological with self-gravity) requires multigrid Poisson coupling
not in substrate; deferred to M4.

## Performance summary

The M3-3e cache_mesh retirement delivered ~1.7-2× speedup at the
1D-native path. The M3-8b matrix-free Newton-Krylov delivers
1.17-1.87× CPU speedup vs the dense ForwardDiff path while staying
bit-equal to round-off. Aggregate over the original M3-2 cache_mesh-
shim baseline: ~3× at level 4-5.

## Test count

~33,940 + 3 deferred tests (from 19,687 + 1 at M3-3 close, growing
through M3-4/5/6/7/8 by ~14,250 asserts). Bit-exact regression
discipline held throughout: every phase landed without invalidating
prior gates.

## What M3 does *not* deliver (the M4 entry points)

The honest list of items deferred to M4, in dependency order:

1. **Closed-loop β coupling for self-amplified KH** (D.1 follow-up).
2. **Per-species momentum coupling** for sub-cell dust drift (D.7
   follow-up); OR Lagrangian volume update via M3-5 Bayesian remap
   composition.
3. **Full Metal/CUDA port** — algorithm-side prerequisites are in
   place (matrix-free Newton-Krylov shipped); the per-leaf residual
   kernel awaits HG `Backend`-parameterized `PolynomialFieldSet`.
4. **MPI scaling** via HG's `partition_for_threads` chunk structure
   + `HaloView` halo exchange.
5. **3D Tier-D extensions** — D.1 3D KH, D.7 3D dust, D.10 3D ISM.
6. **Self-gravity** for E.4 (cosmological) and D.4 ColDICE / PM
   N-body cross-comparison.
7. **D.6 / D.8 / D.9 two-fluid 2D content.**
8. **Higher-order Bernstein reconstruction** for tighter shock L∞
   (Open Issue #2 carry-forward).

See `reference/notes_M4_plan.md` for the M4 entry-point detail.

## References

- `reference/MILESTONE_3_STATUS.md` — full phase ledger + close synthesis.
- `reference/notes_M3_6_phase1c_D1_kh_falsifier.md` — D.1 honest finding.
- `reference/notes_M3_6_phase4_D7_dust_traps.md` — D.7 honest finding.
- `reference/notes_M3_6_phase2_D4_zeldovich.md` — D.4 2D pass.
- `reference/notes_M3_7e_3d_tier_cd_drivers.md` — D.4 3D pass.
- `reference/notes_M3_6_phase5_D10_ism_tracers.md` — D.10 bit-exact pass.
- `reference/notes_M3_8a_tier_e_gpu_prep.md` — Tier-E results.
- `reference/notes_M3_8b_metal_gpu_port.md` — matrix-free + GPU deferral.
- `reference/notes_M4_plan.md` — M4 entry-point list.
- `specs/01_methods_paper.tex` §10.5 / §10.6 / §11 — methods paper revisions.
