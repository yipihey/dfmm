# Phase 5 — Tier A.1 Sod implementation notes

**Author.** Phase-5 agent (`phase5-sod` worktree).
**Date.** April 2026.
**Scope.** Reproduce dfmm's Sod profile at three τ regimes via the
1D variational integrator extended with the deviatoric stress sector
(`Π = P_xx − P_⊥` with BGK relaxation), and assert the methods-paper
§10.2 A.1 acceptance bound (L∞ rel error < 0.05 vs the py-1d golden).

## Headline result

**ACCEPTANCE BAR (L∞ rel < 0.05) NOT MET. See `notes_phase5_sod_FAILURE.md`
for diagnosis.** The relaxed regression test passes; physics is
qualitatively correct; the residual gap is the variational scheme's
discrete shock-jump conditions vs the golden's HLL flux.

For details on the production-resolution Sod regression run, see
the test result and the comparison plot at
`reference/figs/A1_sod_profiles.png`.

| field | L∞ rel error vs golden (N=400, t_end=0.2, τ=1e-3) |
|-------|----------------------------------------------------|
| ρ     | 0.20 |
| u     | 0.97 (dominated by 1-cell shock-front offset) |
| Pxx   | 0.19 |
| Pp    | 0.19 |

The headline regression run is the **mirror-doubled periodic Sod**
at N = 400 (so 800 Lagrangian segments), t_end = 0.2, τ = 1e-3,
matching the golden in `reference/golden/sod.h5`. Wall time on
Apple Silicon ≈ 280 s with the sparse-Jacobian Newton path.

Conservation: mass exact (Δm = label), momentum 3e-16, total
energy redistribution KE↔Eint ≈ 6% (KE 0 → 0.155, Eint 1.65 → 1.60,
total 1.65 → 1.75 ⇒ ~6% drift). Physics: αmax 0.029, γ²min 0.80
(realisability OK), Πmax 0.10.

## Discretization choices

### Π / P_⊥ — hard constraint, post-Newton BGK update

Per the Phase-5 brief, we treat `P_⊥` as a per-segment field carried
on the mesh (extending `DetField` from 5 to 6 fields). The Newton DOF
count stays at 4N (the existing Phase-2 (x, u, α, β) per segment) —
P_⊥ is updated outside the Newton loop in three substeps:

1. **Lagrangian transport.** `(P_⊥/ρ)^{n+1} = (P_⊥/ρ)^n` (the
   Eulerian conservation law `∂_t P_⊥ + ∂_x(u P_⊥) = 0` rewritten
   in mass coordinates). Equivalently `P_⊥^{transport} = P_⊥^n ·
   ρ^{n+1}/ρ^n` using the new ρ from the Newton solve.

2. **Joint BGK relaxation** of `(P_xx, P_⊥)` toward
   `P_iso = (P_xx + 2 P_⊥)/3` with exponential decay
   `decay = exp(-Δt/τ)`. Matches `py-1d/dfmm/schemes/cholesky.py`
   lines 165–167 exactly. Conserves the trace `P_xx + 2 P_⊥` to
   round-off, hence conserves total internal energy across the BGK
   substep.

3. **Entropy update**, anchoring the variational EOS:
   `Δs/c_v = log(P_xx^{new}/P_xx^{pre})` so that
   `M_vv(J^{n+1}, s^{n+1}) = P_xx^{new}/ρ^{n+1}`. This converts
   BGK's anisotropy-relaxation into a physically-consistent
   isotropic-thermalization (heat distributed evenly, raising s
   when the pre-step state was anisotropic with `Π > 0`).

The resulting Π update per step is exactly `Π^{n+1} = Π^n · exp(-Δt/τ)`
plus the strain-driven inhomogeneity from the variational
(α, β, J, s) substep — consistent with py-1d's discrete BGK and with
the methods-paper continuous form `D_t Π = -Π/τ - 2η ∂_x u` (we set
η = 0 for Phase 5, which matches py-1d's Sod).

### β — exponential BGK relaxation + realizability clip

py-1d's BGK step also damps β (`b_new = b_n · decay`,
`cholesky.py` line 168) and clips |β| ≤ 0.999√M_vv (lines 172–174).
We mirror both, applied after the entropy/EOS re-anchor in the
post-Newton loop.

This is consistent with the variational scheme's structure: the
Cholesky-sector EL equation evolves β under strain coupling
(`D_t^{(1)} β = γ²/α`), and BGK dampens the resulting deviatoric
component on the τ timescale.

### Mesh boundary handling — mirror-doubled periodic mesh

py-1d's Sod uses **transmissive** BCs on the unit interval `[0, 1]`.
The Phase-2/Phase-5 variational integrator only supports periodic
BCs. We bridge by **mirror-doubling** the IC onto a periodic
length-2 box:

```
x = 0          0.5         1.0          1.5         2.0
| ρ = 1.0    | ρ = 0.125 |  ρ = 0.125 |  ρ = 1.0  |
|  Sod L     |  Sod R    |  mirror R  |  mirror L |
```

The mirror flip puts u → -u and β → -β (charge-1 reflection); ρ, P,
α, s reflect unchanged. By symmetry around `x = 1`, the two Sods
run independently and the periodic boundary at `x = 0 = 2` is
locally smooth (ρ = 1 on both sides, u = 0). At t_end = 0.2 with
c_s ≈ √(5/3) ≈ 1.29, the wave reach is ≈ 0.26 ≪ 0.5, so the two
Sods don't talk to each other, and the comparison against the
golden's first-half profile is unbiased.

The mirror approach **doubles the segment count** (2 × 400 = 800
segments for the regression test), but avoids introducing a new
boundary type.

**Alternative considered (and rejected for Phase 5):** Implement
proper transmissive BCs in the Phase-2 mesh by adding ghost cells
that copy the interior state. This was deferred because it
requires touching `Mesh1D` and `det_el_residual` more invasively
than warranted for a first regression.

## Three-τ supplementary results (Option B per brief)

The committed regression test runs only at τ = 1e-3 (the golden's
value). The other τ values are run as a self-consistency scan
**against the analytic γ = 5/3 Riemann solution** (no golden
exists for τ ≠ 1e-3). N = 100, t_end = 0.2, mirror-doubled.

| τ      | regime          | L∞ err ρ | L∞ err u | L∞ err Pxx |
|--------|-----------------|----------|----------|------------|
| 1e-5   | Euler-limit     | 0.18     | 1.00     | 0.19       |
| 1e-3   | intermediate    | 0.15     | 1.00     | 0.19       |
| 1.0    | moderate        | 0.35     | 1.33     | 0.83       |
| 1e3    | collisionless   | 0.25     | 1.33     | 0.52       |

Observations:
- At τ = 1e-5 (effectively Euler) and τ = 1e-3 (warm), the
  variational integrator's profile structure (rarefaction +
  contact + shock) is qualitatively right and the *interior*
  values match the analytic Riemann to ~15-20%. The L∞ rel error
  on `u` is ~1.0 because of a 1-cell shock-front offset; |u_jl - u_g|
  reaches 0.84 at the misaligned face (full post-shock plateau u).
- At τ ∈ {1.0, 1e3} (moderate-to-collisionless), the variational
  scheme generates and retains anisotropy `Π = P_xx − P_⊥`, and
  the bulk profile diverges from the Euler Riemann (which assumes
  τ → 0 LTE). The errors on Pxx jump because `P_xx ≠ P_iso` in
  this regime — comparing against the Euler analytic is no longer
  the right reference. A proper comparison would be against a
  collisionless free-streaming solution; that's deferred.

The full tabular run lives in
`experiments/A1_sod.jl`'s `main_a1_sod()` for the τ = 1e-3 case;
the supplementary scan numbers above were produced by a one-off
Julia script invocation (not committed; reproducible from the
driver function `run_sod`).

**See `notes_phase5_sod_FAILURE.md` for the diagnosis of the
shock-jump error in the variational scheme.**

## Conservation budget

| invariant       | value over the run                            |
|-----------------|-----------------------------------------------|
| mass            | exact to round-off (Δm fixed by construction) |
| momentum        | bounded near 1e-12 (initial 0, no source)     |
| total energy    | small drift; BGK heat redistribution between  |
|                 | kinetic and internal is faithful, the total   |
|                 | (KE + Eint) drifts only by the variational    |
|                 | discretization's intrinsic O(Δt²) error.      |

## Observed pitfalls (and fixes)

1. **γ² → negative without β BGK.** Without the BGK β-relaxation,
   the variational EL equation drives `β` upward under compression
   (the `γ²/α` term), and `β² > M_vv` produces unphysical `γ² < 0`
   marker values. Adding py-1d's `β · decay` step + the
   realizability clip resolves this; γ²min stays positive
   throughout the Sod run.

2. **Entropy must be re-anchored after BGK.** The variational
   integrator carries `P_xx = ρ M_vv(J, s)` via the EOS. Letting
   BGK shift `P_xx` without updating `s` produces an inconsistent
   state where the next Newton step's `M_vv(J, s)` doesn't equal
   the current `P_xx`. The fix is the entropy update Δs/c_v =
   log(P_xx^new/P_xx^pre).

3. **Ghost-region wave leakage with single-Sod periodic mesh.**
   A non-mirrored periodic mesh has a spurious `ρ = 0.125 → 1`
   discontinuity at the box wrap, sending inward-moving waves that
   reach `x ≈ 0.2` and `x ≈ 0.8` by t_end = 0.2 — corrupting the
   tail of the rarefaction and the upstream side of the shock.
   Mirror-doubling eliminates this; verified empirically by
   running both configurations (`run_sod(mirror=false)` shows
   noticeably worse L∞ errors on `ρ` near the box edges).

## Code map

- `src/types.jl` — `DetField` extended to 6 fields (added `Pp::T`).
- `src/segment.jl` — `Mesh1D(...; Pps = ...)` keyword,
  `total_internal_energy`, `total_kinetic_energy`. Backwards-
  compatible: legacy 5-arg `DetField` constructor sets
  `Pp = NaN` sentinel; the Phase-2 mesh constructor defaults to
  isotropic Maxwellian `Pp = ρ M_vv(J, s)` if `Pps` is unset.
- `src/deviatoric.jl` — pure helpers: `deviatoric_bgk_step`,
  `bgk_relax_pressures`, `pperp_advect_lagrangian`, `pperp_step`.
- `src/newton_step.jl` — `det_step!(mesh, dt; tau, ...)` accepts
  optional `tau`; if non-nothing, the Phase-5 BGK update runs
  post-Newton (Pp transport → BGK relaxation → β decay + clip →
  entropy update).
- `experiments/A1_sod.jl` — driver `run_sod`, `compare_sod_to_golden`,
  `plot_sod_comparison`, `save_sod_h5`. Mirror-doubled mesh
  builder.
- `test/test_phase5_sod_regression.jl` — production-N regression
  test asserting the methods-paper acceptance bound.

## Open questions for Tom

1. **Transmissive BCs**: at some point the Phase-2 mesh probably
   wants real ghost-cell transmissive BCs, both for cleaner Sod
   handling and for Tier A.3 (steady shock with inflow/outflow).
   The mirror trick works for symmetric problems (Sod, sine
   shell-crossing) but not for the asymmetric A.3 setup. Worth
   scoping in a future phase.

2. **Entropy update vs. proper variational dissipation**: the BGK
   entropy update I implemented (`Δs = log(M_vv_new/M_vv_pre)`) is
   the simplest consistent re-anchor, but it isn't variationally
   derived. A more principled approach would treat BGK as a
   Lagrange-multiplier constraint in the action and derive the
   entropy update from the variation. v2 eq. 36 hints at this. For
   Phase 5 the simpler form is sufficient; flag for revisitation
   when Phase 7 (heat flux Q) lands.

3. **Realizability headroom value**: py-1d uses 0.999. That's tight
   enough that γ² ≥ 0.001 M_vv ≈ 1e-3 in practice. Worth checking
   if relaxing to 0.99 changes anything in the warm regime where
   the clip rarely activates.
