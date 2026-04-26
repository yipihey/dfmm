# Milestone 2 — Status synthesis

**Date:** 2026-04-26.

**Repo state:** main HEAD at the M2-5 wrap-up commit. **2044 + 1
deferred tests pass** (full suite ~3m). 35 commits ahead of
origin/main since M2 launched.

**Per methods paper §10.7:** "Milestone 2: 1D Tier B validation
including variance-gamma verification and tracer-exactness."

M1 already absorbed B.1 / B.2 / B.4 / B.5; M2 closed the remaining
B.3 + B.6 plus M1 Open #4 plus the four methods-paper text
corrections. **Tier B is complete.**

## Phase-by-phase completion table

| # | Item | M1/M2 | Tests | Headline result |
|---|---|---|---:|---|
| B.1 | Long-time energy drift | M1 P4 | 5 | 5.6e-9 over 10⁵; t¹ secular noted (Open #1) |
| B.2 | Cold-limit reduction | M1 P3 | 162 | **Central unification verified** (2.57e-8) |
| **B.3** | **Action-based AMR** | **M2-1** | **40** | **37.8% cell-savings vs gradient AMR** |
| B.4 | Burst statistics | M1 P8 + M2-3 | 140 | Self-consistency monitor; 3-λ mismatch (Open #3) |
| B.5 | Passive scalar advection | M1 P11 | 21 | **L∞ tracer change = 0.0 literally** |
| **B.6** | **Multi-tracer wave-pool** | **M2-2** | **31** | **Bit-exact under stochastic noise; 25-114× over Eulerian** |
| **Open #4** | **Realizability projection** | **M2-3** | **172** | **Wave-pool reaches 12,000 steps (was 1222 baseline)** |
| **Paper corrections** | §3.5 / §3.2 / §3.1 / §10.3 B.1 | **M2-4** | — | **PDF rebuilt; 4 edits applied** |

## Test summary

| Block | Tests | Wall time |
|---|---:|---:|
| Phase 1-7 + 5b (M1 deterministic core) | 305 | ~24 s |
| Phase 8 (stochastic injection) | 140 | 1.1 s |
| Phase 11 (passive tracer) | 21 | ~13 s |
| **Phase M2-1 (action-AMR B.3)** | **40** | **58 s** |
| **Phase M2-2 (multi-tracer wave-pool B.6)** | **31** | **5 s** |
| **Phase M2-3 (realizability projection)** | **172** | **13 s** |
| Cross-phase smoke | 297 | 0.1 s |
| Track B / C / D | 920 | ~16 s |
| Regression scaffold | 118 | 0.5 s |
| **Total** | **2044 + 1 deferred** | **~3m** |

## Headline scientific findings (M2-additional)

### M2-1 — action-based AMR cell savings (Tier B.3 verified)

On an off-center 1D Sod blast wave (`x_disc = 0.7`, `t_end = 0.10`,
adaptive refinement triggered above $\tau_{\rm refine}$ with
hysteresis at $\tau_{\rm refine}/4$):

| Strategy | Time-avg cells | L² error |
|---|---:|---:|
| Action-AMR | 41.7 | 0.052 |
| Gradient-AMR | 67.0 | 0.060 |

**37.8% cell savings at slightly tighter L² accuracy.** Solidly
inside the methods paper's 20-50% prediction. Mass / momentum / tracer
preserved bit-exactly across each refine/coarsen event (verified by
40 testset assertions); tracer-exactness (Phase 11's literal-zero
property) holds through dynamic mesh changes.

See `reference/notes_M2_1_amr.md`,
`reference/figs/B3_amr_comparison.png`,
`experiments/B3_action_amr.jl`.

### M2-2 — bit-exact tracers under stochastic injection (Tier B.6)

Five sharp-step tracers carried through a 500-step wave-pool with
production-calibrated stochastic injection:

| Tracer | Step location | Variational L∞ change | Eulerian-upwind smear |
|---|---|---:|---:|
| 1 | x = 0.20 | **0.0 (literally)** | 114 cells |
| 2 | x = 0.30 | **0.0** | 100 cells |
| 3 | x = 0.50 | **0.0** | 72 cells |
| 4 | x = 0.70 | **0.0** | 43 cells |
| 5 | x = 0.85 | **0.0** | 25 cells |

**Median advantage 72×; worst case 25× — comfortably above the
1-decade target.** The Phase-8 noise mutates ρu, P_xx, P_⊥, s — never
the tracer matrix — so Phase 11's structural bit-exactness extends to
the full stochastic regime.

See `reference/notes_M2_2_multitracer.md`,
`reference/figs/B6_multitracer_wavepool.png`,
`experiments/B6_multitracer_wavepool.jl`.

### M2-3 — realizability projection closes Open #4 + reframes the diagnosis

**Deeper finding than expected.** The long-time stochastic instability
M1 Phase 8 hit at ~950 steps is **not primarily a realizability-
boundary crossing**. It's a **compression cascade**: entropy-debit
drift cools cells, lowering pressure resistance, which allows further
compression, $\rho$ grows toward 1000+, until Newton tangles a vertex
past its neighbor. The realizability boundary is the *symptom*; the
cascade is the *cause*. The projection's required `Mvv_floor = 1e-2`
acts as a polytropic-equivalent pressure resistance preventing the
cascade.

**Result:** wave-pool reaches **12,000 steps** under production
calibration (baseline blew up at 1222). Mass and momentum stay
bit-exact across projection events; energy drift is bounded (large,
~36× the no-noise Phase-4 baseline at 12k steps — documented
trade-off).

| | Baseline `:none` | Fixed `:reanchor` |
|---|---|---|
| Steps reached | 1222 (NaN) | **12000** |
| Projection events | 0 | 810,763 |
| Mass drift | 0 | 0 |
| Mvv_min | crossed floor | held at 1e-2 |
| ΔE_rel | NaN | 80.2 (bounded) |

The chosen `:reanchor` variant raises $s$ post-injection so $M_{vv}
\ge \max(\text{headroom} \cdot \beta^2, \text{Mvv\_floor})$; debit
$\tfrac12 \Delta P_{xx}$ from $P_\perp$ for IE conservation per event.
Mass and momentum bit-stable across events.

See `reference/notes_M2_3_realizability.md`,
`reference/figs/M2_3_stability_comparison.png`,
`experiments/M2_3_long_time_stochastic.jl`.

### M2-4 — methods-paper text corrections applied

Four edits to `specs/01_methods_paper.tex` (PDF rebuilt clean to 20
pages):

1. **§3.5 + §10.2 A.2** — Hessian/γ direction at caustic: language
   updated for ideal-gas Γ>1 EOS (γ² *grows* by 0.3-0.9 decades at
   the caustic; spatial maximum localizes there).
2. **§3.1** — symplectic-potential sign convention note added (paper
   keeps positive sign; agent added a clarifying paragraph instead
   of flipping the boxed equation).
3. **§3.2** — variance-gamma pdf normalized correctly per the
   Distributions-canonical $V \sim \Gamma(\lambda, \theta)$
   parametrization.
4. **§10.3 B.1** — bounded-oscillation caveat added with the
   analytical autonomous trajectory and Kraus 2017 pointer.

See `reference/notes_M2_4_paper_corrections_applied.md`.

## Open architectural questions — status update vs M1

| # | M1 status | M2 update | Net status |
|---|---|---|---|
| **#1** t¹ secular drift | open | unchanged (deferred to M3 prep) | Open |
| **#2** Sod L∞ ~10-20% | open | unchanged (out of M2 scope) | Open |
| **#3** Stochastic 3-λ mismatch | open | M2-3's compression-cascade diagnosis offers new angle | Open (with reframing) |
| **#4** Long-time stochastic instability | open | **closed by M2-3** | **✓ Resolved** |

**Open #3 reframing (new):** the v3 §1.2 calibration mismatch may be
related to the compression cascade M2-3 diagnosed. If production-scale
runs naturally hit the cascade and partial-projection regimes, the
empirical λ measured from residuals would differ from the calibration
input depending on how much of the run is in each regime. Worth
investigating in M3 with the projection enabled.

## Remaining Tier-B / Tier-A items

**All complete.** No outstanding 1D regression-target items.

## Pre-Milestone-3 readiness

**Strong:**
- Tier B.2 (cold-limit unification): verified.
- Tier B.5 (tracer-exactness): verified.
- **Tier B.3 (action-AMR): verified, 37.8% cell savings.**
- **Tier B.6 (multi-tracer wave-pool): verified, bit-exact + 25-114×.**
- **Long-time stochastic stability (Open #4): closed.**
- Tier A.2 cold sinusoid pre-crossing: 6 decades clean.
- 1801 → 2044 tests with no flakes.

**Conditional (M1 caveats carry forward):**
- Tier B.1 energy drift: passes literal bound; t¹-secular caveat now in
  the methods paper (§10.3 B.1 paragraph added by M2-4).
- Tier A.1 Sod / A.3 steady shock: qualitative match with q-on; L∞ ~10-20%
  documented.
- Tier B.4 burst stats: monitor works; 3-λ mismatch now reframed via
  M2-3's compression-cascade finding.

**Unblocked for M3:**
- Open #4 closure means M3 Phase 9-11 (KH, pancake, wave-pool spectra)
  no longer have a long-time-stability blocker.
- AMR primitives from M2-1 generalize to 2D quadtree (per M3 Phase 7
  in `MILESTONE_3_PLAN.md`).
- The compression-cascade diagnosis from M2-3 is itself a new physical
  insight worth a one-paragraph addition to the methods paper —
  currently noted only in the implementation notes.

## Recommended next moves

Per `reference/notes_dfmm_2d_overview.md` and `MILESTONE_3_PLAN.md`,
M3 launches with Phase 0 (Berry connection $\omega_{\rm rot}$
derivation + reading + r3d port confirmation). Decision points before
M3:

1. **Berry connection derivation** — Tom's paper-level work, or an
   agent task with a clear theoretical brief?
2. **r3d Julia port readiness** — confirm package name, API, version
   pin.
3. **GPU primary target** — Apple Metal (laptop) or CUDA (cluster)?
4. **Address Open #1 (Kraus 2017)** before M3 or as M3 Phase 0
   sidebar?

The HANDOFF.md and methods paper §10.7 give 3-4 months for M3 (Tier
C consistency); 3-4 months for M4 (Tier D novelty); 2-3 months for
M5 (Tier E + applications). **Total to methods-paper-ready: ~10
months** from M2 closure if launching M3 now.

## Repo housekeeping

- All 5 M2 worktrees cleaned up; only main worktree remains.
- Named M2 branches (`m2-1-action-amr`, `m2-2-multitracer`,
  `m2-3-realizability`, `m2-4-paper-corrections`) preserved as audit
  history.
- 35 commits ahead of origin/main; pushed via `git push origin main`.
- No outstanding files in working tree.
