# Milestone 1 — Status synthesis

**Date:** 2026-04-25 (Phase 7 just landed during synthesis pass).

**Repo state:** main HEAD at `Phase 7: Tier A.3 steady shock + heat-flux λ_Q`. **1504 + 1 deferred tests pass** (full suite ~2m25s). 26 commits ahead of origin/main since the milestone began. **One inter-phase regression** found and fixed during synthesis: Phase 8's `inject_vg_noise!` mutated segments via 6-arg `DetField{T}(...)` parametric constructor that no longer existed after Phase 7's struct extension; fixed by passing through `seg.state.Q` as the 7th arg.

**No active worktrees.** All 12 stale worktrees cleaned up; only the named phase branches remain as audit history.

This document is the synthesis input for either (a) declaring Milestone 1 complete and moving to Milestone 2, or (b) opening a debugging session to address the open architectural questions before the remaining Tier-A regression tests (A.4, A.5, A.6) land.

## Phase-by-phase completion table

Phases mirror the breakdown in `reference/MILESTONE_1_PLAN.md`.

| # | Phase | Status | Tests | Headline result |
|---|---|---|---:|---|
| 0 | Foundations + reading | ✓ | — | `notes_phaseA0.md` hand-derivation |
| 1 | Cholesky integrator | ✓ | 10 | symplectic; closed-form match to 10⁻¹² |
| 2 | Bulk + entropy | ✓ | 13 | mass/momentum exact; acoustic c_s 0.64% |
| 3 | Tier B.2 cold limit | ✓ headline | 162 | **central unification verified** — Zel'dovich match 2.57e-8 |
| 4 | Tier B.1 energy drift | ✓ literal / open trend | 5 | 5.6e-9 over 10⁵ steps; **t¹ secular** noted |
| 5 | Tier A.1 Sod | ✓ partial | 9 | qualitative dfmm Fig. 2 match; **L∞ rel ~20%** (no q) |
| 5b | Tensor-q opt-in | ✓ | 72 | q-on dampens spike; L∞ improves ~10%, doesn't reach 0.05 |
| 6 | Tier A.2 cold sinusoid τ-scan | ✓ + 1 deferred | 69 + 1 | 6 decades green pre-crossing; post-crossing golden deferred |
| 7 | Tier A.3 steady shock | ✓ | 84 | R-H at M_1∈{1.5..10} to ≥3 decimals; long-horizon shock-capturing limited |
| 8 | Stochastic injection (B.4) | ✓ partial | 140 | wave-pool 3994 bursts; **monitor fires** (mismatch 20×) |
| 9 | B.4 burst statistics | bundled w/ Phase 8 | (incl) | self-consistency monitor working as designed |
| 10 | Tier A.4 KM-LES wave-pool | not started | — | depends on Phase 8 stability fix |
| 11 | Tier B.5 passive scalar | ✓ | 21 | **L∞ tracer change = 0.0** (literally); methods-paper claim verified |
| 12 | Tier A.5 dust-in-gas | not started | — | two-fluid; deferred |
| 13 | Tier A.6 plasma | not started | — | two-fluid; deferred |
| 14 | Wrap-up | this doc | — | (in progress) |

**Aggregate test count: 1504 pass + 1 deferred** (Phase 6 post-crossing golden, awaiting tensor-q-or-equivalent shock fidelity).

**Completed Tiers:**
- Tier A.1 Sod (Phase 5): qualitative ✓, L∞ off-target.
- Tier A.2 cold sinusoid (Phase 6): pre-crossing ✓, post-crossing deferred.
- Tier B.1 energy drift (Phase 4): literal ✓, asymptotic open.
- Tier B.2 cold limit (Phase 3): **headline ✓ — methods paper's central claim verified.**
- Tier B.4 burst statistics (Phase 8): monitor working; **mismatch is the finding.**
- Tier B.5 passive scalar (Phase 11): **headline ✓ — exact at the discrete level.**

**Outstanding Tiers (when ready):**
- Tier A.3 steady shock (Phase 7) — **R-H jumps verified at all M_1; long-horizon (t=3) shock-capturing limited by the same Phase-5/5b discrete-jump-condition issue.** Honest short-horizon plateau preservation: ✓.
- Tier A.4 wave-pool spectra (Phase 10 — depends on Phase 8 stability fix, Open #4).
- Tier A.5 dust-in-gas (Phase 12).
- Tier A.6 plasma equilibration (Phase 13).
- Tier B.3 action-based AMR — out of M1 scope (1D segments stay uniform).

## Figures (artifact gallery)

All under `reference/figs/`:

| File | Phase | What it shows |
|---|---|---|
| `phase3_hessian_degen.png` | 3 | Hessian diagnostic localizes caustic at m=1/2 across the cold-limit run |
| `phase_portrait_phase1.png` | 1 (demo) | (α, β) trajectory under zero-strain Hamiltonian flow |
| `B1_energy_drift.png` | 4 | t¹ secular drift trace, peak \|ΔE\|/E₀ = 5.6e-9 |
| `A1_sod_profiles.png` | 5 | Variational vs py-1d golden Sod, q=:none — visible shock-front spike |
| `A1_sod_q_comparison.png` | 5b | q-off vs q-on Sod overlay; q dampens spike, doesn't fix R-H |
| `A2_cold_sinusoid_tauscan.png` | 6 | 6-panel ρ vs Zel'dovich pre-crossing across τ ∈ [10⁻³, 10⁷] |
| `A2_cold_sinusoid_density.png` | 6 | 6-panel γ²/\|det Hess\| diagnostic |
| `B5_tracer_through_shock.png` | 11 | **bit-exact tracer vs Eulerian-upwind smearing — methods-paper selling point** |
| `B4_burst_statistics.png` | 8 | wave-pool burst histogram + self-consistency monitor mismatch |
| `A3_steady_shock_mach_scan.png` | 7 | Mach scan M_1 ∈ {1.5..10} — sharp R-H plateaus, BGK-relaxed bulk |

## Notes (per-phase journals)

All under `reference/`:

- `notes_phaseA0.md` — Phase-0 hand-derivation
- `notes_phase2_discretization.md` — Phase-2 staggered MAC layout choice
- `notes_phase3_solver.md` — why Newton escalation playbook was unused (cold limit is *easier*, not harder)
- `notes_phase4_energy_drift.md` — t¹ secular diagnosis + Kraus-2017 mitigation note
- `notes_phase5_sod.md` + `notes_phase5_sod_FAILURE.md` — discrete-jump-condition discrepancy
- `notes_phase5b_artificial_viscosity.md` — q saturates at ~10% improvement
- `notes_phase6_cold_sinusoid.md` — 6-decade τ-scan + golden post-crossing deferral
- `notes_phase8_stochastic_injection.md` — three λ values mutually inconsistent + long-time realizability instability
- `notes_phase11_passive_tracer.md` — structural argument for tracer-exactness
- `notes_phase7_steady_shock.md` — heat-flux Q hard-constraint, Mach scan, long-horizon caveat
- **`notes_methods_paper_corrections.md`** — the consolidated paper-edit todo

## Architectural opens (the debugging-session agenda)

Four substantive findings that need decisions/fixes before further Tier-A work:

### Open #1 — t¹ secular energy drift (Phase 4)

**Symptom.** $|\Delta E|/E_0$ over $10^5$ steps is bounded by 5.6e-9 (passes literal $<10^{-8}$ bound) but the trend is **t¹ secular**, not pure bounded oscillation. Extrapolating, drift crosses 10⁻⁸ at ~2×10⁵ steps.

**Diagnosis.** Autonomous (α, β) sector at fixed M_vv has *no closed orbits*; symplectic-averaging cancellation that produces bounded oscillation doesn't apply.

**Mitigations:**
- (a) Accept and document — passes literal acceptance.
- (b) Kraus 2017 *projected variational integrator* enforcing the level-set constraint $\alpha^2(M_{vv} - \beta^2) = \text{const}$ exactly each step. ~1-2 weeks structural rewrite.
- (c) Switch the discretization (Gauss-Legendre 2-stage = 4th order; possibly other symmetry).

**Recommendation:** if the methods paper's "bounded oscillation" claim is structural (must hold ∀t), do (b) before Phase 8 production runs.

### Open #2 — Sod L∞ rel ~10-20% (Phase 5/5b)

**Symptom.** Bare variational on Sod misses post-shock plateau by ~19% (vs analytic Riemann at τ=1e-5 — *not* just py-1d's HLL+MUSCL smearing). q-on (Phase 5b vNR) dampens the post-shock spike but L∞ saturates at ~10% on (ρ, Pxx, Pp). **L1 rel is 3-4%** (bulk profile is fine).

**Diagnosis.** Discrete-jump-condition discrepancy: the variational EL system's discrete shock states don't satisfy Rankine-Hugoniot exactly, even with arbitrary q strength.

**Mitigations:**
- (a) Accept ~10-20% L∞ + 3-4% L1 as "qualitative dfmm Fig. 2 reproduction" — methods paper §10.2 A.1 says "reproduce Fig. 2", no explicit tolerance. The 0.05 L∞ bar was self-imposed in `MILESTONE_1_PLAN.md`.
- (b) Flux-conservative reformulation of the EL system. ~1-2 weeks structural work; loses some variational purity.
- (c) Higher-order reconstruction (e.g. Bernstein cubic per methods paper §9.3) inside each segment. Complex.

**Recommendation:** (a) for Milestone 1 paper claims, (b) or (c) before Phase 7's high-Mach steady shock locks the result in.

### Open #3 — Stochastic injection: three λ values mutually inconsistent (Phase 8 + v3 §1.2)

**Symptom.** On wave-pool with calibrated noise:
- Production calibration `kurt = 3.45` ⇒ `λ_input = 6.7`.
- Burst-shape estimate from histogram: $\hat k = 0.73$.
- Residual-kurtosis-implied: $\hat\lambda_{\rm res} \approx 13-17$.
- v3 §1.2 small-data fit: λ ≈ 1.6.

The self-consistency monitor fires (ratio 20.1 vs warn=2.0) — *exactly its design intent*. The mismatch is real, not an integration bug.

**Diagnosis (three competing):**
- (a) Estimator bias on small samples — $\hat k$ method-of-moments biased low at the heavy-tailed end.
- (b) Burst-duration discretization noise — finite Δt undercounts short bursts.
- (c) Genuine chaotic-divergence floor (v3 §1.2 hypothesis) — the variance-mixed-Gaussian assumption breaks down at production scale.

**Mitigations:**
- (a) Higher Mach + longer run to see if estimator converges.
- (b) Adaptive-Δt burst detection (use sub-step interpolation).
- (c) MLE rather than method-of-moments shape fit (already implemented; agreement is ~0.1).

**Recommendation:** Investigate further with a calibrated production wave-pool run (Phase 10 prerequisite). The mismatch may be diagnostic of a real physical effect.

### Open #4 — Long-time stochastic-injection realizability instability (Phase 8)

**Symptom.** Production-calibrated drift slowly drains entropy on each compressive cell. After ~950 steps a cell crosses $M_{vv} < $ floor and Newton blows up.

**Diagnosis.** No realizability projection in the post-Newton stochastic step. The drift+noise can push individual cells out of the realizable region.

**Mitigations:**
- (a) State projection on the realizability boundary (Caramana-style "balance").
- (b) Adaptive dt that shrinks when the noise gets close to the floor.
- (c) Symmetric debit form: per-cell entropy production matches per-cell KE injection exactly (Phase 8 already does ⅔/⅓ split between Pxx and Pp; refine).

**Recommendation:** Implement (a) before Phase 10's full wave-pool spectra (which need 10⁴+ steps).

## Pre-Milestone-2 readiness

**Strong (ready for paper claims):**
- Tier B.2 unification (Phase 3) — verified.
- Tier B.5 tracer-exactness (Phase 11) — verified.
- Tier A.2 cold sinusoid pre-crossing (Phase 6) — 6 decades clean.

**Conditional (needs caveat):**
- Tier B.1 energy drift (Phase 4) — passes literal bound; cite t¹-secular caveat.
- Tier A.1 Sod (Phase 5/5b) — qualitative match; L∞ off-target. Cite ~10% with q, 3-4% L1.
- Tier B.4 burst stats (Phase 8) — monitor works; mismatch is the finding worth a paragraph.

**Phase 7 result (just landed):**
- R-H jump conditions at $M_1 \in \{1.5, 2, 3, 5, 10\}$ to ≥3 decimal places (machine-precision at $M_1 \le 5$, 5e-4 at $M_1 = 10$). Methods paper §10.2 A.3 acceptance bar (3 decimals) hit at all 5 Mach numbers.
- Golden match at $M_1=3$ to L∞ rel 8.5e-8 on rho — **far below the 0.05 bar.**
- Long-horizon (t=3) instability after ~1000 steps — same root cause as Phase 5/5b's Sod failure (no flux-conservative shock-jump in the bare Lagrangian variational scheme). Short-horizon plateau is machine-precision.

**Out of scope:**
- Tier A.4 wave-pool spectra (Phase 10) — needs Open #4 fixed.
- Tier A.5 dust-in-gas, A.6 plasma (Phases 12, 13) — two-fluid extensions.

## Recommended next moves

**Option A — declare Milestone 1 complete with caveats.** Treat the architectural opens as Milestone-2 inputs. Wrap up Phase 7 + a final wrap-up commit; hand off to M2 via this status doc + `notes_methods_paper_corrections.md`.

**Option B — debug then continue.** Address Opens #1 + #4 now (the structural ones), then do Phase 10 + 12 + 13 against the fixed integrator. ~3-4 weeks.

**Option C — pick the wins.** Accept the qualitative Sod / steady-shock results, fix only Open #4 (realizability projection — necessary for Phase 10 anyway), and skip the long-time-drift work until Phase 10's actual production needs surface it.

My recommendation is **C** — Open #4 is the only blocker for the remaining Tier-A regressions. Opens #1, #2, #3 are documented and citable; the methods paper can describe the qualitative results plus the open questions explicitly. That positions Milestone 1 as a "framework verification" with quantitative B.2 + B.5 wins and qualitative A.1/A.2/A.3.

## Repo housekeeping done in this synthesis pass

- 12 stale agent worktrees removed (Phase 7's still-running worktree intact).
- 11 auto-named `worktree-agent-*` branches deleted.
- Named phase branches (`phase{N}-*`, `agent-{A,B,C,D}-*`) preserved as audit history.
- This document and `notes_methods_paper_corrections.md` added to `reference/`.

Total commits ahead of origin: 25 + (Phase 7 once it lands) + (this housekeeping commit).
