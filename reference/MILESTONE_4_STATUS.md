# Milestone 4 — Status

> **Created (2026-04-26):** at M4 entry, immediately after M3 close
> (`reference/MILESTONE_3_FINAL.md`). Tracks M4-as-shipped phase by
> phase against `reference/notes_M4_plan.md`.

## M4 phase ledger

| Phase | Status | Test delta | Notes |
|---|---|---:|---|
| M4-1: closed-loop β_off ↔ β_a coupling | **CLOSED** (HONEST_FALSIFICATION) | +598 | D.1 KH: closed-loop preserves regression but does not activate eigenmode |
| M4-2: per-species momentum (D.7 follow-up) | not started | — | |
| M4-3: 3D Tier-D headlines | not started | — | |
| M4-4: full Metal/CUDA port | not started (HG `Backend` blocker) | — | |
| M4-5: MPI scaling | not started | — | |
| M4-6: E.4 cosmological + ColDICE | not started | — | |
| M4-7: D.6 / D.8 / D.9 two-fluid 2D | not started | — | |
| M4-8: Bernstein reconstruction | not started | — | |
| M4-9: methods paper resubmission | not started | — | |

## M4-1 close (2026-04-26)

**Headline scientific finding (HONEST_FALSIFICATION):** The closed-
loop back-reaction Hamiltonian
`H_back = c_back · G̃_12 · (α_2·β_12·β_2 + α_1·β_21·β_1)/2` extends
the M3-6 Phase 1a forward strain coupling
`H_rot^off = G̃_12·(α_1·β_21 + α_2·β_12)/2` with a symmetric
β_off ↔ β_a coupling. The closed-loop:

- **Closes the symplectic loop** (∂H_back/∂α_a, ∂H_back/∂β_off,
  ∂H_back/∂β_a are all activated; the discrete EL residual now has
  bidirectional coupling between the diagonal Cholesky sector and
  the off-diagonal sector).
- **Preserves bit-exact regression** at β_off = 0 IC across all M3
  test suites (M3-3c, M3-4, M3-6 Phase 0). The H_back contributions
  are multiplicative in β_off · β_a (in F^β_a) or β_a (in F^β_off)
  so they vanish identically at the M3-3c regression IC.
- **Fixes a missing 1/α_a² normalization** on the M3-6 Phase 1a
  H_rot^off α-derivative term in F^β_a (the previous form was
  correct only at α = 1; M4 Phase 1's form is symplectic-natural at
  all α).

**Falsifier verdict:** at level 4 (16×16) and level 5 (32×32) of
the D.1 KH falsifier, the closed-loop residual produces
`c_off = γ_measured / γ_DR ≈ 1.26` (compared to `c_off ≈ 1.30` for
the M3-6 Phase 1a kinematic-only path). Both are within the methods
paper §10.5 D.1 broad band [0.5, 2.0]; neither tightens into the
M4 Phase 1 aspiration band [0.8, 1.2]. Linear-in-t fits the
trajectory better than exp-in-t (ssr_lin = 4.7e-5 ≪ ssr_exp =
0.078 at level 3 with c_back=1). **The Drazin-Reid eigenmode is
not activated by the closed-loop coupling alone.**

**Phase 1c finding refinement:** the M3-6 Phase 1c finding ("growth
is forced, not self-amplified") is structurally explained: closing
the symplectic loop via H_back is **necessary but not sufficient**
for the eigenmode dynamics. Three honest interpretations:

1. **Methods paper §10.5 D.1 prediction may need a different physics
   extension** — possibly per-cell linearised Rayleigh-equation
   reconstruction at the stencil level, or a higher-order
   Hamiltonian extension (cubic / quartic in perturbation
   amplitudes) that creates a positive-eigenvalue linear-mode
   block.

2. **The H_back form may need an antisymmetric variant** — the
   symmetric form preserves β_off ↔ β_a antisymmetry under axis
   swap but might lack the eigenvalue structure for instability.

3. **The Phase 1a kinematic kludge for F^β_off may be inappropriate**
   — the symplectic-natural EL form (no β̇_off, just α̇'s and θ̇_R
   balancing G̃·α/2) is mathematically more correct but requires
   Newton-system regularization to handle the rank-6 Casimir kernel.

**Test delta:** +598 asserts (129 in `test_M4_phase1_dimension_lift.jl`
4 testsets / 4 GATEs; 469 in `test_M4_phase1_kh_eigenmode.jl` 7
testsets / 7 GATEs).

**LOC delta:** ~50 lines in `src/eom.jl` (closed-loop coupling +
1/α_a² normalization fix), ~5 lines in `src/newton_step_HG.jl`
(c_back keyword threading), ~5 lines in `experiments/D1_KH_growth_rate.jl`
(c_back driver knob).

**Headline plot:** `reference/figs/M4_phase1_kh_eigenmode.png` — 4-panel
CairoMakie figure comparing c_back=0 vs c_back=1 at level 4/5.

**Reference:** `reference/notes_M4_phase1_closed_loop_beta.md`.

### M4 Phase 2 (3D KH lift) handoff items

  1. **3D H_back form**: generalize 2D H_back to 3D SO(3) Berry
     block with three pairs (1,2), (1,3), (2,3); each pair has its
     own G̃_{ab} and α_a·β_{ab}·β_a contribution.

  2. **3D KH IC factory**: lift `tier_d_kh_ic_full` per the M3-7e
     prep pattern.

  3. **3D D.1 falsifier driver**: extend
     `experiments/D1_KH_growth_rate.jl` to 3D (or write a parallel
     `D1_3D_KH_growth_rate.jl`).

  4. **2D ⊂ 3D selectivity**: a 3D KH IC with no z-axis dependence
     should reproduce 2D D.1 verdicts byte-equal.

### Open questions for M4 Phase 1c (if revisited)

  - Does the symplectic-natural F^β_off form (replacing β̇_off with
    pure α̇/θ̇_R constraint) activate the eigenmode? Requires a
    Newton-system regularization for the rank-6 Casimir kernel.
  - Does the antisymmetric H_back^anti form
    `c · G̃ · (α_2·β_12·β_1 - α_1·β_21·β_2)/2` produce different
    spectral structure? Initial test showed it broke GATE 3 of
    Phase 1a's test (symmetric drive identity); needs careful
    sign-bookkeeping.
  - Does a higher-order Hamiltonian (quartic in β) produce the
    eigenmode? This is the M4 Phase 8 (Bernstein reconstruction)
    territory.

---

*M4-1 closed in HONEST_FALSIFICATION mode on 2026-04-26: the
closed-loop symplectic coupling extends M3-6 Phase 1a but does not
activate the Drazin-Reid eigenmode. The methods paper's broad-band
acceptance of c_off ∈ [0.5, 2.0] is preserved; the tighter
[0.8, 1.2] aspiration band is not achieved. M4-2 (per-species
momentum) and M4-3 (3D Tier-D) are unblocked; M4-3a (3D KH) will
revisit the eigenmode question with the same closed-loop form lifted
to 3D, where additional Berry-block degrees of freedom may activate
the eigenvalue structure.*
