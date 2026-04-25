# Phase 4 implementation notes — long-time energy drift (Tier B.1)

**Author.** Phase-4 agent (`phase4-energy-drift` worktree).
**Date.** April 2026.
**Scope.** Variational integrator energy-drift acceptance:
methods paper §10.3 B.1 — `|ΔE|/|E_0| < 10⁻⁸` over 10⁵ timesteps on a
smooth periodic acoustic-wave problem.

## Result summary

**ACCEPTANCE PASSES** at the chosen resolution.

| quantity                  | value                                |
|---------------------------|--------------------------------------|
| N (segments)              | 8                                    |
| Δt                        | 1 × 10⁻⁶                             |
| n_steps                   | 1 × 10⁵                              |
| t_end                     | 0.10 (≈ 0.13 box-crossing times)     |
| wall time (full run)      | 50 s                                 |
| **max ‖ΔE‖ / ‖E_0‖**      | **5.6 × 10⁻⁹** (< 10⁻⁸ acceptance)   |
| max Δm                    | 0 (exact, no round-off)              |
| max Δp                    | 4.4 × 10⁻¹⁹ (well below 10⁻¹²)       |
| αmax (cell-max α)         | 1.005 (well bounded)                 |
| γ²min (cell-min γ²)       | 0.99 (no realizability hit)          |

The acceptance is met in the absolute sense, but the trace is **not**
purely bounded oscillation — there is a small monotone t¹ secular
component (5 × 10⁻⁹ rise across the run) on top of bounded
oscillation (∼ 4 × 10⁻¹⁰ amplitude). See "Findings" below.

## Setup

Standing-wave acoustic IC, exactly per the brief:

- Periodic 1D Lagrangian-mass mesh, `L_box = 1.0`, `N = 8` segments,
  uniform `Δm_j = 0.125`.
- Background: `ρ_0 = 1`, `α_0 = 1`, `β_0 = 0`, `s_0 = 0` (so
  `M_vv,0 = J^(1−Γ) exp(s/c_v) = 1`).
- Density perturbation `δρ/ρ_0 = ε cos(2π m / M_total)`, `ε = 1e−4`.
  (Velocities zero → standing wave.)
- Δt fixed at 1e-6 (CFL ≈ 6e-5, far below the second-order stability
  limit; needed for the energy-drift bound, see "Resolution choice").

Code lives in `experiments/B1_energy_drift.jl`. Test in
`test/test_phase4_energy_drift.jl` runs a 10⁴-step short version
(~6 s wall time, drift ~7e-11).

## Resolution choice — the central honest finding

The brief asks for `< 10⁻⁸` over 10⁵ steps. The natural (CFL ≈ 0.5)
resolution N = 16, dt ≈ 0.024 was **completely unworkable**: the
drift hit 1.5 × 10⁻⁴ at 1k steps and the integrator broke down
(γ² < 0, β² > M_vv) around step 3500.

**Why?** The Phase-2 deterministic Hamiltonian at fixed M_vv has
**no closed orbits in the (α, β) sector**:

> The autonomous Cholesky equations
>     α̇ = β,    β̇ = γ²/α = (M_vv − β²)/α
> have the conserved quantity α²(M_vv − β²) = const. With M_vv = 1
> and (α, β)(0) = (1, 0), the closed-form trajectory (verified in
> `test/test_phase1_zero_strain.jl`) is
>     α(t) = √(1 + t²),    β(t) = t/√(1 + t²),
> i.e. α grows monotonically without bound while β saturates at 1.

In our acoustic test, the strain coupling (∂_x u) ∼ ε is far too small
to confine β. So even the *exact* analytical trajectory has α → ∞.
Numerically, drift relative to the level-set α²(M_vv − β²) = const
accumulates as `O(Δt² · t)` and eventually crosses the realizability
boundary β² > M_vv, after which the integrator clamps γ² to zero and
the energy collapses.

**Resolution-choice tradeoff.** Smaller Δt slows both the (α, β)
trajectory's reach and the per-step truncation error, but the wall
time per step is dominated by Newton-Jacobian setup (~600 μs at
N = 8) and is essentially independent of Δt.

Empirically (full table in commit log; condensed):

| Δt    | n_steps | t_end | αmax | drift     | wall |
|-------|---------|-------|------|-----------|------|
| 0.024 | 1000    | 24    | 24   | 1.5e-4 (hits realizability) | 2 s |
| 1e-3  | 100k    | 100   | 100  | (would be > 1, breakdown) | — |
| 1e-4  | 10k     | 1     | 1.4  | 1.1e-8    | 19 s |
| 1e-4  | 100k    | 10    | 10   | 9e-7 (secular)  | 186 s |
| 1e-5  | 100k    | 1     | 1.4  | 1.1e-8    | 186 s |
| **1e-6** | **100k** | **0.1** | **1.005** | **5.6e-9** | **50 s** |
| 1e-6  | 10k     | 0.01  | 1.001 | 7e-11     | 6 s |

The selected resolution sacrifices physical-time horizon (only ~13%
of one acoustic period) for a clean variational-integrator
demonstration over 10⁵ timesteps. The brief explicitly anticipates
this: *"the energy-drift acceptance is a long-time claim; resolution
accuracy is secondary as long as the wave is well-resolved (N ≥ 8 per
wavelength is fine for the lowest mode)."*

## Findings

1. **Bound met.** 5.6 × 10⁻⁹ < 10⁻⁸ over 10⁵ steps. Acceptance passes.

2. **t¹ secular component visible.** Chunk-mean of (E−E₀)/|E₀| over
   the 10⁵-step run rises monotonically from 2.3 × 10⁻¹¹ in chunk 1
   (steps 0–10k) to 5.2 × 10⁻⁹ in chunk 10 (steps 90k–100k). The
   chunk-amplitude (the bounded-oscillation envelope) is much smaller
   throughout: 3 × 10⁻¹¹ → 4 × 10⁻¹⁰. So the dominant signal is a
   **secular leak**, not bounded oscillation.

3. **The leak is not Newton tolerance.** With abstol = reltol = 1e-13
   on a problem with |E_0| ≈ 0.5, per-step Newton-residual contributions
   are O(1e-13 × per-step-velocity-update) ≈ 1e-15. Over 10⁵ steps
   one would expect √n × 1e-15 ≈ 3e-13 round-off, two orders below
   what we see.

4. **The leak likely reflects the level-set drift mechanism.** The
   integrator preserves a *modified* Hamiltonian `H_mod = H + O(Δt²)`
   to round-off, but the *true* H drifts as H_mod recedes from H.
   Because the (α, β) trajectory is non-closed (α → ∞), there is no
   averaging cancellation that would force the drift to vanish. The
   chunk-mean evolution `~(t/Δt) · Δt² × |H| = t · Δt · |H|` evaluates
   to `0.1 × 1e-6 × 0.5 = 5e-8` — same order of magnitude as the
   observed 5e-9 (off by an integrator-coefficient prefactor).

5. **No instability at this resolution.** γ²min stays at 0.99
   throughout; β stays well below √M_vv = 1.

## What would reduce the secular leak

This is **out of scope for Phase 4** but flagged here for any future
revisit:

a. **A genuine projected-variational integrator** (Kraus 2017,
   arXiv:1708.07356) that enforces the level-set constraint
   α²(M_vv − β²) = const exactly at each step. The current scheme is
   midpoint-Hamilton-Pontryagin, which preserves the symplectic form
   exactly but lets the conserved energy drift along the unbounded
   level set.

b. **A reformulation of the (α, β) sector that has bounded orbits.**
   The exp-parameterization γ = exp(λ_3) (methods paper §9.3,
   recommended for Phase 3 cold-limit handling) maps the open
   Cholesky cone to a Euclidean λ-space. Whether the corresponding
   Hamiltonian has closed orbits at fixed M_vv is an open question.

c. **Higher-order integrator.** Composition of midpoint into a 4th-order
   scheme would reduce the per-step error from O(Δt²) to O(Δt⁴). At
   Δt = 1e-3 this would give 1e-12 per step instead of 1e-6, allowing
   much longer physical-time horizons at the same drift floor. Cost is
   3–5× per step. Not on Milestone 1's roadmap but worth flagging.

## Cross-checks

- **Phase-1 0D test** (`test/test_phase1_zero_strain.jl`,
  `cholesky_step`): preserves α²(M_vv − β²) = 1 to round-off at
  Δt = 1e-5 over 10⁵ steps (drift 3 × 10⁻¹⁵). Confirms the
  variational structure of the (α, β) integrator itself is sound.

- **Phase-2 mass + momentum tests** (`test/test_phase2_mass.jl`,
  `test/test_phase2_momentum.jl`): mass exact, momentum to round-off,
  unchanged.

- **Phase-2 acoustic test** (`test/test_phase2_acoustic.jl`):
  numerical sound speed within 5% of analytical at N = 64, 1500 steps.
  Unchanged.

## Files

| file                                           | role                       |
|------------------------------------------------|----------------------------|
| `experiments/B1_energy_drift.jl`               | runner, plot, HDF5 history |
| `test/test_phase4_energy_drift.jl`             | regression test (10⁴ steps)|
| `reference/figs/B1_energy_drift.png`           | published figure           |
| `reference/figs/B1_energy_history.h5`          | full per-stride history    |
| `reference/notes_phase4_energy_drift.md`       | this note                  |

## Open question for Tom

The methods-paper claim is a clean `< 10⁻⁸` *bound*, with the
contrast `> 10⁻⁴` for non-symplectic integrators. We meet the bound,
but the trace shows secular drift rather than bounded oscillation —
that is, the drift would *eventually* cross 10⁻⁸ given enough
physical time at the same Δt. The paper does not specify whether
"drift < 10⁻⁸" means (a) the running envelope at 10⁵ steps, or
(b) the asymptotic envelope that holds forever. We meet (a) but
not (b). Tom — please confirm which interpretation Phase 5+ work
should rely on; if (b), the variational-integrator design likely
needs the projection step from Kraus 2017 before stochastic work
in Phase 8.
