# Milestone 4 — Status

> **Created (2026-04-26):** at M4 entry, immediately after M3 close
> (`reference/MILESTONE_3_FINAL.md`). Tracks M4-as-shipped phase by
> phase against `reference/notes_M4_plan.md`.

## M4 phase ledger

| Phase | Status | Test delta | Notes |
|---|---|---:|---|
| M4-1: closed-loop β_off ↔ β_a coupling | **CLOSED** (HONEST_FALSIFICATION) | +598 | D.1 KH: closed-loop preserves regression but does not activate eigenmode |
| M4-2: 3D D.1 KH falsifier (lift of M4-1) | **CLOSED** (HONEST_FALSIFICATION_LIFTED) | +1187 | 2D kinematic-only finding generalizes to 3D; c_off ≈ 1.57 at L=3 |
| M4-3: per-species momentum (D.7 follow-up) | **CLOSED** (HONEST_PARTIAL) | +1708 | substrate + kernel + IC factory in place; τ_drag regimes differentiate with u_dust_offset bias; static gas equilibrium leaves regimes identical without offset |
| M4-4: 3D Tier-D headlines (D.7, D.10) | not started | — | |
| M4-5: full Metal/CUDA port | not started (HG `Backend` blocker) | — | |
| M4-6: MPI scaling | not started | — | |
| M4-7: E.4 cosmological + ColDICE | not started | — | |
| M4-8: D.6 / D.8 / D.9 two-fluid 2D | not started | — | |
| M4-9: Bernstein reconstruction | not started | — | |
| M4-10: methods paper resubmission | not started | — | |

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

## M4-2 close (2026-04-26)

**Headline scientific finding (HONEST_FALSIFICATION_LIFTED):** The
M4 Phase 1 2D kinematic-only finding **generalizes to 3D**. Lifting
the closed-loop β_off ↔ β_a coupling to 3D — three off-diagonal
Cholesky pairs (1,2), (1,3), (2,3) with their own H_rot^off and
H_back per pair — produces the same linear-in-t (forced) growth
rather than exp-in-t (eigenmode self-amplification) at the level-3
3D KH falsifier (8³ = 512 cells).

**3D residual structure:**
- 21 dof per leaf cell (15 base M3-7c + 6 off-diagonal `β_{ab}` slots).
- Six cross-axis velocity-gradient pairs: (∂_2 u_1, ∂_1 u_2),
  (∂_3 u_1, ∂_1 u_3), (∂_3 u_2, ∂_2 u_3).
- Three H_rot^off pair contributions: H^off_{12}, H^off_{13},
  H^off_{23} per cell.
- Closed-loop H_back per pair: `H_back^(ab) = c_back · G̃_{ab} ·
  (α_b·β_{ab}·β_b + α_a·β_{ba}·β_a) / 2`.
- Six F^θ_{ab} drives via per-pair vorticity W_{ab} · F_off^(ab).

**Falsifier verdict at level 3 (8³ = 512 cells, c_back=1):**
- γ_DR = 3.333 (U/(2w) with U=1, w=0.15).
- γ_measured = 5.24 ⇒ c_off ≈ 1.57 (broad band [0.5, 2.0] PASSED;
  aspiration [0.8, 1.2] NOT ACHIEVED).
- ssr_lin = 1.4e-5 ≪ ssr_exp = 0.037 ⇒ linear-in-t fits ~2700×
  better than exp-in-t.
- Wall-time per step: 3.41 s at level 3 (30 steps in 102 s wall).
- n_negative_jacobian = 0 throughout the trajectory (6-component
  β cone realizability holds).

**Bit-exact regression preservation:**
At β_off = 0 IC + axis-aligned u (the M3-7c regression configuration),
the 21-dof residual reduces byte-equal to the M3-7c 15-dof Berry
residual on the first 15 slots; rows 16..21 reduce to trivial
kinematic drives. Verified by GATE 2 of `test_M4_phase2_3d_kh_falsifier.jl`.

**Test delta:** +1187 asserts (8 GATEs in
`test_M4_phase2_3d_kh_falsifier.jl`).

**LOC delta:**
- `src/eom.jl`: +468 lines (`pack_state_3d_kh`, `unpack_state_3d_kh!`,
  `build_residual_aux_3D_kh`, `cholesky_el_residual_3D_berry_kh!` +
  allocating wrapper).
- `src/newton_step_HG.jl`: +110 lines (`det_step_3d_berry_kh_HG!`).
- `src/setups_2d.jl`: +199 lines (`allocate_cholesky_3d_kh_fields`,
  `tier_d_kh_3d_ic_full`).
- `src/dfmm.jl`: +9 lines (M4 Phase 2 export block).

**Headline plot:** `reference/figs/M4_phase2_3d_kh_falsifier.png` —
4-panel figure (log|A_rms(t)| with linear & exp fit overlays; 3D
γ_measured/γ_DR vs 2D Phase 1 reference; 6-component β cone diagnostics;
per-axis γ_a max/min trajectories).

**3D ⊂ 2D dimension-lift property (verified by GATE 8):**
With u_3 = 0 in the IC, ∂_3 u_a = 0 for any a and the (1, 3) and
(2, 3) pair velocity gradients vanish. The (1, 3)/(2, 3) off-diag β
slots stay at zero throughout the trajectory; the dominant (1, 2)
pair tilt mode is the only active growth mode. The 2D KH dynamics
embed in the 3D path byte-equally on the z-symmetric sub-slice.

**Reference:** `reference/notes_M4_phase2_3d_kh_falsifier.md`.

### M4 Phase 3 (per-species momentum for D.7) handoff items

  1. **Per-species momentum on 3D field set**: extend
     `tier_d_dust_trap_ic_full` (2D) to 3D Taylor-Green / ABC flow
     with multi-species `TracerMeshHG3D[:gas, :dust]`.
  2. **Drag relaxation kernel**: `∂_t (ρ_k u_k) = -ρ_k (u_k - u_gas)
     / τ_drag(size)` per species per axis.
  3. **2D ⊂ 3D selectivity**: a 3D dust-trap IC with z-symmetric
     base flow should reproduce the 2D D.7 verdicts byte-equal on
     axes 1, 2.

### Open questions for M4 Phase 2c (if revisited)

  - Does the 3D path's additional Berry block (axis-3) admit a
    higher-order H_back^cubic that activates the eigenmode without
    breaking 3D ⊂ 2D parity?
  - Does the antisymmetric H_back^anti variant (tested briefly in 2D
    Phase 1, broke GATE 3) extend cleanly to 3D's six-component
    sector?
  - Bernstein reconstruction (M4-9) is the natural place to resolve
    the eigenmode question — it lifts the per-cell linearised
    Rayleigh equation as a substrate-level extension rather than a
    Hamiltonian addition.

---

*M4-2 closed in HONEST_FALSIFICATION_LIFTED mode on 2026-04-26:
the 2D kinematic-only finding from M4 Phase 1 generalizes to 3D.
The 21-dof 3D KH-active residual closes the symplectic structure
across all three Berry pairs and preserves bit-exact regression at
β_off = 0; the falsifier verdict at level 3 confirms c_off ≈ 1.57
in the broad band, with linear-in-t fitting ~2700× better than
exp-in-t. The 6-component β cone realizability holds (n_negative_jacobian
= 0). The methods paper §10.5 D.1 broad-band claim is preserved in
3D as in 2D; the tighter aspiration band remains unmet, and the
Drazin-Reid eigenmode requires a different physics extension
(higher-order Hamiltonian, per-cell Rayleigh reconstruction, or
Bernstein-order substrate lift). M4 Phase 3 (per-species momentum
for D.7) is unblocked.*

## M4-3 close (2026-04-26)

**Headline scientific finding (HONEST_PARTIAL):** The
`PerSpeciesMomentumHG2D` opt-in extension of `TracerMeshHG2D`
adds per-species (u_x, u_y) per cell + per-species drag
relaxation timescale `τ_drag` + per-species Lagrangian position
offsets. The substrate, kernel, and IC factory all work
correctly:

  - **Drag relaxation kernel** `drag_relax_per_species!` —
    exponential Stokes-drag toward gas; verified at three limits
    (τ→0 passive scalar, τ→∞ decoupled, intermediate analytic
    formula `1 − exp(−dt/τ)` at machine precision).
  - **Position kernel** `advance_positions_per_species!` —
    kinematic `dx ← dx + dt·u`; verified bit-exact.
  - **Mass-conservative remap** `accumulate_species_to_cells!` —
    `Σ new_c == Σ source_c` to ≤ 1e-10.
  - **4-component realizability** under per-species momentum —
    `n_negative_jacobian = 0` throughout L=3 T_factor=0.1
    trajectory.
  - **Bit-exact regression contract** — at zero per-species
    momentum (no `PerSpeciesMomentumHG2D` constructed), the M3-6
    Phase 4 path is byte-equal to the pre-M4-3 codebase.

**Falsifier verdict (PARTIAL):** under the M3-6 Phase 4 Taylor-
Green vortex IC, the gas equilibrium is **static** under
`det_step_2d_berry_HG!` (the cold-limit Cholesky-sector residual
has no time evolution for incompressible ∇·u=0 + uniform pressure
ICs). With co-moving dust IC, all τ_drag regimes integrate the
same constant velocity field and produce identical drift
trajectories. The driver provides a `u_dust_offset` knob that
introduces a controlled IC bias; with `u_dust_offset = 0.5` the
three regimes differentiate as expected:

  | τ_drag | u_dust_speed_mean[end] | dx_dust_max[end] |
  |---:|---:|---:|
  | 1e-6 (tight) | 0.683 | 0.087 |
  | 0.1 (intermediate) | 0.697 | 0.117 |
  | 1e6 (decoupled) | 0.794 | 0.136 |

The literal D.7 centrifugal-accumulation prediction in the strict
sense (driven by gas time-evolution + size-dependent inertia)
requires either:

  1. **Non-stationary base flow** (e.g., decaying Taylor-Green
     under M3-7d viscosity, or a forced KH with explicit gas time
     dependence) — activates differential drag on top of the
     kinematic-only kernel.
  2. **Explicit centrifugal-force computation** in the drag kernel
     (`(u_k · ∇)u_gas` finite-difference term) — produces inward
     spiral drift for decoupled dust in a steady vortex.

Both are deferred to **M4 Phase 4 or beyond**.

**Test delta:** +1708 asserts (1 new test file
`test_M4_phase3_per_species_momentum.jl`, 8 GATEs / 8 testsets).

**LOC delta:**
- `src/newton_step_HG_M3_2.jl`: +325 LOC (struct + 4 kernels +
  diagnostics).
- `src/setups_2d.jl`: +99 LOC (`tier_d_dust_trap_per_species_ic_full`).
- `src/dfmm.jl`: +21 LOC re-exports.
- `experiments/D7_dust_traps.jl`: +517 LOC (driver + sweep + plot).
- `test/test_M4_phase3_per_species_momentum.jl`: NEW, ~310 LOC.
- `test/runtests.jl`: APPEND-ONLY new Phase M4-3 testset block.

**Headline plot:** `reference/figs/M4_phase3_dust_accumulation.png` —
4-panel figure (gas |u| vortex map; remapped dust concentration at
intermediate τ; peak/mean (remapped) vs t at three τ values; max
|dx_dust| Lagrangian drift magnitude vs t).

**Reference:** `reference/notes_M4_phase3_per_species_momentum.md`.

### M4 Phase 4 (Metal port post-HG-Backend) handoff items

  1. **Centrifugal-force kernel** (`apply_centrifugal_drift!`) —
     adds `(u_k · ∇)u_gas` finite-difference term to the drag
     relaxation. Activates inward spiral drift for decoupled dust
     in steady vortices; reproduces the literal D.7 prediction.
  2. **Two-way momentum exchange** — extends the gas Cholesky-
     sector residual with the back-reaction `Σ_k≠gas ρ_k (u_k −
     u_gas) / τ_drag_k` term.
  3. **3D per-species momentum** (`PerSpeciesMomentumHG3D` +
     3D dust-trap IC factory + `run_D7_dust_traps_3d_per_species`)
     — closes the M4-2 handoff for 3D D.7.
  4. **Refine/coarsen listener** for `PerSpeciesMomentumHG2D`
     (mirrors `register_tracers_on_refine_2d!` for the per-species
     `u` and `dx` arrays).
  5. **HG `Backend`-parameterized `PolynomialFieldSet`** (M3-8c
     blocker) — required for the per-leaf residual GPU port.
  6. **Non-stationary IC** — decaying Taylor-Green or forced KH
     base flow that activates differential drag.

---

*M4-3 closed in HONEST_PARTIAL mode on 2026-04-26: the substrate,
kernel, and IC factory for per-species momentum coupling all land
correctly with full bit-exact regression preservation and 1708
test asserts. The literal D.7 centrifugal-accumulation prediction
is not reproduced in the strict-static-gas regime because the cold-
limit Cholesky-sector residual leaves the Taylor-Green vortex
equilibrium static — all τ_drag regimes integrate the same
constant velocity field. The driver provides a `u_dust_offset`
knob that does differentiate regimes (verified by GATE 6); the
full physics requires either a non-stationary base flow or
explicit centrifugal-force computation, both deferred to M4 Phase
4. M4 Phase 4 (3D D.7, two-way momentum exchange, Metal port post-
HG-Backend) is unblocked.*
