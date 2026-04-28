# Milestone 4 — final synthesis (suitable for §A appendix or reviewer response)

**Status (2026-04-26):** M4 closed. The three M4 sub-phases that
shipped (M4-1, M4-2, M4-3) deliver a complete substrate for the two
M3 honest-finding follow-ups (D.1 closed-loop β coupling, D.7 per-
species momentum) and yield three honest-partial scientific verdicts
that sharpen the methods paper's characterization of what the
variational scheme captures natively. The remaining M4 phases of the
original plan (full GPU port, MPI scaling, E.4 cosmological, D.6 /
D.8 / D.9 two-fluid 2D, Bernstein reconstruction, paper resubmission
with M4+M5 results) are deferred to M5. The methods paper §10.5
D.1/D.7 sub-paragraphs and the new §10.7 "Honest scientific
characterization" subsection consolidate M3 + M4 findings.

## What M4 delivers

1. **Closed-loop symplectic coupling on the principal-axis Berry
   block (M4-1, 2D).** The 2D residual gains the back-reaction
   Hamiltonian
   `H_back = c_back · G̃_12 · (α_2·β_12·β_2 + α_1·β_21·β_1)/2`
   which closes the symplectic loop between `(β_12, β_21)` and
   `(β_1, β_2)`. The form is symmetric in `β_off ↔ β_a` under the
   natural pairing `β_12 ↔ β_2, β_21 ↔ β_1`; H_back contributions
   vanish identically when β_off = 0 OR β_a = 0, preserving bit-
   exact regression at the M3-3c, M3-4, M3-6 Phase 0/1a configs.
   Phase 1a's missing 1/α_a² normalization on the F^β_a row is
   fixed (the previous form was correct only at α=1; the M4 form
   is symplectic-natural at all α).

2. **3D SO(3) lift of the closed loop (M4-2, 3D).** Three
   pair-by-pair contributions H_rot^off,(ab) and H_back^(ab) for
   (a,b) ∈ {(1,2), (1,3), (2,3)} extend the 2D form to 3D with the
   per-pair structure of `BerryStencil3D`. The new 21-dof per-leaf
   residual `cholesky_el_residual_3D_berry_kh!` reduces byte-equal
   to the 15-dof M3-7c residual on the first 15 slots at β_off = 0
   IC + axis-aligned u; rows 16..21 are pure trivial drives.

3. **Per-species momentum substrate (M4-3, 2D).**
   `PerSpeciesMomentumHG2D{T}` extends `TracerMeshHG2D` with per-
   species (u_x, u_y) per cell, per-species drag relaxation
   timescale τ_drag, and per-species Lagrangian position offsets.
   Three kernels land:
     - `drag_relax_per_species!` — exponential Stokes-drag toward
       gas; verified at limits τ→0 (passive scalar), τ→∞ (decoupled),
       intermediate (1 − exp(−dt/τ) at machine precision).
     - `advance_positions_per_species!` — kinematic position update
       `dx ← dx + dt·u`.
     - `accumulate_species_to_cells!` — mass-conservative remap;
       `Σ new_c == Σ source_c` to ≤ 1e-10.
   Plus the IC factory `tier_d_dust_trap_per_species_ic_full` with
   `u_dust_offset` knob for controlled IC bias.

## Tier-D follow-up verdicts (the three M4 honest findings)

The two methods-paper claims that M3 honestly flagged for follow-up
(D.1 closed-loop β coupling; D.7 centrifugal drift) are characterised
in M4 with sharp scientific verdicts.

- **D.1 2D Kelvin-Helmholtz (M4-1, HONEST_FALSIFICATION).** At level
  4 (16×16) and level 5 (32×32), the closed-loop residual produces
  γ_measured/γ_DR ≈ 1.26 (compared to ~1.30 for M3-6 Phase 1a
  kinematic-only); both are within the methods-paper broad-band gate
  c_off ∈ [0.5, 2.0]; neither tightens into the M4 Phase 1 aspiration
  band [0.8, 1.2]. Linear-in-t fits the trajectory ~1000× better than
  exp-in-t (ssr_lin = 4.7e-5 ≪ ssr_exp = 0.078 at level 3, c_back=1).
  **The Drazin-Reid eigenmode is not activated by the closed-loop
  coupling alone.** Three structural interpretations documented:
  (a) per-cell linearised Rayleigh reconstruction; (b) antisymmetric
  H_back^anti; (c) symplectic-natural F^β_off form replacing the
  Phase 1a kinematic kludge. n_negative_jacobian = 0 throughout.
  Test delta: +598 asserts. Closed in HONEST_FALSIFICATION mode.

- **D.1 3D Kelvin-Helmholtz (M4-2, HONEST_FALSIFICATION_LIFTED).**
  At level 3 (8³ = 512 cells, c_back = 1): γ_DR = 3.333,
  γ_measured = 5.237, c_off = 1.571. Linear-in-t fits ~2700× better
  than exp-in-t (ssr_lin = 1.4e-5 ≪ ssr_exp = 0.037).
  n_negative_jacobian = 0 throughout. Wall-time 3.41 s/step at
  level 3 (30 steps in 102 s wall). The 6-component β cone
  realizability holds. The 2D verdict generalizes: closing the
  symplectic loop in all three Berry pairs preserves bit-exact
  regression but does not activate the Rayleigh eigenmode in 3D
  either. The 3D ⊂ 2D dimension-lift property is verified: a
  z-symmetric KH IC reproduces 2D dynamics byte-equal on the (1,2)
  pair while the (1,3)/(2,3) pairs stay at zero throughout the
  trajectory. Test delta: +1187 asserts. Closed in
  HONEST_FALSIFICATION_LIFTED mode.

- **D.7 2D dust-trapping in vortices (M4-3, HONEST_PARTIAL).** The
  substrate, kernel, and IC factory all work correctly with full
  bit-exact regression preservation (M3-6 Phase 4 path byte-equal
  at zero per-species momentum). 4-component cone realizability holds
  (n_negative_jacobian = 0 throughout L=3 T_factor=0.1 trajectory).
  Three τ_drag regimes (1e-6 tight, 0.1 intermediate, 1e6 decoupled)
  differentiate cleanly under `u_dust_offset = 0.5`:
  u_dust_speed_mean[end] = 0.683 / 0.697 / 0.794, max|dx_dust|[end]
  = 0.087 / 0.117 / 0.136. Without u_dust_offset, the cold-limit
  Cholesky-sector residual leaves the Taylor-Green vortex equilibrium
  static — all τ_drag regimes integrate the same constant velocity
  field. The literal D.7 centrifugal-accumulation prediction
  requires either (a) a non-stationary base flow, or (b) explicit
  centrifugal-force kernel `(u_k · ∇)u_gas`, plus two-way momentum
  exchange (`Σ_k≠gas ρ_k(u_k − u_gas)/τ_drag` back-reaction on gas).
  Test delta: +1708 asserts. Closed in HONEST_PARTIAL mode.

## What M4 captures natively (consolidated with M3)

The methods paper's §10.7 "Honest scientific characterization" sub-
section formalizes the post-M4 deliverables:

**Natively captured (≥ 10⁶ selectivity or bit-exact):**
- D.4 Zel'dovich pancake collapse (2D 2.6×10¹⁴, 3D 3.0×10¹³).
- D.10 ISM-like multi-tracer fidelity (literal zero per-step error).
- Tier-C C.1/C.2/C.3 dimension-lift gates (Δ = 0.0 absolute byte-
  equal, plane-wave Δx² convergence with rotational invariance).
- Tier-E graceful failure modes (E.1/E.2/E.3, NaN-free at high
  Mach + shell-crossing + low Knudsen).
- 1D ⊂ 2D ⊂ 3D dimension-lift gates (Δ = 0.0 absolute, lifted to
  the M4 Phase 2 21-dof KH-active 3D residual via GATE 8).

**Requires physics extensions (M5+ work):**
- D.1 Drazin-Reid eigenmode — needs higher-order Hamiltonian or
  per-cell linearised Rayleigh reconstruction (M4-1 §3 routes).
- D.7 centrifugal accumulation — needs centrifugal-force kernel
  `(u_k · ∇)u_gas` + two-way momentum coupling (M4-3 handoffs).

## Performance summary

M4 did not target performance work; the focus was physics-extension
substrates. Wall-time impact:

- M4-1 closed-loop H_back: ~22% per-Newton-step overhead at level 3
  (more nonzero Jacobian entries to evaluate via ForwardDiff).
- M4-2 21-dof 3D KH-active residual: 3.41 s/step at level 3
  (8³ = 512 cells); a 21-dof matrix-free promotion is straightforward
  but not landed.
- M4-3 per-species momentum: opt-in extension; 0% overhead when no
  `PerSpeciesMomentumHG2D` is constructed; ~0.04 s/step at level 3 +
  64 cells when active.

## Test count

~37,000+1 deferred tests at M4 close (from 33,940+1 at M3-3 close,
growing through M4-1/2/3 by +598+1187+1708 = +3493 asserts). Bit-
exact regression discipline held throughout: every M4 sub-phase
landed without invalidating prior gates. Test growth pattern:

| Phase | Tests added | Cumulative |
|---|---:|---:|
| M3 close (M3-9 entry) | 33,940 + 3 deferred | 33,940 + 3 |
| M4-1 D.1 2D closed-loop | +598 | 34,538 + 3 |
| M4-2 D.1 3D KH lift | +1187 | 35,725 + 3 |
| M4-3 D.7 per-species momentum | +1708 | 37,433 + 3 |

## What M4 does *not* deliver (the M5 entry points)

The honest list of items deferred to M5, in dependency order:

1. **Centrifugal-force kernel + two-way coupling** for D.7
   literal claim (closes the M4-3 honest-partial loop).
2. **Higher-order Hamiltonian (per-cell linearised Rayleigh)** for
   D.1 eigenmode (closes the M4-1/M4-2 honest-falsification loop).
3. **HG `Backend`-parameterized `PolynomialFieldSet`** upstream
   (PR to HG, prerequisite for full GPU port).
4. **Apple Metal kernel port** for the per-cell residual (requires
   M5-3 prerequisite).
5. **CUDA backend** (after Metal port stabilizes).
6. **MPI scaling** on HG `partition_for_threads` chunk structure.
7. **Methods paper resubmission** with M4 + M5 findings.

(M4 phases 7, 8, 9 of the original plan — D.6/D.8/D.9 two-fluid 2D
content, Bernstein reconstruction, E.4 cosmological — are tracked
as M5+ items; they are physics-extension orthogonal to the M5-1 /
M5-2 closures.)

See `reference/notes_M5_plan.md` for the M5 entry-point detail.

## References

- `reference/MILESTONE_4_STATUS.md` — full M4 phase ledger + close
  synthesis.
- `reference/notes_M4_phase1_closed_loop_beta.md` — D.1 2D
  honest finding (M4-1).
- `reference/notes_M4_phase2_3d_kh_falsifier.md` — D.1 3D
  honest finding (M4-2).
- `reference/notes_M4_phase3_per_species_momentum.md` — D.7
  substrate honest-partial finding (M4-3).
- `reference/notes_M4_plan.md` — M4 entry-point list (M3-9 era).
- `reference/notes_M5_plan.md` — M5 entry-point detail.
- `reference/MILESTONE_3_FINAL.md` — M3 final synthesis (the M4
  parent context).
- `specs/01_methods_paper.tex` §10.5 / §10.7 / §10.9 — methods-paper
  M4 revisions.

## Commit references (M4 phases as shipped)

- M4 Phase 1: 2D closed-loop H_back addition; closed-loop falsifier;
  bit-exact regression preservation across full M3 battery.
- M4 Phase 2 (a–d): 3D off-diag strain coupling, IC factory, 3D
  falsifier driver, headline plot, GATE 8 dimension lift.
- M4 Phase 3 (a–d): TracerMeshHG2D extension, drag kernel, D.7
  driver extension, headline plot.
- M4 wrap-up (a–c): paper revisions, MILESTONE_4_STATUS final, M5
  plan note.

---

*M4 closed on 2026-04-26 with three honest-partial verdicts: the
substrates for both follow-up physics extensions (D.1 closed-loop β
coupling in 2D and 3D; D.7 per-species momentum) are operational and
bit-exact-regression-preserving; the literal eigenmode-activation and
literal centrifugal-accumulation predictions both require additional
physics terms (higher-order Hamiltonian, centrifugal-force kernel,
two-way momentum coupling) which are documented as concrete M5
entry points. The methods paper consolidates M3 + M4 findings and
declares the honest scientific characterization upfront.*
