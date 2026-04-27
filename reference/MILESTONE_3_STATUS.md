# Milestone 3 — Status synthesis (CLOSED — through M3-6 Phase 0)

**Date:** 2026-04-27 (combined close); M3-6 Phase 0 closed 2026-04-26.

**Repo state:** M3-3 closed at `077d6e4` (M3-3e-5 drop cache_mesh).
M3-4 lands in two phases: Phase 1 at `f364b4a` (periodic-x coordinate
wrap on 2D EL residual) and Phase 2 (IC bridge + C.1 / C.2 / C.3
acceptance drivers). M3-5 closed on `m3-5-bayesian-remap` with HG
`compute_overlap` + `polynomial_remap` wiring + IntExact audit harness.
**M3-6 Phase 0** closed on `m3-6-phase-0-offdiag-beta`: re-activates the
off-diagonal Cholesky pair `β_12, β_21` in the 2D residual, prerequisite
for the M3-6 Phase 1 D.1 KH falsifier driver.
**~19687 + 1 deferred tests** pass byte-equal across the 16 phase
blocks. The 2D scientific phase is complete through Tier-C
consistency + off-diagonal β reactivation; the 1D `cache_mesh::Mesh1D`
shim is retired in full; the 1D ⊂ 2D consistency gates fire across
C.1 / C.2 / C.3 IC families; Bayesian L↔E remap is wired with
conservation regression. M3-6 Phase 1 (D.1 KH) is unblocked.

**Per methods paper §10.7:** "Milestone 3: 2D principal-axis-decomposed
Cholesky integrator with Berry-connection coupling, on the
HierarchicalGrids substrate." Numbers hold; M3-5 adds the Bayesian
L↔E remap substrate per §6 / §6.6.

## Phase-by-phase completion table

| # | Item | Sub-phase | Tests | Headline result |
|---|---|---|---:|---|
| **M3-prep** | Berry connection 2D + 3D + off-diag, paper §5/§6 revisions | done | 181 | 7 SymPy CHECKs reproduced numerically; rel-err ≤ 1e-7 |
| **M3-0** | M1 Phase-1 ported onto HG `SimplicialMesh{1, T}` + `PolynomialFieldSet` via `cache_mesh::Mesh1D` shim | done | ~140 | Bit-exact 0.0 parity to M1 |
| **M3-1** | Phases 2/5/5b on HG (cache_mesh shim) | done | ~280 | Bit-exact 0.0 parity to M1 |
| **M3-2** | Phase 7/8/11 + M2-1/M2-3 on HG (cache_mesh shim) | done | 542 | Bit-exact 0.0 parity to M1 |
| **M3-2b** | HG-feature swap-in (Swap 1, 5, 6, 8 landed; 2/3 deferred to M3-3d) | done | ~120 | Periodic BC, IC factories, sparsity, BCKind framework |
| **M3-3a** | 2D field set + per-axis Cholesky decomposition driver | done | ~100 | `cholesky_DD.jl`, `setups_2d.jl`, halo smoke |
| **M3-3b** | Native HG-side **2D** EL residual (no Berry, θ_R fixed) | done | ~80 | Dimension-lift gate at 0.0 absolute |
| **M3-3c** | Berry coupling + θ_R Newton unknown | done | ~70 | All 7 SymPy CHECKs numeric; iso-pullback ε² |
| **M3-3d** | Per-axis γ + AMR/realizability per-axis (closes M3-2b Swaps 2+3 for 2D) | done | ~80 | Per-axis selectivity verified; HierarchicalMesh{2} AMR |
| **M3-3e-1** | Native `det_step_HG!` (deterministic Newton retire) | done | 1344 | Bit-exact 0.0; ~1.7× speedup at N=80 |
| **M3-3e-2** | Native `inject_vg_noise_HG!` + `det_run_stochastic_HG!` (RNG byte-equal) | done | 788 | Bit-exact 0.0 over K=10 stochastic steps |
| **M3-3e-3** | Native AMR refine/coarsen + standalone `TracerMeshHG` storage | done | 5784 | Bit-exact 0.0 across 31 AMR events with 3 tracers |
| **M3-3e-4** | Native `realizability_project_HG!` | done | 708 | Bit-exact 0.0 + ProjectionStats parity |
| **M3-3e-5** | Drop `cache_mesh::Mesh1D` field; close M3-3 | done | 0 (no new) | Field dropped; 13375+1 byte-equal regression |
| **M3-4 P1** | Periodic-x coordinate wrap on 2D EL residual | done | 46 | Translation equivariance; closes M3-3c handoff |
| **M3-4 P2 (a)** | Tier-C IC bridge + primitive recovery | done | 2606 | `(ρ, u, P)` → `(α, β, s, …)` cold-limit isotropic IC |
| **M3-4 P2 (b)** | C.1 1D-symmetric 2D Sod driver | done | 590 | y-independence ≤ 1e-12 per step |
| **M3-4 P2 (c)** | C.2 2D cold sinusoid driver | done | 11 | std(γ_1)/std(γ_2) > 1e10 for k=(1,0) |
| **M3-4 P2 (d)** | C.3 2D plane wave driver | done | 2583 | Rotational invariance + bounded mode amplitude |
| **M3-5** | Bayesian L↔E remap (`compute_overlap` + `polynomial_remap_l_to_e!`/`_e_to_l!` wired via `BayesianRemapState`); IntExact audit; Liouville monotone-necessary diagnostic | done | +86 | 9/9 IntExact battery passes; mass conserved 0..6.7e-16 over 5 cycles; partition-of-unity to 1e-12 |
| **M3-6 Phase 0** | Off-diagonal `β_12, β_21` re-activated in `DetField2D` + 2D residual; 11-dof Newton system; trivial-drive new rows preserve M3-3c byte-equal at β_12=β_21=0 | done | +390 | 9 SymPy CHECKs reproduced numerically; §Dimension-lift gate at 0.0 absolute |

## Test summary

| Block | Tests |
|---|---:|
| M1 (Phase 1-7 + 5b deterministic) | 305 |
| M1 (Phase 8 stochastic) | 140 |
| M1 (Phase 11 tracer) | 21 |
| M2 (M2-1 AMR + M2-2 multi-tracer + M2-3 realizability) | 243 |
| Cross-phase smoke + Track B/C/D + regression | 1335 |
| M3-prep (Berry stencils + Tier-C IC factories) | ~200 |
| M3-0/1/2 (HG ports, native HG-side as of M3-3e) | ~960 |
| M3-2b (HG swaps 1/5/6/8) | ~120 |
| M3-3a/b/c/d (2D native) | ~330 |
| M3-3e-1/2/3/4 (native-vs-cache cross-check tests) | 8624 |
| M3-5 (Bayesian L↔E remap + Liouville + IntExact audit) | 86 |
| M3-4 Phase 1 (periodic-x wrap on 2D EL residual) | 46 |
| M3-4 Phase 2 (a) IC bridge + primitive recovery | 2606 |
| M3-4 Phase 2 (b) C.1 1D-symmetric Sod driver | 590 |
| M3-4 Phase 2 (c) C.2 cold sinusoid driver | 11 |
| M3-4 Phase 2 (d) C.3 plane wave driver | 2583 |
| M3-6 Phase 0 (off-diagonal β reactivation) | 390 |
| **Total** | **~19687 + 1 deferred** |

## M3-3 headline scientific findings

### 2D Berry coupling lands (M3-3c)

The 7 verified Berry checks from the SymPy scripts
(`scripts/verify_berry_connection.py`) reproduce numerically in
the implementation:

- **CHECK 1 closedness:** Newton Jacobian symmetric in
  `(α_a, β_a, θ_R)` block per cyclic-sum identity; verified.
- **CHECK 2 per-axis match:** at `θ̇_R = 0`, the 2D residual reduces
  to two decoupled per-axis 1D residuals at 0.0.
- **CHECK 3 iso-pullback:** at `α_1 = α_2, β_1 = β_2`, Berry block
  vanishes; ε-expansion confirms `O(ε²)` leading order.
- **CHECK 4 H_rot solvability:** the closed-form `∂H_rot/∂θ_R`
  expression from `notes_M3_phase0_berry_connection.md` §6.6
  satisfies `dH ⊥ v_ker` at sampled `(α, β, γ)` points; rel-err ≤ 1e-12.
- **CHECK 5-7 numerical values:** verified against the 5 reference
  points hard-coded in `test_M3_prep_berry_stencil.jl`.

### Per-axis γ selectivity (M3-3d)

In a 1D-symmetric collapsing-axis cold-sinusoid configuration:
`γ_1 → 1e-5` (collapsing axis), `γ_2 → O(1)` (trivial axis); the
action-error indicator fires only along the collapsing axis (per-axis
selectivity verified via `test_M3_3d_selectivity.jl`).

### Native HG-side 2D residual (M3-3b/c)

First native HG-side EL residual in dfmm. Operates on
`HierarchicalMesh{2}` + `PolynomialFieldSet` directly via
`HaloView` + `face_neighbors_with_bcs`. The dimension-lift gate
holds at 0.0 absolute on a 1D-symmetric configuration — i.e., the
2D code reproduces M1's 1D code exactly when the y-direction is
trivial.

## M3-3e status (CLOSED 2026-04-27)

**Goal:** retire the `cache_mesh::Mesh1D` shim from the 1D HG path,
reaching a fully native HG-driven 1D code path with bit-exact 0.0
parity to all currently-passing tests.

**Outcome:** *closed*. All five sub-phases landed bit-exact byte-equal.
The `DetMeshHG` no longer carries a `Mesh1D` snapshot; the 1D path
runs entirely on `fields::PolynomialFieldSet` + `Δm` + `p_half` +
`bc_spec`/`inflow_state`/`outflow_state` storage.

**Sub-phase ledger:**

| Sub-phase | Commit | Tests Δ | Headline |
|---|---|---:|---|
| M3-3e-1 (native `det_step_HG!`) | `4aaf5bb` | +1344 | Newton path native; ~1.7× speedup at N=80 |
| M3-3e-2 (native VG injection + stoch driver) | `79842c0` | +788 | RNG byte-equal at K=10 + reanchor + τ + q |
| M3-3e-3 (native AMR + `TracerMeshHG`) | `d3054ea` | +5784 | 31 AMR events bit-exact with 3 tracers |
| M3-3e-4 (native realizability projection) | `de8986d` | +708 | ProjectionStats + state byte-equal |
| M3-3e-5 (drop cache_mesh field; close M3-3) | *(this commit)* | 0 | Field gone; 13375+1 regression byte-equal |

**Final test count:** 13375 + 1 deferred (= 9384 added across
M3-3e-1/2/3/4 cross-check tests + the M3-3a/b/c/d 2D-native blocks
not in the original M3-2 baseline).

**LOC delta in M3-3e effort (production code only):**

| File | Net LOC |
|---|---:|
| `src/newton_step_HG.jl` | +540 (native Newton path) − 30 (sync helpers) |
| `src/newton_step_HG_M3_2.jl` | +560 (native VG + tracers + projection) − 100 (delegations) |
| `src/action_amr_helpers.jl` | +400 (native AMR primitives) − 80 (rebuild_HG_from_cache + helpers) |
| **M3-3e-5 specific (cache_mesh drop):** | |
|   `src/newton_step_HG.jl` | -75 (field, sync_*_HG!, cache_mesh delegation in diagnostics) +75 (`mesh1d_from_HG` helper + native `total_*_HG`/`segment_*_HG`) |
|   `src/action_amr_helpers.jl` | -100 (`_resize_cache_mesh_HG!` + `rebuild_HG_from_cache!`) |

**Wall-time aggregate (N=80, periodic, deterministic step):**

| Path | ms / step |
|---|---:|
| Original M1 cache_mesh-shim baseline (pre-M3-3e-1) | ~16 |
| M3-3e-5 native (this commit) | ~8.2 |

**Memory footprint reduction:** the dropped `cache_mesh::Mesh1D` was
≈80 bytes/cell (segment array `Vector{Segment{T,DetField{T}}}` plus
`p_half`); on an N=80 mesh ≈6.4 KB; on an N=4096 AMR-driven mesh
≈320 KB. The savings scale linearly in N.

**See `reference/notes_M3_3e_5_cache_mesh_dropped.md`** for the close
report and `reference/notes_M3_3e_1/2/3/4_*.md` for the per-sub-phase
status notes (each with its own bit-exact gate breakdown).

## Open architectural questions — status update vs M2

| # | M2 status | M3 update | Net status |
|---|---|---|---|
| **#1** t¹ secular drift | open | unchanged (deferred to M3 prep, not addressed) | Open |
| **#2** Sod L∞ ~10-20% | open | unchanged (out of M3-3 scope) | Open |
| **#3** Stochastic 3-λ mismatch | open with reframing | unchanged (deferred to M4) | Open |
| **#4** Long-time stochastic instability | resolved by M2-3 | (carries forward; 2D Phase-8 not yet wired in M3-3) | Resolved |
| **#5** cache_mesh::Mesh1D retirement | open (M3-2 handoff) | M3-3e-1/2/3/4/5 all landed; field dropped; native 1D path | **Closed** |

## Pre-Milestone-4 readiness

**Strong:**
- 2D scientific physics (Berry coupling, per-axis γ, per-axis AMR /
  realizability) all verified.
- HG substrate (HaloView + cell_adjacency_sparsity + BCKind framework)
  consumed natively in 2D.
- Dimension-lift gate (1D ⊂ 2D) holds at 0.0 absolute — confirms the
  2D math reduces correctly.
- 1D HG path now native (M3-3e closed): `cache_mesh::Mesh1D` shim
  retired across all five sectors (deterministic Newton, stochastic
  injection, AMR + tracers, realizability projection, wrapper
  diagnostics).

**Unblocked for M4:**
- Tier D (KH instability, pancake collapse, wave-pool spectra) on the
  full 2D substrate.
- M3-6 Phase 1 (D.1 KH falsifier driver) — Phase 0 has scaffolded the
  11-dof residual; Phase 1 plumbs the off-diagonal strain coupling.
- M3-5 higher-order Bernstein per-cell reconstruction.

## Recommended next moves

1. **M3-4** — Tier C consistency tests **CLOSED 2026-04-27**: Phase 1
   periodic-x wrap (`f364b4a`); Phase 2 IC bridge + C.1/C.2/C.3 drivers
   (+5790 asserts). C.1 1D-vs-golden tolerance is at the loose
   variational-solver dispersion level (~10-20% L∞), documented as
   Open Issue #2 (deferred to higher-order Bernstein reconstruction).
   See `reference/notes_M3_4_tier_c_consistency.md`.
2. **M3-5** — Bayesian L↔E remap via `BayesianRemapState` +
   `bayesian_remap_l_to_e!` / `bayesian_remap_e_to_l!` /
   `remap_round_trip!`; IntExact audit gate; Liouville
   monotone-necessary diagnostic. **CLOSED 2026-04-26.** See
   `reference/notes_M3_5_bayesian_remap.md`.
3. **M3-6 Phase 0 — CLOSED 2026-04-26**: re-activated off-diagonal
   β_{12}, β_{21} in `DetField2D` and the 2D residual (11-dof Newton
   system). At β_{12}=β_{21}=0 IC the new rows trivialise and the
   residual reduces byte-equal to M3-3c. The 9 SymPy CHECKs from
   `scripts/verify_berry_connection_offdiag.py` are reproduced
   numerically at the residual / Jacobian level. Test delta: +390
   asserts. See `reference/notes_M3_6_phase0_offdiag_beta.md`.
4. **M3-6 Phase 1 / D.1 KH falsifier** — plumb the off-diagonal
   strain-coupling drive `H_rot^off ∝ G̃_12 · (α_1·β_21 + α_2·β_12)/2`
   into the F^β_a, F^β_12, F^β_21, F^θ_R rows; design the KH IC factory
   `tier_d_kh_ic`; calibrate the `c_off^2 ≈ 1/4` correction to the
   classical Drazin–Reid growth rate. M3-5's
   `det_run_with_remap_HG!` stub provides the integration site.
5. **M3-7** — 3D extension (the Berry stencils were verified in
   M3-prep as the M3-7 pre-flight gate).

## M3-4 close (2026-04-26)

**Phase 1** (`f364b4a`): periodic-x coordinate wrap on the 2D EL
residual. Closes the M3-3c handoff prerequisite. +46 asserts.

**Phase 2** (commits a, b, c, d on `m3-4-phase-2-tier-c-drivers`):

  • IC bridge maps primitive `(ρ, u_x, u_y, P)` cell-averages onto the
    M3-3 12-field Cholesky-sector state under a cold-limit, isotropic
    IC convention (α=1, β=0, θ_R=0, Pp=Q=0; s solved from EOS).
  • C.1 1D-symmetric 2D Sod: y-independence ≤ 1e-12 at every step;
    conservation gates pass; 1D-vs-golden gate captured at the loose
    tolerance documented in Open Issue #2.
  • C.2 2D cold sinusoid: std(γ_1)/std(γ_2) > 1e10 for k = (1, 0);
    qualitative 2D structure for k = (1, 1); conservation gates pass.
  • C.3 2D plane wave: u parallel to k̂ at IC; rotational invariance
    under π/2 to mesh-discretization tolerance; bounded mode amplitude
    under linear-acoustic Newton evolution at θ ∈ {0, π/8, π/4, π/2};
    conservation gates pass.

**M3-4 close: +5836 asserts (13375 + 1 → 19211 + 1).**

## M3-6 Phase 0 close (2026-04-26)

**Phase 0** (`m3-6-phase-0-offdiag-beta`): re-activates the off-diagonal
Cholesky pair `β_12, β_21` in the 2D residual after their omission in
M3-3a Q3. Prerequisite for the M3-6 Phase 1 D.1 KH falsifier driver
(the headline scientific test of M3-6).

  • `DetField2D{T}` extended with `betas_off::NTuple{2, T}` field;
    `n_dof_newton` 10 → 12. Backward-compat constructors default
    `betas_off = (0, 0)` so all pre-M3-6 IC factories continue to
    produce M3-3c-equivalent state byte-equally.
  • `allocate_cholesky_2d_fields` extended to 14 named scalar fields
    (12 prior + `:β_12, :β_21`).
  • `cholesky_el_residual_2D_berry!` extended to 11 dof per cell
    (was 9). Per-axis F^β_a rows pick up off-diag β coupling terms
    derived from rows of `Ω · X = -dH` with the corrected antisymmetric
    Ω entries (per `scripts/verify_berry_connection_offdiag.py`). New
    trivial-drive F^β_12, F^β_21 rows mirror F^θ_R's structure.
  • `det_step_2d_berry_HG!` Jacobian sparsity prototype is now
    `cell_adjacency ⊗ 11×11` (was 9×9).
  • §Dimension-lift gate at β_12=β_21=0: byte-equal to M3-3c (every
    M3-6 addition is multiplied by β_12, β_21, β̇_12, or β̇_21, which
    are pinned at zero by IC + trivial-drive). Verified across single-
    step + 100-step + 8×8 mesh + non-trivial active β_1 IC.
  • §Berry-offdiag CHECKs 1-9 reproduced numerically (FD probes of
    the residual Jacobian against the closed-form Hamilton-equation
    derivation; tolerance 1e-9).
  • §Realizability projection runs cleanly on the 14-named-field set;
    off-diag β unchanged across projection (per-cell-cone projection
    extension is M3-6 Phase 1's job).

**M3-6 Phase 0 close: +390 asserts (19297 + 1 → 19687 + 1).**

## Repo housekeeping

- M3-3a/b/c/d worktrees cleaned up.
- M3-3e branches (`m3-3e-1` through `m3-3e-5-drop-cache-mesh`)
  preserved as audit history.
- 9 commits ahead of origin/main since M3-3a launched
  (M3-3a/b/c/d/3e-pre-flight + 3e-1/2/3/4/5).
- All M3 named branches preserved as audit history.

---

*M3-3 closed with M3-3e-5. M3-4 closed in two phases (Phase 1
periodic-x wrap; Phase 2 IC bridge + C.1/C.2/C.3 drivers). The 1D
path is native; the 2D path is native; the Tier-C C.1 / C.2 / C.3
acceptance gates fire. 1D-path bit-exact 0.0 parity holds across
all 19211 + 1 currently passing tests. Methods paper §10.4 / §10.7
numbers all hold. Ready for M3-5 (in flight in another worktree).*
