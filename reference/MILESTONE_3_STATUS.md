# Milestone 3 — Status synthesis (in progress)

**Date:** 2026-04-26.

**Repo state:** main HEAD at `6a65411` (M3-3e pre-flight). **3991 + 1
deferred tests pass.** The 2D scientific phase (M3-3a/b/c/d) is
complete; the 1D cache_mesh shim retirement (M3-3e) is the open
final sub-phase before M3-3 can close.

**Per methods paper §10.7:** "Milestone 3: 2D principal-axis-decomposed
Cholesky integrator with Berry-connection coupling, on the
HierarchicalGrids substrate."

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
| **M3-3e** | Cache_mesh shim retirement (1D path) | **OPEN** | — | See §"M3-3e status" below |

## Test summary

| Block | Tests |
|---|---:|
| M1 (Phase 1-7 + 5b deterministic) | 305 |
| M1 (Phase 8 stochastic) | 140 |
| M1 (Phase 11 tracer) | 21 |
| M2 (M2-1 AMR + M2-2 multi-tracer + M2-3 realizability) | 243 |
| Cross-phase smoke + Track B/C/D + regression | 1335 |
| M3-prep (Berry stencils + Tier-C IC factories) | ~200 |
| M3-0/1/2 (HG ports, cache_mesh shim) | ~960 |
| M3-2b (HG swaps 1/5/6/8) | ~120 |
| M3-3a/b/c/d (2D native) | ~330 |
| **Total** | **3991 + 1 deferred** |

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

## M3-3e status (OPEN)

**Goal:** retire the `cache_mesh::Mesh1D` shim from the 1D HG path,
reaching a fully native HG-driven 1D code path with bit-exact 0.0
parity to all 3991 + 1 currently-passing tests.

**Status as of 2026-04-26:** investigation complete; scope mismatch
identified. The cache_mesh shim is genuinely load-bearing across
**five** code paths, not one. Retiring it requires sector-by-sector
lifts totaling ~1380 LOC of new HG-side code over ~5 days, not the
~2 days estimated in the M3-3 design note.

**See `reference/notes_M3_3e_cache_mesh_retirement.md`** for the
full investigation, the proposed sub-phase decomposition (M3-3e-1
through M3-3e-5), and the per-sector LOC + risk breakdown.

**Recommended decomposition:**

- **M3-3e-1** (1 day): native deterministic Newton (`det_step_HG!`)
  retirement, including post-Newton BGK / Phase 5b q-dissipation /
  Phase 7 inflow-outflow pinning lifts. Bit-exact gate: M3-1/M3-2
  deterministic tests.
- **M3-3e-2** (1 day): native Phase-8 stochastic injection
  (`inject_vg_noise_HG!`, `det_run_stochastic_HG!`) with RNG-bit-equal
  sequencing. Bit-exact gate: Phase-8 tests.
- **M3-3e-3** (1 day): native AMR refine/coarsen + `TracerMeshHG`
  retirement (Option B: hand-rolled 1D-Lagrangian primitives on
  HG storage). Bit-exact gate: M2-1 + Phase-11 + M2-2 tests.
- **M3-3e-4** (0.5 day): native realizability projection.
  Bit-exact gate: M2-3 tests.
- **M3-3e-5** (0.5 day): drop the `cache_mesh` field; final 3991+1
  regression.

**Files unchanged in this session.** Branch
`m3-3e-cache-mesh-retire` open at `main` HEAD `6a65411` for the
next agent.

## Open architectural questions — status update vs M2

| # | M2 status | M3 update | Net status |
|---|---|---|---|
| **#1** t¹ secular drift | open | unchanged (deferred to M3 prep, not addressed) | Open |
| **#2** Sod L∞ ~10-20% | open | unchanged (out of M3-3 scope) | Open |
| **#3** Stochastic 3-λ mismatch | open with reframing | unchanged (deferred to M4) | Open |
| **#4** Long-time stochastic instability | resolved by M2-3 | (carries forward; 2D Phase-8 not yet wired in M3-3) | Resolved |
| **#5** cache_mesh::Mesh1D retirement | open (M3-2 handoff) | M3-3e investigation underway; sub-divided plan in flight | Open |

## Pre-Milestone-4 readiness

**Strong:**
- 2D scientific physics (Berry coupling, per-axis γ, per-axis AMR /
  realizability) all verified.
- HG substrate (HaloView + cell_adjacency_sparsity + BCKind framework)
  consumed natively in 2D.
- Dimension-lift gate (1D ⊂ 2D) holds at 0.0 absolute — confirms the
  2D math reduces correctly.

**Conditional:**
- M3-3e cache_mesh retirement is genuinely open. The 1D path still
  goes through the shim; the 2D path is native. Until M3-3e closes,
  there are two parallel storage stacks for the 1D fluid code.
- M3-4/M3-5 (Bayesian remap, higher-order Bernstein) layered work
  cannot start until M3-3 closes.

**Unblocked for M4 (if M3-3e closes):**
- Tier D (KH instability, pancake collapse, wave-pool spectra) on the
  full 2D substrate.
- Off-diagonal β_{12}, β_{21} sector activation (M3-6 / D.1 KH).

## Recommended next moves

1. **Execute M3-3e-1** — native deterministic Newton with full
   post-Newton operator splits. The `det_el_residual` already takes
   flat arrays; the lift is mechanical scalar-arithmetic translation.
   Bit-exact gate is automatic for a careful translation.
2. **Then M3-3e-2 through M3-3e-5** in order. Each sub-phase has a
   crisp bit-exact verification gate.
3. **Then M3-4** — higher-order Bernstein per-cell reconstruction on
   the 2D field set; per the methods paper §9.2.

## Repo housekeeping

- M3-3a/b/c/d worktrees cleaned up.
- M3-3e branch (`m3-3e-cache-mesh-retire`) open at `main` HEAD.
- 5 commits ahead of origin/main since M3-3a launched
  (M3-3a/b/c/d/3e-pre-flight).
- All M3 named branches preserved as audit history.

---

*M3-3 closes after M3-3e completes. Estimated remaining wall-time
~5 days for the next agent to execute the sub-phase plan in
`reference/notes_M3_3e_cache_mesh_retirement.md`.*
