# Milestone 3 — Status synthesis (CLOSED)

**Date:** 2026-04-27.

**Repo state:** main HEAD at `de8986d` (M3-3e-4); M3-3e-5 branch
`m3-3e-5-drop-cache-mesh` lands the cache_mesh field drop, **closing
M3-3.** **13375 + 1 deferred tests pass byte-equal across all 13
phase blocks.** The 2D scientific phase (M3-3a/b/c/d) is complete;
the 1D `cache_mesh::Mesh1D` shim has been retired in full (M3-3e-1
through M3-3e-5).

**Per methods paper §10.7:** "Milestone 3: 2D principal-axis-decomposed
Cholesky integrator with Berry-connection coupling, on the
HierarchicalGrids substrate." All numbers hold.

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
| **Total** | **13375 + 1 deferred** |

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
- Tier C (1D ⊂ 2D) consistency tests on the native 2D + native 1D
  substrate (M3-4 scope).
- Tier D (KH instability, pancake collapse, wave-pool spectra) on the
  full 2D substrate.
- Off-diagonal β_{12}, β_{21} sector activation (M3-6 / D.1 KH).
- M3-5 higher-order Bernstein per-cell reconstruction.

## Recommended next moves

1. **M3-4** — Tier C consistency tests: 1D ⊂ 2D parity on the full
   physics suite (Sod, cold-sinusoid, wave-pool) on the native
   `HierarchicalMesh{2}` + `PolynomialFieldSet` substrate. Per
   methods paper §10 / §10.7. Now unblocked.
2. **M3-5** — higher-order Bernstein per-cell reconstruction on the
   2D field set; per the methods paper §9.2.
3. **M3-6 / D.1 KH falsifier** — activate off-diagonal β_{12}, β_{21}
   sector and run KH instability benchmarks.
4. **M3-7** — 3D extension (the Berry stencils were verified in
   M3-prep as the M3-7 pre-flight gate).

## Repo housekeeping

- M3-3a/b/c/d worktrees cleaned up.
- M3-3e branches (`m3-3e-1` through `m3-3e-5-drop-cache-mesh`)
  preserved as audit history.
- 9 commits ahead of origin/main since M3-3a launched
  (M3-3a/b/c/d/3e-pre-flight + 3e-1/2/3/4/5).
- All M3 named branches preserved as audit history.

---

*M3-3 closes with M3-3e-5. The 1D path is native; the 2D path is
native; the cache_mesh shim is gone. Methods paper §10.7 numbers
all hold. Ready for M3-4.*
