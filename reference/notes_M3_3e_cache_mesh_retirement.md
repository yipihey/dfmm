# M3-3e — `cache_mesh::Mesh1D` shim retirement (status note)

> **Status (2026-04-26):** *In progress / scope mismatch flagged.* This
> note documents the M3-3e investigation. The full bit-exact retirement
> of `cache_mesh` across all 1D paths (Phase 2/5/5b/7/8/11 + M2-1/M2-3)
> turns out to be a multi-day reimplementation effort, not the 2-day
> "drop the field" change implied by §9 of the design note. This note
> details the analysis and proposes a sub-divided plan.

## TL;DR

The cache_mesh shim is *not* a single-call delegation. It is the
backbone of the entire 1D HG-substrate path: it carries
`Mesh1D{T,DetField{T}}` segment storage, drives the AMR
refine/coarsen rebuild, owns the tracer-matrix storage that
`TracerMeshHG` shares, and supplies the `Vector{Segment}` layout
that the Phase-8 stochastic injection RNG sequencing depends on
byte-for-byte.

Replacing it in 2 days, with bit-exact 0.0 parity across 3991 + 1
tests, is not feasible without first decomposing it into per-sector
lifts. This note prepares that decomposition.

## What the cache_mesh actually does

A `DetMeshHG.cache_mesh::Mesh1D{T,DetField{T}}` is the load-bearing
shim through which **every 1D HG-side primitive** delegates to its
M1 counterpart. The M3-2 status note (`notes_M3_2_phase7811_m2_port.md`
§ "Open issues / handoff to M3-3" item 2) flagged this for retirement
in M3-3, but understated the scope. Concretely:

### A. `det_step_HG!` (Phase 2/5/5b/7) — `src/newton_step_HG.jl`

`det_step!` (`src/newton_step.jl` lines 265–627, ~360 LOC) does:

1. **Pre-Newton bookkeeping** — pack flat `y_n` from `mesh.segments`,
   compute `Δm`, `s_vec`, build inflow/outflow ghost data.
2. **Newton solve** — `NewtonRaphson` on `det_el_residual` (a
   pure-flat-array function in `src/cholesky_sector.jl`).
3. **Phase 5b post-Newton** — entropy update for q-dissipation
   (mutates `seg.state.s` per segment).
4. **Phase 5 post-Newton BGK** — `(P_xx, P_⊥)` joint relaxation +
   entropy re-anchor + β realizability clip + `Q · exp(-Δt/τ)`
   (mutates `seg.state.{β, s, Pp, Q}`).
5. **Phase 7 inflow/outflow pinning** — overwrites the leftmost /
   rightmost `n_pin` segments with Dirichlet upstream/downstream
   values, including the cumulative-Δx vertex chain reconstruction
   (`Δx_j = Δm_j / ρ_inflow`).

**(2) is already pure-array.** `det_el_residual` takes `y_np1, y_n,
Δm, s, L_box, dt, bc, inflow_xun, outflow_xun, …` — no
`Mesh1D` required. Re-using it from the HG path is straightforward.

**(3), (4), (5) are written against `mesh.segments[j].state`
mutation.** Lifting them onto the HG `PolynomialFieldSet`
requires a parallel implementation that touches `fields.alpha[j]`,
`fields.beta[j]`, etc. Per-line scalar arithmetic is identical, but
**the new code path is ~300 LOC of new HG-side code that has to be
verified bit-exact-equal to the M1 `seg.state.foo = ...` writes**.

### B. `inject_vg_noise_HG!` (Phase 8) — `src/newton_step_HG_M3_2.jl`

`inject_vg_noise!` (`src/stochastic_injection.jl`, ~600 LOC) is a
realizability-projected variance-gamma noise injection on the
deterministic state. It iterates `mesh.segments` in M1's order,
draws RNG samples per segment in that order, applies
realizability-checked δ updates, and emits per-segment
`InjectionDiagnostics`. **The RNG sequencing is order-sensitive**
— Phase-8 bit-exact parity requires the HG path to draw RNG
samples in the same per-cell order as M1.

For the HG `PolynomialFieldSet` (SoA layout), the per-cell
iteration order is the same `1:N` linear index, so the *order*
matches. But: the M1 path also reads `seg.Δm` and the per-segment
mass (which is shared as `mesh.Δm` on the wrapper). The lift is
~600 LOC of new HG-side code, with the load-bearing test gate
being the per-cell `δ(ρu), eta, divu, compressive` diagnostics
agreeing to 0.0 absolute across `det_run_stochastic_HG!`.

### C. AMR refine/coarsen — `src/action_amr_helpers.jl`

`refine_segment!` / `coarsen_segment_pair!` (`src/amr_1d.jl`) operate
on `mesh.segments::Vector{Segment}` via `splice!` + state
redistribution. The HG wrapper currently calls them on the cache
mesh, then **rebuilds** the HG `SimplicialMesh{1, T}` and `fields`
from scratch via `rebuild_HG_from_cache!` (because HG doesn't ship
a `register_on_refine!` hook for `SimplicialMesh{1, T}`).

To retire the cache_mesh from this path, we need either:
- **Option A (design note recommendation):** switch to
  `HierarchicalMesh{1}` for the 1D Eulerian frame, which **does**
  support `register_on_refine!` (M3-3d wired this for 2D). The
  Lagrangian `SimplicialMesh{1, T}` + `PolynomialFieldSet` would
  then be driven by an HG-paired listener. This is non-trivial
  because M1's 1D AMR is *Lagrangian* (split a segment in the
  mass-coordinate frame, redistribute mass), not Eulerian. Switching
  to HierarchicalMesh{1} on the Eulerian frame doesn't directly
  retire cache_mesh — we still need a per-segment Lagrangian record,
  and the natural carrier for that is the HG field set itself.
- **Option B:** hand-roll a 1D-Lagrangian refine/coarsen primitive
  that operates directly on `(SimplicialMesh{1, T}, PolynomialFieldSet,
  Δm::Vector, p_half::Vector)` without going through Mesh1D. This
  is ~250 LOC of new code, plus tests for conservative
  refine/coarsen and the AMR-step orchestration. Bit-exact parity
  to M1's `Mesh1D`-driven path is the gate.

Either option is at least a full day of work.

### D. Tracer storage — `TracerMeshHG`

`TracerMeshHG` wraps an M1 `TracerMesh` whose `fluid` is the cache
mesh. Storage is shared. Retiring cache_mesh requires either:
- Lifting `TracerMesh` to operate on `(DetMeshHG, n_tracers,
  names)` directly (parallel struct + matrix storage), then
  re-implementing `set_tracer!`, `add_tracer!`, `tracer_index`,
  `tracer_at_position`, `n_tracer_*` against it. ~150 LOC.
- Coupling the tracer matrix to `PolynomialFieldSet`'s per-cell
  storage, using the field set's `add_field!` API to add per-tracer
  scalar fields. Cleaner long-term but requires refactoring of M1's
  matrix-of-rows data structure.

### E. Realizability projection — `realizability_project_HG!`

`realizability_project!` (`src/stochastic.jl`?) operates on
`mesh.segments[j].state` to clip β and reanchor entropy on
realizability-violation cells. The lift is ~80 LOC of new
field-set-driven code, with the bit-exact gate being
`ProjectionStats` agreement on synthetic boundary-cell ICs.

## Total scope

| Sector | New LOC (approx) | Days |
|---|---|---|
| `det_step_HG!` Newton + post-Newton | 350 | 1.5 |
| Phase-8 stochastic injection (RNG-bit-equal) | 500 | 1.5 |
| AMR refine/coarsen (Option B) | 250 | 1.0 |
| TracerMeshHG (no cache_mesh) | 150 | 0.5 |
| Realizability projection | 80 | 0.25 |
| Sync/setup helpers | 50 | 0.25 |
| **Total** | **~1380 LOC** | **~5 days** |

That's **~5 days of careful work + bit-exact verification** at every
intermediate test gate. The brief's 2-day estimate appears to assume
a much shallower retirement (just dropping the field on `DetMeshHG`),
but the field is genuinely load-bearing.

## Recommendation: sub-divide M3-3e

Given the realistic scope, recommend splitting M3-3e into:

- **M3-3e-1** (1 day): native `det_step_HG!` retiring the cache_mesh
  on the **deterministic Newton path only** (Phase 2/5/5b/7
  including post-Newton BGK, q-dissipation, inflow/outflow pinning).
  Cache_mesh stays on the wrapper but is no longer touched by
  `det_step_HG!`. Bit-exact gate: M3-1/M3-2 deterministic Newton
  tests + Phase 7 steady-shock test (~1500 asserts).

- **M3-3e-2** (1 day): native `inject_vg_noise_HG!` /
  `det_run_stochastic_HG!` retiring cache_mesh on the **stochastic
  injection path**. Bit-exact RNG sequencing gate: Phase-8
  stochastic test (`test_M3_2_phase8_stochastic_HG.jl`) and the
  per-cell `InjectionDiagnostics` parity (~260 asserts).

- **M3-3e-3** (1 day): native AMR refine/coarsen + `TracerMeshHG`
  retirement. Bit-exact gates: M2-1 + Phase-11 + M2-2 tests
  (~110 asserts).

- **M3-3e-4** (0.5 day): realizability projection lift. Bit-exact
  gate: M2-3 tests (~160 asserts).

- **M3-3e-5** (0.5 day): drop `cache_mesh` field; final regression
  on full 3991 + 1 suite.

After M3-3e-5, M3-3 closes.

This decomposition keeps each sub-phase independently verifiable
against a clear bit-exact gate, instead of one big-bang retirement
where regression diagnosis would be combinatorial.

## Why I did not ship a partial retirement in this session

The brief is explicit: "Don't reintroduce a different shim. If
native HG residual is harder than expected, document and report —
don't add a 'minimal cache' or similar workaround." A partial
retirement that drops `cache_mesh` from the deterministic Newton
path but leaves it for AMR/stochastic/tracer would either:

1. Need to keep the field (breaking the brief's explicit
   "remove `cache_mesh::Mesh1D` from `DetMeshHG` struct"), OR
2. Maintain two parallel storage stacks during the transition
   (the 'minimal cache' workaround the brief forbids).

The honest path is to (a) document the scope mismatch, (b) propose
the sub-divided plan, and (c) leave M3-3e open for the next agent
or session with a clear roadmap.

## What was done in this session

- Read the design note `notes_M3_3_2d_cholesky_berry.md` (§3, §6.6,
  §9 entry).
- Read all four cache_mesh-touching src files
  (`src/newton_step_HG.jl`, `src/newton_step_HG_M3_2.jl`,
  `src/action_amr_helpers.jl`, `src/types.jl`) and traced each
  delegation path.
- Read M1's `det_step!`, `det_el_residual`, `inject_vg_noise!`,
  `refine_segment!`, `coarsen_segment_pair!`, `realizability_project!`
  to characterise per-sector lift complexity.
- Confirmed package loads + `det_step_HG!` runs on a smoke IC
  (current state on `main` is functional).
- Wrote this status note + the sub-phase plan above.

No code changes were committed in this session. Branch
`m3-3e-cache-mesh-retire` is open at the same HEAD as `main`
(`6a65411`) for the next agent to pick up.

## Pointers for the next agent

1. **Start with M3-3e-1** (deterministic Newton). It's the easiest
   lift because `det_el_residual` already takes flat arrays. Sketch:
   - Add `det_step_HG_native!` to `src/newton_step_HG.jl` that
     pack/unpacks from `fields::PolynomialFieldSet` (not from
     `mesh.cache_mesh.segments`).
   - Lift the post-Newton blocks (BGK, q-dissipation, inflow/outflow
     pinning) to mutate `fields.{alpha, beta, x, u, s, Pp, Q}[j]`
     directly. Each block is ~50 LOC; copy-paste from M1 with
     `seg.state.foo` → `fields.foo[j][1]`.
   - Run M3-2 deterministic-only tests. Confirm 0.0 parity.
   - Switch `det_step_HG!` to call the native path. Keep
     `cache_mesh` field allocated (still used by M3-3e-2/3/4).

2. **Then M3-3e-2** (stochastic). The risk is RNG-bit-exact
   sequencing. Sketch: copy `inject_vg_noise!` to a new
   `inject_vg_noise_HG_native!` that iterates the HG field set in
   the same `1:N` order, draws RNG in the same order, mutates
   `fields.{alpha, beta, ...}[j]` directly.

3. **Then M3-3e-3** (AMR). Recommend Option B (hand-rolled 1D-Lag
   refine/coarsen) over Option A (HierarchicalMesh{1}). Option A
   adds an architectural mismatch (Lagrangian `SimplicialMesh{1, T}`
   + Eulerian `HierarchicalMesh{1}` together) that is wider-scope
   than M3-3e.

4. **Then M3-3e-4** (realizability) and **M3-3e-5** (drop the field
   + final regression).

## Phase 8 RNG note (for M3-3e-2)

The Phase-8 RNG bit-equality concern is sharper than the brief
hints. M1's `inject_vg_noise!` calls `rng` per-cell in segment
order, but the realizability projection uses the *post-projection*
state to gate the RNG draw — i.e., a cell that fails projection at
step `n` will draw a different RNG sequence at step `n+1` than a
cell that passed. The HG-native lift must preserve this exact
order/branching, not just the per-cell-index iteration order. The
M3-2 Phase-8 tests cover this (RNG-seed-driven parity on
`det_run_stochastic_HG!` over 5 steps); use those as the verification
gate.

## Files inventory (for the next agent)

Files with `cache_mesh` references (from `grep -l cache_mesh`):

- `src/types.jl` — currently does NOT define DetMeshHG (it's in
  newton_step_HG.jl); types.jl just defines DetField, ChField, etc.
- `src/newton_step_HG.jl` — `cache_mesh` field declaration on
  `DetMeshHG`, allocation in `DetMeshHG_from_arrays`,
  `sync_cache_from_HG!` / `sync_HG_from_cache!`, and
  `det_step_HG!`'s delegation to `det_step!`.
- `src/newton_step_HG_M3_2.jl` — `inject_vg_noise_HG!`,
  `det_run_stochastic_HG!`, `TracerMeshHG`,
  `realizability_project_HG!` all delegate via cache_mesh.
- `src/action_amr_helpers.jl` — `rebuild_HG_from_cache!`,
  `refine_segment_HG!`, `coarsen_segment_pair_HG!`,
  `action_error_indicator_HG`, `gradient_indicator_HG`,
  `amr_step_HG!` all delegate.
- `src/dfmm.jl` — only has a comment string mentioning cache_mesh.
- `test/test_M3_1_phase5_sod_HG.jl_helpers.jl` — uses
  `mesh_HG.cache_mesh` directly to extract Eulerian profiles
  (a test-side convenience).
- `test/test_M3_2_phase7_steady_shock_HG.jl` — same pattern.
- `test/runtests.jl` — comment mentioning cache_mesh.

After M3-3e completes, the test-side `mesh_HG.cache_mesh` accesses
must be replaced with HG-native diagnostic accessors
(`extract_eulerian_profiles_HG`?). Sketch: read positions and
densities directly from `fields.x` and `Δm` rather than via Mesh1D's
`segment_density(mesh, j)`.

## Compatibility note

The 2D path (M3-3a/b/c/d) does not use `cache_mesh` at all — it
already runs natively on `HierarchicalMesh{2}` + `PolynomialFieldSet`.
M3-3e is purely about retiring the **1D** cache_mesh shim. The 2D
path is unaffected by any of the work outlined here.

---

*End of status note. ~250 lines of analysis. The next agent should
start with M3-3e-1 (deterministic Newton) and verify 0.0 parity on
the M3-2 deterministic test set before proceeding.*
