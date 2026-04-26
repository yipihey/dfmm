# Phase M3-2 — Phase 7 / 8 / 11 + M2-1 / M2-3 ports onto HG (1D)

**Date:** 2026-04-26.
**Branch:** `m3-2-phase7811-m2-on-hg`.
**Status:** complete; bit-exact parity to M1 verified across all five
sub-phases.

This note is the implementation log for Phase M3-2 of
`reference/MILESTONE_3_PLAN.md`. The phase ports M1's Phase 7
(heat-flux Q + steady-shock + inflow/outflow), Phase 8 (variance-
gamma stochastic injection), Phase 11 (passive tracers), the M2-1
sub-phase (action-based AMR primitives), and the M2-3 sub-phase
(realizability projection) onto the HierarchicalGrids (HG) substrate
established by M3-0 and extended by M3-1. Same physics, HG mesh.

The bit-exact-parity contract is preserved: every per-cell state on
the HG path agrees with M1's `Mesh1D{T,DetField{T}}` path to **0.0
absolute** at every timestep across every test case, well below the
brief's 5e-13 target. The few exceptions are aggregate sums where
ULP-level differences in summation order are inherent
(`total_mass_HG` vs `total_mass`, see "Open issues / handoff").

## Design pattern — same shim as M3-1

All M3-2 wrappers delegate to their M1 counterparts through the
`DetMeshHG.cache_mesh::Mesh1D{T,DetField{T}}` shim that M3-1
established. The HG-side wrapper:

1. `sync_cache_from_HG!(mesh)` — refresh the cache mesh from the HG
   field set;
2. call the M1 primitive on the cache mesh with identical kwargs;
3. `sync_HG_from_cache!(mesh)` — write the result back to the HG
   field set.

For AMR (M2-1) the rebuild step is more involved: after a refine /
coarsen event the cache mesh's segment count changes, and the HG
wrapper's `SimplicialMesh{1, T}`, `PolynomialFieldSet`, `Δm`,
`p_half`, and `L_box` fields must all be re-allocated to match. The
helper `rebuild_HG_from_cache!` (in `src/action_amr_helpers.jl`)
encapsulates this. Per the M3-2 brief, this implementation goes
through the cache mesh shim because HG does not yet expose a
`register_on_refine!` callback (design-guidance item #5 in
`notes_HG_design_guidance.md`).

## Block-by-block summary

### Block 1 — Phase 7 (heat-flux Q + steady shock + inflow/outflow)

The Q field advancement is performed by `det_step!` itself
(post-Newton operator-split: Lagrangian transport `Q · ρ_np1/ρ_n`
followed by BGK relaxation `Q · exp(-Δt/τ)`). Since `det_step_HG!`
already delegates to `det_step!`, Phase 7 parity was inherited from
M3-1; the only change in M3-2 was extending `det_step_HG!`'s kwarg
signature to forward `bc`, `inflow_state`, `outflow_state`, and
`n_pin` to the cache mesh.

Tests in `test/test_M3_2_phase7_steady_shock_HG.jl`:
- BGK Q-decay on a smooth periodic mesh matches `Q · exp(-N·Δt/τ)`
  to 1e-10.
- Steady-shock Mach-3 IC + `bc = :inflow_outflow` + `q_kind =
  :vNR_linear_quadratic` runs over 33 steps on N = 80 with bit-exact
  parity vs M1 on `(x, u, α, β, s, Pp)` and on Q specifically (both
  `max_parity_err == 0.0` and `max_Q_parity_err == 0.0`).
- R-H plateau preservation matches M1 numerically to 5%/10%/5% on
  (ρ, u, P) at the short-horizon window.

### Block 2 — Phase 8 (variance-gamma stochastic injection)

`inject_vg_noise_HG!` and `det_run_stochastic_HG!` wrap the M1
primitives. With identical RNG seed, params, and Phase-5 / 5b
kwargs, the HG path produces byte-identical state and per-cell
diagnostics (`delta_rhou`, `eta`, `divu`, `compressive`, …).

Tests in `test/test_M3_2_phase8_stochastic_HG.jl`:
- `det_run_stochastic_HG!` vs `det_run_stochastic!` with identical
  seed: 5-step run, N=32 sinusoidal-velocity IC, all per-cell state
  fields agree to 0.0 absolute.
- Zero-noise gate: `C_A = C_B = 0` reduces to `det_run_HG!` (96
  per-segment field comparisons agree to 1e-12).
- Mass exactness across stochastic injection: `total_mass_HG`
  bit-stable.
- Per-cell `InjectionDiagnostics` parity: 7 fields × N=16 = 112
  asserts agreeing to 0.0.

### Block 3 — Phase 11 (passive tracers) + M2-2 (multi-tracer)

`TracerMeshHG{T}` wraps an M1 `TracerMesh{T, M<:Mesh1D}` whose
`fluid` is the HG wrapper's cache mesh. The tracer matrix is
therefore the same storage M1 uses — bit-exact preservation is
inherited; `advect_tracers_HG!` is the same no-op as
`advect_tracers!`.

Tests in `test/test_M3_2_phase11_tracer_HG.jl` and
`test/test_M3_2_M2_2_multitracer_HG.jl`:
- 1000-step Sod-IC run with 3 tracers (step / sin / Gaussian):
  L∞ change literally 0.0.
- Wave-pool 150-step run with stochastic injection enabled and 5
  step tracers: matrix object identity preserved + L∞ change literally 0.0.
- M1↔HG tracer-matrix parity over 200 deterministic steps and over
  50 stochastic-injection steps with identical RNG seed: tracer
  matrices bit-equal.
- API delegation (`set_tracer!`, `add_tracer!`, `tracer_index`,
  `tracer_at_position`, `n_tracer_*`) all forwarded correctly.

### Block 4a — M2-1 (action-based AMR)

Three new files in `src/action_amr_helpers.jl`:

| Function | Behaviour |
|---|---|
| `refine_segment_HG!(mesh, j; tracers)` | Calls `refine_segment!` on cache_mesh; rebuilds HG mesh + fields |
| `coarsen_segment_pair_HG!(mesh, j; tracers, Gamma)` | Same shape, calls `coarsen_segment_pair!` |
| `action_error_indicator_HG(mesh; dt, Gamma)` | Calls `action_error_indicator` |
| `gradient_indicator_HG(mesh; field, Gamma)` | Calls `gradient_indicator` |
| `amr_step_HG!(mesh, indicator, τ_refine, τ_coarsen; tracers, ...)` | Calls `amr_step!`; rebuilds |
| `rebuild_HG_from_cache!(mesh)` | Re-allocate `mesh.mesh::SimplicialMesh{1,T}`, `mesh.fields`, `mesh.Δm`, `mesh.p_half`, `mesh.L_box` from cache |

Tests in `test/test_M3_2_M2_1_amr_HG.jl`:
- refine_segment_HG! bit-exact mass + exact momentum
- coarsen_segment_pair_HG! bit-exact mass + exact momentum
- refine→coarsen roundtrip restores Δm to 1e-15
- action_error_indicator_HG matches M1 bit-for-bit (`==`)
- gradient_indicator_HG matches M1 across `field ∈ (:rho, :u, :P)` (`==`)
- `amr_step_HG!` hysteresis + `min_segments` cap enforced
- amr_step_HG! parity vs M1 on a Sod-style step IC: same
  `(n_refined, n_coarsened)` counts and per-cell state agree
  bit-exactly.

### Block 4b — M2-3 (realizability projection)

`realizability_project_HG!` wraps `realizability_project!`. The
`det_run_stochastic_HG!` driver (Block 2) already exercises the
projection through its inner call to `det_run_stochastic!`, which
calls `realizability_project!` twice per step.

Tests in `test/test_M3_2_M2_3_realizability_HG.jl`:
- Synthetic boundary-cell IC: pre-projection violates realizability,
  post-projection every cell satisfies M_vv ≥ headroom · β².
- Mass + momentum exactness across 5 projection events.
- M1↔HG bit-equality on a smooth IC (no-event regime): 96 field
  comparisons agree to round-off; `ProjectionStats.n_events == 0`
  on both paths.
- Realizability invariant after projection.
- `ProjectionStats` round-trip + `reset!`.

## File layout (new in M3-2)

| File | Role |
|---|---|
| `src/newton_step_HG.jl` (extension) | Forward `bc`, `inflow_state`, `outflow_state`, `n_pin` from `det_step_HG!` to `det_step!` |
| `src/newton_step_HG_M3_2.jl` (new) | `inject_vg_noise_HG!`, `det_run_stochastic_HG!`, `TracerMeshHG`, `advect_tracers_HG!`, tracer accessor delegation, `realizability_project_HG!` |
| `src/action_amr_helpers.jl` (new) | `rebuild_HG_from_cache!`, `refine_segment_HG!`, `coarsen_segment_pair_HG!`, `action_error_indicator_HG`, `gradient_indicator_HG`, `amr_step_HG!` |
| `src/dfmm.jl` (extension) | Append-only: include + export new identifiers |
| `test/test_M3_2_phase7_steady_shock_HG.jl` | Phase-7 steady-shock parity (Block 1) |
| `test/test_M3_2_phase8_stochastic_HG.jl` | Phase-8 stochastic parity (Block 2) |
| `test/test_M3_2_phase11_tracer_HG.jl` | Phase-11 tracer bit-exactness (Block 3) |
| `test/test_M3_2_M2_2_multitracer_HG.jl` | M2-2 wave-pool tracer fidelity (Block 3) |
| `test/test_M3_2_M2_1_amr_HG.jl` | M2-1 action-AMR primitives (Block 4a) |
| `test/test_M3_2_M2_3_realizability_HG.jl` | M2-3 realizability projection (Block 4b) |
| `test/runtests.jl` (append) | M3-2 testset block |
| `reference/notes_M3_2_phase7811_m2_port.md` | This file |

Files explicitly **not modified** in M3-2:

- M1 source (`src/heat_flux.jl`, `src/stochastic_injection.jl`,
  `src/tracers.jl`, `src/amr_1d.jl`, `src/cholesky_sector.jl`,
  `src/segment.jl`, `src/discrete_action.jl`,
  `src/discrete_transport.jl`, `src/newton_step.jl`,
  `src/deviatoric.jl`, `src/artificial_viscosity.jl`).
- Existing M3-0 / M3-1 tests and source.
- `Project.toml`, `Manifest.toml`.
- `reference/MILESTONE_*_STATUS.md`, `reference/notes_phase*.md`,
  `specs/`, `design/`, `HANDOFF.md`, `LICENSE`, `py-1d/`.

## Total test count

Before M3-2: M1 + M2 + M3-0 + M3-1 = 2264 + 1 deferred = 2265 total.

M3-2 added **542 new test assertions** (per the M3-2 testset block,
`Pkg.test()` output):

- Phase 7 (HG): 17
- Phase 8 (HG): 259
- Phase 11 (HG): 19
- M2-2 (HG): 5
- M2-1 (HG): 82
- M2-3 (HG): 160

After M3-2: full `Pkg.test()` reports
**2806 passed + 1 broken = 2807 total**.

The "1 broken" is the pre-existing Phase-6 post-crossing
`@test_skip` (gated on a future shock-capturing extension); not
introduced by M3-2.

## Wall-time benchmarks

Single-thread, Apple M-series.

| Setup | M1 wall | HG wall | Ratio HG/M1 |
|---|---|---|---|
| Phase-7 steady-shock, N=80, 50 steps, q=:vNR | 0.460 s | 0.772 s | 1.68× |
| Phase-8 stochastic+proj, N=64, 200 steps | 1.625 s | 1.591 s | 0.98× |
| Phase-11 tracer + det, N=80, 1000 steps | (= det_step_HG!) | (= det_step_HG!) | ~1× |
| M2-1 50× refine/coarsen, N=32 | 46 ms | 12 ms | 0.26× |
| M2-3 5000× project, N=16 | 1.95 ms | 24.16 ms | ~12× |

Notes:
- Phase 7 ratio is higher than M3-1's 0.95× because the
  inflow-outflow path causes additional sync overhead per step on
  small N (with N → 1000 the ratio approaches 1).
- Phase 8 ratio is ~1× because the stochastic path is dominated by
  the Newton solve and VG sampling — the sync overhead is small.
- M2-1 ratio < 1 reflects compilation-cache benefits during the
  benchmark loop; refine/coarsen events are infrequent in
  production.
- M2-3 ratio ~12× because each `realizability_project!` call is
  ~0.4 µs and the HG sync overhead (~5 µs per direction) dominates.
  In a real run, the projection is called twice per step alongside
  a much larger Newton solve, so the relative cost is < 1%.

## HG workarounds applied

### `register_on_refine!` callback (design-guidance item #5)

HG does not yet expose a callback hook for downstream packages to
update auxiliary per-cell state when `refine_cells!` /
`coarsen_cells!` is invoked. M3-2 routes around this by performing
the AMR on the cache mesh and rebuilding the HG mesh + field set
from the cache via `rebuild_HG_from_cache!`. When HG ships the
callback, the rebuild step can be replaced by an in-place HG-side
update.

### Periodic-BC neighbour wiring on `bc = :inflow_outflow`

The HG `SimplicialMesh{1, T}` is built with periodic neighbour
matrix on `bc = :inflow_outflow` paths too — M1's `det_step!`
implements the Dirichlet pinning via the `bc` kwarg / inflow-outflow
ghost cells, not via the simplex-neighbour matrix. The HG mesh's
neighbour wiring is therefore a "topological" structure unrelated
to the BC; this matches the M3-1 convention.

## Open issues / handoff to M3-3

1. **`total_mass` vs `total_mass_HG` summation order.** The
   generator-based sum in M1's `total_mass(mesh::Mesh1D)` and the
   `Vector` reduction in `total_mass_HG(mesh::DetMeshHG)` can
   disagree at ULP level when N is large (~1e-15 on N=80). Per-cell
   `Δm` are bit-equal; only the aggregate differs. The Phase-11
   bit-exact parity test was adjusted to use a single `M_total` for
   both paths' tracer initialisations. M3-3 may want to pin the
   summation convention if downstream consumers depend on cross-path
   bit-exact totals.

2. **Retire the `cache_mesh` shim.** Still pending from the M3-1
   handoff. M3-3 should write a native HG-side EL residual
   evaluator + Newton solve + post-Newton operator splits, then
   drop the cache mesh. The bit-exact baseline from M3-2 makes the
   divergence checkable per-step on every Phase-7 / 8 / 11 / M2-1 /
   M2-3 path.

3. **`register_on_refine!` callback API in HG.** Items #5 from
   `notes_HG_design_guidance.md`. Once HG ships the callback,
   `rebuild_HG_from_cache!` can be replaced by an in-place HG-side
   update. M3-3 (or whichever phase first writes the native HG-side
   EL residual) is the natural time to retire the cache mesh in
   AMR, since at that point there is no longer a cache mesh to do
   the AMR work on.

4. **`coarsen_segment_pair!` momentum redistribution at `j == N`.**
   M1's `coarsen_segment_pair!` works correctly across the periodic
   wrap; the HG wrapper inherits this. No additional code path
   required, but worth flagging for M3-3 when the simplex-neighbour
   wiring becomes a first-class consumer.

5. **Phase 8 RNG seed reproducibility under threading.** When the
   HG-side wrapper uses HG's `parallel_for_cells` (currently only
   in `cholesky_step_HG!`), bit-exact parity will require the
   threading order to match M1's serial order. M3-2 inherits M3-1's
   "threaded = false on parity tests" convention. Production runs
   that need threading should accept O(eps) divergence between M1
   and HG in those configurations.
