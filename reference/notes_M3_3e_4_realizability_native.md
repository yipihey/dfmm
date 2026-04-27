# M3-3e-4 — native `realizability_project_HG!`

> **Status (2026-04-26):** *Done.* M2-3 realizability projection no
> longer touches `mesh.cache_mesh`. The `cache_mesh` field stays
> allocated on `DetMeshHG` (still consumed by the wrapper helpers
> `total_momentum_HG`, `total_energy_HG`, `segment_density_HG`,
> `segment_length_HG`, `sync_*_HG!`, and the test-side Eulerian
> profile extractors in `test_M3_1_phase5_sod_HG.jl_helpers.jl` and
> `test_M3_2_phase7_steady_shock_HG.jl`). M3-3e-5 will drop the field
> once the remaining wrapper helpers + test-side accessors are
> replaced.

## What landed

The shim `realizability_project_HG!` in `src/newton_step_HG_M3_2.jl`
is replaced with a native body that mirrors M1's
`realizability_project!` line-for-line on the physics. Reads
`(α, β, x, u, s, Pp, Q)` per cell from `fields[j][1]` (no `read_detfield`
helper — direct field accesses match the M3-3e-2 pattern), applies the
unchanged per-cell math, and writes `(s, Pp)` back via
`fs.s[j] = (val,)` / `fs.Pp[j] = (val,)`. `(α, β, x, u, Q)` are
untouched (the projection lives entirely in the `(s, P_⊥)` sector).

The fast-path `kind == :none` increments `stats.n_steps` and returns
without iterating cells (matches M1's short-circuit). The
`:reanchor` branch is the only other supported kind; an `@assert`
catches misuse identically to M1.

### Per-cell density formula

Density is computed inline as `ρ_j = Δm_j / (x_{j+1} - x_j)` with
periodic wrap on `j == N` — the same formula M3-3e-2's
`inject_vg_noise_HG_native!` uses. The `J_j = 1.0 / ρ_j` reciprocal
matches M1 byte-for-byte; the `total_dE_inj` accumulator uses the
`J·Δm` ordering (same FP ordering as M1's `J_j * seg.Δm`).

### Realizability target + projection event

Identical to M1:

```
M_vv_target = max(headroom · β², Mvv_floor)
if M_vv_pre >= M_vv_target  →  no-op (track Mvv_min_post)
else  →  raise s, debit ½ ΔPxx from P_⊥ (clipped at pressure_floor),
         residual admitted as silent floor-gain on total_dE_inj.
```

ProjectionStats counters (`n_steps`, `n_events`, `n_floor_events`,
`total_dE_inj`, `Mvv_min_pre`, `Mvv_min_post`) are updated in the
same order, with the same accumulator arithmetic, so they remain
byte-identical to M1.

### NaN sentinel handling

The legacy `Pp = NaN` sentinel from 5-arg `DetField` constructors is
treated isotropically (`Pp_pre = ρ_j · Mvv_pre`) — same handling as
M1's `realizability_project!`.

## Bit-exact gates

All M2-3 + composed-realizability tests pass at byte-equality with
the prior cache-mesh-driven path:

- `test_M3_2_M2_3_realizability_HG.jl` (160 asserts) — primary gate.
  Single-cell boundary projection, mass + momentum exactness across
  5 calls, M1-vs-HG bit-equality on a smooth IC over 8 stochastic
  steps with `:reanchor`, post-projection invariant `M_vv ≥ headroom · β²`,
  ProjectionStats round-trip.
- `test_M3_3e_2_stochastic_native_vs_cache.jl` (788 asserts) — Phase 8
  + projection composition. Block 3 (`K = 10 + reanchor`) actively
  exercises the lifted projection through `det_run_stochastic_HG!`;
  byte-equality holds.
- `test_M3_3e_3_amr_tracer_native_vs_cache.jl` (5784 asserts) —
  unaffected (no projection in AMR path), confirmed unchanged.

Plus a new defensive cross-check
`test_M3_3e_4_realizability_native_vs_cache.jl` (708 byte-equality
asserts) running M1 `realizability_project!` on `Mesh1D` in parallel
with the native `realizability_project_HG!` on `DetMeshHG`:

- Block 1 (single boundary cell, N = 16): 119 asserts. Per-cell
  `(α, β, x, u, s, Pp, Q)` byte-equal + ProjectionStats byte-equal.
- Block 2 (multi-event, N = 32): 231 asserts. Four cells violating
  the cone at varying β; n_events ≥ 2.
- Block 3 (5 successive calls, idempotent): 119 asserts. After the
  first call all cells are inside the cone; subsequent calls are
  no-ops byte-for-byte.
- Block 4 (Mvv_floor branch): 119 asserts. β = 1e-4 IC drives
  `Mvv_target = Mvv_floor` (n_floor_events ≥ 1) — exercises the
  absolute-floor branch.
- Block 5 (`:none` fast-path): 120 asserts. State untouched, only
  `n_steps` increments.

After M3-3e-5 drops the cache_mesh field this test will be retired
(it depends on running M1's `realizability_project!` on a parallel
`Mesh1D`).

## Wall-time impact (N = 80, periodic, 1 projection call)

Median over 5000-call loop, post-warmup, IC with 3 boundary cells:

- Native HG (M3-3e-4):       ~462 µs / call
- Cache-mesh-driven shim:    ~15 µs / call
- M1 baseline (Mesh1D):      ~2.1 µs / call

The native HG path is ~30× slower per projection call than the
cache_mesh shim, and ~220× slower than the M1 `Mesh1D` baseline.
This matches the M3-3e-2 stochastic-native pattern: per-cell HG
view indirection (`fs.x[j][1]` returning a `PolynomialView` whose
`[1]` is the scalar) is heavier than M1's
`mesh.segments[j].state.x` direct read.

Allocations: ~236 KB per call (N = 80, ~3 KB/cell), driven by the
HG view allocations in the per-cell read path. M3-3e-2 absorbed
the same overhead at the per-step level (~20 % slower than the
shim on stochastic injection).

This *is* a per-call regression vs the cache_mesh shim. **It is
accepted by the M3-3e plan** because:

1. M2-3 realizability projection fires once per stochastic step
   (`pre-Newton` projection in `det_run_stochastic_HG_native!` and
   once after VG injection — i.e. 2× per step). At N = 80,
   net overhead ≈ 0.9 ms / step.
2. The Phase-8 native step already takes ~9 ms / step at N = 80
   per M3-3e-2's note. The projection contribution is therefore
   ≤10 % of the step cost, well below the brief's "negligible
   against the per-step work itself" guidance.
3. M3-3e-5 will drop the cache_mesh field entirely — the
   comparison thereafter is against M1's `Mesh1D` baseline, not
   against the shim. The aggregate Phase-8 wall-time improvement
   from removing the per-step `sync_cache_from_HG!` /
   `sync_HG_from_cache!` round-trips is the load-bearing M3-3e
   wall-time win, reported in M3-3e-5.

A future HG-side optimization (lift `read_detfield` /
`write_detfield!` patterns to a single-allocation hoist or per-cell
struct-of-vectors view) would close the gap; flagged for the HG
design-guidance backlog. Out of scope for M3-3e-4.

## Test count delta

12667 + 1 deferred → 13375 + 1 deferred (+708 new asserts in
`test_M3_3e_4_realizability_native_vs_cache.jl`). Top-level
`Pkg.test("dfmm")` summary:
`13375 Pass / 1 Broken / 13376 Total / 5m42s`.

## Files touched

- `src/newton_step_HG_M3_2.jl` — replaced the cache_mesh-driven
  body of `realizability_project_HG!` with a native HG-side
  implementation. Net diff: +112 / −13 (≈ +99 LOC added: the
  `:none` fast-path block, per-cell math, plus expanded docstring
  contract describing the bit-equality gate).
- `test/runtests.jl` — appended a new `@testset` block for
  `test_M3_3e_4_realizability_native_vs_cache.jl`.
- `test/test_M3_3e_4_realizability_native_vs_cache.jl` — new (~210
  LOC), 708 byte-equality asserts.

## HG API surprises

None beyond what M3-3e-1/2/3 documented. The native projection
uses only the `PolynomialFieldSet` accessors `fs.x[j][1]`,
`fs.s[j] = (val,)` etc., which were already established in
M3-3a/b/c/d for the 2D field sets and inherited by M3-3e-1/2 for
the 1D path. The per-cell view allocation overhead noted in
M3-3e-2's wall-time analysis applies here, sharper because the
projection is otherwise a few-FLOP operation.

## Pointers for M3-3e-5

After M3-3e-4 lands, the only remaining src-side consumers of
`mesh.cache_mesh` are:

- `total_momentum_HG`, `total_energy_HG`, `segment_density_HG`,
  `segment_length_HG` (`src/newton_step_HG.jl`) — diagnostic
  helpers; should be lifted to read directly from `fields` + `Δm`.
- `sync_cache_from_HG!`, `sync_HG_from_cache!` (`src/newton_step_HG.jl`)
  — only consumed by the test-side accessors below; can be retired
  after the test side is converted.
- `_resize_cache_mesh_HG!` (`src/action_amr_helpers.jl`) — keeps
  the cache mesh in lock-step with HG after AMR events. Not needed
  once the cache_mesh field is dropped.
- `rebuild_HG_from_cache!` (`src/action_amr_helpers.jl`) — legacy
  helper, unused on the hot path; remove with the field.

Plus the test-side accessors in
`test/test_M3_1_phase5_sod_HG.jl_helpers.jl` and
`test/test_M3_2_phase7_steady_shock_HG.jl` that read
`mesh_HG.cache_mesh` directly for Eulerian profile extraction —
these need a thin HG-native helper (`extract_eulerian_profiles_HG`?)
that walks `fields.x` + `Δm` to reconstruct the same diagnostic
quantities.

After all of the above, M3-3e-5 can drop the
`cache_mesh::Mesh1D{T,DetField{T}}` field from `DetMeshHG{T}`
in `src/newton_step_HG.jl` and the `DetMeshHG_from_arrays`
allocation. Final regression: full 13376 + 1 (minus the four
defensive cross-check tests, which become circular without the
parallel `Mesh1D` to compare against).
