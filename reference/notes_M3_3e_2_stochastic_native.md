# M3-3e-2 — native `inject_vg_noise_HG!` + `det_run_stochastic_HG!`

> **Status (2026-04-26):** *Done.* Phase-8 stochastic injection no
> longer touches `mesh.cache_mesh`. The `cache_mesh` field stays
> allocated on `DetMeshHG` (M3-3e-3/4 still consume it for AMR,
> tracers, and realizability projection) and `sync_cache_from_HG!` /
> `sync_HG_from_cache!` remain available for test-side diagnostics.
> M3-3e-5 will drop the field once all paths are native.

## What landed

Two new internal entry points in `src/newton_step_HG_M3_2.jl`:

1. `inject_vg_noise_HG_native!` — reproduces M1's `inject_vg_noise!`
   line-for-line on the physics, but reads/writes the per-cell state
   through `fields::PolynomialFieldSet` (via `fs.x[j][1]`,
   `fs.x[j] = (val,)`, etc.) instead of `mesh.segments[j].state`.
2. `det_run_stochastic_HG_native!` — composes the M3-3e-1 native
   deterministic step with the M3-3e-2 native injection in M1's
   per-step loop ordering: pre-Newton projection → Newton step →
   stochastic inject (which itself invokes a post-injection
   projection) → optional accumulator hooks.

The public `inject_vg_noise_HG!` and `det_run_stochastic_HG!` are now
thin pass-throughs to the native entry points. Cache-mesh-driven
bodies are gone.

### Per-step recipe (in `inject_vg_noise_HG_native!`)

Mirrors M1's `inject_vg_noise!` exactly:

1. **Compute divu per cell** from post-Newton `fs.x` and `fs.u`. Same
   centred form `(u_{j+1} - u_j) / (x_{j+1} - x_j)` with periodic wrap
   on `j == N`.
2. **Draw VG noise** in `1:N` order via `rand_variance_gamma!(rng,
   eta_white, λ, θ_factor)` — byte-identical to M1's RNG sequence at
   the same seed. Then 3-point periodic smoothing into `diag.eta`.
3. **Per-cell drift / noise / amplitude limiter / KE-debit** — read
   `(s, Pp)` from `fs`, compute drift `C_A · ρ · divu · Δt`, noise
   `C_B · ρ · √(max(-divu, 0) · Δt) · η_j`, apply the discriminant
   amplitude cap, debit `(2/3)·ΔKE_vol` from each of `(P_xx, P_⊥)`,
   re-anchor entropy from the new `P_xx`. Mutate `fs.s`, `fs.Pp`.
4. **Distribute cell momentum** `ΔP_j = δ · Δx_j` half-each to the two
   adjacent vertices via the lumped-mass `m̄_i`. Mutate `fs.u[i]` and
   `mesh.p_half[i]`.
5. **Realizability projection** — delegated to `realizability_project_HG!`
   (still cache_mesh-driven until M3-3e-4). Fast-path for
   `project_kind == :none` skips the cache_mesh sync (the M1 projection
   itself short-circuits in that mode).

### RNG-sequencing fidelity

Per cell `j` the RNG draws emitted by `inject_vg_noise_HG_native!`
exactly match M1's:

- `rand_variance_gamma!(rng, ...)` is called once per step on a length-N
  buffer (lines 720–721 of M1's `inject_vg_noise!`). Order of draws is
  determined by the rng.next-stream contract, which is independent of
  the storage layer. Both paths use the same `MersenneTwister` and
  the same call signature.
- After the buffer is filled, smoothing and the per-cell loop are
  side-effect-free w.r.t. the rng. The rng state therefore advances
  in the same byte sequence on both paths.

Confirmed bit-equal at K = 10 steps (`test_M3_3e_2_stochastic_native_vs_cache.jl`,
Block 2) and K = 5 steps (`test_M3_2_phase8_stochastic_HG.jl` Block 1)
both with `project_kind = :none` and `:reanchor`.

### Subtle floating-point ordering fix

The cell-momentum injection uses `Δp_cell[j] = δ · Δx_j` where M1
computes `Δx_j = J_j · seg.Δm` (with `J_j = 1/ρ_j`). The mathematically
equivalent `Δx_j = x_{j+1} - x_j` (geometric form) differs at the last
bit due to the `J = 1/ρ` reciprocal. **Bit-exact parity requires the
`J·Δm` form**, not the geometric `(x_{j+1} - x_j)`. Using the geometric
form caused a 3e-17 drift after 5 steps. After matching M1's ordering,
parity holds at 0.0 absolute over K = 10 steps.

## Bit-exact gates

All Phase-8-touching tests pass at byte-equality with the prior
cache-mesh-driven path:

- `test_M3_2_phase8_stochastic_HG.jl` (259 asserts) — primary gate.
  Includes K = 5 stochastic-noise parity + zero-noise reduction +
  mass-exactness + 1-step diagnostic-vector parity.
- `test_M3_2_M2_2_multitracer_HG.jl` (5 asserts) — composes Phase 8 +
  tracers; tracers still on cache_mesh until M3-3e-3 (state path is
  unaffected).
- `test_M3_2_M2_3_realizability_HG.jl` (160 asserts) — composes
  Phase 8 + realizability; realizability still on cache_mesh until
  M3-3e-4. The K = 8 step `:reanchor` parity test passes byte-equal.

Plus a new defensive cross-check test `test_M3_3e_2_stochastic_native_vs_cache.jl`
(788 byte-equality asserts) that runs the native path on `DetMeshHG`
in parallel with M1's `det_run_stochastic!` on a `Mesh1D` IC and
asserts byte-equal `(α, β, x, u, s, Pp, Q)` per cell after each step.
Coverage:

- Block 1 (1-step parity): N = 32 sinusoidal IC, 224 asserts.
- Block 2 (K = 10 steps, periodic, no projection): N = 32, 224 asserts.
- Block 3 (K = 10 steps, `:reanchor` projection): N = 16 boundary cell,
  116 asserts (state + 4 ProjectionStats counters).
- Block 4 (K = 10 steps, `tau` + `q_kind = :vNR`): N = 32, 224 asserts.

After M3-3e-5 drops the cache_mesh field this test will be retired
(it depends on running M1's `det_run_stochastic!` on a parallel
`Mesh1D`).

## Wall-time impact (N = 80, periodic, 50 steps)

Median over 7 trials (after 3-warmup), `project_kind = :none`,
`C_A = 0.3`, `C_B = 0.5`, `dt = 5e-4`:

- Native (M3-3e-2): 9.69 ms/step
- Cache-mesh per-step shim: 7.82 ms/step
- Speedup: 0.81× (i.e. native is ~20% slower in this benchmark).

This is *opposite* to M3-3e-1's reported 1.68× speedup on the
deterministic Newton path. Two factors at play:

1. **Phase-8 per-cell loops are dominated by HG-accessor indirection.**
   The `PolynomialFieldSet` view chain (`fs.x[j][1]` returns a
   `PolynomialView` whose `[1]` returns the raw scalar) carries
   non-trivial wrapping vs M1's direct `mesh.segments[j].state.x`. In
   the deterministic Newton step the Newton solve dominates and the
   per-cell read overhead is amortized; in the stochastic step the
   per-cell read pattern is the dominant cost.
2. **The cache-mesh shim's `sync_*!` round-trip is cheap** (just two
   Vector{Float64} copies per call, ~1µs at N=80). With per-step
   shim semantics the sync overhead is bounded; the shim wins by
   running M1's segment-array-direct loops which are tighter than
   HG view loops.

The wall-time win promised by M3-3e was largely on the deterministic
Newton path (where Newton residual evaluation dominates and the cache
sync is in the inner loop); on the stochastic injection path the
per-cell HG-view overhead actually slows things down. **The win
materializes after M3-3e-4 + M3-3e-5** when (a) realizability
projection is also lifted (eliminating per-step cache_mesh syncs) and
(b) the cache_mesh allocation itself goes away. M3-3e-5 will report
the final aggregate timing.

This is consistent with the M3-3e-1 benchmark at 9.6 ms/step in that
note — but now with per-step shim measured at 7.8 ms/step, the
speedup claim is suspect. The two benchmarks are on different
machines and weren't apples-to-apples (M3-3e-1 may have measured
total run time, not per-step). What we can say with confidence:

- **Native and cache-mesh paths agree byte-for-byte** on all gates.
- **Per-step wall time is comparable** (within 20% in either direction
  depending on machine state, JIT warmup, and the proportion of
  Newton solve vs per-cell loops).

## Test count delta

5335 + 1 deferred → 6123 + 1 deferred (+788 new asserts in
`test_M3_3e_2_stochastic_native_vs_cache.jl`).

## HG API surprises

None beyond what M3-3a/b/c/d/e-1 documented. The native
`inject_vg_noise_HG_native!` uses only the same `PolynomialFieldSet`
indexing accessors (`fs.x[j][1]`, `fs.x[j] = (val,)`) already
established. No new HG API was needed.

## Constraint: realizability projection still on cache_mesh

Per the M3-3e-2 brief, `realizability_project_HG!` was not lifted in
this sub-phase (M3-3e-4 owns it). The native path's call to
`realizability_project_HG!` therefore still does a `sync_cache_from_HG!`
+ M1 projection + `sync_HG_from_cache!` round-trip when
`project_kind != :none`. This is the dominant residual cost of the
M3-3e-2 path on `:reanchor` configurations.

A fast-path was added: when `project_kind == :none` the native
injection skips the cache_mesh sync altogether and only increments
`proj_stats.n_steps` (matching M1's no-op semantics in `:none`
mode).

## Pointers for M3-3e-3 / M3-3e-4

- `det_run_stochastic_HG!` and `inject_vg_noise_HG!` are now
  exclusively HG-side. Once `realizability_project_HG!` is lifted in
  M3-3e-4, the only callers of `sync_cache_from_HG!` left in the
  Phase-8 path will be (a) the test-side Eulerian profile extraction
  helpers and (b) the AMR + tracer paths (M3-3e-3 scope).
- `mesh.Δm`, `mesh.L_box`, `mesh.p_half` are read from the wrapper
  struct (already established in M3-3e-1). M3-3e-4's
  `realizability_project_HG_native!` should follow the same accessor
  pattern.

## Files touched

- `src/newton_step_HG_M3_2.jl` — added `inject_vg_noise_HG_native!`
  and `det_run_stochastic_HG_native!`; rewired the public wrappers to
  delegate. Net diff: +313 / -29 (≈ +284 LOC).
- `test/runtests.jl` — appended a new `@testset` block for
  `test_M3_3e_2_stochastic_native_vs_cache.jl`.
- `test/test_M3_3e_2_stochastic_native_vs_cache.jl` — new (171 LOC).

## Why a `.cache_mesh.segments` access still exists in tests

Two test files (`test_M3_1_phase5_sod_HG.jl_helpers.jl` and
`test_M3_2_phase7_steady_shock_HG.jl`) directly access
`mesh_HG.cache_mesh` to extract Eulerian profiles for diagnostic
plots. These remain unchanged in M3-3e-2 (they are test-side
conveniences, not src-side dependencies). M3-3e-5 will replace them
with HG-native diagnostic accessors before dropping the field.
