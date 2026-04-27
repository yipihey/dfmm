# M3-3e-3 — native 1D AMR refine/coarsen + standalone `TracerMeshHG`

> **Status (2026-04-26):** *Done.* The 1D AMR primitives
> (`refine_segment_HG!`, `coarsen_segment_pair_HG!`,
> `action_error_indicator_HG`, `gradient_indicator_HG`,
> `amr_step_HG!`) and the `TracerMeshHG` storage no longer touch
> `mesh.cache_mesh` for the *hot* arithmetic. Per the M3-3e
> investigation note's Option B (hand-rolled 1D-Lagrangian
> primitives on HG storage), the lift mirrors M1's
> `src/amr_1d.jl` byte-for-byte against the
> `(SimplicialMesh{1, T}, PolynomialFieldSet, Δm::Vector,
> p_half::Vector)` substrate.
>
> The `cache_mesh::Mesh1D{T,DetField{T}}` field stays allocated on
> `DetMeshHG` (M3-3e-4 still consumes it for realizability
> projection; downstream helpers
> `total_momentum_HG`, `total_energy_HG`, `segment_density_HG`,
> `segment_length_HG` still read from it; test-side Eulerian
> profile extraction in `test_M3_1_phase5_sod_HG.jl_helpers.jl` and
> `test_M3_2_phase7_steady_shock_HG.jl` reads `mesh.cache_mesh`
> directly). M3-3e-3 keeps the cache mesh in lock-step with the HG
> side after each AMR event via a new `_resize_cache_mesh_HG!`
> helper so those consumers continue to work. M3-3e-5 will drop
> the field once all paths are native.

## What landed

### Native AMR primitives (`src/action_amr_helpers.jl`)

- **`refine_segment_HG!(mesh, j; tracers)`** — splits simplex `j`
  into two equal-mass daughters at the mid-mass-coordinate.
  Operates directly on `mesh.fields::PolynomialFieldSet`,
  `mesh.Δm`, `mesh.p_half`, and `mesh.mesh::SimplicialMesh{1, T}`.
  Per-cell scalar arithmetic mirrors M1's `refine_segment!`
  byte-equally: parent state read via
  `read_detfield(fields, j)`, daughters written via
  `write_detfield!`, mid-vertex velocity is the linear
  interpolation `(parent.u + u_right) / 2`. Field-set storage is
  reallocated to `N + 1` cells (no in-place insert API on
  `PolynomialFieldSet`); per-event cost is dominated by the
  reallocation + rewrite.

- **`coarsen_segment_pair_HG!(mesh, j; tracers, Gamma)`** — merges
  simplices `j` and `j+1` (cyclic) into one. Reproduces M1's
  law-of-total-covariance arithmetic identically:
  - Mass: bit-exact (`Δm_new = Δm_l + Δm_r`).
  - Momentum: exact via the vertex-velocity redistribution
    `u_left_new = (m̄_left_pre · u_left_pre + (Δm_r/2)·u_seam_pre) / m̄_left_post`.
  - `(α, β)`: mass-weighted + the bilinear between-cell variance
    `(Δm_l Δm_r / Δm_new²)·(u_cell_l - u_cell_r)²` lifts `M_vv_new`
    so γ stays real-valued.
  - Entropy: re-anchored from `M_vv_new` via
    `s_new = c_v · (log(M_vv_new) - (1-Γ)·log(J_new))`.
  - `Pp`, `Q`: mass-weighted average.
  Cache mesh shrinkage is mirrored on the HG side (deleteat-style
  via field-set reallocation); the wrap case `j == N` (j_next == 1)
  is handled identically to M1.

- **`action_error_indicator_HG(mesh; dt, Gamma)`** — per-segment
  `|d²α| + |d²β| + |d²s| + |d²u|/c_s + 0.01·γ_marker` evaluated
  on HG storage. Same formula as M1's `action_error_indicator`,
  same boundary handling (periodic wrap; one-sided / zero on
  inflow/outflow boundaries).

- **`gradient_indicator_HG(mesh; field, Gamma)`** — `:rho`, `:u`,
  `:P` per-segment relative-gradient indicators on HG storage.

- **`amr_step_HG!(mesh, indicator, τ_refine, τ_coarsen; tracers,
  max_segments, min_segments)`** — driver loop mirrors M1's
  `amr_step!`. Coarsens first (right-to-left, no-overlap), then
  refines (right-to-left) when no coarsens fired. Hysteresis
  enforced (`τ_coarsen ≤ τ_refine/4`).

### Standalone `TracerMeshHG` storage (`src/newton_step_HG_M3_2.jl`)

Pre-M3-3e-3 `TracerMeshHG` wrapped an M1
`TracerMesh{T, Mesh1D{T, DetField{T}}}` whose `fluid` was
`mesh.cache_mesh` — storage was *aliased* with the cache mesh.
M3-3e-3 introduces a freestanding `_TracerStorageHG{T}` mutable
struct holding `tracers::Matrix{T}` and `names::Vector{Symbol}`,
preserving the test-side `tm.tm.tracers` / `tm.tm.names` accessors
that `test_M3_2_phase11_tracer_HG.jl` and
`test_M3_2_M2_2_multitracer_HG.jl` use for bit-exact parity gates.

Native methods:
- `TracerMeshHG(fluid::DetMeshHG; n_tracers, names)` — allocates
  the matrix from `length(fluid.Δm)` directly; no cache_mesh sync.
- `set_tracer!(tm, k_or_name, values)` — per-cell write.
- `set_tracer!(tm, k_or_name, f::Function)` — evaluates `f(m_j)`
  at each segment's mass-coordinate cell center, derived from
  cumulative `fluid.Δm`. Mirrors M1's
  `set_tracer!(::TracerMesh, ::Function)` byte-equally — the
  `m_center = cum + Δm[j]/2` derivation is identical to M1's
  `Mesh1D` constructor.
- `add_tracer!(tm, name, values_or_f)` — non-mutating extension
  (returns a new wrapper). Same shape as M1.
- `tracer_at_position(tm, x, k)` — walks the HG field set's
  `fields.x[j]` cyclically; piecewise-constant per-segment value.
- `advect_tracers_HG!(tm, dt)` — pure no-op (matches Phase-11
  semantics in pure-Lagrangian regions; no
  `advect_tracers!(tm.tm, dt)` delegation needed because the inner
  storage is no longer a TracerMesh).

### AMR refine/coarsen tracer helpers

`_refine_tracer_HG!(tracers, j)` and
`_coarsen_tracer_HG!(tracers, j; Δm_l, Δm_r)` operate on the
`_TracerStorageHG{T}` matrix directly. Mirror M1's
`refine_tracer!` / `coarsen_tracer!` byte-equally:
- Refine: bit-exact daughter copy of column `j` into `j+1`.
- Coarsen: mass-weighted merge `(Δm_l · t_l + Δm_r · t_r) / Δm_new`
  with cyclic wrap on `j == N`.

### Cache-mesh lock-step (`_resize_cache_mesh_HG!`)

After each AMR event, `_resize_cache_mesh_HG!` resizes
`mesh.cache_mesh.segments` and `mesh.cache_mesh.p_half` to the new
`N` and writes per-cell HG state into the cache mesh. This keeps
`total_momentum_HG`, `total_energy_HG`, `segment_density_HG`,
`segment_length_HG`, `realizability_project_HG!`, and the test-side
`mesh.cache_mesh` accessors functional during M3-3e-3. M3-3e-4 lifts
realizability projection; M3-3e-5 drops the cache_mesh field
entirely and replaces the test-side accessors with HG-native
diagnostic helpers.

## Bit-exact gates

All AMR + tracer tests pass at byte-equality with the prior
cache-mesh-driven path:

- `test_M3_2_M2_1_amr_HG.jl` — 82 asserts. Conservation
  (mass / momentum exact through refine/coarsen), indicator parity
  vs M1, hysteresis, and full per-segment bit-exact `(x, u, α, β,
  s, Δm)` parity in the post-refine state.
- `test_M3_2_phase11_tracer_HG.jl` — 19 asserts. 1000-step
  shock-survival bit-exactness, M1 ↔ HG tracer-matrix parity over
  200 steps, tracer API delegation smoke.
- `test_M3_2_M2_2_multitracer_HG.jl` — 5 asserts. 150-step
  stochastic-injection bit-exactness, M1 ↔ HG state + tracer parity
  over 50 stochastic steps.

Plus a new defensive cross-check
`test_M3_3e_3_amr_tracer_native_vs_cache.jl` (5784 byte-equality
asserts):

- Block 1 (1730 asserts): K = 10 refine events at varying j on
  N = 16 periodic Sod-style IC; per-cell `(x, u, α, β, s, Pp, Q,
  Δm)` byte-equal to M1 after each event.
- Block 2 (2130 asserts): K = 10 coarsen events at varying j on
  N = 32 acoustic IC; same per-cell byte-equality.
- Block 3 (1065 asserts): 5 (det_step + amr_step) cycles on a
  Sod IC; indicator + amr_step results + per-cell state all
  byte-equal to running M1's `det_step!` + `amr_step!` on a
  parallel `Mesh1D`.
- Block 4 (859 asserts): 6 mixed refine/coarsen events with 3
  tracers (step, sin, gauss); tracer matrices byte-equal to M1's
  at every step (including the mass-weighted-merge events).

After M3-3e-5 drops the cache_mesh field, this test will be
retired (it depends on running M1's `refine_segment!` /
`coarsen_segment_pair!` on a parallel `Mesh1D`).

## Conservation through refine/coarsen

All conservation properties hold byte-equal:
- **Mass**: `==` (Σ Δm invariant; `Δm_new = Δm_l + Δm_r` exact).
- **Momentum**: `isapprox(_, _; atol = 1e-14)` per the M2-1 test
  contract; the redistributed-vertex-velocity arithmetic matches
  M1's character-for-character so the residual is the same
  ULP-level floating-point noise on both paths.
- **Energy** (KE-internal split): inherited from M1's law-of-total-
  covariance; `s_new = c_v · (log M_vv_new - (1-Γ)·log J_new)`
  re-anchors entropy so the merged `P_xx = ρ · M_vv_new` is
  self-consistent. Byte-equal to M1.
- **Tracers**: bit-exact daughter copy on refine; mass-weighted
  average on coarsen — both byte-equal to M1's `_replace_tracer_matrix!`.

## Wall-time impact

Per-AMR-event, on a 64-cell periodic mesh:

- `refine_segment_HG!` (native): ~106 µs / event
- `refine_segment!` (M1 baseline on `Mesh1D`): ~0.45 µs / event
- `coarsen_segment_pair_HG!` (native): ~151 µs / event

The native HG path is ~230× slower per AMR event than M1, dominated
by the `allocate_detfield_HG(N+1)` call (12 named-field
reallocation) plus the `_rebuild_simplicial_mesh_HG!` call (new
`SimplicialMesh{1, T}` allocation). M1's `splice!` / `insert!` on
the segments vector mutates 7 scalar fields per cell in place,
which is much cheaper.

This is **not** a regression vs the pre-M3-3e-3 cache-mesh-driven
shim — that path also did `sync_cache_from_HG!` +
`refine_segment!(cache_mesh, j)` + `rebuild_HG_from_cache!`, which
includes the same field-set + simplicial-mesh reallocation. The
M1 vs HG gap reflects an inherent O(1) (M1) vs O(N · n_fields)
(HG) per-event cost — accepted by the M3-3e brief because *AMR
events are rare relative to timesteps*. At a typical
`det_step_HG!` cost of 10 ms/step and 0–2 AMR events per step, the
AMR overhead is ≤3 % of a step, well within the brief's
"negligible against the AMR work itself" guidance.

## HG API surprises

None beyond what M3-3a/b/c/d/e-1/e-2 documented. The native AMR
path uses only the same `PolynomialFieldSet` allocator
(`HierarchicalGrids.allocate_polynomial_fields` via
`allocate_detfield_HG`) and the same `read_detfield` /
`write_detfield!` accessors already established. No new HG API
was needed.

PolynomialFieldSet does **not** support in-place insert/delete,
so refine/coarsen events reallocate the field set. This is the
source of the per-event O(N · n_fields) cost noted above; an HG
upstream feature for in-place per-cell insert would change this
trade-off. Marked as a candidate for the HG design-guidance
backlog.

## Test count delta

6883 + 1 deferred → 12667 + 1 deferred (+5784 new asserts in
`test_M3_3e_3_amr_tracer_native_vs_cache.jl`). The +5784 includes
~3*N*(state-fields + Δm + p_half) counts per refine/coarsen event,
multiplied across the four cross-check blocks.

## Files touched

- `src/action_amr_helpers.jl` — full rewrite of the 1D AMR
  primitives + indicators + driver + tracer helpers + the
  `_resize_cache_mesh_HG!` lock-step helper. Net diff: +402 / -82
  (≈ +320 LOC).
- `src/newton_step_HG_M3_2.jl` — replaced the M1-`TracerMesh`-
  wrapping `TracerMeshHG` with a standalone `_TracerStorageHG`-
  backed implementation. New native `set_tracer!` /
  `add_tracer!` / `tracer_at_position` / `tracer_index` /
  `n_tracer_*` methods. Net diff: +131 / -57 (≈ +74 LOC).
- `test/runtests.jl` — appended a new `@testset` block for
  `test_M3_3e_3_amr_tracer_native_vs_cache.jl`.
- `test/test_M3_3e_3_amr_tracer_native_vs_cache.jl` — new (~190
  LOC), 5784 byte-equality asserts.

## Pointers for M3-3e-4 / M3-3e-5

- `realizability_project_HG!` is the only remaining src-side
  consumer of `mesh.cache_mesh` for hot-path arithmetic. M3-3e-4
  should lift it to operate directly on `mesh.fields` per the
  M3-3e-1 + M3-3e-2 patterns. The fast-path for
  `project_kind = :none` is already in place; the lift is the
  `:reanchor` branch.
- Once M3-3e-4 lands, the `_resize_cache_mesh_HG!` helper, the
  `sync_cache_from_HG!` / `sync_HG_from_cache!` helpers, the
  `total_momentum_HG` / `total_energy_HG` / `segment_density_HG` /
  `segment_length_HG` cache-mesh delegations, and the test-side
  `mesh.cache_mesh` accessors (in
  `test_M3_1_phase5_sod_HG.jl_helpers.jl` and
  `test_M3_2_phase7_steady_shock_HG.jl`) all need replacement
  before M3-3e-5 can drop the field.
- The `rebuild_HG_from_cache!` legacy helper is retained in
  `src/action_amr_helpers.jl` but unused by the AMR path. It
  remains for diagnostic / migration use only and should be removed
  with the cache_mesh field in M3-3e-5.
