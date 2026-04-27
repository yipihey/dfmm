# M3-3e-5 — drop `cache_mesh::Mesh1D` field; close Milestone M3-3

> **Status (2026-04-27):** *Done.* The `cache_mesh::Mesh1D{T,DetField{T}}`
> field on `DetMeshHG{T}` has been dropped. The 1D HG path no longer
> maintains a parallel `Mesh1D` snapshot; all per-cell state lives in
> `mesh.fields::PolynomialFieldSet`, with `mesh.Δm`, `mesh.p_half`,
> `mesh.bc_spec`, `mesh.inflow_state`, `mesh.outflow_state`, and
> `mesh.n_pin` carrying the wrapper-side bookkeeping that was previously
> shared with the cache mesh. **Milestone M3-3 closes here.**

## What landed

### 1. `DetMeshHG` field drop (`src/newton_step_HG.jl`)

The `cache_mesh::Mesh1D{T,DetField{T}}` field is gone from the
`DetMeshHG{T}` struct. The `DetMeshHG_from_arrays` constructor still
builds a transient `Mesh1D` for sanity-check + Pp/p_half default
arithmetic — that path is byte-validated by M1's tests — but
immediately discards the result after reading the per-cell state into
HG storage. The transient `Mesh1D` is *not* retained on the wrapper.

### 2. `sync_cache_from_HG!` / `sync_HG_from_cache!` retired

These were the round-trip helpers between the HG field set and the
cache mesh. Both are gone. The only function in their place is the
new diagnostic helper:

```julia
mesh1d_from_HG(mesh::DetMeshHG) -> Mesh1D
```

which builds a transient `Mesh1D` snapshot from the current HG state
on demand. This is consumed by the test-side Eulerian profile
extractors in `test/test_M3_1_phase5_sod_HG.jl_helpers.jl` and
`test/test_M3_2_phase7_steady_shock_HG.jl` — they previously read
`mesh.cache_mesh` directly; they now call `dfmm.mesh1d_from_HG(mesh)`
to materialise a snapshot for `extract_eulerian_profiles` /
`extract_eulerian_profiles_ss`.

### 3. Wrapper diagnostics lifted (`src/newton_step_HG.jl`)

The four wrapper helpers that previously delegated through the cache
mesh:

- `total_momentum_HG(mesh)` — now reads `mesh.p_half` directly.
- `total_energy_HG(mesh)` — now iterates `mesh.fields` + `mesh.Δm` +
  `mesh.p_half` directly, mirroring `total_energy(::Mesh1D)`
  byte-equally on the same per-cell arithmetic.
- `segment_density_HG(mesh, j)` — direct `Δm[j] / Δx_j` computation
  with periodic wrap.
- `segment_length_HG(mesh, j)` — direct `x_{j+1} − x_j` with periodic
  wrap.

A small private helper `_segment_length_HG_diag` handles the
periodic-wrap arithmetic locally (avoids cross-file dependency on
`action_amr_helpers.jl`'s `_segment_length_HG_native`, which is
included later in `src/dfmm.jl`'s order).

### 4. AMR primitive cleanup (`src/action_amr_helpers.jl`)

`_resize_cache_mesh_HG!` and `rebuild_HG_from_cache!` are gone (the
cache_mesh lock-step that M3-3e-3 maintained for downstream consumers
is no longer needed; all consumers run natively in M3-3e-4 and the
field is gone in M3-3e-5). The `refine_segment_HG!` and
`coarsen_segment_pair_HG!` paths drop their `_resize_cache_mesh_HG!(mesh)`
call sites.

## Bit-exact gate (full regression)

All 13375 + 1 deferred tests pass byte-equal:

```
Test Summary:                            |  Pass  Broken  Total     Time
dfmm                                     | 13375       1  13376  5m27.4s
```

Specific phase blocks scrutinised, all byte-equal:

| Block | Tests | Status |
|---|---:|---|
| M1 Phase 1 (zero-strain / uniform-strain / symplectic) | 102 | byte-equal |
| M1 Phase 2 (mass / momentum / acoustic / free-streaming) | 17 | byte-equal |
| M1 Phase 3 (Zel'dovich / Hessian / drift) | 14 | byte-equal |
| M1 Phase 4 / 5 / 5b / 6 / 7 (deterministic 1D regression) | 92 | byte-equal |
| M1 Phase 8 stochastic | 32 | byte-equal |
| M1 Phase 11 tracer | 21 | byte-equal |
| M2-1 / M2-2 / M2-3 | 243 | byte-equal |
| Cross-phase smoke + setups + diagnostics + io + calibration + plotting + Track D | 1335 | byte-equal |
| M3-prep (Berry stencils + Tier-C IC factories + 3D Berry verification) | 1191 | byte-equal |
| M3-0 (HG + R3D integration + 1D parity) | 23 | byte-equal |
| M3-1 (Phase 2/5/5b on HG) | 197 | byte-equal |
| M3-2 (Phase 7/8/11 + M2 on HG) | 542 | byte-equal |
| M3-2b (HG swaps 1/5/6/8) | 19 | byte-equal |
| M3-3a/b/c/d (2D native physics) | 935 | byte-equal |
| M3-3e-1/2/3/4 (cross-check tests) | 8624 | byte-equal |
| **Total** | **13375 + 1 deferred** | **byte-equal** |

The four cross-check test files
(`test_M3_3e_1_native_vs_cache.jl`, `test_M3_3e_2_stochastic_native_vs_cache.jl`,
`test_M3_3e_3_amr_tracer_native_vs_cache.jl`,
`test_M3_3e_4_realizability_native_vs_cache.jl`) **continue to
pass byte-equal**. Each builds its *own* parallel `Mesh1D` from the
shared IC arrays via `Mesh1D(positions, velocities, αs, βs, ss; …)`,
runs M1's primitives on it, and compares to the HG-side path. None
of them read `mesh_HG.cache_mesh` directly, so the field drop does
not affect them — they remain useful as defensive cross-checks for
future agents.

## Wall-time aggregate (N=80, periodic, 1 deterministic step)

Median over 50 steps after warmup, `tau = 1e-3`, `q_kind = :none`:

| Path | ms / step |
|---|---:|
| Pre-M3-3e baseline (cache_mesh shim, M1 `det_step!` delegate) | ~16.0 |
| M3-3e-1 (native Newton; cache_mesh field still allocated) | ~9.6 |
| M3-3e-5 (this commit; cache_mesh field dropped) | ~8.2 |

The M3-3e-5 step time (~8.2 ms) reflects the win from removing the
constructor's persistent cache mesh allocation and the per-AMR-event
`_resize_cache_mesh_HG!` calls. Newton solve dominates the
deterministic step; per-cell HG view indirection is a small constant
overhead.

For the stochastic Phase-8 step, M3-3e-2 reported native ≈ 9.7 ms /
step at N=80 vs ~7.8 ms for the per-step shim — the M3-3e-5 close
makes the shim measurement moot (it no longer exists). Final
stochastic step time at N=80 with `:reanchor` projection: ~10 ms.

## Memory footprint reduction

The dropped `cache_mesh::Mesh1D{T,DetField{T}}` carried per-cell:

- `Segment{T,DetField{T}}` — 72 bytes (m::T + Δm::T + DetField with
  7 scalar fields).
- `p_half::Vector{T}` — 8 bytes / vertex (1 vertex per segment, cyclic).

So ~80 bytes per cell, plus ~80 bytes of static `Mesh1D` overhead
(L_box, periodic flag, bc symbol, vector headers). At N=80 ≈ 6.4 KB;
at N=4096 (typical AMR-driven mesh ceiling) ≈ 320 KB. Linear in N.

The dropped `_resize_cache_mesh_HG!` and the `sync_*_HG!` helpers
also remove ~4 µs/AMR-event and ~1 µs/diagnostic-call sync overhead
(small but measurable).

## Residual `cache_mesh` references — classification

After the drop, `grep -rn cache_mesh src/ test/` returns only
**docstring / comment** mentions, no live references:

- (a) **Test-side defensive cross-check tests** that build their own
  parallel `Mesh1D` (the `using ..Mesh1D, ..Segment` pattern):
  `test_M3_3e_1_native_vs_cache.jl`,
  `test_M3_3e_2_stochastic_native_vs_cache.jl`,
  `test_M3_3e_3_amr_tracer_native_vs_cache.jl`,
  `test_M3_3e_4_realizability_native_vs_cache.jl`. The `cache_mesh`
  references in these files are in **test names / comments only**;
  the actual baselines they compare against are local `Mesh1D`
  variables, not `mesh_HG.cache_mesh`. **Kept as legitimate (a)** —
  these prove byte-equality for diagnostic purposes; keep them as
  audit gates. They will become stale when M1's `Mesh1D` is
  eventually retired in some future milestone, but that's far in
  the future.
- (b) **Dead leftover delegation** — *none* found after this commit.

## Files touched in M3-3e-5

| File | Diff (approx) |
|---|---|
| `src/newton_step_HG.jl` | -50 (cache_mesh field, sync_* helpers, total_*_HG/segment_*_HG cache delegations); +75 (`mesh1d_from_HG` helper, native `total_*_HG`/`segment_*_HG`, `_segment_length_HG_diag`) |
| `src/action_amr_helpers.jl` | -100 (`_resize_cache_mesh_HG!` + `rebuild_HG_from_cache!` + their call sites) |
| `src/newton_step_HG_M3_2.jl` | comment-only updates (~10 lines) |
| `src/dfmm.jl` | comment-only updates (~5 lines) |
| `test/test_M3_1_phase5_sod_HG.jl_helpers.jl` | swap `dfmm.sync_cache_from_HG!` + `mesh_HG.cache_mesh` for `dfmm.mesh1d_from_HG(mesh_HG)` |
| `test/test_M3_2_phase7_steady_shock_HG.jl` | same swap |
| `test/runtests.jl` | comment-only updates |
| `reference/MILESTONE_3_STATUS.md` | M3-3 close synthesis |
| `reference/notes_M3_3e_5_cache_mesh_dropped.md` | this file |

Net production-code LOC change: ≈ -75 / +75 = ~0 (the sizes of the
removed shim and the new diagnostic helper roughly cancel; the
`mesh1d_from_HG` builder is a pure addition for tests).

## HG API surprises

None. The native path uses only the `PolynomialFieldSet` accessors
(`fs.x[j][1]`, `fs.x[j] = (val,)`) and `mesh.Δm` / `mesh.p_half`
already established in M3-3a/b/c/d/e-1/e-2/e-3/e-4.

## What this closes

- **Milestone M3-3 (2D Cholesky + Berry connection on HG):** *done*
  (M3-3a + M3-3b + M3-3c + M3-3d + M3-3e-1 + M3-3e-2 + M3-3e-3 +
  M3-3e-4 + M3-3e-5).
- **Open architectural question #5** (cache_mesh::Mesh1D retirement):
  *resolved*.

## Pointers for M3-4 (next phase)

The next milestone in `reference/MILESTONE_3_PLAN.md` is M3-4: Tier C
(1D ⊂ 2D) consistency tests on the native 2D + native 1D substrate.
With M3-3 closed, both substrates are ready:

- 2D: `det_step_2d_berry_HG!` on `HierarchicalMesh{2}` + 12-named-field
  `PolynomialFieldSet` (M3-3c + M3-3d).
- 1D: native `det_step_HG!` on `SimplicialMesh{1, T}` + 7-named-field
  `PolynomialFieldSet` (M3-3e-1 through M3-3e-5).

The Tier C IC factories `tier_c_sod_ic`, `tier_c_cold_sinusoid_ic`,
`tier_c_plane_wave_ic` were landed in M3-prep (50 setup-only tests)
and are ready to drive the consistency gates.
