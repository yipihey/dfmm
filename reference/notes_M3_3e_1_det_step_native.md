# M3-3e-1 — native `det_step_HG!` deterministic Newton path

> **Status (2026-04-26):** *Done.* The deterministic Newton path in
> `det_step_HG!` no longer touches `mesh.cache_mesh`. The
> cache_mesh field stays allocated on `DetMeshHG` (M3-3e-2/3/4 still
> consume it) and `sync_cache_from_HG!` / `sync_HG_from_cache!`
> remain available for test-side diagnostics. M3-3e-5 will drop the
> field once all paths are native.

## What landed

A new internal entry point `det_step_HG_native!` in
`src/newton_step_HG.jl` reproduces M1's `det_step!` line-for-line
on the physics, but reads/writes the per-cell state through
`fields::PolynomialFieldSet` (via `fs.alpha[j]`, `fs.beta[j]`, …)
instead of `mesh.segments[j].state`. The public
`det_step_HG!` is now a thin pass-through to the native function.

Sub-blocks (per §A of `notes_M3_3e_cache_mesh_retirement.md`):

1. **Pre-Newton bookkeeping** — pack flat `y_n`, `s_vec`, cache
   pre-Newton `ρ_n_vec` and `Pp_n_vec` directly from `fields`.
2. **Newton solve** — call `det_el_residual` (already pure
   flat-array) via `NonlinearSolve.NewtonRaphson`. Same
   `det_jac_sparsity` and same kwarg defaults as M1.
3. **Unpack `(x, u, α, β)`** — write the Newton solution back to
   `fields`. Refresh `mesh.p_half[i] = m̄_i u_i` per M1's
   `unpack_state!`.
4. **Phase 5b q-dissipation entropy update** — when
   `q_kind = :vNR_linear_quadratic`, recompute midpoint divu and q
   from `(y_n, post-Newton fields)`, then write `s_pre + Δs_q` back
   into `fs.s[j]`.
5. **Phase 5 BGK joint relaxation** — when `tau` is supplied,
   transport `Pp` Lagrangian, BGK-relax `(P_xx, P_⊥)`, re-anchor
   entropy via `Δs/c_v = log(P_xx_new/P_xx_pre)`, decay β by
   `exp(-Δt/τ)` with realizability clip, and decay Q the same way.
6. **Phase 7 inflow/outflow pinning** — overwrite leftmost `n_pin`
   segments with the upstream Dirichlet state (cumulative-Δx vertex
   chain anchored at `x = 0`); overwrite rightmost `n_pin` segments
   with the downstream state (anchored at `x = L_box`). Zero-gradient
   outflow fallback mirrors M1 exactly.

## Bit-exact gates

All deterministic-Newton tests remain at byte-equality with the
prior cache-mesh-driven path:

- `test_M3_1_phase2_*_HG.jl` (Phase 2 mass / momentum / acoustic /
  free-streaming) — 17 + 8 ≈ 17 asserts.
- `test_M3_1_phase5_sod_HG.jl` (Phase 5 Sod regression) — 12 asserts.
- `test_M3_1_phase5b_*.jl` (Phase 5b q-dissipation: q-formula
  unit, q_none bit-equality, q_vNR Sod) — 144 + 9 + 15 = 168
  asserts.
- `test_M3_2_phase7_steady_shock_HG.jl` (Phase 7 steady shock + Q) —
  17 asserts.
- `test_M3_2b_swap8_bckind_HG.jl` (BC framework path) — 12 asserts.
- All M3-3a/b/c/d 2D tests (no 1D-path dependency) — unchanged.

Plus a new defensive cross-check test
`test_M3_3e_1_native_vs_cache.jl` (1344 byte-equality asserts)
explicitly verifies that running `det_step_HG_native!` on a
`DetMeshHG` and `det_step!` on a parallel `Mesh1D` IC produces
byte-equal `(x, u, α, β, s, Pp, Q)` and `p_half`. Coverage:

- Block 2 (Newton solve / periodic): N = 32 acoustic IC, 5 steps —
  256 asserts.
- Block 3 (Phase 5b q-dissipation): N = 64 Sod IC + q_vNR, 3 steps —
  512 asserts.
- Block 4 (Phase 5 BGK joint relaxation): N = 32 isotropic IC + τ,
  Q = 0.01·cos(2π…), 5 steps — 256 asserts.
- Block 5 (Phase 7 inflow/outflow vertex chain): N = 40 steady-shock
  IC + τ + q_vNR, 4 steps — 320 asserts.

## Wall-time impact (N = 80, periodic, 1 step)

Manual 50-step loop, post-warmup:

- Native (M3-3e-1):       9.6 ms/step
- Cache-mesh-driven shim: 16.1 ms/step
- Speedup: ~1.68× (40% faster), saving the
  `sync_cache_from_HG!` / `sync_HG_from_cache!` round-trip on every
  step.

## Test count delta

3991 + 1 deferred → 5335 + 1 deferred (+1344 new asserts in
`test_M3_3e_1_native_vs_cache.jl`).

## HG API surprises

None. The native path uses only the same `PolynomialFieldSet`
indexing accessors (`fs.x[j][1]`, `fs.x[j] = (val,)`) already
established in M3-3a/b/c/d for the 2D-path read/write pattern. No
new HG API was needed.

## Phase 5 BGK joint relaxation

Confirmed bit-equal: the only change vs M1's
`mesh.segments[j].state.{β, s, Pp, Q} = …` is the storage target.
The per-segment scalar arithmetic (transport, `bgk_relax_pressures`,
re-anchor, `decay`, β realizability clip, Q transport+decay) is
character-for-character identical.

## Phase 7 inflow/outflow vertex chain

Confirmed bit-equal: the cumulative-Δx forward chain (inflow) walks
`x_left_pinned = (j == 1) ? 0 : cum_Δx_in` then increments
`cum_Δx_in += Δm_j / ρ_in`, and the leftward chain (outflow)
walks `cum_Δx += Δm_j / ρ_out` then sets
`x_left_pinned = L_box - cum_Δx`. Both match M1 exactly. The
zero-gradient outflow fallback uses `prev_x = fs.x[j-1][1]` after
the per-iteration overwrite, matching M1's
`mesh.segments[j-1].state.x` read pattern.

## Dead-code residue

The cache-mesh-driven body of `det_step_HG!` is gone (the function
now just delegates to `det_step_HG_native!`). The
`sync_cache_from_HG!` / `sync_HG_from_cache!` helpers and the
`cache_mesh::Mesh1D` field on `DetMeshHG` remain for:

- M3-3e-2 stochastic injection (`inject_vg_noise_HG!`,
  `det_run_stochastic_HG!`) — still delegates via cache_mesh.
- M3-3e-3 AMR + tracers (`refine_segment_HG!`,
  `coarsen_segment_pair_HG!`, `TracerMeshHG`) — still delegates.
- M3-3e-4 realizability projection
  (`realizability_project_HG!`) — still delegates.
- Test-side Eulerian profile extraction
  (`test_M3_1_phase5_sod_HG.jl_helpers.jl`,
  `test_M3_2_phase7_steady_shock_HG.jl`) — calls
  `dfmm.sync_cache_from_HG!(mesh_HG)` then reads
  `mesh_HG.cache_mesh`.

After M3-3e-5 drops the field, the new defensive test
`test_M3_3e_1_native_vs_cache.jl` will be retired (it directly
runs the M1 baseline on a parallel `Mesh1D`).

## Pointers for M3-3e-2

The native `det_step_HG_native!` reads `mesh.Δm`, `mesh.L_box`,
`mesh.bc_spec`, `mesh.inflow_state`, `mesh.outflow_state`,
`mesh.n_pin`, and `mesh.p_half` — all already on the wrapper
struct. The Phase-8 stochastic injection lift can therefore reuse
these without further wrapper-struct surgery; the only new wiring
needed is the `(seed, generator)` plumbing that mirrors M1's
`inject_vg_noise!` 1:N segment iteration order. Bit-exact gate is
the per-cell `InjectionDiagnostics` parity on
`det_run_stochastic_HG!` (~260 asserts) — see
`test_M3_2_phase8_stochastic_HG.jl`.
