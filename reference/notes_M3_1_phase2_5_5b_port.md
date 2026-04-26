# Phase M3-1 — Phase 2 + 5 + 5b ports onto HG (1D)

**Date:** 2026-04-25.
**Branch:** `m3-1-phase2-5-5b-port`.
**Status:** complete; bit-exact parity to M1 verified across all three
sub-phases.

This note is the implementation log for Phase M3-1 of
`reference/MILESTONE_3_PLAN.md`. The phase ports M1's Phase 2 (bulk
position `x`, velocity `u`, entropy `s` on a multi-segment periodic
Lagrangian-mass mesh), Phase 5 (deviatoric `P_⊥` via post-Newton BGK
relaxation), and Phase 5b (opt-in Kuropatenko / vNR tensor-q
artificial viscosity) onto the HierarchicalGrids (HG) substrate
established by M3-0. Same physics, HG mesh.

The bit-exact-parity contract is preserved: every per-cell state on
the HG path agrees with M1's `Mesh1D{T,DetField{T}}` path to **0.0
absolute** at every timestep across every test case, well below the
brief's 5e-13 target.

## Design decisions

### 1. HG storage container: `DetMeshHG{T}` (mirrors M3-0's pattern)

The HG-side state is bundled in a `DetMeshHG{T}` wrapper struct:

```julia
mutable struct DetMeshHG{T<:Real}
    mesh::SimplicialMesh{1, T}        # HG simplicial mesh on mass coord
    fields::PolynomialFieldSet         # (alpha, beta, x, u, s, Pp, Q)
    Δm::Vector{T}                      # per-segment Lagrangian mass
    L_box::T                           # periodic box length
    p_half::Vector{T}                  # vertex half-step momenta
    bc::Symbol                         # :periodic (Phase 2/5/5b)
    cache_mesh::Mesh1D{T,DetField{T}}  # transient M1 mesh, re-used
end
```

The `cache_mesh` field is the bit-equality lever: rather than
re-implementing M1's `det_step!` natively on HG iteration, the M3-1
driver round-trips state through this transient `Mesh1D` and calls
M1's tested `det_step!` byte-identically. This is the same pattern
M3-0 used for `cholesky_step_HG!` (which delegates to `cholesky_step`).

The `Δm`, `L_box`, `p_half`, and `bc` fields ride on the wrapper
because HG's `SimplicialMesh{1, T}` has no native concept of
Lagrangian-mass labelling, periodic-box length, or boundary
conditions. M3-2's M2-1 action-AMR port will need conservative
remap of `Δm` through `refine_segment!` / `coarsen_segment_pair!`
events — the HG `refine_by_indicator!` callback lives on the
wrapper, not on the simplicial mesh.

### 2. Periodic-BC neighbour wiring on `SimplicialMesh{1, T}`

HG's `SimplicialMesh` constructor takes a 2-row simplex-neighbour
matrix `sn::Matrix{Int32}` where `sn[k, j]` is the index of the
simplex sharing the face opposite vertex `k` of simplex `j` (or `0`
for a boundary face). The M3-0 `uniform_simplicial_mesh_1D` set
boundary entries to `0`; the M3-1 `periodic_simplicial_mesh_1D`
links the trailing edge cyclically:

```julia
sn[1, j] = j > 1 ? Int32(j - 1) : Int32(N)   # cyclic wrap left
sn[2, j] = j < N ? Int32(j + 1) : Int32(1)   # cyclic wrap right
```

This is a pure-Julia helper in `dfmm`; HG itself does not yet ship a
`periodic!(mesh)` builder (per the M3-0 design-guidance handoff item
#2). The helper costs nothing — it just fills two integer entries —
and matches the convention M3-3 will need for the 2D periodic
quadtree neighbour wiring.

### 3. Sparse Jacobian: deferred to M3-2

The brief flags HG-side sparsity wiring (`cell_adjacency_sparsity`,
design-guidance item #7) as a port target. We defer this to M3-2
because the M1 `det_step!` already builds and uses the tri-block-banded
sparsity pattern via `det_jac_sparsity(N)`, and the round-trip-to-
`cache_mesh` design re-uses that exact sparsity pattern by
construction. When M3-2 retires the legacy `Mesh1D` and the EL
residual moves natively to HG, the HG-side sparsity will need a
helper; until then the M1 sparsity pattern is bit-exact reused.

### 4. Field-set layout: `(alpha, beta, x, u, s, Pp, Q)`

The HG `PolynomialFieldSet` carries seven Float64 scalar fields,
allocated with `MonomialBasis{1, 0}()` (one coefficient per cell,
piecewise constant) and named per the M1 `DetField{T}` layout. The
field names match the public M1 API so callers familiar with the
1D code can navigate the HG field set by name.

The vertex-staggered `(x, u)` are stored as per-cell quantities (the
left vertex's `x` and `u`) — exactly the M1 `Segment{T,DetField{T}}`
convention. This avoids introducing a separate per-vertex field set
in M3-1 (which would diverge from M1's storage layout); M3-2 will
revisit when the HG-native EL residual moves off the legacy
`Mesh1D` cache.

### 5. `DetFieldND{D, T}` documentation type

Mirrors M3-0's `ChFieldND{D, T}` design: a thin wrapper exposing
`x::NTuple{D, T}`, `u::NTuple{D, T}`, `alphas::NTuple{D, T}`,
`betas::NTuple{D, T}`, plus scalar `s`, `Pp`, `Q`. In 1D this is
the M1 `DetField{T}` layout; in 2D it adds per-axis splitting plus
a Berry rotation angle (M3-3 scope); in 3D it adds three angles
(M3-7 scope). The type is **dispatch / documentation only** in
M3-1 — storage flows through the HG `PolynomialFieldSet`.

## File layout (new in M3-1)

| File | Role |
|---|---|
| `src/types.jl` (extension) | `DetFieldND{D, T}` dimension-generic full deterministic field type. Coexists with M1's `DetField{T}` until M3-2 verifies full M1+M2 parity. |
| `src/eom.jl` (extension) | `read_detfield(fields, j)` / `write_detfield!(fields, j, det)` — bit-preserving HG ↔ `DetField{T}` conversions per cell. |
| `src/newton_step_HG.jl` (extension) | `DetMeshHG{T}` wrapper struct, `DetMeshHG_from_arrays` factory, `periodic_simplicial_mesh_1D`, `allocate_detfield_HG`, `det_step_HG!`, `det_run_HG!`, plus diagnostics (`total_mass_HG`, `total_momentum_HG`, `total_energy_HG`, `segment_density_HG`, `segment_length_HG`). |
| `src/dfmm.jl` (extension) | New M3-1 export block. |
| `test/test_M3_1_phase2_mass_HG.jl` | Phase-2 mass conservation + bit-exact parity vs M1 on a 50-step periodic-mesh run. |
| `test/test_M3_1_phase2_momentum_HG.jl` | Phase-2 momentum conservation (zero + nonzero net momentum). |
| `test/test_M3_1_phase2_free_streaming_HG.jl` | Phase-2 cold-limit ballistic free-streaming. |
| `test/test_M3_1_phase2_acoustic_HG.jl` | Phase-2 linearised acoustic dispersion + 100-step bit-exact parity vs M1. |
| `test/test_M3_1_phase5_sod_HG.jl` | Phase-5 Sod regression vs py-1d golden + per-step bit-exact parity vs M1. |
| `test/test_M3_1_phase5_sod_HG.jl_helpers.jl` | Shared `build_sod_mesh_HG` + `extract_eulerian_profiles_HG` helpers (Phase 5 + 5b tests). |
| `test/test_M3_1_phase5b_qnone_bit_equal.jl` | `q_kind = :none` is bit-equal to default-kwargs path; HG is bit-equal to M1 (4 paths × 7 fields × 8 cells = 144 element-wise asserts after 2 timesteps). |
| `test/test_M3_1_phase5b_qvnr_sod.jl` | Phase-5b Sod with `q_kind = :vNR_linear_quadratic`; matches M1 numerically (L∞ rel < 0.13 / 0.18) and bit-exactly per cell. |
| `test/test_M3_1_phase5b_qformula_unit.jl` | Unit tests for `compute_q_segment` formula (15 hand-computed cases). |
| `test/runtests.jl` (append) | M3-1 testset block. |
| `reference/notes_M3_1_phase2_5_5b_port.md` | This file. |

Files explicitly **not modified** in M3-1:

- M1 source: `src/cholesky_sector.jl`, `src/discrete_action.jl`,
  `src/discrete_transport.jl`, `src/segment.jl`, `src/newton_step.jl`,
  `src/deviatoric.jl`, `src/artificial_viscosity.jl`, M1's `DetField{T}`
  / `ChField{T}` blocks in `src/types.jl`.
- M2 source: `src/heat_flux.jl`, `src/stochastic_injection.jl`,
  `src/tracers.jl`, `src/amr_1d.jl`.
- All M1 / M2 test files. The new M3-1 tests are sibling files, not
  edits to existing tests.
- `Project.toml`, `Manifest.toml` — M3-0 already pinned HG + R3D.
- `reference/MILESTONE_*_STATUS.md`, `reference/notes_phase*.md`,
  `specs/`, `design/`, `HANDOFF.md`, `LICENSE`, `py-1d/`.

## Parity test results

| Test | Setup | Per-step `max(|Δfield|)` | Brief target |
|---|---|---|---|
| Phase-2 mass + bit-exact parity | `N = 16`, periodic, `dt = 1e-3`, 50 steps | **0.0** | < 5e-13 |
| Phase-2 momentum (zero + nonzero) | `N = 32`, periodic, `dt = 5e-3`, 100 steps | conservation `< 1e-12` (zero); `< 1e-10` (drift, nonzero) | < 1e-12; < 1e-10 |
| Phase-2 free-streaming | `N = 16`, cold IC, `dt = 1e-3`, 100 steps | position error `< 1e-10`; velocity error `< 1e-10` | < 1e-10 |
| Phase-2 acoustic + bit-exact parity | `N = 64`, periodic, `dt = 1e-3`, 1500 steps | **0.0** per cell sample at every 100-step | < 5e-13 |
| Phase-5 Sod + bit-exact parity | `N = 100` (mirror to 200 segments), `t_end = 0.2`, `τ = 1e-3`, ~244 steps | **0.0** per cell every 24 steps | < 1e-12 |
| Phase-5b `q_kind = :none` bit-equal | `N = 8`, 2 steps, default vs explicit `:none` × HG vs M1 | **0.0** every cell every field every comparison (144 asserts) | exact equality |
| Phase-5b Sod q=:vNR + bit-exact parity | `N = 100` (mirror), `c_q_quad = 2.0`, `c_q_lin = 1.0`, ~244 steps | **0.0** per cell every 24 steps | < 1e-12 |

The exact-zero result (rather than ULPs) is achieved because
`det_step_HG!` calls `det_step!` byte-identically through the cached
`Mesh1D` shim. The HG storage layer is bit-preserving (Float64 ↔
Float64 single-coefficient writes), so the only divergence between
the two paths is the storage shim.

### Sod golden L∞ rel errors (HG path matches M1 numerically)

Phase 5 (q=:none), N=100 mirror, τ=1e-3, t_end=0.2, ~244 steps:

| Field | M1 / HG L∞ rel | Brief target (M1 baseline) |
|---|---|---|
| ρ   | **0.113** | within 1e-12 of M1 (M1 reports ~0.11) |
| u   | **0.918** | M1 reports ~0.92 |
| Pxx | **0.183** | M1 reports ~0.18 |
| Pp  | **0.183** | M1 reports ~0.18 |

Phase 5b (q=:vNR, `c_q_quad = 2.0`, `c_q_lin = 1.0`), same setup:

| Field | M1 / HG L∞ rel |
|---|---|
| ρ   | **0.104** |
| u   | **0.581** |
| Pxx | **0.145** |
| Pp  | **0.143** |

Both match M1's recorded numbers within bit-exact tolerance — the L∞
errors are *literally identical* between the M1 and HG paths because
`det_step_HG!` produces byte-identical state.

## Wall-time benchmarks

Single-thread, Apple M-series.

| Setup | M1 wall | HG wall | Ratio HG/M1 |
|---|---|---|---|
| Phase-2 acoustic, N=64, 1500 steps | 11.73 s | 11.46 s | **0.977×** |
| Phase-5 Sod, N=100 mirror=200, 244 steps | 7.10 s | 6.72 s | **0.946×** |
| Phase-5b Sod q=:vNR, N=100 mirror=200, 244 steps | 7.25 s | 6.94 s | **0.957×** |

HG path is consistently within 5% of M1's runtime, matching the
M3-0 single-cell benchmark ratio (1.011×) and the M3-0 64-cell
ratio (0.877×). The round-trip-through-`cache_mesh` shim costs
~2-5% of the per-step budget on these problem sizes; M3-2 will
remove the cache when the EL residual moves natively onto HG.

## Total test count

Before M3-1: M1 + M2 + M3-0 = 2044 + 1 deferred + 23 = 2067 + 1.

M3-1 added **197 new test assertions** (per the M3-1 testset block):
- Phase 2 mass: 5
- Phase 2 momentum: 3
- Phase 2 free streaming: 3
- Phase 2 acoustic: 6
- Phase 5 Sod regression: 12
- Phase 5b q=:none bit-equality: 144
- Phase 5b q=:vNR Sod: 9
- Phase 5b compute_q formula unit: 15

After M3-1: full `Pkg.test()` reports
**2264 passed + 1 broken = 2265 total**.

The "1 broken" is a pre-existing `@test_skip` from Phase 6's
post-crossing golden match, gated on a future shock-capturing
extension (per `reference/notes_phase6_cold_sinusoid.md`). It is
**not** introduced by M3-1.

## HG workarounds applied

### Periodic-BC neighbour helper

HG does not yet ship a `periodic!(mesh)` builder (design-guidance
item #2). M3-1 adds the dfmm-side `periodic_simplicial_mesh_1D(N, L)`
which fills the cyclic-wrap entries of the simplex-neighbour matrix
in the same convention M3-0's `uniform_simplicial_mesh_1D` uses for
the boundary case. The helper is single-purpose (1D periodic) and
will be superseded by HG's helper once it lands.

### `n_cells` name collision

`HierarchicalGrids` exports `n_cells`; M3-1 considered exporting
`n_cells(::DetMeshHG)` from dfmm but reverted. The function is
defined on the dfmm side as `n_cells(mesh::DetMeshHG) =
HierarchicalGrids.n_simplices(mesh.mesh)` but is not exported; users
should call `length(mesh.Δm)` or `HierarchicalGrids.n_simplices(mesh.mesh)`
directly. (The same issue applied to M3-0's `spatial_dimension`.)

## Newton iteration counts

The Phase-2/5/5b Newton solves run on the same EL residual M1 uses
(`det_el_residual` from `src/cholesky_sector.jl`), with the same
`NewtonRaphson(; autodiff = AutoForwardDiff())` solver and the same
sparse-Jacobian prototype `det_jac_sparsity(N)` for `N ≥ 16`. Per-step
iteration counts are therefore identical to M1's: 2-4 iterations on
smooth flow (Phase 2 acoustic, Phase-5 IC) and 4-8 iterations near
the shock front in the Sod runs.

## Open questions / handoff to M3-2

1. **Retire the `cache_mesh` shim.** M3-2 should move the EL residual
   evaluation natively onto HG iteration. The handoff path: write a
   native `det_el_residual_HG` that reads from the field set + neighbour
   array directly, then drop the cache mesh. The bit-exact baseline
   from M3-1 makes the divergence checkable per-step.

2. **Sparse-Jacobian native to HG.** When the EL residual moves
   natively, the M1 `det_jac_sparsity(N)` helper has to be replaced
   by an HG-mesh-aware sparsity builder (design-guidance item #7).
   The M1 pattern is the trivial "N-1 cyclic, tri-block-banded"
   form; HG's general adjacency API will support arbitrary AMR
   patterns when M2-1 lands on HG.

3. **HG BMI2 patch upstream.** Pre-existing M3-0 handoff: the
   develop-mounted HG fork still carries the `@static if` patch.
   Once upstream lands the fix, replace the develop pin with an
   upstream commit pin in `Manifest.toml`.

4. **Multi-tracer field set.** M3-2's Phase-11 port adds passive
   tracers — straightforward extension of `allocate_detfield_HG`
   to include user-named tracer fields, since `PolynomialFieldSet`
   accepts arbitrary keyword fields. The bit-exact bookkeeping of
   `TracerMesh` carries over directly (HG's bit-preserving SoA
   storage is identical to M1's `Vector{T}`-of-Vector{T} layout).

5. **`DetField{1, T}` vs `DetFieldND{1, T}` aliasing.** Once M3-2
   retires the legacy `Mesh1D` path, consider renaming `DetFieldND{D, T}`
   to `DetField{D, T}` — but only after `Segment{T,DetField{T}}`
   users have all migrated to the HG path. Don't rename in M3-1.
