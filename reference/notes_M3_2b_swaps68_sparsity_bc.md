# M3-2b — Swaps 6 + 8: HG `cell_adjacency_sparsity` + `BCKind`/`FrameBoundaries`

## Scope

This phase retires two of the M3-2 dfmm-side shims in favour of
upstream HierarchicalGrids primitives (HG HEAD `7c56d47`):

| dfmm shim today | HG primitive | dfmm files | Swap |
|---|---|---|---|
| `det_jac_sparsity(N)` (segment-adjacency hand-rolled, periodic 1D) | `cell_adjacency_sparsity` (face-neighbor graph) | `src/newton_step_HG.jl` | 6 |
| `bc, inflow_state, outflow_state, n_pin` kwargs threaded through `det_step_HG!` | `BCKind` + `FrameBoundaries{1}` | `src/newton_step_HG.jl` | 8 |

Both swaps touch `src/newton_step_HG.jl`, so they are bundled in one
worktree to avoid merge conflicts.

## Bit-exact 0.0 parity gates

Both gates pass at `0.0` exactly:

- **Swap 6 gate** — `det_jac_sparsity_HG(mesh::SimplicialMesh{1, T})`
  produces a `SparseMatrixCSC{Bool}` byte-equal to M1's hand-rolled
  `det_jac_sparsity(N)` on N = 16 periodic mesh and N = 4 wrap-around
  edge case. Verified via `findnz` row/col equality and dense-Bool
  matrix equality (`test/test_M3_2b_swap6_sparsity_HG.jl`, 7 asserts).
- **Swap 8 gate** — Phase-7 Mach-3 steady-shock parity test
  (`test/test_M3_2_phase7_steady_shock_HG.jl`) holds
  `parity_err = 0.0` and `Q_err = 0.0` after 33 timesteps on N = 80,
  identical to the pre-swap snapshot.
- **Indirect gate** — M2-1 AMR HG test (82 asserts) and M3-1 Phase
  2/5/5b parity tests (197 asserts) all green and bit-exact.

Test count: **3037 + 1 → 3056 + 1**. The 19 added asserts are the
two new test files; no existing test was modified except for the
Phase-7 IC factory (see "Caller migration" below).

## Swap 6 — design

### What HG offers and what it doesn't

HG's `cell_adjacency_sparsity(mesh::HierarchicalMesh; depth)` operates
on `HierarchicalMesh{D, M}` (the cartesian-tree mesh used as the
Eulerian remap target), not on `SimplicialMesh{D, T}` (the Lagrangian
segment mesh dfmm 1D evolves on). The 1D Lagrangian path therefore
cannot consume `cell_adjacency_sparsity` directly — the data structures
are incompatible.

What the `SimplicialMesh{1, T}` *does* expose, however, is the
per-simplex face-neighbor matrix `simplex_neighbors[k, s]`:
`face_k`-of-simplex-s points to the simplex sharing that face (`0`
on a domain boundary). This is the same connectivity information
`cell_adjacency_sparsity` walks through `face_neighbors` on the
HG-cartesian side, and is sufficient to build a depth-1 sparsity
pattern matching M1's hand-rolled tridiagonal-block layout.

### Implementation

`det_jac_sparsity_HG(mesh::SimplicialMesh{1, T}; depth = 1)` walks
each simplex `i` and pulls its `(i_lo, i_hi)` neighbours via
`simplex_neighbor(mesh, i, k)` for `k ∈ {1, 2}`. Boundary entries
(`0`) are interpreted as cyclic-wrap for parity with M1's periodic
sparsity (M3-3 will refine this for non-periodic boundaries by
consulting `mesh.bc_spec`). The output is a `SparseMatrixCSC{Bool}`
with the same `(rows, cols)` layout `det_jac_sparsity(N)` produces.

A wrapper overload `det_jac_sparsity_HG(mesh::DetMeshHG)` is provided
for caller convenience.

### Status note

In M3-2b the cache-mesh-driven Newton solve continues to use M1's
helper (the cache mesh is itself a `Mesh1D{T, DetField{T}}` and the
Newton residual evaluator only knows how to consume `det_jac_sparsity(N)`).
The HG-aware helper is a forward hook that M3-3's native HG-side EL
residual will plug into. We provide it now so M3-3 inherits a tested,
parity-gated entry point.

### M3-3 generalisation path

For the future 2D/3D Newton residual, `det_jac_sparsity_HG` will
dispatch on the mesh dimension:

- 1D `SimplicialMesh` — current `simplex_neighbors`-driven path.
- 2D / 3D Eulerian Newton blocks — `cell_adjacency_sparsity` on the
  underlying `HierarchicalMesh` directly (with `leaves_only = true`
  and `depth = 1` for first-cut; `depth = 2` if the residual stencil
  reaches into face-of-face neighbours via the polynomial reconstruction).

## Swap 8 — design

### What HG offers and what it doesn't

HG's `BCKind` is a 5-element enum (`PERIODIC, INFLOW, OUTFLOW,
REFLECTING, DIRICHLET`) and `FrameBoundaries{D}` is a thin wrapper
around an `NTuple{D, NTuple{2, BCKind}}`. Both are *tag-only* types:
they record which kind of BC applies on which axis-side, but they
do NOT carry payload (no `(rho, u, P, …)` for `INFLOW`, no Dirichlet
state vector, no `n_pin`). Payload semantics are explicitly delegated
to PDE-level code (`BoundaryConditions.jl` docstring: "non-periodic
kinds are advisory at this layer").

This is a **partial-Swap-8 situation**: the symbolic BC tag moves to
`FrameBoundaries{1}`, but dfmm's Phase-7 inflow/outflow primitive
literals (the post-shock `(rho, u, P, alpha, beta, s, Pp, Q)` tuples)
must still travel through dfmm-side fields — they have no upstream
home.

### Implementation

The `DetMeshHG{T}` mutable struct gains four new fields (append-only;
existing positional-construction call sites in `DetMeshHG_from_arrays`
were updated in lockstep):

```julia
bc_spec::FrameBoundaries{1}                     # the BC tag
inflow_state::Union{NamedTuple, Nothing}        # post-shock primitives
outflow_state::Union{NamedTuple, Nothing}       # downstream primitives
n_pin::Int                                      # Phase-7 pin count
```

`DetMeshHG_from_arrays` accepts a new `bc_spec::FrameBoundaries{1}`
kwarg and translates the legacy `bc::Symbol` into the new spec when
absent:

| `bc::Symbol` | `bc_spec` |
|---|---|
| `:periodic` | `((PERIODIC, PERIODIC),)` |
| `:inflow_outflow` | `((INFLOW, OUTFLOW),)` |
| (other) | `((REFLECTING, REFLECTING),)` |

`det_step_HG!` retires the `bc, inflow_state, outflow_state, n_pin`
kwargs entirely. It now reads them from the wrapper struct and
translates `FrameBoundaries{1}` → legacy `bc::Symbol` via
`bc_symbol_from_spec` for the cache-mesh delegate (which still
runs through M1's `det_step!`). When M3-3 retires the cache mesh,
the symbol-tag translation can be removed — the native HG-side EL
residual will consume `mesh.bc_spec` directly.

### Caller migration

One caller, the Phase-7 IC factory `build_steady_shock_mesh_HG` in
`test/test_M3_2_phase7_steady_shock_HG.jl`, attached the
inflow/outflow tuples through kwargs to `det_step_HG!`. The factory
now attaches them via the constructor, and the `det_step_HG!` calls
in the same test file dropped the four kwargs. No other dfmm callers
were affected.

### Limits / future work

Three non-periodic `BCKind`s — `INFLOW` (with no payload),
`OUTFLOW` (with no payload), `REFLECTING`, `DIRICHLET` — are not
yet handled by M1's `det_step!` 1D path. The current implementation
maps all "weird" combinations onto `:periodic` for the cache-mesh
delegate. M3-3's native HG residual will need to:

1. Honor `bc_spec` per-axis-half directly (no symbol-tag round-trip).
2. Consume the `inflow_state` / `outflow_state` literals at the
   residual / pinning step.
3. Generalise to mixed BCs in 2D/3D (e.g. `(PERIODIC, PERIODIC)` × `(REFLECTING, REFLECTING)` for a periodic-x reflecting-y channel).

If/when HG adds a payload-carrying `BoundaryState{D, T}` companion
struct (e.g. holding the Dirichlet state per face), dfmm should
migrate the inflow/outflow tuples there.

## LOC delta

- `src/newton_step_HG.jl`: +193 / -20 (net +173): struct field
  additions, new `bc_symbol_from_spec`, new `det_jac_sparsity_HG`,
  kwarg retirement, post-Swap-1 sn layout adoption.
- `src/newton_step.jl`: 0 (untouched — M1's `det_jac_sparsity` is
  preserved as-is for the cache-mesh-driven path).
- `src/dfmm.jl`: +4 / 0 (two new exports).
- `test/test_M3_2_phase7_steady_shock_HG.jl`: +5 / -15 (factory
  attaches BC; call sites drop the four kwargs).
- `test/test_M3_2b_swap6_sparsity_HG.jl`: +60 (new file).
- `test/test_M3_2b_swap8_bckind_HG.jl`: +60 (new file).
- `test/runtests.jl`: +10 (new testset block).

## Interaction with Swap 1

Swap 1 (`M3-2b Swap 1: retire periodic_simplicial_mesh_1D for HG
periodic!`) landed on `main` first. Swap 1 changed the
`simplex_neighbors` layout in `DetMeshHG_from_arrays` from
cyclic-pre-wrap (`sn[1] = j-1`, `sn[2] = j+1`) to interior-only-
with-zero-on-boundary (`sn[1] = j+1` for hi-side neighbour, `sn[2]
= j-1` for lo-side neighbour), letting `HierarchicalGrids.periodic!`
wire the wrap. After this swap rebased onto Swap 1 the new helper
`det_jac_sparsity_HG` therefore reads:

```julia
i_hi_raw = simplex_neighbor(mesh, i, 1)   # face 1 opposite vtx 1 = hi side
i_lo_raw = simplex_neighbor(mesh, i, 2)   # face 2 opposite vtx 2 = lo side
```

(The sparsity is symmetric in `(i_lo, i_hi)`, so the relabeling
doesn't affect the resulting `(row, col)` pattern, but the variable
names match the post-Swap-1 convention for clarity.)

## Verification

```julia
julia --project=. -e 'using Pkg; Pkg.test()'
# dfmm tests passed
# Pass: 3056   Broken: 1   Total: 3057
```

- 3037 → 3056 + 1 = 3057 (exactly the 19 new asserts).
- Phase 7 HG test: `parity_err = 0.0`, `Q_err = 0.0` (bit-exact).
- M2-1 HG test: 82 / 82 pass.
- M3-1 Phase 2/5/5b: 197 / 197 pass with `parity_err = 0.0`.
