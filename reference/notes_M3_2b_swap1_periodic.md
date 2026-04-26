# Phase M3-2b Swap 1 — retire `periodic_simplicial_mesh_1D`, use HG `periodic!`

**Date:** 2026-04-26.
**Branch:** `m3-2b-swap1-periodic`.
**Base:** local `main` at `b8f4dc2` ("M3-2b pre-flight: bump
HierarchicalGrids dev-mount to 7c56d47").
**Status:** complete; bit-exact 0.0 parity gate held; full test suite
passes at 3037 + 1 broken (unchanged).

This note is the implementation log for Swap 1 of M3-2b
(reference/MILESTONE_3_PLAN.md, M3-2b table; adapted plan
~/.claude/plans/look-at-the-directory-proud-mountain.md, Part 3).
HierarchicalGrids 7c56d47 ships an upstream `periodic!(mesh, axes,
bounds)` builder for `SimplicialMesh{D, T}` (HG commit 97c5035 / PR-D).
We retire dfmm's hand-rolled `periodic_simplicial_mesh_1D` helper and
delegate to the upstream builder.

## Design decisions

### 1. Option A: delete the helper, inline `HierarchicalGrids.periodic!`

The dfmm helper `periodic_simplicial_mesh_1D(N, L; T)` lived in
`src/newton_step_HG.jl` and was exported from `src/dfmm.jl`. It was
**never called** from any production path or test — `DetMeshHG_from_arrays`
duplicated the same wiring inline rather than calling the helper.
Removal is therefore safe and reduces the public API surface.

The new `DetMeshHG_from_arrays` body builds the simplex-neighbour
matrix with **interior wiring only** (no manual cyclic wrap), then
calls `HierarchicalGrids.periodic!(hg_mesh, (true,), ((0, cum_m),))`
when `bc == :periodic`. For non-periodic BCs (`:inflow_outflow`) the
boundary entries stay 0, which matches HG convention and is also
strictly more correct than the previous unconditional cyclic wiring
(which was a latent bug; no consumer reads `simplex_neighbors` in
dfmm yet, so it never fired).

### 2. Column-convention switch

HG's `periodic!` follows the strict simplicial convention from HG's
`SimplicialMesh` docstring: `simplex_neighbors[k, s]` is the simplex
sharing the face **opposite vertex k** of simplex `s`. For 1D
segments with `sv = [v_lo; v_hi]`:

- `sn[1, j]` = neighbour opposite v_lo = **high-x (right)** neighbour
- `sn[2, j]` = neighbour opposite v_hi = **low-x (left)** neighbour

This is the OPPOSITE of dfmm's old helper, whose docstring claimed
`sn[1, j]` was the left neighbour but whose code actually filled it
with `j-1` (i.e. left). The new code follows the upstream convention:

```julia
sn[1, j] = j < N ? Int32(j+1) : Int32(0)   # interior; wrap → periodic!
sn[2, j] = j > 1 ? Int32(j-1) : Int32(0)   # interior; wrap → periodic!
```

`HierarchicalGrids.periodic!(hg_mesh, (true,), ((0.0, cum_m),))` then
identifies the lo-mass and hi-mass boundary faces by face-centroid
matching (trivial in 1D — single face per wall) and writes the
matched simplex indices into the boundary entries.

The convention swap is invisible at the parity-gate level because
**no code in dfmm reads `simplex_neighbors`** today (verified via
`grep -rn 'simplex_neighbor' src/ test/`). M3-2/M3-3 phases that
will iterate over neighbour stencils on the HG mesh natively will
inherit the upstream convention from this swap.

### 3. Files modified

- `src/newton_step_HG.jl`: deleted `periodic_simplicial_mesh_1D`
  (~17 LOC of definition + ~20 LOC of docstring), updated the
  `DetMeshHG{T}` docstring to reference HG's upstream builder, and
  rewrote the periodic-wrap block inside `DetMeshHG_from_arrays` to
  call `HierarchicalGrids.periodic!`. Net: -29 LOC.
- `src/dfmm.jl`: removed `periodic_simplicial_mesh_1D` from the
  export list. Net: -1 token (`, periodic_simplicial_mesh_1D`).

No test changes — the helper was never exercised directly.

## Bit-exact parity gate

Full test suite via `julia --project=. -e 'using Pkg; Pkg.test()'`:

| Stage    | Pass | Fail | Broken |
|----------|------|------|--------|
| Baseline | 3037 | 0    | 1      |
| Swap 1   | 3037 | 0    | 1      |

All M3-1 (Phase 2 acoustic / mass / momentum / free-streaming;
Phase 5 Sod; Phase 5b q-none) and M3-2 (Phase 7 steady shock;
Phase 8 stochastic; Phase 11 tracer; M2-1 AMR; M2-2 multitracer;
M2-3 realizability) HG parity tests held bit-exact at 0.0
absolute, matching their pre-swap behaviour.

## HG API surprises

None. HG's `periodic!` documentation, signature, and 1D-segment test
case (`~/.julia/dev/HierarchicalGrids/test/test_periodic_simplicial.jl`)
are a direct fit for dfmm's M3-1 use case. Only the column-convention
mismatch (item #2 above) needed addressing, and that was a doc-only
bug in the old dfmm helper rather than a behavioural mismatch.

## Cross-reference

- **Plan entry:** M3-2b Swap table, item 1 (replace dfmm
  periodic-wiring helper with HG `periodic!`).
- **HG source:** `~/.julia/dev/HierarchicalGrids/src/Mesh/SimplicialMesh.jl`
  lines 489–611.
- **HG tests:** `~/.julia/dev/HierarchicalGrids/test/test_periodic_simplicial.jl`.
- **Predecessor note:** `reference/notes_M3_1_phase2_5_5b_port.md`
  §2 documents the now-retired hand-rolled wiring.
