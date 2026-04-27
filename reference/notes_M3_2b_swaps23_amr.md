# Phase M3-2b — Swaps 2 + 3 (AMR shim retirement) — INVESTIGATION REPORT

**Status: Blocked — architectural mismatch.**

The two swaps as written in the M3-2b plan
(`look-at-the-directory-proud-mountain.md` Part 3 table) cannot be
performed bit-exactly without a substantially larger rewrite that
this milestone does not authorise. This note documents the gap,
points at the correct future-phase fix, and recommends that swaps
2 and 3 be deferred to **M3-3** when the `cache_mesh::Mesh1D` shim
itself retires.

## The plan vs. reality

The M3-2b swap table (Part 3 of the plan) calls for:

| Shim | HG primitive |
|------|--------------|
| `rebuild_HG_from_cache!` | `register_on_refine!(callback)` (HG `3d9847c`) |
| `amr_step_HG!` | `step_with_amr!(...)` (HG `2e39b4f`) |

Both HG primitives target **`HierarchicalMesh{D}`** — HG's
hierarchical, tree-structured Eulerian mesh substrate.

dfmm's M2-1 action-AMR, however, runs on **`Mesh1D`** — the dfmm-native
Lagrangian-mass-coordinate segment mesh — and only afterwards rebuilds
the HG-side `SimplicialMesh{1, T}` from the post-AMR cache mesh state.

The complete `DetMeshHG` field layout (`src/newton_step_HG.jl:213`):

```
mutable struct DetMeshHG{T<:Real}
    mesh::SimplicialMesh{1, T}            # HG simplicial (Lagrangian) mesh
    fields::PolynomialFieldSet            # HG field bundle
    Δm::Vector{T}
    L_box::T
    p_half::Vector{T}
    bc::Symbol
    cache_mesh::Mesh1D{T,DetField{T}}     # dfmm-native shim
end
```

There is **no `HierarchicalMesh{D}`** in `DetMeshHG`, anywhere on its
construction path, or anywhere reachable through its fields. HG's
listener / driver APIs are documented as operating on
`HierarchicalMesh` (see `~/.julia/dev/HierarchicalGrids/src/Mesh/Mesh.jl:675`,
`AMR.jl:68`):

```
register_refinement_listener!(mesh::HierarchicalMesh, callback)
step_with_amr!(state, frame::EulerianFrame{D, T}, step_fn, indicator, n_steps; ...)
```

`SimplicialMesh{1, T}` does not carry a listener registry, does not
fire refinement events, and has no `refine_cells!`/`coarsen_cells!`
hooks. It is purely Lagrangian (vertex-position-only) — there is no
cell-level refinement primitive on it. Confirmed:

- `grep -n "register_refinement_listener\|listener\|on_refine" SimplicialMesh.jl` → 0 hits.
- `grep -n "refine\|coarsen" SimplicialMesh.jl` → only the
  `distortion_metric` docstring mentions "refinement / regeneration"
  as a downstream consumer use case, no actual refinement code.

## Why the swap can't be a pure swap

### Swap 2 (refine listener)

The body of `rebuild_HG_from_cache!` is triggered by mutations on
`mesh.cache_mesh::Mesh1D`. To make it a callback fired by HG's
`register_refinement_listener!`, one of the following must happen:

(a) **Replace `cache_mesh::Mesh1D` with `cache_mesh::HierarchicalMesh{1}`.**
    But the M1 `refine_segment!`/`coarsen_segment_pair!` operate on
    `Mesh1D` segments by mass-coordinate equal-split — these are
    *different* primitives from HG's `refine_cells!`/`coarsen_cells!`,
    which split tree cells geometrically. Different conservation laws
    apply; a 1:1 byte-equal substitution is not possible. The 82-assert
    M2-1 test expects mass-coordinate semantics (`Δm[2] == 0.125` after
    `refine_segment_HG!(mesh, 2)` halves Δm — not what HG's
    `refine_cells!` produces).

(b) **Add a listener registry to `Mesh1D`.** This is dfmm-internal
    work, not "use HG primitive". It also introduces exactly the kind
    of new shim the brief forbids ("Don't introduce new shims").

(c) **Add a listener registry to `DetMeshHG`.** Same objection as (b).

### Swap 3 (`step_with_amr!`)

`step_with_amr!`'s signature requires a `state` plus an `EulerianFrame{D, T}`
whose `frame.mesh::HierarchicalMesh{D}` is the AMR target. The driver
calls `refine_by_indicator!(frame.mesh, ind; ...)` internally — a
`HierarchicalMesh` operation. `DetMeshHG` has no `EulerianFrame` and
no `HierarchicalMesh{D}`; it has a `SimplicialMesh{1, T}` plus the M1
`Mesh1D` shim.

The M2-1 indicator outputs (`action_error_indicator_HG`,
`gradient_indicator_HG`) are computed on `cache_mesh::Mesh1D`'s
segment array, indexed by mass-coordinate position. They have no
representation on a `HierarchicalMesh` cell tree.

Even ignoring topology mismatch, `step_with_amr!` is structured as a
*driver* (it calls `step_fn(state, frame)` on a fixed cadence inside
its own outer loop). dfmm's `amr_step_HG!` is *not* a driver — it
applies one round of refine/coarsen and returns, leaving the time
loop to the caller. Replacing `amr_step_HG!` with `step_with_amr!`
would invert the control flow at every callsite; the M2-1 test alone
would need a rewrite (it asserts on the `(n_refined, n_coarsened)`
return value of one `amr_step_HG!` call — not present in
`step_with_amr!`'s API).

## Recommended path forward

Defer swaps 2 and 3 to **M3-3**, where the `cache_mesh::Mesh1D` shim
itself retires per the existing plan note in
`reference/notes_M3_2_phase7811_m2_port.md` Open Issue #3:

> 3. **`register_on_refine!` callback API in HG.** Items #5 from
>    `notes_HG_design_guidance.md`. Once HG ships the callback,
>    `rebuild_HG_from_cache!` can be replaced by an in-place HG-side
>    update. **M3-3 (or whichever phase first writes the native HG-side
>    EL residual) is the natural time to retire the cache mesh in
>    AMR**, since at that point there is no longer a cache mesh to do
>    the AMR work on.

In M3-3, the M2-1 AMR primitives (`refine_segment_HG!` etc.) will be
rewritten to operate directly on a tree-structured AMR substrate,
which is the right time to wire HG's listener / driver in. The
mass-coordinate semantics of the conservative Cholesky-merger
(law-of-total-covariance in `coarsen_segment_pair!`) will need to be
re-expressed against HG's tree topology — a substantial design task,
not a swap.

## What was investigated

- HG listener API (`Mesh.jl:625-725`): operates on `HierarchicalMesh`
  only.
- HG `step_with_amr!` (`AMR.jl`): operates on `EulerianFrame{D, T}` ⊃
  `HierarchicalMesh{D}` only.
- HG `SimplicialMesh{1, T}` (`SimplicialMesh.jl`): no listener
  registry; no refine/coarsen API.
- HG test files (`test_refinement_listener.jl`,
  `test_step_with_amr.jl`): all examples build a `HierarchicalMesh{D}`
  first and exercise the listener / driver against it.
- dfmm `DetMeshHG` (`src/newton_step_HG.jl:213`): contains
  `SimplicialMesh{1, T}` + `Mesh1D` shim, no `HierarchicalMesh`.
- dfmm M2-1 AMR primitives (`src/amr_1d.jl`,
  `src/action_amr_helpers.jl`): mass-coordinate equal-split on
  `Mesh1D`, with HG-side rebuild from cache.

## Verdict

**No commit produced.** The architectural premise of the swap (that
`rebuild_HG_from_cache!` is a manual workaround for a missing HG
callback that, once shipped, becomes a 1:1 swap) does not match the
actual code: the rebuild is not driven by HG-side topology events; it
is driven by Mesh1D-side topology events that do not flow through HG
at all.

The bit-exact 0.0 parity gate would have to be sacrificed (test 1's
`Δm[2] == 0.125` assertion is mass-coordinate-specific; HG's
`refine_cells!` produces sibling cells in tree topology, not equal-mass
daughters), and per the brief's critical constraint #1, when "asserted
bit-exact tests fail, stop and report — do NOT massage tolerances." So
this report is the deliverable.

## Suggested update to the M3-2b swap table

Strike rows 2 and 3 from the table; merge their work into M3-3's
"native HG-side EL residual + Newton sparsity + halo wiring + AMR"
scope. This is consistent with the existing
`notes_M3_2_phase7811_m2_port.md` Open Issue #3 and the
`MILESTONE_3_PLAN_legacy_pre_HG.md` baseline. The remaining six
swaps (1, 4, 5, 6, 7, 8) are unaffected and may proceed in parallel
worktrees.
