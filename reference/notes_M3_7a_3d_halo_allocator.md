# M3-7a — 3D HaloView smoke + 3D field-set allocator + read/write helpers

> **Status (2026-04-26):** *Implemented + tested*. First sub-phase
> of the M3-7 (3D extension) milestone proper. Branch
> `m3-7a-3d-halo-smoke-allocator`, three commits (a, b, close).
>
> Test delta vs M3-7 prep baseline: **+2581 asserts** in two new
> test files (426 + 2155). All pre-existing tests (~27399 + 1
> deferred + the 736 from M3-7 prep scaffolding) confirmed byte-
> equal across 1D + 2D + 3D-decomposition paths.
>
> M3-7 prep (`64fb1ad`) landed `DetField3D` + 3D Cholesky
> decomposition / recomposition (`src/cholesky_DD_3d.jl`); this
> sub-phase consumes those primitives at the storage layer +
> verifies the 3D HaloView contract. M3-7b (native 3D EL residual)
> is unblocked.

## What landed

| File | Change |
|---|---|
| `test/test_M3_7a_halo_smoke.jl` | NEW (~280 LOC, 426 asserts). 3D analog of `test/test_M3_3a_halo_smoke.jl`: HaloView depth=1 contract for `MonomialBasis{3, 0}` polynomial fields on a 4×4×4 balanced `HierarchicalMesh{3}`. Verifies interior 6-face access, corner-leaf out-of-domain → `nothing`, BC-aware wrap via `face_neighbors_with_bcs`, allocation-free fast path (≤ 64 bytes). Adds depth=2 characterisation block (Q1/Q4 of the M3-7 design note's §11 open questions). |
| `src/setups_2d.jl` | EXTENDED (+150 LOC at end-of-file). New 3D allocator + read/write helpers: `allocate_cholesky_3d_fields(mesh::HierarchicalMesh{3}; T=Float64)`, `write_detfield_3d!(fields, leaf_idx, v)`, `read_detfield_3d(fields, leaf_idx)`. The 3D analog of M3-3a's 2D allocator (+ M3-6 Phase 0's 14-named-field extension): 16 named scalar fields per cell at `MonomialBasis{3, 0}`. |
| `src/dfmm.jl` | APPEND-ONLY (+10 LOC). Three new exports `allocate_cholesky_3d_fields`, `read_detfield_3d`, `write_detfield_3d!` under a new "Phase M3-7a API" comment block. |
| `test/test_M3_7a_field_set_3d.jl` | NEW (~300 LOC, 2155 asserts). Mirrors M3-3a's 2D field-set tests at +1 axis: structural contract, allocator coverage, full byte-equal round-trip on all 64 leaves × all 16 scalars, write-order independence, single-leaf write isolation, T-parameterised allocator (Float32 sanity). |
| `test/runtests.jl` | APPEND-ONLY. New "Phase M3-7a: 3D HaloView smoke + field set" testset block at the end of the M3 testset chain. |
| `reference/notes_M3_7a_3d_halo_allocator.md` | THIS FILE. |
| `reference/MILESTONE_3_STATUS.md` | UPDATED. M3-7a entry added to the phase-by-phase completion table; test summary +2581 → ~29980 + 1. |

## 16-field SoA layout (3D)

`allocate_cholesky_3d_fields` produces a `PolynomialFieldSet` with
the following 16 named scalar fields, each at `MonomialBasis{3, 0}`
(one coefficient per cell — order-0 cell-average storage):

```
:x_1, :x_2, :x_3        — Lagrangian position (charge 0)
:u_1, :u_2, :u_3        — Lagrangian velocity (charge 0)
:α_1, :α_2, :α_3        — principal-axis Cholesky factors
:β_1, :β_2, :β_3        — per-axis conjugate momenta (charge 1)
:θ_12, :θ_13, :θ_23     — Berry rotation angles (Cardan ZYX)
:s                      — specific entropy (operator-split)
```

This matches `DetField3D{T}`'s 13 Newton unknowns + entropy. M3-7
design note §2 boxed equation labels this "13 dof per leaf" (the
boxed formula `D × 4 + binom(D, 2) + 1 = 16` for D=3 counts
position + velocity + α + β + 3 angles + entropy = 16 named scalars,
of which the Newton solver drives 15 — entropy is operator-split
across the Newton step).

## HaloView depth=2 in 3D — open question Q1/Q4 — RESOLVED

The M3-7 design note §11 flagged "HaloView depth=2 in 3D" as Q1
(unverified — does it work as designed?) with the default "stay at
depth=1". The smoke test resolves the open question:

  * **depth=2 constructor accepts the integer.** No exception.
  * **2-hop offsets along an axis (e.g. `(2, 0, 0)`) walk the
    neighbor graph and either return a `PolynomialView` (interior
    2-hop) or `nothing` (boundary).** Verified on a 4×4×4 grid:
    a comfortable majority of the 64 × 6 = 384 axis-aligned 2-hop
    offsets succeed.
  * **2-hop diagonal offsets (e.g. `(1, 1, 0)`) accepted** —
    the depth check `sum(abs.(off)) ≤ depth` permits the call;
    walk semantics are inherited from 1-hop along each axis in
    order. (Whether this is the *physically right* 2-hop walk for
    the M3-7b residual is a separate question; the M3-7 design
    note expects the EL residual to consume axis-aligned ±1
    offsets only, matching M3-3b's 2D pattern.)
  * **3-hop or sum-|off| > 2 throws `ArgumentError`** —
    HaloView's `_check_offset` correctly rejects out-of-depth
    offsets in 3D.

**Verdict on Q1/Q4: depth=2 is not vacuously broken in 3D; the
M3-3a / M3-7 default "stay at depth=1" remains the recommended
operational pattern (matches the M3-7 design note's §3 residual
sketch which uses ±1 face neighbors only), but depth=2 is
available if a future residual variant needs it.**

## Bit-exact 1D + 2D regression — confirmed

Running each upstream test file in isolation against this branch
confirms byte-equal regression:

  * `test_phase1_zero_strain.jl` (1D Phase 1) — 5 asserts pass.
  * `test_M3_3a_halo_smoke.jl` (2D HaloView smoke) — 30 asserts
    pass.
  * `test_M3_3a_field_set_2d.jl` (2D field set) — 295 asserts pass.
  * `test_M3_3a_cholesky_DD.jl` (2D Cholesky decomposition) — 199
    asserts pass.
  * `test_M3_7_prep_3d_scaffolding.jl` (3D Cholesky decomposition
    + DetField3D type definition) — 736 asserts pass.

The append-only changes to `src/setups_2d.jl` (3D allocator at end
of file), `src/dfmm.jl` (3 new exports), and `test/runtests.jl`
(new testset block at end) do not alter any pre-existing 1D / 2D
code paths or their tests.

## Q-resolution against the M3-7 design note's §11 open questions

| Q | Default | Resolution in M3-7a |
|---|---|---|
| Q1 — HaloView depth=2 in 3D | "stay at depth=1" (M3-3a pattern) | **Verified.** depth=2 accepts 2-hop offsets, walks the neighbor graph correctly, throws on 3-hop. The M3-3a default holds for the M3-7b residual but the depth=2 path is available. |
| Q2 — 3×3 eigendecomposition route | "use `LinearAlgebra.eigen`" | **Already adopted** in M3-7 prep (`src/cholesky_DD_3d.jl`). Not touched here. |
| Q3 — off-diagonal β | "pin to zero" | **Adopted.** `allocate_cholesky_3d_fields` does NOT include `β_{ab}` for `a ≠ b`. M3-9 will add a parallel `allocate_cholesky_3d_offdiag_fields` constructor when 3D D.1 KH lands. |
| Q4 — `Pp_a, Q_a` per-axis post-Newton | "stay at 13 dof per leaf for prep" | **Adopted.** `allocate_cholesky_3d_fields` does NOT carry per-axis `Pp` or `Q` storage. M3-7c will extend with these when 3D D.7 / D.10 drivers need them. |
| Q5 — Euler-angle convention | "match SymPy script" | **Already pinned** in M3-7 prep (intrinsic Cardan ZYX). Not touched here. |

## M3-7b launch handoff

When M3-7b launches (native HG-side 3D EL residual without Berry —
3D analog of `cholesky_el_residual_2D!`), the launch agent should
already have everything it needs from M3-7 prep + M3-7a:

### Inputs available

  * `DetField3D{T}` — read-only working struct (`src/types.jl`).
  * `cholesky_decompose_3d`, `cholesky_recompose_3d`,
    `gamma_per_axis_3d` (`src/cholesky_DD_3d.jl`).
  * `rotation_matrix_3d` — SO(3) helper from Cardan ZYX angles.
  * `allocate_cholesky_3d_fields(mesh; T)` — 16-named-field
    `PolynomialFieldSet` allocator.
  * `read_detfield_3d(fields, ci)` / `write_detfield_3d!(fields,
    ci, v)` — bit-exact round-trip helpers.
  * `halo_view(field, mesh, 1)` — depth=1 6-face neighbor access
    on `HierarchicalMesh{3}` (verified by the smoke test;
    allocation-free fast path).
  * `face_neighbors_with_bcs(mesh, ci, spec)` — BC-aware periodic-
    axis wrap, smoke-tested at D=3.

### Pseudo-residual structure (M3-7 design note §2.3)

The 3D EL residual `cholesky_el_residual_3D!` should be the
direct dimension-lift of `cholesky_el_residual_2D!`:

```
F^x_a (cell ci, axis a):
  = (x_a^{n+1} - x_a^n)/dt - midpoint(u_a)
F^u_a (cell ci, axis a):
  = (u_a^{n+1} - u_a^n)/dt
    + (1/Δm) · per-axis pressure stencil at face neighbors (axis a, ±1)
F^α_a (cell ci, axis a):
  = (α_a^{n+1} - α_a^n)/dt - β_a^{n+1/2}
F^β_a (cell ci, axis a):
  = (β_a^{n+1} - β_a^n)/dt - γ_a²/α_a^{n+1/2}
F^θ_{ab} (cell ci, pair (a, b)):
  = trivial-drive in M3-7b ((θ_{ab}^{n+1} - θ_{ab}^n)/dt = 0)
    M3-7c will activate Berry kinetic coupling.
F^s (cell ci):
  = trivial-drive (operator-split, frozen across Newton step).
```

The Newton system grows from M3-3b's `9 N` (2D) to `15 N` rows (3D
Newton-driven count: 3 + 3 + 3 + 3 + 3 = 15 per leaf).

### 3D-specific gotchas

  1. **Coordinate wrap on periodic axes.** M3-4 Phase 1 closed the
     2D periodic-x wrap at the residual level
     (`build_periodic_wrap_tables`). The 3D analog should follow
     the same pattern: per-axis wrap tables for periodic axes
     among `(1, 2, 3)`. Most M3-7 falsifier ICs (Sod, cold
     sinusoid, Zel'dovich pancake) pin one or more axes periodic;
     leaving wrap unimplemented breaks Tier-C 3D consistency
     gates.

  2. **6-face stencil bookkeeping.** The 2D residual indexes face
     neighbors as `(face_lo_idx[1], face_hi_idx[1], face_lo_idx[2],
     face_hi_idx[2])` per cell. The 3D residual needs 6 face
     entries per cell — extend `build_face_neighbor_tables` to
     `D=3` parametrically (already mostly dimension-generic).

  3. **3D pressure stencil sign convention.** The 2D per-axis
     pressure flux is `±(M_vv,a / α_a) · (1 / Δx_a)` at the lo / hi
     face neighbor along axis a. The 3D analog mirrors this for
     each of the 6 face neighbors; the per-axis γ_a entries
     account for off-diagonal pressure terms `M_vv,ab` (which are
     zero in the 13-dof scope; M3-9 activates them).

  4. **Axis-aligned 1D-symmetric ⊂ 3D dimension-lift gates.** M3-7b's
     primary regression target is the 3D 1D-symmetric Sod IC: at
     `θ_12 = θ_13 = θ_23 = 0` and `α_2 = α_3 = const, β_2 = β_3 = 0`
     with `(x_2, x_3, u_2, u_3)` trivial, the 3D residual must
     reduce byte-equally to the 1D EL residual. Pin this gate
     before any non-trivial 3D physics.

  5. **2D-symmetric ⊂ 3D dimension-lift gate.** At `θ_13 = θ_23 = 0`
     and `α_3 = const, β_3 = 0` with `(x_3, u_3)` trivial, the 3D
     residual must reduce byte-equally to the M3-3b 2D residual
     on the top-left 2×2 block. CHECK 3b of
     `notes_M3_prep_3D_berry_verification.md` already verifies the
     Cholesky-decomposition layer; M3-7b needs to verify it at
     the residual layer too.

  6. **Allocation hygiene.** The 2D residual is allocation-free in
     the inner loop; the 3D residual must match this contract on
     the 4×4×4 mesh. The 16-field SoA layout is contiguous per
     field; per-leaf reads should hit the cache without
     allocations. The smoke test verified `hv[i, off]` is ≤ 64
     bytes per call at D=3.

  7. **Newton sparsity.** 2D's Newton sparsity pattern is
     `cell_adjacency ⊗ 11×11`; 3D's should be `cell_adjacency ⊗
     15×15`. The dense Newton fallback at level 4+ in 3D will be
     ~7× slower than 2D at the same level (cell count grows as
     `2^{3 (level)}` vs `2^{2 (level)}`); plan to invoke the
     sparse-Newton pre-conditioner sooner.

## How to extend in M3-7c/d/e

  * **M3-7c** (Berry coupling): consume `berry_partials_3d` from
    `src/berry.jl` directly (already verified at the stencil
    level, 797 asserts in `test_M3_prep_3D_berry_verification.jl`).
    The kernel-orthogonality residual on the θ_23 row per M3-7
    design note §2.2 closed-form formula. The 16-field SoA layout
    here already carries `θ_12, θ_13, θ_23`; add a parallel 19-
    field allocator if `Pp_a, Q_a` per-axis post-Newton sectors
    are needed for Tier-D drivers.

  * **M3-7d** (per-axis γ AMR): `gamma_per_axis_3d` is already
    available. The 3D `gamma_per_axis_3d_field` field-walking
    helper is the natural extension (mirror M3-3d's 2D version in
    `src/diagnostics.jl`; consume `read_detfield_3d` per-leaf).

  * **M3-7e** (Tier-C / D 3D drivers): `tier_c_sod_ic` etc are
    already dimension-generic (D=3 lifts trivially); the IC bridge
    `cholesky_sector_state_from_primitive` needs a 3D variant
    (mirror M3-4 Phase 2's `s_from_pressure_density` +
    `cholesky_sector_state_from_primitive` at +1 axis; α=1, β=0,
    θ_{ab}=0 cold-limit isotropic IC). Headline 3D drivers:
    3D Sod, 3D cold sinusoid, 3D Zel'dovich pancake (D.4 lifted to
    3D — the cosmological reference test).

## Reference

  * `reference/notes_M3_7_3d_extension.md` — full M3-7 design note;
    §3 native HG-side 3D EL residual; §11 open questions Q1 + Q4.
  * `reference/notes_M3_7_prep_3d_scaffolding.md` — predecessor;
    `DetField3D` + `cholesky_DD_3d.jl` round-trip.
  * `reference/notes_M3_3a_field_set_cholesky.md` — 2D analog
    mirrored here at +1 axis.
  * `~/.julia/dev/HierarchicalGrids/src/Storage/HaloView.jl` — HG
    halo API (depth=1 + depth=2 contract recorded in the smoke
    test).
  * `~/.julia/dev/HierarchicalGrids/test/test_halo_view.jl` — HG
    halo usage examples (3D analog folded into the smoke test).
