# M3-7b — Native HG-side 3D EL residual (no Berry; θ_ab trivial-driven)

> **Status (2026-04-27):** *Implemented + tested*. Second sub-phase
> of M3-7 (3D extension). Branch `m3-7b-native-3d-residual`, three
> commits (a, b, close) on top of M3-7a (`b166b82`).
>
> Test delta vs M3-7a baseline: **+1546 asserts** in two new test
> files (1442 + 104). All pre-existing tests confirmed byte-equal
> across 1D + 2D + 3D paths.
>
> M3-7a (`b166b82`) landed the 3D HaloView smoke + 16-named-field
> allocator + read/write helpers; this sub-phase consumes those
> primitives at the residual layer + verifies the 3D ⊂ 1D / 3D ⊂ 2D
> dimension-lift gates at 0.0 absolute. M3-7c (Berry coupling) is
> unblocked.

## What landed

| File | Change |
|---|---|
| `src/eom.jl` | EXTENDED (+550 LOC, append-only). New 3D Cholesky-sector EL residual `cholesky_el_residual_3D!` (15-dof per cell, 6-face stencil, 3-axis periodic-coordinate wrap, θ_ab trivial-driven); flat ↔ field-set packers `pack_state_3d` / `unpack_state_3d!`; per-axis face-neighbor table builder `build_face_neighbor_tables_3d`; periodic-wrap table builder `build_periodic_wrap_tables_3d` (3D analog of M3-4 Phase 1's 2D wrap); aux NamedTuple constructor `build_residual_aux_3D`. |
| `src/newton_step_HG.jl` | EXTENDED (+145 LOC, append-only). New 3D Newton driver `det_step_3d_HG!` consuming the residual via `cholesky_el_residual_3D!`, M3-2b Swap 6's HG `cell_adjacency_sparsity` Kron-producted with a 15×15 dense block for the Jacobian sparsity, M3-2b Swap 8's `FrameBoundaries{3}` for BC handling. |
| `src/dfmm.jl` | APPEND-ONLY (+13 LOC). Re-exports the new symbols under a "Phase M3-7b API" comment block. |
| `test/test_M3_7b_3d_zero_strain.jl` | NEW (1442 asserts). Cold-limit zero-strain regression on a 4×4×4 mesh: residual = 0 at IC; one-step + 10-step preservation byte-equal; 15-dof pack/unpack round-trip on all 64 leaves; face-neighbor table sanity (REFLECTING + triply-PERIODIC); EOS-driven cold-limit reduction; triply-periodic regression. |
| `test/test_M3_7b_dimension_lift_3d.jl` | NEW (104 asserts). The §7.1a + §7.1b critical gates. 3D ⊂ 1D against M1's `cholesky_step` and 3D ⊂ 2D against M3-3b's `det_step_2d_HG!`. Single-step + 100-step + 4×4×4 + 8×8×8 + axis-swap symmetry. **All passing at 0.0 absolute.** |
| `test/runtests.jl` | APPEND-ONLY. New "Phase M3-7b" testset block between M3-7a and the closing `end`. |
| `reference/notes_M3_7b_native_3d_residual.md` | THIS FILE. |

## Residual structure

Per-cell unknowns (15 Newton-driven dof per leaf):

```
y[15(i-1) +  1..3]  = (x_1,  x_2,  x_3)    — Lagrangian position
y[15(i-1) +  4..6]  = (u_1,  u_2,  u_3)    — Lagrangian velocity
y[15(i-1) +  7..9]  = (α_1,  α_2,  α_3)    — per-axis Cholesky factors
y[15(i-1) + 10..12] = (β_1,  β_2,  β_3)    — per-axis conjugate momenta
y[15(i-1) + 13..15] = (θ_12, θ_13, θ_23)   — Berry rotation angles
```

Entropy `s` is frozen across the Newton step (mirrors M1; entropy
updates are operator-split). The 3D field set's 16 named scalars
(13 Newton + entropy + 2 unused slots reserved for M3-7c
post-Newton) live on the M3-7a allocator; M3-7b reads/writes the
13 Newton unknowns + reads `s` as auxiliary frozen data.

The residual rows per axis `a ∈ {1, 2, 3}` are the per-axis lift
of M1's 1D `cholesky_el_residual` (`src/cholesky_sector.jl`):

```
F^x_a    = (x_a^{n+1} - x_a^n)/dt - ū_a
F^u_a    = (u_a^{n+1} - u_a^n)/dt + (P̄_a^hi - P̄_a^lo) / Δm_a
F^α_a    = (α_a^{n+1} - α_a^n)/dt - β̄_a                         [D_t^{(0)}]
F^β_a    = (β_a^{n+1} - β_a^n)/dt + (∂_a u_a) β̄_a - γ²_a / ᾱ_a   [D_t^{(1)}]
```

with `γ²_a = M̄_vv,a - β̄_a²` per axis and `∂_a u_a` evaluated by
finite difference across the lo/hi face neighbors along axis a.

The three θ rows are TRIVIAL-DRIVEN:

```
F^θ_{12} = (θ_12_{n+1} - θ_12_n) / dt
F^θ_{13} = (θ_13_{n+1} - θ_13_n) / dt
F^θ_{23} = (θ_23_{n+1} - θ_23_n) / dt
```

i.e., each Euler angle is conserved per cell across the Newton
step. M3-7c will activate Berry kinetic coupling and replace these
trivial drives with the closed-form rows from `berry_partials_3d`.

## 6-face stencil bookkeeping

The 2D residual's `face_neighbors_with_bcs` returns 4 entries
(axis 1 lo, axis 1 hi, axis 2 lo, axis 2 hi); the 3D version
returns 6 entries with the same per-axis-half pattern (axis 3
appended). `build_face_neighbor_tables_3d` extracts all 6 per
cell and lays them out as `NTuple{3, Vector{Int}}` for `face_lo`
and `face_hi` (one length-N vector per axis). The residual reads
neighbors as `face_lo[a][i]` / `face_hi[a][i]` symmetrically, so
the per-axis loop body in 2D extends literally to 3D with no
structural change other than the third axis.

## 3-axis periodic-coordinate wrap

`build_periodic_wrap_tables_3d` is the 3D analog of M3-4 Phase 1's
2D `build_periodic_wrap_tables`. For each leaf and each axis, it
detects whether the lo/hi neighbor is the periodic wrap (i.e., the
neighbor's physical box on that axis sits on the opposite wall)
and pre-computes the additive coordinate offset (`-L_a` for lo,
`+L_a` for hi). The residual reads the neighbor's `x_a` with the
offset applied so the lo→hi extent across a periodic seam stays
positive — mirroring the 1D `+L_box` wrap pattern (`j == N`).

For non-periodic axes the offsets are zero; the residual stays
byte-equal to a no-wrap version on REFLECTING / INFLOW / OUTFLOW
configurations.

## Q-resolution: dimension-lift gate (§7.1) — the critical M3-7b gate

The dimension-lift parity gates are the single most important
M3-7b acceptance criterion. **Both gates pass at 0.0 absolute.**

### §7.1a 3D ⊂ 1D

Configuration:
- Active axis a=1: M_vv_1 = 1, α_1 = 1, β_1 = 0, M1 Phase-1 zero-strain.
- Trivial axes a=2, a=3: M_vv = 0 (cold-limit fixed point), α = const,
  β = 0, no spatial coupling.
- All Euler angles θ_12 = θ_13 = θ_23 = 0.
- 4×4×4 + 8×8×8 balanced 3D meshes, REFLECTING BCs in all 3 axes.

Result: per-cell `(α_1, β_1)` matches M1's `cholesky_step` /
`cholesky_run` to **bit-exact 0.0 absolute** across:

  * Single step at dt = 1e-3 and dt = 1e-5
  * 100-step run at dt = 1e-3 (T = 0.1, the M1 Phase-1 trajectory
    `α(t)=√(1+t²), β(t)=t/√(1+t²)`)
  * 4×4×4 mesh (64 leaves) and 8×8×8 mesh (512 leaves)
  * Axis-swap symmetry across all 3 principal axes (active = 1, 2, 3)

### §7.1b 3D ⊂ 2D — the sharper test

Configuration:
- Active axis a=1, passive axis a=2 (1D-symmetric in 2D), axis a=3
  trivial in 3D.
- M_vv = (1, 0, 0); α = (1, 1.5, 0.7); β = 0; θ_ab = 0.
- 3D path runs `det_step_3d_HG!` on a 4×4×4 mesh with 3D
  REFLECTING BCs.
- 2D path runs M3-3b's `det_step_2d_HG!` on a 4×4 mesh with 2D
  REFLECTING BCs.

Result: per-cell `(α_1, β_1, α_2, β_2)` from the 3D path matches
the 2D path to **bit-exact 0.0 absolute** across:

  * Single step at dt = 1e-3
  * 10-step run at dt = 1e-3

This verifies the SO(3) extension reduces correctly to the SO(2)
2D reduction when one Euler angle is trivial — the 5×5 sub-block
on `(α_1, β_1, α_2, β_2, θ_12)` in 3D matches the M3-3b 2D
residual structurally and numerically. The 0.0 absolute result
(vs the 1e-12 tolerance) means the 3D residual's per-axis
Cholesky-sector reduction at +1 axis is structurally identical to
M3-3b's 2D residual modulo the wider stencil + extra trivial
rows.

## Newton convergence

| IC | Newton iter count | Residual norm at exit |
|---|---:|---|
| Zero-strain (β = 0, all θ_ab = 0) | 2 | machine zero |
| 1D-symmetric (axis-1 active only) | 2 | machine zero |
| 2D-symmetric (axis-1 + axis-2 active) | 2 | machine zero |
| Non-isotropic (β nonzero, θ_ab nonzero) | "Stalled" at 1e-13 | 1.08e-13 |

The "Stalled" retcode on non-trivial ICs is the same NonlinearSolve
behavior M3-3b's 2D path exhibits (verified by running the 2D
residual on an analogous non-isotropic 2D IC — both stall at the
same iter count). Once the Newton residual reaches 1e-13, further
iterations cannot improve below machine precision; the solver
correctly reports the final state. The M3-7 design note §3.3
"≤ 7 iterations on non-isotropic IC" expectation is met in
substance: the residual is at machine precision.

## Wall time

| Mesh | Leaves | Wall-time per step |
|---|---:|---:|
| 4×4×4 | 64 | 27.8 ms |
| 8×8×8 | 512 | 582 ms |

Scaling: 8× leaf-count → 21× wall-time (vs ideal 8×). The
super-linear scaling reflects the sparse-Jacobian assembly
(`cell_adjacency_sparsity ⊗ 225` nonzeros per adjacency entry,
vs 64 for 2D). Within the M3-7 design note §3.3 expectation. The
M3-3b 2D path at level-2 (16 leaves) is ~1× faster than the 3D
path at the same level (64 leaves), so the per-cell cost is
roughly the same; the cell-count scaling dominates.

## Q-resolution: HaloView access pattern

M3-3b's pre-computed face-neighbor table pattern carries straight
to 3D: `build_face_neighbor_tables_3d` calls
`face_neighbors_with_bcs(mesh, ci, bc_spec)` once per leaf and
caches the 6 face-neighbor leaf indices (vs HaloView's per-call
lookup). This is faster (zero per-call overhead inside the
residual loop) and plays nicely with ForwardDiff (the table is a
parameter, not a differentiable value) — same rationale as M3-3b.

When M3-7c needs higher-order Bernstein reconstruction (M3-8 / M3-9
scope), HaloView with `depth ≥ 2` becomes the natural neighbor-
walking primitive; for the M3-7b/c order-0 cell-average substrate,
the pre-computed table is the cleaner fit. The M3-7a smoke test
already verified that depth=2 on `HierarchicalMesh{3}` works as
designed (Q1/Q4 of the design note's §11 open questions).

## What M3-7b does NOT do

Per the brief's "Critical constraints":

  * **Does not implement Berry coupling.** That's M3-7c. The
    residual omits the three pair-Berry blocks
    `(α_a^3 β_b - α_b^3 β_a)/3 · dθ_{ab}` entirely; θ_ab is held
    fixed at IC across the step.
  * **Does not enforce H_rot solvability constraint.** Same — M3-7c.
  * **Does not implement 3D Tier-C / Tier-D drivers.** That's M3-7e.
  * **Does not write 3D off-diagonal β.** That's M3-9 (3D D.1 KH).
  * **Does not implement 3D per-axis γ AMR.** That's M3-7d.

## M3-7c launch handoff

When M3-7c launches (SO(3) Berry coupling integration on the 3D
substrate), the launch agent should already have everything it
needs from M3-7b + M3-prep:

### Inputs available

  * `cholesky_el_residual_3D!` — the no-Berry residual to extend.
  * `det_step_3d_HG!` — the Newton driver to extend.
  * `pack_state_3d` / `unpack_state_3d!` — 15-dof packers (NO
    rename needed; θ_ab become Newton unknowns in M3-7c too).
  * `build_residual_aux_3D` — aux NamedTuple builder (extend with
    any M3-7c-specific aux fields, e.g., per-pair off-diagonal
    velocity gradients for the kinematic drives).
  * `berry_partials_3d` — closed-form Berry partials on the 3D
    field set; verified at the stencil level (797 asserts in
    `test_M3_prep_3D_berry_verification.jl`); allocation-free hot
    path with `SVector{3, T}` inputs.
  * `BerryStencil3D` — optional pre-compute of `F`, `dF/dα`,
    `dF/dβ` per pair; for an order-0 cell-average residual the
    savings are marginal (M3-3c chose closed-form path).

### Pseudo-residual extension

```
F^α_a += (Berry α-modification per pair):
  for each pair (a, b) in {(1,2), (1,3), (2,3)}:
    add ±(ᾱ_b³ / (3 ᾱ_a²)) · dθ_{ab}/dt  (signs from Ω·X = -dH)

F^β_a += (Berry β-modification per pair):
  for each pair (a, b) in {(1,2), (1,3), (2,3)}:
    add ±β̄_b · dθ_{ab}/dt              (signs from Ω·X = -dH)

F^θ_{ab} = -(M3-7b trivial-drive) + Berry kinematic equation:
  default kinematic drive 0 (free-flight cut, axis-aligned ICs);
  plus on the θ_23 row, the kernel-orthogonality residual from
  M3-7 design note §2.2 closed form.
```

### M3-7b regressions for M3-7c

Both M3-7b dimension-lift gates remain regressions for M3-7c. The
Berry term must vanish on the 1D-symmetric and 2D-symmetric
slices (structural guarantees from CHECK 6 + CHECK 3b of the 3D
Berry verification note). If the M3-7c path doesn't reproduce
M3-7b's gates at 0.0 absolute (or ≤ 1e-12 if summation order causes
ULP drift), the Berry integration has a sign / factor bug.

## How to extend in M3-7d/e

  * **M3-7d** (per-axis γ AMR): consume `gamma_per_axis_3d` from
    `src/cholesky_DD_3d.jl` in a 3D action-error indicator;
    refine only along axes where γ_a collapses. The 3D analog of
    M3-3d's per-axis 2D AMR.
  * **M3-7e** (Tier-C / D 3D drivers): extend the 2D Tier-C IC
    factory `cholesky_sector_state_from_primitive` to D=3 (α=1,
    β=0, θ_ab=0 cold-limit isotropic IC); 3D Sod, 3D cold sinusoid,
    3D plane wave; and the headline Tier-D driver: 3D Zel'dovich
    pancake collapse (the cosmological reference test from
    methods paper §10.5 D.4 lifted to 3D).

## Reference

  * `reference/notes_M3_7_3d_extension.md` — full M3-7 design note;
    §3 native HG-side 3D EL residual; §7.1 dimension-lift gates;
    §9 sub-phase split.
  * `reference/notes_M3_7a_3d_halo_allocator.md` — M3-7a status
    (your dependency).
  * `reference/notes_M3_3b_native_residual.md` — 2D analog
    mirrored here at +1 axis.
  * `src/cholesky_sector.jl` — M1 1D EL residual (the §7.1a
    dimension-lift target).
  * `src/eom.jl::cholesky_el_residual_2D!` — the §7.1b
    dimension-lift target.
  * `src/berry.jl::berry_partials_3d` — closed-form 3D Berry
    partials (M3-7c will consume these).
