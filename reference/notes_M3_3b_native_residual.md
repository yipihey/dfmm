# M3-3b — Native HG-side 2D EL residual (no Berry; θ_R fixed)

> **Status (2026-04-26):** *Implemented + tested*. Second sub-phase of
> M3-3 (`reference/notes_M3_3_2d_cholesky_berry.md` §9). Adds the first
> native HG-side EL residual on the 2D substrate.
>
> Test delta vs M3-3a baseline: **+213 asserts** (3554 + 1 deferred →
> 3767 + 1 deferred). 1D-path bit-exact parity holds. Dimension-lift
> parity gate (§6.1 of the design note): **0.0 absolute** to M1's
> Phase-1 zero-strain trajectory across single-step + 100-step runs on
> 4×4 and 8×8 meshes.

## What landed

| File | Change |
|---|---|
| `src/eom.jl` | EXTENDED: 2D Cholesky-sector EL residual `cholesky_el_residual_2D!`; flat ↔ field-set packers `pack_state_2d` / `unpack_state_2d!`; per-axis face-neighbor table builder `build_face_neighbor_tables`; aux NamedTuple constructor `build_residual_aux_2D`. ~280 LOC added. |
| `src/newton_step_HG.jl` | EXTENDED: 2D Newton driver `det_step_2d_HG!` consuming the M3-3 design note's pieces — `cholesky_el_residual_2D!` for the residual rows, M3-2b Swap 6's HG `cell_adjacency_sparsity` Kron-producted with an 8×8 dense block for the Jacobian sparsity, M3-2b Swap 8's `FrameBoundaries{2}` for BC handling. ~120 LOC added. |
| `src/dfmm.jl` | APPEND-ONLY: re-exports the new symbols. |
| `test/test_M3_3b_2d_zero_strain.jl` | NEW: 173 asserts. Cold-limit zero-strain regression: residual = 0 to machine precision; one-step preservation; pack/unpack round-trip; face-neighbor table sanity. |
| `test/test_M3_3b_dimension_lift_zero_strain.jl` | NEW: 21 asserts. The §6.1 critical gate. Single-step + 100-step + 8×8 mesh + axis-swap symmetry, all passing at 0.0 absolute (well within ≤ 1e-12). |
| `test/runtests.jl` | APPEND-ONLY: new "Phase M3-3b" testset between M3-3a and M3-2. |
| `reference/notes_M3_3b_native_residual.md` | THIS FILE. |

## Residual structure

Per-cell unknowns (8 dof per leaf):

    (x_1, x_2, u_1, u_2, α_1, α_2, β_1, β_2)

Entropy `s` is frozen across the Newton step (mirrors M1; entropy
updates are operator-split). `Pp`, `Q`, `θ_R` ride along on the
field-set storage but are NOT Newton unknowns in M3-3b — `θ_R` is
held fixed, `Pp` and `Q` are post-Newton sectors. Off-diagonal
`β_{12}, β_{21}` are omitted from the field set (per Q3 of the design
note).

The residual rows per axis `a ∈ {1, 2}` are the per-axis lift of M1's
1D `det_el_residual` (`src/cholesky_sector.jl`):

    F^x_a    = (x_a^{n+1} - x_a^n)/dt - ū_a
    F^u_a    = (u_a^{n+1} - u_a^n)/dt + (P̄_a^hi - P̄_a^lo) / Δm_a
    F^α_a    = (α_a^{n+1} - α_a^n)/dt - β̄_a                          [D_t^{(0)}]
    F^β_a    = (β_a^{n+1} - β_a^n)/dt + (∂_a u_a) β̄_a - γ²_a / ᾱ_a   [D_t^{(1)}]

with `γ²_a = M̄_vv,a - β̄_a²` per axis.

## Pressure stencil (M3-3b simplification)

The M3-3b pressure stencil is `P = ρ_ref * M_vv` evaluated per axis:

  • The reference density `ρ_ref` is supplied as a kwarg and held
    constant across cells (zero-strain IC convention).
  • The cell `M_vv,a` comes from either an explicit `M_vv_override`
    NTuple (the M3-3b unit-test path, decoupled from EOS specifics) or
    the EOS branch `Mvv(J, s)` with `J = cell_extent / Δm_a` (used by
    the EOS-driven cold-limit test).
  • Face pressures are the mid-point average of the two adjacent cells'
    `P` values; with uniform IC this gives `P̄_face = P_self` per axis,
    so the F^u_a momentum residual is identically zero — exactly what
    the dimension-lift gate requires.

This simplified stencil suffices for the M3-3b zero-strain regression
and the §6.1 dimension-lift gate. M3-3c will reintroduce the proper
J-dependent EOS coupling so genuine compressible 2D dynamics work.

## Q-resolution: dimension-lift gate (§6.1) — the critical M3-3b gate

The dimension-lift parity gate is the single most important M3-3b
acceptance criterion: **the 2D code in a 1D-symmetric configuration
must reproduce M1's 1D bit-exact results to ≤ 1e-12**.

Configuration:
- Active axis a=1: M_vv_1 = 1, α_1 = 1, β_1 = 0, M1 Phase-1 zero-strain.
- Trivial axis a=2: M_vv_2 = 0 (cold-limit fixed point), α_2 = const,
  β_2 = 0, no spatial coupling.
- 4×4 + 8×8 balanced 2D meshes, REFLECTING BCs.

Result: per-cell `(α_1, β_1)` matches M1's `cholesky_step` /
`cholesky_run` to **bit-exact 0.0 absolute** across:

  • Single step at dt = 1e-3 and dt = 1e-5
  • 100-step run at dt = 1e-3 (T = 0.1, the M1 Phase-1 trajectory
    `α(t)=√(1+t²), β(t)=t/√(1+t²)`)
  • 4×4 mesh (16 leaves) and 8×8 mesh (64 leaves)
  • Axis-swap symmetry: same agreement when the active axis is axis 2

The 0.0 result (vs the 1e-12 tolerance) means the 2D residual's
per-axis Cholesky-sector reduction is structurally identical to M1's
1D residual modulo the residual's own packing — exactly the M3-3b
contract.

## Q-resolution: HaloView access pattern

M3-3a's smoke test (`test/test_M3_3a_halo_smoke.jl`, commit `eb08135`)
documented that HaloView returns `PolynomialView` (NOT a `Tuple`),
indexed by `[1]` for the order-0 cell-average coefficient. M3-3b's
residual does NOT use HaloView directly — instead, we pre-compute a
**face-neighbor index table** via `face_neighbors_with_bcs(mesh, ci, bc)`
once per Newton step and consume that table inside the residual.

Why? HaloView is allocation-free for *one* coefficient column lookup,
but the residual reads ~4 fields × 4 neighbours × N cells per call, and
ForwardDiff differentiates through it. The pre-computed table gives
the same neighbour information (with periodic-wrap already resolved
upstream by `face_neighbors_with_bcs`) at zero per-call overhead and
plays nicely with AD because the table is a parameter — not a
differentiable value.

When the M3-3c residual needs higher-order Bernstein reconstruction
(M3-4 / M3-5 scope), HaloView with `depth ≥ 2` becomes the natural
neighbour-walking primitive — but for the M3-3b/c order-0 cell-average
substrate, the pre-computed table is the cleaner fit.

## What M3-3b does NOT do

Per the brief's "Critical constraints":

  • **Does not implement Berry coupling.** That's M3-3c. The residual
    omits the `(1/3)(α_1³β_2 - α_2³β_1) dθ_R` term entirely; θ_R is
    held fixed at IC across the step and is NOT a Newton unknown.
  • **Does not enforce H_rot solvability constraint.** Same — M3-3c.
  • **Does not retire the cache_mesh shim.** The 1D path continues to
    delegate to M1's `det_step!` via `cache_mesh::Mesh1D`. M3-3e
    retires it.
  • **Does not implement a J-dependent EOS coupling.** The M3-3b
    pressure stencil is `P = ρ_ref * M_vv` (constant ρ_ref). For
    zero-strain ICs density does not change so this is exact;
    genuinely compressible flow needs the full `Mvv(J, s)` per-cell
    EOS coupling, which M3-3c will reinstate alongside Berry.
  • **Does not implement per-axis γ AMR.** That's M3-3d.

## Open issues / handoff to M3-3c

  • **Periodic wrap on x_a coordinates.** The M3-3b pressure stencil
    treats `x_a` as a per-cell scalar but does not handle the periodic
    coordinate wrap that genuine 2D advection needs (cf. M1's 1D
    `det_el_residual` with `wrap = j == N ? L_box : 0`). For
    zero-strain ICs `x_a^{n+1} = x_a^n` so the wrap doesn't fire; for
    M3-3c's active-strain configurations the wrap-aware coordinate
    arithmetic must be added. The M3-3b test suite uses REFLECTING BCs
    to sidestep this; PERIODIC BCs only work for cold-limit ICs.
  • **NonlinearSolve sparse-coloring with the Kron 8×8 block.** The
    `cell_adjacency_sparsity ⊗ ones(8, 8)` construction was verified
    structurally (Newton converges, residuals at 0.0). M3-3c should
    cross-check the coloring via `SparseConnectivityTracer` once the
    Berry block introduces genuine off-cell coupling that the
    `cell_adjacency_sparsity ⊗ dense` pattern must capture.
  • **`mesh.balanced == true` enforced.** Coarse-fine face handling
    is deferred to M3-5.
  • **Wall-time ratio.** 2D-on-HG / 1D-on-cache_mesh ≈ 3.3× at
    N = 16 (4×4 mesh in 2D vs 16-cell 1D path). Reasonable given the
    8N × 8N vs 4N × 4N Jacobian-system size, but a future profiling
    pass could exploit threading or a tighter explicit-Euler initial
    guess to close the gap.

## How to extend in M3-3c/d/e

  • **M3-3c** (Berry coupling): plumb `src/berry.jl::berry_partials_2d`
    into the residual so `θ_R` becomes a Newton unknown and the
    `R_θR` row is non-trivial. Extend the field-set unknown vector
    from 8 to 9 dof per cell. Re-verify the dimension-lift gate
    (the Berry term must vanish on the 1D-symmetric slice — see
    `notes_M3_phase0_berry_connection.md` §6 iso-pullback check).
  • **M3-3d** (per-axis γ AMR): consume `gamma_per_axis_2d` from
    `src/cholesky_DD.jl` in the action-error indicator; refine only
    along axes where γ collapses (Tier-C C.2 selectivity test).
  • **M3-3e** (cache_mesh retirement): drop the 1D `cache_mesh::Mesh1D`
    shim from `src/newton_step_HG.jl` and rebuild the 1D path on the
    native HG residual. Cross-check with the full M1+M2+M3
    regression suite.

## Reference

  • `reference/notes_M3_3_2d_cholesky_berry.md` — full M3-3 design
    note (your sub-phase is §9 entry "M3-3b").
  • `reference/notes_M3_3a_field_set_cholesky.md` — M3-3a status
    (your dependency).
  • `reference/notes_M3_2b_swaps68_sparsity_bc.md` — M3-2b Swaps 6+8
    (sparsity + BCKind APIs you consume).
  • `src/cholesky_sector.jl` — M1 1D EL residual (the dimension-lift
    target).
  • `~/.julia/dev/HierarchicalGrids/src/Storage/HaloView.jl` — HG
    HaloView API (M3-3a recorded the `PolynomialView` return contract).
  • `~/.julia/dev/HierarchicalGrids/src/Mesh/Neighbors.jl` —
    `cell_adjacency_sparsity` and `face_neighbors_with_bcs` APIs.
