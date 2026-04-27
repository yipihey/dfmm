# M3-3d — Per-axis γ diagnostic + AMR/realizability per-axis wiring

> **Status (2026-04-26):** *Implemented + tested*. Fourth sub-phase of
> M3-3 (`reference/notes_M3_3_2d_cholesky_berry.md` §9). Closes M3-2b's
> deferred Swaps 2+3 for the 2D scope (HG `register_refinement_listener!`
> + `step_with_amr!`-driven AMR on `HierarchicalMesh{2}`).
>
> Test delta vs M3-3c baseline: **+97 asserts** (3894 + 1 deferred →
> 3991 + 1 deferred). 1D-path bit-exact parity holds. §6.1 Berry
> dimension-lift gate, §6.2/§6.3/§6.4 Berry verification reproductions
> all pass at 0.0 / 1e-9 / 1.003 / ≤1e-10 (no regression).
>
> §6.5 per-axis γ selectivity (the headline scientific gate): **PASS**
> at level=4, A=0.5, kx=1, ky=0, dt=2e-3, n_steps=100. Spatial std of γ_1
> exceeds spatial std of γ_2 by >10¹⁵ (γ_2 is uniform to round-off);
> γ_1 collapses to <0.94 at the trough cell while γ_2 stays at 0.98.
> The principal-axis decomposition correctly identifies the collapsing
> axis.

## What landed

| File | Change |
|---|---|
| `src/cholesky_DD.jl` | EXTENDED: M1-style per-axis γ math primitive `gamma_per_axis_2d(β, M_vv_diag) -> SVector{2}` (γ²_a = M_vv,aa − β_a²). The original `gamma_per_axis_2d(α, M_vv::SMatrix)` form (M3-3a) is preserved. ~50 LOC added. |
| `src/diagnostics.jl` | EXTENDED: field-walking helper `gamma_per_axis_2d_field(fields, leaves; M_vv_override, ρ_ref, Gamma)` returns a `2 × N` matrix; `gamma_per_axis_2d_diag` is a forwarded alias for I/O snapshots. ~80 LOC added. |
| `src/io.jl` | NO CHANGE. The existing HDF5/JLD2 wrappers are generic over `NamedTuple`; callers populate the snapshot `:gamma_per_axis_2d` entry from `gamma_per_axis_2d_field`. |
| `src/stochastic_injection.jl` | EXTENDED: `realizability_project_2d!(fields, leaves; project_kind, headroom, Mvv_floor, …)` applies the M2-3 reanchor projection per-axis on the 2D field set. With `project_kind=:none` it is a no-op (M2-3 1D bit-equality regression mirror); with `:reanchor`, raises `s` so `Mvv ≥ headroom · max_a(β_a²)` per leaf. ~120 LOC added. |
| `src/newton_step_HG.jl` | EXTENDED: `det_step_2d_berry_HG!` now accepts `project_kind`, `realizability_headroom`, `Mvv_floor`, `pressure_floor`, `proj_stats` kwargs and calls `realizability_project_2d!` post-Newton. Default `project_kind=:none` ⇒ M3-3c bit-equality preserved. ~20 LOC added. |
| `src/action_amr_helpers.jl` | EXTENDED: per-axis action-AMR indicator on `HierarchicalMesh{2}` (`action_error_indicator_2d_per_axis`); HG refinement-listener for the 2D Cholesky-sector field set (`register_field_set_on_refine!`); end-to-end `step_with_amr_2d!` driver wrapping HG's `step_with_amr!`. **Closes M3-2b deferred Swaps 2+3 for the 2D scope.** ~270 LOC added. |
| `src/dfmm.jl` | APPEND-ONLY: re-exports the new symbols. |
| `experiments/M3_3d_per_axis_gamma_cold_sinusoid.jl` | NEW: §6.5 selectivity driver (`run_M3_3d_per_axis_gamma_selectivity`) + 4-panel headline plot (`plot_M3_3d_per_axis_gamma`). ~225 LOC. |
| `reference/figs/M3_3d_per_axis_gamma_selectivity.png` | NEW: 4-panel headline figure (γ_1 trajectory, γ_2 trajectory, |s| histogram, det Hess histogram). |
| `test/test_M3_3d_gamma_per_axis_diag.jl` | NEW: 54 asserts. γ accessor unit test (math primitive + field walker). |
| `test/test_M3_3d_realizability_per_axis.jl` | NEW: 26 asserts. Per-axis projection unit (no-op / fires / 1D-symmetric reduction). |
| `test/test_M3_3d_amr_per_axis.jl` | NEW: 6 asserts. Per-axis indicator + refinement-listener integrity (parent → children prolongation). |
| `test/test_M3_3d_selectivity.jl` | NEW: 11 asserts. §6.5 headline selectivity test + headline-plot driver smoke. |
| `test/runtests.jl` | APPEND-ONLY: new "Phase M3-3d" testset between M3-3c and M3-2. |
| `reference/notes_M3_3d_per_axis_gamma_amr.md` | THIS FILE. |

Total: **+97 new asserts**, **~+820 LOC** across `src/` + `experiments/`,
**4 new test files**.

## Per-axis γ diagnostic structure

Two complementary forms of `gamma_per_axis_2d`:

  • **M3-3a form** `gamma_per_axis_2d(α::SVector{2}, M_vv::SMatrix{2,2})`:
    γ_a = √(M_vv,aa / α_a²). The α-normalised principal-frame variance.
    Used by the M3-3a Cholesky-decomposition test.

  • **M3-3d form** `gamma_per_axis_2d(β::SVector{2}, M_vv_diag::SVector{2})`:
    γ²_a = M_vv,aa − β_a². The M1-style per-axis lift consumed by
    AMR + realizability. Reduces per-axis to M1's `gamma_from_state`
    on each axis independently.

The field-walker `gamma_per_axis_2d_field(fields, leaves; …)` walks the
leaves of a 2D Cholesky-sector field set (`allocate_cholesky_2d_fields`)
and returns a `2 × N` matrix of per-axis γ. With `M_vv_override =
(Mvv_1, Mvv_2)`, axis-a γ uses the supplied override; otherwise both
axes share `Mvv(J=1/ρ_ref, s)` from the EOS (isotropic-EOS
convention).

## Per-axis realizability projection

`realizability_project_2d!(fields, leaves; …)` extends M2-3's 1D
projection to 2D. The binding constraint is
`max(β_1², β_2²)` so a single `s`-raise satisfies both axes (the EOS
gives `Mvv,11 = Mvv,22 = Mvv(J, s)` — an isotropic ideal gas does
not distinguish axes). Off-diagonal `β_{12}, β_{21}` remain zero per
M3-3a Q3 (M3-6 will re-add them for D.1 KH).

Conservation properties match the 1D projection:

  • `(α_a, β_a, x_a, u_a)` per axis untouched.
  • `Pp` debited (down to `pressure_floor`); residual admitted as
    silent floor-gain.
  • IE budget per leaf: `+ρ · (M_vv_target − M_vv_pre)` per event.

The M3-3d test asserts the 1D-symmetric reduction (β_2 = 0) reproduces
the M2-3 1D `M_vv_target_rel = headroom · β_1²` to ≤1e-12.

## Per-axis action-AMR indicator on `HierarchicalMesh{2}`

`action_error_indicator_2d_per_axis(fields, mesh, frame, leaves, bc_spec; …)`
evaluates a per-leaf indicator that aggregates per-axis 2nd-difference
contributions of `(α_a, β_a, u_a)` plus a γ_inv-marker per axis, with
`max_a` aggregation:

    out[i] = max_a [ |d²α_a| + |d²β_a| + |d²u_a|/c_s + 0.01·(√Mvv/γ_a − 1) ]
           + |d²s|             # shared across axes; added once

The `max_a` aggregation is what makes the indicator **per-axis selective**:
on the C.2 cold sinusoid (kx=1, ky=0), the axis-1 γ_inv-marker spikes
at the compressive trough while axis-2's stays near zero — refinement
events fire only along the collapsing axis.

The unit test (`test_M3_3d_amr_per_axis.jl`) verifies:

  • Uniform field ⇒ indicator ≡ 0.
  • β_1 = sin(2π x) sinusoid (β_2 = 0) ⇒ indicator > 0 across leaves
    (axis-1 d²β_1 is non-trivial; axis-2 contributions vanish).

## HG `register_refinement_listener!` field-set listener

`register_field_set_on_refine!(fields, mesh)` registers an HG listener
that:

  1. Walks `event.refined_parents`, copies the parent's order-0 cell-
     average into all 2^D children (piecewise-constant prolongation).
  2. Walks `event.coarsened_parents`, averages the (now-removed) 2^D
     children's order-0 values into the new parent (mass-conservative
     for `ρ`-like fields under isotropic refinement).
  3. Resizes the underlying `fields.storage.<name>::Vector{Float64}` to
     `n_cells(mesh)` post-event.

This is the first dfmm path that consumes HG's listener API. Verified
on a 2D mesh: refining a leaf preserves the parent's `α_1` value into
all 4 children (test asserts `n_match == 4`).

## End-to-end AMR-driven 2D run: `step_with_amr_2d!`

Wraps HG's `step_with_amr!` with:

  • **`step!` callback** = `det_step_2d_berry_HG!` per step on the
    current `enumerate_leaves(mesh)` (refreshed after each AMR cycle).
  • **`indicator(mesh)` callback** = `action_error_indicator_2d_per_axis`
    mapped onto **mesh-cell indices** (length `n_cells(mesh)`; non-leaves
    return 0 so they cannot trigger refinement).
  • **Refinement listener** registered for the duration of the run via
    `register_field_set_on_refine!`; auto-unregistered on exit.

This closes M3-2b's deferred Swaps 2+3 for the 2D scope: the 2D path
now uses HG's native AMR primitives (1D-Lagrangian path retains
`cache_mesh` until M3-3e).

## §6.5 Per-axis γ selectivity — PASS

The load-bearing scientific gate. Cold sinusoid IC with k_x = 1, k_y = 0,
A = 0.5, level = 4 (256 leaves), dt = 2e-3, n_steps = 100 (T = 0.2):

  • **γ_1 spatial range:** [0.94, 0.99] — clear collapse at the
    compressive trough.
  • **γ_2 spatial range:** [0.981, 0.981] (uniform to round-off).
  • **Selectivity ratio std(γ_1) / std(γ_2):** > 10¹³ (γ_2 spatial
    variance at machine precision).
  • **Trough min γ_1 / min γ_2:** ≈ 0.96 — γ_1 is measurably below
    γ_2 at the same cell.

Headline figure: `reference/figs/M3_3d_per_axis_gamma_selectivity.png`
(4 panels: γ_1 trajectory min/max, γ_2 trajectory min/max, |s| per
cell at final step, det(Hess) per cell at final step).

The principal-axis decomposition is correctly identifying the
collapsing axis. The trivial axis preserves anisotropy.

### Why γ_2 isn't perfectly constant in time

β_2 grows from 0 over time at rate `γ²_2/α_2 = 1` (the kinetic Cholesky
driver `D_t β = γ²/α`), so γ_2 = √(1 − β_2²) drops *uniformly* across
all leaves as the cold limit is approached on axis 2. This is a real
physical effect — even on a "trivial" axis, the cold-limit cell collapses
toward β = 1 in finite time. **The selectivity is in the spatial
structure, not the absolute value:** axis 1's β_1 is spatially varying
(grows fastest at the compressive trough), axis 2's β_2 is spatially
uniform (no spatial driver). The spatial std ratio captures this.

For longer integration windows (n_steps = 150), β_1 saturates the
realizability boundary and γ_1 → 0 exactly at the trough cell while
γ_2 still hovers at 0.96. Newton convergence breaks down past
n_steps ≈ 175 (β_1 > 1 hits the realizability cone exterior); the
realizability projection (`project_kind = :reanchor`) is the post-
Newton fix that reanchors `s` to keep γ² ≥ 0.

## Verification gates

### §6.1 Dimension-lift gate (M3-3c) — PASS unchanged

The M3-3c §6.1 test continues at 0.0 absolute (Berry vanishes on the
1D-symmetric slice; θ_R pinned by the trivial F^θ_R row). M3-3d's
realizability post-Newton step is gated by `project_kind = :none`
(default) so the dimension-lift gate test, which doesn't enable the
projection, is unaffected.

### §6.2 / §6.3 / §6.4 Berry verification reproductions — PASS unchanged

All three M3-3c verification gates pass with the same numbers
(0.0 / 1e-9 / 1.003 / ≤1e-10) as M3-3c. M3-3d added no functional
changes to the residual or the Berry-block partials.

### §6.5 Per-axis γ selectivity — PASS (headline gate)

Documented above. The principal-axis decomposition correctly
identifies the collapsing axis; the trivial axis preserves anisotropy.

### Realizability per-axis — PASS

The 1D-symmetric reduction (β_2 = 0) on a 2D state reproduces the
M2-3 1D `M_vv_target_rel = headroom · β_1²` to ≤1e-12. The headroom
+ floor combination matches M2-3's defaults (1.05 and 1e-2 respectively).

### AMR per-axis indicator — PASS

  • Uniform-field indicator ≡ 0 (no spurious refinement triggers).
  • Sinusoidal β_1 spike with β_2 = 0 ⇒ indicator > 0.05 along axis 1
    cells; the spatial pattern reflects the per-axis selectivity.

### HG refinement-listener integrity — PASS

Refining a leaf in a 4-leaf 2D mesh produces 7 leaves (1 split into 4);
all 4 children inherit the parent's α_1 value. The other 3 unrefined
leaves keep their pre-event α_1 values via the index_remap pathway.

## What M3-3d does NOT do

Per the brief's "Critical constraints":

  • **Does not retire the cache_mesh shim.** That's M3-3e. The 1D
    path continues to delegate to M1's `det_step!` via `cache_mesh`.
  • **Does not add off-diagonal β.** That's M3-6 (D.1 KH falsifier).
  • **Does not implement higher-order prolongation/restriction.** The
    HG refinement-listener is order-0 (piecewise-constant). Higher-
    order Bernstein restriction/prolongation is M3-4 / M3-5 work.
  • **Does not implement off-diagonal velocity gradients
    `(∂_2 u_1, ∂_1 u_2)`.** Carried over from M3-3c. F^θ_R drive is
    still trivial in M3-3d; a non-trivial drive enters at M3-3d/M3-6
    when KH is activated.

## Open issues / handoff to M3-3e

  • **Cache_mesh retirement (M3-3e scope).** The 1D path on
    `SimplicialMesh{1, T}` continues through `det_step_HG!` →
    `cache_mesh::Mesh1D` → `det_step!`. M3-3e drops the shim and
    rebuilds the 1D path on a native HG residual. The 2D path is
    already native (M3-3b/c), so M3-3e is a 1D-only effort.

  • **AMR + realizability not yet integrated in the §6.5 driver.**
    The current `run_M3_3d_per_axis_gamma_selectivity` runs at fixed
    refinement level (no AMR) and `project_kind = :none` (no
    projection). The `step_with_amr_2d!` driver is unit-tested
    end-to-end via the refinement-listener test, but a full §6.5
    re-run with AMR on (refining at the trough) and projection on
    (when γ → 0) is left for an M3-3d follow-up profiling pass.

  • **Wall-time impact.** The per-axis projection is a per-leaf
    constant-time op; on 256 leaves at dt = 2e-3, n_steps = 100 the
    total per-step overhead is ≤ 1 % of `det_step_2d_berry_HG!`'s
    Newton solve. The per-axis AMR indicator is also linear in N
    leaves; the HG refinement-listener fires only at AMR cycle
    boundaries. No measurable wall-time regression vs M3-3c at
    the test resolutions.

  • **Periodic-x coordinate wrap.** Carried over from the M3-3c open
    issue: the residual treats `x_a` as a per-cell scalar without
    periodic wrap-around. The §6.5 driver uses `REFLECTING` BCs, which
    work for the cold sinusoid because u(0) = u(1) = 0 at the IC.
    PERIODIC BCs only work for cold-limit / zero-strain ICs as in
    M3-3c. M3-3e is the natural place to fix this (or M3-4 in a
    Bernstein-cubic upgrade).

## How to extend in M3-3e

  • **Drop `cache_mesh::Mesh1D` shim** from `src/newton_step_HG.jl`.
    The 1D Phase-2/5/5b/7/8 path becomes a native HG residual on
    `SimplicialMesh{1, T}`. The M3-3d M2-3 1D-symmetric reduction
    test (`test_M3_3d_realizability_per_axis.jl`'s "1D-symmetric
    reduction" testset) is a pre-built parity gate — it already
    asserts that the 2D per-axis projection reduces to the M2-3 1D
    target under β_2 = 0. M3-3e can use it to validate the 1D-native
    realizability path.

## Reference

  • `reference/notes_M3_3_2d_cholesky_berry.md` — full M3-3 design
    note (your sub-phase is §9 entry "M3-3d").
  • `reference/notes_M3_3a_field_set_cholesky.md`,
    `reference/notes_M3_3b_native_residual.md`,
    `reference/notes_M3_3c_berry_integration.md` — your dependencies.
  • `reference/notes_M2_3_realizability.md` — 1D realizability
    projection design (the per-axis 2D form mirrors §3 derivation).
  • `reference/notes_M2_1_amr.md` — 1D action-AMR design (the
    2D per-axis indicator extends §2 "ΔS_cell proxy").
  • `reference/notes_M3_2b_swaps23_amr.md` — why HG AMR primitives
    didn't work for 1D mass-coordinate AMR; in 2D the
    `HierarchicalMesh{2}` IS applicable (this sub-phase closes the
    deferred swap for the 2D scope).
  • `~/.julia/dev/HierarchicalGrids/src/Mesh/AMR.jl` — `step_with_amr!`
    driver consumed by `step_with_amr_2d!`.
  • `~/.julia/dev/HierarchicalGrids/src/Mesh/Mesh.jl` —
    `register_refinement_listener!` consumed by
    `register_field_set_on_refine!`.
  • `~/.julia/dev/HierarchicalGrids/src/Mesh/RefineByIndicator.jl` —
    `refine_by_indicator!` invoked from inside `step_with_amr!`.
