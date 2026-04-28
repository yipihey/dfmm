# M4 Phase 3 — per-species momentum coupling (D.7 follow-up)

> **Status (2026-04-26):** *Implemented + tested + HONESTLY PARTIAL*.
> Third sub-phase of M4 (`reference/notes_M4_plan.md`). Closes the
> M3-6 Phase 4 D.7 dust-traps honest-finding loop on the **substrate
> level**: `PerSpeciesMomentumHG2D` extends `TracerMeshHG2D` with
> per-species (u_x, u_y) per cell + per-species drag relaxation
> timescale `τ_drag` + per-species Lagrangian position offsets. The
> drag relaxation kernel is exponential Stokes-drag toward gas; the
> position kernel is kinematic; the remap kernel deposits species
> mass back to per-cell concentration.
>
> Test delta: **+1708 asserts** in one new test file
> (`test_M4_phase3_per_species_momentum.jl`, 8 GATEs / 8 testsets).
> Bit-exact opt-in contract verified by GATE 7 (no
> `PerSpeciesMomentumHG2D` constructed ⇒ M3-6 Phase 3 / Phase 4
> path byte-equal).
>
> **Verdict: PARTIAL.** The substrate, kernel, and IC factory work
> correctly (verified by GATE 2/3/4/6). Differentiation of τ_drag
> regimes is *not* observed at the M3-6 Phase 4 Taylor-Green vortex
> IC because the cold-limit Cholesky-sector residual leaves the
> incompressible vortex equilibrium *static* (gas velocity byte-
> stable across `det_step_2d_berry_HG!`). With co-moving dust IC,
> all τ_drag regimes integrate the same constant velocity field
> and produce identical drift. The driver provides a `u_dust_offset`
> knob to introduce a controlled IC bias that does differentiate
> regimes (GATE 6 verified). The literal D.7 centrifugal-
> accumulation prediction in the strict sense (driven by gas time-
> evolution + size-dependent inertia) requires either (a) a non-
> stationary base flow, or (b) explicit centrifugal-force
> computation in the drag kernel — both deferred to M4 Phase 4 or
> beyond.

## What landed

| File | Change |
|---|---|
| `src/newton_step_HG_M3_2.jl` | EXTENDED (+325 LOC, append-only): `PerSpeciesMomentumHG2D{T}` struct + constructor; `n_species`/`species_index` overloads; `drag_relax_per_species!` exponential Stokes-drag kernel; `advance_positions_per_species!` kinematic position update; `accumulate_species_to_cells!` mass-conservative remap; `dust_peak_over_mean_remapped` diagnostic (peak/mean + collapse_fraction + n_occupied). |
| `src/setups_2d.jl` | EXTENDED (+99 LOC, append-only): `tier_d_dust_trap_per_species_ic_full` IC factory (wraps `tier_d_dust_trap_ic_full` + adds `PerSpeciesMomentumHG2D`; supports `u_dust_offset` keyword for biased dust IC). |
| `src/dfmm.jl` | APPEND-ONLY: re-exports under "Phase M4-3 API" comment block. |
| `experiments/D7_dust_traps.jl` | EXTENDED (+517 LOC, append-only): `run_D7_dust_traps_per_species` driver; `run_D7_dust_traps_per_species_sweep` three-regime sweep; `plot_D7_dust_traps_per_species` 4-panel headline plot; `dust_total_mass_remapped` helper. |
| `test/test_M4_phase3_per_species_momentum.jl` | NEW (~310 LOC, 1708 asserts, 8 testsets / 8 GATEs). |
| `test/runtests.jl` | APPEND-ONLY: new `Phase M4-3` testset block following Phase M4-2. |
| `reference/MILESTONE_4_STATUS.md` | UPDATED: M4-3 row + close synthesis. |
| `reference/figs/M4_phase3_dust_accumulation.png` | NEW: 4-panel headline plot (CairoMakie). |
| `reference/notes_M4_phase3_per_species_momentum.md` | THIS FILE. |

## Mathematical structure

### Per-species momentum extension

For each species `k ∈ {1, …, K}` (default K=2: gas + dust) and
each cell `ci`:

```
u_per_species[k, axis, ci] — per-species per-axis velocity
dx_per_species[k, axis, ci] — Lagrangian position offset relative
                                to cell centre at IC
τ_drag_per_species[k] — Stokes-drag relaxation timescale
```

### Drag relaxation kernel

Exponential relaxation toward gas velocity per species per axis:

```
u_k[a]_{n+1} = u_k[a]_n + (1 − exp(−dt/τ_k)) · (u_gas[a] − u_k[a]_n)
```

Limits:
  • `τ_k = 0`: tightly-coupled (passive scalar limit). One-step
    convergence: `u_k = u_gas` exactly.
  • `τ_k = ∞`: decoupled (free particles). `u_k` unchanged.
  • `0 < τ_k < ∞`: Stokes-drag relaxation; for `dt ≪ τ_k` the
    relaxation rate is `dt/τ_k`; for `dt ≫ τ_k` it converges in
    O(1) steps.

### Position update

Kinematic per-species:

```
dx_k[a]_{n+1} = dx_k[a]_n + dt · u_k[a]
```

A species that tracks the gas perfectly (τ → 0) accumulates the
gas Lagrangian path; a decoupled species (τ → ∞) integrates its IC
velocity.

### Remap to per-cell concentration

`accumulate_species_to_cells!(psm, frame, leaves; k, lo, hi, wrap)`
deposits each source cell's concentration `c_i` at the destination
cell whose centre is closest to `(cx_i + dx_k_i, cy_i + dy_k_i)`
(periodically wrapped on each axis when `wrap=true`). Total mass
is preserved exactly: `sum(new_c) == sum(c_i)`.

The `dust_peak_over_mean_remapped` diagnostic returns:
  • `peak / mean` — measure of accumulation (≥ 1)
  • `n_occupied` — count of cells receiving any deposit (`< N` ⇒
    collapse onto a smaller subset)
  • `collapse_fraction` — `1 − n_occupied / N` (0 = no collapse, 1
    = all dust onto one cell)

## Numerical results (level 3, 8×8 = 64 cells, T_factor = 0.1)

### Three τ_drag regimes with `u_dust_offset = 0.5`

| Quantity | τ_drag = 1e-6 (tight) | τ_drag = 0.1 (intermediate) | τ_drag = 1e6 (decoupled) |
|---|---:|---:|---:|
| u_dust_speed_mean[1] | 0.794 | 0.794 | 0.794 |
| u_dust_speed_mean[end] | 0.683 | 0.697 | 0.794 |
| max|dx_dust|[end] | 0.087 | 0.117 | 0.136 |
| peak/mean (remapped)[end] | 2.025 | 2.025 | 2.025 |
| collapse_fraction[end] | 0.250 | 0.125 | 0.250 |
| M_dust_err_max | 0.0 | 0.0 | 0.0 |
| sum(n_negative_jacobian) | 0 | 0 | 0 |
| nan_seen | false | false | false |
| wall_time_per_step | 0.20 s | 0.04 s | 0.03 s |

The u_dust_speed_mean[end] differentiates the three regimes
correctly:
  • Tightly-coupled: dust speed converges to gas speed (0.683 =
    average |u_gas| over the Taylor-Green vortex array).
  • Decoupled: dust speed stays at IC value (0.794 = 0.683 + ~0.11
    average |offset bias|).
  • Intermediate: dust speed in between (0.697; closer to gas
    because dt/τ_drag = 0.05/0.1 = 0.5, so each step relaxes ~40%
    toward gas).

The max|dx_dust|[end] differentiates by drift magnitude:
  • Tightly-coupled: drift = ∫₀ᵀ u_gas dt ≈ 0.087 (purely gas-
    driven).
  • Decoupled: drift = ∫₀ᵀ u_dust_IC dt ≈ 0.136 (purely IC-driven).
  • Intermediate: drift = ∫₀ᵀ u_dust(t) dt ≈ 0.117.

### Honest finding: identical peak/mean across τ regimes

`peak/mean (remapped)[end] = 2.025` for all three regimes is a
*structural artefact* of the L=3 mesh + Taylor-Green periodic
displacement structure: with 64 source cells displaced by ~Δx/2
in regions of |u| ~ U_0, multiple sources collapse onto the same
destination, producing concentration sums of 2.0 at piled-up
cells. The peak/mean ratio is therefore a quantization of the
displacement field, not a centrifugal-accumulation signal.

The `collapse_fraction` shows the regime difference: tight =
decoupled = 0.25 (16/64 cells empty), intermediate = 0.125 (8/64
cells empty), because the intermediate regime has a smaller drift
that lands more sources on their own cells.

### Without u_dust_offset (default IC, co-moving dust)

| τ_drag | peak/mean[end] | dx_max[end] |
|---:|---:|---:|
| 1e-6 | 2.025 | 0.087 |
| 0.1 | 2.025 | 0.087 |
| 1e6 | 2.025 | 0.087 |

Identical across regimes — confirms the static-gas finding. The
gas velocity field is byte-stable through `det_step_2d_berry_HG!`
on the Taylor-Green vortex IC (the cold-limit Cholesky-sector
residual is at equilibrium for incompressible ∇·u=0 + uniform
pressure ICs); with u_dust_IC = u_gas_IC, all τ regimes integrate
the same constant field.

## Verification gates (8 testsets, 1708 asserts)

| GATE | Description | Asserts |
|---|---|---:|
| 1 | PerSpeciesMomentumHG2D IC allocator (co-moving u_dust + offset) | 646 |
| 2 | drag_relax_per_species! exponential limits (τ→0, τ→∞, intermediate) | 194 |
| 3 | advance_positions_per_species! kinematics (dx ← dx + dt·u) | 448 |
| 4 | accumulate_species_to_cells! mass conservation | 3 |
| 5 | 4-component realizability (n_neg_jac = 0) under per-species momentum | 3 |
| 6 | τ_drag regime differentiation (u_dust_offset = 0.5) | 8 |
| 7 | Bit-exact regression at zero per-species momentum (M3-6 Phase 4 path byte-equal) | 388 |
| 8 | Driver smoke at L=3 T_factor=0.1 (NamedTuple shape + finiteness + collapse_fraction bounds) | 18 |
| | **Total** | **1708** |

Wall-time of full M4-3 test suite: ~17 s (GATE 5/6 dominate).

## Bit-exact regression contract

GATE 7 verifies the M4 Phase 3 opt-in contract:

  • Two M3-6 Phase 4 ICs constructed back-to-back have byte-equal
    tracer matrices and field state.
  • `advect_tracers_HG_2d!(tm, dt)` is still a no-op on
    `TracerMeshHG2D` (Phase 3 contract preserved; M4 Phase 3 does
    not perturb the underlying `TracerMeshHG2D` state).
  • Constructing a `PerSpeciesMomentumHG2D` over one IC's tracer
    mesh does not touch the other IC's tracer mesh or fields.
  • Calling `drag_relax_per_species!` and
    `advance_positions_per_species!` on one psm does not touch the
    sibling IC's fields or tracer matrix.

All M3 + M4 Phase 1/2 regression suites stay byte-equal because
the M4 Phase 3 implementation is purely additive: no edits to
`TracerMeshHG2D`, `advect_tracers_HG_2d!`, `det_step_2d_berry_HG!`,
or any pre-existing residual / projection / BC code.

## What M4 Phase 3 does NOT do

  • **Does not extend the gas-fluid Newton residual.** The
    per-species momentum extension is *kinematic only*; the gas
    Cholesky-sector residual is unchanged. There is no two-way
    momentum exchange between gas and dust (the back-reaction
    `+ ρ_dust (u_dust − u_gas) / τ_drag` on gas is not added).
  • **Does not implement explicit centrifugal-force computation.**
    The drag kernel relaxes u_dust toward u_gas; it does not
    compute `-(u_dust · ∇)u_dust` or any explicit curvature term.
    For a Lagrangian dust particle in a steady vortex this is the
    omitted physics that would produce inward spiral drift.
  • **Does not couple per-species momentum to the action AMR
    indicator.** The M2-1 + M3-3d action-AMR refines on per-axis
    γ collapse markers; per-species momentum is not part of the
    refinement signal.
  • **Does not extend to 3D** (`PerSpeciesMomentumHG3D` is left
    for M4 Phase 4 with `tier_d_dust_trap_3d_per_species_ic_full`).
  • **Does not exercise refine/coarsen events** — the
    `PerSpeciesMomentumHG2D` storage is not yet hooked into HG's
    refinement listener (a 4-D analog of
    `register_tracers_on_refine_2d!` is needed). This is fine for
    static-mesh runs (Phase 4 acceptance is L=3 T_factor=0.1
    static); for AMR usage a listener registration is needed.

## M4 Phase 4 (next) handoff items

  1. **Centrifugal-force kernel** (`apply_centrifugal_drift!`):
     For each species k and cell ci, compute `(u_k · ∇)u_gas` via
     finite differences across the cell's four neighbors, and apply
     `du_k = -dt · (u_k · ∇)u_gas` as an inertial drag-like term.
     This activates inward spiral drift for decoupled dust in a
     steady vortex.
  2. **Two-way momentum exchange**: extend
     `cholesky_el_residual_2D_berry!` with a back-reaction term
     `Σ_k≠gas ρ_k (u_k − u_gas) / τ_drag_k` on the gas momentum
     equation. This requires the per-species ρ_k (currently
     implicit in tm.tracers as concentration · ρ_per_cell).
  3. **3D per-species momentum extension**:
     `PerSpeciesMomentumHG3D` + `tier_d_dust_trap_3d_per_species`
     + `run_D7_dust_traps_3d_per_species`. The 2D ⊂ 3D selectivity
     test (z-symmetric IC reproduces 2D verdicts byte-equal) lifts
     directly from M4 Phase 2.
  4. **Refine/coarsen listener** for `PerSpeciesMomentumHG2D`:
     when HG resizes the field set, the per-species `u`, `dx`
     arrays must be resized + populated identically. Mirrors
     `register_tracers_on_refine_2d!`.
  5. **HG `Backend`-parameterized `PolynomialFieldSet` blocker**
     (M3-8c) is still open — needed for the full Metal/CUDA port
     of the per-leaf residual kernel.
  6. **Non-stationary IC**: replace the steady Taylor-Green vortex
     with a time-evolving base flow (e.g., decaying Taylor-Green
     under M3-7d viscosity, or a forced KH where the base flow
     has explicit time dependence). This activates differential
     drag on top of the kinematic-only kernel.

## References

  • `reference/notes_M3_6_phase4_D7_dust_traps.md` — the M3-6
    Phase 4 honest finding this M4-3 closes (substrate level).
  • `reference/notes_M3_6_phase3_2d_substrate.md` — Phase 3 closure
    (`TracerMeshHG2D` + per-species γ).
  • `reference/notes_M4_plan.md` §M4-2 — the plan that this phase
    closes (renumbered to M4-3 in shipped order).
  • `reference/MILESTONE_4_STATUS.md` — M4-3 row + close synthesis.
  • `src/newton_step_HG_M3_2.jl::PerSpeciesMomentumHG2D` — struct.
  • `src/newton_step_HG_M3_2.jl::drag_relax_per_species!` — kernel.
  • `src/newton_step_HG_M3_2.jl::advance_positions_per_species!`
    — kernel.
  • `src/newton_step_HG_M3_2.jl::accumulate_species_to_cells!` —
    remap.
  • `src/setups_2d.jl::tier_d_dust_trap_per_species_ic_full` — IC.
  • `experiments/D7_dust_traps.jl::run_D7_dust_traps_per_species`
    — driver.
  • `test/test_M4_phase3_per_species_momentum.jl` — acceptance
    gates.
  • `reference/figs/M4_phase3_dust_accumulation.png` — headline plot.
