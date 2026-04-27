# M3-6 Phase 3 — 2D substrate (tracers + stoch + per-species γ)

> **Status (2026-04-26):** *Implemented + tested*. Closes M3-6 Phase
> 3 — the 2D substrate work the M3-6 Phase 4 (D.7 dust traps) and
> M3-6 Phase 5 (D.10 ISM tracers) drivers depend on.
>
> Three deliverables, each scoped to extend a 1D substrate to 2D:
>
>  • **(a) `TracerMeshHG2D`** — per-species per-cell passive scalars
>    on `HierarchicalMesh{2}` + 14-named-field 2D Cholesky-sector
>    field set, with refine/coarsen mass conservation via
>    `register_tracers_on_refine_2d!`. Pure-Lagrangian byte-exact
>    preservation (Phase 11 + M2-2 invariants on the 2D path).
>
>  • **(b) `inject_vg_noise_HG_2d!`** — per-axis VG stochastic
>    injection on the 2D field set with explicit `axes` selectivity:
>    axis-1 injection leaves axis-2 fields byte-equal and vice versa.
>    Honours the M3-6 Phase 1b 4-component β-cone via
>    `realizability_project_2d!` post-injection.
>
>  • **(c) `gamma_per_axis_2d_per_species_field`** — per-species
>    wrapper over `gamma_per_axis_2d_field` for D.7 dust-trap and
>    D.10 ISM-tracer per-species γ diagnostics. Plus a math-primitive
>    sibling `gamma_per_axis_2d_per_species(β, M_vv_per_species)` in
>    `src/cholesky_DD.jl`.
>
> **Test delta: +329 asserts** (3 new test files, 21 testsets across
> the three deliverables). Bit-exact 0.0 parity preserved on all M3-3,
> M3-4, M3-5, M3-6 Phase 0/1/2 regression suites — verified by
> running the 1D Phase-8/11/M2-1/M2-2/M2-3 + 2D M3-3d/M3-6 Phase 0/1b
> + M3-6 Phase 2 D.4 Zel'dovich tests through to byte-equal pass.

## What landed

| File | Change |
|---|---|
| `src/newton_step_HG_M3_2.jl` | EXTENDED: +613 LOC. New `TracerMeshHG2D` struct + accessors (`n_species`, `n_cells_2d`, `species_index`, `set_species!`, `advect_tracers_HG_2d!`); new `inject_vg_noise_HG_2d!` per-axis stochastic injection + `InjectionDiagnostics2D`; new `gamma_per_axis_2d_per_species_field` 3D-array walker. |
| `src/cholesky_DD.jl` | EXTENDED: +57 LOC. New `gamma_per_axis_2d_per_species(β, M_vv_diag_per_species) -> Matrix{T}` math primitive — generalises the 1D-style `gamma_per_axis_2d` to multiple species. |
| `src/action_amr_helpers.jl` | EXTENDED: +116 LOC. New `_hierarchical_mesh_dim` helper + `register_tracers_on_refine_2d!` HG refinement listener (mirrors `register_field_set_on_refine!`'s shape, operates on the `tm.tracers::Matrix{T}` storage instead of a `PolynomialFieldSet`). |
| `src/dfmm.jl` | APPEND-ONLY: +21 LOC. Re-exports `TracerMeshHG2D`, `inject_vg_noise_HG_2d!`, `InjectionDiagnostics2D`, `gamma_per_axis_2d_per_species`, `gamma_per_axis_2d_per_species_field`, plus the `set_species!` / `species_index` / `n_species` / `n_cells_2d` / `register_tracers_on_refine_2d!` / `advect_tracers_HG_2d!` accessors. |
| `test/test_M3_6_phase3_tracer_2d.jl` | NEW (~360 LOC, 88 asserts, 8 testsets). |
| `test/test_M3_6_phase3_stochastic_2d.jl` | NEW (~390 LOC, 61 asserts, 9 testsets). |
| `test/test_M3_6_phase3_gamma_per_species.jl` | NEW (~210 LOC, 180 asserts, 8 testsets). |
| `test/runtests.jl` | APPEND-ONLY: +26 LOC. New `Phase M3-6 Phase 3` testset block following Phase 2. |
| `reference/notes_M3_6_phase3_2d_substrate.md` | THIS FILE. |
| `reference/MILESTONE_3_STATUS.md` | UPDATED: M3-6 Phase 3 marked closed; Phase 4 (D.7 dust traps) ready. |

## Deliverable A: 2D `TracerMeshHG2D`

`TracerMeshHG2D{T}` is a freestanding mutable struct storing per-
species per-cell concentrations, alongside the fluid mesh + field set:

```julia
mutable struct TracerMeshHG2D{T<:Real}
    mesh::Any                         # HierarchicalMesh{2, M}
    fields::Any                       # PolynomialFieldSet (14 names)
    tracers::Matrix{T}                # (N_species, n_cells(mesh))
    names::Vector{Symbol}
end
```

### Storage convention

The matrix is sized to `n_cells(mesh)` — including non-leaf parents
— so it tracks the field set's storage contract 1-to-1. This mirrors
the pattern in `setups_2d.jl::allocate_cholesky_2d_fields`. Non-leaf
entries are unused (carry the parent's last-written concentration);
HG's `enumerate_leaves` is the canonical iterator for leaf-only
operations.

### Per-species per-leaf API

  • `TracerMeshHG2D(fields, mesh; n_species, names)` — constructor.
  • `set_species!(tm, name_or_idx, values_or_f, leaves)` — initialise
    from per-leaf vector or function `f(ci)`.
  • `species_index(tm, name)` — name → row index.
  • `n_species(tm)`, `n_cells_2d(tm)` — sizes.
  • `advect_tracers_HG_2d!(tm, dt)` — pure no-op (Lagrangian frame).

### Refinement listener: mass conservation

`register_tracers_on_refine_2d!(tm) -> ListenerHandle` registers an
HG refinement listener mirroring `register_field_set_on_refine!`'s
shape. Per event:

  • **Refine**: parent's per-species concentration `c` is copied
    piecewise-constant into all `2^D = 4` children. Equal-volume
    isotropic refinement gives `V_child = V_parent / 4`, and the
    per-species per-cell mass `c · V` is preserved bit-exactly:
    `Σ c_child · V_child = c_parent · V_parent` (since all children
    have `c_child = c_parent`).
  • **Coarsen**: children's per-species concentrations are averaged
    (volume-weighted; equal-volume children make this the plain
    arithmetic mean). Mass is again exactly preserved.
  • **Index remap**: the listener uses `event.index_remap` to relocate
    each old-mesh-cell's per-species value to its new position
    post-event (HG renumbers mesh cells on refine/coarsen).

Multi-species independence holds by construction: each row is
processed independently; refining or coarsening species `k` does
not touch species `k′ ≠ k`.

### Pure-Lagrangian byte-exact preservation (Phase 11 + M2-2 invariants)

`advect_tracers_HG_2d!` is a no-op. Combined with the variational
scheme's pure-Lagrangian frame (cell boundaries follow `x_a` strictly
in the M3-6 zero-strain limit), this gives bit-exact preservation
through arbitrary numbers of `det_step_2d_berry_HG!` calls — verified
by `tm.tracers == snapshot` after 5 deterministic steps in GATE 4 of
the test.

## Deliverable B: 2D `inject_vg_noise_HG_2d!`

Per-axis variance-gamma stochastic injection on the 2D Cholesky-
sector field set. The signature:

```julia
inject_vg_noise_HG_2d!(fields, mesh, frame, leaves, bc_spec, dt;
                       params::NoiseInjectionParams,
                       rng,
                       axes::Tuple = (1, 2),
                       ρ_ref::Real = 1.0,
                       Gamma::Real = GAMMA_LAW_DEFAULT,
                       diag::InjectionDiagnostics2D = ...,
                       proj_stats::Union{ProjectionStats,Nothing} = nothing)
    -> (fields, diag)
```

### Per-axis selectivity (load-bearing structural property)

The `axes` keyword takes a subset of `(1, 2)`:

  • `axes = (1,)`: only axis-1 fields `(β_1, u_1, s, Pp)` may be
    perturbed. Axis-2 fields `(β_2, u_2, x_2)` and the off-diag pair
    `(β_12, β_21)` are byte-equal pre/post. RNG draws happen only
    for axis 1 (one length-N buffer).
  • `axes = (2,)`: symmetric (axis-1 fields untouched).
  • `axes = (1, 2)`: both axes perturbed. RNG advances first by one
    length-N buffer for axis 1, then by another for axis 2.
  • `axes = ()`: pure no-op (no field writes).

The selectivity contract is verified at the test level by
snapshotting all 14 fields pre/post and asserting byte-equal on
the unselected axis fields (the GATE 2 / 3 testsets in
`test_M3_6_phase3_stochastic_2d.jl`).

### Per-axis recipe (per axis a in `axes`)

1. **Per-axis divu_a**: from the precomputed face-neighbor tables
   `face_lo_idx[a], face_hi_idx[a]`, compute
   `(u_a(hi) - u_a(lo)) / (x_a(hi) - x_a(lo))` per leaf. Periodic
   wrap is honoured via the `+L_a` shift on `Δx ≤ 0` (mirrors the 1D
   path's `+L_box` wrap on `j == N`).
2. **Per-axis VG noise η_a**: `rand_variance_gamma!(rng, eta_white,
   λ, θ_factor)` for `i = 1:N` in leaf order, then 3-point periodic
   smoothing into `diag.eta[a]` (when `params.ell_corr > 0`).
3. **Per-cell drift / noise / amplitude limiter / KE-debit**: drift
   `C_A · ρ · divu_a · Δt`, noise `C_B · ρ · √(max(-divu_a, 0) · Δt) · η_a`,
   amplitude limiter against IE budget, KE-debit `(P_xx, P_⊥) =
   (ρ M_vv, Pp)`, entropy re-anchored from new `M_vv`.
4. **u_a vertex update**: `u_a[ci] += δ / ρ`.

### Cold-limit `Pp = 0` floor handling

The 2D IC bridge sets `Pp = 0` (cold limit). Naively applying the
1D path's `if Pp_new < pressure_floor; Pp_new = pressure_floor`
would silently raise `Pp` from 0 to `pressure_floor` on every step
— breaking the no-noise reduction byte-exactness gate. The fix:
apply the floor ONLY when `ΔKE_vol != 0.0` (i.e. only when an
injection actually fired). Verified bit-equal against pre-IC at
`C_A = C_B = 0` on uniform IC.

### 4-component cone respected (M3-6 Phase 1b)

After per-axis injection, the optional projection
`realizability_project_2d!(fields, leaves; project_kind =
params.project_kind, ...)` enforces both:

  • The per-axis 2-component cone `M_vv,aa ≥ headroom · β_a²` (the
    M3-3d / M2-3 inheritance).
  • The 4-component cone `Q = β_1² + β_2² + 2(β_12² + β_21²) ≤
    M_vv · headroom_offdiag` (the M3-6 Phase 1b extension).

Verified by GATE 6 of the stochastic 2D test:
`n_negative_jacobian == 0` post-injection on a non-trivial
configuration with `project_kind = :reanchor`.

### Wall-time impact (16×16 = 256 leaves)

  • `inject_vg_noise_HG_2d!` (axes=(1,2), project_kind=:none): ~415 ms/call
  • `inject_vg_noise_HG_2d!` (axes=(1,), project_kind=:none): ~415 ms/call
  • `advect_tracers_HG_2d!`: < 1 ns/call (no-op)
  • `gamma_per_axis_2d_per_species_field` (n_species=3): ~0.002 ms/call

The injection wall-time is dominated by `build_face_neighbor_tables`
(~413 ms at 16×16) — the same per-step face-neighbor table cost
that `det_step_2d_berry_HG!` and `realizability_project_2d!` pay.
This is the production performance envelope for 2D paths; M3-6
Phase 1c's "sparse-Newton solver" handoff item would reduce both.

## Deliverable C: per-species per-axis γ diagnostic

Two-tier API:

  • **Math primitive** (`src/cholesky_DD.jl`):
    `gamma_per_axis_2d_per_species(β::SVector{2,T}, M_vv_diag_per_species)
    -> Matrix{T}` with shape `(n_species, 2)`. Generalises
    `gamma_per_axis_2d(β, M_vv_diag)` to multiple species, each with
    its own per-axis `M_vv_diag` 2-tuple.

  • **Field walker** (`src/newton_step_HG_M3_2.jl`):
    `gamma_per_axis_2d_per_species_field(fields, leaves;
    M_vv_override_per_species = nothing, ρ_ref = 1.0, Gamma =
    GAMMA_LAW_DEFAULT, n_species = 1) -> Array{Float64, 3}`
    with shape `(n_species, 2, length(leaves))`. Wraps
    `gamma_per_axis_2d_field` over multiple species; each species
    can carry its own `M_vv` override (e.g. dust ⇒ `M_vv = 0`,
    pressureless cold; gas ⇒ EOS Mvv(J, s)).

### Multi-species independence

Each species' γ is computed independently — no cross-species reads
or writes. The single-species `n_species == 1` path reduces byte-
equally to `gamma_per_axis_2d(β, M_vv_diag_per_species[1])` (math
primitive) or `gamma_per_axis_2d_field(fields, leaves;
M_vv_override = M_vv_override_per_species[1])` (field walker).

### Use cases

  • **D.7 dust traps**: `M_vv_dust = (0, 0)` (pressureless cold dust)
    and `M_vv_gas = (Mvv(J, s), Mvv(J, s))` (gas EOS). Per-species γ
    tracks how cold dust streams collapse independently of the gas
    equation of state. The dust-species `γ_a = 0` everywhere
    (β > 0 ⇒ γ²_a < 0 ⇒ floored).
  • **D.10 ISM tracers**: per-species `M_vv` carries species-
    dependent thermal velocity dispersion (warm/hot/cold ISM
    phases).

## Verification gates (3 testsets, 329 asserts)

| File | Testsets | Asserts |
|---|---|---:|
| `test_M3_6_phase3_tracer_2d.jl` | 8 | **88** |
| `test_M3_6_phase3_stochastic_2d.jl` | 9 | **61** |
| `test_M3_6_phase3_gamma_per_species.jl` | 8 | **180** |
| **Total** | 25 | **329** |

### Tracer 2D coverage

  • Constructor + accessors (n_species, names, validation).
  • `set_species!` vector + functional forms.
  • `advect_tracers_HG_2d!` byte-exact preservation over 100 calls.
  • Phase 11 + M2-2 invariants byte-exact under
    `det_step_2d_berry_HG!` (5 steps).
  • Refine event: parent → 4 children prolongation, mass
    conservation `Σ c_child · V_child == c_parent · V_parent`
    bit-exact, value-multiset preserved post-event.
  • Coarsen event: children → parent volume-weighted mean, mass
    conservation bit-exact, parent value `≈ (1+2+3+4)/4 = 2.5`.
  • Multi-species independence under refine/coarsen: 2-row
    independence verified via tuple-multiset.
  • Refinement listener: column dimension grows correctly
    (`size(tm.tracers, 2) == n_cells(mesh)`).

### Stochastic 2D coverage

  • `InjectionDiagnostics2D` constructor.
  • No-noise reduction (`C_A = C_B = 0`, uniform IC) byte-equal.
  • Axis-1 selectivity (`axes = (1,)` → axis-2 byte-equal):
    asserts `x_2 == x_2'`, `u_2 == u_2'`, `β_2 == β_2'`,
    `β_12 == β_12'`, `β_21 == β_21'` post-injection.
  • Axis-2 selectivity (`axes = (2,)` → axis-1 byte-equal):
    symmetric.
  • Both-axis injection (`axes = (1, 2)`) updates both axes;
    diag.eta[1] and diag.eta[2] both populated.
  • Empty `axes = ()` is a pure no-op.
  • 4-component cone respected (`project_kind = :reanchor` →
    n_negative_jacobian == 0).
  • RNG-bit reproducibility: same seed + same IC → same δ_rhou
    byte-equal across both axes.
  • Zel'dovich-pancake-like axis-aligned IC + `axes = (1,)`:
    axis-2 strict inertness (Phase 2 D.4 selectivity preserved).

### Per-species γ coverage

  • Math primitive: single-species reduces byte-equal to
    `gamma_per_axis_2d` (5 sample points × 2 axes = 10 asserts).
  • Math primitive: per-axis floor `γ² < 0 ⇒ γ = 0` per species.
  • Math primitive: dust + gas multi-species independence.
  • Math primitive: 3-species cross-check against
    `gamma_per_axis_2d` per species.
  • Field walker: single-species reduces byte-equal to
    `gamma_per_axis_2d_field` (16 leaves × 2 axes = 32 asserts).
  • Field walker: 3-species (gas/dust/hot ISM) per-species check
    on 16 leaves × 2 axes per species = 96 asserts (dust → 0
    everywhere; gas → √(1 - β²); hot ISM → √(2 - β²)).
  • Field walker: `nothing` override → EOS-derived M_vv (4 leaves
    × 2 species × 2 axes = 16 asserts).
  • Field walker: input validation (assertion errors on bad inputs).

## Honest scientific finding

The 2D `inject_vg_noise_HG_2d!` design **diverges intentionally**
from the 1D `inject_vg_noise_HG_native!` recipe in two specific
places:

  1. **Pressure floor only when injection fires.** The 2D IC bridge
     sets `Pp = 0` (cold limit) whereas M1's 1D IC sets `Pp = ρ M_vv`
     (isotropic Maxwellian). The 1D code applies `Pp_new < floor ⇒
     Pp_new = floor` unconditionally; in 2D this would silently
     raise `Pp` from 0 to the floor on every step. The fix: apply
     the floor only when `ΔKE_vol != 0` (an injection actually
     fired). This preserves cold-limit byte-exactness across no-op
     steps. The 1D path's bit-exact contract is unaffected (we did
     not modify `inject_vg_noise_HG_native!`).

  2. **u_a stored at cell-center in 2D, vertex in 1D.** The 2D
     variational scheme stores `u_a` per cell-center (one scalar
     per leaf per axis), not per-vertex. The 2D injection writes
     `u_a[ci] += δ / ρ` directly per cell — there is no
     mass-lumped vertex-velocity distribution step (which the 1D
     path needs because `u` is stored at vertices, requiring
     `m̄_i = (Δm_{i-1} + Δm_i) / 2` distribution). This is a
     storage-convention difference, not a physics divergence.

The per-axis selectivity (load-bearing for the M3-6 Phase 2 D.4
Zel'dovich handoff item) is structurally guaranteed by the
`axes::Tuple` argument: the per-axis loop body never touches axes
not in the tuple, and the RNG advances per-axis only when that axis
is selected. Verified by 30+ byte-equality asserts in the test.

## What M3-6 Phase 3 does NOT do

  • **Does not write the D.7 driver.** That's M3-6 Phase 4. The
    `TracerMeshHG2D` substrate makes the per-species mass tracking
    available; a Phase 4 IC factory `tier_d_kh_dust_ic` (and
    optional `tier_d_dust_in_vortex_ic`) builds on M3-6 Phase 1b
    `tier_d_kh_ic_full` plus per-species tracer initialization.
  • **Does not write the D.10 driver.** That's M3-6 Phase 5. The
    multi-tracer infrastructure is ready; Phase 5 needs an ISM IC
    factory and per-species transport accumulator hooks.
  • **Does not exercise large multi-step stochastic 2D runs.** The
    per-call wall-time is ~415 ms at 16×16; a 30-step stochastic 2D
    run takes ~12 s. Larger-N studies would benefit from the
    sparse-Newton solver carried forward from M3-6 Phase 1c.
  • **Does not bit-exact-cross-check 2D ⊃ 1D at axis-aligned ICs.**
    The 1D / 2D paths differ in storage convention (vertex vs cell
    `u`); a strict byte-equal cross-check would require a custom
    bridge mapping the 1D state into a degenerate 1×N 2D mesh and
    reading the cell-centered `u` values via a different accessor.
    The selectivity gate (Zel'dovich-aligned IC + `axes = (1,)` →
    axis-2 inertness) is the structural sufficient condition for
    M3-6 Phase 4/5; a strict 2D ⊃ 1D parity test is not on the
    Phase 3 critical path.

## M3-6 Phase 4 (D.7 dust traps) handoff items

  1. **D.7 dust-trap IC factory** in `src/setups_2d.jl`:
     `tier_d_kh_dust_ic_full(; level, U_jet, w, A_pert, ρ_dust,
     ρ_gas, ...)` — extends `tier_d_kh_ic_full` (Phase 1b) with a
     dust species initialised at `ρ_dust(x, y) = ρ_d0` uniform and
     a gas species at `ρ_gas(x, y) = ρ_g0`. Returns a
     `TracerMeshHG2D` alongside the `(mesh, frame, leaves, fields)`
     tuple.

  2. **D.7 driver** `experiments/D7_dust_trap.jl`:
     Build IC, run `det_step_2d_berry_HG!` for K steps (Phase 1a
     strain coupling + Phase 1b cone), record per-step:
       - per-species per-axis γ via
         `gamma_per_axis_2d_per_species_field`
       - dust-density-in-vortex centroid (track the dust species'
         center of mass over time)
       - vortex-center accumulation diagnostic (the headline gate:
         dust accumulates at vortex centers, gas does not).
     The Phase 4 acceptance gate is the methods paper §10.5 D.7
     dust-trap signature: `[dust]_vortex_center / [dust]_far > 1.5`
     by t = 5 · t_eddy.

  3. **Stochastic injection in Phase 4**: D.7 may want to apply
     `inject_vg_noise_HG_2d!` only on the gas species (the dust
     species has `M_vv = 0` and noise injection on β-0 cells is
     a no-op anyway). The current `inject_vg_noise_HG_2d!` operates
     on the fluid (β_a) state, NOT per-species. Per-species noise
     coupling would require a separate Phase 4 design.

  4. **Sparse-Newton solver** (carried forward from Phase 1c handoff):
     L=5 / L=6 D.7 sweeps require ~10 s/step at L=5 (level 4 is OK
     for the headline gate). The per-step face-neighbor table cost
     dominates; a sparse iterative solver would unlock larger meshes.

## M3-6 Phase 5 (D.10 ISM tracers) handoff items

  1. **D.10 ISM IC factory** in `src/setups_2d.jl`:
     `tier_d_ism_tracers_ic_full(; level, n_species, T_warm, T_hot,
     T_cold, ...)` — multi-phase ISM IC with N_species ≥ 3 (warm,
     hot, cold) and per-species `M_vv` carrying the species-
     dependent thermal velocity dispersion.

  2. **D.10 driver** `experiments/D10_ism_tracers.jl`:
     Build IC, run `det_step_2d_berry_HG!` + optionally
     `inject_vg_noise_HG_2d!` (per-axis stochastic injection on
     the compressive axis only — the M3-6 Phase 2 handoff item)
     for a multi-T_KH run, record per-species:
       - mass conservation per species (multi-species check on
         the `TracerMeshHG2D` matrix)
       - per-species per-axis γ via
         `gamma_per_axis_2d_per_species_field`.
     Phase 5 acceptance gate: methods paper §10.5 D.10 ISM-phase
     mixing signature.

  3. **Per-species realizability projection**: D.10's multi-phase
     ISM may need per-species M_vv-aware projection (current
     `realizability_project_2d!` is fluid-state only). Phase 5
     design item.

  4. **Per-species transport accumulator**: D.10 may want
     `BurstStatsAccumulator2D` that records per-species
     compression bursts. Current `InjectionDiagnostics2D` records
     per-axis statistics (no species dimension). Phase 5 design
     item.

## References

  • `reference/notes_M3_6_phase2_D4_zeldovich.md` — Phase 2 closure,
    immediate predecessor (the source of the multi-tracer + per-axis
    stochastic injection handoff items).
  • `reference/notes_M3_3e_3_amr_tracers_native.md` — 1D
    `TracerMeshHG` substrate the 2D variant lifts.
  • `reference/notes_M3_3e_2_stochastic_native.md` — 1D
    `inject_vg_noise_HG!` recipe the 2D variant lifts per-axis.
  • `reference/notes_M3_3d_per_axis_gamma_amr.md` — per-axis γ
    diagnostic (`gamma_per_axis_2d_field`,
    `realizability_project_2d!`) the per-species variant wraps.
  • `reference/notes_phase11_passive_tracer.md` — original 1D
    `TracerMesh` design (Phase 11).
  • `reference/notes_M2_2_multitracer.md` — multi-tracer M2-2
    refinement-mass-conservation invariants the 2D path inherits.
  • `reference/notes_phase8_stochastic_injection.md` — original 1D
    Phase 8 stochastic injection design.
  • `src/newton_step_HG_M3_2.jl` (`TracerMeshHG2D`,
    `inject_vg_noise_HG_2d!`, `gamma_per_axis_2d_per_species_field`),
    `src/cholesky_DD.jl` (`gamma_per_axis_2d_per_species`),
    `src/action_amr_helpers.jl` (`register_tracers_on_refine_2d!`).
  • `specs/01_methods_paper.tex` §10.5 D.7, §10.5 D.10 — the
    falsifier specifications Phase 3 makes ready for Phase 4 / 5.
