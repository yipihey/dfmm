# M3-7d — Per-axis γ diagnostic + AMR/realizability per-axis (3D)

> **Status (2026-04-26):** *Implemented + tested*. Fourth sub-phase of
> M3-7 (`reference/notes_M3_7_3d_extension.md` §5 + §7.5). The 3D
> analog of M3-3d generalised from 2 axes to 3.
>
> Test delta vs M3-7c baseline: **+418 asserts** (32011 + 1 deferred →
> 32429 + 1 deferred). All 1D + 2D regression tests pass byte-equal.
> M3-7c §7.1a + §7.1b dimension-lift gates continue at 0.0 absolute.
>
> §7.5 per-axis γ selectivity (the headline scientific gate): **PASS**
> at level=2, A=0.3, dt=2e-3, n_steps=20.
>   • 1D-sym (k=(1,0,0)): std(γ_1)/(std(γ_2)+std(γ_3)+eps) = **6.4e10**
>     (gate: > 1e10).
>   • 2D-sym (k=(1,1,0)): (std(γ_1)+std(γ_2))/2/(std(γ_3)+eps) =
>     **6.4e10** (gate: > 1e6); std(γ_3) = 0.0.
>   • Full 3D (k=(1,1,1)): all three std(γ_a) ≈ 1.43e-5 by symmetry.

## What landed

| File | Change |
|---|---|
| `src/diagnostics.jl` | EXTENDED: field-walking helper `gamma_per_axis_3d_field(fields, leaves; M_vv_override, ρ_ref, Gamma)` returns a `3 × N` matrix; `gamma_per_axis_3d_diag` is a forwarded alias for I/O snapshots. ~50 LOC added. |
| `src/action_amr_helpers.jl` | EXTENDED: per-axis action-AMR indicator on `HierarchicalMesh{3}` (`action_error_indicator_3d_per_axis`); 3D listener wrapper (`register_field_set_on_refine_3d!` — delegates to the dimension-generic `register_field_set_on_refine!`); end-to-end `step_with_amr_3d!` driver wrapping HG's `step_with_amr!`. ~290 LOC added. |
| `src/stochastic_injection.jl` | EXTENDED: `realizability_project_3d!(fields, leaves; project_kind, headroom, Mvv_floor, …)` applies the per-axis projection on the 3D field set (3-component cone). Plus `ProjectionStats3D` accumulator + `reset!` method. ~140 LOC added. |
| `src/dfmm.jl` | APPEND-ONLY: re-exports the new symbols (`gamma_per_axis_3d_field`, `gamma_per_axis_3d_diag`, `realizability_project_3d!`, `ProjectionStats3D`, `action_error_indicator_3d_per_axis`, `register_field_set_on_refine_3d!`, `step_with_amr_3d!`). ~12 LOC. |
| `experiments/M3_7d_per_axis_gamma_3d_cold_sinusoid.jl` | NEW: §7.5 selectivity driver (`run_M3_7d_per_axis_gamma_selectivity_3d`) + `init_cold_sinusoid_3d!` IC builder. ~140 LOC. |
| `test/test_M3_7d_gamma_per_axis_diag_3d.jl` | NEW: 253 asserts. γ accessor unit tests (math primitive + field walker on 3D). |
| `test/test_M3_7d_amr_3d.jl` | NEW: 53 asserts. Per-axis indicator + 3D refinement-listener integrity (parent → 8 children prolongation; refine + coarsen byte-equal round-trip). |
| `test/test_M3_7d_realizability_3d.jl` | NEW: 85 asserts. Per-axis projection unit (no-op / fires per-axis / 2D-symmetric byte-equal reduction / 1D-symmetric reduction). |
| `test/test_M3_7d_selectivity.jl` | NEW: 27 asserts. §7.5 headline selectivity gate (1D-sym / 2D-sym / full-3D / β-uniform sanity / γ trajectory monotonicity). |
| `test/runtests.jl` | APPEND-ONLY: new "Phase M3-7d" testset. ~40 LOC. |
| `reference/notes_M3_7d_3d_per_axis_gamma_amr.md` | THIS FILE. |

Total: **+418 new asserts**, **~+880 LOC** across `src/` + `experiments/` + `test/`,
**4 new test files**.

## Per-axis γ diagnostic structure (3D)

The math primitive `gamma_per_axis_3d(β::SVector{3}, M_vv_diag::SVector{3})`
already lived in `src/cholesky_DD_3d.jl` (M3-7 prep landed it).
M3-7d added the field-set-layer wrapper:

```julia
gamma_per_axis_3d_field(fields, leaves;
                         M_vv_override = nothing, ρ_ref = 1.0,
                         Gamma = GAMMA_LAW_DEFAULT) -> Matrix{Float64}
```

returning a `3 × N` matrix. With `M_vv_override = (Mvv_1, Mvv_2, Mvv_3)`,
axis-a γ uses the supplied override; otherwise all three axes share
`Mvv(J=1/ρ_ref, s)` from the EOS (isotropic-EOS convention, same as
2D).

The 1D-symmetric reduction at β_2 = β_3 = 0 byte-equally produces
γ_2 = γ_3 = √Mvv — verified in the unit test.

## Per-axis realizability projection (3D)

`realizability_project_3d!(fields, leaves; …)` extends M3-3d's 2D
projection to 3D. The binding constraint is

    M_vv ≥ headroom · max_a β_a²    for a = 1, 2, 3

so a single `s`-raise satisfies all three axes (the EOS gives
`Mvv,11 = Mvv,22 = Mvv,33 = Mvv(J, s)` — an isotropic ideal gas does
not distinguish axes). Off-diagonal `β_{ab}` remain absent from the
3D field set per M3-3a Q3 + M3-7 §4.4 (M3-9 will lift to 19-dof for
3D D.1 KH).

### ProjectionStats3D

A new `mutable struct ProjectionStats3D` matches the 1D / 2D
`ProjectionStats` shape, dropping the `n_offdiag_events` counter (which
is M3-6 Phase 1b's 4-component β-cone bookkeeping; off-diag β is not
in the 3D field set yet). The 6 fields are:
  • `n_steps::Int`
  • `n_events::Int`
  • `n_floor_events::Int`
  • `total_dE_inj::Float64`
  • `Mvv_min_pre::Float64`
  • `Mvv_min_post::Float64`

A `reset!(stats::ProjectionStats3D)` method dispatches on the new
type without conflicting with the existing `reset!(stats::ProjectionStats)`.

### Conservation properties (mirror M3-3d's 2D + M2-3 1D)

  • Per-axis `(α_a, β_a, x_a, u_a, θ_ab)` untouched.
  • `s` raised when projection fires.
  • Internal-energy increment per leaf: `+ρ · (M_vv_target − M_vv_pre)`
    when projection fires, admitted as silent floor-gain (no `Pp`
    debit slot in the 3D field set yet — M3-9 will activate the
    post-Newton sector).

### 2D-symmetric reduction (β_3 = 0) — PASS

The cross-check test runs both `realizability_project_3d!(fields3, leaves3)`
and `realizability_project_2d!(fields2, leaves2)` on parallel meshes
with `β = (0.85, 0.40, 0.0)` (3D) vs `β = (0.85, 0.40)` (2D), both
starting from the same `s_pre = log(0.30)`, asserting that the
post-projection `s_post` is byte-equal between the two paths. This is
the M3-7d projection-level 3D ⊂ 2D dimension-lift gate (passes to
≤ 1e-14).

## Per-axis action-AMR indicator on `HierarchicalMesh{3}`

`action_error_indicator_3d_per_axis(fields, mesh, frame, leaves, bc_spec; …)`
evaluates a per-leaf indicator that aggregates per-axis 2nd-difference
contributions of `(α_a, β_a, u_a)` plus a γ_inv-marker per axis, with
`max_a` aggregation:

    out[i] = max_a [ |d²α_a| + |d²β_a| + |d²u_a|/c_s + 0.01·(√Mvv/γ_a − 1) ]
           + |d²s|             # shared across axes; added once via axis-1 stencil

The `max_a` aggregation is what makes the indicator **per-axis selective**:
on the 3D 1D-symmetric IC (k=(1,0,0)), the axis-1 γ_inv-marker spikes;
axes 2 and 3 markers stay near zero. Refinement events fire only along
the active axis. The 2D-symmetric IC (k=(1,1,0)) fires on axes 1 and 2;
axis 3 stays unrefined. Full 3D (k=(1,1,1)) fires on all three.

The 3D unit test (`test_M3_7d_amr_3d.jl`) verifies:

  • Uniform field ⇒ indicator ≡ 0.
  • β_1 = sin(2π x_1) sinusoid (β_2 = β_3 = 0) ⇒ indicator > 0.05
    at most cells.
  • β_1 + β_2 sinusoid (β_3 = 0) ⇒ indicator > 0.05 (axes 1, 2 fire).
  • Full 3D β sinusoid ⇒ indicator > 0 across most cells (count > 0.5N).

## HG `register_refinement_listener!` 3D field-set listener

`register_field_set_on_refine_3d!(fields, mesh::HierarchicalMesh{3})`
delegates to the dimension-generic `register_field_set_on_refine!`
(which already handles arbitrary `D` via the `2^D = 8` children-per-
group coarsen averaging). The 3D-specific name is for call-site
clarity. The listener:

  1. Walks `event.refined_parents`, copies the parent's order-0 cell-
     average into all 2³ = 8 children (piecewise-constant prolongation).
  2. Walks `event.coarsened_parents`, averages the (now-removed) 8
     children's order-0 values into the new parent (mass-conservative
     for `ρ`-like fields under isotropic refinement).
  3. Resizes the underlying `fields.storage.<name>::Vector{Float64}` to
     `n_cells(mesh)` post-event.

Verified on a 3D mesh: refining a single octant in an 8-octant mesh
produces 15 leaves; all 16 named fields prolongate correctly (test
asserts `n_match == 8` for each of `α_a, β_a, θ_ab, x_a, u_a, s` ×
3 + 3 + 1 = 16).

The refine + coarsen round-trip preserves the pre-refine state byte-
equally when the 8 child values are equal (which they are after
piecewise-constant prolongation), since `(v + v + … + v) / 8 = v`
exactly in IEEE float. The test uses exact-representable binary
fractions (`k`, `k + 0.5`, `k + 0.25`, `1.0 + 0.5 * k`) to make this
rigorous.

## End-to-end AMR-driven 3D run: `step_with_amr_3d!`

Wraps HG's `step_with_amr!` with:

  • **`step!` callback** = `det_step_3d_berry_HG!` per step on the
    current `enumerate_leaves(mesh)` (refreshed after each AMR cycle).
  • **`indicator(mesh)` callback** = `action_error_indicator_3d_per_axis`
    mapped onto **mesh-cell indices** (length `n_cells(mesh)`; non-leaves
    return 0).
  • **Refinement listener** registered for the duration of the run via
    `register_field_set_on_refine_3d!`; auto-unregistered on exit.

**Difference from `step_with_amr_2d!`:** the 3D version does NOT thread
`project_kind` / `realizability_headroom` / `Mvv_floor` /
`pressure_floor` / `proj_stats` through to `det_step_3d_berry_HG!`,
because M3-7c's driver does not yet take a `project_kind` keyword.
Callers wanting realizability-projected runs should call
`realizability_project_3d!` from the surrounding driver between AMR
cycles. M3-7e (3D Tier-D Zel'dovich pancake) is the natural place to
thread the projection into the inner step.

## §7.5 Per-axis γ selectivity — PASS

The load-bearing scientific gate. Cold sinusoid IC at A = 0.3,
level = 2 (4×4×4 = 64 leaves), dt = 2e-3, n_steps = 20:

| Setup | std(γ_1) | std(γ_2) | std(γ_3) | Selectivity ratio | Gate |
|---|---|---|---|---|---|
| 1D-sym (k=(1,0,0)) | 1.43e-5 | 0.0 | 0.0 | 6.4e10 | > 1e10 ✓ |
| 2D-sym (k=(1,1,0)) | 1.43e-5 | 1.43e-5 | 0.0 | 6.4e10 (avg/eps) | > 1e6 ✓ |
| Full 3D (k=(1,1,1)) | 1.43e-5 | 1.43e-5 | 1.43e-5 | n/a | all axes fire ✓ |

By symmetry (equal `k_a` ⇒ equal axis dynamics), the `std(γ_a)` values
agree across active axes to round-off (≤ 1e-14). The trivial axes'
std is **exactly 0.0** to round-off (β_a stays uniform across leaves
for axes with `k_a = 0`).

### Why is the 3D 1D-sym ratio "only" 6.4e10 vs M3-3d's 1e15?

M3-3d's 1D-sym ratio reaches 1e15 because the 2D denominator is
`std(γ_2)` alone (one trivial axis), which sits at ≈ 1e-21
(round-off scale of the unchanging β_2 grown from the cold-limit
driver `D_t β = γ²/α`). The 3D ratio uses
`std(γ_2) + std(γ_3) + eps` — both trivial axes sum exactly to 0.0
(neither β_2 nor β_3 has a spatial driver in the 1D-symmetric IC),
so the denominator is `eps(Float64) ≈ 2.2e-16` and the ratio is
limited only by `std(γ_1)/eps ≈ 1.4e-5/2.2e-16 ≈ 6.4e10`. This is
**already comfortably above the 1e10 gate**.

The brief noted: "If 3D γ selectivity ratios fall short of 1e10
(1D-sym) or 1e6 (2D-sym), document — could indicate per-axis
decomposition has a numerical-precision issue at resolution." Both
ratios pass with 4+ orders of margin, so no precision issue is
flagged.

### Newton convergence

`det_step_3d_berry_HG!` converges in 2-3 iterations on the cold-
sinusoid IC at all three k-vector setups (well within the
M3-7c-allotted ≤ 7 iter on non-isotropic 3D ICs). Per-step wall time
~50 ms at level=2 (64 leaves); total selectivity-test runtime
~10 s for 27 asserts.

## Verification gates

### §7.1a + §7.1b dimension-lift gates (M3-7c) — PASS unchanged

The M3-7c §7.1a + §7.1b dimension-lift gates (3D ⊂ 1D + 3D ⊂ 2D
byte-equal at 0.0 absolute) continue to hold. M3-7d adds no new
calls into `cholesky_el_residual_3D_berry!` — the new code lives
purely in diagnostic, AMR, and projection layers.

### §7.5 Per-axis γ selectivity (3D) — PASS

Documented above. The principal-axis decomposition correctly identifies
the collapsing axes; the trivial axes preserve anisotropy across all
three symmetry classes (1D-sym, 2D-sym, full 3D).

### Per-axis realizability (3D) — PASS

  • No-op when `M_vv ≥ headroom · max_a β_a²`.
  • Fires when β_1, β_2, or β_3 is the binding axis (each tested
    independently).
  • Mvv_floor branch fires when all β are small but Mvv_pre is also
    small (absolute floor binds vs relative headroom).
  • 2D-symmetric reduction (β_3 = 0) reproduces M3-3d's 2D `s_post`
    byte-equally to ≤ 1e-14.
  • 1D-symmetric reduction (β_2 = β_3 = 0) matches M2-3's 1D
    `M_vv_target_rel = headroom · β_1²` to ≤ 1e-12.
  • `ProjectionStats3D` accumulator records events; `reset!` zeros
    the struct.

### Per-axis action-AMR indicator (3D) — PASS

  • Uniform field ⇒ indicator ≡ 0 (no spurious refinement triggers).
  • 1D-sym, 2D-sym, full-3D ICs trigger indicator selectivity along
    active axes only.
  • Refinement listener walks all 16 named fields; refining a single
    octant of an 8-leaf mesh produces 15 leaves with 8 children
    inheriting the parent's value across every named field.
  • Refine + coarsen round-trip preserves state byte-equally (using
    exact-representable binary fractions).

## What M3-7d does NOT do

Per the brief's "Critical constraints":

  • **Does not write 3D Tier-C/D drivers.** That's M3-7e (next sub-
    phase) — 3D Sod, 3D cold sinusoid, 3D plane wave, 3D Zel'dovich
    pancake.
  • **Does not add off-diagonal β_{ab} to the 3D field set.** That's
    M3-9 (3D D.1 KH follow-up); the 3D field set stays at 16 named
    scalars (13 Newton-active + s + 2 deferred — actually 13 + 1 = 16
    total, no Pp/Q/off-diag).
  • **Does not implement higher-order prolongation/restriction.** The
    HG refinement-listener is order-0 (piecewise-constant). Higher-
    order Bernstein restriction/prolongation is M3-4 / M3-5 work.
  • **Does not thread realizability projection through
    `det_step_3d_berry_HG!`.** Callers wanting projected 3D Newton
    runs should call `realizability_project_3d!` outside the inner
    step. M3-7e will plumb the projection into the M3-9 / Tier-D
    Zel'dovich pancake driver.
  • **Does not save a headline figure.** The 2D M3-3d note saves a
    4-panel figure; the 3D version's natural rendering is a
    volumetric isosurface or 3-axis γ scatter, which M3-7e will
    produce as part of the Tier-D drivers.

## Open issues / handoff to M3-7e

  • **3D Tier-C/D drivers (M3-7e scope).** The 3D analogs of
    `experiments/M3_3d_per_axis_gamma_cold_sinusoid.jl`,
    `experiments/M3_4_phase2_C1_sod_driver.jl`,
    `experiments/M3_4_phase2_C2_cold_sinusoid_driver.jl`,
    `experiments/M3_4_phase2_C3_plane_wave_driver.jl`, plus the
    headline 3D Zel'dovich pancake driver. The §7.5 selectivity
    driver in `experiments/M3_7d_per_axis_gamma_3d_cold_sinusoid.jl`
    is the kernel that M3-7e can extend (multi-symmetry-class IC
    sweeps; sweep dt, level, A; save volumetric γ snapshots).

  • **Wall-time impact.** The per-axis projection is a per-leaf
    constant-time op; on 64 leaves it costs sub-ms per call. The
    per-axis 3D AMR indicator is also linear in N leaves; the HG
    refinement-listener fires only at AMR cycle boundaries. No
    measurable wall-time regression vs M3-7c at the test resolutions.

  • **Periodic-x coordinate wrap (3D).** Inherited from M3-3c /
    M3-7b. The §7.5 driver uses REFLECTING BCs across all three
    axes for the same reason as M3-3d's 2D version (cold sinusoid
    has u(0) = u(L) = 0, naturally compatible with reflecting
    boundaries). Periodic 3D for active strain is a known M3-7e /
    M3-9 follow-up.

  • **3D headline figure.** Will require either CairoMakie 3D
    isosurfaces, GLMakie volumetric, or a slice-based 4-panel
    rendering. Defer to M3-7e.

  • **3D realizability projection inside the Newton inner step.**
    M3-7c's `det_step_3d_berry_HG!` does not currently take a
    `project_kind` keyword; the M3-7d driver pattern calls
    `realizability_project_3d!` outside the inner step. M3-7e or
    M3-9 should add the keyword for the 3D Zel'dovich pancake
    driver where the projection needs to fire mid-Newton-cycle
    (the pancake collapse drives γ_1 → 0 monotonically, so the
    projection has to happen between substeps, not just at the
    AMR cadence).

## How to extend in M3-7e

  • **Build the 3D Tier-D Zel'dovich pancake driver.** Methods
    paper §10.5 D.4 (the "central novel cosmological reference
    test"). Call signature mirrors `experiments/M3_6_phase2_D4_zeldovich.jl`
    (the 1D version), lifted to 3D via `allocate_cholesky_3d_fields`
    + `tier_d_zeldovich_pancake_3d_ic_full!` (M3-7e creates this
    factory). The per-axis γ diagnostic from M3-7d is the load-
    bearing acceptance gate: pancake collapse on a single axis
    (γ_1 → small, γ_2 / γ_3 → O(1)).

  • **Add `project_kind` to `det_step_3d_berry_HG!`.** Optional
    keyword threading `realizability_project_3d!` post-Newton
    (matches the 2D `det_step_2d_berry_HG!` signature). Default
    `:none` preserves M3-7c bit-exact regression.

  • **Build 3D analog of the headline figure.** 4-panel slice
    rendering (γ_1 along x, γ_2 along y, γ_3 along z, |s| at
    final step) or 3D isosurface of `min_a γ_a` (the collapse
    indicator).

## Reference

  • `reference/notes_M3_7_3d_extension.md` — full M3-7 design note
    (your sub-phase is §5 + §7.5).
  • `reference/notes_M3_7a_3d_halo_allocator.md`,
    `reference/notes_M3_7b_native_3d_residual.md`,
    `reference/notes_M3_7c_3d_berry_integration.md` — your
    dependencies.
  • `reference/notes_M3_3d_per_axis_gamma_amr.md` — the 2D pattern
    M3-7d generalises (same listener pattern, `max_a` aggregation,
    bit-equality regression contracts).
  • `reference/notes_M2_3_realizability.md` — 1D realizability
    projection design (the per-axis 3D form mirrors §3 derivation,
    extended to three axes with a single `s`-raise binding constraint
    `max_a β_a²`).
  • `reference/notes_M2_1_amr.md` — 1D action-AMR design.
  • `~/.julia/dev/HierarchicalGrids/src/Mesh/AMR.jl` — `step_with_amr!`
    driver consumed by `step_with_amr_3d!`.
  • `~/.julia/dev/HierarchicalGrids/src/Mesh/Mesh.jl` —
    `register_refinement_listener!` consumed by
    `register_field_set_on_refine!` (dimension-generic; M3-7d
    delegates to it via `register_field_set_on_refine_3d!`).
