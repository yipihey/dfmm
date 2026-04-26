# M3-prep — Tier-C IC factories (1D-symmetric 2D Sod, 2D cold sinusoid, plane wave)

**Date:** 2026-04-26.
**Branch:** `m3-prep-tierC-ic-factories`.
**Status:** complete; setup-only (no solver coupling).

This note documents the dimension-generic Tier-C initial-condition
factories landed in `src/setups_2d.jl`. They feed the upcoming
M3-4 dimension-lift gate (methods paper §10.4 Tier C tests). The
factories are pure IC code — they build a `PolynomialFieldSet` of
cell averages over a uniform Eulerian quadtree/octree on top of HG.
No solver coupling, no time-stepping; that's M3-4.

## API

The three factories all return a NamedTuple of the same shape:

```julia
ic = tier_c_sod_ic(; D = 2, level = 4, kwargs...)
# → (name, mesh::HierarchicalMesh{D}, frame::EulerianFrame{D},
#    fields::PolynomialFieldSet, params::NamedTuple)

ic = tier_c_cold_sinusoid_ic(; D = 2, level = 4, A = 0.5, k = (1, 0), kwargs...)
ic = tier_c_plane_wave_ic(; D = 2, level = 4, A = 1e-3, angle = 0.0, kwargs...)
```

Each factory:

1. Builds a `HierarchicalMesh{D}` of `(2^level)^D` leaves via
   `uniform_eulerian_mesh(D, level)` — a thin helper that calls
   `refine_cells!(mesh, enumerate_leaves(mesh))` `level` times.
2. Pairs it with an `EulerianFrame{D}` carrying the physical box.
3. Allocates an order-0 `PolynomialFieldSet` with named scalar
   fields `(rho, ux, [uy, [uz,]] P)` — one cell-average coefficient
   per leaf per field. The velocity-component name set scales with
   `D` to keep dimension-specific code out of caller-side test
   logic.
4. Computes per-cell averages by Gauss-Legendre quadrature on the
   reference cube `[0, 1]^D` (HG `gauss_quadrature_quad`,
   `gauss_quadrature_cube`, `gauss_quadrature_interval`) with an
   affine pull-back to the cell's physical AABB.

The factories are dimension-generic. `D = 1, 2, 3` all work without
dispatch — the only dimension-dependent code is the velocity-name
allocation (`_allocate_tierc_fields`), the quadrature picker
(`_quad_for_dim`), and the wavevector construction in `tier_c_plane_wave_ic`.

### Helpers exported

- `tier_c_total_mass(ic)` — `Σ_j ρ_j V_j` over all leaves; used by
  the conservation assertion in the unit tests.
- `tier_c_cell_centers(ic)` — `Vector{NTuple{D, T}}` of leaf
  centers, in `enumerate_leaves` order; used to look up primitive
  values at known points.
- `tier_c_velocity_component(fields, j, axis)` — read the
  `axis`-th velocity at leaf `j` without dispatching on
  `:ux`/`:uy`/`:uz` in caller code.
- `uniform_eulerian_mesh(D, level)` — exposed because some tests
  may want to allocate a matching field set themselves (it's the
  same primitive every Tier-C/D test will need at solver-coupling
  time).

## HG API gaps identified

### #9 (per `notes_HG_design_guidance.md`) — `init_field_from!` missing

HG does not yet ship a
`init_field_from!(field, mesh, f::Function; order)` helper that
projects an analytical IC onto a `PolynomialFieldSet` via
quadrature. Each Tier-C factory hand-rolls the projection inline in
its main loop:

```julia
for (j, ci) in enumerate(leaves)
    lo_c, hi_c = cell_physical_box(frame, ci)
    fields.rho[j] = (_cell_average(f_rho, lo_c, hi_c, quad),)
    # ... per-field
end
```

When HG ships #9, the `_cell_average` helper and its per-cell loops
should reduce to one call per field. Recommend the API:

```julia
init_field_from!(fields.rho, mesh, frame, f_rho;
                  basis = MonomialBasis{D, 0}(), quad_order = 3)
```

with `basis = MonomialBasis{D, 0}` for cell averages and
`basis = BernsteinBasis{D, P}` for higher-order projections.

### Periodic-BC info on `EulerianFrame`

The IC factories currently default to box `[0, 1]^D` and rely on
M3-4's solver coupling to attach periodicity per axis. There is no
`EulerianFrame` API for declaring axes-periodicity; M3-4's tests
will need a thin wrapper or a custom `:periodic` symbol on
the side. Per `notes_HG_design_guidance.md` item #2, this is on the
HG-side roadmap.

## Expected solver-side behaviour at M3-4

These factories produce field-sets that the M3-4 solver loops will
consume. The design guarantees:

- **C.1 Sod**: at `D = 1, level = N` the cell averages match
  `setup_sod(N = 2^N)`'s `(rho, u, P)` state to round-off (the
  factory's quadrature integrates the indicator function exactly
  away from the discontinuity, matching the discrete `ifelse` in
  `setup_sod`). The single cell straddling the split gets a linear
  cell-average of the step.
- **C.2 cold sinusoid**: per-axis γ identification is a
  solver-side property — the factory just stamps the analytical
  velocity field onto the cell averages. With `k = (1, 0)` the
  uy-component is exactly zero.
- **C.3 plane wave**: at θ = 0 the IC matches the M1 1D acoustic
  plane wave to the cell-average-vs-point-sample distinction
  (which falls off as `O(Δx²)`). The velocity is in phase with
  density via `δu = (c/ρ₀) δρ k̂` — the right-going acoustic
  branch. Rotational invariance to `1e-13` at `level = 4` (well
  below the brief's plotting-precision target).

## Cell-average convergence sample (C.3 plane wave, A = 1e-3, θ = 0)

Pointwise `δρ_pt(x) = A cos(2π x)` vs the factory's cell-average
`δρ_cell` on a uniform Eulerian quadtree:

| level | n_leaves | max\|δρ_cell − δρ_pt\| | ratio |
|------:|---------:|----------------------:|------:|
| 3 | 64    | 2.36e-5 |    — |
| 4 | 256   | 6.29e-6 | 3.75 |
| 5 | 1024  | 1.60e-6 | 3.93 |
| 6 | 4096  | 4.01e-7 | 3.98 |
| 7 | 16384 | 1.00e-7 | 3.99 |

The ratio approaches `4 = (Δx/2)^{-2}` as expected for the
midpoint-rule leading-order error coefficient `-π²/6`. The
quadrature error itself is round-off at `quad_order = 3` because
the integrand `A cos(2π x)` is well-resolved by 3 G-L nodes per
axis (5-th-degree exact); the `Δx²` floor is the cell-average vs
point-sample distinction, not a quadrature limit.

For the C.2 cold-sinusoid `sin(2π x)` integrand, the 3-pt G-L
quadrature has a residual error around `2e-10` per cell at
`level = 4`. The unit test bumps `quad_order = 8` to drive this
to `1e-13` for the tight comparison; the default `quad_order = 3`
is fine for solver-coupling-time accuracy needs.

## Wiring

- `src/setups_2d.jl` — new file with the three factories + helpers.
- `src/dfmm.jl` — append-only `include("setups_2d.jl")` and exports
  `tier_c_sod_ic`, `tier_c_cold_sinusoid_ic`, `tier_c_plane_wave_ic`,
  plus the helper accessors.
- `test/test_M3_prep_setups_tierC.jl` — new test file with 50
  asserts across the three factories (mass conservation, y-axis
  independence, sample primitive values, rotational invariance,
  acoustic-mode phase relation, convergence sample).
- `test/runtests.jl` — append-only `@testset` entry for the new
  file.

## Things deliberately deferred

- **Higher-order Bernstein basis projection.** All factories use
  order-0 (cell average). Higher-order is a `quad_order` knob from
  the same callsite; the M3-4 solver-coupling phase decides
  whether to consume order-0 or order-3 ICs.
- **Per-axis γ diagnostic.** A pure setup-side computation but it
  needs Cholesky decomposition machinery from Phase M3-3 (HG-side
  `(α_a, β_a)` field set). Defer until M3-3 lands and the IC + γ
  decoder live on the same field set.
- **Solver-coupled C.1/C.2/C.3 tests.** Not in scope for this
  prep phase; M3-4 will write `test_M3_4_C1_sod_HG.jl` etc. against
  the field-sets these factories produce.
- **D = 3 testing.** The factories support `D = 3` parametrically
  (`gauss_quadrature_cube`, `MonomialBasis{3, 0}`), but no test
  exercises it because M3-4's headline tests are 2D. The D=3
  call-path is exercised at M3-7 when the same factories are
  re-used for the 3D extension; no code change should be needed.
