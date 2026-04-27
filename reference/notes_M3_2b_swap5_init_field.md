# M3-2b Swap 5 — Tier-C IC factories on HG `init_field_from!`

Date: 2026-04-26
Branch: `m3-2b-swap5-init-field`
Predecessor: `b8f4dc2 M3-2b pre-flight: bump HierarchicalGrids dev-mount to 7c56d47`

## What changed

`src/setups_2d.jl` previously hand-rolled per-cell tensor-product
Gauss-Legendre quadrature inside each Tier-C IC factory: a private
`_cell_average(f, lo, hi, quad)` helper, an explicit `_quad_for_dim`
dispatcher, and a `for ci in leaves` loop in each of the three factories
(`tier_c_sod_ic`, `tier_c_cold_sinusoid_ic`, `tier_c_plane_wave_ic`)
plumbing affine maps + weighted sums into the field set.

The new HG primitive `init_field_from!(field, frame, f; quadrature_order)`
(`HierarchicalGrids.Storage.Initialization`, commit `4b481da`)
performs an L² projection per cell on the reference cube and writes the
projected coefficients into a `PolynomialFieldSet`. For the order-0
`MonomialBasis{D, 0}` used by Tier-C, the local mass matrix collapses to
`M = [1]` and L² projection reduces to the cell average

    c[1] = ∫_ref f(x(ξ)) dξ ≈ Σ_k w_k * f(x_k),

so the new helper reproduces the legacy hand-rolled cell average exactly
when the same n-point GL tensor rule is used per axis.

The factories now:
1. Build the mesh + frame as before.
2. Allocate the public n_leaves-sized field set with the Tier-C names.
3. For each scalar function (ρ, P, u_x, u_y[, u_z]):
   - Allocate a temporary single-field n_cells-sized
     `PolynomialFieldSet` with the same order-0 monomial basis.
   - Call `init_field_from!(tmp, frame, f; quadrature_order = 2*quad_order - 1)`.
   - Copy the leaf entries (in `enumerate_leaves` order) into the
     master `fields.<name>`.

A small helper `_project_scalar_to_leaves!` encapsulates step 3 so the
three factories' bodies become a flat sequence of `_project_*!` calls
on `(rho, P, ux, uy[, uz])`.

## Why an n_cells temporary, not a direct n_leaves field

`init_field_from!` iterates `1:n_elements(field)` and queries
`cell_physical_box(frame, i)` with each i. The frame indexes *all*
cells of the parent `HierarchicalMesh{D}`, including non-leaves. An
n_leaves-sized field would receive the wrong cell boxes for any
indices beyond the root cell. The HG test at line 113-114 of
`test_init_field_from.jl` confirms this convention: "A non-leaf cell
... is also written but with garbage geometry; the helper writes to
every element of `field` (n_cells)."

The Tier-C public API exposes leaf-indexed storage (callers do
`ic.fields.rho[j]` where `j` runs over leaves), so we keep the public
field n_leaves-sized and project through an n_cells temporary. The
allocation cost is negligible — at level 4 in 2D, n_cells ≈ 341 vs
n_leaves = 256 — and it keeps the public API stable across the swap.

## `quad_order` semantics

dfmm's public `quad_order` kwarg has historically meant "n
Gauss-Legendre points per axis" (legacy `gauss_quadrature_quad(n)`).
HG's `init_field_from!` takes `quadrature_order` defined as
"polynomial degree the rule integrates exactly", with the n-point rule
selected via `n = ceil((order + 1) / 2)`. We translate via

    hg_quadrature_order = 2 * quad_order - 1

so passing `quad_order = 3` selects HG's n=3 (`order=5`), reproducing
the legacy 3-point rule exactly. The translation lives in
`_hg_quadrature_order(::Integer)`.

## Numerical parity

Bit-equal numerical agreement was achieved (no test thresholds had to
move). The 50 Tier-C asserts pass unchanged.

Plane-wave convergence levels 4/5/6 (test
`test_M3_prep_setups_tierC.jl` "C.3 plane wave — convergence δρ vs L₁
refinement"):

| Level | Old (legacy hand-roll) | New (init_field_from!) |
|-------|------------------------|------------------------|
| 4     | 6.3e-6                 | 6.289920216996604e-6   |
| 5     | 1.6e-6                 | 1.597875499962341e-6   |
| 6     | 4.0e-7                 | 4.0106316933274586e-7  |

Convergence ratios:
- Level 4 → 5: 3.94 (target > 3.5)
- Level 5 → 6: 3.98 (target > 3.5)

These match the baseline values quoted in the test (Δx² convergence,
ratios approaching 4 as the leading-order error coefficient becomes
exact).

## HG API gaps

1. **Single-target only.** `init_field_from!` writes the same
   coefficient vector to every named field of its target. This forces
   per-quantity temporary allocations (or projecting a tuple-valued
   function and splitting). For the order-0 Tier-C use case this is
   fine; for high-P fields with multiple correlated components a
   tuple-valued projection helper would save M-factor work. Not
   blocking for this swap.
2. **Iterates over n_cells, not n_leaves.** No knob to restrict
   projection to leaves. Forces the n_cells temporary pattern above.
   For uniform refinement (Tier-C) the overhead is `(2^D - 1) /
   (2^D - 1)`-ish (~30% extra cells projected); for asymmetric AMR
   trees this could waste real work.
3. **`quadrature_order` semantics differ from dfmm's `quad_order`.**
   Translation in `_hg_quadrature_order` works but is a paper cut.
   Could be hidden inside HG with a dfmm-style `n_quad_points_per_axis`
   kwarg, or just documented.

## Files touched

- `src/setups_2d.jl` — entire body refactored; public API unchanged.
- `reference/notes_M3_2b_swap5_init_field.md` — this design note.

The shared module file `src/dfmm.jl` and `test/runtests.jl` are
untouched.

## LOC delta

The LOC count is roughly flat (459 → 487 code-only lines, +28). The
in-factory complexity dropped substantially — each factory body is now
a flat sequence of `_project_scalar_to_leaves!` calls instead of a
nested per-cell affine-map + weighted-sum loop — but the documentation
grew to explain the L²-projection backing and the n_cells/n_leaves
asymmetry. Net win: clarity, single source of truth for the projection
math (HG), and zero numerical regression.
