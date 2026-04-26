# HierarchicalGrids.jl — design guidance for dfmm-2d's purposes

**Date:** 2026-04-26.
**Status:** Initial design-review note. Items 1, 4, 5, 7, 8 are useful
to any downstream package; items 2, 3, 6 are dfmm-specific but
generally needed by hyperbolic-PDE solvers.

This note documents what dfmm-2d uses from
[`HierarchicalGrids.jl`](https://github.com/yipihey/HierarchicalGrids.jl)
(v0.1, published 2026-04-26) and what additions / refinements would
make HG a complete substrate for the dfmm methods paper's 1D / 2D / 3D
implementation.

The companion document `reference/MILESTONE_3_PLAN.md` walks through
how dfmm consumes these features end-to-end.

## What dfmm uses directly (no changes needed)

| HG layer | dfmm use |
|---|---|
| `HierarchicalMesh{D}` | Eulerian quadtree (D=2) / octree (D=3) for Bayesian remap target + AMR |
| `SimplicialMesh{D, T}` | Lagrangian segments (D=1) / triangles (D=2) / tetrahedra (D=3) — the per-parcel mesh dfmm evolves |
| `set_vertex_position!`, `update_lagrangian_positions!` | Per-step vertex evolution from the Newton solve |
| `deformation_gradient`, `volume_jacobian`, `distortion_metric` | The Jacobian $J$ that feeds the EOS $M_{vv}(J, s)$; mesh-quality diagnostic |
| `PolynomialFieldSet` (Bernstein basis) | Per-cell polynomial reconstruction of (x, u, α_a, β_a, θ_R, s, Q, Π, tracers); cubic ($p=3$) reconstruction natural |
| `compute_overlap` + `polynomial_remap` | Bayesian L→E remap (methods paper §6) and the inverse E→L; conservative to round-off |
| `compute_overlap`'s parallel mode | Per-step remap parallelizes for free over (Lag, Eul) leaf pairs |
| `refine_by_indicator!` | Action-based AMR (methods paper §9.7); plug in $\Delta S_{\rm cell}$ as the user function |
| `CompositeMesh` / `PairedMesh` | Paired Lag+Eul with overlap cache; invalidates on refinement (matches methods paper §9.7 algorithm) |
| `enumerate_leaves`, `n_simplices` | Per-step iteration over the active mesh |
| `parallel_for_cells` | The Newton residual evaluation is per-cell-parallel; replaces dfmm's manual `@threads` |
| `Quadrature` (Gauss-Legendre + simplex) | Discrete-action integrals (methods paper §9.4) at chosen polynomial order |
| `Memory` (pools / arenas) | Long-running production runs with pool allocation matching `det_step!`'s scratch needs |

## Recommended HG additions / refinements

These are guidance items to send back to the HG project; some may
already exist and just need confirmation, others are genuine gaps.

### 1. 1D parity confirmation

The README emphasizes 2D/3D. The methods paper needs
`HierarchicalMesh{1}`, `SimplicialMesh{1, T}` (a 1D simplex = a
segment = two vertices), and `compute_overlap` for 1D as trivial
interval intersection. Confirm tests cover D=1 across the stack. If
1D needs a thin shim (intervals as polytopes), contribute upstream —
keeps dfmm dimension-generic.

### 2. Periodic boundary conditions on `SimplicialMesh`

dfmm's wave-pool, cold sinusoid, and KH all require periodic
Lagrangian topology. The current `SimplicialMesh` constructor takes a
neighbor matrix; a `periodic!(mesh; axes)` helper that wires the
wrap-around neighbors would standardize. Same for `HierarchicalMesh`
Eulerian-frame periodicity (currently the user manages this through
the `EulerianFrame` bounding-box convention).

### 3. Inflow / outflow / Dirichlet primitives

Steady-shock Mach scan (Tier A.3) and oblique-shock interaction
(Tier D.3) need per-axis-half Dirichlet on the Eulerian frame, with
the Lagrangian mesh's boundary simplices pinned. A
`BoundaryCondition{:periodic, :inflow, :outflow, :reflecting}`
per-axis-half attachable to `EulerianFrame` would replace dfmm's M1
`Mesh1D{bc}` field cleanly.

### 4. Bernstein-basis positivity certificate

Methods paper §9.3 requires:

- $\det F > 0$ everywhere (no shell crossing) via Bernstein-coefficient
  positivity certificate.
- $L_{ii}(q) > 0$ via the $L_{ii} = \exp(\lambda_i)$ exp-parameterization.

The Bernstein basis is in HG. A primitive
`is_strictly_positive(field::PolynomialField; tolerance)` returning a
bool plus the offending coefficient would be a clean contribution.
Useful for any positivity-aware solver, not just dfmm.

### 5. Refinement-event callback

When `refine_cells!` or `coarsen_cells!` is called, downstream
packages that store per-cell auxiliary data need to be notified to
refine/coarsen their own arrays. `PolynomialFieldSet` already does
this implicitly for fields HG knows about; an exposed
`register_on_refine!(callback)` API for unregistered downstream
storage (e.g. dfmm's per-segment Newton-solver scratch) would prevent
the kind of inter-phase regression M2-1 had to navigate.

### 6. Halo / ghost-cell primitive

dfmm's pressure-gradient face fluxes need values from neighbor cells.
The mesh adjacency is in HG; a thin `halo_view(field, depth=1)` that
returns a per-cell neighbor-tuple view (zero allocation) would
standardize. Similar to how Trixi's `SVector{n_neighbors}` per-cell
stencil works.

### 7. Sparsity-pattern API for downstream Newton solves

dfmm's sparse-AD path (M1 Phase 5b) builds the Jacobian sparsity from
segment-adjacency. In 2D/3D this becomes triangle-adjacency or
tetrahedron-adjacency — already in HG. Exposing
`cell_adjacency_sparsity(mesh; depth=1) :: SparseMatrixCSC{Bool}` would
standardize what every implicit solver needs to pass to
`SparseConnectivityTracer`.

### 8. Liouville-monotone-increase diagnostic in `polynomial_remap`

Methods paper §6.5: the Bayesian remap has $\det L$ monotone
non-decreasing per parcel — physical entropy production. dfmm needs
to monitor this. A `liouville_increment` field returned by
`polynomial_remap` (or a tracking hook) would make the diagnostic
automatic. Same level of visibility as the existing
`total_overlap_volume` mass-conservation check.

### 9. Per-cell field initialization from a polynomial function

Common pattern: `f(x, y) = ...` analytical IC → Bernstein
coefficients per cell. A
`init_field_from!(field, mesh, f::Function; order)` that uses HG's
quadrature + Bernstein-basis projection internally would close the
M3-Phase-3 setup gap (Tier C 1D-symmetric Sod, cold sinusoid, plane
wave). Currently the user has to hand-roll this.

### 10. Stratified time-stepping coupled with AMR

Methods paper §9.7 says refine/coarsen should happen at every output
step; dfmm needs to coordinate Newton-iteration counts (which depend
on cell density) with the per-step refinement hysteresis. A
`step_with_amr!(mesh, step_fn, indicator_fn; hysteresis)` driver
pattern would standardize. Optional, easy to do in dfmm's driver
layer if HG doesn't take it on.

## Out of scope for HG (stays in dfmm)

- The discrete EL residual evaluation (variational integrator).
- Principal-axis Cholesky decomposition + Berry connection coupling.
- Variance-gamma stochastic injection per principal axis.
- Realizability projection (M2-3 result).
- Action-error indicator's specific formula $\Delta S_{\rm cell}$
  (methods paper §9.7) — the *driver* `refine_by_indicator!` is HG;
  the per-cell formula is dfmm.
- EOS coupling $M_{vv}(J, s)$ — dfmm Track C.
- Two-fluid cross-coupling kernels — dfmm.

## How to use this document

- **For Tom (or HG maintainer):** items 1, 4, 5, 7, 8 are upstream PR
  candidates; items 2, 3, 6 are arguably dfmm-internal but general
  enough that HG can host them as optional subpackages.
- **For the M3-0 agent:** items 1 and 4 are pre-launch verification
  steps. Items 2, 3, 6 may need workarounds in dfmm code if HG
  hasn't taken them yet. The M3-Phase-0 brief should call out the
  workarounds explicitly.
