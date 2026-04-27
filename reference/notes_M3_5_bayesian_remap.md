# M3-5 — Bayesian L↔E remap (close)

**Branch:** `m3-5-bayesian-remap`. Worktree:
`.claude/worktrees/agent-a8934fbbbcdcced43/`. **Date:** 2026-04-26.

**Status:** *closed*. Wires HG's `compute_overlap` +
`polynomial_remap_l_to_e!` / `polynomial_remap_e_to_l!` into a
dfmm-side per-step driver via `BayesianRemapState{D, T}` +
`bayesian_remap_l_to_e!` / `bayesian_remap_e_to_l!` /
`remap_round_trip!`. Liouville monotone-increase exposed via
`liouville_monotone_increase_diagnostic`. IntExact backend (HG commit
`cc6ed70`+) adopted as the conservation regression gate via
`audit_overlap_dfmm`. Methods paper §6.6 spec is satisfied to the
extent supported by current HG; the full moment-level diagnostic
awaits HG-design-guidance item #8 (the upstream `liouville_increment`
hook).

Test count: 13375 + 1 deferred → **13461 + 1 deferred** (+86 net,
all in the new "Phase M3-5: Bayesian L↔E remap" testset).

## API surface

`src/remap.jl` (NEW; 410 LOC):

| Symbol | Role |
|---|---|
| `BayesianRemapState{D, T}` | Per-step state: Eulerian mesh + frame + companion `PolynomialFieldSet` + opt-in `IntegerLattice` + reusable `RemapDiagnostics{T}` + Liouville running-max + per-call history. |
| `BayesianRemapState(eul_mesh, frame, lag_fields; lattice = nothing)` | Constructor: companion field set is allocated SoA with the same basis / named-field layout as `lag_fields`, sized to `n_cells(eul_mesh)`. |
| `bayesian_remap_l_to_e!(state, lag_mesh, lag_fields; backend = :float, moment_order = nothing, parallel = false, fields = nothing)` | L→E projection. Computes the overlap once, runs the polynomial remap on every named field. |
| `bayesian_remap_e_to_l!(state, lag_mesh, lag_fields; ...)` | E→L inverse projection. |
| `remap_round_trip!(state, lag_mesh, lag_fields; ...)` | L→E→L convenience wrapper for conservation regression. |
| `liouville_monotone_increase_diagnostic(state)` | Returns `(min_J, max_J, monotone_increase_holds::Bool)`. The bool is the §6.6 *necessary condition*: positive proxies, balanced volumes, zero `n_negative_jacobian_cells`. |
| `audit_overlap_dfmm(; verbose = false, atol = 1e-10)` | Wraps HG's `audit_overlap` for the dfmm IntExact regression gate. |
| `total_mass_weighted_lagrangian(lag_mesh, lag_fields, fieldname)` | `Σ_s a_s · V_s` in canonical 1..n_simplices order. |
| `total_mass_weighted_eulerian(state, fieldname)` | `Σ_j a_j · V_j` over Eulerian leaves only. |

`src/dfmm.jl` (append-only): includes `remap.jl`, exports the
M3-5 API, and adds a `det_run_with_remap_HG!` stub for the M3-6
hand-off (intentionally inert when `remap_every === nothing`,
preserving 1D bit-exact 0.0 parity).

## Conservation gate results

Run on a 16×16 (level = 4) Eulerian mesh + 512-triangle Lagrangian
mesh (2 triangles per leaf, vertices de-duped at leaf boundaries
→ conforming) with sinusoidal-displacement vertex deformation
(A = 0.001 to 0.005, bump-shaped, vanishing at boundary).

| Test scenario | Backend | Mass rel-err | Momentum atol | Energy rel-err |
|---|---|---:|---:|---:|
| Identity-overlap (no deformation) | `:float` | ≤ 1e-12 | 1e-12 | 1e-12 |
| Sinusoidal-displacement, single round trip | `:float` | ≤ 1e-12 | 1e-12 | 1e-12 |
| Multi-field round trip (rho, ux, uy, P, E) | `:float` | ≤ 1e-12 | 1e-12 | 1e-12 |
| 5-cycle round trip on fixed deformed mesh | `:float` | ≤ 6.7e-16 (per cycle) | — | — |
| Identity-overlap uniform field | `:exact` | **0.0** (byte-equal) | — | — |

The `:exact` backend on the *uniform-density* identity-overlap
configuration produces byte-equal totals because every vertex sits
at multiples of `1/N` and the lattice resolution exactly preserves
those positions. On a sinusoidally-deformed mesh the lattice's
16-bit auto-derivation can no longer hold all vertices on the
lattice; HG's `quantize_strict` is not invoked (we use the
permissive `quantize`), so the deformed-mesh exact path returns a
slightly different total (< 5 % drift on benign configurations).

Numerical checks (atol = 1e-10) of HG's canonical 9-polytope IntExact
audit battery: **9/9 pass**, max_volume_relative_diff ≈ 1.33e-16,
max_moment_relative_diff ≈ 1.33e-16 (essentially round-off).

## Liouville monotone-increase verification

The HG `RemapDiagnostics` proxy `entry.volume / source_physical_volume`
is a *geometric proxy* for the moment-level §6.5 theorem. The
necessary-condition check we expose:

```
monotone_increase_holds = (n_negative_jacobian_cells == 0) AND
                          (liouville_min > 0) AND
                          (total_volume_in == total_volume_out)
```

Verification status: **passes on every M3-5 test scenario**. Across
5 progressive deformation cycles on an 8×8 Eulerian + 128-triangle
Lagrangian mesh:

- `n_negative_jacobian_cells == 0` (no inverted Lagrangian simplex).
- `liouville_min > 0` (positive overlap polygon volumes).
- `total_volume_in == total_volume_out` to 1e-12.
- `total_overlap_volume(state.last_overlap) ≈ box_vol = 1.0` to 1e-12
  (partition-of-unity check, complementary to the diagnostic).

The full sufficient verification (`Δ_Liou(C_j) ≥ 0` per-cell) requires
the moment-level `det L_j^new` to be evaluated against
`Σ_i w_ij det L_i`; that is `hg-design-guidance` item #8 and is
deferred to a future HG release. M3-5 stitches the moment-level
conservation gate on top via the `total_mass_weighted_*`
accumulators, which exercises a related — but coarser — global
property.

## IntExact audit harness results

Default `atol = 1e-10`:

```
OverlapAuditReport(checked=9, passed=9, failed=0,
                   max_vol_rel_diff=1.33e-16, max_moment_rel_diff=1.33e-16)
```

The audit covers:

- **D = 2** (5 polytopes): unit triangle, off-axis triangle,
  triangle-clipped-by-quarter-box, triangle-partial-overlap,
  empty case.
- **D = 3** (4 polytopes): unit tetrahedron, Kuhn-cube tetrahedron,
  tet-partial-overlap, empty case.

All pass at `atol = 1e-12` already (the residual is essentially
round-off at the 10-bit audit lattice).

### IntExact caveats observed in dfmm's 2D scenarios

1. **D = 2 0//0 collinear-triangle degeneracy**. *Magnitude:*
   complete failure — `ArgumentError: invalid rational:
   zero(Int128)//zero(Int128)` from
   `R3D.IntExact._vertex_rational_d2`. *Trigger:* a sinusoidally-
   deformed Lagrangian triangulation with multiplicative bump
   factor `0.001 * cycle` at the 16×16 demo. *Mitigation in dfmm:*
   `experiments/M3_5_remap_round_trip_demo.jl` wraps the `:exact`
   call in `try/catch` and reports the caveat. Production code
   should use `:float` (the M3-5 default); `:exact` is the audit
   harness path.

2. **16-bit-lattice volume drift**. *Magnitude:* on the canonical
   audit battery, ≈ 0 (1.33e-16). On the dfmm 4×4 identity-overlap
   setup, total overlap volume drifts by ≤ 1e-3 from the box
   volume. On the deformed dfmm 8×8 setup with uniform field, drift
   is 0 (by-eye exact). *Mitigation:* if tighter audits are needed,
   use a finer lattice (`bits = 24` or `int_type = Int64`) — see
   `~/.julia/dev/HierarchicalGrids/docs/src/exact_backend.md`.

3. **No periodic-ghost support on `:exact`**. *Magnitude:* hard
   error if any periodic axis is attached. *Mitigation:* dfmm uses
   `:float` whenever periodic frame BCs are active; `:exact` is
   for non-periodic (or no-BC) audit comparisons.

## Per-cycle wall time

On a 16×16 Eulerian (256 leaves) + 512-triangle Lagrangian setup,
single-thread:

| Path | s/cycle (L→E→L) |
|---|---:|
| `:float` | ~0.17 s |
| `:exact` (single L→E) | ~0.5 s on benign configs; throws on collinear-triangle deformed configs (caveat #1 above) |

The `:exact` path is a few×slower than `:float` and not yet
parallel-safe (HG comment in `compute.jl` notes the BigInt
accumulator allocations don't compose with per-task builders);
for production runs `:float` is the default.

## HG API observations

- The `polynomial_remap_field!` field-set wrapper at the HG side
  exists, but for our hot loop we go directly through
  `polynomial_coeffs_view` + `polynomial_remap_l_to_e!` / `_e_to_l!`
  — this lets us share the per-cell pullback computation and the
  per-side frame vectors across multiple fields in one pass.
  HG's wrapper recomputes pullbacks per call (per its docstring),
  so dfmm's `_remap_pass!` shaves ~`O(n_cells)` `2D` matrix
  builds per per-call per-field by routing through the lower-level
  primitive.
- The `RemapDiagnostics` accumulator's geometric proxy is
  per-overlap-entry, not per-target-cell. The §6.6 spec calls for
  the per-cell `Δ_Liou(C_j)`; the entry-level proxy is the closest
  HG currently provides.
- HG rejects clockwise (signed-J < 0) source simplices by returning
  zero overlap, NOT as `n_negative_jacobian_cells > 0`. This
  behavior is documented in the M3-5 test layer
  (`test_M3_5_liouville_monotone.jl`'s "no-overlap detection"
  testset) so the failure-mode signaling is well-understood.

## Tests where bit-equality couldn't be held

None. Every M3-5 conservation-regression assertion holds at the
documented relative tolerance (≤ 1e-12 on `:float`, byte-equal on
the uniform-field `:exact` path). The 1D bit-exact 0.0 parity
contract from M3-3e is preserved: all 8624 native-vs-cache cross-
check tests still pass at 0.0 absolute.

## Files added / modified

| Commit | File(s) | Δ LOC |
|---|---|---:|
| ed4bd08 | src/remap.jl (NEW) | +414 |
| ed4bd08 | src/dfmm.jl | +59 |
| 60297d9 | src/remap.jl | -36 / +56 (Liouville-diagnostic refinement) |
| 084c140 | test/test_M3_5_remap_conservation.jl (NEW) | +236 |
| 084c140 | test/test_M3_5_liouville_monotone.jl (NEW) | +163 |
| 084c140 | test/test_M3_5_intexact_audit.jl (NEW) | +103 |
| 084c140 | test/runtests.jl | +28 |
| a42c95f | experiments/M3_5_remap_round_trip_demo.jl (NEW) | +204 |
| a42c95f | reference/figs/M3_5_remap_conservation.png (NEW) | (binary, 203 kB) |
| (this) | reference/notes_M3_5_bayesian_remap.md (NEW) | this file |
| (this) | reference/MILESTONE_3_STATUS.md | M3-5 row marked closed |

`src/eom.jl`, `src/newton_step_HG.jl`, `src/setups_2d.jl` are NOT
modified — those are M3-4 territory.

## M3-6 hand-off items

- Wire the new `det_run_with_remap_HG!` stub into the 2D solver's
  run-driver (M3-3's per-step `det_step_2d_berry_HG!` doesn't have
  a public run-driver yet; M3-6 / D.1 KH falsifier is a natural
  consumer).
- Adopt the upcoming HG `liouville_increment` hook (HG-design-
  guidance item #8) to upgrade `liouville_monotone_increase_diagnostic`
  to the full §6.6 sufficient verification.
- Add Bernstein-basis support: `polynomial_remap_l_to_e!` requires
  monomial-basis storage; the `Bases` module has multi-D
  Bernstein↔Monomial change-of-basis at D = 1 only. M3-6 / Phase 5
  reconstruction will need 2D / 3D bidirectional change-basis.
- IntExact production-path: the documented D=2 0//0 caveat is a
  hard blocker for production use of `:exact` on deformed meshes.
  M3-6 (or its own milestone) should produce a degeneracy-handling
  upstream HG fix.
- Tier-D KH instability benchmark with M3-5 remap + M3-3d AMR
  active. The off-diagonal β_{12}, β_{21} sector activation is
  M3-6 / D.1 territory; M3-5 lays the substrate.

## Pointers

- `specs/01_methods_paper.tex` §6 (and §6.6 in particular).
- `reference/notes_M3_prep_paper_section6_revision_applied.md`.
- `reference/notes_HG_design_guidance.md` items #4 (positivity), #5
  (refinement-event callback), #8 (Liouville monotone-increase hook).
- HG: `src/Overlap/compute.jl`, `src/Overlap/polynomial_remap.jl`,
  `src/Overlap/r3d_int_adapter.jl`, `src/Overlap/quantize.jl`,
  `src/Diagnostics/Diagnostics.jl`, `src/Diagnostics/exact_audit.jl`,
  `docs/src/exact_backend.md`.
- `reference/MILESTONE_3_STATUS.md` (M3-5 row updated).
