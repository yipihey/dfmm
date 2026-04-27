# M3-3a — 2D field set + per-axis Cholesky decomposition driver

> **Status (2026-04-26):** *Implemented + tested*. First sub-phase of
> M3-3 (`reference/notes_M3_3_2d_cholesky_berry.md` §9). Two commits
> on branch `m3-3a-field-set-cholesky`:
>
>   1. `8f0350b` — HaloView coefficient-access smoke test (`test/test_M3_3a_halo_smoke.jl`).
>   2. *(this commit)* — 2D field set + per-axis Cholesky driver.
>
> Test delta vs M3-2 baseline: **+517 asserts** (3037 + 1 deferred → 3554 + 1 deferred).
> 1D-path bit-exact parity holds (no test failures, no test count
> regressions).

## What landed

| File | Change |
|---|---|
| `src/cholesky_DD.jl` | NEW: per-axis Cholesky decomposition driver. Three primitives: `cholesky_decompose_2d`, `cholesky_recompose_2d`, `gamma_per_axis_2d`. All allocation-free with `StaticArrays` inputs. |
| `src/types.jl` | EXTENDED: `DetField2D{T}` working struct carrying the 10 Newton unknowns `(x_a, u_a, α_a, β_a, θ_R, s)` plus the 2 post-Newton sectors `Pp, Q`. Legacy `DetField{T}` (1D) and `DetFieldND{D, T}` (doc tag) untouched. |
| `src/setups_2d.jl` | EXTENDED: `allocate_cholesky_2d_fields(mesh)` builds a 12-named-field `PolynomialFieldSet` over `n_cells(mesh)`; `read_detfield_2d` / `write_detfield_2d!` round-trip a `DetField2D` against it bit-exactly. |
| `src/dfmm.jl` | APPEND-ONLY: include + export the new symbols. The 1D path imports are unchanged. |
| `test/test_M3_3a_halo_smoke.jl` | NEW (committed in `8f0350b`): HaloView contract on an 8×8 balanced 2D mesh. |
| `test/test_M3_3a_cholesky_DD.jl` | NEW: 199 asserts. Round-trip on 50 random `(α, θ_R)`; iso-pullback gauge; per-axis γ on anisotropic / iso `M_vv`; allocation-free hot-path tests. |
| `test/test_M3_3a_field_set_2d.jl` | NEW: 288 asserts. `DetField2D` structural contract; field-set allocation over a 4×4 balanced mesh; full read/write round-trip; mass-conservation analog (uniform write → leaf sum); single-leaf write isolation. |
| `test/runtests.jl` | APPEND-ONLY: new "Phase M3-3a" testset between Berry stencils and M3-2. |
| `reference/notes_M3_3a_field_set_cholesky.md` | THIS FILE. |

## Q-resolution against the design note's §10 open questions

All four §10 default decisions held; no human judgement needed.

| Q | Default | Resolution in M3-3a |
|---|---|---|
| Q1 — HaloView coefficient access for order-0 fields | "write the smoke test first" | **HaloView returns `PolynomialView`, NOT a `Tuple`.** Index `pv[1]` for the scalar order-0 cell-average coefficient. The HaloView docstring claims "Tuple of length n_coeffs(basis) for zero allocation" but the implementation returns `hv.field[Int(nb)]`, a `PolynomialView`. The smoke test (commit `8f0350b`) records this concretely so M3-3b's residual reads `pv[1]`. Indexing in the cached fast path is allocation-free up to ≤ 64 bytes (matches HG's own halo-test tolerance). |
| Q2 — `MonomialBasis` order for the 2D Cholesky-sector fields | "stay at order 0" | Order 0. The 12-named-field `PolynomialFieldSet` allocator uses `MonomialBasis{2, 0}()`. M3-4 / M3-5 will move to higher-order Bernstein per methods paper §9.2. |
| Q3 — Off-diagonal `β_{12}, β_{21}` pinning | "omit" | **Omitted**. `DetField2D` has no `β_{12}, β_{21}` fields; `allocate_cholesky_2d_fields` allocates 12 named scalars (10 Newton unknowns + 2 post-Newton). M3-6 will re-add a parallel `allocate_cholesky_2d_offdiag_fields` constructor when the D.1 KH falsifier activates. |
| Q4 — `mesh.balanced == true` requirement | "balanced only" | **Asserted in the smoke test and the field-set tests**: every 2D HG mesh constructed in M3-3a uses `HierarchicalMesh{2}(; balanced = true)`. M3-3b's residual will inherit this constraint. Hanging-node (coarse-fine face) handling is deferred to a follow-up phase (M3-5 the natural place). |

## Smoke-test results (HaloView contract)

```
HaloView 2D order-0 smoke (M3-3a) | 30 30 0.3s
```

Six sub-blocks:

  1. Mesh sanity: 8×8 balanced 2D HG mesh has 64 leaves, 85 cells.
  2. Basis sanity: `n_coeffs(MonomialBasis{2, 0}()) == 1`.
  3. Interior-leaf self-access: `hv[i, (0, 0)][1]` returns the per-cell tag.
  4. Interior-leaf neighbor access: `hv[i, off]` agrees with `face_neighbors(mesh, i)` cross-check on all 4 face offsets.
  5. Boundary-leaf out-of-domain: corner leaf returns `nothing` for `(-1, 0)` and `(0, -1)`; non-boundary faces return `PolynomialView`.
  6. BC-aware wrap: `face_neighbors_with_bcs` with `((PERIODIC, PERIODIC), (REFLECTING, REFLECTING))` wraps axis 1 boundary faces and leaves axis 2 boundary faces at `0` (the dfmm EL residual will synthesize the reflecting ghost downstream).
  7. Allocation contract: cached `hv[i, off]` is ≤ 64 bytes per call.

## Per-axis Cholesky decomposition driver — math + sign convention

For a 2×2 symmetric positive-semi-definite moment matrix
`M = L · Lᵀ`, eigendecomposition gives

    M = R(θ_R) · diag(λ_1, λ_2) · R(θ_R)ᵀ,    α_a = √λ_a

with the standard 2×2 closed form

    a, b, c   = M[1,1], M[1,2]=M[2,1], M[2,2]
    mean      = (a + c) / 2
    diff      = (a - c) / 2
    D         = √(diff² + b²)
    λ_1       = mean + D     (largest)
    λ_2       = mean - D     (smallest, ≥ 0)
    θ_R       = atan(2b, a − c) / 2 ∈ (-π/2, π/2].

The canonical recomposition is

    L_canonical(α, θ_R) = R(θ_R) · diag(α)

which makes `cholesky_decompose_2d ∘ cholesky_recompose_2d ≡ id` to
≤ 1e-13 absolute (verified on 50 random inputs in
`test/test_M3_3a_cholesky_DD.jl`).

**Sort convention.** We sort `α_1 ≥ α_2 ≥ 0`. This is the "principal
axis = larger ellipsoid axis" convention; it matches the `θ_R = atan(2b,
a−c)/2` formula's sign so that `θ_R = 0` corresponds to a
diagonal `L` with `L[1,1] ≥ L[2,2]`.

**Gauge on iso slice.** When `α_1 = α_2`, the rotation angle is
under-determined (any rotation diagonalizes the iso `M = α² I`). We
return `θ_R = 0` deterministically (`atan(0, 0) = 0` in Julia). This
choice is consistent with the iso-pullback ε-expansion check (§6.3 of
the design note): on the iso slice the Berry kinetic 1-form
`Θ_rot^(2D) = (1/3)(α_1³ β_2 − α_2³ β_1) dθ_R` vanishes, so the
gauge choice does not affect the EL residual.

**Per-axis γ diagnostic.** `gamma_per_axis_2d(α, M_vv)` returns
`γ_a = √max(M_vv[a,a] / α_a², 0)`. For the M1 1D form
`γ² = M_vv − β²` generalized to 2D, callers should pre-subtract
`β · βᵀ` before passing `M_vv` here; the helper enforces only the
realizability floor `γ² ≥ 0`.

## What M3-3a does NOT do

Per the brief's "Critical constraints":

  • **Does not write the EL residual.** That is M3-3b. `det_el_residual_HG_2D!` is left for the M3-3b agent to draft.
  • **Does not touch `src/newton_step_HG.jl`.** M3-2b's Swaps 6+8 are in flight there; the 2D Newton solve is M3-3c+ work.
  • **Does not retire `cache_mesh`.** That is M3-3e (after dimension-lift parity gates pass).
  • **Does not initialise the 2D Cholesky state in the Tier-C IC factories.** The `tier_c_*_ic` factories still return only the 5-named-field `(rho, ux, uy, P)` `PolynomialFieldSet`; M3-3b will plumb a separate `(α, β, θ_R, s, ...)`-aware allocator into the IC pipeline once it has the EL residual to drive.

## Open issues / handoff to M3-3b

  • **HaloView returns `PolynomialView`, not `Tuple`.** M3-3b residuals must read `pv[1]` for the scalar order-0 cell-average. The smoke test pins this; the HG-side docstring claim of "Tuple of length n_coeffs(basis)" should eventually be corrected upstream, but for M3-3 we work around it.
  • **`balanced = true` is the working assumption.** All `HierarchicalMesh{2}` constructions in M3-3a explicitly pass `balanced = true`. M3-3b's residual + Newton solve should mirror this; hanging-node support is M3-5.
  • **The 12-named-field allocator includes `Pp, Q`.** These are post-Newton sectors but we provision storage for them now so the M3-3b residual + the M2-3 realizability projection don't have to thread two field-sets. M3-6's D.1 KH activation will add `β_{12}, β_{21}` via a parallel allocator.
  • **`DetField2D` does not currently subtype `DetFieldND`.** `DetFieldND` is a doc-tag; `DetField2D` is the working struct. They coexist; if M3-7 wants a 3D variant we will likely promote `DetField3D` similarly.

## How to extend in M3-3b/c/d/e

  • **M3-3b** (native HG-side EL residual without Berry): use
    `read_detfield_2d` to materialize the 10-dof state per leaf;
    call `cholesky_recompose_2d` only when assembling the per-axis
    pressure stencil; consume `halo_view`'s `PolynomialView` returns
    via `pv[1]` for the scalar coefficient.
  • **M3-3c** (Berry coupling): consume `src/berry.jl::berry_partials_2d`
    in the residual; θ_R becomes a Newton unknown driven by the
    Berry term.
  • **M3-3d** (per-axis γ AMR): `gamma_per_axis_2d` is the inputs to
    the per-axis action-error indicator.
  • **M3-3e** (cache_mesh retirement): the `DetField2D` storage layout
    here matches the M3-3b native residual so cache_mesh is no longer
    needed on the 2D path. Drop the 1D `cache_mesh` shim once
    dimension-lift parity passes.

## Reference

  • `reference/notes_M3_3_2d_cholesky_berry.md` — full M3-3 design note (your sub-phase is §9 entry "M3-3a").
  • `reference/notes_M3_phase0_berry_connection.md` — 2D Berry derivation (§3 kinematic equation, §6 verifications, §6.6 H_rot constraint).
  • `reference/notes_M3_prep_berry_stencil.md` — Berry stencil API (`src/berry.jl`).
  • `~/.julia/dev/HierarchicalGrids/src/Storage/HaloView.jl` — HG halo API (the `PolynomialView` return contract recorded here).
  • `~/.julia/dev/HierarchicalGrids/test/test_halo_view.jl` — HG halo usage examples that the smoke test mirrors.
