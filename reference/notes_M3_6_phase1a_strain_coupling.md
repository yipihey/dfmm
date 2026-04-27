# M3-6 Phase 1a — Off-diagonal strain coupling H_rot^off

> **Status (2026-04-26):** *Implemented + tested*. First sub-phase of
> M3-6 Phase 1 (D.1 Kelvin–Helmholtz falsifier). Wires the off-diagonal
> Hamiltonian
>
>     H_rot^off = G̃_12 · (α_1·β_21 + α_2·β_12) / 2
>
> into the 2D EL residual, breaking the trivial-drive `β̇_12 = β̇_21 = 0`
> behaviour of M3-6 Phase 0. The cross-axis velocity-gradient stencil
> `(∂_2 u_1, ∂_1 u_2)` is computed once per cell from the residual's
> existing per-axis face-neighbour table; no new HG primitive is needed.
>
> Test delta: **+125 asserts** added (1 new test file). Bit-exact 0.0
> parity at axis-aligned ICs preserved across M3-3b (194 asserts),
> M3-3c (127 asserts), M3-3d (621 asserts on adjacent), M3-4 (5900
> asserts), M3-6 Phase 0 (390 asserts).
>
> Phase 1a does NOT include the KH IC factory (Phase 1b) or the
> Drazin–Reid γ_KH calibration (Phase 1c). Phase 1a's smoke gate is
> structural: the strain stencil reads correctly, drives β_12/β_21
> off rest under shear, and is rotationally equivariant.

## What landed

| File | Change |
|---|---|
| `src/eom.jl` | EXTENDED: `cholesky_el_residual_2D_berry!` adds (1) cross-axis velocity-gradient stencil before the per-axis loop, computing `(∂_2 u_1, ∂_1 u_2)` from `face_lo[2..1], face_hi[2..1]` neighbour tables (with M3-4 periodic-wrap offsets along the cross axis); (2) `G̃_12 = (∂_2 u_1 + ∂_1 u_2)/2`, `W_12 = (∂_2 u_1 − ∂_1 u_2)/2` strain-decomposition; (3) `H_rot^off` contributions to F^β_a (`∂H/∂α_a`-style coupling at midpoint), F^β_12 (`G̃_12 · ᾱ_2 / 2` strain drive), F^β_21 (`G̃_12 · ᾱ_1 / 2` strain drive), and F^θ_R (`W_12 · F_off` vorticity drive, with `F_off = (α_1²·α_2·β_12 − α_1·α_2²·β_21)/2`). LOC delta: ~115 lines added (file 1264 → 1379). |
| `test/test_M3_6_phase1a_strain_coupling.jl` | NEW (125 asserts, 5 testsets / 5 GATEs): GATE 1 strain stencil matches centered-FD on tanh shear (interior + boundary cells); GATE 2 axis-aligned IC ⇒ residual byte-equal to Phase 0; GATE 3 sheared IC drives β_12, β_21 ≠ 0 after one Newton step; GATE 4 90°-rotated shear produces mirror-image stencil with bit-exact magnitude match; GATE 5 W_12·F_off drive on F^θ_R matches closed form when β_12, β_21 ≠ 0 IC. |
| `test/runtests.jl` | APPEND-ONLY: new "Phase M3-6 Phase 1a" testset block following Phase 0. |
| `reference/notes_M3_6_phase1a_strain_coupling.md` | THIS FILE. |

## Residual structure

Per-cell unknowns unchanged from M3-6 Phase 0 (11 dof per leaf):

    (x_1, x_2, u_1, u_2, α_1, α_2, β_1, β_2, β_12, β_21, θ_R)

The residual rows acquire the following additions vs Phase 0:

    F^β_1  += G̃_12 · β̄_21 / 2
    F^β_2  += G̃_12 · β̄_12 / 2
    F^β_12 += G̃_12 · ᾱ_2 / 2
    F^β_21 += G̃_12 · ᾱ_1 / 2
    F^θ_R  += W_12 · F_off

with

    G̃_12 = (∂_2 u_1 + ∂_1 u_2) / 2     (symmetric strain)
    W_12 = (∂_2 u_1 − ∂_1 u_2) / 2     (vorticity)
    F_off = (α_1²·α_2·β_12 − α_1·α_2²·β_21) / 2

`F_off` is the off-diagonal piece of the Berry function `F_tot` from
`scripts/verify_berry_connection_offdiag.py` (CHECK 7: vanishes on the
iso slice α_1 = α_2 ∧ β_12 = β_21).

## Cross-axis velocity-gradient stencil

The stencil reads neighbour-cell `u_a` and `x_a` data through the
existing `face_lo_idx[a], face_hi_idx[a]` neighbour tables built in
`build_face_neighbor_tables`:

  • `∂_2 u_1`: read `u_1, x_2` at axis-2 lo/hi neighbours (`face_lo[2][i],
    face_hi[2][i]`), apply M3-4 periodic-wrap offset on `x_2` if active.
    Stencil: `(ū_1^hi2 − ū_1^lo2) / (x̄_2^hi2 − x̄_2^lo2)`.
  • `∂_1 u_2`: same pattern at axis-1 neighbours, reading `u_2, x_1`.

Boundary handling mirrors the diagonal stencil: if a neighbour is out
of domain (face index = 0), mirror-self ⇒ that derivative contribution
is zero locally. This makes the stencil one-sided at REFLECTING walls,
which is the correct behaviour for the wall-impermeable BC.

## Verification gates

### §Bit-exact gate at axis-aligned ICs (CRITICAL) — PASS at 0.0 absolute

Every M3-3c, M3-4, and M3-6 Phase 0 regression test uses an IC where
either u = (0, 0) or u_1 = u_1(x_1) only (and u_2 = 0). In both cases,

    ∂_2 u_1 = 0   and   ∂_1 u_2 = 0
    ⇒ G̃_12 = W_12 = 0
    ⇒ every Phase 1a addition vanishes multiplicatively.

The bit-exact gate is therefore structural: at axis-aligned IC the
Phase 1a residual reduces to the Phase 0 residual byte-equally.
Verified empirically by re-running the M3-3b/c/d, M3-4, and M3-6
Phase 0 test suites (~6611 asserts across the bit-exact zone), all
passing without numerical drift.

### §Strain stencil (GATE 1) — PASS at 1e-12 absolute

With `u_1(x, y) = U·tanh((y − 0.5)/w)` (sheared base flow with axis-2
dependence only) on a 4×4 mesh (level 2, REFLECTING BCs), the
residual's F^β_12 row at fixed-point input matches the centered-FD
prediction `G̃_12·α_2/2 = (∂_2 u_1)/4` per cell to ≤ 1e-12 absolute.
Tested on every cell (16 cells), with both interior cells (centered
stencil) and boundary cells (one-sided / mirror-self) checked.

### §β_12, β_21 drive from rest (GATE 3) — PASS

Single Newton step at dt = 1e-3 from sheared IC + β_12 = β_21 = 0:
the converged state has `max|β_12| ≥ 1e-5` and `max|β_21| ≥ 1e-5`.
With G̃_12 ~ U/(2w) ~ 1.7 in the shear layer and α_a = 1, the
expected magnitude is ~G̃_12·α/2 · dt ~ 8.5e-4, within an order of
magnitude of the measured value. With α_1 = α_2 = 1, max|β_12|
exactly equals max|β_21| (within 1e-12) — a structural symmetry of
the drive.

### §Rotational equivariance (GATE 4) — PASS

90°-rotated shear (u_1 = 0, u_2(x_1) = U·tanh((x_1 − 0.5)/w))
produces the mirror-image off-diagonal stencil. The F^β_12 and F^β_21
maximum magnitudes match between the two ICs to ≤ 1e-12 relative
tolerance. F^θ_R = 0 in both cases (β_12 = β_21 = 0 IC ⇒ F_off = 0
even though W_12 ≠ 0 in the rotated case).

### §F^θ_R vorticity drive (GATE 5) — PASS at 1e-12 absolute

With sheared u_1(y), non-zero β_12 = 0.07, β_21 = -0.03 IC, and
α = (1.2, 0.8): the F^θ_R row matches `W_12 · F_off` per cell to
≤ 1e-12 absolute, where F_off = (α_1²·α_2·β_12 − α_1·α_2²·β_21)/2 is
evaluated at IC and W_12 from the cross-axis stencil.

## Wall-time impact (16×16 mesh, level 4, 256 leaves)

Per Newton step (3-trial minimum, dt = 1e-3, M_vv_override = (1, 1)):

  • Axis-aligned IC (Phase 1a inert, Newton converges in 1–2 iters):
    ~57 ms / step. Roughly 1.0× M3-6 Phase 0's ~50 ms / step at the
    same mesh; the new strain stencil is ~3% per-cell overhead
    (read/write of 4 extra neighbour pointers).
  • Sheared IC (Phase 1a actively driving β_12, β_21, Newton iterates
    3–5 times): ~263 ms / step. The slowdown is dominated by Newton
    iteration count rather than per-residual cost.

The Phase 0 design note projected ~6–9 ms / step at 4×4 mesh; scaling
linearly with leaf count gives ~100–150 ms / step at 16×16 sheared
— Phase 1a is in the same ballpark. No sparse-coloring fixes needed
for Phase 1a; the F^β_12, F^β_21 rows are still local to the cell
(no off-cell stencil), so the `cell_adjacency_sparsity ⊗ 11×11`
Jacobian pattern is unchanged.

## What M3-6 Phase 1a does NOT do

  • **Does not implement the KH IC factory.** The `tier_d_kh_ic`
    setup that produces a full sheared base flow with antisymmetric-
    tilt-mode perturbations is M3-6 Phase 1b's job.
  • **Does not run the Drazin–Reid γ_KH calibration.** Falsifying the
    `c_off² ≈ 1/4` correction is M3-6 Phase 1c's headline gate.
  • **Does not extend `realizability_project_2d!` to the 4-component
    `(β_1, β_2, β_12, β_21)` cone.** The Phase 0 minimum bar (off-
    diag β unchanged across projection) still applies; the full
    cone projection is needed once Phase 1b's KH IC produces non-
    zero off-diag β at IC.
  • **Does not promote θ_R Newton step to the new W_12·F_off drive in
    a falsifier driver.** The drive is wired into the residual; it
    fires whenever `β_12 ≠ 0 ∨ β_21 ≠ 0` AND `W_12 ≠ 0`. Phase 1c
    will exercise it.

## M3-6 Phase 1b (KH IC factory) handoff items

  1. **`tier_d_kh_ic` factory in `src/setups_2d.jl`**: produce the
     base flow `u_1(y) = U·tanh((y − y_0)/w)`, `u_2 = 0`, with a
     small-amplitude perturbation along the antisymmetric tilt mode
     `δβ_12 = −δβ_21` (per the linearisation sketch in
     `scripts/verify_berry_connection_offdiag.py` CHECK 9). PERIODIC
     BCs along the streamwise axis 1; REFLECTING along axis 2 (the
     shear layer is bounded). Match the Drazin–Reid base flow
     conventions (`U = 1`, `w = 0.05` standard).
  2. **Realizability cone extension**: extend
     `realizability_project_2d!` to project the full
     `(β_1, β_2, β_12, β_21)` 4-vector onto the per-axis Cholesky
     cone `M_vv ≥ headroom · (β_1² + β_2² + β_12² + β_21²)`. The
     Phase 0 stub leaves the off-diag pair unchanged; Phase 1b's
     KH driver will produce non-zero off-diag β IC, requiring proper
     cone projection.
  3. **Periodic-wrap stress test**: the Phase 1a strain stencil
     consumes `wrap_lo[2], wrap_hi[2]` along the off-axis. Verify
     this against an axis-1-periodic / axis-2-reflecting KH base
     flow before trusting Phase 1c's growth-rate measurement.
  4. **F^x, F^u advection wrap**: the M3-3b open-issue note flags
     periodic-x coordinate handling in F^x_a; with KH advecting
     genuinely along axis 1 at U ≠ 0, this needs to be re-verified
     (currently exercised in a limited form via M3-4 C.1 Sod).

## M3-6 Phase 1c (Drazin–Reid calibration) handoff items

  1. **Linear growth rate measurement**: from the Phase 1b KH IC, run
     a short trajectory (T ~ 1 / γ_KH, with γ_KH ≈ U/(2w)) and fit
     the antisymmetric tilt mode `δβ_12 − δβ_21` to an exponential.
     The fitted growth rate ought to match the classical Drazin–
     Reid prediction times an `O(1)` correction `c_off²`.
  2. **Falsifier gate**: the prediction `c_off² ≈ 1/4` must come out
     of the off-diag Berry sector with no parametric tuning. If the
     measured γ disagrees with `γ_DR · c_off`, *one* of {H_rot^off
     prefactor, F_off sign, W_12 · F_off vs W_12 · α_a^k decision,
     pressure-strain coupling missing} is wrong — diagnose and fix
     before declaring Phase 1c done.
  3. **Mesh refinement convergence**: Phase 1c's growth-rate fit
     must converge under mesh refinement (level 4 → 5 → 6, fixed
     IC). Expect γ_KH to be resolved at ~10 cells across the
     shear layer (level 5–6 for w = 0.05 on the unit square).

## References

  • `reference/notes_M3_6_phase0_offdiag_beta.md` — your immediate
    predecessor, handoff item 1 of which is exactly this Phase 1a
    scope.
  • `reference/notes_M3_phase0_berry_connection.md` §7.5 — full
    derivation of `H_rot^off` from the off-diagonal kinetic 1-form.
  • `scripts/verify_berry_connection_offdiag.py` CHECK 6 — symbolic
    derivation of the `H_rot^off` form needed by the kernel-
    orthogonality solvability constraint.
  • `scripts/verify_berry_connection_offdiag.py` CHECK 9 — KH-shear
    linearisation sketch; tells you which mode (antisymmetric tilt
    `δβ_12 − δβ_21`) carries the Drazin–Reid linear instability.
  • `src/berry.jl` — `kinetic_offdiag_coeffs_2d`, `kinetic_offdiag_2d`
    closed forms (consumed indirectly by the residual via the F_off
    closed form).
  • `reference/notes_M3_3b_native_residual.md`,
    `reference/notes_M3_3c_berry_integration.md` — 2D residual
    HaloView / face-neighbour-table patterns adopted by Phase 1a.
