# M3-3c — Berry coupling + θ_R as Newton unknown

> **Status (2026-04-26):** *Implemented + tested*. Third sub-phase of
> M3-3 (`reference/notes_M3_3_2d_cholesky_berry.md` §9). Adds the
> Berry-connection coupling to the 2D EL residual and promotes θ_R
> from a fixed IC value to a Newton unknown.
>
> Test delta vs M3-3b baseline: **+127 asserts** (3767 + 1 deferred →
> 3894 + 1 deferred). 1D-path bit-exact parity holds. Dimension-lift
> parity gate (§6.1 of the design note): **0.0 absolute** to M1's
> Phase-1 zero-strain trajectory across single-step + 100-step runs
> on 4×4 and 8×8 meshes (Berry vanishes identically on the
> 1D-symmetric slice — that's the sharp test).

## What landed

| File | Change |
|---|---|
| `src/eom.jl` | EXTENDED: `cholesky_el_residual_2D_berry!` evaluates the 9-dof Cholesky-sector EL residual with closed-form Berry partials; `pack_state_2d_berry` / `unpack_state_2d_berry!` are the 9-dof pack/unpack helpers. ~280 LOC added (file total ~880 LOC). |
| `src/newton_step_HG.jl` | EXTENDED: `det_step_2d_berry_HG!` — Newton driver that wraps `cholesky_el_residual_2D_berry!` with `cell_adjacency_sparsity ⊗ 9×9` Jacobian. ~120 LOC added. |
| `src/cholesky_DD.jl` | EXTENDED: `h_rot_partial_dtheta` (closed-form `∂H_rot/∂θ_R` from the kernel-orthogonality constraint) + `h_rot_kernel_orthogonality_residual` (numerical probe). ~80 LOC added. |
| `src/dfmm.jl` | APPEND-ONLY: re-exports the new symbols. |
| `test/test_M3_3c_dimension_lift_with_berry.jl` | NEW: 22 asserts. §6.1 critical gate re-verification with Berry coupling enabled. |
| `test/test_M3_3c_berry_residual.jl` | NEW: 72 asserts (8 random samples × 9 sub-tests each). §6.2 Berry verification — residual partials match `berry_partials_2d`. |
| `test/test_M3_3c_iso_pullback.jl` | NEW: 10 asserts. §6.3 iso-pullback ε-expansion in three blocks. |
| `test/test_M3_3c_h_rot_solvability.jl` | NEW: 23 asserts. §6.4 H_rot solvability — closed-form vs kernel-orthogonality + Newton convergence at non-isotropic IC. |
| `test/runtests.jl` | APPEND-ONLY: new "Phase M3-3c" testset between M3-3b and M3-2. |
| `reference/notes_M3_3c_berry_integration.md` | THIS FILE. |

## Residual structure

Per-cell unknowns (9 dof per leaf):

    (x_1, x_2, u_1, u_2, α_1, α_2, β_1, β_2, θ_R)

θ_R is now a Newton unknown (was held fixed in M3-3b). Entropy `s` is
still frozen across the Newton step (M1 convention; updates are
operator-split). `Pp`, `Q` ride along on the field set as post-Newton
sectors. Off-diagonal `β_{12}, β_{21}` remain pinned to zero (per Q3 of
the design note).

The residual rows per axis `a ∈ {1, 2}` are the M3-3b form augmented
with Berry coupling derived from the rows of `Ω · X = -dH` for the
closed 5×5 symplectic / Poisson form
`ω_{2D} = α_1²dα_1∧dβ_1 + α_2²dα_2∧dβ_2 + dF∧dθ_R` with
`F = (α_1³β_2 − α_2³β_1)/3`:

    F^x_a    = (x_a_np1 − x_a_n)/dt − ū_a
    F^u_a    = (u_a_np1 − u_a_n)/dt + (P̄_a^hi − P̄_a^lo)/m̄_a
    F^α_1    = (α_1_np1 − α_1_n)/dt − β̄_1 + (ᾱ_2³/(3ᾱ_1²)) θ̇_R_h
    F^α_2    = (α_2_np1 − α_2_n)/dt − β̄_2 − (ᾱ_1³/(3ᾱ_2²)) θ̇_R_h
    F^β_1    = (β_1_np1 − β_1_n)/dt + (∂_1 u_1) β̄_1 − γ̄_1²/ᾱ_1 + β̄_2 θ̇_R_h
    F^β_2    = (β_2_np1 − β_2_n)/dt + (∂_2 u_2) β̄_2 − γ̄_2²/ᾱ_2 − β̄_1 θ̇_R_h
    F^θ_R    = (θ_R_np1 − θ_R_n)/dt

with `θ̇_R_h = (θ_R_np1 − θ_R_n)/dt`. The 9th row F^θ_R encodes the
kinematic equation `θ̇_R = drive`, where `drive = W̃_12 − S̃_12·… +
2M̃_xv,12/…` is the off-diagonal strain/vorticity in the principal-axis
frame. In M3-3c's first cut on the M3-3b pressure stencil:

  • The off-diagonal velocity-gradient stencil is not yet implemented
    (`(∂_2 u_1, ∂_1 u_2)` are zero in the residual). M3-3d / M3-6
    activate it for the D.1 KH falsifier.
  • Off-diagonal β is pinned ⇒ M̃_xv,12 = 0.

So `drive = 0` in M3-3c; the residual pins θ_R to its IC value across
each step. This is consistent with the dimension-lift gate (§6.1):
on the 1D-symmetric slice, all Berry α/β-modification terms vanish
identically because β̄_2 = 0, θ̇_R_h = 0 (F^θ_R pins it).

## Q-resolution against the design note's §10 open questions

All four §10 default decisions held; no human judgement needed. Q4
(`mesh.balanced == true`) is asserted in `det_step_2d_berry_HG!`. The
4 sub-phase Q3 decision (omit `β_{12}, β_{21}`) carries through —
they remain absent from the field set.

## Verification gates

### §6.1 Dimension-lift gate (CRITICAL) — PASS at 0.0 absolute

The single most important M3-3c acceptance criterion: 2D 1D-symmetric
configuration must reproduce M1's 1D bit-exact results to ≤ 1e-12.
With Berry now in the residual, the test is sharper than M3-3b's:
the Berry α/β-modification terms must vanish identically on the
1D-symmetric slice (β_2 = 0, θ_R = 0, α_2 = const).

Result: per-cell `(α_1, β_1)` matches M1's `cholesky_step` to
**bit-exact 0.0 absolute** across:

  • Single step at dt = 1e-3 (4×4 mesh)
  • 100-step run at dt = 1e-3 (T = 0.1, M1 Phase-1 trajectory)
  • 8×8 mesh (level 3, 64 leaves) — single step + 10-step
  • Axis-swap symmetry (active axis = 2)

θ_R stays at 0.0 across every step (the kinematic-drive row pins it).

The 0.0-absolute result is structurally guaranteed: on the 1D-symmetric
slice, β̄_2 = 0 and the F^θ_R row enforces θ̇_R = 0, so every Berry
term in the per-axis rows is multiplied by 0. The Newton system
factorizes into the 8-dof M3-3b sub-system + a trivial θ_R row.

### §6.2 Berry verification reproduction — PASS

Sampled 8 random `(α, β, θ_R)` configurations on a 4×4 balanced 2D
mesh. Verified that the numerical partials of `cholesky_el_residual_2D_berry!`
w.r.t. `θ_R^{n+1}` match the closed forms

    ∂F^α_1 / ∂θ_R_np1 = (ᾱ_2³ / (3 ᾱ_1²)) / dt
    ∂F^α_2 / ∂θ_R_np1 = -(ᾱ_1³ / (3 ᾱ_2²)) / dt
    ∂F^β_1 / ∂θ_R_np1 = β̄_2 / dt
    ∂F^β_2 / ∂θ_R_np1 = -β̄_1 / dt
    ∂F^θ_R / ∂θ_R_np1 = 1/dt

to FD tolerance 1e-9 (with Δθ_R probe = 1e-7). Cross-checked against
`src/berry.jl::berry_partials_2d` using the symplectic-weight identity
`∂F^*/∂θ̇_R · α_a² = berry_partials_2d entry`.

### §6.3 Iso-pullback ε-expansion — PASS, slope ≈ 1.003

Three sub-blocks:

  1. **Berry function `F` itself** scales linearly in `(α_1 − α_2)`
     when β_1 = β_2 is held off-iso: at α_1 = α_0, α_2 = α_0+ε,
     β_1 = β_2 = β_0:
         F = ((α_0+ε)³ − α_0³) · β_0 / 3 ≈ α_0² β_0 · ε.
     Measured slope: **1.003** (target 1.0 ± 0.05).
  2. **F vanishes identically on iso submanifold**: at α_1 = α_2 =
     α_0+εδ_α, β_1 = β_2 = β_0+εδ_β, F == 0 exactly (axis-swap
     antisymmetry preserved). Tested at three ε values; all returned
     `F == 0.0`.
  3. **Action contribution F · θ̇_R on a 4×4 mesh** at α_1 = α_2+ε,
     β_1 = β_2 = β_0, θ̇_R = 50 — same slope ≈ 1.003.

The brief specified "scales as O(ε)" — measured 1.003 confirms.

### §6.4 H_rot solvability — PASS

  • Closed-form `h_rot_partial_dtheta(α, β, γ²)` matches the
    kernel-orthogonality identity `dH · v_ker = 0` to ≤ 1e-10
    absolute at 5 random generic `(α, β, γ²)` points × 3 random
    `θ̇_R` values per point (15 evaluations of
    `h_rot_kernel_orthogonality_residual`). Iso-slice value of
    `h_rot_partial_dtheta` evaluates to 0 at α_1 = α_2 + γ_1² = γ_2².
  • Newton converges in ≤ 5 iterations at non-isotropic IC
    (α = (1.2, 0.8), β = (0.15, -0.10), θ_R = 0.13). Re-running
    Newton from the converged state with maxiters = 5 succeeds.
  • Post-Newton residual norm ≤ 1e-10 at generic 2D IC.

## Sign convention for ∂H_rot/∂θ_R

The closed form in `h_rot_partial_dtheta` is

    ∂H_rot/∂θ_R = -(γ_1² α_2³)/(3 α_1) + (γ_2² α_1³)/(3 α_2)
                   + (α_1² − α_2²) β_1 β_2.

This has the **opposite overall sign** of the magnitude written in
`reference/notes_M3_phase0_berry_connection.md` §6.6 — but matches
the SymPy verification output of `scripts/verify_berry_connection.py`
CHECK 7. The §6.6 design-note text writes the magnitude only; the
SymPy script solves the kernel-orthogonality constraint directly and
gives the sign explicitly. We adopt the SymPy convention for
algebraic consistency. The discrete EL residual rows do NOT consume
this closed form directly (the per-axis Berry-modification terms
implement the rows of `Ω · X = -dH` from the closed-form Berry
partials in `src/berry.jl`); the helper exists as a verification-gate
artefact for §6.4.

## What M3-3c does NOT do

Per the brief's "Critical constraints":

  • **Does not retire the cache_mesh shim.** That's M3-3e. The 1D
    path continues to delegate to M1's `det_step!`.
  • **Does not implement per-axis γ AMR.** That's M3-3d.
  • **Does not add off-diagonal β.** That's M3-6 (D.1 KH falsifier).
  • **Does not implement the periodic-x coordinate wrap.** Carried
    over from the M3-3b open issue: the residual treats `x_a` as
    per-cell scalar and does not handle periodic coordinate wrap-
    around for genuinely advecting flows. M3-3c's tests use
    REFLECTING BCs and `M_vv_override = constant` to sidestep this;
    PERIODIC BCs only work for cold-limit / zero-strain ICs. M3-3d /
    M3-3e are the natural place to implement.
  • **Does not implement off-diagonal velocity gradients
    `(∂_2 u_1, ∂_1 u_2)`.** Required for non-trivial `θ̇_R = drive`
    in F^θ_R. M3-3c's first cut leaves drive = 0 (θ_R conserved per
    cell). M3-3d / M3-6 implement this for the D.1 KH falsifier.

## Open issues / handoff to M3-3d

  • **F^θ_R kinematic drive is trivial.** With drive = 0, θ_R is
    conserved per cell. This is fine for the M3-3c verification gates
    (which all probe Berry coupling structurally) but is physically
    incomplete for genuinely 2D dynamics. M3-3d / M3-6 must add the
    off-diagonal velocity-gradient stencil and wire the
    `W̃_12 − S̃_12·…` drive into F^θ_R.
  • **Per-axis γ AMR.** `gamma_per_axis_2d` from M3-3a is ready; M3-3d
    will plumb it through the action-error indicator.
  • **Wall-time ratio M3-3c / M3-3b ≈ 2.1× at N = 16.** Reasonable
    given the 9N × 9N vs 8N × 8N Jacobian system and the per-cell
    cached α midpoints for cross-axis Berry terms. Future profiling
    pass could exploit a tighter explicit-Euler initial guess for
    Berry-active configurations.
  • **NonlinearSolve sparse-coloring with the 9×9 Kron block.** The
    `cell_adjacency_sparsity ⊗ ones(9, 9)` construction was verified
    structurally (Newton converges, residuals ≤ 1e-10). M3-3d should
    cross-check the coloring via `SparseConnectivityTracer` once
    F^θ_R has a non-trivial drive (the current trivial drive makes
    θ_R block-diagonal at the structural level, which the
    full-9×9-dense pattern over-estimates).

## How to extend in M3-3d/e

  • **M3-3d** (per-axis γ AMR): consume `gamma_per_axis_2d` from
    `src/cholesky_DD.jl` in the action-error indicator; refine only
    along axes where γ collapses (Tier-C C.2 selectivity test).
    Optionally implement off-diagonal velocity-gradient stencil and
    wire the F^θ_R kinematic drive.
  • **M3-3e** (cache_mesh retirement): drop the 1D `cache_mesh::Mesh1D`
    shim from `src/newton_step_HG.jl` and rebuild the 1D path on the
    native HG residual.

## Reference

  • `reference/notes_M3_3_2d_cholesky_berry.md` — full M3-3 design
    note (your sub-phase is §9 entry "M3-3c").
  • `reference/notes_M3_3a_field_set_cholesky.md`,
    `reference/notes_M3_3b_native_residual.md` — your dependencies.
  • `reference/notes_M3_phase0_berry_connection.md` — Berry derivation
    (§3 kinematic equation, §6 verifications, §6.6 H_rot constraint).
  • `reference/notes_M3_prep_berry_stencil.md` — Berry stencil API
    (`src/berry.jl`).
  • `scripts/verify_berry_connection.py` — SymPy 2D verification
    (CHECK 7 has the kernel-direction + h_rot solvability output).
  • `src/berry.jl` — closed-form Berry stencils consumed by the
    residual.
