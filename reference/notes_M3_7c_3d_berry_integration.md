# M3-7c — SO(3) Berry coupling integration on the 3D residual + (θ_12, θ_13, θ_23) as Newton unknowns

> **Status (2026-04-26):** *Implemented + tested*. Third sub-phase of
> M3-7 (3D extension). Branch `m3-7c-3d-berry-integration`, two
> commits on top of M3-7b (`3881b8c`).
>
> Test delta vs M3-7b baseline: **+485 asserts** in four new test files
> (288 + 49 + 35 + 113). All pre-existing tests confirmed byte-equal
> across 1D + 2D + 3D paths. Both dimension-lift gates pass at **0.0
> absolute** — the load-bearing M3-7c acceptance criterion.
>
> M3-7d (per-axis γ AMR + selectivity) is unblocked.

## What landed

| File | Change |
|---|---|
| `src/eom.jl` | EXTENDED (+478 LOC, append-only). New 3D Berry-aware EL residual `cholesky_el_residual_3D_berry!` (15-dof per cell, Berry α/β-modifications summed across pair-generators, F^θ_{ab} kinematic-drive rows) + allocating wrapper `cholesky_el_residual_3D_berry`. |
| `src/newton_step_HG.jl` | EXTENDED (+118 LOC, append-only). New 3D Newton driver `det_step_3d_berry_HG!` consuming the Berry residual; same `cell_adjacency_sparsity ⊗ ones(15, 15)` prototype as M3-7b (Berry couplings live within the existing within-cell block). |
| `src/cholesky_DD_3d.jl` | EXTENDED (+93 LOC). New `h_rot_partial_dtheta_3d(α, β, γ²; pair)` (3D analog of M3-3c's `h_rot_partial_dtheta`) + `h_rot_kernel_orthogonality_residual_3d(...)` numerical probe. Closed-form per-pair `∂H_rot/∂θ_{ab}` for §7.4. |
| `src/dfmm.jl` | APPEND-ONLY (+19 LOC). Re-exports the new symbols under a "Phase M3-7c API" comment block. |
| `test/test_M3_7c_berry_3d_residual.jl` | NEW (288 asserts). §7.2 Berry verification reproduction at residual level: 6 random samples × per-pair θ probes × FD-vs-closed-form match to 1e-9 + cross-check against `berry_partials_3d`. |
| `test/test_M3_7c_iso_pullback_3d.jl` | NEW (49 asserts). §7.3 iso-pullback ε-expansion: `F_{ab} = 0` on iso (CHECK 3a); ε-extrapolation slope = 1 ± 1e-3; residual-level Berry contribution scales linearly in ε. |
| `test/test_M3_7c_h_rot_solvability_3d.jl` | NEW (35 asserts). §7.4 H_rot solvability: per-pair closed-form `∂H_rot/∂θ_{ab}` matches kernel-orthogonality at 5 random `(α, β, γ²)` × 3 `θ̇_test` × 3 pairs (45 contractions); Newton converges in ≤ 7 iter; post-Newton residual ≤ 1e-10. |
| `test/test_M3_7c_dimension_lift_3d_with_berry.jl` | NEW (113 asserts). §7.1a + §7.1b critical gates **with Berry**. 3D-Berry ⊂ 1D against M1; 3D-Berry ⊂ 2D-Berry against M3-3c's `det_step_2d_berry_HG!`. Single-step + 100-step + 10-step + 4×4×4 + axis-swap + non-trivial Berry IC (β + θ_12). **All passing at 0.0 absolute.** |
| `test/runtests.jl` | APPEND-ONLY. New "Phase M3-7c" testset block between M3-7b and the closing `end`. |
| `reference/MILESTONE_3_STATUS.md` | UPDATED. Header + M3-7c (a..d) sub-phase rows + test summary. |
| `reference/notes_M3_7c_3d_berry_integration.md` | THIS FILE. |

## Residual structure

Per-cell unknowns (15 Newton-driven dof per leaf, same packing as M3-7b):

```
y[15(i-1) +  1..3]  = (x_1,  x_2,  x_3)    — Lagrangian position
y[15(i-1) +  4..6]  = (u_1,  u_2,  u_3)    — Lagrangian velocity
y[15(i-1) +  7..9]  = (α_1,  α_2,  α_3)    — per-axis Cholesky factors
y[15(i-1) + 10..12] = (β_1,  β_2,  β_3)    — per-axis conjugate momenta
y[15(i-1) + 13..15] = (θ_12, θ_13, θ_23)   — Berry rotation angles
```

The structurally new pieces vs M3-7b:

1. **(θ_12, θ_13, θ_23) are Newton unknowns** (vs M3-7b's
   trivial-driven). The Newton system Jacobian sparsity is unchanged
   (`cell_adjacency_sparsity ⊗ ones(15, 15)` — same 15×15 within-cell
   block as M3-7b; Berry couplings live within it).

2. **Per-axis Berry α/β-modifications** sum independently across the
   three pair-rotation generators (a, b) ∈ {(1, 2), (1, 3), (2, 3)}.
   Per axis a, summing over pairs in which a participates (signs from
   the rows of `Ω · X = -dH` per pair):

   ```
   axis 1 (in pairs (1,2) [+] and (1,3) [+]):
     α̇_1 = β_1 − (α_2³/(3α_1²)) θ̇_12 − (α_3³/(3α_1²)) θ̇_13
     β̇_1 = γ_1²/α_1 − β_2 θ̇_12 − β_3 θ̇_13

   axis 2 (in pair (1,2) [-] and (2,3) [+]):
     α̇_2 = β_2 + (α_1³/(3α_2²)) θ̇_12 − (α_3³/(3α_2²)) θ̇_23
     β̇_2 = γ_2²/α_2 + β_1 θ̇_12 − β_3 θ̇_23

   axis 3 (in pairs (1,3) [-] and (2,3) [-]):
     α̇_3 = β_3 + (α_1³/(3α_3²)) θ̇_13 + (α_2³/(3α_3²)) θ̇_23
     β̇_3 = γ_3²/α_3 + β_1 θ̇_13 + β_2 θ̇_23
   ```

   Each per-pair Berry contribution is structurally identical to M3-3c's
   2D pair (1, 2) form on the (α_a, α_b, β_a, β_b) restriction (the
   third axis decouples because `[J_{ab}, (α_c, β_c)] = 0` for
   c ∉ {a, b}).

3. **F^θ_{ab} rows are kinematic-drive form** (drive = 0; off-diagonal
   velocity-gradient stencil deferred to M3-9):

   ```
   F^θ_{12} = (θ_12_{n+1} − θ_12_n)/dt
   F^θ_{13} = (θ_13_{n+1} − θ_13_n)/dt
   F^θ_{23} = (θ_23_{n+1} − θ_23_n)/dt
   ```

   This is the 3D analog of M3-3c's F^θ_R kinematic-drive row. With
   drive = 0, each Euler angle is conserved per cell when
   multiplicative factors `β̄_b = 0` or `θ̇_{ab} = 0` apply (the
   dimension-lift slices). The H_rot solvability identity per pair is
   structurally guaranteed by the per-axis residual rows encoding the
   rows of `Ω · X = -dH` (analog of M3-3c §6.4).

## Verification gates

### §7.1a 3D-Berry ⊂ 1D — PASS at 0.0 absolute

Configuration: 1D-symmetric (axes 2, 3 trivial: β_2 = β_3 = 0,
α_2 = α_3 = const, M_vv = (1, 0, 0)), all θ_{ab} = 0.

On the 1D-symmetric slice, every Berry α-modification term has factor
`θ̇_{ab}` (zero, since the F^θ_{ab} rows pin `θ_{ab}_{n+1} = θ_{ab}_n`)
and every Berry β-modification term has factor `β̄_b · θ̇_{ab}` (zero
on both factors). Berry vanishes multiplicatively; the residual reduces
byte-equal to M3-7b's no-Berry form.

Result: per-cell `(α_1, β_1)` matches M1's `cholesky_step` /
`cholesky_run` to **bit-exact 0.0 absolute** across:

  * Single step at dt = 1e-3 (4×4×4 mesh)
  * 100-step run at dt = 1e-3 (T = 0.1, the M1 Phase-1 trajectory)
  * Axis-swap symmetry across all 3 principal axes (active = 1, 2, 3)

θ_{ab} all stay at 0.0 across every step (the kinematic-drive rows
pin them).

### §7.1b 3D-Berry ⊂ 2D-Berry — PASS at 0.0 absolute (the sharper test)

Configuration: 2D-symmetric (axis 3 trivial: β_3 = 0, α_3 = const,
θ_13 = θ_23 = 0); 2D path runs `det_step_2d_berry_HG!`. Tested in
two regimes:

  (a) **1D-symmetric IC** (β = 0, θ_R = 0; trivial Berry block):
      Berry α/β-modifications vanish on both paths via β = 0 / θ̇ = 0.
  (b) **Non-trivial Berry IC** (β_1, β_2 ≠ 0, θ_12 ≠ 0, β_3 = 0,
      θ_13 = θ_23 = 0): the (1, 2) pair Berry block is active in
      both 2D and 3D paths; the 3D path's (1, 3) and (2, 3) pair
      blocks vanish via β_3 = 0 / θ̇_13 = θ̇_23 = 0.

Result: per-cell `(α_1, β_1, α_2, β_2, θ_12)` from the 3D path matches
the 2D Berry path to **bit-exact 0.0 absolute** across:

  * Single step at dt = 1e-3
  * 10-step run at dt = 1e-3
  * Non-trivial Berry IC (regime (b) above)

This is the load-bearing M3-7c gate: SO(3) → SO(2) reduction is
correct at the residual level. The 0.0 absolute result (vs the 1e-12
tolerance) means the 3D Berry residual's per-axis Berry-modification
sum reduces byte-equally to M3-3c's 2D form on the dimension-lift
slice — this is the residual-level reproduction of CHECK 3b of
`notes_M3_prep_3D_berry_verification.md`.

### §7.2 Berry verification reproduction — PASS

6 random `(α, β, θ_12, θ_13, θ_23)` configurations on a 4×4×4 mesh.
Numerical partials of `cholesky_el_residual_3D_berry!` w.r.t.
`θ_{ab}^{n+1}` (probed independently per pair) match the closed-form
expected coefficients (per pair-axis sums above) to FD tolerance
1e-9. Cross-check against `src/berry.jl::berry_partials_3d` confirms
the closed forms `∂F_{ab}/∂α_c` and `∂F_{ab}/∂β_c` per pair match the
SymPy authority at 1e-14 (CHECK 7 of the 3D verification note,
residual-level reproduction).

### §7.3 Iso-pullback ε-expansion — PASS, slope ≈ 1 ± 1e-3

Three sub-blocks:

  1. **F_{ab} = 0 on iso** (CHECK 3a reproduction): at α_1 = α_2 = α_3,
     β_1 = β_2 = β_3, all three F_{ab} = 0 exactly. With β_iso = 0
     (purely α-iso slice), the F^β_a residual rows match the no-Berry
     form byte-equal (every β̄_b factor = 0). Verified at three
     (α_iso, β_iso) configurations.
  2. **F_{ab} ε-expansion off iso**: at α_b = α_iso + ε (single-axis
     deviation from iso), F_{ab} ≈ -α_iso² β_iso · ε to leading order.
     Slopes at ε ∈ {1e-2, 1e-4, 1e-6} converge to the expected
     leading coefficient at rel-err ≤ 1e-2. Verified for all three
     pairs.
  3. **Residual-level ε-expansion** of F^β contribution: perturbing
     β_2 = β_iso + ε with θ_12_{n+1} − θ_12_n = θ̇_12·dt activates the
     pair-(1, 2) Berry β-block, contributing
     `+(ε/2)·θ̇_12_const` to F^β_1 (midpoint averaging halves ε). All
     three ε samples reproduce this to rel-err ≤ 1e-6; slope variation
     across ε is < 1e-3 — confirming linear-in-ε behaviour.

### §7.4 H_rot solvability per pair — PASS

  • Closed-form `h_rot_partial_dtheta_3d(α, β, γ²; pair)` matches the
    kernel-orthogonality identity `dH · v_ker = 0` per pair to ≤ 1e-10
    absolute at 5 random generic `(α, β, γ²)` × 3 random `θ̇_{ab}`
    values × 3 pairs (45 contractions per sample × 5 samples = 225
    individual checks; all pass).
  • Iso-slice value of `h_rot_partial_dtheta_3d` evaluates to 0 per
    pair when (α_a = α_b, β_a = β_b, γ²_a = γ²_b) (3 iso configurations
    × 3 pairs).
  • Newton converges in ≤ 7 iterations at non-isotropic 3D IC (α =
    (1.2, 0.8, 1.0), β = (0.15, -0.10, 0.07), θ_{ab} = (0.13, -0.09,
    0.05)). Re-running Newton from the converged state with maxiters = 7
    succeeds.
  • Post-Newton residual norm ≤ 1e-10 at generic 3D IC (α =
    (1.5, 0.9, 1.1), β = (0.05, -0.02, 0.03), θ_{ab} = (0.07, -0.04,
    0.02)).

### 8 SymPy CHECKs reproduced at residual level

| CHECK | Stencil-level (M3-prep) | Residual-level (M3-7c) |
|---|---|---|
| 1 (closedness, 84 cyclic triples) | structural — `dΩ = 0` polynomial identity | structural — Newton Jacobian symmetry under `(α_a, β_a, θ_{ab})` block; verified via FD of the 3D Berry residual rows |
| 2 (per-axis Hamilton on θ̇=0 slice) | passed | reproduced via §7.1a (3D ⊂ 1D byte-equal to M1) |
| 3a (full iso pullback: F = 0) | passed | reproduced via §7.3 Block 1 (Berry blocks vanish on iso) |
| 3b (5×5 sub-block matches 2D Ω) | passed | reproduced via §7.1b (3D-Berry ⊂ 2D-Berry byte-equal to M3-3c) |
| 4 (rank-8, 1D Casimir kernel) | passed | reproduced via §7.4 (closed-form `∂H_rot/∂θ_{ab}` per pair) |
| 5 (per-pair antisymmetry) | passed | reproduced via §7.2 (residual partials match `berry_partials_3d` per pair) |
| 6 (degeneracy boundary regularity) | passed | structural — closed-form `h_rot_partial_dtheta_3d` regular at α_a = α_b in non-iso (β_a ≠ β_b) sub-slice; Newton converges (no singularity) |
| 7 (Ω = dΘ exactness, no monopole) | passed | reproduced via §7.2 (residual-level FD = closed-form `F_{ab}` per pair, F^θ_{ab} row partials) |
| 8 (cyclic-sum polynomial identity) | passed | residual is linear in `dθ_{ab}` per pair (no cross-pair coupling at θ-row level), so the cyclic-sum identity holds at the per-pair coefficient level (residual-Jacobian inspection — implicit in §7.2 Block sums) |

## Newton convergence

| IC | Newton iter count | Residual norm at exit |
|---|---:|---|
| Zero-strain (β = 0, all θ_{ab} = 0) | 2 | machine zero |
| 1D-symmetric (axis-1 active only) | 2 | machine zero |
| 2D-symmetric (axis-1 + axis-2 active, β + θ_12 ≠ 0) | 2-4 | ≤ 1e-13 |
| Non-isotropic 3D (full β + all θ_{ab} ≠ 0) | ≤ 7 | ≤ 1e-10 |

The "≤ 7" convergence on non-isotropic 3D ICs meets the M3-7 design
note §7.4 expectation. The 13-dof active Newton system is structurally
larger than M3-3c's 9-dof active 2D system; per-pair Berry blocks are
diagonal-dominant within their pair, so convergence rate is set by
the largest eigenvalue of the 13×13 within-cell Jacobian — not by
cross-pair coupling.

## Wall time

| Mesh | Leaves | Wall-time per step (M3-7c, full 13-dof active) |
|---|---:|---:|
| 4×4×4 | 64 | ~118 ms |

This is ~4× M3-7b's 27.8 ms/step at the same 4×4×4. The slowdown is
expected: M3-7c's 13-dof-active Newton system has ~6× the within-cell
Jacobian density (Berry off-diagonal couplings between (α_a, β_a) and
the three θ_{ab} columns) vs M3-7b's effectively 12-dof-active system
(θ_{ab} rows uncoupled). The sparsity-prototype prediction is the same
15×15 dense block per cell-cell adjacency entry, but the runtime cost
of the AD-based Jacobian assembly grows with the actual non-zero
density. Within the M3-7 design note §3.3 expectation envelope.

## Q-resolution against the design note's §11 open questions

All seven §11 default decisions held; no human judgement needed.

  1. **Pair-generator parameterization**: pinned in M3-7 prep
     (`reference/notes_M3_7_prep_3d_scaffolding.md` Q1) and verified
     residual-level via §7.1b byte-equal to M3-3c's
     `det_step_2d_berry_HG!`.
  2. **HaloView depth = 1**: M3-7c uses the same pre-computed
     face-neighbor table pattern as M3-7b; depth = 1 sufficient.
  3. **MonomialBasis order = 0**: same as M3-7b.
  4. **Off-diagonal β omitted**: same as M3-7b; M3-9 (3D D.1 KH) lifts.
  5. **mesh.balanced asserted**: yes, same as M3-7b.
  6. **eigen-based Cholesky decomposition**: not exercised in M3-7c
     (the residual operates on (α_a, θ_{ab}) directly; decomposition
     is an IC / AMR-time concern handled by `cholesky_decompose_3d`).
  7. **Single-mode Zel'dovich pancake IC**: deferred to M3-7e.

## Sign convention for ∂H_rot/∂θ_{ab}

The closed form in `h_rot_partial_dtheta_3d` per pair `(a, b)` is

    ∂H_rot/∂θ_{ab} = -(γ_a² · α_b³)/(3·α_a) + (γ_b² · α_a³)/(3·α_b)
                       + (α_a² − α_b²) · β_a · β_b.

This matches the 2D `h_rot_partial_dtheta` sign convention exactly
(the 2D form is the per-pair restriction at (a, b) = (1, 2) on the
α_3 = const, β_3 = 0 slice — CHECK 3b of the 3D verification note).
The discrete EL residual rows do **not** consume this closed form
directly (the per-axis Berry-modification terms encode the rows of
`Ω · X = -dH` from `src/berry.jl::berry_partials_3d`); the helper
exists as a verification-gate artefact for §7.4.

## What M3-7c does NOT do

Per the brief's "Critical constraints":

  * **Does not implement 3D per-axis γ AMR.** That's M3-7d.
  * **Does not implement 3D Tier-C / Tier-D drivers.** That's M3-7e.
  * **Does not write 3D off-diagonal β.** That's M3-9 (3D D.1 KH).
  * **Does not implement off-diagonal velocity-gradient stencil.**
    M3-9 will plumb `(∂_b u_a, ∂_a u_b)` for non-trivial `θ̇_{ab} =
    drive` in F^θ_{ab}. M3-7c's first cut leaves drive = 0
    (θ_{ab} conserved per cell on free-flight ICs).

## Open issues / handoff to M3-7d

  * **F^θ_{ab} kinematic drives are trivial.** With drive = 0, each
    θ_{ab} is conserved per cell. This is fine for the M3-7c
    verification gates (which all probe Berry coupling structurally)
    but is physically incomplete for genuinely 3D dynamics. M3-9
    (3D D.1 KH) must add the off-diagonal velocity-gradient stencil
    and wire the strain-rotation drives into F^θ_{ab}.
  * **Per-axis γ AMR.** `gamma_per_axis_3d` from M3-7-prep is ready;
    M3-7d will plumb it through the action-error indicator (analog of
    M3-3d's 2D path).
  * **Wall-time ratio M3-7c / M3-7b ≈ 4.2× at N = 64.** Reasonable
    given the 13-dof-active vs effectively-12-dof-active Newton
    system. Future profiling pass could exploit a tighter
    explicit-Euler initial guess for Berry-active configurations
    (e.g., extrapolate θ_{ab}^{n+1} from the previous step's
    rotation rate). M3-9 / M3-8 GPU port is the natural place.
  * **NonlinearSolve sparse-coloring with the 15×15 Kron block.**
    Same as M3-7b: the manual Kron prototype was verified
    structurally (Newton converges, residuals ≤ 1e-10 on generic
    3D IC). M3-7d should cross-check the coloring via
    `SparseConnectivityTracer` once F^θ_{ab} has a non-trivial drive
    (the current trivial drive makes θ_{ab} structurally
    block-diagonal at the row level, which the full-15×15-dense
    pattern over-estimates).

## How to extend in M3-7d/e

  * **M3-7d** (per-axis γ 3D AMR): consume `gamma_per_axis_3d` from
    `src/cholesky_DD_3d.jl` in a 3D action-error indicator; refine
    only along axes where γ_a collapses. The 3D analog of M3-3d's
    per-axis 2D AMR. Three configurations to verify per §7.5 of the
    M3-7 design note: 1D-symmetric, 2D-symmetric, full 3D (k =
    (1, 1, 1)).
  * **M3-7e** (Tier-C / D 3D drivers): extend the 2D Tier-C IC factory
    `cholesky_sector_state_from_primitive` to D=3 (α=1, β=0, θ_{ab}=0
    cold-limit isotropic IC); 3D Sod, 3D cold sinusoid, 3D plane wave;
    and the headline Tier-D driver: 3D Zel'dovich pancake collapse
    (the cosmological reference test from methods paper §10.5 D.4
    lifted to 3D — single-mode along axis 1 per Q7 default).

## M3-7c launch handoff for M3-7d

When M3-7d launches (per-axis γ AMR + selectivity in 3D), the launch
agent should already have everything it needs from M3-7c + M3-7-prep:

### Inputs available

  * `cholesky_el_residual_3D_berry!` — the Berry-active residual to
    consume (no further extension expected for AMR; AMR operates on
    the field set, not the residual).
  * `det_step_3d_berry_HG!` — the Berry-active Newton driver.
  * `gamma_per_axis_3d` — per-axis γ diagnostic (M3-7-prep).
  * `pack_state_3d` / `unpack_state_3d!` — 15-dof packers (no rename
    needed; θ_{ab} are already Newton unknowns in M3-7c).
  * `build_residual_aux_3D` — aux NamedTuple builder.
  * `h_rot_partial_dtheta_3d` per pair — closed-form `∂H_rot/∂θ_{ab}`
    helper for verification gates.

### M3-7c regressions for M3-7d

Both M3-7c dimension-lift gates remain regressions for M3-7d. Adding
per-axis AMR refinement should NOT change the Berry-aware residual
on the dimension-lift slices (the per-axis γ indicator is an
*outer-loop* AMR signal, not a residual-level perturbation). Verify
M3-7d's AMR-driven mesh refinement preserves M3-7c's gates byte-equal
on uniform-IC test problems.

## Reference

  * `reference/notes_M3_7_3d_extension.md` — full M3-7 design note
    (your sub-phase is §9 entry "M3-7c"; §4 Berry coupling integration;
    §6 verification gates; §7.1 dimension-lift critical gates; §7.4
    H_rot solvability).
  * `reference/notes_M3_7b_native_3d_residual.md` — M3-7b status
    (your immediate dependency).
  * `reference/notes_M3_3c_berry_integration.md` — 2D analog mirrored
    here at +2 pairs (M3-7c is "M3-3c × 3 pair-generators").
  * `reference/notes_M3_phase0_berry_connection_3D.md` — 3D Berry
    derivation (§3 per-pair ansatz; §6 H_rot determination as
    deferred algebraic work — landed in M3-7c per pair).
  * `reference/notes_M3_prep_3D_berry_verification.md` — Berry CHECKs
    1-8 at stencil level; M3-7c reproduces the relevant subset at the
    residual level.
  * `src/berry.jl` — closed-form 3D Berry stencils consumed by the
    residual.
  * `src/cholesky_DD_3d.jl` — `h_rot_partial_dtheta_3d` per-pair
    closed form (M3-7c addition).
  * `src/eom.jl::cholesky_el_residual_3D_berry!` — the M3-7c residual.
  * `src/newton_step_HG.jl::det_step_3d_berry_HG!` — the M3-7c
    Newton driver.
