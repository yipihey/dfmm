# M4 Phase 2 — 3D D.1 Kelvin-Helmholtz falsifier (lift of M4 Phase 1)

> **Status (2026-04-26):** *Implemented + tested + HONESTLY FALSIFICATION_LIFTED*.
> Second sub-phase of M4 (`reference/notes_M4_plan.md`). Closes the
> M4 Phase 1 honest-finding loop in 3D: the closed-loop β_off ↔ β_a
> coupling lifted to the SO(3) Berry block — three pairs (1, 2),
> (1, 3), (2, 3) with their own H_rot^off and H_back per pair —
> produces the same linear-in-t (forced) growth rather than the
> Drazin-Reid exp-in-t (eigenmode) growth. The 2D verdict
> generalizes to 3D.
>
> Test delta: **+1187 asserts** in one new test file
> (`test_M4_phase2_3d_kh_falsifier.jl`, 8 GATEs / 8 testsets). Bit-
> exact 0.0 parity on the M3-7c regression IC (β_off = 0 +
> axis-aligned u) verified by GATE 2.
>
> **Falsifier verdict: HONEST_FALSIFICATION_LIFTED.** At level 3
> (8³ = 512 cells, c_back = 1, T_factor = 0.7): γ_DR = 3.333,
> γ_measured = 5.237, c_off = 1.571. Linear-in-t fits 2700× better
> than exp-in-t (ssr_lin = 1.4e-5 ≪ ssr_exp = 0.037). The methods
> paper §10.5 D.1 broad-band [0.5, 2.0] passes; the aspiration
> [0.8, 1.2] does not (the M4 Phase 1 finding generalizes).

## What landed

| File | Change |
|---|---|
| `src/eom.jl` | EXTENDED (+468 LOC, append-only): `pack_state_3d_kh`, `unpack_state_3d_kh!`, `build_residual_aux_3D_kh`, `cholesky_el_residual_3D_berry_kh!`, `cholesky_el_residual_3D_berry_kh`. |
| `src/newton_step_HG.jl` | EXTENDED (+110 LOC, append-only): `det_step_3d_berry_kh_HG!` Newton driver consuming the 21-dof residual; sparsity prototype `cell_adjacency_sparsity ⊗ ones(21, 21)`. |
| `src/setups_2d.jl` | EXTENDED (+199 LOC, append-only): `allocate_cholesky_3d_kh_fields` (22-named-field allocator), `tier_d_kh_3d_ic_full` (3D KH IC factory). |
| `src/dfmm.jl` | APPEND-ONLY (+9 LOC): re-exports under "Phase M4-2 API" comment block. |
| `experiments/D1_3D_KH_growth_rate.jl` | NEW (~410 LOC): 3D analog of `experiments/D1_KH_growth_rate.jl`. `run_D1_3D_KH_growth_rate`, `fit_linear_growth_rate_3d`, `fit_linear_vs_exp_3d`, `plot_D1_3D_KH_growth_rate`. |
| `test/test_M4_phase2_3d_kh_falsifier.jl` | NEW (~270 LOC, 1187 asserts, 8 testsets / 8 GATEs). |
| `test/runtests.jl` | APPEND-ONLY: new `Phase M4-2` testset block following Phase M4-1. |
| `reference/MILESTONE_4_STATUS.md` | UPDATED: M4-2 row + close synthesis. |
| `reference/figs/M4_phase2_3d_kh_falsifier.png` | NEW: 4-panel headline plot (CairoMakie). |
| `reference/notes_M4_phase2_3d_kh_falsifier.md` | THIS FILE. |

## Mathematical structure

### Per-pair lift

The 2D forward strain coupling H_rot^off and closed-loop H_back are
lifted to 3D pair-by-pair. For each pair (a, b) ∈ {(1, 2), (1, 3),
(2, 3)}:

  H_rot^off,(ab) = G̃_{ab} · (α_a · β_{ba} + α_b · β_{ab}) / 2
  H_back^(ab)    = c_back · G̃_{ab} · (α_b · β_{ab} · β_b
                                       + α_a · β_{ba} · β_a) / 2

with G̃_{ab} = (∂_b u_a + ∂_a u_b)/2 (symmetric strain) and W_{ab} =
(∂_b u_a − ∂_a u_b)/2 (vorticity). Each pair contributes:

  • F^β_{ab} += G̃_{ab} · ᾱ_b / 2 + c_back · G̃_{ab} · ᾱ_b · β̄_b / 2
  • F^β_{ba} += G̃_{ab} · ᾱ_a / 2 + c_back · G̃_{ab} · ᾱ_a · β̄_a / 2
  • F^θ_{ab} += W_{ab} · F_off^(ab) with
        F_off^(ab) = (ᾱ_a²·ᾱ_b·β̄_{ab} − ᾱ_a·ᾱ_b²·β̄_{ba}) / 2

The per-axis Berry α/β-modifications gain pair contributions from each
pair in which the axis participates (axis 1: pairs (1,2) [+] and
(1,3) [+]; axis 2: pair (1,2) [-] and pair (2,3) [+]; axis 3: pairs
(1,3) [-] and (2,3) [-]). The 2D Phase 1 form is preserved per pair
on the SO(2) sub-block.

### Per-cell unknowns (21 dof per leaf)

```
y[21(i-1) +  1.. 3] = (x_1, x_2, x_3)        — Lagrangian positions
y[21(i-1) +  4.. 6] = (u_1, u_2, u_3)        — Lagrangian velocities
y[21(i-1) +  7.. 9] = (α_1, α_2, α_3)        — per-axis Cholesky factors
y[21(i-1) + 10..12] = (β_1, β_2, β_3)        — per-axis conjugate momenta
y[21(i-1) + 13..15] = (θ_12, θ_13, θ_23)     — Berry rotation angles
y[21(i-1) + 16..17] = (β_12, β_21)           — pair (1, 2) off-diag
y[21(i-1) + 18..19] = (β_13, β_31)           — pair (1, 3) off-diag
y[21(i-1) + 20..21] = (β_23, β_32)           — pair (2, 3) off-diag
```

The brief refers to "19 dof" by counting only the Cholesky-sector
unknowns (positions + velocities + α + β + θ_{ab} = 15 base, of which
9 are per-axis Cholesky-driven, plus 6 off-diag β = 19). The actual
residual carries 21 floats per cell once positions/velocities are
included; the entropy `s` is operator-split (not a Newton unknown).

### Bit-exact regression

At β_off = 0 IC + axis-aligned u (every M3-7c regression test):
  • G̃_{ab} = W_{ab} = 0 ∀ pairs (no cross-axis strain).
  • β_{ab} = 0 ∀ slots.
  • All H_rot^off,(ab) and H_back^(ab) contributions vanish
    multiplicatively.
  • F^β_{ab} rows reduce to pure trivial drives (β_{ab}_np1 −
    β_{ab}_n)/dt = 0 at fixed-point input.
  • F^θ_{ab} += W_{ab} · F_off^(ab) = 0.
  • Per-axis Berry α/β modifications match the M3-7c form byte-equal.

The 21-dof residual reduces to the M3-7c 15-dof Berry residual
byte-equal on the first 15 slots; rows 16..21 are trivial. Verified
empirically by GATE 2 (max diff = 0.0 across all 64 cells in a level-2
non-trivial Berry IC with α ≠ 1, β ≠ 0, θ_12 ≠ 0).

## Numerical results (level 3, 8³ = 512 cells)

| Quantity | Value |
|---|---:|
| γ_DR | 3.333 |
| γ_measured (linear fit) | 5.237 |
| c_off | 1.571 |
| ssr_lin | 1.4e-5 |
| ssr_exp | 3.7e-2 |
| linear / exp better fit | LINEAR |
| n_negative_jacobian (total) | 0 |
| Wall-time per step | 3.41 s |
| n_steps for T_factor=0.7 | 30 |
| Total wall (level 3) | 102 s |

## Verification gates (8 testsets, 1187 asserts)

| GATE | Description | Asserts |
|---|---|---:|
| 1 | IC mass conservation + uniform-ρ + α=1 + β=0 + θ=0 + β_off antisymmetric tilt mode | ~899 |
| 2 | Bit-exact: 21-dof residual ≡ 15-dof M3-7c Berry residual on first 15 slots at β_off = 0 IC + axis-aligned u, also at c_back = 0 fallback | 3 |
| 3 | Driver smoke at level 2, T_factor = 0.7: γ_measured > 0, c_off bounded, no NaN | 38 |
| 4 | Level 3 broad-band acceptance c_off ∈ [0.5, 2.0] (methods paper) | 4 |
| 5 | Linear-in-t vs exp-in-t fit comparison: HONEST verdict (which fits better) | 4 |
| 6 | 6-component β cone: n_negative_jacobian = 0 throughout level-3 trajectory | 217 |
| 7 | c_back parameter sweep ∈ {0, 0.5, 1, 2}: no NaN, all return finite γ_measured | 16 |
| 8 | Dimension-lift: (1, 2) pair tilt mode dominates; (1, 3) / (2, 3) pairs stay at 0 (z-symmetric IC has no z-axis cross-axis strain) | 6 |

Total: **+1187 asserts** added. Wall-time of full test suite: ~5 minutes
(GATEs 4, 5, 6 each run a level-3 trajectory).

## c_back parameter sweep (level 2)

| c_back | c_off (level 2) |
|---:|---:|
| 0.0 | 2.660 |
| 0.5 | 2.649 |
| 1.0 | 2.640 |
| 2.0 | 2.626 |

The c_back sweep at level 2 is consistent with M4 Phase 1's 2D
finding: the closed-loop coupling slightly *reduces* c_off (from
2.66 to 2.63, a ~1% effect). Neither value is in the [0.8, 1.2]
aspiration band; both are above the upper edge of [0.5, 2.0] at
level 2 (level 2 trajectories are far from converged — the 30-sample
fit window only spans a fraction of T_KH). At level 3 (the
acceptance gate), c_off = 1.57, in the broad band.

## Honest scientific finding

The M4 Phase 2 lift successfully:

1. **Closes the symplectic loop in 3D** across all three Berry pairs.
   Each pair's `(α_a, β_a, α_b, β_b, β_{ab}, β_{ba}, θ_{ab})` block
   has bidirectional coupling via H_rot^off,(ab) and H_back^(ab),
   identical to the 2D Phase 1 form per pair.

2. **Preserves bit-exact regression** at β_off = 0 IC across the
   entire M3-7c, M3-7d, M3-7e Berry-active 3D test suite. The new
   21-dof residual reduces to the M3-7c 15-dof Berry residual
   byte-equal on the first 15 slots, and the 6 new rows are pure
   kinematic drives at the regression IC.

3. **Does NOT activate the Drazin-Reid eigenmode in 3D either.**
   The KH growth trajectory remains better-fit by linear-in-t than
   exp-in-t at level 3, with c_off ≈ 1.57. The 2D Phase 1 finding
   ("closing the symplectic loop is necessary but not sufficient
   for the eigenmode dynamics") generalizes to 3D: even with three
   additional Berry pairs and the full SO(3) coupling, the
   closed-loop residual reproduces the kinematic strain response
   without exponential amplification.

This is a real scientific result: **the variational scheme's
symplectic structure on the 3D Cholesky-sector Berry block does not
admit the Rayleigh eigenmode at the linear order, even with the
closed-loop H_back per pair**. The interpretation aligns with the
2D Phase 1 reading: the eigenmode requires either (a) a higher-order
Hamiltonian extension (cubic / quartic in perturbation amplitudes
that creates a positive-eigenvalue block), or (b) a per-cell
linearised Rayleigh-equation reconstruction at the stencil level
(Bernstein basis lift, M4 Phase 9 territory), or (c) the symplectic-
natural F^β_off form (no β̇_off; α̇'s and θ̇_R provide the constraint),
which requires a Newton-system regularization for the rank-6 Casimir
kernel.

For the methods paper, the M4 Phase 2 finding is:

  > "The variational scheme's 3D off-diagonal Cholesky sector
  > captures the kinematic strain response under KH-style shear in
  > all three Berry pairs (axes 1-2, 1-3, 2-3) up to the c_off ∈
  > [0.5, 2.0] broad band. The closed-loop symplectic coupling
  > between β_off and β_a in 3D (M4 Phase 2) preserves the bit-exact
  > regression at the M3-7c IC and produces c_off ≈ 1.57 at the
  > level-3 falsifier (8³ = 512 cells). However, the Drazin-Reid
  > eigenmode growth (exp-in-t with γ_KH = U/(2w)) is not activated
  > in 3D, just as in 2D (M4 Phase 1). The methods paper §10.5 D.1
  > broad-band claim is preserved in 3D as in 2D; the tighter [0.8,
  > 1.2] aspiration band requires further physics extensions
  > orthogonal to the symplectic-loop closure."

## 3D ⊂ 2D dimension-lift (verified by GATE 8)

At z-symmetric IC (u_3 = 0, no z-axis structure):
  • ∂_3 u_a = 0 ∀ a (no axis-3 velocity gradient).
  • ∂_a u_3 = 0 ∀ a.
  • G̃_{13} = G̃_{23} = 0 (symmetric pair-strains zero).
  • W_{13} = W_{23} = 0 (vorticity zero).
  • The (1, 3) and (2, 3) pair forcing terms vanish on F^β_{13},
    F^β_{31}, F^β_{23}, F^β_{32}, and F^θ_{13}, F^θ_{23}.
  • The (1, 3) and (2, 3) off-diag β slots stay at 0 throughout the
    trajectory.
  • The dominant tilt mode in pair (1, 2) drives β_12 / β_21 with
    the 2D KH dynamics byte-equal.

GATE 8 verifies these properties at level 2: max|β_13| = max|β_23|
= 0 throughout T_factor = 0.7 trajectory.

## What M4 Phase 2 does NOT do

  • **Does not implement per-cell linearised Rayleigh reconstruction.**
    A per-cell linearised eigenvector projection at the stencil level
    is the natural place to activate the eigenmode dynamics. This is
    M4 Phase 9 (Bernstein reconstruction) territory.
  • **Does not extend `realizability_project_3d!` to the 6-component
    `(β_1, β_2, β_3, β_{12}, β_{21}, β_{13}, β_{31}, β_{23}, β_{32})`
    cone.** The full cone projection would be the 3D analog of M3-6
    Phase 1b's 4-component projection; M4 Phase 2 leaves the
    realizability projection at the 3-component β cone level (as in
    M3-7d). At the level-3 trajectory the 6-component cone is
    well-respected (n_negative_jacobian = 0) but a stress test under
    severe shear could probe the projection extension.
  • **Does not promote the 21-dof residual to a matrix-free
    Newton-Krylov variant.** The M3-8b matrix-free pattern lives at
    the 13-dof and 15-dof level; lifting to 21-dof is straightforward
    (cell_adjacency × 21×21 prototype) but not landed here.
  • **Does not close the methods paper §10.5 D.1 aspiration gate in
    3D.** The c_off ∈ [0.8, 1.2] band is not achieved. The honest
    finding is the deliverable.

## M4 Phase 3 (per-species momentum for D.7) handoff items

  1. **3D dust-trap IC**: extend `tier_d_dust_trap_ic_full` (2D
     Taylor-Green vortex) to 3D ABC flow; multi-species
     `TracerMeshHG3D[:gas, :dust]`.
  2. **Per-species drag relaxation kernel**:
     `∂_t (ρ_k u_k) = -ρ_k (u_k - u_gas) / τ_drag(size)` per species
     per axis, with size-dependent τ_drag.
  3. **2D ⊂ 3D selectivity**: a 3D dust-trap IC with z-symmetric
     base flow should reproduce the 2D D.7 verdicts byte-equal on
     axes 1, 2.

## References

  • `reference/notes_M4_phase1_closed_loop_beta.md` — the 2D
    finding that this phase generalizes.
  • `reference/notes_M3_6_phase1a_strain_coupling.md` — the 2D
    forward strain coupling per pair.
  • `reference/notes_M3_7c_3d_berry_integration.md` — the 3D Berry
    block this phase extends.
  • `reference/notes_M3_7e_3d_tier_cd_drivers.md` — 3D substrate
    (Tier-C / D 3D drivers).
  • `reference/notes_M4_plan.md` §M4-3 — the plan that this phase
    closes (renumbered to M4-2 in shipped order).
  • `src/eom.jl::cholesky_el_residual_3D_berry_kh!` — the 21-dof
    3D residual (M4 Phase 2 lift).
  • `src/newton_step_HG.jl::det_step_3d_berry_kh_HG!` — the M4
    Phase 2 Newton driver.
  • `src/setups_2d.jl::tier_d_kh_3d_ic_full` — the 3D KH IC factory.
  • `experiments/D1_3D_KH_growth_rate.jl` — the M4 Phase 2 falsifier
    driver.
  • `reference/figs/M4_phase2_3d_kh_falsifier.png` — headline plot.
  • `scripts/verify_berry_connection_offdiag.py` — the SymPy
    authority for the 2D form per pair (3D applies pair-by-pair).
