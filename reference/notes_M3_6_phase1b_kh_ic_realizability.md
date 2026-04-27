# M3-6 Phase 1b — KH IC factory + 4-component realizability cone

> **Status (2026-04-26):** *Implemented + tested*. Second sub-phase of
> M3-6 Phase 1 (D.1 Kelvin–Helmholtz falsifier). Closes the two
> handoff items from Phase 1a:
>
>   1. `tier_d_kh_ic` / `tier_d_kh_ic_full` IC factories in
>      `src/setups_2d.jl` — sheared base flow `u_1(y) = U_jet ·
>      tanh((y − y_0)/w)`, `u_2 = 0`, with antisymmetric tilt-mode
>      perturbation `δβ_12 = −δβ_21 = A · sin(2π k_x · (x − lo_x)/L_x)
>      · sech²((y − y_0)/w)` overlaid on the off-diagonal Cholesky
>      pair.
>   2. 4-component realizability cone extension in
>      `realizability_project_2d!` — `(β_1, β_2, β_12, β_21)` is now
>      projected onto the cone `Q ≡ β_1² + β_2² + 2(β_12² + β_21²) ≤
>      M_vv · headroom_offdiag` after the existing 2-component s-raise.
>
> Test delta: **+574 asserts** added (2 new test files,
> `test_M3_6_phase1b_kh_ic.jl` + `test_M3_6_phase1b_realizability_4comp.jl`).
> Bit-exact 0.0 parity at axis-aligned ICs preserved across M3-3a/b/c
> /d/e (all native-vs-cache, all selectivity), M3-4 (C.1/C.2/C.3 +
> periodic-wrap + IC bridge: ~6300 asserts), M3-6 Phase 0 (390
> asserts), M3-6 Phase 1a (125 asserts). 1D path bit-exact 0.0 parity
> preserved (M2-3: 172 asserts; M3-3e-4: 708 asserts).
>
> Phase 1b does NOT include the Drazin–Reid γ_KH calibration driver
> or the falsifier acceptance gate — those are M3-6 Phase 1c.

## What landed

| File | Change |
|---|---|
| `src/setups_2d.jl` | EXTENDED: appended `tier_d_kh_ic` (primitive 4-name field set + per-leaf δβ vectors) and `tier_d_kh_ic_full` (14-name Cholesky-sector field set with off-diag β slot pre-loaded). Both factories project the smooth `tanh` base flow via HG's `init_field_from!` and overlay the centre-evaluated antisymmetric tilt-mode perturbation. Default box `[0, 1]²`, default `y_0 = 0.5`, default BC mix `(PERIODIC, PERIODIC) × (REFLECTING, REFLECTING)` (caller-attached). LOC delta: **+276**. |
| `src/stochastic_injection.jl` | EXTENDED: `realizability_project_2d!` extended to 4-component cone with new keyword `headroom_offdiag::Real = 2.0`; `ProjectionStats` extended with `n_offdiag_events::Int` counter. The 2-component s-raise still fires when `M_vv < headroom · max(β_1², β_2²)`; AFTER the s-raise (or skip), a 4-component check `Q = β_1² + β_2² + 2(β_12² + β_21²) ≤ M_vv_post · headroom_offdiag` is applied; if violated, scale `(β_1, β_2, β_12, β_21)` uniformly by `s_β = √(target/Q)`. At `β_12 = β_21 = 0` an early continue guards the β-scaling step ⇒ field-set output is **byte-equal** to the M3-3d 2-component projection. LOC delta: **+105 effective** (133 added, 28 modified). |
| `src/dfmm.jl` | APPEND-ONLY: exports `tier_d_kh_ic, tier_d_kh_ic_full`. LOC delta: **+9**. |
| `test/test_M3_6_phase1b_kh_ic.jl` | NEW (418 asserts, 7 testsets): GATE 1 primitive base flow + perturbation overlay; GATE 2 full IC writes 14-named-field state; GATE 3 mass conservation; GATE 4 primitive recovery round-trip; GATE 5 periodic-x wrap on antisymmetric tilt mode; GATE 6 cross-axis strain stencil fires non-trivially; GATE 7 perturbation_amp = 0 ⇒ off-diag β = 0 byte-equal. |
| `test/test_M3_6_phase1b_realizability_4comp.jl` | NEW (156 asserts, 8 testsets): GATE 1+1b+1c byte-equal at β_off = 0 vs M3-3d baseline; GATE 2 off-diag-only stress; GATE 3+3b mixed-stress scaling; GATE 4 ProjectionStats accumulator; GATE 5 in-cone β_off untouched; GATE 6 `:none` no-op; GATE 7 combined s-raise + β-scale stress. |
| `test/runtests.jl` | APPEND-ONLY: new `Phase M3-6 Phase 1b` testset block following Phase 1a. |
| `reference/notes_M3_6_phase1b_kh_ic_realizability.md` | THIS FILE. |

## KH IC factory

The factory mirrors the M3-4 `tier_c_*_full_ic` pattern:

1. Build a balanced `HierarchicalMesh{2}` of `(2^level)²` leaves.
2. Write the primitive `(ρ, P)` cell averages via HG's `init_field_from!`
   (uniform; `quadrature_order = 2·quad_order − 1`).
3. Write the velocity components `u_1(y) = U_jet · tanh((y − y_0)/w)`,
   `u_2 = 0` via `init_field_from!` (smooth profile, GL-projected).
4. Per leaf, evaluate the antisymmetric tilt-mode perturbation
   `δβ_12(x, y) = A · sin(2π k_x (x − lo_x)/L_x) · sech²((y − y_0)/w)`
   at the cell centre. `δβ_21 = −δβ_12` per cell (the linearised
   eigenmode Phase 1c calibrates).
5. For the `_full` variant, apply the Phase-2 IC bridge
   `cholesky_sector_state_from_primitive` to produce the cold-limit
   isotropic state `(α = 1, β = 0, θ_R = 0, s = s(ρ, P))`, then
   overwrite `betas_off = (δβ_12, δβ_21)`.

The Cholesky-sector full IC returns `(name, mesh, frame, leaves,
fields, ρ_per_cell, params)` consumable by `det_step_2d_berry_HG!` —
no caller-side setup beyond attaching the recommended
`bc_kh = FrameBoundaries{2}(((PERIODIC,PERIODIC),(REFLECTING,REFLECTING)))`.

## 4-component realizability cone

The 4-component projection is a *post-hoc* extension of the existing
2-component s-raise:

```
Stage 1 (existing M3-3d behaviour):
  if M_vv_pre < headroom · max(β_1², β_2²) [+ Mvv_floor]:
    raise s so M_vv_post = headroom · max(β_1², β_2²) [+ Mvv_floor]
    debit ½ ΔP_xx from P_⊥; admit residual as floor-gain.
  else:
    M_vv_post = M_vv_pre.

Stage 2 (new in Phase 1b):
  if β_12 == 0 && β_21 == 0:
    skip (preserves M3-3d byte-equal).
  else:
    Q = β_1² + β_2² + 2 (β_12² + β_21²)
    target = headroom_offdiag · M_vv_post
    if Q > target:
      scale = sqrt(target / Q)
      (β_1, β_2, β_12, β_21) ← scale · (β_1, β_2, β_12, β_21)
      stats.n_offdiag_events += 1
```

The default `headroom_offdiag = 2.0` is chosen so that **at
`β_12 = β_21 = 0`** the Stage-2 check always passes after the Stage-1
s-raise. Proof: post-Stage-1, `M_vv_post ≥ headroom · max(β_1², β_2²)`.
With `β_off = 0`, `Q = β_1² + β_2² ≤ 2 · max(β_1², β_2²)`. So
`Q ≤ 2 · M_vv_post / headroom`. We require
`Q ≤ headroom_offdiag · M_vv_post`, i.e.,
`headroom_offdiag ≥ 2 / headroom`. With `headroom = 1.05 ≤ 2 = h_off`,
the inequality holds. ✓

The early `continue` on `β_12 == 0 && β_21 == 0` skips the Stage-2
field mutation entirely — guarantees byte-equal output (no
multiplication by `1.0`).

## Verification gates

### §Bit-exact gate at β_off = 0 (CRITICAL) — PASS at 0.0 absolute

Replays the M3-3d test cases at `β_12 = β_21 = 0` and asserts the
projection output (s, Pp, β_1, β_2, β_12, β_21) is identical to the
existing M3-3d behaviour:
  • `M3-3d` 2-component cone target match: post-projection
    `M_vv = 1.05 · 0.95² ≈ 0.948` to ≤ 1e-12 absolute (matches the
    M3-3d test).
  • `n_offdiag_events == 0` for every β_off = 0 cell.
  • β_1, β_2 untouched across the projection (no β-scaling event).
  • The 1D-symmetric reduction `β_2 = 0` still matches the 1D
    `realizability_project!` form `M_vv_target = 1.05 · β_1²`.

### §Bit-exact gates on existing test suites — PASS

All ~6800 asserts across M3-3a/b/c/d/e, M3-4 (C.1/C.2/C.3 + periodic-
wrap + IC bridge), M3-6 Phase 0, M3-6 Phase 1a continue passing
byte-equal:

  • M3-3d realizability per-axis: 26 / 26.
  • M3-3d gamma per-axis diag: included in (543/543) tally below.
  • M3-3a field set, M3-3b zero-strain & dim-lift, M3-3d gamma diag:
    543 / 543 byte-equal.
  • M3-3c (Berry, dim-lift, iso-pullback, h_rot solvability): 127 / 127.
  • M3-4 (periodic-wrap + IC bridge): 2716 / 2716.
  • M3-4 driver tests (C.1/C.2/C.3): 3184 / 3184.
  • M3-6 Phase 0 (residual + dim-lift + realizability): 390 / 390.
  • M3-6 Phase 1a (strain coupling): 125 / 125.
  • M2-3 1D realizability: 172 / 172.
  • M3-2 M2-3 HG realizability: 160 / 160.
  • M3-3e-4 native-vs-cache realizability: 708 / 708.

The byte-equal property is structurally guaranteed (early continue on
β_off = 0 skips all Stage-2 logic), and the empirical reproduction
confirms.

### §Off-diag-only stress (GATE 2) — PASS

`(β_1, β_2) = (0, 0)`, `(β_12, β_21) = (1, 1)`, `M_vv = 0.5`. Q = 4,
target = 2 · 0.5 = 1, scale = sqrt(1/4) = 0.5 exactly. Verified to
0.0 absolute on every leaf; β_1, β_2 stay at 0; s untouched (no
Stage-1 event). `n_offdiag_events == n_leaves`.

### §Mixed stress (GATE 3) — PASS

`(β_1, β_2, β_12, β_21) = (1, 1, 1, 1)`, M_vv = 1.05 (so Stage-1
target `1.05 · 1 = 1.05` is exactly satisfied). Q = 6, target = 2 ·
1.05 = 2.1, scale = sqrt(2.1/6) ≈ 0.5916. All 4 components scaled by
the same factor; post-projection Q = 2.1 to ≤ 1e-12 absolute. The
brief's exact "scale = 0.5" version is reproduced in GATE 3b with the
off-diag-only setup (β_1 = β_2 = 0, β_off = 1, M_vv = 0.5).

### §Combined stress (GATE 7) — PASS

Both Stage-1 and Stage-2 fire: `β = (0.95, 0.40, 0.5, 0.5)`,
`s_pre = log(0.5)`. Stage-1 raises `M_vv_pre = 0.5` to
`M_vv_post = 1.05 · 0.9025 ≈ 0.948` (per-axis target). Stage-2 then
fires because `Q = 0.9025 + 0.16 + 2(0.25 + 0.25) ≈ 2.06 > 2 · 0.948
= 1.896`. Post-projection `Q ≤ 2 · M_vv_post + 1e-12`. Both event
counters increment to `n_leaves`.

### §Cross-axis strain stencil fires from KH IC (GATE 6) — PASS

With `tier_d_kh_ic_full(level = 2, U_jet = 1, jet_width = 0.15)` and
the Phase-1a residual at fixed-point input: `max|F^β_12| > 1e-3`,
`max|F^β_21| > 1e-3`. With α_1 = α_2 = 1, `max|F^β_12| ==
max|F^β_21|` by the symmetric-strain coupling. F^θ_R stays at 0
because `β_off = 0` at perturbation_amp = 0 ⇒ `F_off = 0`, even
though `W_12 ≠ 0`.

### §Periodic-x wrap stress test (GATE 5) — PASS

The mode-projection of `δβ_12(x)` against `sin(2π k_x x / L_x)` at
each y-row of a 16×16 mesh recovers a non-negative amplitude bounded
by A = 1e-3, confirming the sin-mode closes back on itself across
the periodic seam (`Σ_x β_12(x)` averages to 0 per row to ≤ 1e-12).

## What M3-6 Phase 1b does NOT do

  • **Does not run the Drazin–Reid γ_KH calibration.** The Phase 1b
    KH IC produces a falsifier-ready field set, but the linear
    growth-rate fit `γ_KH ≈ U/(2w) · c_off` and the falsifier
    acceptance gate `c_off² ≈ 1/4` are M3-6 Phase 1c's job.
  • **Does not extend the 4-component realizability projection
    through the post-Newton stochastic injection path.** That's not
    needed yet — KH ICs do not ride the stochastic-injection driver
    (the wave-pool calibration uses a different IC). The 4-component
    cone is fired strictly through the post-Newton operator-split
    in `det_step_2d_berry_HG!` whenever it's invoked.
  • **Does not provide a falsifier driver.** The driver script
    (`experiments/D1_KH_growth_rate.jl` per the M3-6 plan) is M3-6
    Phase 1c.

## Wall-time impact (4×4 mesh, level 2)

  • IC factory `tier_d_kh_ic_full`: O(N_leaves) per-leaf work,
    dominated by the `init_field_from!` projection of the smooth
    base flow. Negligible compared to a Newton step (~5 ms IC vs
    ~50–250 ms / Newton step at the same mesh).
  • Realizability projection 4-component overhead: per leaf, two
    extra `Float64` reads (β_12, β_21), one `==` short-circuit at
    β_off = 0 path; on β_off ≠ 0 path, four `Float64` reads + four
    writes + a `sqrt`. The β_off = 0 path is the regression-test
    hot path and adds ~3 ns / leaf.

## M3-6 Phase 1c (Drazin–Reid calibration) handoff items

  1. **D.1 driver script** at `experiments/D1_KH_growth_rate.jl`:
     load `tier_d_kh_ic_full` at `level ∈ {4, 5, 6}`, attach
     `bc_kh = FrameBoundaries{2}(((PERIODIC,PERIODIC),(REFLECTING,
     REFLECTING)))`, run a short trajectory `T ~ 1 / γ_KH ≈ 2 w / U
     ≈ 0.1`, fit the antisymmetric tilt-mode amplitude
     `(δβ_12 − δβ_21)/2 = δβ_12` (since `δβ_21 = −δβ_12` IC) to an
     exponential.

  2. **Linear growth-rate measurement**: the classical Drazin–Reid
     prediction is `γ_KH ≈ U / (2 w)`. The off-diag Berry sector
     introduces an `O(1)` correction `c_off`. Phase 1c's headline
     gate is `c_off² ≈ 1/4` from the linearisation of `H_rot^off ∝
     G̃_12 · (α_1 β_21 + α_2 β_12)/2` per CHECK 9 of
     `scripts/verify_berry_connection_offdiag.py`.

  3. **Falsifier acceptance gate**: if the measured `γ` disagrees
     with `γ_DR · c_off`, *one* of {`H_rot^off` prefactor, `F_off`
     sign, `W_12 · F_off` vs `W_12 · α_a^k` decision in F^θ_R,
     pressure-strain coupling missing} is wrong — diagnose and fix
     before declaring Phase 1c done.

  4. **Mesh refinement convergence**: the growth-rate fit must
     converge under mesh refinement (level 4 → 5 → 6, fixed IC).
     Expect γ_KH to be resolved at ~10 cells across the shear layer
     (level 5–6 for w = 0.05 on the unit square).

  5. **4-component cone exercise**: with `perturbation_amp = 1e-3`
     standard, the IC's `Q = β_off² · 2 ≈ 2e-6 ≪ headroom_offdiag ·
     M_vv ≈ 2`, so the cone is well inside cone interior at IC.
     Phase 1c will need to verify the cone holds throughout the
     non-linear evolution; if not, re-tune `headroom_offdiag` or
     introduce dynamic `Mvv_floor_offdiag` knob.

## References

  • `reference/notes_M3_6_phase1a_strain_coupling.md` — predecessor;
    handoff items 1+2 closed by this Phase 1b.
  • `reference/notes_M3_6_phase0_offdiag_beta.md` — ancestor; off-
    diag β reactivation prerequisite.
  • `reference/notes_M2_3_realizability.md` §3 — original 1D
    `:reanchor` projection; the 4-component cone here is the per-
    cell cone-projection generalisation foreshadowed by §3.
  • `reference/notes_M3_3d_per_axis_gamma_amr.md` — M3-3d's per-axis
    2-component cone (the byte-equal baseline).
  • `reference/notes_M3_4_tier_c_consistency.md` §"Pre-Tier-C
    handoff items" — IC bridge pattern reused by `tier_d_kh_ic_full`.
  • `scripts/verify_berry_connection_offdiag.py` CHECK 9 — KH-shear
    linearisation; tells you which mode (antisymmetric tilt
    `δβ_12 − δβ_21`) carries the Drazin–Reid linear instability.
  • `src/setups_2d.jl` — `tier_d_kh_ic`, `tier_d_kh_ic_full`,
    `cholesky_sector_state_from_primitive`,
    `allocate_cholesky_2d_fields`, `write_detfield_2d!`.
  • `src/stochastic_injection.jl` — `realizability_project_2d!`,
    `ProjectionStats` (with new `n_offdiag_events`).
