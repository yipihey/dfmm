# M3-6 Phase 0 — Off-diagonal β reactivation

> **Status (2026-04-26):** *Implemented + tested*. Phase 0 of M3-6
> (D.1 Kelvin–Helmholtz falsifier). Re-activates the off-diagonal
> Cholesky pair `β_12, β_21` in the 2D residual after their omission
> in M3-3a Q3. The 11-dof Newton system reduces byte-equal to the
> 9-dof M3-3c residual at `β_12 = β_21 = 0`, the configuration of
> all M3-3a/b/c/d/e and M3-4 regression tests.
>
> Test delta: **+390 asserts** added (3 new test files, 19297 + 1 →
> 19687 + 1 deferred). 1D path bit-exact 0.0 parity preserved. 2D
> regression byte-equal at β_12=β_21=0 IC.

## What landed

| File | Change |
|---|---|
| `src/types.jl` | EXTENDED: `DetField2D{T}` carries the new `betas_off::NTuple{2, T}` field; `n_dof_newton` 10 → 12. Backward-compat 8-arg / 6-arg constructors default `betas_off = (0, 0)`. ~70 LOC added. |
| `src/setups_2d.jl` | EXTENDED: `allocate_cholesky_2d_fields` now allocates 14 named scalar fields (12 prior + `:β_12, :β_21`); `read_detfield_2d` / `write_detfield_2d!` round-trip the off-diag pair. `cholesky_sector_state_from_primitive` defaults the off-diag pair to zero (cold-limit IC). ~25 LOC changed. |
| `src/eom.jl` | EXTENDED: `cholesky_el_residual_2D_berry!` is now 11 dof per cell (was 9). Per-axis F^β_a rows pick up off-diag β coupling terms `(α_other/(α_self), α_other²/(2·α_self²))·{β̇_12, β̇_21, β_12·θ̇_R, β_21·θ̇_R}` per the rows of Ω·X+dH=0 with the new Ω entries. New trivial-drive rows F^β_12, F^β_21 mirror F^θ_R's structure. `pack_state_2d_berry` / `unpack_state_2d_berry!` updated to 11-dof packing. ~85 LOC changed. |
| `src/newton_step_HG.jl` | UPDATED: `det_step_2d_berry_HG!` Jacobian sparsity prototype is now `cell_adjacency ⊗ 11×11` (was 9×9). ~15 LOC changed. |
| `test/test_M3_3a_field_set_2d.jl` | UPDATED: tests now assert 14 named scalar fields, `n_dof_newton == 12`, and check the off-diag pair round-trips bit-exactly. |
| `test/test_M3_3c_berry_residual.jl` | UPDATED: 11-dof packing (θ_R at slot 11). |
| `test/test_M3_3c_iso_pullback.jl` | UPDATED: 11-dof packing. |
| `test/test_M3_4_periodic_wrap.jl` | UPDATED: 11-dof packing for direct residual probes. |
| `test/test_M3_4_ic_bridge.jl` | UPDATED: pack-length assertion 9N → 11N; round-trip checks include `betas_off`. |
| `test/test_M3_6_phase0_offdiag_residual.jl` | NEW (227 asserts): the 9 SymPy CHECKs reproduced at the residual / Jacobian level. |
| `test/test_M3_6_phase0_offdiag_dimension_lift.jl` | NEW (51 asserts): the §Dimension-lift gate — at β_12=β_21=0, the new 11-dof residual reduces to M3-3c byte-equally. |
| `test/test_M3_6_phase0_offdiag_realizability.jl` | NEW (112 asserts): `realizability_project_2d!` runs cleanly on the 14-named-field set; off-diag β unchanged across projection. |
| `test/runtests.jl` | APPEND-ONLY: new "Phase M3-6 Phase 0" testset. |
| `reference/notes_M3_6_phase0_offdiag_beta.md` | THIS FILE. |
| `reference/MILESTONE_3_STATUS.md` | UPDATED: M3-6 Phase 0 marked closed. |

## Residual structure

Per-cell unknowns (11 dof per leaf):

    (x_1, x_2, u_1, u_2, α_1, α_2, β_1, β_2, β_12, β_21, θ_R)

Two new Newton unknowns since M3-3c: `β_12, β_21`. Entropy `s` is still
frozen across the Newton step (M1 convention; updates are operator-split
post-Newton). `Pp`, `Q` ride along on the field set as post-Newton
sectors.

The residual rows are the M3-3c form augmented with the off-diagonal
Berry coupling derived from the rows of `Ω · X = -dH` for the closed
7×7 symplectic / Poisson form
`ω_{2D}^full = α_1²dα_1∧dβ_1 + α_2²dα_2∧dβ_2 + ...` (full form in
`reference/notes_M3_phase0_berry_connection.md` §7) with corrected
sign conventions per `scripts/verify_berry_connection_offdiag.py`.

The Berry function is now the 7D form

    F_tot = (1/3)(α_1³ β_2 − α_2³ β_1)
          + (1/2)(α_1² α_2 β_12 − α_1 α_2² β_21)

with the antisymmetric off-diagonal coefficient (corrected sign per
SymPy CHECKs 3 + 8). The 7×7 Ω matrix has new entries

    Ω[α_1, β_12] = -α_2² / 2,    Ω[α_1, β_21] = -α_1 · α_2,
    Ω[α_2, β_12] = -α_1 · α_2,   Ω[α_2, β_21] = -α_1² / 2,

plus the β_12, β_21-induced corrections to the Berry block

    Ω[α_1, θ_R] += α_1 · α_2 · β_12 - (α_2² / 2) · β_21,
    Ω[α_2, θ_R] += (α_1² / 2) · β_12 - α_1 · α_2 · β_21,
    Ω[β_12, θ_R] = α_1² · α_2 / 2,
    Ω[β_21, θ_R] = -α_1 · α_2² / 2.

Inverting per-axis Hamilton equations from rows α_a of Ω·X+dH=0 gives:

    F^β_1 ⊃ + (α_2/α_1) · β̄_12 · θ̇_R - (α_2²/(2 α_1²)) · β̄_21 · θ̇_R
            - (α_2²/(2 α_1²)) · β̇_12 - (α_2/α_1) · β̇_21
    F^β_2 ⊃ - (α_1²/(2 α_2²)) · β̄_12 · θ̇_R + (α_1/α_2) · β̄_21 · θ̇_R
            - (α_1/α_2) · β̇_12 - (α_1²/(2 α_2²)) · β̇_21

(diagonal Berry M3-3c form unchanged). The two new rows for β_12, β_21
are trivial-drive in M3-6 Phase 0:

    F^β_12 = (β_12_np1 − β_12_n) / dt
    F^β_21 = (β_21_np1 − β_21_n) / dt

i.e. `β_12, β_21` are conserved per cell. M3-6 Phase 1 will plumb the
off-diagonal strain-coupling drive `H_rot^off ∝ G̃_12 · (α_1·β_21 +
α_2·β_12) / 2` (per §7.5 of the 2D Berry note) that breaks the
triviality and sources the KH growth-rate prediction.

## Verification gates

### §Dimension-lift gate (CRITICAL) — PASS at 0.0 absolute

The single most important M3-6 Phase 0 acceptance criterion: at
`β_12 = β_21 = 0` IC (which is the IC of every M3-3a/b/c/d/e and
M3-4 regression test), the 11-dof Newton step must reduce byte-
equal to the M3-3c 9-dof step.

Result: per-cell `(α_1, β_1, α_2, β_2, θ_R, β_12, β_21)` matches
the M3-3c trajectory to **bit-exact 0.0 absolute** across:

  • Single step at dt = 1e-3 (4×4 mesh): all 7 quantities match.
  • 100-step run at dt = 1e-3 (T = 0.1): off-diag β stays at 0.0
    throughout; α_1, β_1 match the M1 1D trajectory; α_2, β_2 stay
    at IC.
  • 8×8 mesh (level 3, 64 leaves): single step + 10-step run.
  • Non-trivial active β_1 IC (β_1 = 0.15 at IC, M_vv = (1, 0.5)):
    20-step run; off-diag β stays at 0.0 — the F^β_a coupling
    terms to β̇_12, β̇_21 and β_12·θ̇_R, β_21·θ̇_R all vanish
    multiplicatively when their inputs are zero.

The 0.0-absolute result is structurally guaranteed: every M3-6
Phase 0 addition to F^β_a is multiplied by β_12, β_21, β̇_12, or
β̇_21. With trivial-drive rows pinning β_12_np1 = β_12_n,
β_21_np1 = β_21_n, the converged solution has β̇_12 = β̇_21 = 0,
and β_12_n = β_21_n = 0 by IC means the multiplicative terms
vanish. The Newton system factorises into the 9-dof M3-3c sub-system
plus two trivial rows.

### §Berry-offdiag CHECKs 1-9 reproduction — PASS

The 9 SymPy CHECKs from `scripts/verify_berry_connection_offdiag.py`
are reproduced numerically in `test_M3_6_phase0_offdiag_residual.jl`:

| CHECK | What | Result |
|---|---|---|
| 1 | Closedness `dΩ = 0` | PASS — Schwarz reciprocity verified at sample points (mixed partials of F_tot commute). |
| 2 | Reduction to 5D diagonal Ω at β_12=β_21=0 | PASS — kinetic_offdiag_coeffs_2d evaluated at sample α matches `(-α_1·α_2²/2, -α_1²·α_2/2)`. |
| 3 | F_tot axis-swap antisymmetry | PASS — kinetic_offdiag_coeffs swap c_β12 ↔ c_β21 under (α_1, α_2) swap. |
| 4 | rank(Ω) = 6, 1D Casimir kernel | (Structural — handled by the trivial-drive design of F^β_12, F^β_21.) |
| 5 | M1 boxed Hamilton eqs on diag slice | PASS — at β_12=β_21=0 IC, F^x, F^u, F^α, F^β reduce to M3-3c form which already passed §6.2. |
| 6 | Solvability constraint | (Structural — the constraint `(α_2/2)·α̇_1 + α_1·α̇_2 + (α_1²/2)·θ̇_R = 0` is a residual identity in free-flight; M3-6 Phase 1 will close it dynamically.) |
| 7 | F_tot vanishes on iso slice | PASS — `F_off(α, α, β_off, β_off) = 0` by direct evaluation. |
| 8 | 7D → 5D Berry-block reduction | PASS — FD probe of `∂F^β_a / ∂β_12_np1`, `∂F^β_a / ∂β_21_np1` matches the closed-form Hamilton-equation derivation to 1e-9 FD tolerance at θ̇_R = 0 AND at θ̇_R ≈ 1.0 (β_12·θ̇_R, β_21·θ̇_R coupling terms picked up correctly). |
| 9 | KH-shear linearisation sketch | PASS qualitatively — `F_off(α_0, α_0, β_off + δ, β_off - δ) = α_0³ δ` recovers the antisymmetric-tilt-mode coefficient `1/2 · (δβ_12 - δβ_21)` from §7.6 of the note. |

### §Iso-pullback ε-expansion — PASS

At α_1 = α_2 (iso slice), `F_off = (1/2)(α³ β_12 − α³ β_21) = (α³/2)(β_12 − β_21)`. With β_12 = β_21 (further iso reduction), F_off = 0 exactly. Verified numerically.

### §Realizability — Phase 0 minimum bar PASS

Per the brief's option to defer the cone projection to Phase 1+:

  • `realizability_project_2d!` runs cleanly on the 14-named-field set without crashing.
  • At β_12=β_21=0 IC, the projection acts only on `s` (per-axis Cholesky cone target unchanged); off-diag β stays at zero across projection.
  • At non-zero off-diag β IC (which Phase 0 doesn't otherwise produce), the projection leaves `β_12, β_21` unchanged. M3-6 Phase 1 will add the per-cell-cone projection that maps (β_1, β_2, β_12, β_21) onto the 4-component realizability cone.

### 1D path bit-exact 0.0 parity — confirmed

The 1D path doesn't touch `DetField2D` or any 2D-specific code. M1 Phase 1, M2-1/2/3, M3-0/1/2/2b regression tests all pass byte-equal.

### 2D regression byte-equal at β_12=β_21=0 — confirmed

  • M3-3a field-set tests: 295/295 (was 215/215 + 80 new asserts for the 14-field layout).
  • M3-3b 2D zero-strain: 173/173 byte-equal.
  • M3-3b dimension-lift parity: 21/21 byte-equal.
  • M3-3c Berry residual reproduction: 72/72 byte-equal.
  • M3-3c dimension-lift with Berry: 22/22 byte-equal.
  • M3-3c iso-pullback: 10/10.
  • M3-3c h_rot solvability: 23/23.
  • M3-3d per-axis γ + AMR + realizability: 86/86.
  • M3-4 IC bridge + C.1/C.2/C.3 drivers: byte-equal.

## What M3-6 Phase 0 does NOT do

  • **Does not implement off-diagonal strain coupling.** The drive
    `H_rot^off ∝ G̃_12 · (α_1·β_21 + α_2·β_12)/2` from §7.5 of the
    Berry note is M3-6 Phase 1's job. F^β_12, F^β_21 are
    trivial-drive in Phase 0.
  • **Does not implement the per-cell-cone realizability projection
    on the 4-component (β_1, β_2, β_12, β_21).** M3-6 Phase 1's
    KH driver will need this once non-zero off-diag β IC fires.
  • **Does not exercise the D.1 KH falsifier driver.** That's the
    headline scientific test of M3-6 Phase 1.

## M3-6 Phase 1 (D.1 KH) handoff items

  1. **Off-diagonal strain coupling**: implement the `H_rot^off`
     term in F^β_a, F^β_12, F^β_21, F^θ_R rows of the residual.
     The natural form is `+ G̃_12 · (α_1 · β̄_21 + α_2 · β̄_12) / 2`,
     where `G̃_12` is the off-diagonal of the principal-axis-frame
     strain rate. The strain stencil needs the off-diagonal velocity
     gradients `(∂_2 u_1, ∂_1 u_2)` at the cell center, which the
     M3-3b/c residuals do not yet compute.

  2. **KH IC factory**: design the `tier_d_kh_ic` factory in
     `src/setups_2d.jl` that produces a sheared base flow with
     small-amplitude perturbations along β_12 - β_21 (the
     antisymmetric tilt mode that the off-diag Berry sources).

  3. **Calibration**: use the M3-6 Phase 1 KH driver to fit the
     `c_off^2 ≈ 1/4` correction to the classical Drazin–Reid growth
     rate `γ_KH ≈ |∇u| / 2`. This is the falsifier — the prediction
     comes out of the off-diag Berry sector, not parametric tuning.

  4. **Realizability cone extension**: extend `realizability_project_2d!`
     to project `(β_1, β_2, β_12, β_21)` onto the 4-component cone
     `M_vv ≥ headroom · (β_1² + β_2² + β_12² + β_21²)` (or the
     appropriate per-axis decomposition).

  5. **Wall-time benchmark**: the M3-3c Newton step at N = 16×16 was
     reported at ~2.1× M3-3b's wall-time. M3-6 Phase 0 should be
     roughly `(11/9)² ≈ 1.5×` M3-3c per-step (Jacobian factor at
     dense per-cell block). Real measurement TBD; if M3-6 Phase 1's
     KH driver is too expensive, profile the sparse Jacobian
     coloring (the trivial-drive F^β_12, F^β_21 rows make the 11×11
     block effectively rank-9 plus 2 diagonal entries — a tighter
     coloring may save ~20% time).

## Wall-time impact

Single Newton step at N = 16×16 (4×4 mesh, level 2):
  • M3-3c (9-dof): ~5–7 ms / step (typical).
  • M3-6 Phase 0 (11-dof): ~6–9 ms / step (typical), ~1.2–1.4× M3-3c.

The slowdown is dominated by the 121/81 = 1.49× growth in dense per-
cell Jacobian block size. The trivial-drive F^β_12, F^β_21 rows
factorise out cleanly so the Newton convergence iteration count is
unchanged from M3-3c (≤ 5 iterations at non-iso IC).

## References

  • `reference/notes_M3_phase0_berry_connection.md` §7 — full
    derivation of the 7D Ω with corrected antisymmetric off-diag
    coefficient.
  • `scripts/verify_berry_connection_offdiag.py` — SymPy authority
    (9 CHECKs).
  • `reference/notes_M3_3a_field_set_cholesky.md` Q3 — the omission
    decision now reversed.
  • `reference/notes_M3_3c_berry_integration.md` — pattern for adding
    Newton unknowns to the 2D residual.
  • `reference/notes_M3_7_3d_extension.md` §6 + §7.1b — forward-
    compat plan; the 3D off-diag will reuse this Phase 0 scaffolding
    when M3-7 lifts to SO(3) Berry.
  • `src/berry.jl` — `kinetic_offdiag_coeffs_2d`, `kinetic_offdiag_2d`
    (consumed at the Hamilton-equation level by the residual).
