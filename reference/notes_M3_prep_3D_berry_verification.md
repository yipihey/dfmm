# M3-prep — 3D Berry-connection verification reproduction in Julia

Status: **complete**, all 8 SymPy CHECKs reproduced in Julia (pre-flight for M3-7).

This is a pre-flight task ahead of Milestone **M3-7 (3D extension)**. The
3D SO(3) Berry connection has been derived and verified symbolically via
SymPy; here we reproduce the same 8 CHECKs in Julia, asserting that the
M3-prep 3D stencils in `src/berry.jl` (`berry_F_3d`, `berry_term_3d`,
`berry_partials_3d`, `BerryStencil3D`) match the SymPy verification at
random sample points to round-off precision.

## The 3D SO(3) symplectic potential

```
Θ_rot^{(3D)} = (1/3) Σ_{a<b} (α_a³ β_b − α_b³ β_a) · dθ_{ab}
```

summed over (a,b) ∈ {(1,2), (1,3), (2,3)}, on the 9D phase space
(α_a, β_a)_{a=1,2,3} ∪ (θ_12, θ_13, θ_23).

## Reproduction results

All 8 SymPy CHECKs reproduced in
`test/test_M3_prep_3D_berry_verification.jl` (797 asserts total).

| CHECK | Description | Tolerance | Status |
|------:|:-------------|:---------|:-------|
| 1 | dΩ = 0 (84 cyclic triples in 9D, polynomial arithmetic) | `== 0.0` exact | pass |
| 2 | Per-axis Hamilton eqs on slice θ̇_ab = 0: α̇_a = β_a, β̇_a = (M_a − β_a²)/α_a | atol = 1e-13 | pass |
| 3a | Full iso pullback (α_a = α, β_a = β): Berry blocks vanish; ∂α-∂β coupling = 3α² | atol = 1e-14 (or `== 0.0`) | pass |
| 3b | 5×5 sub-block on (α_1, β_1, α_2, β_2, θ_12) matches 2D Ω | atol = 1e-14 | pass |
| 4 | rank(Ω) = 8, dim ker(Ω) = 1; closed-form kernel direction matches | atol/rtol = 1e-9 (SVD limited) | pass |
| 5 | F_{ab} antisymmetry under axis swap; per-pair 2D-sector reduction | atol = 1e-14 | pass |
| 6 | Degeneracy boundary α_a = α_b: F_{ab} = (α³/3)(β_b − β_a) | atol = 1e-14 (or `== 0.0`) | pass |
| 7 | Ω = dΘ exactness: ∂Θ/∂θ_{ab} = F_{ab} (FD vs closed form); polynomial → no monopole | atol = 1e-9 (FD) or 1e-14 (closed form) | pass |
| 8 | Cyclic-sum polynomial: F_12 + F_23 − F_13 = (1/3)[α_1³(β_2−β_3) + α_2³(β_3−β_1) + α_3³(β_1−β_2)] | atol = 1e-14 | pass |

### Tolerance commentary

Most asserts hit exact `0.0` or atol = 1e-14 (round-off) because the 3D
Berry stencils are polynomial in (α, β) and the Euler-angle increment
matrix `dθ`. Higher tolerances appear in:

- **CHECK 2** (1e-13): Hamilton's equation solve `Ω_6 \ (-dH_6)` goes
  through a 6×6 linear solve, accumulating round-off.
- **CHECK 4** (1e-9): kernel direction recovered from `svd(Matrix(Ω))`,
  so the closed-form match is at SVD-precision rather than exact. The
  closed form has compound numerator/denominator polynomials which can
  reach magnitudes ~10⁵ at random points in [-1, 1]^3 × [0.5, 1.5]^3,
  driving the relative tolerance into 1e-9 territory.
- **CHECK 7** finite-difference parts (1e-9): central FD with h = 1e-6
  has truncation error ~h² and round-off ~eps/h; both ~1e-12.

### File:line cross-references

- `test/test_M3_prep_3D_berry_verification.jl:191–207`  CHECK 1 (504 asserts)
- `test/test_M3_prep_3D_berry_verification.jl:209–242`  CHECK 2 (36 asserts)
- `test/test_M3_prep_3D_berry_verification.jl:249–280`  CHECK 3a (60 asserts)
- `test/test_M3_prep_3D_berry_verification.jl:282–319`  CHECK 3b (6 asserts)
- `test/test_M3_prep_3D_berry_verification.jl:321–391`  CHECK 4 (55 asserts)
- `test/test_M3_prep_3D_berry_verification.jl:393–445`  CHECK 5 (25 asserts)
- `test/test_M3_prep_3D_berry_verification.jl:447–491`  CHECK 6 (18 asserts)
- `test/test_M3_prep_3D_berry_verification.jl:497–565`  CHECK 7 (35 asserts)
- `test/test_M3_prep_3D_berry_verification.jl:567–593`  CHECK 8 (18 asserts)
- `test/test_M3_prep_3D_berry_verification.jl:599–639`  smoke (40 asserts)

Total: **797 asserts**, all passing.

### `src/berry.jl` discrepancies

**None.** The Julia implementation in `src/berry.jl` matches the SymPy
authority exactly (round-off only) at every sample point tested. No
edits to `src/berry.jl` were made or needed.

## M3-7 launch implications

All 8 CHECKs gate M3-7's 3D Cholesky-sector EL residual integration.
Specifically:

- **CHECK 2** is the "M3-7 reduces to M1's per-axis 1D box on the slice"
  acceptance criterion (analog of M3-3b's dimension-lift gate).
- **CHECK 3b** is the "2D residual is recovered exactly when one axis is
  frozen" gate (analog of M3-3c's 2D dimension-lift).
- **CHECK 4** (rank-8 + 1D kernel) tells M3-7 that the 9-row-per-cell
  Newton system has exactly one Casimir direction, so M3-7 will need a
  closed-form `h_rot_partial_dθ_3d` (the 3D analog of
  `h_rot_partial_dtheta` in `src/cholesky_DD.jl`) to ensure
  `dH · v_ker = 0` at every Newton iter.
- **CHECK 7** confirms that no Chern-class obstruction will arise — Ω is
  globally exact, so the 3D Berry term enters the discrete action as a
  globally well-defined boundary term.

### Pseudo-code for M3-7's 3D EL residual integration

```julia
# Per-cell residual rows (M3-7 will lift M3-3c's 9-row 2D pattern to 12 rows in 3D):
#   3 × x_a      (positions, like M3-3b)
#   3 × u_a      (velocities, like M3-3b)
#   3 × α_a      (Cholesky diagonal, like M3-3b)
#   3 × β_a      (per-axis L_2 diag, like M3-3b)
#   3 × θ_{ab}   (Berry/SO(3) Newton unknowns — NEW for M3-7)

# In each Newton iter, the per-cell α/β Berry contributions come from:
function add_berry_3d_to_residual!(R_α, R_β, R_θ, α, β, dθ, dt)
    s = BerryStencil3D(α, β)        # zero-alloc pre-compute
    (grad_αβ, F) = berry_partials_3d(α, β, dθ)
    # α-row contribution: ∂Θ/∂α_a
    @. R_α += grad_αβ[SVector(1, 2, 3)] * dt
    # β-row contribution: ∂Θ/∂β_a
    @. R_β += grad_αβ[SVector(4, 5, 6)] * dt
    # θ-row contribution (kinematic eq): F is the coefficient.
    R_θ[1] += F[1] * dt    # F_12 → θ_12 row
    R_θ[2] += F[2] * dt    # F_13 → θ_13 row
    R_θ[3] += F[3] * dt    # F_23 → θ_23 row
end
```

The kernel-orthogonality residual (M3-3c's `h_rot_kernel_orthogonality_residual`
analog) must be added to the θ_{23} row in 3D — this is the row where
the Casimir kernel direction has the largest (=1) component (CHECK 4).

## Bit-exact regression

The new test file is **append-only**: 797 new asserts are added on top
of the existing test suite. No existing tests modified. Full test suite
runs unchanged.

## References

- `scripts/verify_berry_connection_3D.py`  the SymPy authority.
- `reference/notes_M3_phase0_berry_connection_3D.md`  the 3D derivation.
- `reference/notes_M3_prep_berry_stencil.md`  the `src/berry.jl` API.
- `src/berry.jl`  the Julia stencils under test.
- `test/test_M3_prep_berry_stencil.jl`  existing 3D test patterns
  (FD-vs-closed-form, cyclic permutation, 2D reduction).
- `test/test_M3_3c_berry_residual.jl`  the 2D analog of what M3-7 will
  do for residual integration.
