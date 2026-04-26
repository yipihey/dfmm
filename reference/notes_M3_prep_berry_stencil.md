# M3-prep — Berry-connection stencil pre-compute (`src/berry.jl`)

Tier-1 pre-compute kernels for the upcoming **M3-3** (2D Cholesky +
Berry connection) phase. Pure-functional building blocks; no solver
wiring yet — M3-3 will consume these.

## Verified symbolic forms (from §4 / §7 / §8 of the 2D doc and the 3D doc)

**2D principal-axis Berry 1-form**

$$
\Theta_{\rm rot}^{(2D)} \;=\; \tfrac{1}{3}\!\left(\alpha_1^{3}\beta_2 - \alpha_2^{3}\beta_1\right)\,d\theta_R .
$$

**3D SO(3) Berry 1-form** (the three pair-rotation generators)

$$
\Theta_{\rm rot}^{(3D)} \;=\; \tfrac{1}{3}\!\sum_{a<b}\!\left(\alpha_a^{3}\beta_b - \alpha_b^{3}\beta_a\right)\,d\theta_{ab},\qquad
(a,b)\in\{(1,2),(1,3),(2,3)\}.
$$

**Off-diagonal-$L_2$ kinetic 1-form** (corrected per agent B in M3-prep)

$$
\theta_{\rm offdiag} \;=\; -\tfrac{1}{2}\!\left(\alpha_1^{2}\alpha_2\,d\beta_{21} + \alpha_1\alpha_2^{2}\,d\beta_{12}\right).
$$

These are checked symbolically by
`scripts/verify_berry_connection.py`,
`..._3D.py`, and
`..._offdiag.py` (closedness $d\omega = 0$, axis-swap antisymmetry,
1D and iso reductions, kernel structure of $\Omega$).

## Module API (re-exported from `dfmm`)

### 2D

```julia
berry_F_2d(α::SVector{2,T}, β::SVector{2,T}) -> T
    # Berry function F = (1/3)(α_1³ β_2 − α_2³ β_1).

berry_term_2d(α::SVector{2,T}, β::SVector{2,T}, dθ_R::Real) -> T
    # Θ_rot^(2D) evaluated at the rotation increment dθ_R, i.e. F · dθ_R.

berry_partials_2d(α, β, dθ_R) -> SVector{5,T}
    # (∂Θ/∂α_1, ∂Θ/∂α_2, ∂Θ/∂β_1, ∂Θ/∂β_2, ∂Θ/∂θ_R).
    # Closed-form:
    #   ∂Θ/∂α_1 =  α_1² β_2 · dθ_R
    #   ∂Θ/∂α_2 = -α_2² β_1 · dθ_R
    #   ∂Θ/∂β_1 = -α_2³/3 · dθ_R
    #   ∂Θ/∂β_2 =  α_1³/3 · dθ_R
    #   ∂Θ/∂θ_R = F (the coefficient of dθ_R itself).

struct BerryStencil2D{T}; F; dF_dα::SVector{2,T}; dF_dβ::SVector{2,T}; end
BerryStencil2D(α, β) -> BerryStencil2D
dfmm.apply(stencil::BerryStencil2D, dθ_R) -> T   # = F · dθ_R
```

### 3D

```julia
berry_F_3d(α::SVector{3,T}, β::SVector{3,T}) -> SVector{3,T}
    # (F_12, F_13, F_23) with F_{ab} = (1/3)(α_a³ β_b − α_b³ β_a).

berry_term_3d(α, β, dθ::SMatrix{3,3,T,9}) -> T
    # Θ_rot^(3D) = Σ_{a<b} F_{ab} · dθ_{ab} (only the strict-upper
    # triangle of dθ is read).

berry_partials_3d(α, β, dθ) -> (grad_αβ::SVector{6,T}, F::SVector{3,T})
    # grad_αβ = (∂Θ/∂α_1, ∂α_2, ∂α_3, ∂β_1, ∂β_2, ∂β_3).
    # Coefficients of the dθ_{ab} partials are F_{ab}.

struct BerryStencil3D{T}
    F::SVector{3,T}
    dF_dα::SMatrix{3,3,T,9}   # dF_dα[pair, axis], pair ∈ {1=12, 2=13, 3=23}
    dF_dβ::SMatrix{3,3,T,9}
end
BerryStencil3D(α, β) -> BerryStencil3D
dfmm.apply(stencil::BerryStencil3D, dθ::SMatrix{3,3,...}) -> T
```

### Off-diagonal-$L_2$ kinetic 1-form

```julia
kinetic_offdiag_coeffs_2d(α::SVector{2,T}) -> SVector{2,T}
    # (c_β12, c_β21) = (-(α_1 α_2²)/2, -(α_1² α_2)/2).
    # These are the coefficients of dβ_12 and dβ_21 in θ_offdiag.

kinetic_offdiag_2d(α::SVector{2,T}, β::SMatrix{2,2,T,4}) -> T
    # Bilinear evaluation: θ_offdiag(α) · β
    #   = -(1/2)(α_1² α_2 · β[2,1] + α_1 α_2² · β[1,2]).
    # The diagonal entries β[1,1], β[2,2] are ignored (those live in the
    # diagonal kinetic 1-form -(α_a³/3) dβ_a).
```

## Properties verified by `test/test_M3_prep_berry_stencil.jl` (181 asserts)

1. **Explicit verification points** — bit-equal reproduction of the
   SymPy-computed values at five 2D points and three 3D points
   (including iso-limits, generic, and axis-swapped pairs).
2. **Closed-form partials** match the symbolic derivation against
   second-order central finite differences (`h = 1e-6`, rel-error
   `≤ 1e-7`) for both 2D and 3D.
3. **Axis-swap antisymmetry**: `F_{ab}(swap)` = `-F_{ab}` for all
   three pairs in 3D, and the 2D analog.
4. **Cyclic-permutation behavior** of the three 3D pair-functions
   under $\sigma:(1\to 2\to 3\to 1)$: $F'_{12}=-F_{13}$,
   $F'_{13}=-F_{23}$, $F'_{23}=F_{12}$.
5. **Iso reduction**: at $\alpha_1=\alpha_2$ (resp. $\alpha_a=\alpha_b$
   in 3D) with $\beta_a=\beta_b$, all $F$ vanish. This matches the
   $\varepsilon$-derivation of §6 of the 2D doc.
6. **2D ↔ 3D reduction**: at $\alpha_3$ const, $\beta_3=0$,
   $d\theta_{13}=d\theta_{23}=0$, the 3D Berry term reduces exactly
   to the 2D Berry term (CHECK 3b of `verify_berry_connection_3D.py`).
7. **Off-diagonal coefficients** match the §7 corrected derivation:
   $-(1/2)\alpha_1\alpha_2^{2}$ for $d\beta_{12}$ and
   $-(1/2)\alpha_1^{2}\alpha_2$ for $d\beta_{21}$.
8. **Allocation count = 0** on the hot-path call
   `berry_term_2d(::SVector{2}, ::SVector{2}, ::Real)` and the 3D /
   stencil-construction analogs (verified via `@allocated` after
   warm-up). M3-3 will call these once per cell per Newton iteration.

## What this module does NOT do

- It does not solve Hamilton's equations on the 5D / 7D / 9D Poisson
  manifold; the kernel-orthogonality solvability constraint
  (CHECK 6 / 7 of the SymPy scripts) is the M3-3 solver's job.
- It does not wire any term into the discrete EL residual or the
  Newton step. M3-3 will consume the `BerryStencil2D` / `…3D` carriers
  inside `det_el_residual_HG_2D` (or its 3D analog) once that kernel
  is written.
- The principal-axis-rotation angle $\theta_R$ (and the three
  $\theta_{ab}$ in 3D) is a per-step quantity computed from
  consecutive $L_2$-tensor diagonalizations; this is also M3-3's job.
  The stencils take $d\theta$ as a callee-supplied scalar (2D) or
  antisymmetric matrix (3D) so the call site is decoupled from the
  diagonalization detail.

## References

- `reference/notes_M3_phase0_berry_connection.md` (2D + off-diagonal §7)
- `reference/notes_M3_phase0_berry_connection_3D.md` (SO(3) extension)
- `scripts/verify_berry_connection.py` / `…_3D.py` / `…_offdiag.py`
  (SymPy verifications)
- `src/berry.jl` (this module)
- `test/test_M3_prep_berry_stencil.jl` (181 asserts cross-checked
  against the SymPy reference values)
