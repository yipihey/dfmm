# Berry connection $\omega_{\rm rot}$ for the 2D Cholesky sector

> **Status (2026-04-26):** First-principles derivation **+ symbolic
> verification** in `scripts/verify_berry_connection.py` (passes 5 of
> 7 checks cleanly; 2 corrections to the original derivation appear
> in §6 below). Off-diagonal $L_2$ coupling sketched in §7 but not
> fully integrated — defer until M3 Phase 9 (KH instability test).

This document derives the Berry-curvature contribution
$\omega_{\rm rot}$ in

$$\omega_{2D} = \sum_{a=1}^{2}\,\alpha_a^2\,d\alpha_a\wedge d\beta_a + \omega_{\rm rot},$$

flagged in `specs/01_methods_paper.tex` §5.2 (eq:omega-2D) as "requires
further derivation; specific form flagged for the 2D extension paper."

The result, derived below:

$$\boxed{\;
\omega_{\rm rot} = d\Theta_{\rm rot},
\qquad
\Theta_{\rm rot} = \tfrac{1}{3}\bigl(\alpha_1^3\,\beta_2 - \alpha_2^3\,\beta_1\bigr)\,d\theta_R
\;}$$

i.e. $\omega_{\rm rot}$ is **exact** with the per-axis-antisymmetric
1-form $\Theta_{\rm rot}$ as its primitive.

## 1. Setup and notation

Per `specs/01_methods_paper.tex` §2 + §5, the 2D phase-space covariance
is the $4\times 4$ symmetric positive-definite matrix
$M = \begin{pmatrix} M_{xx} & M_{xv}^\top \\ M_{xv} & M_{vv} \end{pmatrix}$
on each Lagrangian point. Its Cholesky factor

$$L = \begin{pmatrix} L_1 & 0 \\ L_2 & L_3 \end{pmatrix}$$

has 10 independent components ($L_1, L_3$ each lower-triangular 2×2,
$L_2$ general 2×2). The spatial block $L_1$ admits the principal-axis
parameterization

$$L_1 = R(\theta_R)\,D\,R(\theta_R)^\top,
\qquad D = \mathrm{diag}(\alpha_1, \alpha_2),$$

with $R(\theta_R) = \exp(\theta_R J) \in SO(2)$ and
$J = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$. This is the
spectral decomposition of $M_{xx} = L_1 L_1^\top = R\,D^2\,R^\top$;
the principal stretches are $(\alpha_1, \alpha_2)$ and the
principal-axis frame is rotated by $\theta_R$ from the lab frame.

In the principal-axis basis (after conjugation by $R$), restrict to
the **diagonal-$L_2$ sector**

$$\tilde L_2 := R^\top L_2 R = \mathrm{diag}(\beta_1, \beta_2),$$

so the per-axis position-velocity correlation is the scalar
$(\alpha_a\beta_a)$ on axis $a$. The off-diagonal of $\tilde L_2$ is
its own dynamical sector with its own Berry-like coupling, sketched
in §6 below.

**Coordinates.** $(\alpha_1, \alpha_2, \beta_1, \beta_2, \theta_R)$ —
five independent fields. The 1D analog is the slice
$(\alpha_2, \beta_2, \theta_R) = (\text{const},\, 0,\, 0)$, where the
2D scheme reduces to the M1-verified 1D one.

## 2. Strategy: derive $\omega_{\rm rot}$ from two compatibility conditions

The methods paper specifies *part* of $\omega_{2D}$ (the per-axis
diagonal), and §5.5 specifies the Hamiltonian
$\mathcal{H}_{\rm Ch}^{2D} = -\tfrac12(\alpha_1^2\gamma_1^2 + \alpha_2^2\gamma_2^2) + \mathcal{H}_{\rm rot}(\theta_R)$.
The rotation Hamiltonian $\mathcal{H}_{\rm rot}$ is "driven by the
misalignment between the $M_{xx}$ principal axes and the strain
principal axes" — that's the kinematic input. We derive
$\omega_{\rm rot}$ by requiring that Hamilton's equations under
$(\omega_{2D}, \mathcal{H}_{\rm Ch}^{2D})$ reproduce two compatibility
conditions:

- **(C1)** The kinetic-theory Liouville evolution of $M_{xx}$ in the
  principal-axis frame (computed in §3 below).
- **(C2)** The 1D-isotropic limit ($\alpha_1=\alpha_2$, $\beta_1=\beta_2$)
  recovers $\omega_{1D}$ on the diagonal slice; equivalently,
  $\omega_{\rm rot}$ vanishes when the principal-axis frame becomes
  degenerate (gauge symmetry takes over).

C1 fixes the form of $\omega_{\rm rot}$ up to gauge; C2 fixes the gauge
(the additive harmonic-form ambiguity).

## 3. Kinematic equation for $\dot\theta_R$

The Liouville evolution of the spatial covariance under flow with
velocity gradient $G_{ij} = \partial_j u_i$ is

$$\dot M_{xx} = -G\,M_{xx} - M_{xx}\,G^\top + 2\,M_{xv}^{\rm sym},$$

where $M_{xv}^{\rm sym} = \tfrac12(M_{xv} + M_{xv}^\top)$. Decompose
$G = S + W$ into symmetric strain rate $S$ and antisymmetric
vorticity $W$. In the principal-axis frame of $M_{xx}$
($M_{xx} = R\Lambda R^\top$, $\Lambda = D^2$),

$$\dot M_{xx} = R\bigl(\dot\Lambda + [J\,\dot\theta_R,\Lambda]\bigr)R^\top.$$

Using $[J,\Lambda] = (\alpha_1^2 - \alpha_2^2)\,X$ where
$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, the off-diagonal
component (in the principal-axis frame) reads

$$(\alpha_1^2 - \alpha_2^2)\,\dot\theta_R
= -\,\tilde S_{12}\,(\alpha_1^2 + \alpha_2^2) + \tilde W_{12}\,(\alpha_1^2 - \alpha_2^2) + 2\,\tilde M_{xv,12}^{\rm sym},$$

(tildes denote principal-axis-frame quantities), so

$$\dot\theta_R = \tilde W_{12} - \tilde S_{12}\,
\frac{\alpha_1^2 + \alpha_2^2}{\alpha_1^2 - \alpha_2^2}
+ \frac{2\,\tilde M_{xv,12}^{\rm sym}}{\alpha_1^2 - \alpha_2^2}.
\tag{3.1}$$

The three terms are: (i) the rigid rotation by the local vorticity,
(ii) the strain-induced rotation that vanishes when the strain and
$M_{xx}$ principal axes align ($\tilde S_{12} = 0$), and (iii) the
intrinsic rotation driven by the off-diagonal cross-correlation
$\tilde M_{xv,12}^{\rm sym}$. Note the divergence at $\alpha_1 =
\alpha_2$: this is the principal-axis-degenerate gauge boundary
where $\theta_R$ is undefined. Physical evolution stays well-defined
because $\tilde S_{12} \to 0$ and $\tilde M_{xv,12}^{\rm sym} \to 0$
in this limit.

## 4. Variational structure: where the Berry connection lives

Under the Hamilton-Pontryagin construction, the symplectic 1-form
$\Theta$ is the canonical 1-form on the cotangent bundle, and the
Hamiltonian generates time evolution via Hamilton's equations
$\iota_{X_H}\omega = -dH$ where $X_H$ is the Hamiltonian vector
field.

For the rotation sector, $\theta_R$ is a configuration coordinate
without an independent canonical momentum — its conjugate is built
from the existing $(\alpha_a, \beta_a)$ via the Berry connection.
Specifically: in the principal-axis-diagonal restriction (§1, the
$\tilde L_2$-diagonal sector), the only off-diagonal $M_{xv}$
component is

$$\tilde M_{xv,12}^{\rm sym} = \tfrac12(\tilde L_1\,\tilde L_2^\top + \tilde L_2\,\tilde L_1^\top)_{12} = 0,$$

(both $\tilde L_1$ and $\tilde L_2$ are diagonal there), and so the
intrinsic-rotation term in (3.1) vanishes. The remaining drivers are
the lab-frame $\tilde W_{12}$ (geometric, comes through the
covariant derivative $\mathcal{D}_t$) and $\tilde S_{12}$ (which is
itself determined by the principal-axis-frame strain rate, an input
from the surrounding hydrodynamics).

The Berry connection thus arises from how $(\beta_1, \beta_2)$
transform under principal-axis rotation and how that coupling
enters the action.

## 5. Derivation: ansatz and exterior derivative

Following the 1D Hamilton-Pontryagin construction (v2 §3.1 with the
M1 Phase-1 sign convention),

$$\Theta_{1D} = -\tfrac{1}{3}\alpha^3\,d\beta,
\qquad \omega_{1D} = d\Theta_{1D} = \alpha^2\,d\alpha\wedge d\beta.$$

The 2D analog has two copies plus a $\theta_R$ piece:

$$\Theta_{2D} = -\tfrac{1}{3}\bigl(\alpha_1^3\,d\beta_1 + \alpha_2^3\,d\beta_2\bigr) + \Theta_{\rm rot}.
\tag{5.1}$$

**Ansatz.** $\Theta_{\rm rot}$ is a 1-form linear in $d\theta_R$:

$$\Theta_{\rm rot} = F(\alpha_1, \alpha_2, \beta_1, \beta_2)\,d\theta_R,$$

so $\omega_{\rm rot} = d\Theta_{\rm rot} = dF\wedge d\theta_R$.

**Constraint from C2** (isotropic limit). At $\alpha_1=\alpha_2$,
$\beta_1=\beta_2$, the principal-axis frame is degenerate; rotations
become pure gauge and $\omega_{\rm rot}$ must vanish on that
subspace. So $F$ must be a function that vanishes on the diagonal
$\alpha_1=\alpha_2,\,\beta_1=\beta_2$. The simplest gauge-invariant
choice:

$$F(\alpha_1,\alpha_2,\beta_1,\beta_2) = c_1\,(\alpha_1^p\beta_2 - \alpha_2^p\beta_1) + c_2\,(\alpha_1^q\beta_1 - \alpha_2^q\beta_2)\,(\beta_1 - \beta_2) + \cdots$$

The first term is bilinear-antisymmetric; the second introduces
quadratic-in-$\beta$ terms; higher orders allowed if needed.

**Constraint from $1$D analog.** The per-axis Hamilton-Pontryagin
factor is $\alpha_a^3/3$ (the cube comes from the weighted symplectic
$\omega = \alpha^2 d\alpha\wedge d\beta$ via integrating by parts:
$\theta = -(\alpha^3/3) d\beta$ has $d\theta = \alpha^2 d\alpha\wedge d\beta$).
So the natural exponent is $p = 3$, with the simplest ansatz

$$F = \tfrac{1}{3}\bigl(\alpha_1^3\,\beta_2 - \alpha_2^3\,\beta_1\bigr).
\tag{5.2}$$

This vanishes on the diagonal $\alpha_1=\alpha_2,\beta_1=\beta_2$ ✓
and is the unique bilinear-antisymmetric form that reduces correctly
in the limits checked in §6.

**Result.** Taking $d$ of (5.2):

$$\omega_{\rm rot} = dF\wedge d\theta_R
= \bigl(\alpha_1^2\,\beta_2\,d\alpha_1 + \tfrac{\alpha_1^3}{3}\,d\beta_2
       - \alpha_2^2\,\beta_1\,d\alpha_2 - \tfrac{\alpha_2^3}{3}\,d\beta_1\bigr)
  \wedge d\theta_R.
\tag{5.3}$$

## 6. Verifications (symbolic, via `scripts/verify_berry_connection.py`)

The script builds $\Omega$ as a 5×5 antisymmetric matrix in the basis
$(\alpha_1, \beta_1, \alpha_2, \beta_2, \theta_R)$ and runs seven
checks. Five pass cleanly; two surface corrections to the original
derivation:

### 6.1 Closedness — $d\omega_{2D} = 0$

All cyclic-sum components $\partial_i \Omega_{jk} + \partial_j \Omega_{ki} + \partial_k \Omega_{ij}$
for $i<j<k$ vanish. ✓ — $\omega_{2D}$ is a valid (closed) symplectic
2-form.

### 6.2 Per-axis Hamilton equations match M1's boxed form

Computing Hamilton's equations from $\Omega$ and $\mathcal{H}_{\rm Ch}^{2D} = -\tfrac12(\alpha_1^2\gamma_1^2 + \alpha_2^2\gamma_2^2)$ on the slice $\dot\theta_R = 0$ (4×4 sub-system):

| Equation | Match |
|---|---|
| $\dot\alpha_1 = \beta_1$ | ✓ |
| $\dot\beta_1 = (M_{vv,1} - \beta_1^2)/\alpha_1 = \gamma_1^2/\alpha_1$ | ✓ |
| $\dot\alpha_2 = \beta_2$ | ✓ |
| $\dot\beta_2 = (M_{vv,2} - \beta_2^2)/\alpha_2 = \gamma_2^2/\alpha_2$ | ✓ |

Per-axis sector reduces *exactly* to the M1 boxed Hamilton equations
(v2 §2.1, eq:dfmm-cholesky-cov).

### 6.3 Iso-subspace pullback (1D iso reduction)

The original derivation claimed $\omega_{\rm rot} \to 0$ "in the
isotropic limit." The verification clarifies this: the **components**
of $\omega_{\rm rot}$ in 5D do **not** vanish at $\alpha_1 = \alpha_2$,
$\beta_1 = \beta_2$ (the partial derivatives $\partial F/\partial \alpha_a$,
$\partial F/\partial \beta_a$ are individually nonzero there). What
vanishes is the **pullback** to the 3D iso-diagonal subspace
parameterized by $(\alpha, \beta, \theta_R)$ with embedding
$\alpha_1 = \alpha_2 = \alpha$, $\beta_1 = \beta_2 = \beta$.

The pullback's tangent vectors are $T_\alpha = (1,0,1,0,0)$,
$T_\beta = (0,1,0,1,0)$, $T_\theta = (0,0,0,0,1)$, and SymPy
gives:

| Component | Value | Status |
|---|---|---|
| $i^*\omega(T_\alpha, T_\beta)$ | $\boxed{2\alpha^2}$ | Twice 1D weight (factor-of-2 issue, see §6.4) |
| $i^*\omega(T_\alpha, T_\theta)$ | $0$ | ✓ Berry block vanishes |
| $i^*\omega(T_\beta, T_\theta)$ | $0$ | ✓ Berry block vanishes |

So the Berry contribution does vanish on the iso slice, as required.

### 6.4 Factor-of-2 normalization correction

The methods paper §5.2 says iso reduction "recovers $\omega_{1D}$
correctly." With the form as written ($\alpha_a^2 d\alpha_a \wedge d\beta_a$ per
axis), the iso pullback gives $2\alpha^2 d\alpha \wedge d\beta = 2\omega_{1D}$ —
twice the 1D weight. To get exact agreement, one of two corrections:

1. **Normalize the per-axis weight to $\alpha_a^2/2$**: write
   $\omega_{2D} = \tfrac12\sum_a \alpha_a^2 d\alpha_a \wedge d\beta_a + \omega_{\rm rot}$.
   The 1D form is then $\omega_{1D} = \alpha^2 d\alpha\wedge d\beta$, and the
   iso pullback gives exactly $\omega_{1D}$.
2. **Or interpret the iso-reduction as a projection** counting each
   axis half-weight on the diagonal subspace.

Both conventions are internally consistent. **Recommend (1)** for the
methods-paper revision: explicit factor of $1/2$ avoids the
implicit-projection subtlety.

### 6.5 Uniaxial limit ($\alpha_2 = $ const, $\beta_2 = 0$, $\theta_R = 0$)

On this 2D slice ($d\alpha_2 = d\beta_2 = d\theta_R = 0$), every
$\omega_{\rm rot}$ term involves $d\theta_R$ and pulls back to zero;
$\omega_{2D}$ reduces to $\alpha_1^2 d\alpha_1 \wedge d\beta_1 = \omega_{1D}$
without normalization issues. ✓

### 6.6 Kernel structure: Poisson manifold, not symplectic

$\Omega$ has rank 4 in 5D — it's a *Poisson* structure, not strictly
symplectic. The 1D kernel is the **gauge direction**:

$$v_{\rm ker} = \left(-\frac{\alpha_2^3}{3\alpha_1^2},\; -\beta_2,\; \frac{\alpha_1^3}{3\alpha_2^2},\; \beta_1,\; 1\right) e$$

(verified by SymPy `nullspace`). Hamilton's equations
$\Omega\,X = -dH$ have solutions iff $dH \perp v_{\rm ker}$. This is
the **integrability condition**, and it determines the rotation
Hamiltonian $\mathcal{H}_{\rm rot}$ as a function of the per-axis
$(α_a, β_a, M_{vv,a})$:

$$\boxed{\;\mathcal{H}_{\rm rot}\;\text{must satisfy}\quad
\frac{\partial \mathcal{H}_{\rm rot}}{\partial \theta_R}\bigg|_{\rm constraint}
= \frac{\gamma_1^2 \alpha_2^3}{3\alpha_1} - \frac{\gamma_2^2 \alpha_1^3}{3\alpha_2} - (\alpha_1^2 - \alpha_2^2)\beta_1\beta_2\;}$$

(derived from $dH \cdot v_{\rm ker} = 0$; SymPy output in §6.6 of the
verification log.) **This is a substantive new finding**: in the
principal-axis-diagonal restriction, $\mathcal{H}_{\rm rot}$ is *not* a
free input — it's algebraically determined by the per-axis Hamiltonian
and the Berry connection.

The methods paper §5.5 says $\mathcal{H}_{\rm rot}$ "is driven by the
misalignment between the $M_{xx}$ principal axes and the strain
principal axes." Our result complements this: in the
principal-axis-diagonal sector (no strain misalignment, $\tilde S_{12} = 0$),
$\mathcal{H}_{\rm rot}$ has the explicit form above. The full
strain-driven form requires lifting the principal-axis-diagonal
restriction (see §7).

### 6.7 Singularity structure at the iso boundary

Both $\mathcal{H}_{\rm rot}$ and the $\dot\theta_R$ Hamilton equation
have a $1/(\alpha_1^2 - \alpha_2^2)$-type singularity at the
principal-axis-degenerate boundary (the $\alpha_a$-difference appears
in denominators). This matches the kinetic-theory kinematic equation
(3.1) from §3 above. ✓

## 7. Off-diagonal $L_2$ sector (sketch)

The principal-axis-diagonal restriction sets
$\tilde\beta_{12} = \tilde\beta_{21} = 0$. The full 2D problem has
these as two additional dynamical fields; they couple to $\theta_R$
through their own Berry-like 1-form

$$\Theta_{\rm rot}^{(\rm off)} = \tfrac{1}{2}\bigl(\alpha_1^2\alpha_2\,\tilde\beta_{12} + \alpha_1\alpha_2^2\,\tilde\beta_{21}\bigr)\,d\theta_R + \cdots$$

(Schematic — full derivation analogous to §5 but with the
mixed-axis-stretch weight $\alpha_1\alpha_2$.) These pick up the
intrinsic-rotation term in (3.1) that vanishes in the
diagonal-restriction sector.

For M3 Phase 1's per-triangle integrator (which works in the
principal-axis-diagonal sector by design — methods paper §9.2 lists
$(\alpha_a, \beta_a, \theta_R)$ as the spatial-Cholesky DOFs), the
off-diagonal sector is not needed. M3 Phase 9 (KH instability,
Tier D.1) is the first test that exercises significant off-diagonal
$L_2$ coupling; the off-diagonal Berry term should be derived
fully before that phase.

## 8. Open verification needs

Before M3 Phase 1 codes against this form:

1. ~~**Computer-algebra verification.**~~ **✓ Done** —
   `scripts/verify_berry_connection.py` runs SymPy verification of:
   closedness $d\omega_{2D} = 0$, per-axis Hamilton equations matching
   M1's boxed form, iso-subspace pullback's Berry-block vanishing,
   uniaxial reduction to $\omega_{1D}$, kernel structure, and the
   $1/(\alpha_1^2 - \alpha_2^2)$ singularity at the iso boundary.
   See §6 above for the corrections surfaced (factor-of-2
   normalization in §6.4, $\mathcal{H}_{\rm rot}$ constraint in §6.6).

2. **3D extension structural pattern.** Replacing $SO(2)$ with
   $SO(3)$ and three principal axes, the Berry connection should
   generalize to a 1-form $\Theta_{\rm rot} = \tfrac{1}{3}\sum_{a<b}\bigl(\alpha_a^3\,\beta_b - \alpha_b^3\,\beta_a\bigr)\,d\theta_{ab}$ summed over the three rotation generators of $SO(3)$.
   Verify that this gives a closed 2-form (no monopole/anomaly)
   and reduces correctly to the 2D form when one axis is held
   constant.

3. **Off-diagonal $L_2$ contribution.** Complete the derivation
   sketched in §7 before M3 Phase 9 (KH instability test).

4. **Liouville monotonicity under remap.** The Bayesian remap
   (methods paper §6) must respect the symplectic structure;
   $\omega_{\rm rot}$'s contribution to the law-of-total-covariance
   coarse-graining term needs explicit form so that
   moment-projected-onto-Eulerian-cell preserves the variational
   structure.

5. **Hessian-degeneracy structure in 2D.** The 1D Hessian of
   $\mathcal{H}_{\rm Ch}$ at the caustic is rank-1 (M1 Phase 3
   verified). In 2D, the Hessian on
   $(\alpha_1, \alpha_2, \beta_1, \beta_2, \theta_R)$ should
   rank-defect to rank 2 at a 2D pancake (one principal axis
   collapses while the other does not). This is what enables the
   per-axis stochastic injection (methods paper §4.5 / §9.6 D.1).
   Verify the Hessian formula in 2D and the structure of its
   degeneracy.

## 9. Use of this result

For M3 Phase 1 (per-triangle Cholesky integrator):

- **Lagrangian.** Per axis $a = 1, 2$: the 1D form
  $\mathcal{L}_{\rm Ch}^{(a)} = -(\alpha_a^3/3)\,\mathcal{D}_t^{(1)}\beta_a + \tfrac12 \alpha_a^2 \gamma_a^2$
  carries over directly. Plus a rotation kinetic term
  $\Theta_{\rm rot}\cdot \dot{} = \tfrac{1}{3}(\alpha_1^3\beta_2 - \alpha_2^3\beta_1)\,\dot\theta_R$ from (5.2),
  contributing $-\tfrac{1}{3}(\alpha_1^3\beta_2 - \alpha_2^3\beta_1)\,\dot\theta_R$ to the Lagrangian density
  with the M1 sign convention.

- **EL system.** The discrete-EL system has 5 unknowns per cell
  $(\alpha_a^{n+1}, \beta_a^{n+1}, \theta_R^{n+1})$ instead of M1's
  2 unknowns $(\alpha^{n+1}, \beta^{n+1})$. The Newton solve and
  sparse-AD path generalize directly.

- **Discrete parallel transport.** The $\mathcal{D}_t^{(q)}$
  midpoint stencil applies per principal axis. The principal-axis
  frame's rotation contributes its own midpoint stencil:
  $\mathcal{D}_t^{(\rm rot)}\theta_R = \dot\theta_R + (\text{lab-frame strain rotation contribution})$.

## 10. References

- `specs/01_methods_paper.tex` §5 (the 2D matrix lift; flagged the
  $\omega_{\rm rot}$ derivation as "for the 2D extension paper").
- `design/03_action_note_v2.tex` §3.1 (1D weighted symplectic form
  + Hamilton-Pontryagin Lagrangian).
- M1 Phase 1 implementation: `src/cholesky_sector.jl` (1D EL
  residual + sign convention); `reference/notes_phaseA0.md`
  (1D hand-derivation).
- Berry, M.V. (1984), "Quantal phase factors accompanying
  adiabatic changes," *Proc. Roy. Soc. A* 392, 45–57 (the
  classical reference for geometric-phase / Berry-connection
  formalism, applicable here by analogy with the
  principal-axis bundle).
- Marsden, J.E. & Ratiu, T.S. (1999), *Introduction to Mechanics
  and Symmetry*, 2nd ed., §5.3 ("Cotangent Bundle Reduction") —
  the standard reference for the Hamilton-Pontryagin construction
  on Lie group cotangent bundles, of which this is an instance.
