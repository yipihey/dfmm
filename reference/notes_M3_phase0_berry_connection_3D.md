# Berry connection $\omega_{\rm rot}^{3D}$ for the 3D Cholesky sector

> **Status (2026-04-25):** First-principles derivation **+ symbolic
> verification** in `scripts/verify_berry_connection_3D.py` (passes all
> 8 structural checks). This is the SO(3) generalization of the 2D-SO(2)
> Berry connection in `notes_M3_phase0_berry_connection.md`. Required
> for M3 Phase M3-7 (3D extension).

This document derives the SO(3) generalization of the rotation-Berry
contribution

$$\omega_{3D} = \sum_{a=1}^{3}\,\alpha_a^2\,d\alpha_a\wedge d\beta_a + \omega_{\rm rot}^{3D}$$

flagged in `specs/01_methods_paper.tex` §5.4 ("3D extension is
straightforward; replace SO(2) with SO(3)") and as Open Question 3 of
`reference/MILESTONE_3_PLAN.md` ("3D Berry connection. The proposed
SO(3) form needs a verification analogous to the 2D one. Run before
M3-7.").

The result, derived below:

$$\boxed{\;
\Theta_{\rm rot}^{3D} = \tfrac{1}{3}\sum_{a<b}\bigl(\alpha_a^3\,\beta_b - \alpha_b^3\,\beta_a\bigr)\,d\theta_{ab},
\qquad
\omega_{\rm rot}^{3D} = d\Theta_{\rm rot}^{3D}
\;}$$

i.e. one bilinear-antisymmetric Berry block per pair-rotation generator
$\theta_{ab}$ for $(a,b)\in\{(1,2),(1,3),(2,3)\}$. The 2D form is the
restriction to a single pair.

## 1. Setup

The 3D phase-space covariance is the $6\times 6$ symmetric SPD matrix
$M$ on each Lagrangian point. Its Cholesky factor

$$L = \begin{pmatrix} L_1 & 0 \\ L_2 & L_3 \end{pmatrix}$$

has $L_1, L_3$ each lower-triangular $3\times 3$, $L_2$ general $3\times 3$.
The spatial block $L_1$ admits the principal-axis parameterization

$$L_1 = R\,D\,R^\top,
\qquad D = \mathrm{diag}(\alpha_1, \alpha_2, \alpha_3),$$

with $R \in SO(3)$ parameterized by three rotation angles
$(\theta_{12}, \theta_{13}, \theta_{23})$ — one per pair-rotation
generator $J_{ab}$ (the $3\times 3$ generator of rotation in the
$(e_a, e_b)$ plane).

In the principal-axis basis, restrict to the **diagonal-$\tilde L_2$
sector**

$$\tilde L_2 = R^\top L_2 R = \mathrm{diag}(\beta_1, \beta_2, \beta_3),$$

so the per-axis position-velocity correlation is the scalar $\alpha_a\beta_a$
on axis $a$.

**Coordinates.** $(\alpha_1, \alpha_2, \alpha_3, \beta_1, \beta_2, \beta_3, \theta_{12}, \theta_{13}, \theta_{23})$
— nine independent fields. The 2D analog is the slice
$(\alpha_3, \beta_3, \theta_{13}, \theta_{23}) = (\text{const}, 0, 0, 0)$,
where the 3D scheme reduces to the 2D Berry derivation in
`notes_M3_phase0_berry_connection.md`.

**Reference for shared structure.** §1–§4 of the 2D doc apply per
pair-generator $\theta_{ab}$; we don't repeat them here. The new
content is:
- Three rotation generators acting independently in the symplectic
  structure (whereas SO(2) has one).
- The Liouville kinematic equation has three components — one per
  pair-rotation — each with its own $1/(\alpha_a^2 - \alpha_b^2)$
  singularity at the corresponding axis-degeneracy boundary.
- A potential Chern-class obstruction (does NOT arise; see §5).

## 2. Strategy

Apply the same compatibility logic as the 2D doc, per pair-generator:

- **(C1)** The Liouville kinematic for the $(a,b)$ pair-rotation gives
  $\dot\theta_{ab}$ in terms of strain rate, vorticity, and intrinsic
  rotation — analog of (3.1) of the 2D doc, restricted to the $(a,b)$
  off-diagonal of the $3\times 3$ kinematic equation. The structure is
  identical (per pair).

- **(C2)** Each pair $(a,b)$ has its own iso-degenerate boundary
  $\alpha_a = \alpha_b$ where the Berry block must vanish on the
  corresponding sub-slice (gauge symmetry takes over there).

C1 fixes the form of $\Theta_{\rm rot}^{3D}$ up to gauge per pair; C2
fixes the gauge.

## 3. Derivation: per-pair ansatz

Per 2D doc §5, the ansatz on each pair is bilinear-antisymmetric
$F_{ab} = c\,(\alpha_a^p\beta_b - \alpha_b^p\beta_a)$ with $p=3$ from
1D analogy and $c=1/3$ from the integral $\int\alpha^2\,d\alpha = \alpha^3/3$.
So per pair:

$$F_{ab}(\alpha_a, \alpha_b, \beta_a, \beta_b) = \tfrac{1}{3}(\alpha_a^3\beta_b - \alpha_b^3\beta_a),$$

and the 3D Berry function is the sum over all three pairs:

$$\Theta_{\rm rot}^{3D} = \sum_{a<b} F_{ab}\,d\theta_{ab}
= \tfrac{1}{3}\bigl[(\alpha_1^3\beta_2 - \alpha_2^3\beta_1)\,d\theta_{12}
+ (\alpha_1^3\beta_3 - \alpha_3^3\beta_1)\,d\theta_{13}
+ (\alpha_2^3\beta_3 - \alpha_3^3\beta_2)\,d\theta_{23}\bigr].$$

The 2-form is

$$\omega_{\rm rot}^{3D} = d\Theta_{\rm rot}^{3D} = \sum_{a<b} dF_{ab}\wedge d\theta_{ab}.$$

**Important symmetric structure.** Each $F_{ab}$ involves only the
pair $(\alpha_a, \alpha_b, \beta_a, \beta_b)$ — there are no
cross-pair terms (e.g. no $\alpha_1^2\alpha_2\beta_3$). This is
because each pair-rotation generator $J_{ab}$ commutes with the
$\alpha_c, \beta_c$ for $c\not\in\{a,b\}$ (the "third axis" is fixed
under the $(a,b)$-rotation). So the per-pair ansatz adds independently
across the three SO(3) generators.

## 4. Verification

Implemented in `scripts/verify_berry_connection_3D.py` and run as
`py-1d/.venv/bin/python scripts/verify_berry_connection_3D.py`. All
8 checks pass.

### 4.1 Closedness $d\omega_{3D} = 0$

For a closed 2-form $\omega = (1/2)\sum \Omega_{ij}\,dx^i\wedge dx^j$,
the cyclic sum
$\partial_i \Omega_{jk} + \partial_j \Omega_{ki} + \partial_k \Omega_{ij}$
vanishes for every $(i, j, k)$ ordered triple. With $n=9$,
there are $\binom{9}{3} = 84$ such triples; SymPy verifies all 84 vanish. ✓

### 4.2 Per-axis Hamilton equations

On the slice $\dot\theta_{ab} = 0\;\forall(a,b)$, the $6\times 6$
sub-system $(\alpha_a, \beta_a)_{a=1,2,3}$ decouples, and Hamilton's
equations under
$\mathcal{H}_{\rm Ch}^{3D} = -\tfrac12\sum_a \alpha_a^2(M_{vv,a} - \beta_a^2)$
match the M1 boxed form per axis:

| Axis $a$ | Hamilton equations |
|---|---|
| 1 | $\dot\alpha_1 = \beta_1$, $\dot\beta_1 = (M_{vv,1} - \beta_1^2)/\alpha_1$ |
| 2 | $\dot\alpha_2 = \beta_2$, $\dot\beta_2 = (M_{vv,2} - \beta_2^2)/\alpha_2$ |
| 3 | $\dot\alpha_3 = \beta_3$, $\dot\beta_3 = (M_{vv,3} - \beta_3^2)/\alpha_3$ |

All four equations per axis match exactly. ✓

### 4.3a Full iso reduction

At $\alpha_1 = \alpha_2 = \alpha_3 = \alpha$, $\beta_1 = \beta_2 = \beta_3 = \beta$:
- $F_{12}, F_{13}, F_{23}$ all vanish identically.
- Pullback to the iso-diagonal subspace
  $(\alpha, \beta, \theta_{12}, \theta_{13}, \theta_{23})$ has all
  Berry-block components zero (gauge symmetry takes over).
- The $(\partial_\alpha, \partial_\beta)$ pullback is $3\alpha^2$ — three
  times the 1D weight $\alpha^2$, matching the factor-of-2 pattern of the
  2D iso reduction (§6.4 of the 2D doc) generalized to 3D.

### 4.3b 2D reduction

On the slice $\alpha_3 = $ const, $\beta_3 = 0$,
$\theta_{13} = \theta_{23} = 0$ (with $d\alpha_3 = d\beta_3 = d\theta_{13} = d\theta_{23} = 0$),
the 9D form pulls back to the 5D $(\alpha_1, \beta_1, \alpha_2, \beta_2, \theta_{12})$
subspace and reproduces **exactly** the 2D Omega from
`scripts/verify_berry_connection.py`. ✓

### 4.4 Kernel structure

In 9D (odd-dim Poisson), $\omega_{3D}$ has rank 8 and a 1D Casimir
kernel. SymPy `nullspace` gives the kernel direction explicitly (large
expression printed in the verification script output).

The kernel direction has nonzero components in $(\alpha_a, \beta_a, \theta_{ab})$
generically, with $\theta_{23}$-component normalized to 1 in the
default SymPy output.

### 4.5 Per-pair antisymmetry

Each $F_{ab}$ is antisymmetric under the corresponding axis swap
$(a\leftrightarrow b)$:
- $F_{12}$ under $(1\leftrightarrow 2)$,
- $F_{13}$ under $(1\leftrightarrow 3)$,
- $F_{23}$ under $(2\leftrightarrow 3)$. ✓

### 4.6 Singularity at $\alpha_a = \alpha_b$ boundaries

The Berry function $F_{ab}$ itself is regular at $\alpha_a = \alpha_b$:
$F_{ab}|_{\alpha_a = \alpha_b} = \tfrac{1}{3}\alpha_a^3(\beta_b - \beta_a)$,
which vanishes only when $\beta_a = \beta_b$ as well (the higher-codimension
iso boundary). The singularity that matters is in the inverse symplectic
structure, where $1/(\alpha_a^2 - \alpha_b^2)$ appears — same structure as
the 2D doc §6.7. Each of the three pair-degeneracy boundaries is a
gauge-degeneracy direction analogous to the 2D one. ✓

### 4.7 No Chern-class obstruction (no monopole)

By construction $\omega_{3D}$ is exact ($\omega = d\Theta$) with $\Theta$
a globally-defined 1-form on the principal-axis bundle (away from the
codimension-2 axis-degeneracy boundary). Each $F_{ab}$ is a polynomial
in $(\alpha, \beta)$, so there's no patch-gluing required.

**Topological caveat (informational, not a derivation gap).** SO(3)
has $\pi_1(SO(3)) = \mathbb{Z}/2$, so the principal-axis bundle has a
double cover by $\mathrm{Spin}(3) = SU(2)$. The Berry connection
$\Theta_{\rm rot}^{3D}$ is invariant under the $\mathbb{Z}/2$ covering
action because $F_{ab}$ has no spinor-valued fields — the Berry
formula only sees the SO(3) representation, which is well-defined.
For the dfmm-2D project, this means the 3D extension does NOT require
any non-trivial monopole/anomaly handling.

This is in contrast to the standard angular-momentum Berry curvature
on $S^2$, which DOES carry a non-trivial Chern class (the
"monopole-at-the-band-touching-point" of condensed-matter topology).
The dfmm-2D Berry curvature is fundamentally classical (kinetic-theory
Liouville structure) rather than quantum (parameter-space adiabatic
holonomy of spinor states), so the Dirac-monopole structure does not
arise. ✓

### 4.8 Bianchi-like identity

The three $F_{ab}$'s do not satisfy a simple linear cyclic identity
(they're linearly independent polynomials in $\alpha, \beta$). The
SO(3) commutator structure $[J_{ab}, J_{cd}] = $ (cyclic) is encoded
not in a linear relation among $F_{ab}$ but in the closedness $d^2 = 0$
checked in CHECK 1. ✓

## 5. Use of this result for M3 Phase M3-7

For M3 Phase M3-7 (3D mesh + 3D Cholesky integrator):

- **Lagrangian.** Per axis $a = 1, 2, 3$: the 1D form
  $\mathcal{L}_{\rm Ch}^{(a)} = -(\alpha_a^3/3)\,\mathcal{D}_t^{(1)}\beta_a + \tfrac12 \alpha_a^2 \gamma_a^2$
  carries over directly. Plus three rotation kinetic terms
  $F_{ab}\,\dot\theta_{ab}$ contributing to the action density.

- **EL system.** The discrete-EL system has 9 unknowns per cell:
  3 × $(\alpha_a^{n+1}, \beta_a^{n+1})$ + 3 × $\theta_{ab}^{n+1}$.
  The Newton solve and sparse-AD path generalize directly from M1's 2
  unknowns and M3-1's 5 unknowns.

- **Discrete parallel transport.** $\mathcal{D}_t^{(q)}$ midpoint
  stencil applies per principal axis. Each pair-rotation
  $\theta_{ab}$ contributes its own midpoint stencil:
  $\mathcal{D}_t^{(\rm rot,ab)}\theta_{ab} = \dot\theta_{ab} + (\text{lab-frame strain rotation contribution per pair})$.

- **Singularity handling.** Each of the three $\alpha_a = \alpha_b$
  boundaries needs the gauge-invariant projection (analog of the 2D
  uniaxial degeneracy handling).

## 6. Open issues and follow-ups

- **Off-diagonal 3D $\tilde L_2$.** The 3D analog of the 2D off-diagonal
  $L_2$ sector has six off-diagonal entries
  $(\tilde\beta_{12}, \tilde\beta_{13}, \tilde\beta_{23}, \tilde\beta_{21}, \tilde\beta_{31}, \tilde\beta_{32})$,
  with three antisymmetric combinations (angular-momentum-like) and
  three symmetric (intrinsic-rotation-like). The 2D off-diagonal
  derivation in `notes_M3_phase0_berry_connection.md` §7
  generalizes: each pair-generator $\theta_{ab}$ couples to the
  corresponding $\tilde\beta_{ab}, \tilde\beta_{ba}$ via a 1-form
  $\tfrac12(\alpha_a^2\alpha_b\,\beta_{ab} - \alpha_a\alpha_b^2\,\beta_{ba})\,d\theta_{ab}$
  per pair, by direct extension. This is needed for the 3D analog of
  M3 Phase 9 (KH instability test), which is not yet on the M3 plan.

- **Rotation-Hamiltonian determination.** In the diagonal restriction,
  the constraint $dH \perp v_{\rm ker}$ in the 5D 2D case fixed
  $\mathcal{H}_{\rm rot}(\theta_R)$ algebraically (§6.6 of the 2D doc).
  In 9D 3D, the kernel direction is more complex, and the constraint
  fixes a 3-component analog: a 3-vector $\mathcal{H}_{\rm rot}^{ab}(\theta_{ab})$
  with one entry per pair-generator. Deriving the explicit form is
  algebraically straightforward but tedious; deferred to M3 Phase
  M3-7 implementation when an actual numerical 3D test is at hand.

## 7. References

- `reference/notes_M3_phase0_berry_connection.md` — the 2D derivation
  (this document is the 3D extension; §1–§4 of the 2D doc apply per
  pair-generator).
- `scripts/verify_berry_connection.py` — 2D SymPy verification.
- `scripts/verify_berry_connection_3D.py` — 3D SymPy verification
  (this document's verification, all 8 checks pass).
- `specs/01_methods_paper.tex` §5.4 (3D extension flagged as
  "straightforward").
- `reference/MILESTONE_3_PLAN.md` Open Question 3, Phase M3-7.
- Berry, M.V. (1984). [as in 2D doc]
- Marsden, J.E. & Ratiu, T.S. (1999), §15.1 ("Cotangent Bundle
  Reduction by SO(3)") — the standard reference for the SO(3) version
  of the Hamilton-Pontryagin construction.
