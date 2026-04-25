# Phase 2 discretization note

**Author.** Agent A, Phase 2 worktree (`phase2-bulk`).
**Purpose.** Document the staggered mesh, time-stepping, and discrete EL
system used by the multi-segment deterministic integrator implementing
$\mathcal{L}_{\rm det} = \tfrac12\dot x^2 + \mathcal{L}_{\rm Ch}$ (v2 §3.2;
methods paper §3.2 / §9.4–§9.5).

## Mesh layout

Periodic 1D Lagrangian mass-coordinate mesh with $N$ segments $j=1,\dots,N$,
fixed Lagrangian masses $\Delta m_j$.

* **Vertex variables (charge 0 under strain):** positions $x_i$ and
  velocities $u_i$, $i=1,\dots,N$. Periodicity ties
  $x_{N+1} \equiv x_1 + L_{\rm box}$.
* **Cell-centered:** Cholesky $(\alpha_j, \beta_j)$ (charges 0, 1) and
  specific entropy $s_j$ (charge 0).

Vertex-associated mass is
$\bar m_i = \tfrac12 (\Delta m_{i-1} + \Delta m_i)$, cyclic in $i$.

`DetField` packages each segment with its left-vertex `(x, u)` and the
cell-centered `(α, β, s)` for storage convenience; the Newton system
treats positions and velocities as vertex-indexed and Cholesky as
segment-indexed (no double-counting).

## Per-segment density, J, M_vv, P_xx

* $J_j = (x_{j+1}-x_j)/\Delta m_j$ (specific volume).
* $\rho_j = 1/J_j$.
* $M_{vv,j} = J_j^{1-\Gamma}\,\exp(s_j/c_v)$ via Track-C `eos.jl` `Mvv(J, s)`.
* $P_{xx,j} = \rho_j\,M_{vv,j}$.
* For diagnostics, $\gamma_j = \sqrt{\max(M_{vv,j}-\beta_j^2, 0)}$. Not
  evolved.

## Cell-centered strain rate

$$
  (\partial_x u)_j = \frac{u_{j+1} - u_j}{x_{j+1} - x_j}.
$$

At the midpoint of a step we evaluate this with the midpoint averages
$\bar u_i = (u_i^n + u_i^{n+1})/2$, $\bar x_i = (x_i^n + x_i^{n+1})/2$.

## Discrete action (single step)

Per-step (midpoint-rule quadrature):
$$
  \Delta S_n = \Delta t\,\Big[\sum_i \frac{\bar m_i}{2}\,\bar u_i^2
              + \sum_j \Delta m_j\,\mathcal{L}_{\rm Ch,j}^{n+1/2}\Big],
$$
where $\bar u_i = (u_i^n + u_i^{n+1})/2$ and the Cholesky-sector
contribution per segment is the same midpoint quadrature used in
Phase 1 (see `cholesky_one_step_action`):
$$
  \Delta t\,\mathcal{L}_{\rm Ch,j}^{n+1/2}
    = -(\bar\alpha_j^3/3)(\beta_j^{n+1}-\beta_j^n)
      - \Delta t (\bar\alpha_j^3/3)(\partial_x u)_j^{n+1/2}\,\bar\beta_j
      + (\Delta t/2)\bar\alpha_j^2 (\bar M_{vv,j} - \bar\beta_j^2),
$$
with $\bar M_{vv,j} = J_j^{n+1/2,\,1-\Gamma}\exp(s_j/c_v)$ evaluated at
the midpoint $J$. The position kinetic uses $\bar u_i$ directly (it is
the same as the implicit-midpoint quadrature of $\tfrac12 \dot x^2$).

## Discrete EL system (Newton residual)

The midpoint-rule applied to the canonical Hamilton equations
$\dot x = u$, $\dot u = -\partial_m P_{xx}$ for the bulk and the boxed
Cholesky equations of v2 §2.1 yields a $4N$-DOF coupled nonlinear
system in the unknowns $(x_i^{n+1}, u_i^{n+1}, \alpha_j^{n+1}, \beta_j^{n+1})$
with $s_j$ frozen ($\dot s = 0$, deterministic adiabatic). Per
segment $j$ and per vertex $i$ (cyclic in both):
$$
\begin{aligned}
  F^x_i &= \frac{x_i^{n+1} - x_i^n}{\Delta t} - \bar u_i = 0, \\
  F^u_i &= \frac{u_i^{n+1} - u_i^n}{\Delta t}
          + \frac{\bar P_{xx,i} - \bar P_{xx,i-1}}{\bar m_i} = 0, \\
  F^\alpha_j &= \frac{\alpha_j^{n+1} - \alpha_j^n}{\Delta t} - \bar\beta_j = 0,\\
  F^\beta_j  &= \frac{\beta_j^{n+1} - \beta_j^n}{\Delta t}
              + (\partial_x u)_j^{n+1/2}\,\bar\beta_j
              - (\bar M_{vv,j} - \bar\beta_j^2)/\bar\alpha_j = 0.
\end{aligned}
$$

Sign convention on $\mathcal{L}_{\rm Ch}$ matches Phase 1's
`cholesky_sector.jl` (see "sign note" there). Phase 1 is recovered
when $u_i^{n} = 0$, $x_i^n$ static (so no positions evolve and $J$
stays constant), and $\bar M_{vv}$ is therefore the Phase-1 constant.

## Sparsity

Banding on the natural ordering
$(\dots, x_i, u_i, \alpha_i, \beta_i, x_{i+1}, u_{i+1}, \dots)$:

* $F^x_i$ : depends on $x_i^{n+1}, u_i^{n+1}$.
* $F^u_i$ : depends on $u_i^{n+1}$, $x_{i-1}^{n+1}, x_i^{n+1}, x_{i+1}^{n+1}$
  (via flanking pressures), and $s_{i-1}, s_i$ (frozen).
  Pressures depend only on positions and entropies (not on $\alpha,\beta$).
  So $F^u_i$ is *independent* of $\alpha,\beta$ in Phase 2.
* $F^\alpha_j$ : depends on $\alpha_j^{n+1}, \beta_j^{n+1}$ only.
* $F^\beta_j$ : depends on $\alpha_j^{n+1}, \beta_j^{n+1}$,
  $x_j^{n+1}, x_{j+1}^{n+1}$ (via $J^{n+1/2}$ for $\bar M_{vv}$ and
  via $(\partial_x u)$ for the strain), and on
  $u_j^{n+1}, u_{j+1}^{n+1}$ (via $\bar u$ in the strain).

Net bandwidth: 8 (about $\pm 2$ blocks of 4) in the natural ordering.
For $N \le 128$ a dense ForwardDiff Jacobian is fast; we use
NonlinearSolve `NewtonRaphson()` with `AutoForwardDiff()` and accept
$O(N^2)$ Jacobian build cost as Phase-2 acceptable.

## Time-stepping (single-step, no carried half-step state)

The implicit-midpoint rule above is an *autonomous* one-step map
$(q^n, u^n) \mapsto (q^{n+1}, u^{n+1})$. No half-step momentum needs
to be carried across calls. `Mesh1D.p_half` is left in the struct as
diagnostic / convenience storage but the Newton solve does not consult
it; instead it reads `seg.state.u` directly. This matches the
Hamilton form of v2 §3.2 (where $u$ is a primary variable).

`s` is updated trivially after the Newton solve: $s_j^{n+1} = s_j^n$.

## Conservation properties

* **Mass.** $\Delta m_j$ are labels: `total_mass` is bit-stable across
  steps. Asserted in `test_phase2_mass.jl`.
* **Momentum.** Translation invariance of the discrete action under
  $x_i \mapsto x_i + c$ (for periodic BC) implies discrete Noether
  conservation of $\sum_i \bar m_i u_i$. The pressure-difference
  telescoping $\sum_i (\bar P_{xx,i} - \bar P_{xx,i-1}) = 0$ for any
  periodic state proves this directly from the residual $F^u_i$.
  Asserted in `test_phase2_momentum.jl`.
* **Energy.** Approximate, $O(\Delta t^2)$ bounded oscillation. Tested
  globally only in Phase 4 (B.1).

## Phase-1 reduction check

Set $u_i^n = 0$, fix $x_i$ at uniform $\Delta x = J_0\,\Delta m$
positions, freeze $s$. With no pressure gradient (uniform state),
$F^u_i = 0$ identically; $F^x_i = 0$ enforces $u^{n+1} = -u^n = 0$ ⇒
$x^{n+1} = x^n$ (positions static). With $J$ static, $\bar M_{vv}$
is the Phase-1 constant. With $u_i \equiv 0$, $(\partial_x u) = 0$.
Then $F^\alpha_j$ and $F^\beta_j$ reduce *bit-identically* to
`cholesky_el_residual` per segment. Phase-1 tests pass unchanged on
any single-segment Phase-1 mesh because the Phase-1 driver
(`cholesky_step` / `cholesky_run`) bypasses the multi-segment path
entirely.
