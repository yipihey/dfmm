# Phase 0 sanity check: covariant ↔ bare-derivative reduction

**Author.** Agent A, Phase 1 worktree.
**Purpose.** Verify by hand algebra that the boxed Hamilton equations
in `design/03_action_note_v2.tex` (eq. `eq:dfmm-cholesky-cov`)
reduce to the bare-derivative form `eq:dfmm-cholesky-bare`. This is
HANDOFF.md sanity check #2 and the exit gate of Phase 0.

## Algebra

The covariant time derivative on the principal $GL(d)$ bundle is
(v2 eq. `eq:Dtq-1D`)

$$
  \mathcal{D}_t^{(q)}\phi \;=\; \dot\phi + q\,(\partial_x u)\,\phi.
$$

Apply with $q=0$ to $\alpha$ and with $q=1$ to $\beta$ in v2
eq. `eq:dfmm-cholesky-cov`:

1. $\mathcal{D}_t^{(0)}\alpha = \beta
   \;\Longleftrightarrow\;
   \dot\alpha + 0\cdot(\partial_x u)\,\alpha = \beta
   \;\Longleftrightarrow\;
   \dot\alpha = \beta.$

2. $\mathcal{D}_t^{(1)}\beta = \gamma^2/\alpha
   \;\Longleftrightarrow\;
   \dot\beta + 1\cdot(\partial_x u)\,\beta = \gamma^2/\alpha
   \;\Longleftrightarrow\;
   \dot\beta + (\partial_x u)\,\beta = \gamma^2/\alpha.$

Both lines reproduce v2 eq. `eq:dfmm-cholesky-bare` term-for-term, and
match the update rule in `py-1d/dfmm/schemes/cholesky.py` (lines 26–27
of the docstring: "D alpha / Dt = beta", "D beta / Dt =
gamma^2/alpha - (du/dx) * beta").

## Consequence for the discrete scheme

Strain coupling lives entirely in the connection $\mathcal{D}_t^{(q)}$,
not in the Hamiltonian $\mathcal{H}_{\rm Ch} = -\tfrac12\alpha^2\gamma^2$
(v2 §3.1, eq. `eq:H-Ch`). The discrete integrator therefore implements
the strain term as a parallel-transport operator on time slices, not as
an explicit source in the Hamiltonian. This is the structural fact the
Phase 1 integrator must respect, and it is what differentiates the
variational scheme from a port of the py-1d HLL finite-volume
implementation.
