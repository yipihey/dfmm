# M3-prep: Methods-paper §5 Berry-connection revision applied

Branch: `m3-prep-paper-section5`. Worktree:
`.claude/worktrees/agent-a014d1c0160b03b82/`.

This note records the four edits applied to `specs/01_methods_paper.tex`
(plus PDF rebuild) per the M3-prep task and
`reference/notes_M3_phase0_berry_connection.md`.

## Summary of edits

All four edits (three required + one optional cross-reference cleanup)
applied. PDF rebuilt to 22 pages (was 21 pre-edit). The only LaTeX
warning is the pre-existing `sec:appendix-soft` undefined reference
(documented as pre-existing in M2-4).

| # | Section | Action | Approx lines added |
|---|---------|--------|--------------------|
| 1 | §5.2 | Replaced "flagged for the 2D extension paper" wording with explicit $\Theta_{\rm rot}$ / $\omega_{\rm rot}$ form (eq.~\ref{eq:Theta-rot-2D}, \ref{eq:omega-rot-2D}) plus verification pointer | ~20 |
| 2 | §5.2 | Added `\paragraph{Normalization.}` factor-of-2 clarification (convention (a) chosen) | ~10 |
| 3 | §5.5 | Replaced "flagged for further derivation" wording; added `\paragraph{Poisson-manifold constraint on $\mathcal{H}_{\rm rot}$.}` paragraph with eq.~\ref{eq:H-rot-constraint} | ~17 |
| 4 | §11 (Conclusions, "What this framework does not yet provide") | Updated item 1 to point at the new derivation in §5.2 (eq.~\ref{eq:Theta-rot-2D}--\ref{eq:omega-rot-2D}, eq.~\ref{eq:H-rot-constraint}) and reframe remaining work as the off-diagonal $L_2$ sector | ~5 (replaces 3) |

## Per-edit detail

### Edit 1 — §5.2 explicit form (lines 568–600 after edit)

**Before** (lines 567–571 pre-edit):

> The first two terms are exactly two copies of the 1D weighted
> symplectic form; reduction to 1D (set $\alpha_2=\alpha_1$,
> $\beta_2=\beta_1$, $\theta_R=0$) recovers $\omega_{1D}$ correctly.
> The Berry-connection term $\omega_{\rm rot}$ requires further
> derivation; its specific form is flagged for the 2D extension paper.

**After:** the first sentence retained; the "requires further derivation"
sentence replaced by the explicit $\Theta_{\rm rot} = \tfrac{1}{3}(\alpha_1^3\beta_2 - \alpha_2^3\beta_1)\,d\theta_R$
(eq:Theta-rot-2D) and $\omega_{\rm rot} = d\Theta_{\rm rot}$ (eq:omega-rot-2D)
plus a paragraph noting bilinear-antisymmetry, vanishing on the
iso-diagonal subspace, dimensional matching to the 1D Hamilton-Pontryagin
factor, and pointing to `scripts/verify_berry_connection.py` for symbolic
verification.

### Edit 2 — §5.2 factor-of-2 normalization paragraph

Inserted as a `\paragraph{Normalization.}` block immediately after the
new explicit-form paragraph from Edit 1. Notes that on the iso-diagonal
subspace the per-axis sum pulls back to $2\alpha^2\,d\alpha\wedge d\beta$,
gives the two consistent normalization conventions
((a) explicit $1/2$ in the per-axis weight, (b) iso-pullback as a
half-weight projection), and adopts (a) for explicitness in subsequent
work (matching the recommendation in
`notes_M3_phase0_berry_connection.md` §6.4).

### Edit 3 — §5.5 Poisson-manifold $\mathcal{H}_{\rm rot}$ constraint

**Before** (lines 600–605 pre-edit):

> with the Berry-connection contribution to the rotation Hamiltonian
> flagged for further derivation. Hamilton's equations on the diagonal
> sectors give two independent copies of the 1D dfmm Cholesky
> evolution, one per principal axis. The rotation sector evolves on
> $SO(2)$ driven by the misalignment between the $M_{xx}$ principal
> axes and the strain principal axes.

**After:** replaced "flagged for further derivation" with "specified by
the Poisson-manifold integrability constraint below"; the existing
paragraph about per-axis Hamilton equations and the $SO(2)$ rotation is
retained verbatim. Added a new `\paragraph{Poisson-manifold constraint
on $\mathcal{H}_{\rm rot}$.}` block stating:

- $\omega_{2D}$ is rank-4 in 5D (Poisson, not strictly symplectic);
- the 1D kernel is the gauge direction;
- $dH \cdot v_{\rm ker} = 0$ fixes $\partial_{\theta_R} \mathcal{H}_{\rm rot}|_{\rm constraint}$ algebraically (eq:H-rot-constraint);
- the strain-driven part requires lifting the diagonal restriction (off-diagonal $L_2$ sector).

### Edit 4 — §11 cross-reference (optional)

**Before:**
> The Berry-connection $\omega_{\rm rot}$ in the 2D matrix lift is
> sketched but not explicitly derived. The 1D reduction is verified;
> the full 2D form requires further work.

**After:** replaced with a sentence pointing at §\ref{sec:matrix-lift},
eq.~\ref{eq:Theta-rot-2D}--\ref{eq:omega-rot-2D}, and eq.~\ref{eq:H-rot-constraint};
notes the SymPy verification; reframes remaining work as the
off-diagonal $L_2$ sector for 2D Kelvin-Helmholtz tests in M3 Phase 9.

A grep for `extension paper\|further derivation` after the edits confirms
no remaining stale references; the only surviving "2D-extension paper"
phrase is intentional and refers (in the new §5.5 paragraph) to
implementation notes for follow-on work on the off-diagonal $L_2$ sector.

## PDF rebuild

Command (run from `specs/`):

```bash
pdflatex -interaction=nonstopmode 01_methods_paper.tex
pdflatex -interaction=nonstopmode 01_methods_paper.tex
pdflatex -interaction=nonstopmode 01_methods_paper.tex   # third pass settled cross-refs
```

Output:

```
Output written on 01_methods_paper.pdf (22 pages, ~378 kB).
```

Three new equation labels registered in the .aux file:

```
\newlabel{eq:Theta-rot-2D}{{28}{10}{...}{equation.28}{}}
\newlabel{eq:omega-rot-2D}{{29}{10}{...}{equation.29}{}}
\newlabel{eq:H-rot-constraint}{{31}{11}{Poisson-manifold ...}{equation.31}{}}
```

The only LaTeX warning is the pre-existing
`Reference 'sec:appendix-soft' undefined`, documented in
`notes_M2_4_paper_corrections_applied.md` as pre-existing and not
introduced by paper-revision work.

## Files touched

- `specs/01_methods_paper.tex` — ~52 insertions, ~7 deletions across §5.2, §5.5, §11.
- `specs/01_methods_paper.pdf` — rebuilt (21 → 22 pages).
- `reference/notes_M3_prep_paper_section5_revision_applied.md` — this file.

## Items not applied / scope limits

- All four edits applied as specified.
- The narrative claim in §5.5 ("rotation sector ... driven by the
  misalignment between the $M_{xx}$ principal axes and the strain
  principal axes") is retained verbatim, as instructed; the new
  Poisson-manifold paragraph supplies the quantitative no-misalignment
  baseline ($\tilde S_{12} = 0$ sector).
- The pre-existing `sec:appendix-soft` undefined reference is left
  alone (predates this work, M2-4-documented).
- The off-diagonal $L_2$ sector remains flagged as future work
  (consistent with `notes_M3_phase0_berry_connection.md` §7).
