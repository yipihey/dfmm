# M2-4: Methods-paper text corrections applied

Branch: `m2-4-paper-corrections`. Worktree:
`.claude/worktrees/agent-afcc9df89e0db2240/`.

This note records the four corrections applied to
`specs/01_methods_paper.tex` (and PDF rebuild) per the M2-4 task and
`reference/notes_methods_paper_corrections.md`.

## Summary of edits

All four corrections from `notes_methods_paper_corrections.md` were
applied. PDF rebuild succeeded with no new errors. One pre-existing
undefined cross-reference (`sec:appendix-soft`) remains; it was not
introduced by this work and lives outside the scope of these
corrections.

| # | Section | Action | Lines added (approx) |
|---|---------|--------|----------------------|
| 1 | §3.5, §10.2 A.2 | Added "Note on the magnitude direction" paragraph + replaced A.2 wording | ~14 |
| 2 | §3.1 | Added "Note on the symplectic-potential sign convention" paragraph | ~14 |
| 3 | §3.2 (eq:VG) | Replaced VG pdf with correctly-normalized form + parametrization mapping note | ~10 |
| 4 | §10.3 B.1 | Added Phase-4 caveat paragraph explicitly | ~15 |

## Per-correction detail

### Correction 1 — Hessian/γ direction at caustic (§3.5, §10.2 A.2)

**§3.5 (around line 379, after the existing rank-1 paragraph):**
added a new `\paragraph{Note on the magnitude direction at the
caustic.}` clarifying that for adiabatic Γ > 1 the *evolved*
$M_{vv}$ grows at the caustic (because $J\to 0$), so $\gamma^2$ and
$|\det\,\mathrm{Hess}|$ have spatial *maxima* at the caustic, not
minima. The cold-limit *initial* state has $\gamma\to 0$ uniformly;
the *evolved* caustic is where $\beta\to\sqrt{M_{vv}}$ saturates while
$M_{vv}$ diverges. The variational signature is the spatial *shape*
(symmetric peak at rank-1 ridge), not the magnitude direction.
Pointers to `reference/figs/phase3_hessian_degen.png` and
`reference/figs/A2_cold_sinusoid_density.png` included.

**§10.2 A.2 (was line 941, now ~line 956):** replaced
> "$\gamma$ drops by $\sim 6$ decades at caustics"

with
> "$\gamma^2$ spatial-max grows by 0.3–0.9 decades at caustics across
> six $\tau$ decades (verified across $\tau\in\{10^{-3},...,10^{7}\}$
> in M1 Phase 6)."

### Correction 2 — eq:L-Ch sign convention (§3.1)

**Diagnosis:** the methods paper §3.1 (line 293) uses the v2
positive-sign convention $\mathcal{L}_{\rm Ch} = +(\alpha^3/3)\,
\mathcal{D}_t^{(1)}\beta - \mathcal{H}_{\rm Ch}$. M1 Phase 1's
implementation in `src/cholesky_sector.jl` (header lines 36–43)
uses the negative sign with $\theta = -(\alpha^3/3)\,\dd\beta$ so
that standard EL signs reproduce the boxed Hamilton equations.

**Action chosen:** option (b) — added a new `\paragraph{Note on
the symplectic-potential sign convention.}` immediately after the
existing Symplectic-potential paragraph in §3.1 (Cholesky sector,
~line 295). The note flags the convention discrepancy, points to
the M1 implementation, and notes that downstream EL machinery must
use the matched convention. This is the "1 sentence + sign flip"
fix scoped in the task — chose the convention-clarification rather
than flipping the sign in eq:L-Ch itself, since the boxed equation
is referenced elsewhere and (per the task instructions) flipping
the sign would have wider downstream impact.

A `\ref{sec:discrete}` to the discrete-EL section §9 was used (the
section's existing label, line 730).

### Correction 3 — VG pdf normalization (§3.2)

Replaced equation `eq:VG` (was lines 431–436) with the
correctly-normalized form derived in
`src/stochastic.jl::pdf_variance_gamma`:

$$f(\epsilon) = \sqrt{\tfrac{2}{\pi}}\,\theta^{-\lambda}\,
\frac{(|\epsilon|\sqrt{\theta/2})^{\lambda-1/2}\,
K_{\lambda-1/2}(|\epsilon|\sqrt{2/\theta})}{\Gamma(\lambda)}.$$

Also added: explicit declaration that $\theta$ is in the
variance-mixture-canonical (Distributions.jl) parametrization
($\mathbb{E}[V]=\lambda\theta$); reference to
`test_vg_sampler.jl` quadrature confirmation; the $\theta_{\rm GH}
= 2\theta$ mapping to the textbook generalised-hyperbolic form.

Chose option (a) per task guidance ("pick (a) for clarity").

### Correction 4 — bounded-oscillation caveat (§10.3 B.1)

Added a one-paragraph `\emph{Caveat (M1 Phase 4).}` block inside
the B.1 itemize entry (was line 957, now ~line 985), giving:

- the analytical autonomous trajectory $\alpha(t)=\sqrt{1+t^2}$,
  $\beta(t)=t/\sqrt{1+t^2}$ from $\dot\alpha=\beta$,
  $\dot\beta=\gamma^2/\alpha$ at fixed $\gamma$;
- explicit statement that bounded-oscillation cancellation does
  not apply in the autonomous-(α,β) limit;
- M1 Phase 4 numbers: $5.6\times 10^{-9}$ over $10^5$ steps,
  $t^1$-secular within bound;
- pointer to Kraus 2017 projected variational integrator
  (arXiv:1708.07356) as future work to enforce the level-set
  constraint $\alpha^2(M_{vv}-\beta^2)=\mathrm{const}$.

Pointer to `reference/notes_phase4_energy_drift.md`.

## PDF rebuild

Command (run from `specs/`):

```bash
pdflatex -interaction=nonstopmode 01_methods_paper.tex
pdflatex -interaction=nonstopmode 01_methods_paper.tex
```

Two passes were needed for cross-references to resolve. No
bibliography step is needed (the document uses only inline
citations referencing `~\cite{action-note-v3}` etc., resolved at the
bib-key level). Output:

```
Output written on 01_methods_paper.pdf (20 pages, ~372 kB).
```

The pre-existing `LaTeX Warning: Reference 'sec:appendix-soft'
undefined` is unrelated to this work and was present before the
edits.

## Files touched

- `specs/01_methods_paper.tex` — 61 insertions, 8 deletions.
- `specs/01_methods_paper.pdf` — rebuilt.
- `reference/notes_M2_4_paper_corrections_applied.md` — this file.

## Items not applied / scope limits

- **None of the four corrections was skipped.** All four sections in
  the methods paper contained the issue described in the task.
- The v2 design note `design/03_action_note_v2.tex` was deliberately
  left untouched per the file-ownership constraints (the v2 sign
  convention is referenced in the new §3.1 paragraph but the v2
  source itself is in `design/`, which is out of scope).
- The pre-existing `sec:appendix-soft` undefined reference is left
  alone; it predates the M2-4 work.
