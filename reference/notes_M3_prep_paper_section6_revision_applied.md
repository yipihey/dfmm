# M3-prep: Methods-paper §6 (Bayesian remap) HG-aware revision applied

Branch: `m3-prep-paper-section6-revision`. Worktree:
`.claude/worktrees/agent-ad4d80f98b886c9b4/`.

This note records the four edits applied to `specs/01_methods_paper.tex`
(plus PDF rebuild) per the M3-prep §6 task,
`reference/notes_HG_design_guidance.md` items #5 and #8, and
`reference/MILESTONE_3_PLAN.md` Phase M3-5.

The paper file in this project lives at `specs/01_methods_paper.tex`
(not `paper/paper.tex` as written in the original task description);
the §5 revision note `notes_M3_prep_paper_section5_revision_applied.md`
established this convention and we follow it.

## Summary of edits

All edits applied. PDF rebuilt to 23 pages (was 22 pre-edit). The only
LaTeX warning is the pre-existing `sec:appendix-soft` undefined reference
(documented as pre-existing in M2-4 / §5-revision note).

| # | Section | Action | Approx lines added |
|---|---------|--------|--------------------|
| 1 | §6.2 | Renamed "Geometric overlap via r3d" → "Geometric overlap and polynomial moment transfer"; preserved the original r3d / dimension-lifting paragraph; added a new `\paragraph{Implementation via HierarchicalGrids.}` that names `HierarchicalGrids.jl`, `r3djl`, `SimplicialMesh`, `HierarchicalMesh`, `PairedMesh`/`CompositeMesh`, `PolynomialFieldSet`, `compute_overlap`, `polynomial_remap`, and explicitly states that the math content is unchanged | ~22 |
| 2 | §6.3 | Added `\label{sec:remap-LTC}` so the new HG paragraph in §6.2 can refer to "the law of total cumulants specified in §6.3" | 1 |
| 3 | §6.5 | Added `\label{sec:remap-liouville-thm}` so the new §6.6 can refer to it cleanly without `\S\ref{sec:remap}.5` syntax fragility | 1 |
| 4 | §6.6 (new) | Added `\subsection{Liouville monotone-increase diagnostic}` with: (a) per-cell increment definition (eq:liouville-increment), (b) global `liouville_increment` driver-layer diagnostic alongside `total_overlap_volume`, (c) HG-design-guidance #8 status (driver-layer until HG hook lands), (d) two-failure-mode interpretation (overlap bug vs polynomial-order shortfall, with refinement remedy via `refine_by_indicator!`), (e) complementarity with `total_overlap_volume` (positive-semidefiniteness vs partition of unity) | ~52 |
| 5 | Bibliography | Added three new `\bibitem`s: `hg-paper`, `r3djl-paper`, `hg-design-guidance` | ~16 |

## Per-edit detail

### Edit 1 — §6.2 HG-aware framing

**Before:** Title was "Geometric overlap via r3d". Single paragraph
described the polygonal overlap, r3d moment integration, and
dimension-lifting for curved edges.

**After:** Title is "Geometric overlap and polynomial moment transfer".
The original paragraph is preserved verbatim (with light copy-edit to
remove the imperative voice). A new
`\paragraph{Implementation via HierarchicalGrids.}` follows, naming
`HierarchicalGrids.jl` (`hg-paper`) as the dimension-generic substrate,
`r3djl` (`r3djl-paper`) as the Julia r3d port, and listing the four
HG types (`SimplicialMesh`, `HierarchicalMesh`, `PairedMesh`/
`CompositeMesh`, `PolynomialFieldSet`) and two HG primitives
(`compute_overlap`, `polynomial_remap`) that do the geometric work.
Closes with: "the mathematical content of the remap --- priors,
likelihoods, and posterior moment matching --- is unchanged by this
implementation choice; HG and r3djl supply the conservative geometric
transfer, and dfmm specifies the Bayesian moment-matching rule on top."

### Edits 2–3 — Subsection labels for clean cross-refs

Added `\label{sec:remap-LTC}` to §6.3 and
`\label{sec:remap-liouville-thm}` to §6.5. No prose change. Used by
the new §6.6 to cite the law of total cumulants and the monotonicity
theorem without resorting to manual section-number arithmetic.

### Edit 4 — New §6.6 Liouville monotone-increase diagnostic

Inserted between the existing §6.5 "Liouville behavior" (theorem +
sketch + physical-interpretation paragraphs, all preserved verbatim)
and §7 "Passive scalar advection". Structure:

- Lead paragraph defines $\Delta_{\rm Liou}(C_j) = \det L_j^{\rm new}
  - \sum_i w_{ij} \det L_i$ (eq:liouville-increment) and notes that
  it is recorded per remap step alongside `total_overlap_volume`,
  per the M3 acceptance criterion.
- `\paragraph{Status of the diagnostic.}` cites
  `hg-design-guidance` item #8 (built-in HG hook proposed; until
  then dfmm computes it in the driver layer; cost negligible).
- `\paragraph{Failure modes detected.}` two-bullet list:
  (a) numerical bug in `compute_overlap` or moment integrator;
  (b) polynomial reconstruction order too low for local geometry,
  with the remedy "refine locally via HG `refine_by_indicator!`,
  with $|\Delta_{\rm Liou}|$ as the indicator, rather than reduce
  the time step".
- Closing sentence emphasizes complementarity: per-cell increment
  field localizes faults more informatively than a global
  mass-conservation residual; checks
  *positive-semidefiniteness* of moment update, while
  `total_overlap_volume` checks *partition of unity* of geometry.

### Edit 5 — Three new bibliography entries

```
\bibitem{hg-paper}            HierarchicalGrids.jl v0.1, Apr 2026
\bibitem{r3djl-paper}         r3djl Julia port, Apr 2026
\bibitem{hg-design-guidance}  reference/notes_HG_design_guidance.md
```

The dfmm repository hosts `notes_HG_design_guidance.md` as the
canonical reference; the bibitem points at it.

## Cross-references

The grep for `sec:remap` cross-references in the paper before edit
returned three sites (intro paragraph at line 126, Liouville-invariant
backreference at line 248, the section label itself at line 659). All
three remain valid; numbering of §6 is unchanged at the section level
and §6.1–§6.5 retain their original numbers.

The new §6.6 is the only renumbering-relevant addition; nothing in the
rest of the paper cited a hypothetical §6.6 previously, so no
cross-reference is broken.

## PDF rebuild

```bash
cd specs && \
  pdflatex -interaction=nonstopmode 01_methods_paper.tex && \
  pdflatex -interaction=nonstopmode 01_methods_paper.tex && \
  pdflatex -interaction=nonstopmode 01_methods_paper.tex
```

Output: `01_methods_paper.pdf` (23 pages, 386 kB; was 22 pages, 378 kB).

The +1-page increase comes from the new §6.6 subsection (~52
lines) which pushes the rest of §7 to a new page break.

Three new labels registered in the .aux:

```
\newlabel{sec:remap-overlap}{{6.2}{...}}
\newlabel{sec:remap-LTC}{{6.3}{...}}
\newlabel{sec:remap-liouville-thm}{{6.5}{...}}
\newlabel{sec:remap-liouville-diag}{{6.6}{13}{Liouville monotone-increase diagnostic}{subsection.6.6}{}}
\newlabel{eq:liouville-increment}{{36}{13}{...}}
```

The only LaTeX warning is the pre-existing
`Reference 'sec:appendix-soft' undefined`, documented in
`notes_M2_4_paper_corrections_applied.md` and
`notes_M3_prep_paper_section5_revision_applied.md` as pre-existing
and not introduced by paper-revision work.

## Files touched

- `specs/01_methods_paper.tex` — ~92 insertions, ~13 deletions across
  §6.2, §6.3 (label only), §6.5 (label only), §6.6 (new), bibliography.
- `specs/01_methods_paper.pdf` — rebuilt (22 → 23 pages).
- `specs/01_methods_paper.aux`, `.toc`, `.out`, `.log` — regenerated
  by the build (untracked; .gitignore-equivalent behavior under `make
  clean`).
- `reference/notes_M3_prep_paper_section6_revision_applied.md` — this
  file.

## Items not applied / scope limits

- **Math content unchanged.** Eqs. (32)–(35) (mass, momentum, M_j^new,
  Q_j^new under law of total covariance) preserved verbatim. Theorem
  + sketch + physical-interpretation in §6.5 preserved verbatim.
- **No edits to other sections.** Algorithm summary §9.4 still says
  "r3d geometric overlap, Bayesian moment projection" — this is
  consistent with the §6.2 reframing (HG hosts the r3d kernel; the
  one-liner remains accurate). Abstract and intro mentions of "r3d"
  remain — they are correct (r3d is the polynomial-moment kernel)
  and the HG framing is layered on top in §6.2.
- The pre-existing `sec:appendix-soft` undefined reference is left
  alone (predates this work).

## Pointers

- `reference/notes_HG_design_guidance.md` items #5 (refinement-event
  callback, used by the §6.6 paragraph's `refine_by_indicator!`
  remedy) and #8 (`liouville_increment` HG hook, cited explicitly in
  §6.6 status paragraph).
- `reference/MILESTONE_3_PLAN.md` Phase M3-5 acceptance criterion is
  the source of the "match the box volume to round-off" wording in
  §6.6.
- `reference/notes_M3_prep_paper_section5_revision_applied.md` is the
  pattern this note mirrors.
