# M3-3 prep: Methods-paper §10 (validation tests) M3-aware revision applied

Branch: `m3-3-prep-paper-section10-revision`. Worktree:
`.claude/worktrees/agent-af879d5bc5964f2cb/`. Local main HEAD at branch
point: `6f2fc94`.

This note records the edits applied to `specs/01_methods_paper.tex`
(plus PDF rebuild) to incorporate the M3-3a/b/c/d/e numerical results
into §10 (validation tests). Pattern follows
`reference/notes_M3_prep_paper_section5_revision_applied.md` and
`reference/notes_M3_prep_paper_section6_revision_applied.md`.

## Summary of edits

All edits applied. PDF rebuilt to **26 pages** (was 23 pre-edit).
The +3-page increase comes from the new §10.7 subsection (~95 lines)
plus the new C.2 figure (~half page); together they push two §11
subsection breaks to new pages.

The only LaTeX warning is the pre-existing
`Reference 'sec:appendix-soft' on page 6 undefined`, documented as
pre-existing in `notes_M2_4_paper_corrections_applied.md`,
`notes_M3_prep_paper_section5_revision_applied.md`, and
`notes_M3_prep_paper_section6_revision_applied.md`. No new warnings.

| # | Section / target            | Action                                                                                                                                                                                                                                                                                                                                                                                                                | Approx lines added |
|---|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| 1 | §10.1 Validation hierarchy  | Added a new `\paragraph{Bit-exact regression protocol across substrate ports.}` paragraph to the end of the hierarchy, summarising the M1+M2 → HG bit-exact carry-forward (3037+ asserts) via the `cache_mesh::Mesh1D` shim and the M3-3e sub-phase decomposition.                                                                                                                                                     | ~20                |
| 2 | §10.2–§10.6 labels          | Added `\label{sec:tier-A}`, `\label{sec:tier-B}`, `\label{sec:tier-C}`, `\label{sec:tier-D}`, `\label{sec:tier-E}`, `\label{sec:tier-acceptance}` so the new §10.7 (and any future cross-references) can refer to the tiers without manual section-number arithmetic. No prose change in those subsections.                                                                                                            | 6                  |
| 3 | §10.4 Tier C C.2 figure     | New `\begin{figure}` block immediately after the C.1–C.3 `itemize`. Includes `reference/figs/M3_3d_per_axis_gamma_selectivity.png` at `width=0.95\textwidth`. Caption gives the cold-sinusoid IC (k_x=1, k_y=0, A=0.5, level=4, dt=2e-3, 100 steps), the γ_1/γ_2 spatial ranges, and the >10^13 spatial-std ratio. `\label{fig:M3-3d-selectivity}` for cross-ref. **First figure in the paper.**                       | ~17                |
| 4 | §10.4 Tier C conservation   | New `\paragraph{Tier C conservation-properties acceptance.}` mentioning HG IntExact backend availability at HG commit `81914c2` (`overlap_simplex_box_exact!`, `IntegerLattice`, `compute_overlap(...; backend=:exact)`), and that M3-5 will adopt `:exact` as the conservation-regression gate.                                                                                                                       | ~14                |
| 5 | §10.7 (new)                 | **New subsection** `\subsection{M3 implementation gate results}` between Tier E (§10.5) and Acceptance criteria (§10.8). Eight `\paragraph` blocks: (a) dimension-lift gate (0.0 abs), (b) Berry-coupling residual verification (72 asserts, FD tol 1e-9), (c) iso-pullback ε-expansion (slope 1.003), (d) H_rot solvability (≤1e-10), (e) per-axis γ selectivity (>10^13 ratio, with `\ref{fig:M3-3d-selectivity}`), (f) plane-wave convergence (Δx² rate 3.93→3.98), (g) bit-exact 0.0 parity protocol, (h) HG IntExact backend availability. Each cites the relevant design note under `reference/notes_M3_3*.md`. | ~95                |

## Per-edit detail

### Edit 1 — §10.1 bit-exact regression protocol paragraph

Inserted after the Tier E paragraph in the Validation hierarchy. Lead
sentence: "The Tier A and Tier B 1D regression suites are not only
verified against \texttt{dfmm} reference output: they are
additionally held \emph{bit-exact}…". Cites M1 (305+140+21=466) +
M2 (243) + 1335 cross-phase + M3-port-specific tests = 3037+ asserts
through the `cache_mesh::Mesh1D` shim. Cites
`reference/MILESTONE_3_STATUS.md` for the full counts. Closes with
the philosophical statement that bit-exactness is the
\emph{strongest} acceptance criterion when attainable.

### Edit 2 — Subsection labels

Added six new `\label{...}` lines (sec:tier-A through
sec:tier-E plus sec:tier-acceptance). No prose change. Used by the
new §10.7 paragraphs to cite specific tiers when discussing bridges
between tiers (e.g. dimension-lift gate cited as Tier A
↔ Tier C bridge).

### Edit 3 — C.2 headline figure

`\begin{figure}[ht]` with the M3-3d selectivity 4-panel chart.
Figure path: `../reference/figs/M3_3d_per_axis_gamma_selectivity.png`
(relative from `specs/`). Caption gives the IC parameters, the
spatial-range numerics ([0.940, 0.987] for γ_1, [0.981, 0.981] for
γ_2), and the >10^13 std ratio with the explanatory observation
that γ_2 spatial variance is at machine precision. Calls out that
this is the load-bearing scientific gate for the 2D matrix lift of
§\ref{sec:matrix-lift}.

This is the **first figure** in the paper. The `graphicx` package
was already loaded in the preamble.

### Edit 4 — Tier C conservation-properties paragraph

Added between the C.1–C.3 itemize and Tier D. Names the HG
`Overlap/r3d_int_adapter.jl` API (`overlap_simplex_box_exact!`,
`IntegerLattice`, the `backend=:exact` keyword) and the HG commit
`81914c2` at which it landed. Frames the protocol: every
conservation gate is run under both `:float` and `:exact`, with the
exact-rational result bounding the geometric round-off contribution.
M3-5 acceptance gate.

### Edit 5 — New §10.7 M3 implementation gate results

Eight `\paragraph` blocks summarising the M3-3 numerical findings.
Each block:

1. **Dimension-lift gate** — 0.0 absolute byte-equal, all
   resolutions / dt / mesh sizes / axis-swap; reference
   `notes_M3_3b_native_residual.md`.
2. **Berry-coupling residual verification** — closed-form
   ᾱ_2³/(3ᾱ_1²)/dt-style partials match to FD tol 1e-9 across 8
   samples × 9 sub-tests; symplectic-weight identity
   ∂F^*/∂θ̇_R · α_a² = berry_partials cross-check.
3. **Iso-pullback ε-expansion** — F-function scales linearly in
   (α_1−α_2), measured slope 1.003; F vanishes on iso submanifold;
   action contribution scales as O(ε).
4. **H_rot solvability** — closed-form vs kernel-orthogonality ≤
   1e-10 at 5 random pts × 3 θ̇_R; Newton ≤ 5 iters at non-iso IC
   (α=(1.2,0.8), β=(0.15,−0.10), θ_R=0.13); post-Newton residual ≤
   1e-10.
5. **Per-axis γ selectivity** — std-ratio > 10^13, with figure
   `\ref`.
6. **Plane-wave convergence (Tier C C.3)** — levels 4/5/6 give
   6.29e-6, 1.60e-6, 4.01e-7; ratios 3.93, 3.98 (Δx² midpoint
   rule).
7. **Bit-exact 0.0 parity protocol** — M1+M2 → HG carry-forward,
   3037+ asserts, M3-3e sub-phase decomposition.
8. **HG IntExact backend availability** — `:exact` backend at
   commit `81914c2`, M3-5 acceptance gate.

Each block ends with a `\verb|reference/notes_M3_3*.md|` pointer,
mirroring the §6 revision's bibliography style for design notes.

## Cross-references

- `\ref{fig:M3-3d-selectivity}` from §10.7 paragraph (e) → C.2
  figure in §10.4.
- `\S\ref{sec:matrix-lift}` from §10.7 paragraphs (b), (c) and the
  C.2 figure caption → §5 of the paper.
- `\S\ref{sec:tier-A}`, `\S\ref{sec:tier-C}` from §10.7 paragraph
  (a) → tier subsections.
- `\S\ref{sec:remap-liouville-diag}` from the new §10.4
  conservation paragraph → §6.6.
- `\eqref{eq:Theta-rot-2D}`–`\eqref{eq:omega-rot-2D}` and
  `\eqref{eq:H-rot-constraint}` cited from §10.7 paragraphs (b)
  and (d), respectively. Pre-existing labels.

The §1 introduction's roadmap paragraph and the §11
"Implementation milestones" `description` list are unchanged
(no §10.X cross-reference into them was previously present, and
the explicit milestone wording was outside the brief's scope).

## PDF rebuild

```bash
cd specs && \
  pdflatex -interaction=nonstopmode 01_methods_paper.tex && \
  pdflatex -interaction=nonstopmode 01_methods_paper.tex && \
  pdflatex -interaction=nonstopmode 01_methods_paper.tex
```

Output: `01_methods_paper.pdf` (26 pages, 514 kB; was 23 pages,
~387 kB).

The +3-page increase distributes as:
- ~half page from the new C.2 figure (`width=0.95\textwidth`,
  4-panel matplotlib aspect ratio).
- ~one page from the new conservation-properties Tier C paragraph
  + bit-exact regression-protocol paragraph in §10.1 + tier labels.
- ~1.5 pages from the new §10.7 with eight gate paragraphs.

New labels registered in the .aux:

```
\newlabel{sec:tier-A}{...}
\newlabel{sec:tier-B}{...}
\newlabel{sec:tier-C}{...}
\newlabel{sec:tier-D}{...}
\newlabel{sec:tier-E}{...}
\newlabel{sec:tier-acceptance}{...}
\newlabel{sec:tier-M3-gates}{...}
\newlabel{fig:M3-3d-selectivity}{...}
```

The only LaTeX warning is the pre-existing
`Reference 'sec:appendix-soft' on page 6 undefined`. No new warnings.

## Files touched

- `specs/01_methods_paper.tex` — ~152 insertions across §10.1
  (paragraph), §10.2/§10.3/§10.4/§10.5/§10.6 (labels), §10.4
  (figure + paragraph), §10.7 (new subsection).
- `specs/01_methods_paper.pdf` — rebuilt (23 → 26 pages).
- `specs/01_methods_paper.aux`, `.toc`, `.out`, `.log` —
  regenerated by the build.
- `reference/notes_M3_3_prep_paper_section10_revision_applied.md`
  — this file.

## Items not applied / scope limits

- **Test specifications unchanged.** The math content of §10
  (Tier A/B/C/D/E test definitions, expected behaviors,
  acceptance criteria enumeration) is preserved verbatim.
- **§11 Implementation milestones list left alone.** It still
  reads "Milestone 3. 2D Rust implementation" with an `Estimated
  3--4 months` note; the brief did not authorise rescoping
  that prose, and the new §10.7 already records actual M3-3
  progress with HG/Julia framing. A future paper revision may
  want to refresh the milestone table.
- **Abstract not touched.** It mentions "r3d on curved elements"
  which remains correct; HG framing was added in §6.2 by the
  earlier §6 revision.
- **No bibliography additions.** All design-note references in
  §10.7 use `\verb|reference/notes_M3_3*.md|` inline, matching
  the §6.6 Liouville-diagnostic precedent (which does the same
  for `notes_HG_design_guidance.md`).
- The pre-existing `sec:appendix-soft` undefined reference is
  left alone (predates this work; tracked in
  `notes_M2_4_paper_corrections_applied.md`).

## TODO / follow-ups

- §10.7 paragraph (b)'s closed-form Berry partial display
  abbreviates with "...". A future expansion could enumerate all
  four cross-axis partials; the brief did not call for the full
  list, and the design note has them.
- §11 "Implementation milestones" still says "Rust"; would benefit
  from a one-line update to "Rust → Julia (HG-backed)" once the
  project lead is ready to revise §11 broadly.
- The C.3 plane-wave convergence numerics in §10.7(f) are quoted
  but not plotted. A small log–log convergence figure would
  complement the C.2 selectivity figure if the project lead
  wants it; not in the brief.

## Pointers

- `reference/notes_M3_3a_field_set_cholesky.md`,
  `notes_M3_3b_native_residual.md`,
  `notes_M3_3c_berry_integration.md`,
  `notes_M3_3d_per_axis_gamma_amr.md`,
  `notes_M3_3e_cache_mesh_retirement.md` — sources of the
  individual gate numerics.
- `reference/MILESTONE_3_STATUS.md` — global view, 3037+ assert
  count, M3-3e status block.
- `reference/notes_M3_prep_tierC_ic_factories.md`,
  `notes_M3_2b_swap5_init_field.md` — plane-wave Δx² convergence
  table.
- `reference/notes_M3_prep_paper_section6_revision_applied.md`
  — pattern this note mirrors.
