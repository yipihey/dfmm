# M3-9 prep: Methods-paper §11 (Implementation milestones) Rust→Julia revision applied

Branch: `m3-9-prep-paper-section11-revision`. Worktree:
`.claude/worktrees/agent-a68484011439aa33d/`. Local main HEAD at branch
point: `f69a0a1` (post-M3-6 Phase 2 close).

This note records the edits applied to `specs/01_methods_paper.tex`
(plus PDF rebuild) to retire the Rust framing in §11 (Implementation
milestones) and replace the speculative 5-milestone schedule with the
HG-adapted plan reflecting the actual M1 → M2 → M3-0/.../M3-9
progression. Pattern follows
`reference/notes_M3_prep_paper_section5_revision_applied.md`,
`reference/notes_M3_prep_paper_section6_revision_applied.md`, and
`reference/notes_M3_3_prep_paper_section10_revision_applied.md`.

## Summary of edits

All edits applied. PDF rebuilt to **28 pages** (was 26 pre-edit). The
+2-page increase is the new §11 schedule (longer prose + the new
"Language and substrate" paragraph + the 11-item milestone
`description` list against the original 5-item one).

The only LaTeX warning is the pre-existing
`Reference 'sec:appendix-soft' on page 6 undefined`, documented as
pre-existing in
`notes_M2_4_paper_corrections_applied.md`,
`notes_M3_prep_paper_section5_revision_applied.md`,
`notes_M3_prep_paper_section6_revision_applied.md`, and
`notes_M3_3_prep_paper_section10_revision_applied.md`. No new warnings.

## Rust mention audit

`grep -ni rust specs/01_methods_paper.tex` before this revision:

| Line | Context |
|-----:|---------|
| 1112 | "the standalone Rust/Julia 1D code" (in §10.1 bit-exact protocol paragraph) |
| 1502 | "Milestone 1. 1D Rust implementation of the unified scheme" |
| 1508 | "Milestone 3. 2D Rust implementation. Tier C consistency tests" |

Three mentions found (the brief estimated 5; the actual count was 3).
After revision, only one mention remains (line 1504), in the new
`\paragraph{Language and substrate.}` historical-context sentence:

> "the original design document called for a Rust port of the dfmm-1D
> Python+Numba code (`py-1d`); the Julia choice was made before M1 to
> obtain a better numerical-computing ecosystem [...]"

This single retained mention is intentional and necessary: it
documents the original design intent and the reason the Julia choice
was made, so a future reader of the design document and the methods
paper sees a coherent rationale rather than an unexplained
substitution.

## Per-edit detail

### Edit 1 — §10.1 bit-exact protocol "Rust/Julia" → "Julia"

Single-token edit at line 1112. The Rust/Julia disjunction was a
remnant from when the M3 substrate-port narrative was being drafted
against a hypothetical Rust standalone code. The standalone 1D code
is and always was Julia (M1 / M2). Now reads:

> "across the M3 substrate port from the standalone Julia 1D code
> onto `HierarchicalGrids.jl`."

### Edit 2 — §11.3 "Implementation milestones" full revision

The original `\subsection{Implementation milestones}` had:
- A two-line lead ("The framework is designed for staged
  implementation:").
- A 5-item `description` list: Milestone 1 (1D Rust), 2 (1D Tier B),
  3 (2D Rust), 4 (2D Tier D), 5 (Tier E + applications), with
  "Estimated N--M months" annotations and a "Total: 12--18 months"
  closer.

The revision replaces both with:

1. **`\paragraph{Language and substrate.}`** — names the Julia
   reference implementation (`https://github.com/yipihey/dfmm-2d`),
   explains the Rust→Julia choice (better numerical-computing
   ecosystem; sparse-AD via `SparseConnectivityTracer.jl` +
   `ForwardDiff.jl`; faster iteration; HG.jl substrate), names HG and
   r3djl as the substrate + clipper, names the IntegerLattice / `:exact`
   backend as the conservation-regression reference, and notes that HG
   is dimension-generic from day 1 (so 3D moves into M3 rather than
   downstream). Mentions the `cache_mesh::Mesh1D` shim and its M3-3e
   retirement.
2. **`\paragraph{Schedule against this design document.}`** — explains
   the design-doc → code-milestone mapping. Single sentence: design-M1
   → code-M1, design-M2 → code-M2, design-M3 ⊂ code-M3-0..M3-4,
   design-M4 ⊂ code-M3-6, design-M5 ⊂ code-M3-8.
3. **An 11-item `description` list** of code milestones M1, M2,
   M3-0/1/2, M3-2b, M3-3, M3-4, M3-5, M3-6 (Phases 0/1/2 done; 3+
   ahead), M3-7, M3-8, M3-9. Each item:
   - tagged `(DONE)` or `(ahead)` to set reader expectation;
   - one-paragraph headline with the load-bearing numerics
     (e.g., M2 "37.8% cell savings"; M3-3 "dimension-lift gate at
     Δ = 0 absolute"; M3-5 "0–6.7e-16 mass over 5 cycles + 1e-12
     POU"; M3-6 Phase 1c "γ_measured/γ_DR = 1.34, mesh-converged
     1.2%, n_neg_jac = 0"; M3-6 Phase 2 "std(γ_1)/std(γ_2) ≈
     2.6e14 + γ_2 uniform to round-off + max|β_off| = 0");
   - a `\verb|reference/notes_M3_*.md|` or
     `\verb|MILESTONE_*_STATUS.md|` pointer for the per-phase
     numerics.
4. **A closing paragraph** spelling out the design-doc → code mapping
   explicitly so a reader of the original design document can see the
   plan unchanged in scope, just compressed in engineering unit (3D
   moves into code-M3 rather than as a separate "Outlook" item).

### Edit 3 — `\label{sec:milestones}`

Added immediately after the `\subsection{Implementation milestones}`
heading so the milestone block can be cited cleanly from §1, §10, or
the abstract in future revisions. No prose change. Not yet referenced
from anywhere in the current paper text — present for future use,
mirrors the §10 revision's pre-emptive labels (`sec:tier-A`,
`sec:tier-B`, etc.).

### Edit 4 — `\bibitem{sct-paper}` for `SparseConnectivityTracer.jl`

Added at the end of `\begin{thebibliography}{99}` block, after
`hg-design-guidance`. The bibliography style follows the existing
`hg-paper` / `r3djl-paper` entries (author + descriptive title +
"April 2026" + code repository URL). Cited by the new
`\paragraph{Language and substrate.}` paragraph in §11. Without this
entry, `\cite{sct-paper}` would render as a `[?]`.

## Cross-references

- `\S\ref{sec:remap}` (§6 Bayesian remap) — cited in
  `\paragraph{Language and substrate.}` for the IntegerLattice
  conservation-regression reference. Pre-existing label, resolves
  cleanly.
- `\S\ref{sec:tier-C}`, `\S\ref{sec:tier-D}`, `\S\ref{sec:tier-E}` —
  cited in the milestone descriptions for M3-4 (Tier-C), M3-6 (D.1
  KH falsifier and D.4 Zel'dovich pancake context), and M3-8 (Tier-E
  stress tests). All pre-existing labels added in the §10 revision.
- `\S\ref{sec:tier-M3-gates}` — cited in the
  `\paragraph{Language and substrate.}` paragraph and in M3-3.
  Pre-existing label.
- `\S\ref{sec:remap-liouville-diag}` — cited in M3-5 description for
  the Liouville monotone-necessary diagnostic. Pre-existing label
  (added in the §6 revision).

No cross-reference into the new `\label{sec:milestones}` was added in
this pass; the §10 references to "Milestone 1 / 2 / 3" elsewhere in
the paper continue to resolve to the prose nouns rather than the
numbered subsection (the `\ref{sec:milestones}` would render as
"§11.3", which is less informative than the prose "Milestone 1").

## PDF rebuild

```bash
cd specs && \
  pdflatex -interaction=nonstopmode 01_methods_paper.tex && \
  pdflatex -interaction=nonstopmode 01_methods_paper.tex
```

(The paper uses an inline `\begin{thebibliography}{99}` block, not a
`.bib` file; bibtex is not part of the build sequence.)

Output: `01_methods_paper.pdf` (28 pages, ≈516 kB; was 26 pages
before this revision).

The +2-page increase distributes as:

- ≈half page from the new `\paragraph{Language and substrate.}` block
  (Julia / HG / r3djl / IntegerLattice / sparse-AD framing).
- ≈1.5 pages from the expanded `description` list (11 items at
  one paragraph each, with embedded numerics + `\verb|...|`
  pointers, against the original 5-item ≈1-line-each list).

The only LaTeX warning is the pre-existing
`Reference 'sec:appendix-soft' on page 6 undefined`. No new warnings.
The hyperref `Token not allowed in a PDF string (Unicode)` warnings
and the `OMS/cmtt/m/n` font-shape warning are pre-existing (verified
by stashing the §11 changes and rebuilding from HEAD: identical
warning set).

New labels registered in the `.aux`:

```
\newlabel{sec:milestones}{...}
```

## Files touched

- `specs/01_methods_paper.tex` — single-token edit at line 1112
  (Rust/Julia → Julia); full rewrite of §11.3 milestones block
  (≈90 lines added against ≈18 lines removed); one new bibliography
  entry. Net ≈+95 lines.
- `specs/01_methods_paper.pdf` — rebuilt (26 → 28 pages).
- `specs/01_methods_paper.aux`, `.toc`, `.out`, `.log` — regenerated
  by the build.
- `reference/notes_M3_9_prep_paper_section11_revision_applied.md` —
  this file.

## Items not applied / scope limits

- **§11 `\subsection{Outlook}` left intact.** The 3D / cosmological /
  MHD / hybrid kinetic-fluid extensions remain as outlook items. The
  3D extension is *also* now an in-scope code milestone (M3-7), but
  the structural-mathematical content of "Outlook §3D extension" is
  the right level of detail for the methods paper; no edit needed.
- **§11.1 `\subsection{What this framework provides}` and §11.2
  `\subsection{What this framework does not yet provide}` left
  unchanged.** They contain no language-or-implementation-specific
  claims. The §11.2 mentions Berry-off-diagonal status with a
  forward reference to "M3 Phase 9" in the original phrasing
  ("this is needed before 2D Kelvin-Helmholtz tests in M3 Phase 9");
  this should arguably be updated to "M3-6 Phase 1" since that
  phase has now landed and the off-diagonal sector is reactivated,
  but the brief explicitly limited the revision to §11.3 and left
  §11.1 / §11.2 alone unless cross-references break. They don't,
  so the Berry-off-diagonal claim in §11.2 is preserved — it is not
  factually wrong (just dated by the M3-6 close), and the §10.7
  M3 implementation gate results subsection records the actual M3-6
  Phase 0/1/2 results.
- **Abstract not touched.** Abstract makes no language or milestone
  claim.
- **§1 introduction not touched.** It lists the section roadmap but
  never references "Rust", "Julia", or specific milestones; the
  §-level cross-references are unchanged.
- **No equation / theorem / figure changes.** Pure schedule-text
  revision.

## TODO / follow-ups

- §11.2 "What this framework does not yet provide" item 1 mentions
  "this is needed before 2D Kelvin-Helmholtz tests in M3 Phase 9".
  M3-6 Phase 1 has now landed the off-diagonal Cholesky-Berry sector
  and the D.1 KH falsifier passed (broadband, mesh-converged at L4→L5
  to 1.2%; γ/γ_DR = 1.34). A follow-up §11.2 sub-revision should
  promote the Berry-off-diagonal item from "sketched" to "wired and
  KH-verified", and either retire the bullet or rephrase it as
  "the off-diagonal sector is now wired through M3-6 Phase 1; an
  outlook item for the 3D extension".
- The §11.3 closing paragraph ("design-doc M_n → code-M_m mapping")
  could stand to be a small table rather than a sentence. Not done
  here; would be ≈4 rows.
- §10.5 cites D.7 / D.10 as "M3-6 Phase 3+" implicitly; the new
  §11.3 milestones are explicit about Phase 3 = D.7 dust trapping
  in vortices, Phase 4 = D.7 close, Phase 5 = D.10 ISM-like
  metallicity. A future §10 revision could match the §11
  numbering once Phase 3 lands.
- The single retained "Rust" mention (the historical-context
  sentence) should probably stay until the methods paper is
  submitted; once submitted, a final pass could either retire it
  or fold it into a footnote / acknowledgment of the original
  design intent.

## Pointers

- `reference/MILESTONE_1_STATUS.md` — M1 close report, 1504+1
  deferred tests, B.2 = 2.57e-8 Zel'dovich match.
- `reference/MILESTONE_2_STATUS.md` — M2 close report, B.3 cell
  savings, B.6 multi-tracer numerics, M2-3 realizability lifting
  the wave-pool stable horizon to 12,000 steps.
- `reference/MILESTONE_3_STATUS.md` — M3 close-through-Phase-2
  status, ~24,669+1 deferred test count, M3-3e cache_mesh
  retirement ledger, M3-6 Phase 0/1/2 close blocks.
- `reference/notes_M3_3_2d_cholesky_berry.md` — M3-3 design note.
- `reference/notes_M3_4_tier_c_consistency.md` — M3-4 Tier-C close.
- `reference/notes_M3_5_bayesian_remap.md` — M3-5 close.
- `reference/notes_M3_6_phase0_offdiag_beta.md`,
  `notes_M3_6_phase1c_D1_kh_falsifier.md`,
  `notes_M3_6_phase2_D4_zeldovich.md` — M3-6 close blocks.
- `reference/notes_M3_7_3d_extension.md` — M3-7 design note.
- `reference/MILESTONE_3_PLAN.md` — global M3 plan (HG-adapted).
- `reference/notes_HG_design_guidance.md` — HG substrate guidance.
- `reference/notes_M3_3_prep_paper_section10_revision_applied.md`
  — pattern this note mirrors.
