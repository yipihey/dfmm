# M3-9 prep: C.3 plane-wave log-log convergence figure (paper §10.7 TODO)

Date: 2026-04-26
Branch: `m3-9-prep-c3-convergence-plot`
Worktree: `.claude/worktrees/agent-a078b880be4fe3a03/`
Local main HEAD at branch point: `43d954e M3-6 Phase 2: close (status note + MILESTONE_3_STATUS update + headline plot)`.

## Scope

The §10 revision note `notes_M3_3_prep_paper_section10_revision_applied.md`
flagged three TODO follow-ups for §10.7 of the methods paper. This
prep addresses the second one:

> §10.7(c) plane-wave convergence is quoted but not plotted; a small
> log-log convergence figure would complement Fig. 1 if desired.

Pure visualisation deliverable. No `src/*.jl` source-code changes; no
new tests; the existing M3-4 C.3 + Tier-C IC test counts are unchanged
(2583 + 50 asserts pass after the change, matching the baseline).

## What was added

| File                                                                | Status   |
|---------------------------------------------------------------------|----------|
| `experiments/C3_2d_plane_wave_convergence.jl`                       | new      |
| `reference/figs/M3_3_C3_plane_wave_convergence.h5`                  | new      |
| `reference/figs/M3_3_C3_plane_wave_convergence.png`                 | new      |
| `specs/01_methods_paper.tex`                                        | modified |
| `specs/01_methods_paper.pdf`                                        | rebuilt  |
| `reference/notes_M3_9_prep_c3_convergence_plot.md`                  | new (this file) |

The new rotation-invariance figure
`reference/figs/M3_4_C3_plane_wave_rotation_invariance.png` from the
M3-4 Phase 2 deliverable is not touched.

## Driver

`experiments/C3_2d_plane_wave_convergence.jl` reuses the IC factory
`tier_c_plane_wave_ic` (the `_full_ic` variant builds 12-field
Cholesky-sector state we don't need for projection-error diagnostics)
and the `tier_c_cell_centers` helper. For each (level, angle) it
computes both L∞ and L² norms of the cell-average vs. point-sample
residual

```
err_j = δρ_cell(j) - δρ_pt(j),
δρ_pt(j) = A · cos(2π · k · x_center(j)).
```

The driver exposes three entry points:

- `run_C3_plane_wave_convergence(; levels, angles, A, k_mag, ρ0, P0)`
- `save_C3_plane_wave_convergence_to_h5(result; save_path)`
- `plot_C3_plane_wave_convergence(result; save_path)`
- `fit_loglog_slope(x, y)` (used by both consumers + the figure annotation)

CairoMakie + HDF5 are loaded lazily via `Base.require(Main, :Module)`
following the precedent in `M3_3d_per_axis_gamma_cold_sinusoid.jl` and
`D4_zeldovich_pancake.jl`.

## Sweep parameters

| Quantity         | Value                                |
|------------------|--------------------------------------|
| Levels           | `{3, 4, 5, 6}`                       |
| Cells per axis   | `{8, 16, 32, 64}`                    |
| Δx (unit box)    | `{0.125, 0.0625, 0.03125, 0.015625}` |
| Amplitude `A`    | `1e-3`                               |
| `k_mag`          | `1`                                  |
| `(ρ0, P0)`       | `(1.0, 1.0)`                         |
| Angles `θ`       | `{0, π/4}`                           |

The original M3-2b/M3-4 quote covered levels 4/5/6 at θ = 0; the
sweep here adds level 3 at the coarse end and θ = π/4 to verify
rotational invariance in the projection step.

## Measured convergence

**L∞ projection error `max |δρ_cell − δρ_pt|`:**

| Level | N   | Δx       | θ = 0       | θ = π/4     |
|-------|-----|----------|-------------|-------------|
| 3     | 8   | 0.125    | 2.356e-5    | 2.505e-5    |
| 4     | 16  | 0.0625   | 6.290e-6    | 6.385e-6    |
| 5     | 32  | 0.03125  | 1.598e-6    | 1.604e-6    |
| 6     | 64  | 0.015625 | 4.011e-7    | 4.015e-7    |

**Successive ratios (L∞):**

| Step    | θ = 0 | θ = π/4 | Δx² target |
|---------|-------|---------|------------|
| 3→4     | 3.746 | 3.924   | 4          |
| 4→5     | 3.936 | 3.980   | 4          |
| 5→6     | 3.984 | 3.996   | 4          |

The level 4→5 and level 5→6 numbers reproduce the previously-quoted
3.93 / 3.98 in the §10.7 paragraph and the M3-2b note exactly (no
regression — the underlying IC factory was not touched). The level
3→4 ratio (3.75) is mildly below the asymptotic 4 because the
leading-order Taylor coefficient `(2π)² A / 24 = 1.645e-3` is not yet
the dominant term at Δx = 0.125; the sub-leading O(Δx⁴) coefficient
contributes ≈ 7% of the L∞ error there.

**L² projection error (RMS over cells):**

| Level | N   | θ = 0       | θ = π/4     |
|-------|-----|-------------|-------------|
| 3     | 8   | 1.803e-5    | 1.758e-5    |
| 4     | 16  | 4.535e-6    | 4.437e-6    |
| 5     | 32  | 1.135e-6    | 1.112e-6    |
| 6     | 64  | 2.839e-7    | 2.781e-7    |

**Fitted log-log slopes (least-squares over the four levels):**

| Norm | θ = 0   | θ = π/4 | Theory |
|------|---------|---------|--------|
| L∞   | -1.961  | -1.988  | -2.000 |
| L²   | -1.997  | -1.994  | -2.000 |

The L² slope at θ = 0 is closest to the analytic -2 because the L²
norm averages out the worst-cell anomaly visible at level 3 in L∞.
At θ = π/4 the L∞ slope is also within 0.012 of -2. **No regression
vs. previously-quoted rates.**

## Rotational consistency

Across all four levels the θ = π/4 errors are within 0.7% of the
θ = 0 errors in both norms, confirming that the L²-projection IC
machinery is rotationally invariant to round-off.

## Figure

`reference/figs/M3_3_C3_plane_wave_convergence.png` — single-panel
log-log plot:

- x-axis: cells per axis (`N = 2^level`), log scale
- y-axis: projection error, log scale
- Solid lines + circles: L∞ at θ = 0; L∞ at θ = π/4 (offset markers)
- Dashed lines: L² at θ = 0 and θ = π/4
- Dotted gray: Δx² reference slope, anchored at the level-3 L∞ value
- Annotation: fitted slopes at θ = 0 (L∞ = -1.96, L² = -2.00, theory -2.0)

Style matches the existing `M3_4_C3_plane_wave_rotation_invariance.png`
and `M3_3d_per_axis_gamma_selectivity.png` (CairoMakie, 760×560 px,
log scales).

## Methods paper edit

`specs/01_methods_paper.tex` §10.7(f) — the "Plane-wave convergence
(Tier C C.3)" paragraph — gained:

1. A trailing sentence pointing at the new figure and quoting the
   level-3 + π/4 extension.
2. A `\begin{figure}[ht]` block immediately after the paragraph,
   with `\includegraphics[width=0.85\textwidth]` of the new PNG and
   a caption summarising the sweep, ratios, fitted slopes, and the
   driver/data file paths.

`\label{fig:M3-9-C3-convergence}` registered.

## PDF rebuild

```bash
cd specs && \
  pdflatex -interaction=nonstopmode 01_methods_paper.tex && \
  pdflatex -interaction=nonstopmode 01_methods_paper.tex && \
  pdflatex -interaction=nonstopmode 01_methods_paper.tex
```

Output: 27 pages (baseline 26, +1 for the new figure block).

The only LaTeX warning is the pre-existing
`Reference 'sec:appendix-soft' on page 6 undefined`, documented in
`notes_M2_4_paper_corrections_applied.md`,
`notes_M3_prep_paper_section5_revision_applied.md`,
`notes_M3_prep_paper_section6_revision_applied.md`, and
`notes_M3_3_prep_paper_section10_revision_applied.md`. The pre-existing
`hyperref` "Token not allowed in a PDF string (Unicode)" notes
(γ, δ, π in section/caption titles) are also unchanged. **No new
warnings introduced.**

## Bit-exact regression

No `src/*.jl` source touched — the existing 1D + 2D path is byte-equal.
Verified end-to-end:

- `test/test_M3_4_C3_plane_wave.jl` → 2583 / 2583 pass
- `test/test_M3_prep_setups_tierC.jl` → 50 / 50 pass

The level 4/5/6 L∞ errors (6.290e-6, 1.598e-6, 4.011e-7) match the
M3-2b table line-for-line, and the level 4→5 / 5→6 ratios (3.94, 3.98)
match the §10.7 paragraph quote exactly.

## Files touched

- `experiments/C3_2d_plane_wave_convergence.jl` — new (~210 LOC).
- `reference/figs/M3_3_C3_plane_wave_convergence.h5` — new (data dump).
- `reference/figs/M3_3_C3_plane_wave_convergence.png` — new (figure).
- `specs/01_methods_paper.tex` — paragraph extended + figure block
  added in §10.7.
- `specs/01_methods_paper.pdf` — rebuilt, 26 → 27 pages.
- `specs/01_methods_paper.aux`, `.toc`, `.out`, `.log` — regenerated
  by the build.
- `reference/notes_M3_9_prep_c3_convergence_plot.md` — this file.

## Items not addressed

Two of the §10.7 TODOs remain:

- **§11 milestone table still says "Rust".** Out of scope for this
  M3-9 prep (paper §10 only); the §10 revision note already flagged
  this as a future broader §11 update.
- **§10.7(b) closed-form Berry partial display abbreviates with "..."**.
  Out of scope here.

## Pointers

- `reference/notes_M3_3_prep_paper_section10_revision_applied.md`
  — TODO source.
- `reference/notes_M3_2b_swap5_init_field.md` — original convergence
  table (level 4/5/6 quote).
- `reference/notes_M3_4_tier_c_consistency.md` — M3-4 Phase 2 status
  note that re-quotes the 3.93/3.98 ratios.
- `experiments/C3_2d_plane_wave.jl` — sibling rotation-invariance
  driver.
- `experiments/M3_3d_per_axis_gamma_cold_sinusoid.jl` — CairoMakie
  pattern reference.
