# Milestone 3 implementation plan — multi-D dfmm on HierarchicalGrids.jl

> **Status (2026-04-26):** Adapted from the original M3 plan after
> `yipihey/HierarchicalGrids.jl` (HG, v0.1) was published. HG provides
> the dimension-generic mesh + AMR + polynomial-fields + remap +
> threading substrate the original plan was going to build from
> scratch, and enables a unified 1D / 2D / **3D** codebase from day 1.
>
> The original 2D-only plan is preserved as
> `MILESTONE_3_PLAN_legacy_pre_HG.md` for audit. HG-specific design
> guidance is in `notes_HG_design_guidance.md`.

## Scope

A dimension-generic Julia implementation of the unified variational
dfmm framework on top of HG's `HierarchicalMesh{D}` + `SimplicialMesh{D, T}`,
running in 1D, 2D, and 3D from the same source. Targets methods paper
§10's Tier-C (consistency), Tier-D (novelty), and most of Tier-E
(stress) tests across all three dimensions.

**In scope:**
- Refactor M1's 1D solver onto HG (verify 1D parity to 5e-13).
- Add 2D principal-axis Cholesky + Berry connection (per derivation in
  `notes_M3_phase0_berry_connection.md`).
- Bayesian remap via `compute_overlap` + `polynomial_remap`.
- Action-based AMR via `refine_by_indicator!`.
- Tier C consistency tests (1D-symmetric reductions in 2D/3D).
- Tier D novelty (KH, Zel'dovich pancake, dust traps, ISM tracers).
- 3D extension (SO(3) Berry connection — derivation extends naturally).
- Tier E stress tests + GPU/MPI distribution.

**Out of scope (subsequent milestones):**
- Self-gravity (cosmological multigrid Poisson on the quadtree/octree).
- MHD / magnetic fields.
- Hybrid kinetic-fluid coupling.

## Reframing vs methods paper §10.7

The methods-paper schedule has **M1 (1D) → M2 (1D Tier B) → M3 (2D
Tier C) → M4 (2D Tier D) → M5 (Tier E + applications)**. With HG, the
engineering unit shrinks to a single dimension-generic codebase:

- **M1 (1D, `Mesh1D`-based):** DONE — 2044+1 deferred tests on `main`.
- **M2 (1D Tier B closure):** DONE — same code, B.3/B.6 added.
- **M3-α (refactor):** port M1's solver onto HG; verify 1D parity.
- **M3 (2D Tier C + D):** same solver, 2D mesh; Berry connection live.
- **M3-β (3D Tier C + D):** same solver, 3D mesh; SO(3) Berry.
- **M4 (Tier E + applications):** multi-D physics tests.

This collapses methods-paper M3 + M4 + most of M5 into one engineering
effort because HG already supplies mesh / AMR / remap / threading.

## Repository layout (post-refactor)

```
src/
├── dfmm.jl                   # entry; using HierarchicalGrids, R3D, ...
├── eos.jl                    # M_vv(J, s) — Track C carry-over
├── stochastic.jl             # VG sampler — Track D carry-over
├── stochastic_injection.jl   # post-Newton mutator — M1 P8 + M2-3
├── tracers.jl                # passive-tracer fields (replaced by HG PolynomialFieldSet)
├── diagnostics.jl            # γ_a, |s|, det Hess — Track C carry-over
├── calibration.jl            # NPZ loader — Track C carry-over
├── plotting.jl               # CairoMakie helpers — Track C carry-over
├── io.jl                     # HDF5/JLD2 — Track C carry-over
├── regression.jl             # Tier-A golden loader — scaffold carry-over
├── setups.jl                 # Tier A + C IC factories — generalize to D
│
├── types.jl                  # ChField{D,T}, DetField{D,T} — d-generic
├── cholesky_DD.jl            # principal-axis decomp, Berry coupling
├── eom.jl                    # discrete EL residual on HG mesh
├── newton_step.jl            # Newton wrapper using HG sparsity
├── deviatoric.jl             # Π relaxation
├── heat_flux.jl              # Q Lagrange multiplier
├── artificial_viscosity.jl   # tensor-q opt-in (M1 P5b)
├── action_error.jl           # ΔS_cell indicator for HG's refine_by_indicator
├── action_amr_helpers.jl     # M2-1 carry-over: Cholesky merger via law-of-total-covariance
└── (no more Mesh1D / Segment / amr_1d.jl — replaced by HG)
```

The post-Newton mutators (`stochastic_injection`, `tracers`,
`deviatoric`, `heat_flux`, `artificial_viscosity`) carry over almost
unchanged — they operate on per-cell fields, which become HG's
`PolynomialFieldSet` entries.

## Phase plan

### Phase M3-0 — HG + r3djl integration + 1D parity

**Goal.** Add HG and r3djl as deps. Build a thin shim that runs M1's
1D Phase-1 zero-strain Cholesky test on a `SimplicialMesh{1, Float64}`
+ `PolynomialFieldSet` instead of `Mesh1D`+`Segment`. Verify 1D parity
with M1's bit-exact tests.

**Files.**
- `Project.toml` — add `HierarchicalGrids` (URL), `R3D` (URL).
- `src/types.jl` — refactor `ChField{T}` to `ChField{D,T}`; in 1D this
  is a single (α, β); in 2D it's $(\alpha_a, \beta_a, \theta_R)$; in 3D
  $(\alpha_a, \beta_a, \theta_{ab})$.
- `src/eom.jl` — port `cholesky_el_residual` to take HG meshes.
- `test/test_M3_0_parity_1D.jl` — re-run M1 Phase 1's three tests
  on the HG-based code; assert byte-equal output to 5e-13.

**Acceptance.** All M1 Phase 1 tests pass on the HG-based code at the
same tolerances. Total wall time within 2× of M1's baseline.

**Wall time.** 1-2 weeks.

### Phase M3-1 — Phase 2 + 5 + 5b ports (1D)

**Goal.** Port M1's Phase 2 (bulk + entropy), Phase 5 (deviatoric),
Phase 5b (tensor-q) onto HG. Same physics, HG mesh.

**Acceptance.** All M1 Phase 2/5/5b tests pass.

**Wall time.** 1 week.

### Phase M3-2 — Phase 7/8/11 + M2 ports (1D)

**Goal.** Port heat-flux Q (Phase 7), stochastic injection (Phase 8),
realizability projection (M2-3), tracers (Phase 11), action-AMR
(M2-1) onto HG. Phase 11 plugs into HG's `PolynomialFieldSet`
directly (tracers are just additional fields; HG's bit-exact
preservation mirrors dfmm's).

**Note on M2-1 action-AMR.** HG's `refine_by_indicator!` replaces
dfmm's hand-rolled `amr_step!`. The **conservative refine/coarsen
operators** (mass-weighted Cholesky merge via law-of-total-covariance)
become callbacks registered with HG (per design-guidance item #5
in `notes_HG_design_guidance.md`; until that exists, we do the
bookkeeping in dfmm's driver layer).

**Acceptance.** All M1 + M2 1D tests pass on HG.

**Wall time.** 2 weeks.

### Phase M3-3 — 2D Cholesky + Berry connection

**Goal.** Add the 2D principal-axis decomposition + Berry-connection
coupling per `notes_M3_phase0_berry_connection.md`. The discrete EL
system grows from 4N (M1) to 5N per cell:
$(\alpha_a, \beta_a, \theta_R)$. Berry term

$$\Theta_{\rm rot} = \tfrac{1}{3}\bigl(\alpha_1^3\beta_2 - \alpha_2^3\beta_1\bigr)\,d\theta_R$$

in the per-cell Lagrangian.

**Files.** `src/cholesky_DD.jl` — per-axis decomposition + Berry term.
`src/eom.jl` — extends EL residual to include Berry coupling.
`src/types.jl` — `DetField{2, T}` with the new fields.

**Acceptance.** Verification document's CHECKs 1-7 (closedness,
per-axis match, iso pullback, kernel structure) reproduce in the
implementation. Per-axis γ diagnostic is correctly computed.

**Wall time.** 2 weeks.

### Phase M3-4 — Tier C consistency tests

**Goal.** Run the 2D code in 1D-symmetric configurations; verify it
bit-equals (or matches to round-off) the 1D results from M3-2. Methods
paper §10.4 Tier C.

- C.1 1D-symmetric 2D Sod (y-periodic).
- C.2 1D-symmetric 2D cold sinusoid (per-axis γ correctly identifies
  the collapsing axis: $\gamma_1 \to 10^{-5}$, $\gamma_2 \to O(1)$).
- C.3 Plane wave at arbitrary direction (rotational invariance).

**Files.** `experiments/C1_*.jl` etc.; tests under `test/test_M3_4_C*.jl`.

**Acceptance.** C.1: y-independence to 1e-12. C.2: per-axis γ
selectivity. C.3: rotational invariance to plotting precision.

**Wall time.** 2-3 weeks.

### Phase M3-5 — Bayesian remap

**Goal.** Wire `compute_overlap` + `polynomial_remap` into the
per-step driver. Call once per AMR step (or every K steps as
configured). Verify mass / momentum / energy / tracer conservation
to round-off through remap.

**Acceptance.** `total_overlap_volume(ov) ≈ box_volume` (round-off);
`polynomial_remap` round-trips conserved quantities to round-off;
Liouville monotone-increase diagnostic enabled (per HG design-guidance
item #8).

**Wall time.** 1-2 weeks.

### Phase M3-6 — Tier D headlines (the methods-paper deliverables)

The headline scientific results, per methods paper §10.5:

- **D.1 2D KH** — falsifiable per-axis noise prediction. Via M2-3's
  realizability projection + Phase 8's stochastic injection on
  per-axis decomposition.
- **D.4 2D Zel'dovich pancake** — central novel test. Per-axis γ
  correctly identifies pancake-collapse direction; stochastic
  regularization on the compressive axis only.
- **D.7 2D dust-trapping** — realistic test (vortex-center dust
  accumulation matches reference codes).
- **D.10 ISM-like 2D multi-tracer** — community-impact test.
  Multi-tracer fidelity in 2D shocked turbulence.

**Wall time.** 4-6 weeks. **Risk.** The compression-cascade we
diagnosed in M2-3 will resurface at higher Mach in 2D; expect to
iterate on the realizability projection.

### Phase M3-7 — 3D extension

**Goal.** With the dimension-generic code, run the 2D Tier-C/D
headlines in 3D. The Berry connection extends to SO(3) per
`notes_M3_phase0_berry_connection.md` §8 step 2:

$$\Theta_{\rm rot}^{3D} = \tfrac{1}{3}\sum_{a<b}\bigl(\alpha_a^3\beta_b - \alpha_b^3\beta_a\bigr)\,d\theta_{ab}$$

Run 3D Zel'dovich pancake collapse (the cosmological reference test).
3D KH instability. 3D dust traps in turbulent boxes.

**Wall time.** 3-5 weeks.

### Phase M3-8 — Tier E stress tests + GPU / MPI distribution

**Goal.** Per methods paper §10.6 (Tier E):
- E.1 high-Mach 2D shocks (graceful failure).
- E.2 severe shell-crossing geometries.
- E.3 very low Knudsen.
- E.4 cosmological IC (with self-gravity — defer if not in scope).

Plus GPU port via HG's threading layer (KernelAbstractions.jl
backend) and MPI scaling via HG's chunk structure.

**Wall time.** 4-6 weeks (mostly GPU porting).

### Phase M3-9 — Wrap-up

`reference/MILESTONE_3_STATUS.md`; performance comparison; paper
figures; methods-paper revision pass.

**Wall time.** 2-3 weeks.

## Total estimated wall time

| Phase block | Time |
|---|---|
| M3-0 + M3-1 + M3-2 (1D refactor) | 4-5 weeks |
| M3-3 + M3-4 (2D + Tier C) | 4-5 weeks |
| M3-5 + M3-6 (Bayesian remap + Tier D) | 5-8 weeks |
| M3-7 (3D extension) | 3-5 weeks |
| M3-8 + M3-9 (Tier E + GPU/MPI + wrap-up) | 6-9 weeks |
| **Total** | **~22-32 weeks (~5-8 months)** |

Comparable to the original M3 estimate (~5-7 months), but now
includes 3D **and** has all the substrate work done by HG.

## File ownership map

For parallel-agent dispatch (per the M1 successful pattern):

| File | Phase | Notes |
|---|---|---|
| `Project.toml`, `Manifest.toml` | M3-0 | Add HG + R3D deps |
| `src/types.jl` | M3-0, M3-3 | Dim-generic field types |
| `src/eom.jl` | M3-0..M3-3 | Discrete EL residual on HG mesh |
| `src/cholesky_DD.jl` | M3-3 | Per-axis decomp + Berry coupling |
| `src/setups.jl` | M3-4 | Tier-C IC factories (extend existing) |
| `src/action_error.jl` | M3-2 | ΔS_cell indicator |
| `src/action_amr_helpers.jl` | M3-2 | Conservative refine/coarsen callbacks |
| `experiments/Cn_*.jl`, `experiments/Dn_*.jl` | M3-4, M3-6, M3-7 | Per-test drivers |
| `test/test_M3_n_*.jl` | per-phase | Test scaffolding |
| `reference/notes_M3_n_*.md` | per-phase | Per-phase notes |

## What we keep from M1/M2 unchanged

These files transfer to the HG-based code essentially as-is:

- `src/eos.jl` — pure functions on $(J, s)$.
- `src/diagnostics.jl` — pure functions on per-cell state.
- `src/stochastic.jl` (Track D) — pure functions on samples.
- `src/calibration.jl` — pure I/O.
- `src/plotting.jl` — CairoMakie helpers.
- `src/io.jl` — HDF5/JLD2 wrappers (extend for HG mesh save/load).
- `src/regression.jl` — Tier-A golden loader.

These need adaptation but the physics is the same:

- `src/stochastic_injection.jl` — operates on `PolynomialFieldSet`
  per-cell instead of `Mesh1D` segments.
- `src/tracers.jl` — replaced by `PolynomialFieldSet` of tracer fields.
- `src/deviatoric.jl`, `src/heat_flux.jl`, `src/artificial_viscosity.jl`
  — same physics, HG mesh.

These are **discarded** (replaced by HG):

- `src/segment.jl` — superseded by `SimplicialMesh{1, T}`.
- `src/amr_1d.jl` — superseded by `refine_by_indicator!` + callbacks.
- M1 Phase-1's hand-rolled mesh + Newton scaffolding — now uses HG's
  `parallel_for_cells` + sparsity API.

## Verification gates

End-to-end checks at each phase boundary:

1. **M3-0 parity gate.** All M1 Phase 1 tests pass on HG-based code
   to 5e-13. If not, HG's polynomial-field representation has a bug
   relative to the M1 hand-rolled mesh; debug before continuing.

2. **M3-2 full M1+M2 parity.** Full M1 + M2 Tier-A/B regression
   passes on HG-based code (1D). 2044 + 1 deferred tests reproduced.

3. **M3-4 dimension-lift gate.** C.1, C.2, C.3 pass. If 2D code can't
   reduce cleanly to 1D, the Berry-connection implementation has a
   bug; debug before Tier-D.

4. **M3-5 Bayesian remap conservation.** Mass / momentum / energy /
   tracer conservation through remap to round-off; Liouville
   monotone-increase diagnostic in expected range.

5. **M3-6 Tier-D headlines.** D.1 (KH falsifier), D.4 (pancake),
   D.7 (dust traps), D.10 (ISM tracers) all pass per methods paper §10
   acceptance.

6. **M3-7 3D parity.** 3D code reduces to 2D and 1D under the
   appropriate symmetry restrictions.

## Pre-launch prerequisites

Before M3-0:

1. **Confirm HG 1D coverage.** `HierarchicalMesh{1}` + `SimplicialMesh{1, T}`
   tested? If not, contribute upstream or work around.
2. **Confirm r3djl 1D coverage.** Trivial interval intersection in 1D.
   If r3djl is 2D/3D-only, dfmm provides the 1D shim.
3. **Pin the HG + r3djl versions** (commits) in `Manifest.toml`.
4. **Berry-connection methods-paper revision.** §5.2-§5.5 needs the
   explicit $\Theta_{\rm rot}$ form + Poisson-manifold $\mathcal{H}_{\rm rot}$
   constraint from `notes_M3_phase0_berry_connection.md`. This is
   paper-text work, not blocking M3-0 code.

## Open questions to resolve at M3-0 launch

1. **Refactor depth.** Do we delete `src/segment.jl`/`amr_1d.jl`/etc.
   immediately after M3-2 parity, or keep them as a comparison
   reference for one milestone? Recommend: delete once M3-2 verifies
   parity to 5e-13 — keeping two implementations rots.

2. **GPU target.** Apple Metal (Tom's laptop) vs CUDA (production
   cluster)? HG's threading-layer abstraction makes either viable;
   first-cut Metal is recommended for dev iteration speed.

3. **3D Berry connection.** The proposed SO(3) form needs a
   verification analogous to the 2D one
   (`scripts/verify_berry_connection.py`). Run before M3-7.

4. **Off-diagonal $L_2$ sector.** Methods paper §5.5; sketched in
   `notes_M3_phase0_berry_connection.md` §7. Needed for D.1 KH where
   $\beta_{12}$ becomes dynamical. Derive before M3-6.

5. **Production-scale calibration mismatch (M1 Open #3).** The 3-λ
   inconsistency from Phase 8 may resurface in 2D. Plan to monitor
   via the self-consistency monitor in M3-6 (KH calibration step).

## Critical files referenced

- `reference/MILESTONE_1_STATUS.md` — M1 baseline (1801+1 → 2044+1).
- `reference/MILESTONE_2_STATUS.md` — M2 baseline.
- `reference/MILESTONE_3_PLAN_legacy_pre_HG.md` — pre-HG M3 plan.
- `reference/notes_HG_design_guidance.md` — HG features dfmm needs.
- `reference/notes_dfmm_2d_overview.md` — eight differences for 2D.
- `reference/notes_M3_phase0_berry_connection.md` — Berry derivation
  + SymPy verification.
- `scripts/verify_berry_connection.py` — SymPy verification driver.
- `specs/01_methods_paper.tex` §5 + §6 + §9 + §10 — design + validation.
- HierarchicalGrids.jl `docs/src/architecture.md` + `overlap.md`.
- r3djl `docs/overlap_example.md`.

## Recommended next moves

1. **Send `notes_HG_design_guidance.md` to the HG project** as a
   design-review note (issue or PR comment). Items 1, 4, 5, 7, 8 are
   useful to any downstream package; items 2, 3, 6 are dfmm-specific
   but generally needed by hyperbolic-PDE solvers.

2. **Launch M3-0** (HG + r3djl integration + 1D parity verification).
   Single agent in a worktree; ~1-2 weeks. Use the M1 lesson-learned
   protocol (worktree-fix on launch + commit-before-return).

3. **Update `notes_dfmm_2d_overview.md`**'s implementation-stack
   section to call out HG as the substrate.
