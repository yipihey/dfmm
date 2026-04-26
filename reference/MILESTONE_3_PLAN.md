# Milestone 3 implementation plan — 2D extension

**Scope.** A 2D Julia implementation of the unified variational dfmm
framework on a Lagrangian simplicial triangulation coupled to an
Eulerian quadtree, with adaptive refinement, GPU-portable kernels,
and the full Tier-C/D/E validation hierarchy.

**Out of scope.** Self-gravity (Milestone 4), 3D extension (a
different milestone), the methods paper revision pass (Tom's work).

**Authoritative references** (cite section numbers when working):

- `HANDOFF.md` — the original design handoff, scope-split table.
- `reference/notes_dfmm_2d_overview.md` — synthesis of what changes
  for 2D and what carries over from M1. **Read this first.**
- `reference/MILESTONE_1_STATUS.md` — what M1 verified + four open
  architectural questions.
- `reference/notes_methods_paper_corrections.md` — paper-edit items
  surfaced during M1; relevant when 2D paper text is written.
- `specs/01_methods_paper.tex` — canonical design.
  - §2-§3 (4×4 phase-space Cholesky + action) — 2D Cholesky structure.
  - §5 (the 2D matrix lift via principal-axis decomposition).
  - §6 (Bayesian remap with the law of total cumulants).
  - §7 (passive scalar advection: exact at the discrete level).
  - §9 (the discrete scheme — mesh, polynomial reconstruction, EL,
    stochastic injection, action-based AMR, algorithm summary).
  - §10 (validation hierarchy and acceptance criteria).
- `design/03_action_note_v2.tex` §3 (deterministic action), §4
  (stochastic dressing).
- `design/04_action_note_v3_FINAL.pdf` (variance-gamma; supersedes v2).
- `specs/05_julia_ecosystem_survey.md` §13 (the implementation stack).

> **Status (2026-04-25):** Not started. M1 substantially complete
> (1801 + 1 deferred tests on main). M3 launches once Tom decides
> to proceed; recommended prerequisites listed at end of this plan.

The 2D symplectic form (methods paper §5.4):

> $\omega_{2D} = \sum_{a=1}^{2} \alpha_a^2\,d\alpha_a \wedge d\beta_a + \omega_{\rm rot}$

The two diagonal terms are the per-axis 1D analog (each verified by
M1). The Berry connection $\omega_{\rm rot}$ is *the open theoretical
work* — derive before coding (Phase 0 below).

---

## Phase ordering rationale

The M1 handoff's "Three things you should NOT do" generalize for M3:

1. **Don't bolt the principal-axis decomposition on top of a
   working 2D non-axis scheme.** Build the principal-axis structure
   foundationally; everything else derives.
2. **Test the dimension-lift consistency (Tier C) early.** If the 1D
   limits don't reduce cleanly in the 2D code, the framework's
   axis-decomposition is broken and the headline claims fail.
3. **Don't skip the Berry-connection derivation.** Coding without
   it produces a scheme that decouples the principal axes
   incorrectly; the 2D-novel claims (D.1, D.4, D.7, D.10) all rely
   on the rotation-coupling being right.

Phases 0-2 form the "2D variational core works" gate. Phases 3-5
are Tier-C consistency. Phases 6-8 add infrastructure (remap, AMR,
GPU). Phases 9-13 are Tier-D headlines. Phase 14 is wrap-up.

A reasonable estimate (HANDOFF.md "Estimated time"): **~4-5 months
total** at agent cadence; the bottlenecks are the Berry-connection
derivation (Phase 0; theory work) and Tier-D.4 pancake collapse
(Phase 10; the central novel test).

---

## Phase 0 — Foundations

**Status:** *theoretical + reading work; no code yet.*

**Goal.** Derive the Berry connection $\omega_{\rm rot}$ for the
2D action, verify the dimension-lift reduction to 1D analytically,
read the M1 implementation thoroughly, and confirm the r3d Julia
port is ready.

### Theoretical deliverables (must precede Phase 1)

1. **Derive $\omega_{\rm rot}$.** The structural form: a 2-form on
   the principal-axis $GL(2)$ bundle parameterized by the rotation
   angle $\theta_R$. Two natural candidates from gauge theory:
   - Pure-curvature form: $\omega_{\rm rot} = c\,\alpha_1\alpha_2\,d\theta_R \wedge d(\alpha_1\beta_2 - \alpha_2\beta_1)$.
   - Geometric-phase form: $\omega_{\rm rot}$ as the canonical Berry curvature on $SO(2) \subset GL(2)$.
   Pick by requiring (a) reduction to the diagonal terms when
   $\alpha_1 = \alpha_2$ and the rotation decouples, and (b)
   gauge-invariance under the full strain group $GL(2)$.
2. **Verify dimension-lift consistency.** Setting $\beta_2 = 0$,
   $\alpha_2 = $ const (no $y$-dynamics) must reduce the 2D action
   to the 1D action verified in M1 Phase 3.
3. **Cold-limit reduction in 2D.** With $\gamma_a \to 0$ for both
   $a = 1, 2$, the action collapses to the mass-coordinate dust
   action $\int \tfrac12 \dot{\bm x}^2$. Verify analytically.
4. **Hessian-degeneracy structure in 2D.** The 2D Hessian's
   rank-defect signals at the caustic; two distinct degeneracy
   directions corresponding to the two principal axes.

**Output:** `reference/notes_phase0_M3_berry_connection.md`
documenting the derivation with explicit formulas. Ideally also a
methods-paper §5.5 patch (or a new appendix) capturing the result
for publication.

### Reading deliverables

1. Re-read M1's `reference/notes_phaseA0.md`,
   `notes_phase2_discretization.md`, `notes_phase3_solver.md`,
   `notes_phase5_sod_FAILURE.md`,
   `notes_phase5b_artificial_viscosity.md`,
   `notes_phase8_stochastic_injection.md`. Internalize the 1D
   findings.
2. Read `specs/01_methods_paper.tex` §5 (matrix lift) end-to-end.
3. Read Trixi.jl source: `src/callbacks_step/amr.jl`,
   `src/solvers/dg_p4est.jl`, `IndicatorHennemannGassner` (per
   `specs/05` §2.2).
4. Read p4est paper (Burstedde-Wilcox-Ghattas 2011, SIAM J. Sci.
   Comput.) to understand the quadtree data structure.
5. Skim Caramana-Shashkov-Whalen 1998 §4 (tensor-q in 2D) and
   Powell-Abel 2015 (r3d original paper).

### Code prerequisites (verify, don't write)

- `r3d` Julia port available as a registered package or Pkg.add'able.
  Confirm signature: `polygon_polygon_overlap_with_moments(P1, P2, max_degree)` returning the overlap polygon plus a vector of polynomial moments up to degree 6 (cubic reconstruction means up-to-6th-degree integrals).
- M1 `src/eos.jl`, `src/stochastic.jl`, `src/diagnostics.jl`,
  `src/regression.jl`, `src/calibration.jl` — confirm they don't
  bake in 1D-specific assumptions. (Spot-check; most should be
  scalar/per-cell.)

**Exit gate.** The agent can write $\omega_{\rm rot}$ from memory,
explain why $\beta_2 = 0$ reduces 2D to 1D, and point to the r3d
package on GitHub.

---

## Phase 1 — Per-triangle Cholesky-sector variational integrator

**Goal.** A standalone Julia module implementing the 2D
Hamilton-Pontryagin integrator for the principal-axis Cholesky
sector $(\alpha_a, \beta_a, \theta_R)$ alone, with $\gamma_a$
supplied externally. Analog of M1 Phase 1; verifies the 2D
variational structure produces the right axis-decoupled equations
plus the rotation coupling from $\omega_{\rm rot}$.

**Files to create:**

- `src/types_2d.jl` — `TriField` struct (per-triangle state:
  $x_1, x_2, u_1, u_2, \alpha_1, \alpha_2, \beta_1, \beta_2,
  \theta_R, s, $ heat flux components, deviatoric stress, etc.).
- `src/triangle_mesh.jl` — `TriangleMesh` data structure: vertex
  positions, triangle topology (triplets), face adjacency,
  per-triangle field storage.
- `src/cholesky_2d.jl` — 2D Hamilton-Pontryagin Lagrangian, EL
  residual on the per-triangle 5-DOF Cholesky sector
  $(\alpha_1, \alpha_2, \beta_1, \beta_2, \theta_R)$.
- `src/discrete_transport_2d.jl` — discrete covariant derivative
  $\mathcal{D}_t^{(q,a)}$ per principal axis; midpoint stencil per
  methods paper §9.5.
- `src/newton_step_2d.jl` — multi-DOF Newton wrapper; sparse
  Jacobian via `SparseConnectivityTracer` + triangle adjacency
  graph.

**Tests:**

1. **Single-triangle zero-strain free evolution** — analog of
   M1 `test_phase1_zero_strain.jl`. Closed-form trajectory under
   the autonomous Cholesky Hamiltonian; integrator matches to
   $10^{-12}$.
2. **1D-symmetric reduction.** Set $\beta_2 = 0$ initially with
   uniform $y$-direction; integrator output matches 1D reference
   at the corresponding mass coordinates.
3. **Pure rotation under uniform shear strain.** $\partial_x u_y =
   \omega$ const; analytical $\theta_R(t)$ rotation rate. Verify
   the Berry-connection contribution to $\dot\theta_R$.
4. **2D symplecticity check.** Stokes-on-loop-of-ICs analog of M1
   Phase 1; $\oint \theta$ along closed orbit in $(\alpha_a, \beta_a, \theta_R)$ to round-off.

---

## Phase 2 — Bulk + entropy + multi-triangle coupling

**Goal.** Add bulk fields (position, velocity, entropy) on the
triangle mesh; couple triangles through pressure-gradient face
fluxes; full deterministic action $\mathcal{L}_{\rm det}$ in 2D.

**Files:**

- Extend `src/cholesky_2d.jl` with the full action sum.
- Extend `src/triangle_mesh.jl` with face-flux assembly.
- Extend `src/newton_step_2d.jl` for the full coupled system.
- `src/discrete_action_2d.jl` — sum over triangles × timesteps.

**Tests:**

1. **Mass conservation** per triangle (Δm_T fixed by construction).
2. **Translation + rotation invariance** ⇒ momentum + angular
   momentum conservation in periodic-BC problems.
3. **Linearized acoustic wave at arbitrary direction.** Compare
   numerical phase speed to $c_s$; rotational invariance check.
4. **2D free streaming dust** (γ → 0 limit): particles ballistic
   in 2D until caustic.

---

## Phase 3 — Tier C.1: 1D-symmetric 2D Sod

**Goal.** A 2D Sod problem with $y$-periodic + $x$-direction
inflow/outflow; verify the solution is exactly $y$-independent to
numerical precision (i.e. the 2D code reduces to the 1D code we
verified in M1 Phase 5).

**Files:**

- `experiments/C1_sod_2d.jl` — driver, comparison to M1's
  `reference/golden/sod.h5`.
- `test/test_phase3_M3_sod_2d.jl` — assertions on
  $y$-independence and per-row L∞ rel error vs the 1D golden.

**Acceptance** (methods paper §10.4 C.1):

1. Solution $y$-independent to numerical precision (relative
   spread across $y$ < $10^{-12}$).
2. Per-row L∞ rel error vs the M1 1D golden < 0.05 (matching
   M1's qualitative bar; the residual ~10-20% from M1 propagates
   here unless Open #2 from M1 is addressed first).

**Risk:** if the 1D-symmetric reduction does not produce
$y$-independent output, the principal-axis decomposition has a
bug. Stop and debug — *do not proceed to Phase 4.*

---

## Phase 4 — Tier C.2: 1D-symmetric cold sinusoid axis-decomposition

**Goal.** A 2D cold sinusoid at $y$-periodic; verify per-axis γ
diagnostic correctly distinguishes the directions: $\gamma_1 \to 0$
at the shell-crossing, $\gamma_2 \to O(1)$ throughout. **This is
the central Tier-C test for the principal-axis structure.**

**Acceptance** (methods paper §10.4 C.2):

1. $\gamma_1$ at the caustic location reaches $\le 10^{-5}$.
2. $\gamma_2$ stays $\ge 0.1$ throughout (no spurious $y$-axis
   collapse).
3. Pre-crossing density profile matches Zel'dovich analytical to
   the same tolerance as M1 Phase 6 (~$10^{-4}$).

**Risk:** if $\gamma_2$ collapses spuriously, the Berry-connection
derivation is wrong or the discrete scheme's rotation coupling has
a sign error. **The Phase-0 derivation must be re-verified in this
case.**

---

## Phase 5 — Tier C.3: rotational invariance

**Goal.** A 2D plane wave at an arbitrary angle (e.g. 30° or 45°);
verify the solution is rotational-invariant to plotting precision —
i.e. rotating the mesh by an angle and re-running gives the same
result modulo the rotation.

**Tests:**

- Run the same IC at angles {0°, 15°, 30°, 45°, 60°, 90°}; all
  should produce identical primitives modulo rigid rotation.
- L∞ residual after de-rotation < $10^{-3}$ at moderate resolution.

---

## Phase 6 — Bayesian remap (Lagrangian → Eulerian)

**Goal.** Implement the periodic remap step using Tom's r3d Julia
port: triangle-quadcell overlap with polynomial-moment integration.

**Files:**

- `src/eulerian_mesh.jl` — Eulerian quadtree-cell data structure
  (initially uniform grid; AMR enters in Phase 7).
- `src/remap.jl` — `remap_lagrangian_to_eulerian!(em, lm, mode)`.
  Calls `r3d.polygon_polygon_overlap_with_moments` per
  triangle-quadcell pair; assembles cell-averaged moments.
- `src/bayesian.jl` — law-of-total-covariance application to
  the moment hierarchy (methods paper §6.3).

**Acceptance** (methods paper §10.4):

1. **Conservation.** Mass, momentum, energy preserved to round-off
   through the remap (modulo the documented Liouville monotone
   increase per methods paper §6.5).
2. **Positivity.** Cell-averaged density and Cholesky diagonals
   stay positive (Bernstein-basis positivity certificate per §9.3).
3. **Consistency.** A trivial remap (Lagrangian and Eulerian
   identical) reproduces the input bit-exactly.
4. **Tracer-exactness preservation in pure-Lagrangian regions.**
   M1 Phase 11's L∞ tracer change = 0 must continue to hold when
   no remap step is invoked.

---

## Phase 7 — Adaptive mesh refinement via p4est

**Goal.** Quadtree AMR on the Eulerian mesh, refinement triggered
by the action-error indicator $\Delta S_{\rm cell}$ per methods
paper §9.7.

**Files:**

- `src/amr.jl` — p4est wrapper, refinement/coarsening callbacks.
- `src/action_error_indicator.jl` — per-cell $\Delta S_{\rm cell}$
  (eq. 348-349 in v2 / methods paper §9.7).
- `src/amr_callbacks.jl` — Trixi-style callback architecture.

**Acceptance:**

1. **Off-center blast wave (Tier B.3 generalization to 2D).**
   Action-AMR achieves a target L² accuracy with 20-50% fewer
   cells than gradient-based AMR (the Trixi
   `IndicatorHennemannGassner` baseline).
2. **Hysteresis prevents flicker.** Refine threshold > coarsen
   threshold by factor ≥ 4 (per methods paper §9.7).
3. **Conservation through refinement.** Mass / momentum / energy
   preserved across refine/coarsen events to round-off.

**Risk:** p4est-Julia interop fragility. Allocate extra time if
P4est.jl's parallel APIs prove fragile on Apple Silicon.

---

## Phase 8 — GPU porting via KernelAbstractions

**Goal.** Per-triangle and per-cell kernels expressed as
`@kernel` functions executable on multiple GPU backends. Single
source code targets CUDA / Metal / AMDGPU / oneAPI / multi-thread
CPU.

**Files:**

- `src/kernels.jl` — KernelAbstractions kernels for: per-triangle
  EL residual, per-triangle q-term, per-cell remap-overlap,
  AMR-step utilities.
- `src/backend.jl` — backend-selector glue.

**Acceptance:**

1. **Correctness.** Kernel outputs match the reference CPU
   implementation to round-off across all 4 GPU backends + CPU.
2. **Performance.** ≥ 70% of theoretical peak GPU bandwidth on
   one of CUDA/Metal (per `specs/05` §5.3 ParallelStencil baseline).
3. **Portability.** No code path requires #ifdef-style backend
   forking. (Vendor-library calls like cuFFT for the wave-pool
   spectra are an allowed exception.)

---

## Phase 9 — Tier D.1: 2D Kelvin-Helmholtz (the falsifiable noise prediction)

**Goal.** 2D KH instability with the per-principal-axis stochastic
injection enabled; verify that noise lives on the *one* (compressive)
axis only — KH's incompressible shear has no noise on the shear
axis. **This is the falsifiable scientific prediction of the
methods paper.**

**Acceptance** (methods paper §10.5 D.1):

1. **Noise-axis selectivity.** $C_B^\parallel = O(1)$ (compressive
   axis, calibrated); $C_B^\perp = 0$ (shear axis, *measured to
   zero* within sampling noise). If $C_B^\perp \ne 0$, the
   per-axis decomposition is wrong.
2. **KH linear growth rate.** Numerical growth rate matches
   analytical inviscid KH dispersion to < 5% over the linear
   regime.
3. **Self-consistency monitor.** $\hat k_a$ vs $\hat\lambda_{a,\rm res}$
   per axis; no axis fires the warning.

**Risk:** the v3 §1.2 / M1 Phase 8 calibration mismatch
(production kurt 3.45 → λ ≈ 6.7 vs small-data λ ≈ 1.6) re-surfaces
in 2D; expect to debug here. **The KH test is the natural place
to nail down which λ value is right.**

---

## Phase 10 — Tier D.4: 2D Zel'dovich pancake collapse (*central novel test*)

**Goal.** The methods paper's signature 2D demonstration: a 2D
cosmological-style Zel'dovich pancake collapse, with per-axis γ
diagnostics tracking the 2D shell-crossing geometry, and
stochastic injection regularizing the caustic without spurious
$y$-direction noise.

**Acceptance** (methods paper §10.5 D.4):

1. **Pre-pancake.** Density matches the 2D Zel'dovich analytical
   solution to L∞ rel < $10^{-3}$ (the 2D analog of M1 Phase 3).
2. **Pancake formation.** $\gamma_1 \to 0$ at the pancake plane
   while $\gamma_2 \sim 1$. The principal-axis decomposition
   correctly identifies the collapse direction.
3. **Stochastic regularization.** Injection on the compressive
   ($\gamma_1 \to 0$) axis only; no $y$-direction broadening.
4. **Second crossing.** Filament formation with both $\gamma \to 0$.
5. **Comparison to ColDICE 2D + PM N-body.** Density profile match
   within 5% L∞ rel through pre-crossing; statistical match
   post-crossing.

**This is the deliverable that makes the methods paper.**

---

## Phase 11 — Tier D.5: 2D wave-pool noise calibration

**Goal.** Calibrate $C_A^\parallel, C_A^\perp, C_B^\parallel,
C_B^\perp$ on a 2D wave-pool problem; verify variance-gamma
residuals per axis with self-consistency on burst-shape parameters.

**Acceptance** (methods paper §10.5 D.5):

1. Per-axis calibration constants converge.
2. $C_B^\perp \approx 0$ in pure-shear regions.
3. Self-consistency monitor green per axis.
4. Spectra match fine-DNS reference within methods-paper tolerance.

---

## Phase 12 — Tier D.7: dust-trapping in 2D vortices (*realistic test*)

**Goal.** KH instability with passive dust species; verify
vortex-center dust accumulation matches reference codes
(PencilCode, AREPO with dust); per-species γ diagnostics fire
selectively (dust collapses, gas does not).

**Acceptance** (methods paper §10.5 D.7):

1. Vortex-center dust density enhancement within 10% of reference.
2. Per-species γ: dust $\gamma_1 \to 0$, gas $\gamma_a \sim 1$.
3. Tracer fidelity (multi-tracer) preserved.

---

## Phase 13 — Tier D.10: ISM-like 2D multi-tracer (*community-impact test*)

**Goal.** Multi-tracer 2D shocked turbulence with metallicity
labels; verify metallicity PDF and spatial structure consistent
with Lagrangian methods (SPH, AREPO) and significantly better than
Eulerian methods.

**Acceptance** (methods paper §10.5 D.10):

1. Metallicity PDF KS-distance from Lagrangian reference < 0.1.
2. Spatial-structure interface sharpness 2-3× better than
   Eulerian PPM/WENO reference.
3. Conservation invariants intact (mass, momentum per species).

---

## Phase 14 — Remaining Tier D + Tier E + wrap-up

**Tier D remaining:**
- D.2 cylindrical blast (cylindrical symmetry preservation).
- D.3 oblique shock interaction.
- D.6 2D dust-in-gas cold sinusoid.
- D.8 2D plasma anisotropy.
- D.9 KH tracer fidelity.

**Tier E stress tests** (per methods paper §10.6):
- E.1 high-Mach 2D shocks (graceful failure).
- E.2 severe shell-crossing geometries.
- E.3 very low Knudsen (Navier-Stokes reduction).
- E.4 cosmological IC (with self-gravity — explicitly Milestone 4 if not done here).

**Wrap-up deliverables:**

- `reference/MILESTONE_3_STATUS.md` — synthesis analog of M1's status doc.
- `reference/notes_performance_2d.md` — performance baseline at production scale.
- Performance comparison vs 1D M1 baseline (per `reference/notes_performance.md`).
- Paper figures (per methods paper §10's headline tests).
- Handoff to Milestone 4 (self-gravity + cosmological-scale runs).

---

## Cross-cutting concerns

### Code organization (recommended `src/` layout)

```
src/
├── dfmm.jl                          module entry; using/exports
├── (M1 carry-over) types.jl, eos.jl, diagnostics.jl, plotting.jl,
│                   io.jl, calibration.jl, regression.jl,
│                   stochastic.jl, tracers.jl
├── types_2d.jl                      TriField, principal-axis state
├── triangle_mesh.jl                 Lagrangian simplicial mesh
├── eulerian_mesh.jl                 Eulerian quadtree (Phase 6+)
├── cholesky_2d.jl                   2D Hamilton-Pontryagin EL
├── discrete_transport_2d.jl         D_t^(q,a) midpoint stencils
├── discrete_action_2d.jl            ΔS_n sum over triangles
├── newton_step_2d.jl                multi-DOF sparse Newton
├── deviatoric_2d.jl                 2-component π
├── heat_flux_2d.jl                  4-component Q
├── stochastic_injection_2d.jl       per-axis VG injection
├── artificial_viscosity_2d.jl       tensor-q (Caramana-Shashkov)
├── remap.jl                         r3d-based Lagrangian↔Eulerian
├── bayesian.jl                      law of total cumulants
├── amr.jl                           p4est wrapper
├── action_error_indicator.jl        ΔS_cell
├── amr_callbacks.jl                 Trixi-style callbacks
├── kernels.jl                       KernelAbstractions kernels
├── backend.jl                       GPU/CPU backend select
└── setups_2d.jl                     2D Tier-C/D IC factories
```

### Newton convergence playbook (2D-specific)

The per-triangle EL system has ~21 unknowns × N_tri triangles. The
Jacobian sparsity is the triangle adjacency graph (each triangle
couples to its 3 neighbors). Use `SparseConnectivityTracer` +
`SparseMatrixColorings` (already pulled in by M1). Newton
iterations should converge in 2-3 steps on smooth problems
(matching M1's 1D experience).

When stalls happen near caustics (Phase 10 pancake), apply M1's
escalation playbook generalized to 2D:
1. Exp-parameterize $\gamma_a = \exp(\lambda_{3,a})$ per axis.
2. Damped Newton + line search.
3. TrustRegion.
4. Per-axis $\gamma_\epsilon$ continuation.
5. Kraus 2017 projected variational integrator.

### Regression-testing pattern

For each Tier-C/D phase: a Julia experiment writes an HDF5 output;
a paired test loads both the Julia output and a reference (either
M1 1D goldens for Tier-C, or external code outputs for Tier-D).
Goldens for Tier-C are derived from M1; Tier-D goldens come from
ColDICE / Athena / FLASH / PencilCode runs Tom provides.

### File ownership for parallel agents

Following M1's pattern, M3 phases that touch shared state should
be sequenced; phases that touch disjoint files can run in parallel.
The natural parallel groupings:

- **Phase 0** (theory + reading) — single agent, foundational.
- **Phases 1-2** — sequential (Phase 2 builds on Phase 1's mesh).
- **Phases 3-5** (Tier-C) — can parallelize after Phase 2.
- **Phases 6, 7, 8** (infrastructure: remap, AMR, GPU) — can
  parallelize among themselves; each touches its own file area.
- **Phases 9-13** (Tier-D headlines) — can parallelize after
  Phases 6-8; each is its own experiment + test + notes triple.

### Things explicitly *not* in M3 scope

- **Self-gravity** (Milestone 4 with Zel'dovich pancake + cosmological IC).
- **3D extension** (a separate milestone).
- **Methods paper revision pass** — Tom's work; this milestone
  produces results that justify the paper, not the paper text.
- **The four M1 architectural opens** (Phase-4 t¹ drift, Phase-5
  L∞ shock, Phase-8 calibration, Phase-8 long-time stability).
  These are inputs to M3 (resolve before launching) and outputs
  in the sense that their resolution is verified in 2D.

### Pre-M3 prerequisites

Before launching M3:

1. **Resolve M1 Open #4** (long-time stochastic realizability
   instability). Carries directly into M3 Phase 11+.
2. **Berry connection derivation** complete (Phase 0 deliverable).
3. **r3d Julia port** confirmed available + API confirmed.
4. **Methods paper revision pass** applied (per
   `reference/notes_methods_paper_corrections.md`).
5. **Optional: Milestone 2** (1D variational verification). The
   handoff explicitly leaves M2 scope open. Tom's call.

---

## Estimated wall-clock per phase

| Phase | Wall-clock | Risk |
|---|---:|---|
| 0 — foundations + Berry derivation | 2-3 weeks | high (theory) |
| 1 — per-triangle integrator | 1-2 weeks | medium |
| 2 — bulk + entropy + multi-tri | 2-3 weeks | medium |
| 3 — Tier C.1 Sod | 0.5-1 week | low |
| 4 — Tier C.2 cold sinusoid | 1-2 weeks | high (Phase-0 verifier) |
| 5 — Tier C.3 rotational | 0.5-1 week | low |
| 6 — Bayesian remap | 2-3 weeks | medium-high |
| 7 — AMR via p4est | 2-3 weeks | medium-high (interop) |
| 8 — GPU via KernelAbstractions | 2-4 weeks | medium |
| 9 — Tier D.1 KH | 1-2 weeks | medium |
| 10 — Tier D.4 pancake | 2-3 weeks | high (central test) |
| 11 — Tier D.5 wave-pool | 1-2 weeks | medium |
| 12 — Tier D.7 dust-trapping | 1-2 weeks | medium |
| 13 — Tier D.10 ISM | 1-2 weeks | low |
| 14 — wrap-up + Tier E | 2-3 weeks | low |
| **Total** | **~18-28 weeks** | **~5-7 months** |

## Acceptance for "Milestone 3 complete"

Per the methods paper §10's central-demonstration list, M3 is
complete if:

1. All Tier C tests pass (dimension-lift correct).
2. **D.1** passes (per-principal-axis noise verified —
   *the falsifier*).
3. **D.4** passes (central novel 2D test — Zel'dovich pancake).
4. **D.7** passes (realistic two-fluid 2D — dust traps).
5. **D.10** passes (community-impact tracer test — ISM).
6. Tier E tests pass in graceful-degradation mode.

(The methods paper §10.7 acceptance list also includes Tier B —
that's M1 territory, already done — and the remaining D items
which are nice-to-have but not central.)

---

## Open questions for Tom (carry through M3 phases)

1. **r3d Julia port API** — confirm signature and version pin
   when M3 starts.
2. **GPU primary target** — Apple Metal (dev laptop) or NVIDIA CUDA
   (production cluster)? Affects Phase 8 testing emphasis.
3. **External reference goldens for Tier-D** — Tom provides
   ColDICE / AREPO / PencilCode outputs for D.4, D.7. Need format
   spec.
4. **Cosmological IC source for D.4** — Zel'dovich amplitude,
   pancake angle, simulation box size; affects all of Phase 10.
5. **Self-gravity scope** — does D.4 need self-gravity (then it's
   Milestone 4 work) or is the linear-growth pre-pancake
   sufficient (then it's M3)?
6. **Methods paper §10.7 milestone schedule** — what's the actual
   text say? I haven't read §10.7 directly; this plan extrapolates
   from §10.2-§10.6.
