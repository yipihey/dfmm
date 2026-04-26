# dfmm-2d extension — overview

**Audience:** Tom and the agent picking up Milestone 3 (the 2D
implementation). Synthesizes `HANDOFF.md`, `specs/01_methods_paper.tex`,
`design/03_action_note_v2.tex`, `design/04_action_note_v3_FINAL.pdf`,
and `specs/05_julia_ecosystem_survey.md` into a single high-level
picture of what changes for 2D and what carries over from M1.

For phase-level implementation detail see `reference/MILESTONE_3_PLAN.md`.
For M1 status see `reference/MILESTONE_1_STATUS.md`.

## What changes structurally for 2D — eight differences

### 1. Mesh: from one to two coupled meshes

- **1D:** single Lagrangian mass-coordinate mesh, periodic or inflow/outflow.
- **2D:** **Lagrangian simplicial triangulation** in $\bm{q}$-space (fixed topology, evolving vertex positions $\bm{x}(\bm{q}, t)$) **+ Eulerian quadtree** in $\bm{x}$-space (adaptive refinement). Coupled by r3d-based geometric overlap. Methods paper §9.1.

### 2. Cholesky factor: 2×2 → 4×4 in principal-axis form

- **1D:** scalars $(\alpha, \beta)$, $\gamma$ derived from EOS.
- **2D:** 4×4 block-lower-tri $L = \begin{pmatrix} L_1 & 0 \\ L_2 & L_3 \end{pmatrix}$, 10 independent components, parameterized as $(\alpha_a, \beta_a, \theta_R)$ for $a = 1, 2$ — two principal axes plus a rotation angle. Methods paper §5.2-§5.4.

### 3. Per-triangle field count: ~5 → ~21 (× 10 polynomial coefficients = ~210/triangle)

Per methods paper §9.2: position (2), velocity (2), Cholesky $L$ (10), entropy (1), heat flux (4), deviatoric stress (2), tracers (any). Each scalar field is a cubic ($p=3$) reconstruction over each triangle ⇒ 10 Bernstein coefficients per scalar.

### 4. Positivity is non-trivial

- **1D:** $\gamma > 0$ via clamp / exp-parameterization.
- **2D:** TWO constraints per triangle (methods paper §9.3):
  - $\det F > 0$ everywhere (no shell-crossing) — **Bernstein-basis positivity certificates**.
  - $L_{ii}(\bm{q}) > 0$ for all diagonals — **exp-parameterization** $L_{ii} = \exp(\lambda_i)$.

### 5. Stochastic injection: scalar → per-principal-axis

- **1D:** single compression axis, scalar VG noise on $\rho u$.
- **2D:** eigendecompose strain rate $S_{\rm sym} = R\Lambda R^\top$; per compressive axis $\lambda_a < 0$ inject VG independently; transform back to lab frame; paired-pressure debit (methods paper §9.6).
- **Falsifiable prediction:** KH instability has noise on *one* (compressive) axis only — Tier D.1 in the validation hierarchy.

### 6. Bayesian remap: interval → polygon

- **1D:** trivial interval intersection (Milestone 1 doesn't even need it).
- **2D:** r3d-style polygon-polygon overlap with polynomial moment integration. **Tom's Julia r3d port is being prepared independently** — when M3 starts, just `using r3d` and call directly. The `specs/05` §6.3 fallback ("GeometryOps.jl + custom moment integration") becomes obsolete.
- Methods paper §6 ("Bayesian remap with the law of total cumulants").

### 7. AMR

- **1D:** not needed (segments stay uniform).
- **2D:** **p4est-based quadtree** on the Eulerian mesh. Refinement triggered by action-error per cell $\Delta S_{\rm cell}$ — the unified detector for shocks, shell-crossings, and turbulence cascades (methods paper §9.7).

### 8. Berry connection $\omega_{\rm rot}$ — *the open theoretical work*

The 2D symplectic form is

$$\omega_{2D} = \sum_a \alpha_a^2\,d\alpha_a \wedge d\beta_a + \omega_{\rm rot}$$

where $\omega_{\rm rot}$ is the Berry curvature on the principal-axis $GL(d)$ bundle. **It is explicitly deferred from M1** (HANDOFF.md "Open" #2; v3 §4.2: "derive in 2D paper, not before Milestone 1"). The two diagonal terms reduce cleanly to the 1D scheme M1 verified; $\omega_{\rm rot}$ is what couples the principal axes.

**This is the one piece of new physics the 2D agent needs to derive before coding.** Everything else is engineering.

## What carries over from Milestone 1

The 1D Julia infrastructure is largely directly applicable:

| M1 component | Carries to 2D? | Notes |
|---|---|---|
| Variational integrator structure | ✓ | Phase 1-3 implementation generalizes |
| EOS (`src/eos.jl`) | ✓ direct | Track C |
| Stochastic primitives (`src/stochastic.jl`) | ✓ direct | Track D — VG sampler, burst stats, monitor |
| Diagnostics | ✓ generalize | γ-rank, realizability, Hessian — component-wise |
| Sparse-AD Newton | ✓ | Triangle adjacency replaces segment tri-band |
| Tracers (`src/tracers.jl`) | ✓ | `TracerMesh` design generalizes; advection still no-op in pure-Lagrangian |
| Goldens / regression scaffold | ✓ generalize | `src/regression.jl` generalizes to triangle-mesh outputs |
| Tensor-q artificial viscosity | ✓ generalize | Caramana-Shashkov tensor form for 2D |
| Heat-flux Lagrange multiplier | ✓ generalize | Q is 4-component in 2D (was 1) |
| Deviatoric stress | ✓ generalize | π is 2-component traceless symmetric (was 1) |
| Calibration / IO / plotting | ✓ direct | Track C |

What does **not** carry over:

- 1D `Mesh1D` / `Segment` structures → replaced by triangle mesh.
- 1D pressure-gradient stencil → replaced by triangle face-flux assembly.
- 1D cubic reconstruction → replaced by triangle Bernstein basis.
- 1D inflow/outflow boundary handling → 2D needs proper boundary conditions per problem.

## Implementation stack — incremental over M1

Per `specs/05` §13, M1 already pulled in:

> StaticArrays, NonlinearSolve, LinearSolve, Enzyme, ForwardDiff, Distributions, GeometricalPredicates, GeometricIntegrators, OrdinaryDiffEq, HDF5, JLD2, WriteVTK, CairoMakie, NPZ, QuadGK, SpecialFunctions, StatsBase, SparseArrays, SparseConnectivityTracer, SparseMatrixColorings.

For M3 add:

| Package | Purpose | Reference |
|---|---|---|
| **P4est.jl + P4estTypes.jl** | quadtree AMR | `specs/05` §2.1 |
| **Tom's `r3d` Julia port** | polygon overlap + polynomial moments | replaces `specs/05` §6.3 fallback |
| **KernelAbstractions.jl** | vendor-neutral GPU kernels | `specs/05` §5.1 |
| **AcceleratedKernels.jl** | sort/scan/reduce primitives | `specs/05` §5.4 |
| **CUDA.jl / Metal.jl / AMDGPU.jl / oneAPI.jl** | backend selectors | `specs/05` §5.2 |
| **MPI.jl** + **ImplicitGlobalGrid.jl** | distributed parallelism | `specs/05` §11.1 |
| **ParallelStencil.jl** | regular Eulerian-quadtree stencils | `specs/05` §5.3 |
| **Tensors.jl** (optional) | clean deviatoric-stress arithmetic | `specs/05` §9.2 |
| **HOHQMesh.jl** (optional) | curvilinear mesh generation | `specs/05` §6.5 |

**Don't add to the M3 stack:**

- Trixi.jl (study its source per `specs/05` §2.2; do not depend on it).
- Gridap.jl / GalerkinToolkit.jl (heavyweight FEM frameworks; mismatched paradigm per `specs/05` §10).

## Validation hierarchy (methods paper §10)

The 2D acceptance criteria ramp from "reduces correctly to 1D" up through novel 2D physics:

### Tier C — 2D consistency (verifies dimension lift correctness)

- **C.1 1D-symmetric 2D Sod.** Solution exactly $y$-independent to numerical precision.
- **C.2 1D-symmetric cold sinusoid.** Per-axis $\gamma$ diagnostic correctly distinguishes 1D shell-crossing within 2D flow: $\gamma_1 \to 10^{-5}$, $\gamma_2 \to O(1)$.
- **C.3 Plane wave at arbitrary direction.** Rotational invariance to plotting precision.

These are cheap; lands in week 1-2 of M3 once the integrator is up.

### Tier D — 2D novelty (the headline scientific results)

- **D.1 2D Kelvin-Helmholtz.** Per-principal-axis noise prediction: incompressible shear has noise on *one* (compressive) axis, not both. KH linear growth rate matches analytic. **Falsifiable noise-structure prediction.**
- **D.2 2D cylindrical blast.** Cylindrical symmetry preserved to grid scale; reference comparison to FLASH/Athena.
- **D.3 Oblique shock interaction.** Mach reflection geometry, anisotropic post-shock stress structure.
- **D.4 2D Zel'dovich pancake collapse. *Central novel test.*** Pre-pancake matches Zel'dovich analytic; $\gamma_1 \to 0$ at pancake formation while $\gamma_2 \sim 1$; stochastic injection regularizes shell-crossing without spurious $y$-direction noise; second crossing produces filament with both $\gamma \to 0$. Compared to ColDICE 2D and PM N-body references.
- **D.5 2D wave-pool noise calibration.** 2D KM-LES calibration with axis-decomposed coefficients $C_A^\parallel, C_A^\perp, C_B^\parallel, C_B^\perp$; verify $C_B^\perp \approx 0$ in pure-shear regions; verify variance-gamma residuals per axis with self-consistency on burst-shape parameter.
- **D.6 2D dust-in-gas cold sinusoid.** Per-species per-axis diagnostics: directional shell-crossing in dust, gas remains rank-full. **Genuinely new 2D two-fluid physics.**
- **D.7 2D dust-trapping in vortices.** KH instability with passive dust; vortex-center accumulation matches reference codes (PencilCode, AREPO with dust); per-species diagnostics fire selectively. **Realistic astrophysical test: proto-planetary disk dust traps.**
- **D.8 2D plasma equilibration with anisotropy.** Spatially uniform plasma, anisotropic initial $T_{ij}$; intra-species isotropization via $\tau_P$ vs. inter-species via $\nu^T_{AB}$. **Verifies separation of intra- and inter-species relaxation in 2D.**
- **D.9 Tracer fidelity in 2D Kelvin-Helmholtz.** Sharp-interface preservation 2-3× better than standard schemes.
- **D.10 ISM-like 2D problem with metallicity tracking.** Multi-tracer 2D shocked turbulence; metallicity PDF and spatial structure consistent with Lagrangian methods (SPH, AREPO); significantly better than Eulerian methods. **Methods-paper community-impact test.**

The methods paper has its central demonstration if **D.1, D.4, D.7, D.10** all pass.

### Tier E — Stress tests (graceful degradation)

- E.1 High-Mach 2D shocks: $|s|$ approaches realizability bound; scheme reports failure cleanly.
- E.2 Severe shell-crossing geometries: multiple intersecting caustics; Hessian degeneracy at all caustic locations.
- E.3 Very low Knudsen: stiff timescales handled by implicit Newton; reduces to Navier-Stokes.
- E.4 Cosmological initial conditions: CDM-style 2D collapse with self-gravity; comparison to ColDICE and PM N-body.

## Estimated scope (very rough)

Per HANDOFF.md "Estimated time": M1 was projected at 2-3 months; we're closer to ~5-6 weeks of agent-cadence work for what's done so far. M3 is substantially larger:

| Phase block | Wall-clock estimate |
|---|---|
| Berry connection derivation (paper-level work) + 2D mesh data structures + r3d wiring | 2-3 weeks |
| Per-triangle Bernstein reconstruction + positivity certificates + 2D EL action + Newton wiring | 2-3 weeks |
| Tier C consistency tests (1D code paths exercised in 2D) | 2-3 weeks |
| Tier D novel tests (KH, pancake, dust-traps, ISM) | 4-6 weeks |
| GPU porting via KernelAbstractions, AMR via p4est, distributed via MPI | 2-4 weeks |
| Wrap-up, performance tuning, paper figures | 2-3 weeks |
| **Total** | **~14-22 weeks (~4-5 months)** |

With Tom's r3d port already done and M1's variational core proven, the highest-risk paths are:

1. **Berry connection derivation** (theory, not coding).
2. **Lagrangian-Eulerian remap accuracy at high resolution** (engineering — does the polynomial-moment integration over polygon overlaps preserve all the conserved quantities at the rates we need?).
3. **Newton convergence on the 4N+M_seg coupled triangle-EL system** (the 1D analog converged in 2-3 iterations; 2D is structurally similar but the per-triangle DOF growth from 4 to ~21 may surface new conditioning issues).

## Recommended pre-M3 next steps

Before launching M3:

1. **Resolve M1 Open #4** (long-time stochastic realizability instability). Carries directly into M3's stochastic injection — without the realizability projection, 2D stochastic runs will hit the same instability faster.

2. **Berry connection derivation** — paper-level work, not coding. The structural form of $\omega_{\rm rot}$ is the entry point for the 2D action.

3. **r3d Julia port readiness check** — confirm Tom's port covers polynomial moments to required degree (likely cubic, $p=3$, so up through 6th-degree integrals over polygon overlaps). Confirm the API signature.

4. **Pick a target architecture for first GPU run** — Apple Metal is the natural pick if Tom's laptop is the dev machine; CUDA for production cluster.

5. **Read Trixi.jl source** carefully (per `specs/05` §2.2) — its mesh + AMR + callback architecture is the reference design even though we're not building on it. Specifically:
   - `src/callbacks_step/amr.jl` — AMR callback pattern.
   - `src/solvers/dg_p4est.jl` — p4est integration.
   - `IndicatorHennemannGassner` — the entropy-based AMR indicator (analogous to our action-error indicator).

6. **Methods paper revision pass** — apply the corrections in `reference/notes_methods_paper_corrections.md` (Hessian/γ direction, eq:L-Ch sign, VG pdf normalization, "bounded oscillation" caveat). The 2D paper builds on the 1D paper's text, so the corrections propagate.

7. **Optionally: Milestone 2** — formal verification / additional rigorous validation of M1 claims. The handoff explicitly leaves M2 scope open between "formal proof" and "more thorough numerical validation". Tom's call.

## Where the methods paper's central pitch is verified vs. open

| Methods-paper claim | M1 status | M3 work needed |
|---|---|---|
| Variational integrator with bounded energy | ✓ on smooth bounded problems; t¹ secular open | Same (Open #1) |
| Cold-limit unification (B.2) | ✓ verified at 2.57e-8 in 1D | C.2 axis-decomposition; D.4 pancake |
| Tracer-exactness (B.5) | ✓ literally exact in 1D | D.9 in 2D KH; D.10 ISM multi-tracer |
| Variance-gamma stochastic injection | ✓ infrastructure; 3-λ mismatch open | D.1 KH per-axis prediction (the falsifier) |
| Per-axis principal-axis decomposition | not exercised in 1D | D.1, D.5, D.6 — central 2D claim |
| Anisotropic dust traps | not exercised in 1D | D.7 — realistic test |
| Action-based AMR | not needed in 1D | full p4est + ΔS_cell indicator |
| Bayesian remap | not needed in 1D | full r3d + polynomial moments |
| Berry connection ω_rot | open theoretical | derive before coding |

The central 2D-novel scientific claims (per-axis noise structure, anisotropic dust traps, action-based AMR) are the meat of the M3 work; everything else is "1D code paths exercised in a 2D mesh."
