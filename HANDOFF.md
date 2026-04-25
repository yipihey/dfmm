# dfmm-2d implementation handoff

**For:** the agent picking up Milestone 1 (1D Julia implementation of the
unified dfmm framework).
**From:** Tom Abel, via design-iteration sessions through April 2026.
**Date:** April 25, 2026.

---

## What this is

A complete design package for **dfmm-2d**, a variational unification of
Tom Abel's existing 1D moderate-Knudsen moment scheme (`dfmm`) with the
phase-space-sheet picture of collisionless dynamics, extended to 2D via
principal-axis decomposition of the 4×4 phase-space Cholesky factor.

Six documents are bundled here. Read them in the order below. By the end
you should know exactly what to build, in what order, and using which
Julia packages.

The methods paper (`specs/01_methods_paper.pdf`) is the canonical design
document. The three action-note revisions (`design/02–04`) are the
iterative working notes that produced the final design — read them
in order to understand *why* each design choice was made, not just
*what* it is. The Julia ecosystem survey (`specs/05`) is the
implementation roadmap.

There is no code yet. **Your job is to write Milestone 1.**

---

## Read order

```
1. specs/01_methods_paper.pdf            (21 pp)  — the design, complete
2. design/02_action_note_v1.pdf          ( 6 pp)  — first variational draft
3. design/03_action_note_v2.tex          (10 pp)  — connection form, 4×4 lift
4. design/04_action_note_v3_FINAL.pdf    ( 6 pp)  — empirical findings, FINAL action
5. specs/05_julia_ecosystem_survey.md    (~35 KB) — the implementation stack
6. THIS README                                    — milestone & sanity checks
```

The action-note v3 supersedes v2 supersedes v1. v1 and v2 are kept
because they document the chain of reasoning that led to v3, including
two false starts (exponential burst durations; naive matrix-trace
pairing) that were corrected on empirical and algebraic grounds. If
you find yourself disagreeing with v3, re-read v1 and v2 first — there's
a good chance the disagreement is one Tom and the previous agents
already worked through.

---

## What's settled vs. what's open

### Settled (don't relitigate without strong cause)

- **The deterministic action.** Connection-form on the principal $GL(d)$
  bundle, weighted symplectic form $\omega_{1D} = \alpha^2\,d\alpha\wedge d\beta$,
  $\alpha$-local Hamiltonian $\mathcal{H}_{\rm Ch} = -\tfrac12\alpha^2\gamma^2$.
  Verified against dfmm by direct algebra in v2 §2.1.

- **Cold-limit reduction.** The action collapses to the Abel–Hahn–Kaehler
  mass-coordinate dust action; $\gamma\to 0$ is Hessian degeneracy.

- **Stochastic dressing.** Variance-gamma noise, derived from
  gamma-distributed compression burst durations (empirically verified
  in v3, KS test rejects exponential at $p\to 0$). Self-consistency:
  $\lambda_{\rm residual}$ = (gamma-shape of bursts).

- **2D principal-axis structure.** $\omega_{2D} = \sum_a \alpha_a^2
  d\alpha_a\wedge d\beta_a + \omega_{\rm rot}$. The two diagonal terms
  reduce cleanly to 1D; the Berry connection $\omega_{\rm rot}$ is
  flagged as "derive in 2D paper, not before Milestone 1" (v3 §4.2).

- **Bayesian remap.** Law of total covariance, monotone Liouville
  increase = physical entropy production. The geometric overlap is
  computed exactly via r3d (or Julia equivalent — see survey §6.3
  for the three options).

- **Implementation language.** Julia. The survey §14 explains why for
  this specific project (NonlinearSolve.jl, Enzyme.jl, KernelAbstractions.jl,
  P4est.jl, Makie.jl + WriteVTK.jl all production-grade).

### Open (your work)

- **Two empirical checks** Tom can run on production wave-pool calibration
  data (v3 §1.1 and §1.2 final paragraphs):
  1. $b_{\rm It\hat o}$ vs $b_{\rm closure}$ decomposition. Predicted
     ~0.15 + ~0.19 = 0.34. Either result is fine for the framework;
     matters for the physical interpretation of the noise drift.
  2. Full burst-duration vs. residual-kurtosis self-consistency at
     production scale. The small-data fit gave $\lambda \approx 1.6$;
     production gives kurt 3.45 = $\lambda \approx 1$. The mismatch is
     consistent with chaotic-divergence floor biasing toward Gaussian.
     Worth confirming.

- **The Berry connection $\omega_{\rm rot}$** in 2D. Required for the
  2D scheme. Not required for Milestone 1 (1D code). Derive when needed.

- **Polynomial-moment integration over polygons.** Survey §6.3 and §12
  explains the design choice. Recommendation: use GeometryOps.jl for
  the geometric overlap, write a custom polynomial-moment integration
  module on top using divergence-theorem closed forms. Few hundred
  lines, dimension-2 only. **Defer to Milestone 3 (when 2D code begins).**
  In Milestone 1 the moment computation is trivial 1D interval
  intersection.

- **Variational integrator implementation.** Read Kraus's two
  references in the survey before writing this:
  - Kraus, M. (2017). *Projected Variational Integrators for
    Degenerate Lagrangian Systems.* arXiv:1708.07356.
  - Kraus, M. & Tyranowski, T. M. (2019). *Variational Integrators for
    Stochastic Dissipative Hamiltonian Systems.* arXiv:1909.07202.
  
  GeometricIntegrators.jl provides reference implementations.
  Specialize for our connection-form weighted-symplectic structure.

---

## Milestone 1 plan

The methods paper §10.7 gives the full milestone schedule. Milestone 1
is your scope. Concretely:

### Goal
A 1D Julia implementation of the unified scheme that reproduces dfmm
benchmark results bit-level (deterministic limit) or statistically
(stochastic limit), demonstrating the variational structure works in
practice before any 2D claims are made.

### Inputs
- Tom's existing 1D dfmm code: https://github.com/yipihey/dfmm
  (~2600 lines Python + Numba; you'll need this as the regression target.)
- The variational action from `design/04_action_note_v3_FINAL.pdf`.
- The Julia stack from `specs/05_julia_ecosystem_survey.md` §13.

### Deliverable
A fresh Julia package `dfmm.jl` (working name) implementing:

1. **Field structure.** The 1D specialization of the 4×4 Cholesky factor
   $L = (L_1, L_2; 0, L_3) = (\alpha, \beta; 0, \gamma)$, with
   $\alpha, \beta$ evolved and $\gamma$ derived from the EOS via
   $\gamma = \sqrt{M_{vv} - \beta^2}$. Plus the bulk fields
   $(\rho, \rho u, P_{xx}, P_\perp, \rho L_1, Q)$ from the dfmm paper.

2. **Discrete action.** The discrete Hamilton–Pontryagin form with
   discrete parallel transport on the charge-1 sectors (β, γ, $P_{xx}$).
   The weighted symplectic form discretized correctly.

3. **Variational integrator.** Implicit step via NonlinearSolve.jl with
   automatic Jacobian (Enzyme.jl backend, ForwardDiff fallback).
   Energy drift target: $< 10^{-8}$ relative over $10^5$ steps (the B.1
   acceptance criterion in the methods paper §10.3).

4. **Stochastic injection.** Variance-gamma noise per compression event,
   compound-Poisson-with-gamma-durations sampling. Energy debit applied
   to the trace of $M_{vv}$. Self-consistency monitor that compares
   burst-shape and residual-kurtosis at output cadence.

5. **Tier A regression suite.** Reproduce dfmm benchmarks A.1 through A.6
   (Sod, cold sinusoid, steady shock, KM-LES wave-pool, dust-in-gas,
   plasma equilibration). The methods paper §10.2 lists them.

6. **Tier B extension tests.**
   - B.1 long-time energy drift (bounds quoted above).
   - B.2 cold-limit reduction (1D Zel'dovich, must match analytical
     pre-crossing; Hessian degeneracy at predicted location).
   - B.4 compression-burst statistics (gamma-shape verification).
   - B.5 passive scalar exact advection (zero deterministic diffusion
     in pure-Lagrangian regions).

### Acceptance criteria
- All Tier A tests pass to specified tolerances.
- Tier B.2 demonstrates the unification: cold limit recovers
  phase-space-sheet evolution; warm limit recovers dfmm.
- Tier B.4 verifies the variance-gamma derivation.
- Tier B.5 verifies the tracer-exactness claim.

### Estimated time
2–3 months for a capable agent working full-time, given the design is
this complete. The bottleneck will likely be the variational integrator
(item 3): the Newton convergence near the Hessian degeneracy is the
genuinely subtle part. Expect to spend a week reading Kraus's papers
before writing any code.

### Out of scope for Milestone 1
- 2D code (Milestone 3).
- AMR (not needed in 1D; segments stay uniform).
- GPU acceleration (1D is fast enough on CPU; defer to 2D).
- Self-gravity (Milestone 4 with Zel'dovich pancake test).
- Two-fluid extensions (Milestone 4).
- The Berry connection in the 2D matrix lift.

---

## Sanity checks before you start

1. **Read the v3 note carefully.** Especially the empirical findings in §1
   and the Hessian-degeneracy interpretation in §3.4 of v2 (preserved in
   v3). If you don't understand why the 2×2 Hessian becomes singular at
   $\gamma=\beta=0$, you don't yet understand the unification.

2. **Verify the Hamilton-equation derivation in v2 §2.1** against
   dfmm's `experiments/02_sine_shell_crossing.py` and the Cholesky
   evolution in dfmm's source code. The boxed equation
   $\mathcal{D}_t^{(0)}\alpha = \beta$, $\mathcal{D}_t^{(1)}\beta = \gamma^2/\alpha$
   should match dfmm's discrete update modulo the connection-form vs.
   bare-derivative bookkeeping. If it doesn't, raise the question with
   Tom before proceeding.

3. **Run dfmm's existing test suite first.** Get the wave-pool and
   shell-crossing benchmarks working in pure dfmm before writing a single
   line of Julia. You need to know what "correct" looks like.

4. **Set up a clean Julia environment** with the Milestone 1 stack
   from the survey §13. Use Julia 1.11 or later. Pkg.activate a fresh
   environment; document the Project.toml and Manifest.toml in the
   repository.

5. **Check in early and often with Tom.** The design choices are
   substantial enough that small implementation decisions can compound
   into big problems if uncorrected. The previous design iterations
   produced four documents over roughly ten checkpoint conversations.
   Plan for similar interaction density during implementation.

---

## Files in this bundle

```
dfmm2d_handoff/
├── README.md                                ← this file
├── specs/
│   ├── 01_methods_paper.pdf                 ← canonical design (21 pp)
│   ├── 01_methods_paper.tex                 ← LaTeX source
│   └── 05_julia_ecosystem_survey.md         ← implementation stack
├── design/
│   ├── 02_action_note_v1.pdf                ← first variational draft
│   ├── 03_action_note_v2.tex                ← connection form + 4×4 lift
│   └── 04_action_note_v3_FINAL.pdf          ← empirical findings; FINAL action
└── reference/                               ← (empty; populated as Milestone 1 progresses)
```

The reference/ directory is empty by design; it's where the
Milestone-1 agent should accumulate:
- Implementation notes, design decisions made during coding, gotchas.
- Performance benchmark data.
- Validation outputs from Tier A and Tier B tests.
- Any new findings (e.g., the production-scale empirical checks Tom
  flagged in v3 §1.1 and §1.2 final paragraphs).

These become the inputs to Milestone 2 (1D variational verification)
and ultimately Milestone 3 (the 2D agent's starting point).

---

## Three things you should NOT do

1. **Don't rewrite dfmm in Julia first and then try to add the
   variational structure.** That tempting path leads to a Julia port of
   dfmm with a variational sticker on top, not the unified framework.
   The variational structure has to be foundational from the start.
   Build the action evaluation and the discrete EL equations first;
   verify they reproduce dfmm's update rules; *then* add the bulk fields.

2. **Don't skip the cold-limit test (Tier B.2).** It's the single most
   important verification in the entire program. If the cold limit
   doesn't reduce cleanly to phase-space-sheet evolution, the
   unification claim fails and the methods paper is dead. Test it
   early, before investing in the more complex stochastic injection.

3. **Don't try to handle the 2D Berry connection** unless we explicitly
   ask for it. The 2D agent gets that work. Your scope is 1D.

---

## Contact

Tom Abel (KIPAC, Stanford). Active collaborator on every milestone.
GitHub: yipihey. The dfmm code at github.com/yipihey/dfmm is the
reference implementation; the impress suite at github.com/yipihey/impress
is the broader research-tooling ecosystem (separate project, but worth
knowing about for code style and Rust-side conventions if you find
yourself wanting to expose dfmm-2d functionality to those tools).

The previous design-iteration agent (the one that produced v1, v2, v3
of the action note) is reachable through Tom; they have substantial
context on the framework's reasoning that didn't make it into the
documents.

Good luck.
