# Milestone 1 implementation plan

> **Status (2026-04-25):** Phases 0-11 + 5b complete. **1801 + 1
> deferred tests** on main, full suite ~2m25s. The two methods-paper
> central claims (Tier B.2 cold-limit unification, Tier B.5 tracer
> exactness) are verified. Tier A.1/A.2/A.3 land qualitatively;
> Phase 6's post-crossing golden L∞ is `@test_skip`'d pending
> shock-capturing improvement. Phases 10/12/13 not started (10 needs
> Open #4 fixed; 12/13 are two-fluid extensions). See
> `reference/MILESTONE_1_STATUS.md` for the synthesis.

**Scope.** A 1D Julia implementation of the unified variational dfmm
framework, built foundationally on the connection-form action so that
the variational structure is primary and the dfmm regression match is
*derived* (not bolted on after). Reproduces the existing 1D
Python+Numba dfmm at `py-1d/` — bit-level in the deterministic limit,
statistically in the stochastic limit.

**Out of scope.** 2D code, the Berry connection $\omega_{\rm rot}$,
AMR (segments stay uniform in 1D), GPU, self-gravity, two-fluid 2D
extensions. See `HANDOFF.md` for the milestone-2/3/4 scope split.

**Authoritative references** (cite section numbers when working):
- `HANDOFF.md` — entry point, deliverables list, three-things-not-to-do.
- `specs/01_methods_paper.tex` §3 (action), §4 (stochastic dressing),
  §9 (discrete scheme), §10 (validation hierarchy and acceptance).
- `design/03_action_note_v2.tex` §2.1 (boxed Hamilton equations),
  §3 (deterministic action), §3.5 (Hessian degeneracy at $\gamma=0$).
- `design/04_action_note_v3_FINAL.pdf` (variance-gamma; supersedes v2).
- `specs/05_julia_ecosystem_survey.md` §13 (the implementation stack).

The boxed Hamilton equations driving the Cholesky sector
(v2 §2.1, eq. (eq:dfmm-cholesky-cov)):

> $\mathcal{D}_t^{(0)}\alpha = \beta,
>   \quad \mathcal{D}_t^{(1)}\beta = \gamma^2/\alpha
>   = (M_{vv}(J,s) - \beta^2)/\alpha$

with the weighted symplectic form $\omega = \alpha^2\,d\alpha\wedge d\beta$
and Hamilton–Pontryagin Lagrangian
$\mathcal{L}_{\rm Ch} = (\alpha^3/3)\,\mathcal{D}_t^{(1)}\beta - \mathcal{H}_{\rm Ch}$
where $\mathcal{H}_{\rm Ch} = -\tfrac12\alpha^2\gamma^2$. **Strain
coupling lives in $\mathcal{D}_t^{(q)}$, not in $\mathcal{H}$** — this
is the structural fact the discrete scheme must implement.

---

## Phase ordering rationale

The handoff's "Three things you should NOT do" prescribes the order:
1. Build the variational core *first*; verify it reproduces dfmm's
   bare-derivative update rule; *then* add bulk fields.
2. Test the cold limit (B.2) **early**, before stochastic injection.
   This is the central unification claim — if it fails, the framework
   fails and the methods paper is dead. Don't invest in stochastic work
   until B.2 is green.
3. Don't touch the 2D Berry connection.

Phases 1–4 form the "variational core works" gate. Phases 5–7 are
deterministic Tier-A regression. Phases 8–9 add stochastic dressing.
Phases 10–11 are stochastic Tier-A/B. Phases 12–13 are two-fluid.
Phase 14 is the handoff to Milestone 2.

A reasonable estimate (handoff: "2–3 months for a capable agent
working full-time"): Phases 1–4 are a month of work concentrated on
the variational integrator and the Hessian-degeneracy debugging
("the genuinely subtle part"). The remaining phases are roughly
proportional to lines-of-code in the regression target.

---

## Phase 0 — Foundations *(✓ done)*

**Status:** repo restructured; Julia package activates and tests
pass on Julia 1.12.6; `py-1d/` regression target installed and its
87-test pytest suite passes.

**Remaining housekeeping** before Phase 1:

- [ ] Read `design/04_action_note_v3_FINAL.pdf` end to end. Key
      sections: §1 (empirical findings), §3.4 (Hessian degeneracy
      interpretation, preserved from v2), §4.2 (Berry deferral).
- [ ] Read `design/03_action_note_v2.tex` §2.1–§3.5 in full. Verify
      by hand algebra that
      $\mathcal{D}_t^{(0)}\alpha = \beta,\;\mathcal{D}_t^{(1)}\beta = \gamma^2/\alpha$
      reduces to $\dot\alpha = \beta,\; \dot\beta + (\partial_x u)\beta = \gamma^2/\alpha$
      (v2 eq:dfmm-cholesky-bare). This is HANDOFF sanity-check #2.
- [ ] Trace `py-1d/dfmm/schemes/cholesky.py` (591 lines) and confirm
      its update rule matches the bare-derivative form. The
      `hll_step` / `hll_step_muscl` functions are HLL finite-volume —
      *not* a variational integrator. The Julia code does **not**
      reproduce HLL; it reproduces the *output* of HLL via a
      different (variational) discretization.
- [ ] Read Kraus's two papers:
      - Kraus 2017, *Projected Variational Integrators for Degenerate
        Lagrangian Systems* (arXiv:1708.07356).
      - Kraus & Tyranowski 2019, *Variational Integrators for Stochastic
        Dissipative Hamiltonian Systems* (arXiv:1909.07202).
      Skim `GeometricIntegrators.jl` source for reference.
- [ ] Decide: hand-rolled discrete EL, or `GeometricIntegrators.jl`
      with our own Lagrangian? **Recommendation:** hand-rolled. The
      weighted symplectic form is non-standard and the Hessian
      degeneracy needs custom care.
- [ ] (optional) CI: GitHub Actions running `Pkg.test()` on Julia 1.11
      and 1.12; `JuliaFormatter.jl` config. Defer if the agent is
      working alone for the first month.

**Exit gate.** The agent can write the boxed Hamilton equations from
memory and explain why the strain term has to live in $\mathcal{D}_t$.

---

## Phase 1 — Cholesky-sector variational integrator (deterministic, no bulk) *(✓ done)*

**Goal.** A standalone Julia module implementing the discrete
Hamilton–Pontryagin integrator for the Cholesky sector $(\alpha, \beta)$
alone, with $\gamma = \sqrt{M_{vv}(J,s) - \beta^2}$ supplied externally
(constant initial $J,s$). Verifies the variational structure produces
the boxed equations.

**Files to create:**
- `src/cholesky_sector.jl` — Hamilton–Pontryagin Lagrangian
  $\mathcal{L}_{\rm Ch}(\alpha,\beta,\dot\beta; \gamma)$, the discrete
  action $\Delta S_n$ for one timestep, the discrete EL residual
  $F(\mathbf{q}_{n+1};\mathbf{q}_n) = 0$.
- `src/discrete_transport.jl` — discrete covariant derivative
  $\mathcal{D}_t^{(q)}$ per methods-paper §9.5: midpoint strain rate
  $\overline{(\partial_x u)}_{n+1/2}$ acting on charge-$q$ fields.
  In Phase 1 the strain rate is supplied as a fixed external field.
- `src/newton_step.jl` — wraps `NonlinearSolve.NewtonRaphson` with the
  EL residual; Jacobian via `ForwardDiff.jl` (forward-mode is right
  for the small per-cell system; bring in Enzyme later when the
  per-step problem grows).
- `src/types.jl` — `StaticArrays.MVector` field type for one cell.

**Tests** (`test/test_phase1_cholesky.jl`):
1. **Zero-strain free evolution.** $\partial_x u = 0$, $\gamma$ fixed.
   $\dot\alpha = \beta$, $\dot\beta = \gamma^2/\alpha$. Closed-form
   solution: SHO-like in $(\alpha,\beta)$. Julia integrator matches
   to $10^{-12}$ over 100 steps. **Closes Phase 1.**
2. **Uniform strain.** $\partial_x u = \kappa$ const. Bare equation
   $\dot\beta + \kappa\beta = \gamma^2/\alpha$ has analytical
   exponential factor; verify Julia matches. This isolates the
   parallel-transport operator.
3. **Symplecticity.** Phase-portrait area $\int\alpha^2\,d\alpha\,d\beta$
   on a closed orbit conserved to round-off.

**Pitfalls.**
- The symplectic form is *weighted* ($f = \alpha^2$) — standard
  symplectic-Euler from `OrdinaryDiffEq.jl` does not apply. Implement
  the Hamilton–Pontryagin discretization directly.
- Don't add Hessian-degeneracy regularization yet — Phase 1 keeps
  $\gamma$ bounded away from zero via initial condition. Cold-limit
  handling enters Phase 3.

---

## Phase 2 — Bulk + entropy coupling (full deterministic action) *(✓ done)*

**Goal.** Add position $x$, velocity $u$ (charge-0), and entropy $s$
(charge-0). The full deterministic Lagrangian (methods-paper eq.
$\mathcal{L}_{\rm det}$ minus the deviatoric and heat-flux pieces;
1D specializations of those are scalar and added in Phase 5):

> $\mathcal{L}_{\rm det} = \tfrac12\dot x^2 + \mathcal{L}_{\rm Ch}$

EL of $x$ gives $\ddot x = -\partial_m P_{xx}$ (v2 eq. 17); EL of $s$
gives $\dot s = 0$ (adiabatic).

**Files:**
- Extend `src/cholesky_sector.jl` to a full per-cell action
  $\Delta S_n$ over the Lagrangian segment, with neighbor coupling
  through $\partial_m P_{xx}$.
- `src/segment.jl` — the 1D Lagrangian mass-coordinate mesh: ordered
  list of segments, fixed labels $m_i$, evolving positions $x_i(t)$.
- `src/discrete_action.jl` — sum over segments, sum over timesteps.
  Sparsity of the discrete EL Jacobian matches segment adjacency
  (tri-band in 1D).

**Tests** (`test/test_phase2_bulk.jl`):
1. **Mass conservation.** Per-segment mass $\Delta m_i$ fixed by
   construction; assert at every step.
2. **Translation invariance ⇒ momentum conservation** in periodic
   BC. Total $\sum_i (\rho u)_i \Delta m_i$ to round-off.
3. **Free-streaming dust** ($\gamma = 0$ limit hard-coded; cold
   sinusoid pre-crossing). Particles stream ballistically;
   $x_i(t) = x_i(0) + u_i(0) t$ to $10^{-10}$. *Closes Phase 2.*
4. **Linearized acoustic wave.** Small-amplitude sinusoid; compare
   to dispersion relation $\omega = c_s k$.

**Note.** "Discrete momentum exact" requires care with the parallel
transport on charged fields. Methods paper §9.5 "Conservation
properties" claims exactness by construction; verify it.

---

## Phase 3 — Cold-limit reduction (Tier B.2) — *central unification test* *(✓ done — methods-paper claim verified)*

**Goal.** Cold sinusoid through shell-crossing, deterministic only.
This is the "is the framework right at all" test. Per HANDOFF
"Three things you should NOT do" #2: don't skip this, don't defer it
behind stochastic work.

**Setup.** 1D Zel'dovich-style: uniform $\rho_0$, sinusoidal velocity
$u_0(m) = A \sin(2\pi m/L)$, $\gamma$ initialized at machine epsilon
(or $|s|$ at large negative). Periodic BC.

**Acceptance** (methods paper §10.3 B.2):
- **Pre-crossing.** Density profile matches the analytical Zel'dovich
  solution to $10^{-6}$ absolute. (Zel'dovich: $\rho(q,t) = \rho_0/(1-tDu_0/\partial q)$.)
- **Hessian-degeneracy at the predicted location.** Detect
  $\det\,\mathrm{Hess}(\mathcal{H}_{\rm Ch}) \to 0$ at $t \to t_{\rm cross}$
  per v2 §3.5; the integrator does not blow up there.
- **Post-crossing.** $\gamma$ stays $\sim 0$; multi-stream behavior
  emerges in the parcel positions (not in $\rho$) without spurious
  density spikes. *In the deterministic limit, the integrator should
  smoothly handle the rank-1 caustic — stochastic regularization is
  added in Phase 8 and is not required for this phase to pass.*

**Files:**
- `experiments/B2_cold_zeldovich.jl` — setup, runner, comparison.
- `src/diagnostics.jl` — $\det\,\mathrm{Hess}(\mathcal{H}_{\rm Ch})$ per cell,
  $\gamma$ rank diagnostic, $|s|$ realizability marker.

**Pitfalls — the genuinely subtle part.**
- Newton may stall at the caustic. HANDOFF flags this as the bottleneck.
  Mitigations to try in order:
  - Damped Newton (line search via NonlinearSolve's `LineSearchAlgorithm`).
  - Trust region (`TrustRegion()` from NonlinearSolve).
  - Pseudo-time continuation: add small artificial $\gamma_\epsilon$
    that decays to zero, run a continuation in $\gamma_\epsilon$.
  - As a last resort, Kraus 2017 *projected* variational integrator.
- Numerical $\gamma$ at machine epsilon vs. exact zero: prefer the
  exp-parameterization $\gamma = \exp(\lambda_3)$ from methods paper
  §9.3, with $\lambda_3 \to -\infty$. Then $\gamma > 0$ is automatic
  and the cold limit is the limit $\lambda_3 \to -\infty$.

**Exit gate.** B.2 acceptance criteria are hit. **Do not proceed past
Phase 3 if this fails.** Open a discussion with Tom; revisit the v3
note's §3.4 on the Hessian-degeneracy interpretation.

---

## Phase 4 — Long-time energy drift (Tier B.1) *(✓ literal pass; t¹ secular open — see notes_phase4)*

**Goal.** Run a smooth periodic acoustic-wave problem for $10^5$
timesteps with the full deterministic integrator from Phases 1–3.
Demonstrate variational structure produces no secular energy drift.

**Acceptance** (methods paper §10.3 B.1, HANDOFF deliverable item 3):
- $|\Delta E| / E_0 < 10^{-8}$ over $10^5$ steps. Standard non-symplectic
  integrators accumulate $> 10^{-4}$ for comparison.

**Files:**
- `experiments/B1_energy_drift.jl` — small-amplitude periodic wave;
  output energy at every step; plot $\Delta E(t)$ on log scale.

**Pitfalls.**
- Newton tolerance must be tighter than the drift target. Plan for
  $\|F\|_\infty < 10^{-12}$ on the EL residual.
- Energy drift is the *acceptance test for the variational integrator
  itself*; if it fails, the discrete action discretization has a
  bookkeeping error in the parallel transport. Revisit Phase 1 tests.

---

## Phase 5 — Tier A.1 Sod (warm-limit dfmm regression) *(✓ qualitative; L∞ ~10-20%, L1 ~3-4% — see notes_phase5)*

**Goal.** Reproduce dfmm's Sod profile in the deterministic warm
limit. The first regression test against `py-1d/`.

**Adds physics.** Pressure: $P_{xx} = \alpha^2\gamma^2 + (\text{deviatoric})$
in the closed Cholesky form (v2 §3.3 1D specialization with
$\Pi = P_{xx} - P_\perp$ relaxing under BGK). Heat-flux constraint
optional in Phase 5 (defer to Phase 7 if not needed for Sod
fidelity at three $\tau$ regimes).

**Acceptance** (methods paper §10.2 A.1):
- Reproduce dfmm Fig. 2 across three $\tau$ regimes ($\tau \to \infty$
  collisionless, $\tau \sim 1$ moderate, $\tau \to 0$ Euler-like).
- Density, velocity, pressure profiles match `py-1d`'s
  `experiments/01_sod_validation.py` output to L^∞ rel error $< 0.05$.

**Files:**
- `src/eos.jl` — $M_{vv}(J,s)$, ideal-gas closed form
  ($M_{vv} = \exp((\gamma-1)s/c_v)\cdot \rho^{\gamma-1}$ or whatever
  py-1d uses; check `py-1d/dfmm/schemes/_common.py`).
- `src/deviatoric.jl` — 1D scalar $\Pi = P_{xx} - P_\perp$ with BGK
  relaxation $\dot\Pi + \Pi/\tau = -2\eta S^{\rm dev}$ (v2 eq. 36).
- `experiments/A1_sod.jl` — setup mirroring `py-1d/dfmm/setups/sod.py`.
- `test/test_phase5_sod_regression.jl` — load py-1d output via
  `PyCall.jl` *or* a checked-in HDF5 snapshot; assert profile match.

**Pitfalls.** This phase is where the variational approach *must*
re-derive dfmm's bare update under uniform-strain regions. Bug here
is an EL-discretization error, not a physics error.

---

## Phase 6 — Tier A.2 cold sinusoid across six $\tau$ decades *(✓ pre-crossing across 6 decades; 1 deferred test — see notes_phase6)*

**Goal.** Same setup as B.2 (Phase 3), but parameterized over the
$\tau$ scan and now matching dfmm Fig. 3. Verifies the warm-to-cold
crossover.

**Acceptance:** $\gamma$ drops by $\sim 6$ decades at caustics
(methods paper §10.2 A.2); pre-crossing matches Zel'dovich; profiles
match `py-1d`'s `experiments/02_sine_shell_crossing.py`.

**Files:** `experiments/A2_cold_sinusoid_scan.jl`.

---

## Phase 7 — Tier A.3 steady shock Mach scan (+heat flux) *(✓ R-H to ≥3 decimals at all M_1; long-horizon limited — see notes_phase7)*

**Goal.** Reproduce dfmm Fig. 4 across $M_1 \in \{1.5, \ldots, 10\}$.
Rankine–Hugoniot to 3 decimal places.

**Adds physics.** Heat-flux Lagrange multiplier $\lambda_Q$ in
$\mathcal{L}_{\rm det}$ (v2 eq. 17). Implement the hard-constraint
form first (per v2 footnote: soft-constraint deferred to appendix).

**Files:**
- `src/heat_flux.jl` — the $\lambda_Q(\dot Q + Q/\tau - \mathcal{S}_Q)$
  constraint and its discrete realization.
- `experiments/A3_steady_shock.jl` — mirroring
  `py-1d/dfmm/setups/shock.py` and `experiments/03_steady_shock.py`.

---

## Phase 8 — Stochastic injection (variance-gamma) *(✓ infrastructure; calibration mismatch + long-time instability open — see notes_phase8)*

**Goal.** Implement compression-activated variance-gamma noise per
methods paper §4 (replacing v2's Laplace special case) and §9.6
(per-axis injection, paired-pressure energy debit).

**Sampler** (v3 §1; ecosystem survey §7.1):
```julia
function rand_variance_gamma(rng, λ::Real, θ::Real)
    V = rand(rng, Gamma(λ, θ))
    return sqrt(V) * randn(rng)
end
```

**Per-step recipe** (methods paper §9.6 — note 1D principal-axis is
trivial: one axis):
1. Compute $\partial_x u$ at cell.
2. If $\partial_x u < 0$ (compressive):
   - drift: $\delta(\rho u)^{\rm drift} = C_A\,\rho\,\partial_x u\,\Delta t$
   - noise: $\delta(\rho u)^{\rm noise} = C_B\,\rho\,\sqrt{|\partial_x u|\,\Delta t}\,\eta$,
     $\eta \sim \mathrm{VG}(\lambda, \theta)$.
3. Paired-pressure energy debit:
   $\Delta KE = u\,\delta(\rho u) + |\delta(\rho u)|^2/(2\rho)$,
   debited from $M_{vv}$ trace.
4. Spatial smoothing: 1D Gaussian kernel, correlation length
   $\ell_{\rm corr} \sim 2$ cells.
5. **Self-consistency monitor.** At each output cadence: empirical
   residual kurtosis from compression-active cells vs. gamma-shape
   parameter from burst-duration histogram. Warn if ratio outside
   $[0.5, 2.0]$.

**Files:**
- `src/stochastic.jl` — `inject_variance_gamma_noise!(...)` and
  the compression detector.
- `src/diagnostics.jl` — extend with burst-stats accumulator and
  residual-kurtosis estimator.

**Calibration.** $\lambda$, $\theta$, $C_A$, $C_B$ from
`py-1d/data/noise_model_params.npz`. Read with `NPZ.jl` or convert
to JLD2 once.

**Pitfalls.** v3 §1.2 last paragraph: production small-data fit gave
$\lambda \approx 1.6$; production scale gives kurt 3.45 ⇒
$\lambda \approx 1$. Mismatch is a known open question (HANDOFF
"Open" item 1.ii); document in `reference/notes_calibration.md` but
don't try to resolve in Milestone 1.

---

## Phase 9 — Tier B.4 compression-burst statistics *(✓ bundled into Phase 8 — self-consistency monitor working)*

**Goal.** Run a wave-pool problem (Phase 8 active) with instrumented
burst detection. Verify the cascade structure: empirical
burst-duration distribution is $\Gamma(k)$, residual kurtosis
matches $\lambda = k$.

**Acceptance** (methods paper §10.3 B.4): KS test on burst durations
fits gamma; residual kurtosis is $3 + 3/\lambda \approx 3 + 3/k$.

**Files:** `experiments/B4_burst_stats.jl`.

---

## Phase 10 — Tier A.4 KM-LES wave-pool spectra *(not started — depends on Phase 8 long-time stability fix)*

**Goal.** Reproduce dfmm Fig. 8 spectra in the stochastic regime.
Spectral distance to dfmm output below the table-410 threshold.

**Reference py-1d code:** `experiments/10_kmles_wavepool.py`,
`11_kmles_calibrate.py`, `13_kmles_energy_conservation.py`,
`14_kmles_ensemble.py`.

**Files:** `experiments/A4_kmles_wavepool.jl`. Spectral comparison
helper in `src/spectra.jl`.

---

## Phase 11 — Tier B.5 passive scalar advection through shock *(✓ done — L∞ tracer change = 0.0 literally; methods-paper claim verified)*

**Goal.** Tracer fidelity 1–3 decades better than PPM/WENO in
pure-Lagrangian regions; bounded remap diffusion only.

**Adds.** Passive scalar field on the Lagrangian segments (charge 0,
trivially advected; no remap diffusion in Milestone 1 since 1D
remap is interval intersection — see HANDOFF "Open" item 3:
"In Milestone 1 the moment computation is trivial 1D interval
intersection").

**Reference:** py-1d's tracer infrastructure in
`py-1d/dfmm/tracers.py` and `tests/test_tracers*.py`.

**Files:** `src/tracers.jl`, `experiments/B5_tracer_shock.jl`.

---

## Phase 12 — Tier A.5 two-fluid dust-in-gas *(not started — two-fluid Track A extension)*

**Goal.** Two species with separate Cholesky factors and the
cross-coupling kernels from `py-1d/dfmm/schemes/two_fluid.py`.
Reproduce dfmm Fig. 11; per-species $\gamma$ selectivity (dust
$\gamma \to 0$, gas $\gamma \sim 1$).

**Reference py-1d code:** `dfmm/schemes/two_fluid.py` (543 lines),
`experiments/05_two_fluid_dust_gas.py`.

**Files:** `src/two_fluid.jl`, `experiments/A5_dust_gas.jl`.

---

## Phase 13 — Tier A.6 Coulomb plasma equilibration *(not started — two-fluid Track A extension)*

**Goal.** Reproduce dfmm Fig. 12: thermal-vs.-momentum relaxation
mass-ratio scaling.

**Reference py-1d:** `experiments/07_eion_equilibration.py`.

**Files:** `experiments/A6_eion.jl`. Cross-coupling kernel for
plasma case from `py-1d/dfmm/schemes/two_fluid.py`.

---

## Phase 14 — Wrap-up: handoff to Milestone 2 *(in progress — see MILESTONE_1_STATUS.md)*

**Deliverables:**
- All Tier-A tests green; B.1, B.2, B.4, B.5 green.
- Performance benchmark: per-step wall time vs. py-1d at matched
  resolution. Capture in `reference/perf_milestone1.md`.
- Implementation notes (gotchas during Newton convergence, Hessian
  degeneracy, calibration drift) in `reference/notes_*.md`.
- Validation outputs (HDF5 snapshots used by tests) in
  `reference/golden/` (small files only; the suite regenerates them
  from py-1d on demand).
- Open empirical checks from HANDOFF "Open" §1: $b_{\rm Itô}$ vs
  $b_{\rm closure}$ decomposition (predicted $\sim 0.34$); production
  burst-duration vs. residual-kurtosis self-consistency. These need
  Tom's wave-pool calibration data; coordinate.
- A `MILESTONE_2_HANDOFF.md` analogous to the present document,
  scoping the 1D variational verification (per the methods paper's
  §10 acceptance list — items 1, 2, 3, 4 should all be hit by end
  of Milestone 1; Milestones 2/3 cover Tier C/D 2D work).

---

## Cross-cutting concerns

### Code organization

Recommended `src/` layout (grow as phases land):

```
src/
├── dfmm.jl              # module entry; using/exports
├── types.jl             # Field types (StaticArrays-backed)
├── segment.jl           # 1D Lagrangian mass-coordinate mesh
├── eos.jl               # M_vv(J, s)
├── cholesky_sector.jl   # L_Ch and discrete EL of (α, β)
├── discrete_transport.jl# D_t^(q) on charged fields
├── discrete_action.jl   # full S_d sum over segments × timesteps
├── newton_step.jl       # NonlinearSolve wrapper
├── deviatoric.jl        # Π relaxation (Phase 5)
├── heat_flux.jl         # λ_Q constraint (Phase 7)
├── stochastic.jl        # variance-gamma noise (Phase 8)
├── two_fluid.jl         # cross-coupling kernels (Phase 12)
├── tracers.jl           # passive scalars (Phase 11)
├── diagnostics.jl       # γ, |s|, Hessian, burst-stats
└── io.jl                # HDF5/JLD2 + WriteVTK output
```

### Newton convergence playbook

Order to escalate when Newton stalls (Phase 3 onward):
1. Tighten initial guess (use Phase 1 zero-strain solution as warmer
   guess for the implicit step).
2. Switch from `NewtonRaphson()` to `LevenbergMarquardt()` or
   `TrustRegion()` in `NonlinearSolve.jl`.
3. Continuation in $\gamma_\epsilon$ (artificial floor → 0).
4. Switch to Kraus 2017 projected variational integrator. This is a
   bigger rewrite — only if 1–3 fail.

### Regression-testing pattern

For each Tier-A phase: a Julia experiment script writes an HDF5
output; a paired Julia test loads both the Julia output and the
py-1d golden snapshot from `reference/golden/*.h5` and asserts
profile match within tolerance. Goldens are checked in (small)
or regenerated by `make goldens` from py-1d (the Makefile target
to add at the start of Phase 5).

### Things explicitly *not* in scope (per HANDOFF)

- 2D code, Berry connection $\omega_{\rm rot}$.
- AMR — segments stay uniform.
- GPU.
- Self-gravity (Milestone 4).
- Two-fluid 2D extensions (Milestone 4).
- The polynomial-moment integration over polygons. 1D remap is trivial
  interval intersection, so M1 does not need it. **For M3 (2D),
  Tom is independently porting r3d (Powell-Abel 2015) to Julia;
  assume that package is available as a dep when 2D work begins,
  and skip the GeometryOps.jl + custom moment-integration alternative
  recommended in `specs/05_julia_ecosystem_survey.md` §6.3.**

---

## Open questions for Tom (carry through phases)

1. **Calibration mismatch** (HANDOFF "Open" 1.ii): production
   wave-pool kurt 3.45 ⇒ $\lambda \approx 1$, but small-data fit
   gives $\lambda \approx 1.6$. Worth a production-scale check. Not
   blocking; flag during Phase 9.
2. **Drift decomposition** (HANDOFF "Open" 1.i): predicted
   $b_{\rm Itô} \approx 0.15 + b_{\rm closure} \approx 0.19 = 0.34$.
   Either result is fine for the framework; matters for physical
   interpretation. Flag during Phase 8.
3. **PyCall vs. checked-in goldens** for Tier-A regression. Tradeoff:
   PyCall keeps tests live against py-1d but adds a Python runtime
   dependency in CI; goldens are static but drift if py-1d evolves.
   Recommend: HDF5 goldens in `reference/golden/`, `make goldens`
   regenerator. Confirm with Tom at start of Phase 5.
