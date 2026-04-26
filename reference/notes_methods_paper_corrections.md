# Methods-paper text corrections from Milestone-1 implementation

> **Status (2026-04-26):** All four corrections **✓ applied in M2-4**
> (commit `ed5338c`). PDF rebuilt to 20 pages. See
> `reference/notes_M2_4_paper_corrections_applied.md` for the
> per-correction line-number diffs.

These are concrete edits to `specs/01_methods_paper.tex` and
`design/03_action_note_v2.tex` that surfaced during Milestone-1
implementation. Each was independently noted by an agent in their
phase report; collected here for a single-pass paper edit when Tom
has the energy.

None of these are framework-level issues. They are language /
convention discrepancies between the design corpus and the
implementation that emerged. The implementation is the source of
truth (verified end-to-end); the paper text needs alignment.

## 1. Hessian and γ at the caustic — *grows*, not *drops*

**Affected sections:**
- `specs/01_methods_paper.tex` §3.5 "Hessian degeneracy at γ→0"
- `specs/01_methods_paper.tex` §10.2 A.2: "γ drops by ∼6 decades at caustics"
- `design/03_action_note_v2.tex` §3.5 (preserved from v1)

**Issue.** With the ideal-gas EOS

> $M_{vv}(J, s) = J^{1-\Gamma} \exp(s/c_v),\quad \Gamma = 5/3$

implemented in `src/eos.jl` (matching py-1d's kinetic-moment
convention), and γ defined via $\gamma^2 = M_{vv} - \beta^2$, the
caustic *grows* both γ² and $|\det\,\mathrm{Hess}(\mathcal{H}_{\rm Ch})|$
because compression shrinks $J \to 0$ which grows $M_{vv} \to \infty$
for $\Gamma > 1$. The paper text reads as if γ → 0 at the caustic
(the cold-limit *initial* state has γ → 0; the *evolved* state at the
caustic does not).

**Empirical confirmation:**
- Phase 3 (`reference/figs/phase3_hessian_degen.png`): spatial *max*
  of $|\det\,\mathrm{Hess}|$ at $m = 1/2$ rises ~1.3 decades; spatial
  *min* at $m \in \{0, 1\}$ stays ~constant.
- Phase 6 (`reference/figs/A2_cold_sinusoid_density.png`, six τ
  panels): $\log_{10}(\gamma^2/\gamma_0^2)$ at the caustic
  $= +0.31$ to $+0.86$ across τ ∈ {10⁻³, ..., 10⁷}.

**Suggested edit.** Replace the descriptive language with the
correct form, e.g.:

> §3.5: "At the caustic, $\beta \to \sqrt{M_{vv}}$ saturates and
> $\gamma^2 \to 0$ on the trajectory's level set; for adiabatic
> $\Gamma > 1$ the *evolved* M_vv grows as $J \to 0$, so the
> Hessian's spatial *max* localizes at the caustic."

> §10.2 A.2: "γ² spatial-max grows by 0.3–0.9 decades at caustics
> across six τ decades" (replace the "drops by ∼6 decades" wording).

The variational signature of the caustic is the spatial *shape*
(symmetric peak at the rank-1 ridge), not the magnitude direction.

## 2. v2 §3.1 eq:L-Ch sign convention

**Affected section:** `design/03_action_note_v2.tex` §3.1, around
line 168–177 "Symplectic potential" paragraph + eq:L-Ch.

**Issue.** v2 states

$$\mathcal{L}_{\rm Ch} = +(\alpha^3/3)\,\mathcal{D}_t^{(1)}\beta - \mathcal{H}_{\rm Ch}$$

but applying standard EL (`δL/δq − d/dt(δL/δq̇) = 0`) to that form
produces the *wrong sign* for the boxed Hamilton equations. The
implementation in `src/cholesky_sector.jl` (Phase 1, Agent A's
report Open Question #1) uses

$$\mathcal{L}_{\rm Ch} = -(\alpha^3/3)\,\mathcal{D}_t^{(1)}\beta - \mathcal{H}_{\rm Ch}$$

equivalent to choosing $\theta = -(\alpha^3/3)\,d\beta$ as the
symplectic potential, or swapping the wedge order in $\omega$. The
two conventions describe the same physics but the discrete EL
implementation is sign-sensitive.

The header comment in `src/cholesky_sector.jl` documents this
explicitly; see lines ~36-43 of that file.

**Suggested edit.** Either:
- Flip the sign in v2 eq:L-Ch to match the implementation, or
- Add a paragraph clarifying the symplectic-potential convention
  ($\theta$ vs $-\theta$) and note that downstream EL machinery
  must use the matched convention.

## 3. Methods-paper §3.2 variance-gamma pdf normalization

**Affected section:** `specs/01_methods_paper.tex` §3.2 (around the
boxed eq:VG, the variance-gamma marginal pdf).

**Issue.** The formula in the paper

$$f(\epsilon) = \frac{|\epsilon|^{\lambda - 1/2}\,K_{\lambda - 1/2}(|\epsilon|/\sqrt\theta)}{\sqrt\pi\,\Gamma(\lambda)\,(2\theta)^{\lambda - 1/2}}$$

does **not** integrate to one under the variance-mixture parametrization
$V \sim \Gamma(\lambda, \theta)$ (Distributions-canonical, with
$\mathbb{E}[V] = \lambda\theta$, $\mathrm{Var}(\epsilon) = \lambda\theta$).

Track D's `src/stochastic.jl::pdf_variance_gamma` derives the
correct normalized form by direct integration of the variance
mixture:

$$f(\epsilon) = \sqrt{\frac{2}{\pi}}\,\theta^{-\lambda}\,\frac{\bigl(|\epsilon|\sqrt{\theta/2}\bigr)^{\lambda - 1/2}\,K_{\lambda - 1/2}\bigl(|\epsilon|\sqrt{2/\theta}\bigr)}{\Gamma(\lambda)}$$

Quadrature in `test_vg_sampler.jl` confirms unit normalization;
KS test against this CDF passes (p ∈ [0.23, 0.80] across four
$(\lambda, \theta)$ points).

The two forms are equivalent under the rescaling
$\theta_{\rm paper} := \theta_{\rm var-mix}/2$ — i.e. the paper
implicitly uses a *scaled* gamma $V \sim \Gamma(\lambda, \theta_p)$
where $\theta_p = 2\theta_{\rm var-mix}$.

**Suggested edit.** Either (a) declare the explicit
parametrization mapping ("$V \sim \Gamma(\lambda, \theta_p)$ with
$\theta_p$ as below") in §3.2, or (b) rescale the formula to use
the Distributions-canonical $\theta = \mathrm{Var}(V)/\lambda^2$
parametrization that matches the implementation.

## 4. v2 §3.1 / methods-paper §3.1 symplectic-form claim

**Affected section:** `design/03_action_note_v2.tex` §3.1, the
"weighted symplectic structure" paragraph, and the symplecticity
diagnostic in `test/test_phase1_symplectic.jl` per Agent A's
Open Question #2.

**Issue.** The boxed Hamilton equations
$\mathcal{D}_t^{(0)}\alpha = \beta$,
$\mathcal{D}_t^{(1)}\beta = \gamma^2/\alpha$ have **no closed
orbits** at fixed $M_{vv}$ in the autonomous $(\alpha, \beta)$ flow.
Specifically, with $M_{vv}$ frozen, $\beta(t)$ saturates at
$\sqrt{M_{vv}}$ and $\alpha(t) \to \infty$ along the level set
$\alpha^2(M_{vv} - \beta^2) = \mathrm{const}$; trajectories are
unbounded.

The "symplecticity check" in Phase 1 (`test_phase1_symplectic.jl`)
therefore could not use the textbook "closed orbit + $\oint\theta$
to round-off" form. Agent A used Stokes-on-a-loop-of-initial-conditions
instead (transport a closed loop in IC-space through 100 steps;
verify the loop integral $\oint(\alpha^3/3)\,d\beta$ on the evolved
loop returns its initial value to $10^{-10}$).

**Implication for the paper.** The "bounded oscillation" promise of
B.1 (methods paper §10.3) is *conditional on bounded orbits* — it
holds for the bulk-coupled system on smooth acoustic problems
(small N, short t), but not for the autonomous (α, β) sector where
trajectories are unbounded. Phase 4 confirmed: t¹ secular drift on
the acoustic test, $5.6\times 10^{-9}$ over $10^5$ steps, *literally*
within the methods-paper $10^{-8}$ bound but not pure bounded
oscillation.

**Suggested edit.** Clarify in §10.3 B.1 that "bounded oscillation"
holds for the bulk-coupled flow on smooth bounded problems; note
that the autonomous (α, β) sector has unbounded orbits and a Kraus
2017 projected variational integrator is the next step if pure
bounded oscillation is required. (See `reference/notes_phase4_energy_drift.md`.)

---

## Summary table

| Correction | Location | Severity | Fix scope |
|---|---|---|---|
| 1. Hessian/γ direction at caustic | §3.5, §10.2 A.2; v2 §3.5 | text-language | 2-3 sentences |
| 2. v2 eq:L-Ch sign convention | v2 §3.1 (line ~170) | discrete-implementation | 1 sentence + sign flip |
| 3. VG pdf normalization | §3.2 (eq:VG) | math-text | 1 line of formula or 1 sentence of param mapping |
| 4. "Bounded oscillation" claim B.1 | §10.3 B.1 | empirical caveat | 1 paragraph |

None block Milestone 1 acceptance. All are paper-edit items for the
methods paper revision pass.
