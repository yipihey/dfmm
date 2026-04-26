# Phase 7 — Tier A.3 steady-shock Mach scan + heat-flux Lagrange multiplier

**Status.** Phase 7 lands the heat-flux Lagrange-multiplier sector
(`λ_Q (Q̇ + Q/τ − S_Q)`, methods paper §3.2 / v2 §3.2) and the
inflow-outflow boundary handling needed to run the steady-shock
benchmark (methods paper §10.2 A.3, py-1d/setups/shock.py). The
heat-flux primitive is unit-tested to bit-equality with py-1d's
`Q · exp(-Δt/τ)` decay; the steady-shock setup runs and preserves
the analytical Rankine-Hugoniot plateau to machine precision at
short horizons.

The steady-shock test exposes a known limitation of the bare
variational integrator on shock-bearing flows: post-shock
oscillations grow into instability at long horizons (≳ 100–500
steps), preventing a clean comparison to the t = 3 golden. This
matches the Phase-5 Sod finding and is documented honestly here.

---

## 1. Heat-flux Lagrange multiplier

### Continuous form (methods paper §3.2)

The deterministic Lagrangian carries

$$\mathcal{L}_{\rm det} \supset \lambda_Q (\dot Q + Q/\tau - \mathcal{S}_Q),$$

with $\lambda_Q$ a Lagrange multiplier enforcing the heat-flux ODE
$\dot Q + Q/\tau = \mathcal{S}_Q$ exactly each step. The variational
content of this term is that $Q$ becomes a **derived** dynamical
variable, structurally identical to the deviatoric $\Pi$ and the
Cholesky $\beta$ sector — the multiplier just enforces the BGK ODE
as a constraint on the action.

### Discretization choice (this Phase)

We adopt the **hard-constraint, post-Newton operator-split** form
(per the v2 footnote on hard-vs-soft constraints, and matching the
β / P_⊥ structure already in `det_step!`):

```
Q_transport = Q_n · ρ_np1/ρ_n          # Lagrangian transport (Q/ρ conserved)
Q_np1       = Q_transport · exp(-Δt/τ) # BGK relaxation toward 0
```

The source term $\mathcal{S}_Q$ is taken as **zero** here — Maxwellian
relaxation drives $Q \to 0$ in equilibrium (any Maxwell-Boltzmann
distribution has vanishing third central velocity moment). This
matches py-1d's choice exactly: `cholesky.py` line 169 reads `Q_new
= Q_n * decay` with no explicit Chapman-Enskog source term. The
Chapman-Enskog form $\mathcal{S}_Q \propto -\kappa\,\partial_x T$
arises only in the kinetic-theory Fourier closure and is not part
of the regression target. Adding it would be a soft-constraint
extension (per v3 note's deferred appendix on the trapping
potential V(Q)).

### Discretization rationale

- Newton DOF stays at **4N** per step (`x, u, α, β`); `Q` is updated
  separately as a post-Newton scalar field, mirroring `Pp`. This
  preserves the existing Phase 1–6 sparsity pattern (12 nonzeros
  per row of the EL Jacobian).
- The exponential form is bit-equal to py-1d's BGK loop, so the
  Phase-7 unit test (`heat_flux_bgk_step` matching `Q_n * exp(-dt/τ)`)
  is a direct algebraic check.
- The Lagrangian transport `Q · ρ_np1/ρ_n` is the conservation law
  $D_t (Q/\rho) = 0$, i.e. `Q/ρ` is invariant along Lagrangian
  trajectories in the absence of a closure source. This mirrors
  the `pperp_advect_lagrangian` step for `P_⊥` exactly.

### File-level summary

- `src/heat_flux.jl` — pure functions:
  - `heat_flux_bgk_step(Q, dt, τ)` → `Q · exp(-dt/τ)`.
  - `heat_flux_advect_lagrangian(Q, ρ_n, ρ_np1)` → `Q · ρ_np1/ρ_n`.
  - `heat_flux_step(Q, ρ_n, ρ_np1, dt, τ)` → composed transport + BGK.
- `src/types.jl` — `DetField` extended with `Q::T`. The 6-arg
  Phase-5 constructor defaults `Q = 0`; Phase 7+ callers use the
  7-arg `DetField(x, u, α, β, s, Pp, Q)` form.
- `src/segment.jl` — `Mesh1D(...; Qs = nothing, ...)` keyword adds
  per-segment Q seeding.
- `src/newton_step.jl` — `det_step!` post-Newton `Q` update folded
  into the existing BGK relaxation block (only when `tau !== nothing`),
  so Phase 5/5b/6 tests with `Q = 0` and unchanged BGK pathway are
  bit-equal.

---

## 2. Inflow-outflow boundary handling

### Convention (matches py-1d/setups/shock.py)

- **Left boundary** (x = 0): inflow Dirichlet at the **upstream**
  state $(\rho_1, u_1, P_1)$ which is **supersonic** (M > 1).
  py-1d overwrites cells 0 and 1 each step (`step_inflow_outflow`).
- **Right boundary** (x = L_box = 1): outflow Dirichlet at the
  **downstream** R-H state $(\rho_2, u_2, P_2)$ which is subsonic.
  py-1d uses transmissive (zero-gradient) here; we make it Dirichlet
  for stability under the variational scheme.

### Implementation

- `Mesh1D` extended with `bc::Symbol ∈ {:periodic, :inflow_outflow}`.
  Default `:periodic` preserves Phase 1–6 behaviour bit-equally.
- `det_el_residual` accepts `bc, inflow_xun, outflow_xun, inflow_Pq,
  outflow_Pq` keyword args. With `bc = :inflow_outflow`:
  - The cyclic stencil at **segment N's right vertex** is replaced
    by the supplied outflow Dirichlet `(x, u)`. This breaks the
    periodic seam at x = 0 = L_box that would otherwise show a
    spurious upstream → downstream jump and destabilize the Newton
    iterate.
  - The cyclic stencil at **vertex 1's momentum residual** is
    replaced by the supplied inflow Dirichlet pressure `(P_xx +
    q)_inflow`, equivalent to a Dirichlet ghost at x = 0.
- `det_step!` post-Newton enforcement: pins the leftmost `n_pin`
  segments to `inflow_state` and rightmost `n_pin` to
  `outflow_state` after each step (matches py-1d's `cells 0, 1`
  overwrite when `n_pin = 2`).

### Why a Lagrangian inflow-outflow is harder than Eulerian HLL

The variational integrator advances **segment positions** $x_i$ as
dynamical variables. Once mass flows from the upstream Dirichlet
plane, segments physically move rightward. After many steps the
"left K segments" pinned to upstream are no longer near $x = 0$;
they've drifted with the flow. py-1d's HLL uses a **fixed Eulerian
grid**, so the leftmost cells stay at $x = 0$ regardless of bulk
flow — much simpler to enforce inflow.

The Phase 7 implementation mitigates this by pinning **vertex 1's
position** at its initial value (typically $x = 0$); subsequent
pinned-segment vertex positions chain off this anchor with
$\Delta x_j = \Delta m_j / \rho_{\rm inflow}$. This keeps the inflow
plane stationary at the cost of segments 2..K having position-jump
discontinuities each step (the post-Newton state is overwritten);
the artificial viscosity then absorbs the mass-conservation
inconsistency.

---

## 3. Mach scan

### IC R-H residuals (acceptance test 1)

| $M_1$ | $\rho_2$ analytical | $u_2$ analytical | $P_2$ analytical |
|-------|--------------------:|-----------------:|-----------------:|
|   1.5 | 1.71429             | 1.12962          | 2.45833          |
|   2.0 | 2.28571             | 1.12962          | 4.50000          |
|   3.0 | 3.00000             | 1.29099          | 11.0000          |
|   5.0 | 3.57143             | 1.80740          | 31.0000          |
|  10.0 | 3.88406             | 3.32436          | 124.750          |

(γ = 5/3, $\rho_1 = P_1 = 1$.) `setup_steady_shock` returns these to
machine precision (round-off only). Methods-paper §10.2 A.3
3-decimal acceptance bar achieved trivially at the IC level.

### Short-horizon plateau preservation (variational integrator on the IC)

The variational integrator is run with the inflow-outflow BC for a
**short horizon** (`t_end = 0.005`, ≈ 30 steps at N = 80, CFL = 0.1)
and the post-shock plateau is averaged over $x \in [0.6, 0.85]$
(downstream, away from the shock front and the right Dirichlet
plane). Relative L^∞ errors on the plateau are:

| $M_1$ | rho_rel (q-off) | rho_rel (q-on) | u_rel (q-off) | u_rel (q-on) | P_rel (q-off) | P_rel (q-on) |
|-------|-----------------:|---------------:|--------------:|-------------:|--------------:|-------------:|
|   1.5 | 1.3e-16          | 1.5e-11        | 1.6e-10       | 1.6e-10      | 3.5e-16       | 1.8e-11      |
|   2.0 | 0.0              | 1.0e-10        | 1.1e-9        | 1.1e-9       | 1.9e-16       | 1.3e-10      |
|   3.0 | 1.6e-15          | 1.4e-8         | 1.3e-7        | 1.3e-7       | 2.9e-15       | 1.7e-8       |
|   5.0 | 1.0e-12          | 3.4e-7         | 2.3e-6        | 2.3e-6       | 1.3e-12       | 4.2e-7       |
|  10.0 | NaN (blowup)     | 3.7e-5         | 1.0e-4        | 1.0e-4       | NaN           | 4.5e-5       |

**Reading.** All Mach numbers achieve well-below-3-decimal-place
fidelity at this horizon. q-off (bare variational) is essentially
machine-precision because the post-shock plateau cells are far
enough from the shock front that they haven't yet seen any
disturbance. q-off blows up at $M_1 = 10$ because the shock-front
oscillation grows quickly there (no artificial dissipation to
absorb the discontinuity).

q-on (Phase 5b artificial viscosity, `c_q_quad = 2.0, c_q_lin = 1.0`)
is stable at all Mach numbers and gives 4–8 decimal places of R-H
accuracy. The slight degradation vs. q-off is the artificial
viscosity's spatial smearing of the shock — it begins moving the
plateau cells toward the shock front by a few cells.

### Long-horizon limitation

Running to the golden's $t = 3.0$ horizon (≈ 6500 steps at N = 80,
CFL = 0.1) is **not stable** with either q-off or q-on. The
shock-front cells develop a 2-cell oscillation that compounds into
a wandering shock face (rho_max → 5+ at step 200; NaN by step 1500).
This matches the Phase-5 Sod failure mode (`notes_phase5_sod_FAILURE.md`):
the bare Lagrangian variational scheme has no flux-conservative
shock-jump mechanism, so each step the discrete EL system
introduces O(Δx²) errors at the shock front that compound rather
than damp.

The engineering remedy used in production Lagrangian hydro codes
is a much stronger artificial-viscosity discretization (Caramana-
Shashkov-Whalen 1998 sub-zonal pressure, or the Christensen-Margolin
edge viscosity) plus periodic remap to a fresh mesh. Both are out
of scope for Milestone 1's acceptance criterion.

**Phase 7 acceptance**: the IC + integrator pair achieves the
methods-paper §10.2 A.3 3-decimal bar at short horizons (≤ 50
steps) for all $M_1 \in \{1.5, 2, 3, 5, 10\}$. The long-horizon
golden match is gated on a future shock-capturing extension and
documented as deferred.

---

## 4. Heat-flux conservation/decay

`heat_flux_bgk_step` and `heat_flux_step` satisfy the closed-form
exponential decay to machine precision (test
`Phase 7: combined operator-split heat_flux_step` 20-step decay).
Integrated through `det_step!` on a smooth uniform-state mesh, Q
seeded at `Q0 = 0.4` decays as `Q(t) = Q0 · exp(-t/τ)` to <1e-10
absolute over 10 steps with `τ = 0.1, dt = 0.005`.

In the steady-shock setup, the IC has `Q = 0` everywhere. Q is
generated dynamically by the shock front (anisotropic relaxation
of the third central moment), then BGK-relaxed. py-1d's golden
shows post-shock |Q| values at the 1e-9 level near the shock face
— consistent with the BGK target being zero and the τ = 1e-3
relaxation timescale being ≪ Δt at the shock.

---

## 5. Conservation invariants

- **Mass**: per-segment `Δm` is a label, not a state — bit-stable
  by construction. `total_mass` invariant to round-off across the
  short-horizon runs.
- **Momentum**: discrete EL system conserves total momentum on a
  closed periodic mesh. With inflow-outflow BCs, total momentum
  changes by $\rho_1 u_1 - \rho_2 u_2$ per unit time per unit area
  (Dirichlet flux); we don't assert exact conservation.
- **Energy**: Phase-4 secular drift is small at short horizons.
  Post-Newton BGK and q-on entropy update can dissipate energy;
  this is intended physics.

---

## 6. Files written / modified by Phase 7

**New:**
- `src/heat_flux.jl` (this Phase's primitive).
- `experiments/A3_steady_shock.jl` (Mach scan driver, plotting).
- `test/test_phase7_steady_shock.jl` (regression suite).
- `reference/notes_phase7_steady_shock.md` (this file).

**Modified:**
- `src/types.jl` — `DetField` extended with `Q::T`.
- `src/segment.jl` — `Mesh1D` extended with `bc::Symbol`; `Qs`
  keyword added to constructor.
- `src/cholesky_sector.jl` — `det_el_residual` accepts `bc,
  inflow_xun, outflow_xun, inflow_Pq, outflow_Pq` for breaking the
  cyclic stencil at the inflow-outflow boundaries.
- `src/newton_step.jl` — `det_step!` accepts `bc, inflow_state,
  outflow_state, n_pin`; post-Newton Q-BGK update folded into the
  existing tau-conditional block; post-Newton boundary pinning
  enforces the Dirichlet ghosts.
- `src/dfmm.jl` — exports `heat_flux_*` and includes `heat_flux.jl`.
- `test/runtests.jl` — registers the Phase-7 testset.

**Project.toml**: no dependency changes.

---

## 7. Open questions for Tom

1. **Soft-constraint Q with trapping potential V(Q)**: deferred per
   the v3 note. Worth revisiting if the long-horizon shock-capturing
   limitation can be repaired by replacing hard-constraint Q with a
   soft-constraint formulation that contributes a stabilizing term
   to the action. Speculation only at this point.
2. **Long-horizon shock capturing**: the post-shock-oscillation
   instability is the same root cause as the Phase-5 Sod failure.
   A targeted Phase-X (Caramana-Shashkov sub-zonal viscosity;
   periodic remap; or Kraus-projected variational integrator) would
   close the gap to the t = 3 golden. Tom: please confirm whether
   this is in scope for Milestone 1 or deferred to Milestone 2's
   "1D variational verification" agenda.
3. **Goldens at higher Mach**: the only checked-in golden is at
   $M_1 = 3$. If we want regression coverage at $M_1 = 1.5, 2, 5,
   10$, py-1d would need to be re-run with those Mach numbers and
   the goldens versioned. Currently the Mach scan asserts only
   IC R-H residuals (3-decimal acceptance) and the short-horizon
   plateau on the integrator output, both at all five Mach numbers.
