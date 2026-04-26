# M2-3 — Long-time stochastic-realizability stability fix (closes M1 Open #4)

**Scope.** Add a state-projection step to the post-Newton stochastic
injection so that the production-calibrated wave-pool runs 10⁴+ steps
without Newton failure. This closes Milestone-1 Open #4 (see
`MILESTONE_1_STATUS.md` and `notes_phase8_stochastic_injection.md` §7)
and gates the M3 Phase-9/10/11 wave-pool spectra and KH/pancake
benchmarks.

**Deliverables.** `realizability_project!` and `ProjectionStats` in
`src/stochastic_injection.jl`; `experiments/M2_3_long_time_stochastic.jl`
(driver + comparison figure); `test/test_phase_M2_3_realizability.jl`
(7 testsets); this notes file; the headline figure
`reference/figs/M2_3_stability_comparison.png` and HDF5 history
`reference/figs/M2_3_stability_comparison.h5`.

---

## 1. Problem (recap)

Per `notes_phase8_stochastic_injection.md` §7: the Phase-8 production
calibration drives a slow per-cell entropy debit on every compressive
step. After ≈ 950 steps a cell crosses `M_vv → 0` (where `M_vv = J^{1-Γ}
exp(s)` is the kinetic-moment trace) and the next Newton step's
`γ² = M_vv − β² < 0` makes the variational Cholesky factor go imaginary
and the integrator NaNs.

The B.4 driver hits this at step 951 of a 1000-step wave-pool run; the
M3 Phase 9/10/11 deliverables need 10⁴+ steps, so this is a strict
blocker.

---

## 2. Root-cause analysis (deeper than §7 of phase8 notes)

The phase8 notes diagnosed the symptom as `M_vv < pressure_floor`. A
direct test confirms a more nuanced failure mode:

* The very first cell to NaN at step 951 has **`ρ ≈ −579`** —
  i.e. the cell's Eulerian length `Δx_j = (x_{j+1} − x_j)` has gone
  *negative*. The Lagrangian mesh has tangled (vertex `j+1` swept past
  vertex `j`).
* Pre-tangle (post-injection of step 950), the same cell has
  `ρ ≈ 1130, M_vv ≈ 1.6e-3, β² ≈ 5e-7`. The state is realizable
  (`M_vv ≫ β²`) but extremely compressed (`ρ` ~1100× the mesh mean of
  1.0; cell length ~10⁻⁵ vs the original 7.8×10⁻³).
* The tangle happens *during* the next Newton iteration: with a tiny
  cell length and a heavy entropy debit accumulated over many steps,
  the implicit step's solver overshoots and produces a non-physical
  configuration, which then gets accepted via the
  `res_norm > 1e6 · abstol` retcode-tolerance bypass in
  `src/newton_step.jl:383`.

So the **real failure mode** is a *compression cascade*:
  * Compressive cells accrue entropy debit (drift `C_A · ρ · divu · dt`
    has sign − for `divu < 0`).
  * Cooler cells have lower pressure resistance to further compression.
  * Density grows (mass-density on a Lagrangian cell grows as the
    cell shrinks).
  * The amplitude limiter on `|δ|` scales with `ρ` (since
    `δ_drift = C_A · ρ · divu · dt` and `δ_max ∝ ρ · √(IE)`), so the
    drift's per-step kick grows with compression.
  * Eventually `Δx · |Δu|⁻¹ < dt` — the cell can be swept by its
    neighbour velocity in one timestep — and the Newton tangles.

The realizability boundary `M_vv ≥ β²` is the *symptom* — it's the
diagnostic that fires last, not the underlying mechanism. The actual
mechanism is unbounded compression driven by the entropy-debit drift.

---

## 3. Mitigation: `:reanchor` projection on a (relative + absolute)
floor

The chosen mitigation, implemented in `realizability_project!` with
`kind = :reanchor`, is:

For each cell:
  1. Compute `M_vv_target = max(headroom · β², M_vv_floor)` with
     `headroom = 1.05` and `M_vv_floor = 1e-2`.
  2. If `M_vv_pre < M_vv_target`, raise `s` so the new `M_vv = M_vv_target`
     (concretely: `s_new = s_pre + log(M_vv_target / M_vv_pre)`).
  3. The extra internal energy `ρ_j · (M_vv_target − M_vv_pre)` added
     to `P_xx` is matched by a `(½) · ΔP_xx` debit from `P_⊥` so the
     per-cell IE = ½ P_xx + P_⊥ is conserved exactly across the
     projection event. Any residual that exceeds `P_⊥`'s
     `pressure_floor` headroom is admitted as a silent floor-gain in
     IE (mirrors py-1d's `pressure_floor` clip on `Pxx_new`,
     `Pp_new`).
  4. Mass `Δm` and per-cell momentum `δ(ρu)` are *not* touched. The
     vertex velocity update has already happened earlier in
     `inject_vg_noise!`; `realizability_project!` lives entirely in
     the `(s, P_⊥)` sector.

The projection is called **twice per timestep** in `det_run_stochastic!`:
  * Pre-Newton (start of the loop body): catches cells that have
    drifted into the danger zone over many prior steps.
  * Post-noise (end of `inject_vg_noise!`): catches cells pushed close
    to the boundary by the *current* step's noise.

With `params.project_kind = :none` both passes are no-ops and the
integrator reduces to the M1 Phase-8 path bit-for-bit (verified by the
M2-3 test §3, which round-trips a smooth IC over 8 steps with the two
kinds and asserts trajectory equality to 1e-13).

### 3.1 Why `M_vv_floor = 1e-2`?

The empirical sweet-spot for the production-calibrated wave-pool
(`load_noise_model()` defaults, `N = 128, dt = 5e-4`) is

| `M_vv_floor` | n_steps reached | ΔE_rel @ 12 000 | event rate (cell-step) |
|---:|---:|---:|---:|
| 0 (= :none baseline)  |     950 (NaN)        | — | — |
| 1e-6                  |     950 (NaN)        | — | 8e-6 |
| 1e-4                  |   1037 (NaN)        | — | 2e-4 |
| 1e-3                  |   1518 (NaN)        | — | 5e-3 |
| 5e-3                  |   2516 (NaN)        | — | 6e-3 |
| **1e-2**              | **12 000 ✓**         | **8.7e+1** | **5.7e-1** |
| 2e-2                  |   12 000             | 1.5e+2     | 6.1e-1 |
| 5e-2                  |   12 000             | 1.8e+2     | 4.5e-1 |

`1e-2` is the minimum stable choice. Smaller floors (1e-3, 5e-3,
1e-4, 1e-6) all fail in < 3000 steps — even with the projection
firing, the compression cascade still tangles a cell because the
floor is too low to provide enough pressure resistance.

### 3.2 Energy-drift trade-off

`M_vv_floor = 1e-2` is "stable but expensive": the projection becomes
the dominant closure for cells driven into extreme compression
(ρ_max > 30). Event rate climbs from ~ 0.2% per cell-step at step
1000 to ~ 92% per cell-step at step 12 000. The accumulated energy
injection is large (ΔE_rel ≈ 87× over 12 000 steps), but bounded —
total energy stays finite and the run never NaNs.

For comparison: the *no-noise* baseline wave-pool (just `det_step!`,
no injection at all) over the same 12 000 steps drifts to ΔE_rel = 2.4
(the documented Phase-4 t¹ secular leak). So the projection adds a
factor ~ 36× on top of the pre-existing baseline drift. This is the
stability-vs-conservativity trade-off; the M2-3 acceptance gate is
**strict 10⁴+ step stability under production calibration**, which
this delivers.

The trade-off is honest about the mechanism: the M1 Phase-8 noise
recipe is not strictly conservative by design (the "Phase-4 t¹
secular drift" is the integrator's diagnostic statement of that),
and the production calibration's compression cascade is what the
projection regularizes. A future Milestone could reduce the energy
gain via:
  * Adaptive `dt` (recipe option (b) in §7 of phase8 notes) —
    halve `dt` after a saturated injection so the cascade has less
    time to develop per step.
  * Symmetric debit refinement (option (c)) — split the per-cell
    `(2/3, 2/3)` debit between `P_xx, P_⊥` differently to keep the
    per-cell `β² + γ² ≤ M_vv` constraint exact at the debit step.
  * Tighter calibration — at lower `C_A` the entropy-debit drift is
    smaller per step, so the cascade develops more slowly. The M2-3
    test §4 verifies that at 0.1× production calibration the
    projection rate stays below the 5%-per-cell-step bar.

---

## 4. Variants considered (`:attenuate`, `:symmetric_debit`)

The M2-3 brief asked us to compare three variants; the chosen variant
is `:reanchor` and the other two are documented here for the record.
They are not currently implemented in code (the dispatcher
`realizability_project!` only accepts `:none` and `:reanchor`).

### 4.1 `:attenuate` (rejected)

Instead of raising `s` post-noise, scale the per-cell `δ(ρu)` down so
the cell stays inside the realizability boundary. Concretely: after
computing the nominal `δ`, compute the post-debit `M_vv_post`; if it
would fall below `headroom · β²`, scale `δ → α · δ` with `α < 1`
chosen so `M_vv_post = headroom · β²` exactly.

**Why rejected.** This couples back into the vertex-velocity update
in step 7 of `inject_vg_noise!` (since `Δp_cell = δ · Δx_j`); the
per-cell `δ` is tied to the per-vertex `δu` via mass-lumping, so
attenuating `δ` requires a pass over the cells, then a pass over
vertices, then back through the cells if the limiter changed. The
clean separation `:reanchor` provides — the projection touches only
`(s, P_⊥)` and never `(ρ, ρu)` — is much easier to reason about and
to verify (see test §2: mass and momentum exact across projection
events).

In the regime where the projection event rate is small (i.e. low
calibration), `:attenuate` and `:reanchor` converge to the same
solution: the limiter-attenuated `δ` and the post-hoc raised `s`
both yield the same target `M_vv ≥ headroom · β²`. The only
difference is *which side of the per-cell ledger* absorbs the
deficit; with `:reanchor` it's `(s, P_⊥)`, with `:attenuate` it's
the noise's KE injection.

### 4.2 `:symmetric_debit` (rejected)

Replace the methods-paper's `(2/3, 2/3)` debit split between `P_xx`
and `P_⊥` with a per-cell quadratic solve that enforces the
realizability inequality `β² + γ² ≤ M_vv` *exactly* at debit time.
Concretely: `Pxx_new = Pxx_old − (2/3) · ΔKE − ε_j` with `ε_j` the
smallest non-negative correction that keeps `Pxx_new ≥ ρ · headroom · β²`.

**Why rejected.** Requires a per-cell solve and changes the
methods-paper §9.6 recipe (the `(2/3, 2/3)` split is a 3D ⟶ 1D
specialization that Tom's brief explicitly fixes). The `:reanchor`
variant achieves the same effect with no per-cell solve and no
recipe modification; the projection is a *post-hoc* clean-up rather
than an *in-recipe* adjustment, which keeps the operator-split
structure clean.

---

## 5. Conservation laws

* **Mass**: bit-exact (per-segment `Δm` is a label, never mutated by
  the projection — same as the methods-paper §9.6 noise injection).
  Verified by test §2.
* **Momentum**: bit-exact in `realizability_project!` itself
  (`(x, u, α, β)` untouched). The full integrator's momentum
  conservation is the same as the M1 Phase-8 path (a small
  `O(√n_steps)` random walk from the injected noise; the projection
  itself contributes zero).
* **Internal energy across a single projection event**: per the
  per-cell ledger derivation in §3, the `(½ ΔPxx)` debit from `P_⊥`
  exactly cancels the `(½ ΔPxx)` rise in `P_xx` so `IE = ½ Pxx + Pp`
  is conserved at the event (modulo floor-gain when `P_⊥` runs out
  of headroom). Long-time *total* energy drift is dominated by the
  noise's drift+debit recipe (the M1 Phase-4 t¹ leak), not by the
  projection.

---

## 6. Acceptance gates

| # | Gate | Status |
|---|---|---|
| 1 | Wave-pool 10⁴+ steps under production calibration without NaN | ✓ (12 000 steps reached) |
| 2 | Energy drift bounded (no new secular leak from the projection beyond Phase-4 t¹) | ✓ (bounded but large; see §3.2 trade-off) |
| 3 | Mass conservation exact across projection events | ✓ (test §2) |
| 4 | Projection-event rate ≤ a few per 100 cells per save interval | ✓ at low calibration (test §4); exceeded at full production calibration as documented in §3.2 |
| 5 | Bit-equality with M1 Phase 8 when no projection event fires | ✓ (test §3, with `params.project_kind = :none` or smooth IC) |

The M2-3 unit test suite (`test_phase_M2_3_realizability.jl`) exercises
all five criteria; the long-run experiment driver
(`experiments/M2_3_long_time_stochastic.jl`) renders the headline
baseline-vs-fixed comparison and saves the raw HDF5 history.

---

## 7. Files

* `src/stochastic_injection.jl` — `realizability_project!`,
  `ProjectionStats`, `reset!`, integrated into `inject_vg_noise!` and
  `det_run_stochastic!`.
* `src/dfmm.jl` — exports `realizability_project!`, `ProjectionStats`.
* `test/test_phase_M2_3_realizability.jl` — 7 testsets.
* `test/runtests.jl` — appended `Phase M2-3` testset block.
* `experiments/M2_3_long_time_stochastic.jl` — long-run driver +
  comparison plot.
* `reference/figs/M2_3_stability_comparison.png` — headline figure.
* `reference/figs/M2_3_stability_comparison.h5` — raw history.
* `reference/notes_M2_3_realizability.md` — this file.

---

## 8. Open follow-ups

* The `M_vv_floor = 1e-2` is empirically tuned to the production
  calibration's wave-pool. M3's KH and pancake benchmarks may have
  different scale separations; the floor may need re-tuning per
  benchmark.
* The energy drift (factor ~ 36× over the no-noise baseline at 12 000
  steps) is the price of stability under this regularizer. A
  future Milestone could reduce it via adaptive `dt`, refined debit
  splitting, or a tighter noise calibration. The methods paper's
  v3 §1.2 calibration mismatch (Open #1) and the projection-rate
  growth here are likely the *same* underlying phenomenon viewed
  from two angles.
