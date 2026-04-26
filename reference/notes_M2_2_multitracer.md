# Phase M2-2 — Tier B.6 multi-tracer fidelity in 1D wave-pool turbulence

**Status:** complete. Variational tracer transport remains
**bit-exact** under Phase-8 stochastic injection in 1D wave-pool
turbulence; the Eulerian upwind reference smears each step interface
to 25–114 cells over 500 stochastic timesteps at N = 128. Bit-
exactness holds even when the wave-pool fluid hits the long-time
realizability instability (the tracer matrix is never written to,
regardless of fluid state).

## The structural argument (carries over from Phase 11)

The B.6 acceptance is a strict superset of B.5: the same Phase-11
structural argument applies, and one only needs to verify that the
Phase-8 stochastic-injection module also leaves the tracer matrix
out of its write set.

Concretely:

* `det_step!`'s write set: `(x, u, α, β, s, Pp, Q, p_half)` —
  see `src/newton_step.jl`.
* `inject_vg_noise!`'s write set: `(u, s, Pp, p_half)` — see
  `src/stochastic_injection.jl` lines 376–399.
* `det_run_stochastic!` is `det_step! + inject_vg_noise!` per
  step, so the union write set is still
  `(x, u, α, β, s, Pp, Q, p_half)` — *not* `tm.tracers`.

Hence in 1D / Milestone 2 (no remap yet), the deterministic
*and* stochastic numerical diffusion of any passive scalar carried
in a `TracerMesh` is **literally 0.0**. The matrix array is never
in the write set of the integrator.

This is the strongest possible bit-exactness statement: not
"machine epsilon", not "round-off bounded" — *literal zero*, by
inspection of the code.

## Bit-exactness verification (production run)

`experiments/B6_multitracer_wavepool.jl::main_b6_multitracer_wavepool()`
runs the variational + stochastic integrator on a periodic
broadband wave-pool IC (`setup_kmles_wavepool` with `u_RMS = 0.3`,
`P_0 = 1.0`, `K_max = 8`, seed = 2026) for 500 stochastic timesteps
with five concurrent step tracers at `x ∈ {0.2, 0.3, 0.5, 0.7, 0.85}`,
under calibrated `(C_A = 0.336, C_B/2 = 0.274, λ ≈ 6.67)` noise
parameters (half-amplitude `C_B` to stay clear of the long-time
realizability instability that M2-3 will fix).

Acceptance:

* `tm.tracers === tracers_initial` (same object identity)
* `tm.tracers == tracers_initial` (elementwise equal)
* `maximum(abs.(tm.tracers .- tracers_initial)) === 0.0`

All three pass. The same is true under the unscaled production
calibration `(C_B = 0.548)` for the 841 steps that complete before
the realizability instability triggers; even at the moment of the
fluid blow-up, the tracer matrix is bit-identical to its initial
state.

## Fidelity comparison vs Eulerian upwind

Headline production numbers at `N = 128`, `n_steps = 500`,
`dt = 1e-3`, `cb_scale = 0.5` (5 step tracers at distinct positions):

| tracer | step at x | variational width | Eulerian width | ratio  |
|--------|-----------|-------------------|----------------|--------|
| 1      | 0.20      | **0 cells**       | 114 cells      | 114×   |
| 2      | 0.30      | **0 cells**       | 100 cells      | 100×   |
| 3      | 0.50      | **0 cells**       |  72 cells      |  72×   |
| 4      | 0.70      | **0 cells**       |  43 cells      |  43×   |
| 5      | 0.85      | **0 cells**       |  25 cells      |  25×   |

The variational widths are *identically* zero — every step retains
its sharp `0 → 1` discontinuity at the original cell boundary; no
cell sits in the `[0.05, 0.95]` transition band. The Eulerian
upwind reference, fed the same coarse-grained velocity history,
smears each step over tens to hundreds of cells. The position-
dependence of the Eulerian smear (114 cells at x = 0.2, 25 cells at
x = 0.85) reflects the random-phase wave-pool's per-tracer
exposure to compressive vs expansive flow during the run.

Even on the *worst* tracer (smallest Eulerian smear, 25 cells), the
ratio is 25×, well above the "≥ 1 decade" target. The median ratio
across the five tracers is 72×.

## No cross-tracer contamination

`@testset "M2-2.2"` initialises 6 step tracers at distinct
positions, runs 100 stochastic steps, and verifies:

* per-row bit-equality with the IC (`tm.tracers[k, :] == initial[k]`),
* per-row L1 mass conservation (`sum(tm.tracers[k, :])` exact),
* per-row value set is `⊆ {0.0, 1.0}` (no smearing into intermediate
  values).

All pass. Cross-tracer contamination is structurally impossible
since the integrator never writes to *any* row of the tracer
matrix.

## Long-time behaviour and the M2-3 realizability instability

At unscaled production calibration (`C_B = 0.548`), the wave-pool
hits a `non-finite divu` at step **842** (consistent with the
~950-step instability documented in `notes_phase8_stochastic_injection.md`).
The proximate cause is one cell crossing `M_vv → 0` after a heavy-
tail VG draw extracts more internal energy than the local IE
reservoir; the next Newton iteration then sees `M_vv < floor` and
diverges to NaN.

M2-3 is implementing the realizability projection that fixes this.
Until that lands, the recommended driver knob is `cb_scale = 0.5`
(half-amplitude `C_B`), which completes 500+ steps reliably while
still exercising the stochastic regime. With `cb_scale = 0.5` the
500-step run shows no instability and tracer bit-exactness holds
end-to-end.

**Crucially, the bit-exactness assertion is independent of the
fluid solution being valid.** Even at step 842 of the unscaled
run — the moment the fluid blows up — the tracer matrix is still
bit-identical to its initial state (`L_inf_tracer_change = 0.0`,
`same_object_identity = true`). This is the structural strength of
the variational tracer scheme: tracer fidelity does not require
fluid stability.

## Why the structural argument is "trivially" true here

A reviewer might object that the test is vacuous: of course the
tracer matrix doesn't change if no code writes to it. The point of
the test is **defensive** — to catch a future refactor that
accidentally puts `tm.tracers` in the write set of either
`det_step!` or `inject_vg_noise!`. The bit-exactness assertion in
`@testset "M2-2.1"` is a regression guard, not an existence proof.

The non-vacuousness check is the "stochastic injection actually
fires" sub-test (line 113 of `test_phase_M2_2_multitracer.jl`),
which compares the post-run velocity field against a pure-
deterministic run of the same length and asserts they differ on a
meaningful subset of cells. This guarantees the `inject_vg_noise!`
path was exercised during the bit-exactness window — not silently
no-op'd by `C_B = 0` or some other guard.

## Files

* `experiments/B6_multitracer_wavepool.jl` — production driver
  (`main_b6_multitracer_wavepool`), wave-pool mesh builder, 5-step
  tracer factory, Eulerian-reference replay, fidelity-table
  computation, multi-panel figure renderer.
* `test/test_phase_M2_2_multitracer.jl` — three test sets covering
  bit-exact preservation, no cross-tracer contamination, and
  Eulerian-comparison fidelity. Wall time ≈ 33 s at `N = 64`,
  150 stochastic steps.
* `reference/figs/B6_multitracer_wavepool.png` — headline figure:
  density profile + variational tracers + Eulerian tracers +
  interface-width-vs-time, for the 5-tracer 500-step production
  run.
* This notes file.

## Cross-references

* `reference/notes_phase11_passive_tracer.md` — Phase 11 / Tier B.5
  structural argument (the deterministic version of the same
  bit-exactness claim).
* `reference/notes_phase8_stochastic_injection.md` — Phase 8 /
  Tier B.4 stochastic injection mechanics + the long-time
  realizability instability flagged for M2-3.
* `specs/01_methods_paper.tex` §10.3 B.6 — methods-paper acceptance
  criterion.
* `experiments/B5_passive_tracer.jl` — the B.5 Sod-style driver
  whose multi-tracer recipe we mirror in B.6.

## Open questions / Milestone-3 hooks

* The remap-diffusion bound (Bayesian remap onto the Eulerian mesh,
  fourth-order with cubic moments, methods paper §7) is the
  natural Milestone-3 test. The `TracerMesh` API is already shaped
  to take a future `remap_tracers!(tm, eulerian_mesh)` step; B.6's
  bit-exactness will *not* survive that step (it's not supposed
  to — the remap is the place where tracer diffusion enters the
  variational scheme by construction).
* When M2-3's realizability projection lands, `cb_scale` can be
  increased to 1.0 and the production-amplitude run extended to
  many thousands of steps. The bit-exactness assertion remains
  valid by construction; the Eulerian reference will diverge
  (smear) further.
* Two-fluid extension (Milestone 3) carries a `TracerMesh` per
  species; the current 1D test infrastructure slots in without an
  API change.
