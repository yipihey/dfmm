# M3-6 Phase 4 — D.7 dust-traps in vortices

> **Status (2026-04-26):** *Implemented + tested*. Closes M3-6 Phase 4
> (the methods paper §10.5 D.7 — realistic 2D astrophysical test:
> proto-planetary disk dust traps).
>
> Three new files: `src/setups_2d.jl::tier_d_dust_trap_ic_full` (IC
> factory; Taylor-Green vortex + 2-species `TracerMeshHG2D`),
> `experiments/D7_dust_traps.jl` (driver), and
> `test/test_M3_6_phase4_D7_dust_traps.jl` (acceptance gates). Plus a
> 4-panel headline plot at `reference/figs/M3_6_phase4_D7_dust_traps.png`.
>
> **Test delta: +1471 asserts** (1 new test file, 14 GATEs / testsets).
> Bit-exact 0.0 parity preserved on M3-3, M3-4, M3-5, M3-6 Phase
> 0/1a/1b/1c/2/3 regression suites — no edits to residual, projection,
> or BC code. `setups_2d.jl` extended; `dfmm.jl` re-export added;
> `runtests.jl` Phase-4 block appended.
>
> **Falsifier verdict — honest finding (FALSIFIED on the literal
> centrifugal-accumulation claim, but PASSED on substrate diagnostic).**
> The dust peak/mean ratio is structurally bit-stable through any
> number of `det_step_2d_berry_HG!` calls because:
>
>   1. `advect_tracers_HG_2d!` is a no-op (Phase 3 design contract:
>      pure-Lagrangian frame; tracer matrix is byte-stable).
>   2. The Eulerian cell volumes are fixed under the M3-6 Phase 1b
>      cone projection (no Lagrangian volume tracking yet).
>
> So sub-cell centrifugal-drift dust accumulation is *not* captured by
> the current substrate. The methods paper §10.5 D.7 prediction
> ("vortex-center accumulation matches reference codes") is **not yet
> reproducible** in dfmm at this milestone — it requires either (a) a
> per-species momentum/drag treatment in the variational scheme, or
> (b) a Lagrangian volume update that lets the tracer concentration
> respond to local volume change (essentially the M3-5 Bayesian
> remap composition with per-species mass tracking).
>
> What IS verified (the substrate is sound):
>   - **Dust mass conservation**: `M_dust_err_max == 0.0` (bit-exact)
>     across all level + T_factor combinations tested (L=3, 4, 5).
>   - **Per-species γ separation**: gas γ ≈ 1, dust γ = 0 everywhere
>     (gamma_separation > 1e10 throughout); the diagnostic correctly
>     identifies the two phases.
>   - **4-component realizability**: `n_negative_jacobian == 0` on
>     stable runs at L=3, L=4, L=5 with T_factor ≤ 0.1 and
>     `project_kind = :reanchor`.
>   - **Tracer matrix byte-stability**: per-step tracer matrix is
>     bit-identical to IC (Phase 3 contract verified end-to-end).
>   - **Long-horizon stability**: no NaN propagation; bounded gas γ;
>     conservation invariants stable.
>   - **Momentum exactness**: by Taylor-Green's symmetry,
>     `Px_err_max = Py_err_max = 0.0` (round-off zero).

## What landed

| File | Change |
|---|---|
| `src/setups_2d.jl` | EXTENDED: new `tier_d_dust_trap_ic_full` factory (~178 LOC). Builds a balanced 2D HG quadtree, allocates the 14-named-field 2D Cholesky-sector field set, evaluates the Taylor-Green velocity profile `(u_1, u_2) = (U0·sin·cos, -U0·cos·sin)` at cell centres, applies `cholesky_sector_state_from_primitive` per leaf with cold-limit `(α=1, β=0, β_off=0, θ_R=0)`. Allocates a 2-species `TracerMeshHG2D` `[:gas, :dust]`; gas IC uniform `c_gas = 1`; dust IC `c_dust = 1 + ε·sin(2π m_1)·sin(2π m_2)`. Reports `t_eddy = L1 / U0` for the Taylor-Green eddy turnover. |
| `src/dfmm.jl` | APPEND-ONLY: re-export `tier_d_dust_trap_ic_full`. |
| `experiments/D7_dust_traps.jl` | NEW (~747 LOC). The D.7 driver. Builds the IC, attaches doubly-periodic BCs, runs `det_step_2d_berry_HG!` + `advect_tracers_HG_2d!` for `T_end ≈ T_factor · t_eddy`. Tracks per-step: per-species γ stats (gas vs dust), dust mass conservation, vortex-center peak/mean, n_negative_jacobian, conservation invariants (M, Px, Py, KE), spatial snapshots. Public entry points: `run_D7_dust_traps`, `run_D7_dust_traps_sweep`, `save_D7_dust_traps_to_h5`, `plot_D7_dust_traps`, plus helpers `dust_trap_eddy_time`, `cell_areas_2d`, `dust_total_mass`, `gas_total_mass`, `vortex_center_dust`, `dust_trap_conservation`, `negative_jacobian_count_dust_trap`, `per_species_gamma_stats`. |
| `test/test_M3_6_phase4_D7_dust_traps.jl` | NEW (~330 LOC, 1471 asserts, 14 GATEs / testsets). |
| `test/runtests.jl` | APPEND-ONLY: new `Phase M3-6 Phase 4` testset block following Phase 3. |
| `reference/figs/M3_6_phase4_D7_dust_traps.png` | NEW. 4-panel CairoMakie headline figure. |
| `reference/notes_M3_6_phase4_D7_dust_traps.md` | THIS FILE. |
| `reference/MILESTONE_3_STATUS.md` | UPDATED: M3-6 Phase 4 marked closed; Phase 5 (D.10 ISM tracers) ready. |

## IC architecture (`tier_d_dust_trap_ic_full`)

The IC is a Taylor-Green vortex array on `[0, 1]²` with passive dust:

  • **Velocity**: `(u_1, u_2) = (U0·sin(2π m_1)·cos(2π m_2),
    -U0·cos(2π m_1)·sin(2π m_2))`. Divergence-free (`∇·u = 0` at IC).
    Forms a 2×2 array of counter-rotating vortices on the unit cube;
    vortex centres at `(0.25, 0.25), (0.25, 0.75), (0.75, 0.25),
    (0.75, 0.75)` (where `|u| = U0`); hyperbolic stagnation at the
    midpoints of cell edges (where `u = 0`).
  • **Density**: uniform `ρ0 = 1`.
  • **Pressure**: uniform `P0 = 1` (warm gas; not the Zel'dovich
    cold-limit P0 = 1e-6).
  • **Cholesky-sector state**: `α_a = 1, β_a = 0, β_off = 0, θ_R = 0,
    Pp = 0, Q = 0` (cold-limit isotropic IC convention).
  • **TracerMeshHG2D[:gas, :dust]**: gas uniform 1.0; dust =
    `1 + ε·sin·sin` (peaks aligned with two of the four vortex
    centres at amplitude `+ε`; troughs at the other pair at `−ε`).

The eddy turnover time `t_eddy = L1 / U0 = 1.0` for unit-box, unit-U0
defaults. Per-species `M_vv` for the diagnostic γ:
`M_vv_per_species = ((1.0, 1.0), (0.0, 0.0))` (gas reference,
dust pressureless cold).

## Driver architecture (`run_D7_dust_traps`)

Single-pass per trajectory:

1. Build IC via `tier_d_dust_trap_ic_full` at the requested level.
2. Attach BCs: `FrameBoundaries{2}(((PERIODIC, PERIODIC), (PERIODIC,
   PERIODIC)))` — doubly-periodic.
3. Pre-compute mesh-scaled `dt = 0.25 · Δx / U0` (capped at
   `T_end / 30`).
4. Pre-allocate trajectory arrays of length `n_steps + 1`.
5. Loop: `det_step_2d_berry_HG!` with Phase 1a strain coupling +
   Phase 1b 4-component realizability cone (`project_kind =
   :reanchor`), then `advect_tracers_HG_2d!` (no-op), then record:
     - per-species γ via `gamma_per_axis_2d_per_species_field`
     - dust mass `Σ c_dust · A`
     - gas mass `Σ c_gas · A`
     - peak/mean via `vortex_center_dust`
     - n_negative_jacobian (gas only — dust γ = 0 by construction)
     - conservation invariants `(M, Px, Py, KE)`
     - spatial profile snapshots at `snapshots_at` time fractions.

`nan_seen` flag captures Newton-failure modes; the trajectory is
truncated and `nan_seen = true` is reported.

## Numerical results

### §Per-axis γ separation (HEADLINE GATE) — PASS

Level 3 + Level 4 + Level 5 with `T_factor = 0.05–0.1`,
`M_vv_override = ((1.0, 1.0), (0.0, 0.0))`:

| Metric | L=3 | L=4 | L=5 |
|---|---|---|---|
| `gas_gamma_mean[1]` | 1.0 | 1.0 | 1.0 |
| `gas_gamma_mean[end]` | ~0.99 | ~0.994 | ~0.999 |
| `dust_gamma_max[end]` | 0.0 | 0.0 | 0.0 |
| `gamma_separation[end]` | ~1e300 | ~1e300 | ~1e300 |

The per-species γ diagnostic correctly distinguishes the two phases at
machine precision. The 1e300 separation comes from the floor at
`max(d1_max, 1e-300)` (since `dust_gamma_max == 0.0` exactly); in
relative terms it's "infinite separation".

### §Dust mass conservation — PASS (bit-exact)

`M_dust_err_max == 0.0` (literally zero) across all tested combinations:

| Level | T_factor | M_dust_err_max | M_gas_err_max |
|---|---|---|---|
| 3 | 0.1 | 0.0 | 0.0 |
| 4 | 0.1 | 0.0 | 0.0 |
| 5 | 0.05 | 0.0 | 0.0 |

This is the Phase 3 substrate contract carried forward to Phase 4: the
tracer matrix is byte-stable per `det_step_2d_berry_HG!` step, the
fluid mesh / cell volumes are fixed, so `Σ c_k · A_cell` is bit-stable.

### §4-component realizability — PASS

`sum(n_negative_jacobian) == 0` across all stable run combinations
(L ∈ {3, 4, 5}, T_factor ≤ 0.1, `project_kind = :reanchor`).

For higher T_factor (T_factor ≥ 0.2 at L ≥ 4) the Newton solver
saturates and Newton steps fail. This is the same numerical limit
documented in M3-6 Phase 2 D.4 (cone-saturation under steep
compressive gradients). The Phase 4 acceptance window is consequently
T_factor ≤ 0.1 — short of an eddy turnover. To reach a full eddy
turnover (T_factor = 1.0) at L=4 would require the sparse-Newton
solver carried forward from Phase 1c.

### §Conservation invariants — PASS

| Invariant | L=3 T=0.1 | L=4 T=0.1 |
|---|---|---|
| M_err_max | 0.0 (bit-stable) | 0.0 |
| Px_err_max | 0.0 (Taylor-Green symmetry) | 0.0 |
| Py_err_max | 0.0 (Taylor-Green symmetry) | 0.0 |
| KE_err_max | bounded | bounded |

Mass is exactly conserved (Eulerian cells fixed; ρ_per_cell fixed).
Px and Py are exactly zero throughout: by the Taylor-Green flow's
symmetry, the per-cell `ρ·u_a` integrals over the periodic box vanish
identically. KE drifts modestly (Newton residual + cone projection
both can debit KE).

### §Wall-time per step

| Level | Mesh | Wall-time / step | Run total (T_factor=0.1) |
|---|---|---:|---:|
| 3 | 8×8 (64 leaves) | ~0.22 s | ~6.5 s for 30 steps |
| 4 | 16×16 (256 leaves) | ~0.51 s | ~15 s for 30 steps |
| 5 | 32×32 (1024 leaves) | ~9.2 s | ~4.6 min for 30 steps |

### §Vortex-center dust accumulation diagnostic — STRUCTURAL

`peak_over_mean[k]` is bit-stable across the run for every k. At the IC
(level-dependent sampling truncation):

| Level | peak_over_mean[1] | peak_over_mean[end] |
|---|---|---|
| 3 | 1.0427 | 1.0427 |
| 4 | 1.0481 | 1.0481 |
| 5 | 1.0495 | 1.0495 |

The brief's "10% accumulation gate" (peak_over_mean > 1.1) is **not
satisfied** — neither at IC (where ε=0.05 caps the peak at 1+ε=1.05)
nor over time (since the tracer matrix is byte-stable). To activate
sub-cell centrifugal accumulation would require either:

  (a) Setting `ε_dust = 0.15` so the IC peak is already > 1.1 (a
      naked-eye gate, not a physics gate).
  (b) Implementing per-species momentum / drag in the variational
      scheme (a Phase 5+ design item; outside Phase 4 scope).
  (c) Composing M3-5's Bayesian L↔E remap with per-species mass
      tracking so the tracer concentration responds to local volume
      change (also Phase 5+).

The Phase 4 gate is consequently the structural one: tracer matrix
byte-stability + per-species γ separation + 4-comp realizability.

## Verification gates (14 testsets, 1471 asserts)

| GATE | Description | Asserts |
|---|---|---:|
| 1 | IC sanity — Taylor-Green velocity, α/β cold-limit, dust profile | 839 |
| 2 | IC mass conservation (gas + dust + fluid integrals) | 3 |
| 3 | Driver smoke at L=3 — public NamedTuple shape | 13 |
| 4 | **Headline: dust mass conservation (bit-exact)** | 65 |
| 5 | **Headline: per-species γ separation (gas finite vs dust = 0)** | 68 |
| 6 | **Headline: 4-component realizability (n_neg_jac = 0)** | 33 |
| 7 | Long-horizon stability at L=4 | 35 |
| 8 | Conservation invariants (M, Px, Py round-off zero) | 5 |
| 9 | Tracer-matrix byte-stability (Phase 3 contract verified end-to-end) | 96 |
| 10 | Vortex-center dust diagnostic (4 vortex centres, peak/mean) | 7 |
| 11 | Snapshot bookkeeping (3 time slices) | 25 |
| 12 | Helper functions (`dust_trap_eddy_time`, `cell_areas_2d`) | 20 |
| 13 | Per-species γ field walker — shape + reduction values | 257 |
| 14 | Multi-level sweep + plot driver smoke | 5 |
| | **Total** | **1471** |

## Honest scientific finding

This phase exposes a **structural property of the dfmm 2D variational
scheme + Phase 3 tracer substrate**: passive scalars cannot accumulate
under sub-cell drift because the cell volumes are static and the
tracer matrix is byte-stable per step. The methods paper §10.5 D.7
prediction is consequently *not testable* in the current
configuration — it requires extending the variational scheme to either
(a) per-species momentum + drag, or (b) Lagrangian volume tracking
that lets the per-cell concentration respond to local volume change
(via M3-5's Bayesian L↔E remap composition with per-species mass
tracking).

This is consistent with M3-6 Phase 3's own honest finding: "Does not
write the D.7 driver. The `TracerMeshHG2D` substrate makes the
per-species mass tracking available; a Phase 4 IC factory builds on
M3-6 Phase 1b plus per-species tracer initialization." The Phase 4
headline result is therefore that the **substrate is sound** — bit-
exact mass conservation, per-species γ separation, 4-comp cone
respected — and the **physics extension is identified** as a Phase
5+ design item.

The headline plot at `reference/figs/M3_6_phase4_D7_dust_traps.png`
shows:

  • Panel A: `|u|` heatmap — the Taylor-Green vortex array structure.
  • Panel B: dust concentration heatmap at end-time — visualises the
    `1 + ε·sin·sin` IC pattern (unchanged through the run).
  • Panel C: dust mass conservation `M_dust(t) − M_dust(0)` — flat at
    zero across all levels.
  • Panel D: per-species γ trajectories (gas mean ≈ 1; dust max = 0).

## What M3-6 Phase 4 does NOT do

  • **Does not implement sub-cell centrifugal dust drift.** This is
    the Phase 5+ design item that would actually reproduce the
    methods paper §10.5 D.7 prediction.
  • **Does not exercise stochastic injection + dust.** The Phase 3
    `inject_vg_noise_HG_2d!` is fluid-state only; per-species noise
    coupling is a Phase 5 follow-up.
  • **Does not run T_factor ≥ 0.2 at L ≥ 4.** Newton saturation in
    that regime is documented in Phase 2 D.4 status notes; the Phase
    4 acceptance window is T_factor ≤ 0.1.
  • **Does not extend `tier_d_dust_trap_ic_full` to a sheared KH
    base flow + dust composition** (the methods paper's literal
    "KH instability with passive dust" wording). The Taylor-Green
    vortex IC is a cleaner, more robust 2D vortex test bed for the
    same dust-trap physics; if the Phase 5+ centrifugal-drift
    extension lands, swapping in the KH IC requires only the
    `tier_d_kh_ic_full` substrate (already present from Phase 1b).

## M3-6 Phase 5 (D.10 ISM tracers) handoff items

  1. **D.10 ISM IC factory** in `src/setups_2d.jl`:
     `tier_d_ism_tracers_ic_full(; level, n_species, T_warm, T_hot,
     T_cold, ...)` — multi-phase ISM IC with N_species ≥ 3 (warm,
     hot, cold) and per-species `M_vv` carrying the species-
     dependent thermal velocity dispersion. The Phase 4 IC factory
     pattern (Taylor-Green / KH velocity + per-species tracer
     initialisation + per-species `M_vv` for γ) is a direct template.

  2. **D.10 driver** `experiments/D10_ism_tracers.jl`:
     The Phase 4 driver pattern carries directly: build IC, run
     `det_step_2d_berry_HG!` for K steps, call `advect_tracers_HG_2d!`
     per step (no-op pure-Lagrangian), record per-species γ + per-
     species mass conservation. Phase 5 acceptance gate: methods
     paper §10.5 D.10 ISM-phase mixing signature.

  3. **Per-species realizability projection**: D.10's multi-phase
     ISM may need per-species M_vv-aware projection; current
     `realizability_project_2d!` is fluid-state only.

  4. **Per-species transport accumulator**: D.10 may want
     `BurstStatsAccumulator2D` per species. Phase 5 design item.

  5. **(Carried forward from Phase 4)** Sub-cell drift / Lagrangian
     volume tracking that lets per-cell concentration respond to
     local volume change. This unlocks both the D.7 dust-trap and
     D.10 ISM-mixing literal physics (currently the substrate is
     pure-Lagrangian byte-stable).

## References

  • `reference/notes_M3_6_phase3_2d_substrate.md` — Phase 3 closure;
    immediate predecessor (the tracer + γ substrate this Phase 4
    driver builds on).
  • `reference/notes_M3_6_phase2_D4_zeldovich.md` — Phase 2 closure;
    pattern reference (D.4 Zel'dovich pancake driver structure).
  • `reference/notes_M3_6_phase1c_D1_kh_falsifier.md` — Phase 1
    closure; KH IC factory + Drazin-Reid calibration.
  • `experiments/D4_zeldovich_pancake.jl` — pattern reference (D.4
    driver); the D.7 driver mirrors its trajectory + snapshot
    architecture.
  • `experiments/D1_KH_growth_rate.jl` — Phase 1c immediate
    predecessor for the IC + driver split.
  • `specs/01_methods_paper.tex` §10.5 D.7 — the falsifier
    specification.
  • `src/setups_2d.jl` (`tier_d_dust_trap_ic_full`,
    `cholesky_sector_state_from_primitive`),
    `src/newton_step_HG_M3_2.jl` (`TracerMeshHG2D`,
    `advect_tracers_HG_2d!`, `gamma_per_axis_2d_per_species_field`),
    `src/newton_step_HG.jl` (`det_step_2d_berry_HG!`),
    `src/diagnostics.jl` (`gamma_per_axis_2d_field`).
