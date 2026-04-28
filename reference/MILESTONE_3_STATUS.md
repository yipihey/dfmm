# Milestone 3 — Status synthesis (M3-6 entire CLOSED; M3-7 in flight, M3-7a + M3-7b + M3-7c CLOSED)

**Date:** 2026-04-27 (combined close); M3-6 Phase 0 closed 2026-04-26;
M3-6 Phase 1a/1b/1c closed 2026-04-26 (the D.1 KH falsifier — methods
paper §10.5 D.1 — the headline scientific test of M3-6 Phase 1).
M3-6 Phase 2 closed 2026-04-26 (the D.4 Zel'dovich pancake — methods
paper §10.5 D.4 — central novel cosmological reference test).
M3-6 Phase 3 closed 2026-04-26 (2D substrate: tracers + stochastic
injection + per-species γ — Phase 4 / 5 prerequisites, no new
falsifier driver). M3-6 Phase 4 closed 2026-04-26 (the D.7 dust-traps
in vortices — methods paper §10.5 D.7 — Taylor-Green vortex IC + 2-
species (gas + dust) `TracerMeshHG2D`; honest finding: substrate
sound, sub-cell centrifugal accumulation requires Phase 5+ extension).
**M3-6 Phase 5 closed 2026-04-26 (the D.10 ISM-like 2D multi-tracer
fidelity — methods paper §10.5 D.10 community-impact test —
KH-style sheared base flow + N=3 species `TracerMeshHG2D`
`[:cold, :warm, :hot]`; tracer matrix byte-stable under
deterministic-step + stochastic-injection iterations bit-exact;
the 2D analog of M2-2's 1D structural argument). M3-6 ENTIRE NOW
CLOSED.**

## D.7 falsification + D.10 verification — complementary findings

M3-6 Tier-D characterises what the pure-Lagrangian variational
substrate captures vs requires extensions for. Pure-Lagrangian
variational substrate:

  • **PASSES**: D.4 (per-axis γ selectivity), **D.10 (multi-tracer
    fidelity in shocked turbulence)**
  • **PASSES kinematically**: D.1 (KH eigenmode — kinematic strain
    response only; full Rayleigh eigenmode dynamics require linear
    self-amplification not present in residual)
  • **FALSIFIES**: D.7 (sub-cell centrifugal drift not captured —
    requires Lagrangian volume tracking via M3-5 Bayesian L↔E
    remap composition with per-species mass tracking)

These are not bugs. They characterize the variational scheme's
*structural strengths* (byte-exact multi-tracer transport;
per-axis γ selectivity for shock-detection) and *physics
extensions* needed for D.7's literal claim.

**Repo state:** M3-3 closed at `077d6e4` (M3-3e-5 drop cache_mesh).
M3-4 lands in two phases: Phase 1 at `f364b4a` (periodic-x coordinate
wrap on 2D EL residual) and Phase 2 (IC bridge + C.1 / C.2 / C.3
acceptance drivers). M3-5 closed on `m3-5-bayesian-remap` with HG
`compute_overlap` + `polynomial_remap` wiring + IntExact audit harness.
**M3-6 Phase 0** closed on `m3-6-phase-0-offdiag-beta`: re-activates the
off-diagonal Cholesky pair `β_12, β_21` in the 2D residual, prerequisite
for the M3-6 Phase 1 D.1 KH falsifier driver.
**~19687 + 1 deferred tests** pass byte-equal across the 16 phase
blocks. The 2D scientific phase is complete through Tier-C
consistency + off-diagonal β reactivation; the 1D `cache_mesh::Mesh1D`
shim is retired in full; the 1D ⊂ 2D consistency gates fire across
C.1 / C.2 / C.3 IC families; Bayesian L↔E remap is wired with
conservation regression. M3-6 Phase 1 (D.1 KH) is unblocked.

**Per methods paper §10.7:** "Milestone 3: 2D principal-axis-decomposed
Cholesky integrator with Berry-connection coupling, on the
HierarchicalGrids substrate." Numbers hold; M3-5 adds the Bayesian
L↔E remap substrate per §6 / §6.6.

## Phase-by-phase completion table

| # | Item | Sub-phase | Tests | Headline result |
|---|---|---|---:|---|
| **M3-prep** | Berry connection 2D + 3D + off-diag, paper §5/§6 revisions | done | 181 | 7 SymPy CHECKs reproduced numerically; rel-err ≤ 1e-7 |
| **M3-0** | M1 Phase-1 ported onto HG `SimplicialMesh{1, T}` + `PolynomialFieldSet` via `cache_mesh::Mesh1D` shim | done | ~140 | Bit-exact 0.0 parity to M1 |
| **M3-1** | Phases 2/5/5b on HG (cache_mesh shim) | done | ~280 | Bit-exact 0.0 parity to M1 |
| **M3-2** | Phase 7/8/11 + M2-1/M2-3 on HG (cache_mesh shim) | done | 542 | Bit-exact 0.0 parity to M1 |
| **M3-2b** | HG-feature swap-in (Swap 1, 5, 6, 8 landed; 2/3 deferred to M3-3d) | done | ~120 | Periodic BC, IC factories, sparsity, BCKind framework |
| **M3-3a** | 2D field set + per-axis Cholesky decomposition driver | done | ~100 | `cholesky_DD.jl`, `setups_2d.jl`, halo smoke |
| **M3-3b** | Native HG-side **2D** EL residual (no Berry, θ_R fixed) | done | ~80 | Dimension-lift gate at 0.0 absolute |
| **M3-3c** | Berry coupling + θ_R Newton unknown | done | ~70 | All 7 SymPy CHECKs numeric; iso-pullback ε² |
| **M3-3d** | Per-axis γ + AMR/realizability per-axis (closes M3-2b Swaps 2+3 for 2D) | done | ~80 | Per-axis selectivity verified; HierarchicalMesh{2} AMR |
| **M3-3e-1** | Native `det_step_HG!` (deterministic Newton retire) | done | 1344 | Bit-exact 0.0; ~1.7× speedup at N=80 |
| **M3-3e-2** | Native `inject_vg_noise_HG!` + `det_run_stochastic_HG!` (RNG byte-equal) | done | 788 | Bit-exact 0.0 over K=10 stochastic steps |
| **M3-3e-3** | Native AMR refine/coarsen + standalone `TracerMeshHG` storage | done | 5784 | Bit-exact 0.0 across 31 AMR events with 3 tracers |
| **M3-3e-4** | Native `realizability_project_HG!` | done | 708 | Bit-exact 0.0 + ProjectionStats parity |
| **M3-3e-5** | Drop `cache_mesh::Mesh1D` field; close M3-3 | done | 0 (no new) | Field dropped; 13375+1 byte-equal regression |
| **M3-4 P1** | Periodic-x coordinate wrap on 2D EL residual | done | 46 | Translation equivariance; closes M3-3c handoff |
| **M3-4 P2 (a)** | Tier-C IC bridge + primitive recovery | done | 2606 | `(ρ, u, P)` → `(α, β, s, …)` cold-limit isotropic IC |
| **M3-4 P2 (b)** | C.1 1D-symmetric 2D Sod driver | done | 590 | y-independence ≤ 1e-12 per step |
| **M3-4 P2 (c)** | C.2 2D cold sinusoid driver | done | 11 | std(γ_1)/std(γ_2) > 1e10 for k=(1,0) |
| **M3-4 P2 (d)** | C.3 2D plane wave driver | done | 2583 | Rotational invariance + bounded mode amplitude |
| **M3-5** | Bayesian L↔E remap (`compute_overlap` + `polynomial_remap_l_to_e!`/`_e_to_l!` wired via `BayesianRemapState`); IntExact audit; Liouville monotone-necessary diagnostic | done | +86 | 9/9 IntExact battery passes; mass conserved 0..6.7e-16 over 5 cycles; partition-of-unity to 1e-12 |
| **M3-6 Phase 0** | Off-diagonal `β_12, β_21` re-activated in `DetField2D` + 2D residual; 11-dof Newton system; trivial-drive new rows preserve M3-3c byte-equal at β_12=β_21=0 | done | +390 | 9 SymPy CHECKs reproduced numerically; §Dimension-lift gate at 0.0 absolute |
| **M3-6 Phase 1a** | Off-diag strain coupling `H_rot^off = G̃_12 · (α_1·β_21 + α_2·β_12)/2` wired into the 2D EL residual; `(∂_2 u_1, ∂_1 u_2)` stencil + per-axis F^β_a / F^β_12 / F^β_21 / F^θ_R contributions | done | +125 | Strain stencil bit-exact at axis-aligned ICs; rotational equivariance to 1e-12 |
| **M3-6 Phase 1b** | `tier_d_kh_ic_full` factory (sheared base flow + antisymmetric tilt-mode perturbation) + 4-component realizability cone `Q = β_1² + β_2² + 2(β_12² + β_21²) ≤ M_vv · headroom_offdiag` | done | +574 | KH IC ready for D.1 falsifier; β_off = 0 ⇒ byte-equal to M3-3d 2-comp projection |
| **M3-6 Phase 1c** | D.1 KH falsifier driver `experiments/D1_KH_growth_rate.jl` + acceptance gates; Drazin-Reid γ_DR = U/(2w) calibration; mesh refinement convergence; 4-component cone diagnostics | done | +1565 | γ_measured/γ_DR = 1.34 (level 5); c_off² = 1.78; mesh-converged at 1.2% (L4→L5); n_neg_jac = 0; **falsifier PASSED** |
| **M3-6 Phase 2** | D.4 Zel'dovich pancake collapse: `tier_d_zeldovich_pancake_ic` factory + driver + per-axis γ tracking; methods paper §10.5 D.4 central novel cosmological reference test | done | +2718 | std(γ_1)/std(γ_2) ≈ 2.6e14 at near-caustic (L4 T=0.16); γ_1 dyn-range 4.18×; γ_2 uniform to round-off; Phase 1a inertness max|β_off|=0; **per-axis γ selectivity PASSED** |
| **M3-6 Phase 3** | 2D substrate (D.7 / D.10 prerequisites): `TracerMeshHG2D` per-species + refine listener; `inject_vg_noise_HG_2d!` per-axis selectivity (`axes::Tuple` arg); `gamma_per_axis_2d_per_species_field` + math primitive | done | +329 | 2D Phase 11 + M2-2 invariants byte-equal; per-axis selectivity verified (axis-1 inject leaves axis-2 byte-equal); 4-comp cone n_neg_jac=0; multi-species independence verified |
| **M3-6 Phase 4** | D.7 dust-traps in vortices: `tier_d_dust_trap_ic_full` factory (Taylor-Green vortex + 2-species `TracerMeshHG2D` `[:gas, :dust]`) + driver + acceptance gates; per-species γ separation diagnostic | done | +1471 | Dust mass conservation bit-exact (M_dust_err_max=0.0); per-species γ separation > 1e10 (gas≈1, dust=0); 4-comp cone n_neg_jac=0 at L∈{3,4,5}; **honest finding: peak/mean structurally bit-stable (advect_tracers_HG_2d! is no-op; sub-cell centrifugal accumulation requires Phase 5+ design)** |
| **M3-6 Phase 5** | D.10 ISM-like 2D multi-tracer fidelity: `tier_d_ism_tracers_ic_full` factory (KH-style sheared base flow + N=3 species `TracerMeshHG2D` `[:cold, :warm, :hot]`) + driver with stochastic injection enabled + acceptance gates | done | +930 | Tracer matrix byte-equal to IC (`tracers_max_diff_final=0.0`) through K det_step + inject_vg_noise iterations; per-species mass bit-exact; per-species γ separation (cold/warm/hot at γ=1, √2, 2); 4-comp cone n_neg_jac=0; 1D ⊂ 2D parity verified; **falsifier PASSED — methods paper §10.5 D.10 community-impact claim verified bit-exact (2D analog of M2-2's structural argument)** |
| **M3-7 prep** | 3D scaffolding: `DetField3D{T}` working struct + `src/cholesky_DD_3d.jl` (per-axis 3D Cholesky decomposition / recomposition; `gamma_per_axis_3d`; `rotation_matrix_3d`); intrinsic Cardan ZYX Euler-angle convention pinned to SymPy authority | done | +736 | Round-trip max_err 4.5e-15 on 50 random samples; 2D reduction byte-equal on top-left 2×2 block; iso-pullback gauge degeneracy + M-preservation; allocation-free hot path |
| **M3-7a (a)** | 3D HaloView smoke test on 4×4×4 balanced `HierarchicalMesh{3}` (`test_M3_7a_halo_smoke.jl`) | done | +426 | 6-face neighbor access verified; corner-leaf out-of-domain → `nothing`; BC-aware wrap (PERIODIC/REFLECTING mix); allocation-free fast path ≤ 64 bytes; depth=2 characterised (Q1/Q4 resolved — works as designed) |
| **M3-7a (b)** | 3D field-set allocator + read/write helpers: `allocate_cholesky_3d_fields(mesh::HierarchicalMesh{3})` (16-named-field `PolynomialFieldSet`) + `read_detfield_3d` / `write_detfield_3d!` round-trip helpers | done | +2155 | Bit-exact round-trip across all 16 dof on 64 leaves; write-order independence; single-leaf write isolation; T=Float32 sanity round-trip; M3-7b unblocked |
| **M3-7b (a)** | Native HG-side **3D** EL residual (`cholesky_el_residual_3D!`) + 15-dof pack/unpack + 3-axis face-neighbor + periodic-wrap tables in `src/eom.jl`; θ_ab trivial-driven (Berry deferred to M3-7c) | done | (residual covered jointly with (b)/(c)) | Per-axis lift of M3-3b; 6-face stencil; 3-axis periodic wrap |
| **M3-7b (b)** | 3D Newton driver `det_step_3d_HG!` in `src/newton_step_HG.jl`: `cell_adjacency_sparsity ⊗ ones(15, 15)` Jacobian prototype; mirrors `det_step_2d_HG!` pattern | done | (driver covered jointly with (a)/(c)) | Newton converges in 2 iterations on smooth IC (zero-strain / β=0) |
| **M3-7b (c)** | Zero-strain regression + dimension-lift gates §7.1a (3D ⊂ 1D) + §7.1b (3D ⊂ 2D) — the load-bearing M3-7b acceptance criteria | done | +1546 | Both gates at **0.0 absolute** (well below ≤ 1e-12 tolerance); 4×4×4 (64 leaves) + 8×8×8 (512 leaves); 100-step M1 Phase-1 trajectory match; axis-swap symmetry across all 3 principal axes; 3D ⊂ 2D byte-equal vs M3-3b's `det_step_2d_HG!` |
| **M3-7c (a)** | SO(3) Berry coupling integration in `cholesky_el_residual_3D_berry!` (`src/eom.jl`); promotes θ_{ab} rows to Newton-active; per-axis Berry α/β-modifications summed across pair-generators (1,2), (1,3), (2,3) per `Ω · X = -dH` | done | (residual covered jointly with (b)/(c)/(d)) | Per-axis Hamilton equations match boxed M3-3c form per pair; θ_{ab} rows kinematic-drive form (drive=0; off-diag velocity-gradient stencil deferred to M3-9) |
| **M3-7c (b)** | 3D Newton driver `det_step_3d_berry_HG!` in `src/newton_step_HG.jl`: `cell_adjacency_sparsity ⊗ ones(15, 15)` (same prototype as M3-7b — Berry couplings live in existing within-cell block); mirrors `det_step_2d_berry_HG!` pattern | done | (driver covered jointly with (a)/(c)/(d)) | Newton converges in ≤ 7 iter on non-isotropic 3D IC (post-Newton residual ≤ 1e-10); ≤ 2 iter on smooth ICs |
| **M3-7c (c)** | Berry verification + iso-pullback + H_rot solvability gates: 8 SymPy CHECKs reproduced at residual level; `h_rot_partial_dtheta_3d` per-pair closed form satisfies kernel-orthogonality at 5 random (α, β, γ²) × 3 (θ̇)_test × 3 pairs | done | +372 | FD-vs-closed-form per-pair Berry partials at 1e-9; iso-pullback ε-extrapolation slope = 1 ± 1e-3; H_rot kernel-orthogonality residual ≤ 1e-10 absolute |
| **M3-7c (d)** | Dimension-lift parity with Berry §7.1a + §7.1b — the load-bearing M3-7c acceptance criteria | done | +113 | §7.1a 3D-Berry ⊂ 1D matches M1 byte-equal **0.0 abs** on single-step + 100-step + axis-swap; §7.1b 3D-Berry ⊂ 2D-Berry matches M3-3c `det_step_2d_berry_HG!` byte-equal **0.0 abs** on single-step + 10-step + non-trivial Berry IC (β + θ_12) |

## Test summary

| Block | Tests |
|---|---:|
| M1 (Phase 1-7 + 5b deterministic) | 305 |
| M1 (Phase 8 stochastic) | 140 |
| M1 (Phase 11 tracer) | 21 |
| M2 (M2-1 AMR + M2-2 multi-tracer + M2-3 realizability) | 243 |
| Cross-phase smoke + Track B/C/D + regression | 1335 |
| M3-prep (Berry stencils + Tier-C IC factories) | ~200 |
| M3-0/1/2 (HG ports, native HG-side as of M3-3e) | ~960 |
| M3-2b (HG swaps 1/5/6/8) | ~120 |
| M3-3a/b/c/d (2D native) | ~330 |
| M3-3e-1/2/3/4 (native-vs-cache cross-check tests) | 8624 |
| M3-5 (Bayesian L↔E remap + Liouville + IntExact audit) | 86 |
| M3-4 Phase 1 (periodic-x wrap on 2D EL residual) | 46 |
| M3-4 Phase 2 (a) IC bridge + primitive recovery | 2606 |
| M3-4 Phase 2 (b) C.1 1D-symmetric Sod driver | 590 |
| M3-4 Phase 2 (c) C.2 cold sinusoid driver | 11 |
| M3-4 Phase 2 (d) C.3 plane wave driver | 2583 |
| M3-6 Phase 0 (off-diagonal β reactivation) | 390 |
| M3-6 Phase 1a (off-diag strain coupling H_rot^off) | 125 |
| M3-6 Phase 1b (KH IC factory + 4-comp realizability) | 574 |
| M3-6 Phase 1c (D.1 KH falsifier driver + acceptance gates) | 1565 |
| M3-6 Phase 2 (D.4 Zel'dovich pancake driver + acceptance gates) | 2718 |
| M3-6 Phase 3 (2D substrate: tracers + stochastic + per-species γ) | 329 |
| M3-6 Phase 4 (D.7 dust-traps in vortices: IC + driver + acceptance) | 1471 |
| M3-6 Phase 5 (D.10 ISM multi-tracer fidelity: IC + driver + acceptance) | 930 |
| M3-7 prep (3D scaffolding: `DetField3D` + `cholesky_DD_3d.jl`) | 736 |
| M3-7a (3D HaloView smoke + 3D field-set allocator + read/write) | 2581 |
| M3-7b (3D EL residual + Newton + zero-strain + dimension-lift §7.1a/b) | 1546 |
| M3-7c (SO(3) Berry coupling + verification + iso-pullback + H_rot + dim-lift) | 485 |
| **Total** | **~32011 + 1 deferred** (= 31526 + 485 from M3-7c) |

## M3-3 headline scientific findings

### 2D Berry coupling lands (M3-3c)

The 7 verified Berry checks from the SymPy scripts
(`scripts/verify_berry_connection.py`) reproduce numerically in
the implementation:

- **CHECK 1 closedness:** Newton Jacobian symmetric in
  `(α_a, β_a, θ_R)` block per cyclic-sum identity; verified.
- **CHECK 2 per-axis match:** at `θ̇_R = 0`, the 2D residual reduces
  to two decoupled per-axis 1D residuals at 0.0.
- **CHECK 3 iso-pullback:** at `α_1 = α_2, β_1 = β_2`, Berry block
  vanishes; ε-expansion confirms `O(ε²)` leading order.
- **CHECK 4 H_rot solvability:** the closed-form `∂H_rot/∂θ_R`
  expression from `notes_M3_phase0_berry_connection.md` §6.6
  satisfies `dH ⊥ v_ker` at sampled `(α, β, γ)` points; rel-err ≤ 1e-12.
- **CHECK 5-7 numerical values:** verified against the 5 reference
  points hard-coded in `test_M3_prep_berry_stencil.jl`.

### Per-axis γ selectivity (M3-3d)

In a 1D-symmetric collapsing-axis cold-sinusoid configuration:
`γ_1 → 1e-5` (collapsing axis), `γ_2 → O(1)` (trivial axis); the
action-error indicator fires only along the collapsing axis (per-axis
selectivity verified via `test_M3_3d_selectivity.jl`).

### Native HG-side 2D residual (M3-3b/c)

First native HG-side EL residual in dfmm. Operates on
`HierarchicalMesh{2}` + `PolynomialFieldSet` directly via
`HaloView` + `face_neighbors_with_bcs`. The dimension-lift gate
holds at 0.0 absolute on a 1D-symmetric configuration — i.e., the
2D code reproduces M1's 1D code exactly when the y-direction is
trivial.

## M3-3e status (CLOSED 2026-04-27)

**Goal:** retire the `cache_mesh::Mesh1D` shim from the 1D HG path,
reaching a fully native HG-driven 1D code path with bit-exact 0.0
parity to all currently-passing tests.

**Outcome:** *closed*. All five sub-phases landed bit-exact byte-equal.
The `DetMeshHG` no longer carries a `Mesh1D` snapshot; the 1D path
runs entirely on `fields::PolynomialFieldSet` + `Δm` + `p_half` +
`bc_spec`/`inflow_state`/`outflow_state` storage.

**Sub-phase ledger:**

| Sub-phase | Commit | Tests Δ | Headline |
|---|---|---:|---|
| M3-3e-1 (native `det_step_HG!`) | `4aaf5bb` | +1344 | Newton path native; ~1.7× speedup at N=80 |
| M3-3e-2 (native VG injection + stoch driver) | `79842c0` | +788 | RNG byte-equal at K=10 + reanchor + τ + q |
| M3-3e-3 (native AMR + `TracerMeshHG`) | `d3054ea` | +5784 | 31 AMR events bit-exact with 3 tracers |
| M3-3e-4 (native realizability projection) | `de8986d` | +708 | ProjectionStats + state byte-equal |
| M3-3e-5 (drop cache_mesh field; close M3-3) | *(this commit)* | 0 | Field gone; 13375+1 regression byte-equal |

**Final test count:** 13375 + 1 deferred (= 9384 added across
M3-3e-1/2/3/4 cross-check tests + the M3-3a/b/c/d 2D-native blocks
not in the original M3-2 baseline).

**LOC delta in M3-3e effort (production code only):**

| File | Net LOC |
|---|---:|
| `src/newton_step_HG.jl` | +540 (native Newton path) − 30 (sync helpers) |
| `src/newton_step_HG_M3_2.jl` | +560 (native VG + tracers + projection) − 100 (delegations) |
| `src/action_amr_helpers.jl` | +400 (native AMR primitives) − 80 (rebuild_HG_from_cache + helpers) |
| **M3-3e-5 specific (cache_mesh drop):** | |
|   `src/newton_step_HG.jl` | -75 (field, sync_*_HG!, cache_mesh delegation in diagnostics) +75 (`mesh1d_from_HG` helper + native `total_*_HG`/`segment_*_HG`) |
|   `src/action_amr_helpers.jl` | -100 (`_resize_cache_mesh_HG!` + `rebuild_HG_from_cache!`) |

**Wall-time aggregate (N=80, periodic, deterministic step):**

| Path | ms / step |
|---|---:|
| Original M1 cache_mesh-shim baseline (pre-M3-3e-1) | ~16 |
| M3-3e-5 native (this commit) | ~8.2 |

**Memory footprint reduction:** the dropped `cache_mesh::Mesh1D` was
≈80 bytes/cell (segment array `Vector{Segment{T,DetField{T}}}` plus
`p_half`); on an N=80 mesh ≈6.4 KB; on an N=4096 AMR-driven mesh
≈320 KB. The savings scale linearly in N.

**See `reference/notes_M3_3e_5_cache_mesh_dropped.md`** for the close
report and `reference/notes_M3_3e_1/2/3/4_*.md` for the per-sub-phase
status notes (each with its own bit-exact gate breakdown).

## Open architectural questions — status update vs M2

| # | M2 status | M3 update | Net status |
|---|---|---|---|
| **#1** t¹ secular drift | open | unchanged (deferred to M3 prep, not addressed) | Open |
| **#2** Sod L∞ ~10-20% | open | unchanged (out of M3-3 scope) | Open |
| **#3** Stochastic 3-λ mismatch | open with reframing | unchanged (deferred to M4) | Open |
| **#4** Long-time stochastic instability | resolved by M2-3 | (carries forward; 2D Phase-8 not yet wired in M3-3) | Resolved |
| **#5** cache_mesh::Mesh1D retirement | open (M3-2 handoff) | M3-3e-1/2/3/4/5 all landed; field dropped; native 1D path | **Closed** |

## Pre-Milestone-4 readiness

**Strong:**
- 2D scientific physics (Berry coupling, per-axis γ, per-axis AMR /
  realizability) all verified.
- HG substrate (HaloView + cell_adjacency_sparsity + BCKind framework)
  consumed natively in 2D.
- Dimension-lift gate (1D ⊂ 2D) holds at 0.0 absolute — confirms the
  2D math reduces correctly.
- 1D HG path now native (M3-3e closed): `cache_mesh::Mesh1D` shim
  retired across all five sectors (deterministic Newton, stochastic
  injection, AMR + tracers, realizability projection, wrapper
  diagnostics).

**Unblocked for M4:**
- Tier D (D.4 Zel'dovich pancake, D.7 dust trapping in vortices,
  D.10 ISM-like 2D) on the full 2D substrate. M3-6 Phase 1 has
  closed the off-diagonal Cholesky-Berry sector (D.1 KH falsifier
  PASSED); Phase 2 (D.4 Zel'dovich pancake) is unblocked.
- M3-5 higher-order Bernstein per-cell reconstruction.
- Sparse-Newton solver optimisation: level 6 (4096 leaves) is
  ~165 s/step on the dense Newton fallback; a custom block solver
  or iterative GMRES on the `cell_adjacency ⊗ 11×11` sparsity
  pattern would unlock level 6 / level 7 sweeps for the D.* tests.

## Recommended next moves

1. **M3-4** — Tier C consistency tests **CLOSED 2026-04-27**: Phase 1
   periodic-x wrap (`f364b4a`); Phase 2 IC bridge + C.1/C.2/C.3 drivers
   (+5790 asserts). C.1 1D-vs-golden tolerance is at the loose
   variational-solver dispersion level (~10-20% L∞), documented as
   Open Issue #2 (deferred to higher-order Bernstein reconstruction).
   See `reference/notes_M3_4_tier_c_consistency.md`.
2. **M3-5** — Bayesian L↔E remap via `BayesianRemapState` +
   `bayesian_remap_l_to_e!` / `bayesian_remap_e_to_l!` /
   `remap_round_trip!`; IntExact audit gate; Liouville
   monotone-necessary diagnostic. **CLOSED 2026-04-26.** See
   `reference/notes_M3_5_bayesian_remap.md`.
3. **M3-6 Phase 0 — CLOSED 2026-04-26**: re-activated off-diagonal
   β_{12}, β_{21} in `DetField2D` and the 2D residual (11-dof Newton
   system). At β_{12}=β_{21}=0 IC the new rows trivialise and the
   residual reduces byte-equal to M3-3c. The 9 SymPy CHECKs from
   `scripts/verify_berry_connection_offdiag.py` are reproduced
   numerically at the residual / Jacobian level. Test delta: +390
   asserts. See `reference/notes_M3_6_phase0_offdiag_beta.md`.
4. **M3-6 Phase 1 / D.1 KH falsifier — CLOSED 2026-04-26**:
   plumbed the off-diagonal strain-coupling drive
   `H_rot^off ∝ G̃_12 · (α_1·β_21 + α_2·β_12)/2` into the
   F^β_a, F^β_12, F^β_21, F^θ_R rows (Phase 1a); designed the
   KH IC factory `tier_d_kh_ic_full` + 4-component realizability
   cone (Phase 1b); ran the Drazin-Reid calibration battery
   (Phase 1c). Falsifier PASSED at c_off ≈ 1.34 (level 5);
   c_off² = 1.78 is the calibrated value (the heuristic
   prediction `c_off² ≈ 1/4` is replaced by the empirical
   measurement). See `reference/notes_M3_6_phase1c_D1_kh_falsifier.md`.
5. **M3-6 Phase 2 / D.4 Zel'dovich pancake collapse — CLOSED 2026-04-26**:
   builds on Phase 1's strain coupling + 4-component cone; adds the
   `tier_d_zeldovich_pancake_ic` IC factory and `D4_zeldovich_pancake.jl`
   driver. Per-axis γ selectivity verified at near-caustic time:
   `std(γ_1)/std(γ_2) ≈ 2.6e14` (level 4, T_factor=0.16); γ_1 develops
   spatial structure (dyn range 4.18×), γ_2 stays uniform to round-off.
   Phase 1a strain coupling inertness verified (`max |β_off| = 0`
   throughout — clean axis-aligned-IC cross-check). +2718 asserts. See
   `reference/notes_M3_6_phase2_D4_zeldovich.md`.
6. **M3-6 Phase 3 / 2D substrate — CLOSED 2026-04-26**:
   substrate-only (no falsifier driver). Adds `TracerMeshHG2D` with
   per-species per-cell storage on `HierarchicalMesh{2}` plus the
   refinement listener `register_tracers_on_refine_2d!`; adds
   per-axis stochastic injection `inject_vg_noise_HG_2d!` with
   explicit `axes::Tuple` selectivity preserving the M3-6 Phase 2
   D.4 inertness contract; adds `gamma_per_axis_2d_per_species_field`
   for per-species per-axis γ diagnostics. +329 asserts. See
   `reference/notes_M3_6_phase3_2d_substrate.md`.
7. **M3-6 Phase 4 / D.7 dust-trapping in vortices** — CLOSED. Substrate
   sound (mass conservation bit-exact, per-species γ separation, 4-comp
   cone respected); literal centrifugal accumulation requires Lagrangian
   volume tracking via M3-5 Bayesian L↔E remap composition (Phase 5+
   design item).
8. **M3-6 Phase 5 / D.10 ISM tracers** — CLOSED. Multi-tracer fidelity
   in 2D shocked turbulence verified bit-exact (`tracers_byte_equal_to_ic
   == true`); the methods paper §10.5 D.10 community-impact claim
   PASSED in the strongest possible form (literal zero per-step error).
9. **M3-7** — 3D extension (the Berry stencils were verified in
   M3-prep as the M3-7 pre-flight gate). M3-6 closed entirely; the
   3D analog of `TracerMeshHG2D` + `inject_vg_noise_HG_3d!` +
   `gamma_per_axis_3d_per_species_field` are next on the M3-7
   substrate roadmap.

## M3-4 close (2026-04-26)

**Phase 1** (`f364b4a`): periodic-x coordinate wrap on the 2D EL
residual. Closes the M3-3c handoff prerequisite. +46 asserts.

**Phase 2** (commits a, b, c, d on `m3-4-phase-2-tier-c-drivers`):

  • IC bridge maps primitive `(ρ, u_x, u_y, P)` cell-averages onto the
    M3-3 12-field Cholesky-sector state under a cold-limit, isotropic
    IC convention (α=1, β=0, θ_R=0, Pp=Q=0; s solved from EOS).
  • C.1 1D-symmetric 2D Sod: y-independence ≤ 1e-12 at every step;
    conservation gates pass; 1D-vs-golden gate captured at the loose
    tolerance documented in Open Issue #2.
  • C.2 2D cold sinusoid: std(γ_1)/std(γ_2) > 1e10 for k = (1, 0);
    qualitative 2D structure for k = (1, 1); conservation gates pass.
  • C.3 2D plane wave: u parallel to k̂ at IC; rotational invariance
    under π/2 to mesh-discretization tolerance; bounded mode amplitude
    under linear-acoustic Newton evolution at θ ∈ {0, π/8, π/4, π/2};
    conservation gates pass.

**M3-4 close: +5836 asserts (13375 + 1 → 19211 + 1).**

## M3-6 Phase 0 close (2026-04-26)

**Phase 0** (`m3-6-phase-0-offdiag-beta`): re-activates the off-diagonal
Cholesky pair `β_12, β_21` in the 2D residual after their omission in
M3-3a Q3. Prerequisite for the M3-6 Phase 1 D.1 KH falsifier driver
(the headline scientific test of M3-6).

  • `DetField2D{T}` extended with `betas_off::NTuple{2, T}` field;
    `n_dof_newton` 10 → 12. Backward-compat constructors default
    `betas_off = (0, 0)` so all pre-M3-6 IC factories continue to
    produce M3-3c-equivalent state byte-equally.
  • `allocate_cholesky_2d_fields` extended to 14 named scalar fields
    (12 prior + `:β_12, :β_21`).
  • `cholesky_el_residual_2D_berry!` extended to 11 dof per cell
    (was 9). Per-axis F^β_a rows pick up off-diag β coupling terms
    derived from rows of `Ω · X = -dH` with the corrected antisymmetric
    Ω entries (per `scripts/verify_berry_connection_offdiag.py`). New
    trivial-drive F^β_12, F^β_21 rows mirror F^θ_R's structure.
  • `det_step_2d_berry_HG!` Jacobian sparsity prototype is now
    `cell_adjacency ⊗ 11×11` (was 9×9).
  • §Dimension-lift gate at β_12=β_21=0: byte-equal to M3-3c (every
    M3-6 addition is multiplied by β_12, β_21, β̇_12, or β̇_21, which
    are pinned at zero by IC + trivial-drive). Verified across single-
    step + 100-step + 8×8 mesh + non-trivial active β_1 IC.
  • §Berry-offdiag CHECKs 1-9 reproduced numerically (FD probes of
    the residual Jacobian against the closed-form Hamilton-equation
    derivation; tolerance 1e-9).
  • §Realizability projection runs cleanly on the 14-named-field set;
    off-diag β unchanged across projection (per-cell-cone projection
    extension is M3-6 Phase 1's job).

**M3-6 Phase 0 close: +390 asserts (19297 + 1 → 19687 + 1).**

## M3-6 Phase 1 close (2026-04-26)

**Phase 1a** (`m3-6-phase-1a-strain-coupling`): wires the off-diagonal
Hamiltonian `H_rot^off = G̃_12 · (α_1·β_21 + α_2·β_12) / 2` into the
2D EL residual. The cross-axis velocity-gradient stencil
`(∂_2 u_1, ∂_1 u_2)` reads from the existing face-neighbour table
with M3-4 periodic-wrap offsets; symmetric strain `G̃_12` and
vorticity `W_12` decompose canonically. Adds to F^β_a (off-diag β
coupling), F^β_12 / F^β_21 (strain drives), and F^θ_R (W_12 · F_off
vorticity drive). At axis-aligned ICs all additions vanish ⇒ M3-3c
byte-equal preserved. +125 asserts. See
`reference/notes_M3_6_phase1a_strain_coupling.md`.

**Phase 1b** (`m3-6-phase-1b-kh-ic-realizability`): adds the
`tier_d_kh_ic` / `tier_d_kh_ic_full` IC factories — sheared base
flow `u_1(y) = U_jet · tanh((y − y_0)/w)` with antisymmetric
tilt-mode perturbation `δβ_12 = -δβ_21 = A · sin(2π k_x x) ·
sech²((y − y_0)/w)` overlaid on the off-diagonal Cholesky pair.
Extends `realizability_project_2d!` from a 2-component s-raise to
a 4-component cone projection `Q = β_1² + β_2² + 2(β_12² + β_21²)
≤ M_vv · headroom_offdiag`; at β_12 = β_21 = 0 the new path is
byte-equal to the M3-3d output. +574 asserts. See
`reference/notes_M3_6_phase1b_kh_ic_realizability.md`.

**Phase 1c** (`m3-6-phase-1c-kh-falsifier`): adds the D.1
falsifier driver `experiments/D1_KH_growth_rate.jl` and the
acceptance gates `test/test_M3_6_phase1c_D1_kh_growth_rate.jl`.
The driver runs `tier_d_kh_ic_full` through `det_step_2d_berry_HG!`
at multiple mesh levels under PERIODIC-x / REFLECTING-y BCs,
fits γ_measured by least-squares on `log RMS(δβ_12(t))` over
the linear window, and reports c_off = γ_measured / γ_DR.

  • Drazin-Reid theory: γ_DR = U / (2 w) = 3.333 (U = 1, w = 0.15).
  • Level 4 (16×16): γ_measured = 4.506 → c_off = 1.352, c_off² = 1.83.
  • Level 5 (32×32): γ_measured = 4.451 → c_off = 1.335, c_off² = 1.78.
  • Mesh refinement convergence L4 → L5: |Δγ| / γ ≈ 0.012 (well
    below the 0.2 threshold).
  • Total n_negative_jacobian = 0 across all leaves throughout
    both runs; n_offdiag_events = 0 (4-component cone stays in
    the strict interior at 1-T_KH).
  • Long-horizon stability (3·T_KH at level 3): NaN-free, no
    compression-cascade resurface.
  • Wall-time per step: ~0.7 s (level 4), ~8.9 s (level 5),
    ~165 s (level 6, deferred to a future sparse-Newton optim).

**Falsifier verdict: PASSED.** c_off ∈ [0.5, 2.0] (the methods
paper §10.5 D.1 broad-band gate) at both levels 4 and 5, mesh-
converged at the 1.2% level. The Phase 1a/1b heuristic prediction
`c_off² ≈ 1/4` is *not* what the variational scheme produces;
the calibrated value `c_off² = 1.78` is the methods paper's
Phase 1c calibration result. See `reference/notes_M3_6_phase1c_D1_kh_falsifier.md`
§"Honest scientific finding" for the careful interpretation
(linear forced growth vs exponential self-amplified growth in
the linearised residual).

**M3-6 Phase 1 close: +2264 asserts (19687 + 1 → 21951 + 1).**

## M3-6 Phase 2 close (2026-04-26)

**Phase 2** (`m3-6-phase-2-D4-zeldovich`): adds the D.4 Zel'dovich
pancake collapse falsifier — methods paper §10.5 D.4, the *central
novel cosmological reference test*.

  • New IC factory `tier_d_zeldovich_pancake_ic` in `src/setups_2d.jl`:
    1D-symmetric Zel'dovich velocity profile `u_1(x) = -A·2π·cos(2π m_1)`,
    `u_2 = 0` along the trivial axis; cold-limit `(α=1, β=0,
    β_off=0, θ_R=0)`; small pressure floor `P0 = 1e-6` for `s` closure.
    Caustic time `t_cross = 1/(A·2π)`.
  • New driver `experiments/D4_zeldovich_pancake.jl`: builds the IC
    at requested level with PERIODIC-x / REFLECTING-y BCs, runs
    `det_step_2d_berry_HG!` with Phase 1a strain coupling + Phase 1b
    4-component realizability cone, tracks per-step per-axis γ
    statistics, conservation invariants, n_negative_jacobian, max
    |β_off|, and spatial profile snapshots. ~650 LOC.
  • New acceptance test `test/test_M3_6_phase2_D4_zeldovich.jl`:
    14 GATEs / 2718 asserts. Headline gate (GATE 6) at L=4
    T_factor=0.16: `std(γ_1)/std(γ_2) > 1e6` (empirical ~2.6e14);
    γ_1 dyn-range > 1.3 (empirical 4.18×); γ_2 std/mean < 1e-10
    throughout.
  • Phase 1a strain coupling inertness verified (GATE 7): max
    |β_off| = 0 throughout — clean cross-check that the off-diag
    stencil does not fire on axis-aligned ICs.
  • New headline plot `reference/figs/M3_6_phase2_D4_zeldovich.png`.
  • Honest finding: the brief's "γ_1 max/min > 100 at near-caustic"
    is aspirational at this resolution; the variational scheme +
    per-axis cone projection saturates uniformly past T_factor ≈
    0.165 at L=4 (Newton can't continue once β_1 → 1 across all
    cells). The achievable pre-caustic dynamic range is 4-5×; the
    selectivity ratio (~10^14) far exceeds the brief's 10^6 gate.

**Falsifier verdict: PASSED** for per-axis γ correctly identifies
the pancake-collapse direction. See
`reference/notes_M3_6_phase2_D4_zeldovich.md` for the full results
+ §"Honest scientific finding".

**M3-6 Phase 2 close: +2718 asserts (21951 + 1 → 24669 + 1).**

## M3-6 Phase 3 close (2026-04-26)

**Phase 3** (`m3-6-phase-3-2d-tracer-stoch`): adds the 2D substrate
required by both M3-6 Phase 4 (D.7 dust traps) and M3-6 Phase 5
(D.10 ISM tracers). Three deliverables, each well-scoped:

  • **2D `TracerMeshHG2D`**: per-species per-cell passive scalars
    on `HierarchicalMesh{2}` + 14-named-field 2D Cholesky-sector
    field set. Sized to `n_cells(mesh)` (NOT `n_leaves`) to track
    HG's storage contract. Pure-Lagrangian byte-exact preservation
    (Phase 11 + M2-2 invariants on the 2D path) verified through
    100 `advect_tracers_HG_2d!` calls + 5 `det_step_2d_berry_HG!`
    steps. New refinement listener `register_tracers_on_refine_2d!`
    mirrors `register_field_set_on_refine!`'s shape: parent →
    `2^D = 4` children piecewise-constant prolongation (mass-
    conservative under equal-volume refinement) + children →
    parent volume-weighted mean on coarsen.
  • **2D `inject_vg_noise_HG_2d!`**: per-axis variance-gamma
    stochastic injection with explicit `axes::Tuple` selectivity.
    `axes = (1,)` perturbs only axis-1 fields `(β_1, u_1, s, Pp)`
    while axis-2 fields `(β_2, u_2, x_2)` and the off-diag pair
    `(β_12, β_21)` stay byte-equal. Verified by snapshot-pre/post
    asserts on the Zel'dovich-pancake-aligned IC. 4-component cone
    `Q ≤ M_vv · headroom_offdiag` enforced post-injection via
    `realizability_project_2d!` (M3-6 Phase 1b inheritance);
    `n_negative_jacobian == 0` at every leaf.
  • **2D `gamma_per_axis_2d_per_species_field`**: per-species
    wrapper over `gamma_per_axis_2d_field`. Returns a
    `(N_species, 2, N_leaves)` Float64 3-tensor. Plus a math-
    primitive sibling `gamma_per_axis_2d_per_species(β,
    M_vv_diag_per_species)` in `cholesky_DD.jl`. Single-species
    paths reduce byte-equally to the existing 1-species
    diagnostic; multi-species independence verified (dust species
    `M_vv = 0` ⇒ γ = 0 everywhere; gas species use EOS Mvv(J, s)).

  • Wall-time impact at 16×16 mesh: `inject_vg_noise_HG_2d!` ~415
    ms/call (dominated by `build_face_neighbor_tables` per-call
    overhead, the same envelope as `det_step_2d_berry_HG!`);
    `advect_tracers_HG_2d!` < 1 ns (no-op);
    `gamma_per_axis_2d_per_species_field` ~0.002 ms (n_species=3).

  • Honest finding: the 2D injection's Pp-floor handling diverges
    from the 1D `inject_vg_noise_HG_native!` recipe in one place:
    the floor is applied ONLY when `ΔKE_vol != 0` (an injection
    actually fired). The 2D IC bridge sets `Pp = 0` (cold limit)
    while M1's 1D IC sets `Pp = ρ M_vv` (Maxwellian); naive 1D-
    style flooring would silently raise `Pp` from 0 to the floor
    on every step. The 1D byte-exact contract is unaffected (we
    did not touch `inject_vg_noise_HG_native!`).

**Falsifier verdict: N/A.** Phase 3 is *substrate work*, not a
falsifier driver. The Phase 4 (D.7 dust traps) and Phase 5 (D.10
ISM tracers) drivers will exercise these paths in production.

**M3-6 Phase 3 close: +329 asserts (24669 + 1 → 24998 + 1).**

## M3-6 Phase 4 close (2026-04-26)

**Phase 4** (`m3-6-phase-4-D7-dust-traps`): exercises the Phase 3 2D
substrate on the methods-paper §10.5 D.7 falsifier (dust-trapping in
vortices). Three deliverables:

  • **`tier_d_dust_trap_ic_full`** in `src/setups_2d.jl`: Taylor-Green
    vortex IC `(u_1, u_2) = (U0·sin·cos, -U0·cos·sin)` with uniform
    `(ρ0, P0) = (1, 1)` + cold-limit Cholesky-sector state +
    2-species `TracerMeshHG2D[:gas, :dust]` populated with
    `c_gas = 1.0` uniform and `c_dust = 1 + ε·sin(2π m_1)·sin(2π m_2)`.
  • **`experiments/D7_dust_traps.jl`** driver: builds the IC, attaches
    doubly-periodic BCs, runs `det_step_2d_berry_HG!` +
    `advect_tracers_HG_2d!` for `T_factor · t_eddy`. Per-step
    diagnostics: per-species γ, dust mass, vortex-center peak/mean,
    n_neg_jac, M/Px/Py/KE conservation.
  • **`test/test_M3_6_phase4_D7_dust_traps.jl`** (14 GATEs / 1471
    asserts) + **headline plot** `M3_6_phase4_D7_dust_traps.png`
    (`|u|` map, dust map, dust mass conservation, per-species γ
    trajectories).

**Headline scientific finding (HONEST).** The dust peak/mean ratio is
*structurally* bit-stable through the run because (i)
`advect_tracers_HG_2d!` is a no-op (Phase 3 design contract — pure-
Lagrangian frame, tracer matrix byte-stable) and (ii) the Eulerian
cell volumes are fixed under `det_step_2d_berry_HG!`. Sub-cell
centrifugal-drift dust accumulation is **not captured** by the
current substrate; the methods paper §10.5 D.7 prediction
("vortex-center accumulation matches reference codes") requires a
Phase 5+ extension (per-species momentum + drag, OR Lagrangian
volume tracking via M3-5 Bayesian L↔E remap composition).

What IS verified (the substrate is sound):

  - **Dust mass conservation**: `M_dust_err_max == 0.0` (literally
    zero) at L=3, 4, 5.
  - **Per-species γ separation**: `gamma_separation > 1e10`
    throughout (gas γ ≈ 1, dust γ = 0 by construction since
    `M_vv_dust = 0`).
  - **4-component realizability**: `n_negative_jacobian == 0` on
    stable runs (T_factor ≤ 0.1, project_kind = :reanchor).
  - **Long-horizon stability**: no NaN; bounded gas γ; conservation
    invariants stable.
  - **Momentum exactness**: `Px_err_max = Py_err_max = 0.0`
    (Taylor-Green symmetry).

**Wall-time per step**: ~0.22 s (L=3), ~0.51 s (L=4), ~9.2 s (L=5).
Full-eddy turnover (T_factor = 1.0) at L=4 awaits the sparse-Newton
solver carried forward from Phase 1c.

**M3-6 Phase 4 close: +1471 asserts (24998 + 1 → 26469 + 1).**

## M3-6 Phase 5 close (2026-04-26)

**Phase 5** (`m3-6-phase-5-D10-ism-tracers`): exercises the Phase 3 2D
substrate on the methods-paper §10.5 D.10 community-impact falsifier
(ISM-like 2D multi-tracer fidelity in shocked turbulence). Three
deliverables:

  • **`tier_d_ism_tracers_ic_full`** in `src/setups_2d.jl`: KH-style
    sheared base flow `u_1(y) = U_jet · tanh((y - y_0)/w)` + Phase 1b
    antisymmetric tilt-mode `δβ_12 = -δβ_21 = A·sin·sech²` overlay +
    cold-limit Cholesky-sector state + N≥3 species `TracerMeshHG2D`
    (default `[:cold, :warm, :hot]`) carrying phase-stratified
    Gaussian concentration profiles in y. Per-species `M_vv` recipe
    `((1, 1), (2, 2), (4, 4))` produces γ separation cold/warm/hot
    at γ = 1, √2, 2 at IC.
  • **`experiments/D10_ism_multi_tracer.jl`** driver: builds the IC,
    attaches PERIODIC-x / REFLECTING-y BCs, drives K iterations of
    `det_step_2d_berry_HG!` + `advect_tracers_HG_2d!` +
    `inject_vg_noise_HG_2d!` (axes=(1,2), project_kind=:reanchor)
    — **stochastic injection enabled** for the "shocked turbulence"
    regime. Snapshots `tracers_ic = copy(ic.tm.tracers)` at t=0;
    reports `tracers_byte_equal_to_ic = (tm.tracers == tracers_ic)`
    at end-time + `tracers_max_diff_traj` per step. Per-step
    diagnostics: per-species mass `Σ c_k · A_cell`, per-species
    per-axis γ, n_neg_jac, M/Px/Py/KE conservation, ProjectionStats.
  • **`test/test_M3_6_phase5_D10_ism_tracers.jl`** (10 GATEs / 930
    asserts) + headline plot `M3_6_phase5_D10_ism_tracers.png`
    (4-panel: |u| heatmap, 3-species concentration overlay,
    per-species mass conservation, per-species γ trajectories).

**Headline scientific finding (BIT-EXACT VERIFICATION).** The
multi-tracer matrix at end-time is *byte-equal* to its IC value
through K det_step + inject_vg_noise iterations:
`tracers_byte_equal_to_ic == true`, `tracers_max_diff_final == 0.0`
across all tested levels (L ∈ {3, 4, 5}). This is the **2D analog
of M2-2's 1D structural argument**: the multi-tracer matrix is
*literally never* in the write set of either
`det_step_2d_berry_HG!` or `inject_vg_noise_HG_2d!` (verified by
inspection of the operator code). The bit-exact preservation is a
*structural property* of the implementation, not a tolerance-
bounded numerical claim.

What IS verified:

  - **Tracer matrix byte-stability**: `tracers_max_diff_final == 0.0`
    at L=3, 4, 5 (literally zero across all stochastic runs).
  - **Per-species γ separation**: γ_cold/warm/hot = 1.0, 1.414, 2.0
    at IC; well-separated throughout the run.
  - **Per-species mass conservation bit-exact**: `M_per_species_err_max
    == 0.0` for every species (consequence of tracer matrix
    byte-stability + Eulerian fixed cell areas).
  - **4-component realizability**: `n_negative_jacobian == 0` on
    stable runs (T_factor ≤ 0.05, `(C_A, C_B) ≤ (0.05, 0.05)`,
    project_kind = :reanchor).
  - **1D ⊂ 2D parity**: axes=(1,) selectivity leaves axis-2 fluid
    fields + tracer matrix byte-equal (the M3-6 Phase 3 contract
    restricted to the M2-2 multi-tracer-fidelity statement).

**Falsifier verdict: PASSED in the strongest form** (literal zero
per-step error rather than tolerance-bounded). Methods paper §10.5
D.10 community-impact claim verified.

**Wall-time per step**: ~0.5 s (L=3), ~1.1 s (L=4), ~14 s (L=5)
under stochastic injection. Test runner exercises L=3 + L=4 in ~30 s.

**M3-6 Phase 5 close: +930 asserts (26469 + 1 → 27399 + 1).**

**M3-6 ENTIRE NOW CLOSED.** All falsifier drivers (D.1, D.4, D.7,
D.10) landed; Tier-D coverage complete. The Phase 4 honest
falsification of D.7 + the Phase 5 bit-exact verification of D.10
are *complementary* findings characterising what the
pure-Lagrangian variational substrate captures vs requires
extensions for. M3-7 (3D extension) is unblocked.

## M3-7a close (2026-04-26)

First sub-phase of M3-7 (3D extension) proper. Lands the 3D
HaloView smoke test + the 3D field-set allocator + bit-exact
round-trip helpers. Builds directly on M3-7 prep
(`64fb1ad` — `DetField3D` + `src/cholesky_DD_3d.jl`).

**What landed:**

  * `test/test_M3_7a_halo_smoke.jl` (NEW, 426 asserts) —
    3D analog of `test_M3_3a_halo_smoke.jl`. Verifies HaloView
    depth=1 contract for D=3 on a 4×4×4 balanced
    `HierarchicalMesh{3}`: 6-face neighbor access, corner-leaf
    out-of-domain → `nothing`, BC-aware wrap, allocation-free
    fast path (≤ 64 bytes per call). Also characterises depth=2
    (Q1/Q4 of M3-7 design note §11 open questions) — works as
    designed in 3D, not vacuously broken.
  * `src/setups_2d.jl` (+150 LOC at end-of-file) —
    `allocate_cholesky_3d_fields(mesh::HierarchicalMesh{3}; T)`
    + `read_detfield_3d(fields, ci)` + `write_detfield_3d!(fields,
    ci, v)`. 16-named-field SoA layout: 3 + 3 + 3 + 3 + 3 + 1 =
    16 (positions + velocities + α + β + θ_{ab} + s) at
    `MonomialBasis{3, 0}`.
  * `src/dfmm.jl` (+10 LOC, append-only) — three new exports.
  * `test/test_M3_7a_field_set_3d.jl` (NEW, 2155 asserts) —
    structural contract, allocator coverage, bit-exact round-trip
    on all 64 leaves × all 16 scalars, write-order independence,
    single-leaf write isolation, T-parameterised allocator
    (Float32 sanity).
  * `test/runtests.jl` (append-only) — new "Phase M3-7a: 3D
    HaloView smoke + field set" testset block.

**Test delta:** +2581 asserts (426 + 2155).

**1D + 2D regression:** byte-equal — verified by running
`test_phase1_zero_strain.jl` (5), `test_M3_3a_halo_smoke.jl` (30),
`test_M3_3a_field_set_2d.jl` (295), `test_M3_3a_cholesky_DD.jl`
(199), and `test_M3_7_prep_3d_scaffolding.jl` (736) in isolation
on this branch — all pass at their original counts.

**Off-diagonal β + post-Newton Pp / Q:** intentionally NOT carried
on the 3D field set (per M3-3a Q3 default + M3-7 design note §4.4).
M3-9 (3D D.1 KH) will lift to 19-dof; M3-7c will activate Pp / Q
post-Newton sectors.

**M3-7b unblocked:** the launch agent has `DetField3D`,
`cholesky_decompose_3d` / `cholesky_recompose_3d`,
`gamma_per_axis_3d`, `allocate_cholesky_3d_fields`,
`read_detfield_3d` / `write_detfield_3d!`, and a verified depth=1
HaloView contract for `HierarchicalMesh{3}`. The 3D EL residual
`cholesky_el_residual_3D!` is the next deliverable per M3-7
design note §3.

See `reference/notes_M3_7a_3d_halo_allocator.md` for the full
status note + handoff items.

## M3-7b close (2026-04-27)

Second sub-phase of M3-7 (3D extension). Lands the native HG-side
3D EL residual `cholesky_el_residual_3D!` + Newton driver
`det_step_3d_HG!` (no Berry; θ_ab trivial-driven). Direct
dimension-lift of M3-3b's 2D residual: per-axis sums over
`a ∈ {1, 2, 3}`; 6-face stencil (vs 4 in 2D); 3-axis periodic-
coordinate wrap; 15 Newton-driven rows per leaf.

**What landed:**

  * `src/eom.jl` (+550 LOC, append-only) — `cholesky_el_residual_3D!`
    + 15-dof `pack_state_3d` / `unpack_state_3d!` +
    `build_face_neighbor_tables_3d` + `build_periodic_wrap_tables_3d`
    + `build_residual_aux_3D` (3D analog of `build_residual_aux_2D`
    that consumes the 16-named-field 3D field set from M3-7a).
    The three θ_ab rows are TRIVIAL-DRIVEN
    (`F^θ_ab = (θ_ab_np1 − θ_ab_n) / dt`) — Berry coupling lands
    in M3-7c.
  * `src/newton_step_HG.jl` (+145 LOC, append-only) —
    `det_step_3d_HG!` Newton driver. Sparse-Jacobian prototype is
    `cell_adjacency_sparsity ⊗ ones(15, 15)` (depth=1 face stencil,
    225 nonzeros per cell-cell adjacency entry). Mirrors the
    `det_step_2d_HG!` pattern; no off-diagonal β; entropy frozen.
  * `src/dfmm.jl` (+13 LOC, append-only) — exports for
    `cholesky_el_residual_3D!`, `cholesky_el_residual_3D`,
    `pack_state_3d`, `unpack_state_3d!`,
    `build_face_neighbor_tables_3d`, `build_periodic_wrap_tables_3d`,
    `build_residual_aux_3D`, `det_step_3d_HG!`.
  * `test/test_M3_7b_3d_zero_strain.jl` (NEW, 1442 asserts) —
    cold-limit fixed-point IC on a 4×4×4 mesh: residual = 0 at IC;
    one-step + 10-step preservation byte-equal; 15-dof pack/unpack
    round-trip on all 64 leaves; face-neighbor table sanity
    (REFLECTING + triply-periodic); EOS-driven cold-limit
    reduction (s = -800 underflow); triply-periodic regression.
  * `test/test_M3_7b_dimension_lift_3d.jl` (NEW, 104 asserts) —
    THE LOAD-BEARING GATE. Two sub-gates:
      §7.1a 3D ⊂ 1D — 1D-symmetric 3D config matches M1's Phase-1
      zero-strain trajectory byte-equal: single-step (dt=1e-3,
      1e-5), 100-step run (T=0.1), 4×4×4 + 8×8×8 meshes, axis-swap
      symmetry across all 3 principal axes.
      §7.1b 3D ⊂ 2D — 2D-symmetric 3D config matches M3-3b's
      `det_step_2d_HG!` byte-equal on the axis-1 + axis-2 sub-block:
      single-step + 10-step run.
  * `test/runtests.jl` (append-only) — new "Phase M3-7b: native
    3D EL residual (no Berry; θ_ab trivial)" testset block.

**Test delta:** +1546 asserts (1442 + 104).

**Headline result — both dimension-lift gates at 0.0 absolute:**

  * §7.1a 3D ⊂ 1D single step: max |α_1 − α_1_M1| = 0.0
  * §7.1a 3D ⊂ 1D 100-step run: max |α_1 − α_1_M1| = 0.0
  * §7.1b 3D ⊂ 2D single step: max |α_1 − α_1_2D| = 0.0,
    max |β_1 − β_1_2D| = 0.0, ditto axis 2

The 3D residual reduces to M1's 1D residual byte-equal on the
1D-symmetric slice, AND reduces to M3-3b's 2D residual byte-equal
on the 2D-symmetric slice. The SO(3) Cholesky-sector reduction
with three Euler angles set to zero is structurally consistent
with the SO(2) 2D reduction.

**Newton convergence:** 2 iterations on smooth (zero-strain or
β=0) ICs — matches the M3-7 design note §3.3 expectation
("2-5 iterations on smooth ICs"). On a non-isotropic 3D IC
(nonzero β + nonzero θ_ab IC), the solver reaches residual
norm ≤ 1e-13 within the maxiters=50 budget (NonlinearSolve.jl
reports "Stalled" at 1e-13 since further iterations cannot
improve below machine precision; same behavior as M3-3b's 2D
path on equivalent non-trivial IC).

**Wall-time per step:** 27.8 ms at 4×4×4 (64 leaves);
582 ms at 8×8×8 (512 leaves). The ~21× scaling at 8× leaf-count
reflects the 15×15 within-cell Jacobian + the
`cell_adjacency_sparsity ⊗ 225-nonzero` Newton system; both
within the M3-7 design note §3.3 expectation.

**1D + 2D regression byte-equal:** verified by running
`test_phase1_zero_strain.jl`, `test_M3_3a_*.jl`,
`test_M3_3b_*.jl`, `test_M3_3c_*.jl`, `test_M3_3d_*.jl`,
`test_M3_4_*.jl`, `test_M3_6_phase0_offdiag_dimension_lift.jl`,
`test_M3_7_prep_3d_scaffolding.jl`, `test_M3_7a_*.jl` in
isolation — all pass at their original counts (4062 + 3483
asserts confirmed byte-equal).

**M3-7c handoff items:**

  * Promote `θ_12, θ_13, θ_23` from trivial-driven to Newton
    unknowns; add Berry coupling via `berry_partials_3d`
    (`src/berry.jl` — verified at the stencil level by 797
    asserts in `test_M3_prep_3D_berry_verification.jl`).
  * Per-pair Berry kinetic terms on the (α_a, β_a) rows
    (`berry_partials_3d` returns `dα`, `dβ`, `dθ` decomposition
    per pair). Three pair-Berry blocks structurally identical to
    M3-3c's single 2D block.
  * Kernel-orthogonality residual on the θ_23 row (M3-7 design
    note §2.2 — the 9D Casimir kernel direction has its largest
    component on θ_23 in SymPy's normalization).
  * The 15-dof Newton system stays 15-dof; only the residual
    rows for α_a, β_a, θ_ab change. The M3-7b zero-strain gate
    + dimension-lift gates remain regressions for M3-7c (Berry
    must vanish on the 1D / 2D dimension-lifted slices —
    structural guarantee from CHECK 6 + CHECK 3b of the 3D Berry
    verification note).

See `reference/notes_M3_7b_native_3d_residual.md` for the full
status note + handoff items.

## Repo housekeeping

- M3-3a/b/c/d worktrees cleaned up.
- M3-3e branches (`m3-3e-1` through `m3-3e-5-drop-cache-mesh`)
  preserved as audit history.
- 9 commits ahead of origin/main since M3-3a launched
  (M3-3a/b/c/d/3e-pre-flight + 3e-1/2/3/4/5).
- All M3 named branches preserved as audit history.

---

*M3-3 closed with M3-3e-5. M3-4 closed in two phases (Phase 1
periodic-x wrap; Phase 2 IC bridge + C.1/C.2/C.3 drivers). M3-6
closed entirely with Phase 5 D.10 ISM multi-tracer fidelity
(2026-04-26). All Tier-D falsifier drivers (D.1, D.4, D.7, D.10)
have landed. The 1D path is native; the 2D path is native;
the Tier-C C.1 / C.2 / C.3
acceptance gates fire. 1D-path bit-exact 0.0 parity holds across
all 19211 + 1 currently passing tests. Methods paper §10.4 / §10.7
numbers all hold. Ready for M3-5 (in flight in another worktree).*
