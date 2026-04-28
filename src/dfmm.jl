module dfmm

# 1D / 2D Julia implementation of the unified dfmm framework.
#
# Design corpus:
#   HANDOFF.md
#   specs/01_methods_paper.pdf
#   design/04_action_note_v3_FINAL.pdf
#   specs/05_julia_ecosystem_survey.md
#
# Milestone 1 scope: a 1D Julia implementation reproducing the
# regression target at py-1d/. See HANDOFF.md "Milestone 1 plan" and
# reference/MILESTONE_1_PLAN.md for the phase-by-phase breakdown.
#
# Phase 1: Cholesky-sector variational integrator with externally
# supplied γ. See `src/cholesky_sector.jl` for the action and discrete
# EL system.
#
# Phase 2: full deterministic action L_det = ½ ẋ² + L_Ch on a
# multi-segment periodic Lagrangian-mass mesh, with bulk position x,
# velocity u, and frozen entropy s. See
# `reference/notes_phase2_discretization.md` and `src/discrete_action.jl`
# / `src/cholesky_sector.jl`'s `det_el_residual`.

# Field types live first so downstream files (including eos.jl) can
# refer to them via Mvv-method overloading.
include("types.jl")

# EOS is included before segment.jl so that segment.jl helpers
# (`total_energy`) and the Phase-2 EL residual can reach the
# `Mvv(J, s)` adiabat method.
include("eos.jl")

include("segment.jl")
include("discrete_transport.jl")
include("discrete_action.jl")
include("cholesky_sector.jl")
# Phase 5: deviatoric / P_⊥ BGK update (used by det_step!).
include("deviatoric.jl")
# Phase 5b: opt-in artificial viscosity (tensor-q) for shock capture.
include("artificial_viscosity.jl")
# Phase 7: heat-flux Lagrange-multiplier sector (used by det_step!).
include("heat_flux.jl")
include("newton_step.jl")

# --- Phase M3-0 API: HierarchicalGrids-based Cholesky-sector path ----------
# Foundation phase for the M3 (multi-D) refactor. Adds a thin shim that
# runs the Phase-1 Cholesky-sector integrator on top of HG's
# `SimplicialMesh{1, T}` + `PolynomialFieldSet` substrate, while reusing
# M1's `cholesky_el_residual` kernel byte-identically. M1's hand-rolled
# `Mesh1D` + `Segment` path remains the regression baseline; the new
# files coexist with it.
#
# See `reference/MILESTONE_3_PLAN.md` Phase M3-0 and
# `reference/notes_M3_0_hg_integration.md` for the full design.
import HierarchicalGrids
import R3D
include("eom.jl")
include("newton_step_HG.jl")

# --- Phase 1 API ----------------------------------------------------------
export ChField, gamma_from_Mvv
export Segment, Mesh1D, single_segment_mesh
export D_t_q
export cholesky_one_step_action, cholesky_el_residual, cholesky_hamiltonian
export cholesky_step, cholesky_run

# --- Phase M3-0 API (HG substrate) ----------------------------------------
# Note: `spatial_dimension(::ChFieldND)` is intentionally NOT exported —
# `HierarchicalGrids` already exports the same name for its meshes, and
# we'd rather keep the user's namespace unambiguous. Call via
# `dfmm.spatial_dimension(...)` if needed.
export ChFieldND
export read_alphabeta, write_alphabeta!, cholesky_el_residual_HG
export cholesky_step_HG!, cholesky_run_HG!
export single_cell_simplicial_mesh_1D, uniform_simplicial_mesh_1D,
       allocate_chfield_HG

# --- Phase M3-1 API (HG substrate; Phase-2/5/5b) --------------------------
# Adds the multi-segment Phase-2 (`(x, u, α, β, s)` evolution), the
# post-Newton Phase-5 BGK relaxation of `(P_xx, P_⊥)`, and the opt-in
# Phase-5b artificial viscosity (Kuropatenko / vNR) on the HG substrate.
# The driver delegates to M1's `det_step!` for bit-exact parity. See
# `reference/notes_M3_1_phase2_5_5b_port.md` for the design write-up.
export DetFieldND
export DetMeshHG, DetMeshHG_from_arrays
export allocate_detfield_HG
export read_detfield, write_detfield!
export det_step_HG!, det_run_HG!
export total_mass_HG, total_momentum_HG, total_energy_HG
export segment_density_HG, segment_length_HG
# M3-2b Swap 6: HG-aware Newton-Jacobian sparsity helper.
export det_jac_sparsity_HG
# M3-2b Swap 8: BC translation helper for the cache-mesh delegate path.
export bc_symbol_from_spec
# Note: `n_cells(::DetMeshHG)` is intentionally NOT exported —
# `HierarchicalGrids` already exports an `n_cells` method on its own
# mesh types. Call via `dfmm.n_cells(...)` if needed; the regular
# user-facing path is `length(mesh.Δm)` or
# `HierarchicalGrids.n_simplices(mesh.mesh)`.

# --- Phase 2 API (multi-segment full deterministic) ------------------------
export DetField
export n_segments, vertex_mass, segment_length, segment_density,
       total_mass, total_momentum, total_energy,
       total_internal_energy, total_kinetic_energy
export discrete_action_sum, segment_cholesky_action, midpoint_strain, midpoint_J
export det_el_residual
export det_step!, det_run!, pack_state, pack_state!, unpack_state!

# --- Phase 5 API (deviatoric / P_⊥ BGK relaxation) -------------------------
export deviatoric_bgk_step, deviatoric_bgk_step_exponential,
       pperp_advect_lagrangian, bgk_relax_pressures, pperp_step

# --- Phase 5b API (opt-in artificial viscosity / tensor-q) -----------------
export compute_q_segment, q_kind_supported, q_active,
       Q_KIND_NONE, Q_KIND_VNR_LINEAR_QUADRATIC

# --- Phase 7 API (heat-flux Lagrange multiplier) ---------------------------
export heat_flux_bgk_step, heat_flux_advect_lagrangian, heat_flux_step

# --- Track C/F infrastructure layer (Milestone 1, Agent C) -----------------
# Pure infrastructure modules: diagnostics, I/O, plotting, calibration.
# (eos.jl is included higher up so Phase-2 segment.jl can use it.)
include("diagnostics.jl")
include("io.jl")
include("calibration.jl")
include("plotting.jl")

# --- Track D stochastic primitives (Milestone 1, Agent D) ------------------
# Variance-gamma sampling, burst-statistics, self-consistency monitor.
# Will be wired into the integrator's stochastic injection in Phase 8.
include("stochastic.jl")
using .Stochastic

# Public re-exports for convenient REPL use; these are the entry points
# the milestone plan calls out for cross-track consumers.
export Mvv, gamma_from_state, Mvv_from_pressure, pressure_from_Mvv
export gamma_rank_indicator, realizability_marker,
       realizability_marker_from_Mvv, hessian_HCh, det_hessian_HCh
export save_state, load_state
export load_noise_model
export plot_profile, plot_phase_portrait, plot_diagnostics
export Stochastic,
       rand_variance_gamma,
       rand_variance_gamma!,
       pdf_variance_gamma,
       cdf_variance_gamma,
       burst_detect,
       estimate_gamma_shape,
       residual_kurtosis,
       gamma_shape_from_kurtosis,
       ks_test

# --- Phase 8 API (variance-gamma stochastic injection) --------------------
# Wires Track D's VG sampler + burst-stats primitives into the
# deterministic integrator as a post-Newton operator-split step. See
# `src/stochastic_injection.jl` and `reference/notes_phase8_stochastic_injection.md`.
include("stochastic_injection.jl")
export NoiseInjectionParams, from_calibration,
       inject_vg_noise!, det_run_stochastic!,
       BurstStatsAccumulator, record_step!, burst_durations,
       self_consistency_check, InjectionDiagnostics,
       smooth_periodic_3pt!,
       realizability_project!, ProjectionStats

# Tier-A initial-condition factories. Mirrors the six benchmarks listed
# in the methods paper §10.2; the corresponding HDF5 goldens are
# generated by `scripts/make_goldens.jl` and live under
# `reference/golden/`.
include("setups.jl")
export setup_sod, setup_cold_sinusoid, setup_steady_shock,
       setup_kmles_wavepool, setup_dust_in_gas, setup_eion

# Tier-C dimension-generic IC factories (HG substrate). Used by
# Milestone M3-4 (2D 1D-symmetric consistency tests). The factories
# are dimension-generic over `D ∈ {1, 2, 3}` via HG's
# `HierarchicalMesh{D}` + `EulerianFrame{D}`. See
# `reference/notes_M3_prep_tierC_ic_factories.md`.
include("setups_2d.jl")
export tier_c_sod_ic, tier_c_cold_sinusoid_ic, tier_c_plane_wave_ic
export uniform_eulerian_mesh, tier_c_total_mass,
       tier_c_cell_centers, tier_c_velocity_component

# Tier-A regression-scaffold: HDF5 golden loader. The full
# integrator-vs-golden comparison enters at Phase 5; this module
# lands the loader plumbing in advance.
include("regression.jl")
export load_tier_a_golden

# --- Phase 11 API (Tier B.5 passive scalar advection) ---------------------
# A `TracerMesh` carries one or more passive-scalar fields per
# Lagrangian segment, in parallel to a `Mesh1D`. In pure-Lagrangian
# regions (Milestone 1) the advection step is a no-op: the tracer
# matrix is never written to, so deterministic numerical diffusion
# is exactly zero. See methods paper §7 for the structural argument
# and `reference/notes_phase11_passive_tracer.md` for the benchmark.
include("tracers.jl")
export TracerMesh, advect_tracers!, add_tracer!, set_tracer!,
       tracer_at_position, tracer_index,
       n_tracer_fields, n_tracer_segments,
       eulerian_upwind_advect!, interface_width

# --- Phase M2-1 API (Tier B.3 1D action-based AMR) ------------------------
# Refine/coarsen primitives on the Lagrangian segment mesh + the action-
# error and gradient indicators used to compare action-AMR vs. gradient-
# AMR per methods paper §10.3 B.3. See `src/amr_1d.jl` and
# `experiments/B3_action_amr.jl`.
include("amr_1d.jl")
export refine_segment!, coarsen_segment_pair!,
       action_error_indicator, gradient_indicator,
       amr_step!, refresh_p_half!

# --- Phase M3-3a API: per-axis Cholesky decomposition driver ---------------
# Per-axis (principal-axis) Cholesky decomposition for the dimension-lifted
# 2D Cholesky-sector EL residual (consumed by M3-3b). Provides
# `cholesky_decompose_2d`, `cholesky_recompose_2d`, and the per-axis γ
# diagnostic `gamma_per_axis_2d`. See `src/cholesky_DD.jl` and
# `reference/notes_M3_3_2d_cholesky_berry.md` §4.1 / §4.3.
include("cholesky_DD.jl")
export cholesky_decompose_2d, cholesky_recompose_2d, gamma_per_axis_2d
# M3-3c: closed-form ∂H_rot/∂θ_R + kernel-orthogonality residual helper.
export h_rot_partial_dtheta, h_rot_kernel_orthogonality_residual

# --- Phase M3-3a API: 2D Cholesky-sector field types -----------------------
# `DetField2D{T}` is the working struct for the M3-3 2D 10-dof Newton
# unknown set `(x_a, u_a, α_a, β_a, θ_R, s)`; the 12-named-field
# PolynomialFieldSet allocator `allocate_cholesky_2d_fields` plus the
# read/write helpers `read_detfield_2d` / `write_detfield_2d!` ride on
# top. See `src/types.jl` (DetField2D) and `src/setups_2d.jl`
# (allocator + accessors).
export DetField2D, n_dof_newton
export allocate_cholesky_2d_fields, read_detfield_2d, write_detfield_2d!

# --- Phase M3-3b API: native HG-side 2D EL residual + Newton driver --------
# `cholesky_el_residual_2D!` evaluates the per-axis Cholesky-sector EL
# residual (no Berry; θ_R fixed) on the 2D Eulerian quadtree mesh,
# consuming `HaloView`-style face-neighbor lookups via a pre-computed
# `face_lo_idx / face_hi_idx` table. `det_step_2d_HG!` is the Newton
# driver that wraps it. See `src/eom.jl` and
# `reference/notes_M3_3_2d_cholesky_berry.md` §3 (Berry deferred to M3-3c).
export cholesky_el_residual_2D!, cholesky_el_residual_2D
export pack_state_2d, unpack_state_2d!,
       build_face_neighbor_tables, build_residual_aux_2D
export det_step_2d_HG!

# --- Phase M3-3c → M3-6 Phase 0 API: 2D EL residual + Newton WITH Berry ---
# coupling AND off-diagonal `β_12, β_21` re-activated.
#
# M3-3c: promotes θ_R from a fixed IC value to a Newton unknown. Adds the
# closed-form Berry partials (`src/berry.jl::berry_partials_2d`) into the
# per-axis residual rows; adds a 9th `F^θ_R` row encoding the kinematic-
# equation evolution (trivial drive in M3-3c — off-diagonal velocity
# gradients enter at M3-3d/M3-6).
#
# M3-6 Phase 0: re-activates the off-diagonal Cholesky pair `β_12, β_21`
# in the 11-dof Newton residual (was 9 in M3-3c). The two new rows
# F^β_12, F^β_21 are trivial-drive (free-flight cut); per-axis F^β_a rows
# pick up off-diagonal Berry coupling terms per the corrected
# antisymmetric extension of §7 of the 2D Berry note (verified against
# `scripts/verify_berry_connection_offdiag.py`). The Newton system grows
# from 9N to 11N. M3-6 Phase 1 (D.1 KH falsifier) will plumb the off-
# diagonal strain-coupling drive that breaks the trivial-drive pattern.
# See `reference/notes_M3_3_2d_cholesky_berry.md` §4 + §6,
# `reference/notes_M3_3c_berry_integration.md`, and
# `reference/notes_M3_6_phase0_offdiag_beta.md`.
export cholesky_el_residual_2D_berry!, cholesky_el_residual_2D_berry
export pack_state_2d_berry, unpack_state_2d_berry!
export det_step_2d_berry_HG!

# --- Phase M3-prep API: Berry-connection 2D/3D stencils -------------------
# Pure-functional building blocks for the M3-3 (2D Cholesky + Berry
# connection) phase. Implements the verified symbolic forms
#
#   Θ_rot^(2D) = (1/3)(α_1³ β_2 − α_2³ β_1) · dθ_R
#   Θ_rot^(3D) = (1/3) Σ_{a<b} (α_a³ β_b − α_b³ β_a) · dθ_{ab}
#   θ_offdiag  = -(1/2)(α_1² α_2 dβ_{21} + α_1 α_2² dβ_{12})
#
# from `reference/notes_M3_phase0_berry_connection.md` (2D + off-diag) and
# `reference/notes_M3_phase0_berry_connection_3D.md` (SO(3)). Cross-checked
# against `scripts/verify_berry_connection*.py`. Not yet wired into the
# solver — M3-3 will consume these stencils.
include("berry.jl")
export BerryStencil2D, BerryStencil3D
export berry_F_2d, berry_term_2d, berry_partials_2d
export berry_F_3d, berry_term_3d, berry_partials_3d
export kinetic_offdiag_coeffs_2d, kinetic_offdiag_2d

# --- Phase M3-2 API (HG substrate; Phase 7/8/11 + M2-1 / M2-3) ----------
# Ports the M1 Phase 7 (heat-flux Q via det_step!), Phase 8 (variance-
# gamma stochastic injection), Phase 11 (passive tracers), the M2-1
# action-based AMR primitives (refine/coarsen + indicators + driver),
# and the M2-3 realizability projection onto the HG substrate.
# **M3-3e-5:** all wrappers run natively on the HG `PolynomialFieldSet`
# substrate; the legacy `cache_mesh::Mesh1D` shim has been retired.
# See `reference/notes_M3_2_phase7811_m2_port.md`,
# `reference/notes_M3_3e_5_cache_mesh_dropped.md`, and
# `reference/MILESTONE_3_PLAN.md` Phase M3-2 / M3-3.
include("newton_step_HG_M3_2.jl")
include("action_amr_helpers.jl")
export inject_vg_noise_HG!, det_run_stochastic_HG!
export TracerMeshHG, advect_tracers_HG!
export realizability_project_HG!
export refine_segment_HG!, coarsen_segment_pair_HG!,
       action_error_indicator_HG, gradient_indicator_HG,
       amr_step_HG!

# --- Phase M3-3d API: per-axis γ diagnostic + AMR/realizability per-axis -----
# Per-axis γ diagnostic on the 2D Cholesky-sector field set, per-axis AMR
# indicator on `HierarchicalMesh{2}`, per-axis realizability projection,
# and the HG `step_with_amr!`-driven AMR loop wired through
# `register_refinement_listener!`. Closes M3-2b's deferred Swaps 2+3 for
# the 2D scope. See `reference/notes_M3_3d_per_axis_gamma_amr.md`.
export gamma_per_axis_2d_field, gamma_per_axis_2d_diag
export realizability_project_2d!
export action_error_indicator_2d_per_axis,
       register_field_set_on_refine!,
       step_with_amr_2d!

# --- Phase M3-4 API: periodic-x coordinate wrap on the 2D EL residual ------
# Per-axis-per-cell coordinate-wrap offsets for the 2D Cholesky-sector
# residual. Closes the M3-3c handoff item flagged as "the periodic-x
# coordinate wrap for active strain is a noted M3-3c handoff item": the
# 2D residual now correctly handles periodic boundaries on active /
# advecting flows, mirroring the 1D path's `+L_box` wrap in
# `cholesky_sector.jl::det_el_residual` at `j == N`.
#
# `build_residual_aux_2D` now populates `wrap_lo_idx, wrap_hi_idx` in the
# returned NamedTuple. Both 2D residuals (`cholesky_el_residual_2D!` and
# `cholesky_el_residual_2D_berry!`) consume them when present and fall
# back to zero offsets when absent (legacy callers continue to work
# byte-equally on REFLECTING configurations). See
# `reference/notes_M3_4_tier_c_consistency.md`.
export build_periodic_wrap_tables

# --- Phase M3-5 API: Bayesian L↔E remap ----------------------------------
# Wires HG's `compute_overlap` + `polynomial_remap_l_to_e!` /
# `polynomial_remap_e_to_l!` into a dfmm-side per-step driver. The remap
# conservatively transfers per-cell polynomial fields between a deforming
# Lagrangian `SimplicialMesh{D, T}` and a fixed Eulerian
# `HierarchicalMesh{D}` background, per the methods paper §6 (Bayesian
# remap with the law of total cumulants). HG's `RemapDiagnostics` is
# exposed via `liouville_monotone_increase_diagnostic` per §6.6.
#
# IntExact backend (HG commit `cc6ed70`+) is opt-in via `backend = :exact`
# on `bayesian_remap_*!`; `:float` remains the production default. See
# `~/.julia/dev/HierarchicalGrids/docs/src/exact_backend.md` for the
# documented caveats (D=2 0//0 collinear-triangle degeneracy; D=2/3 16-bit
# lattice volume drift up to ~30 % on near-degenerate configurations).
include("remap.jl")
export BayesianRemapState
export bayesian_remap_l_to_e!, bayesian_remap_e_to_l!
export remap_round_trip!
export liouville_monotone_increase_diagnostic
export audit_overlap_dfmm
export total_mass_weighted_lagrangian, total_mass_weighted_eulerian

"""
    det_run_with_remap_HG!(state_args...; remap_every, remap_state, lag_mesh,
                            lag_fields, kwargs...)

Thin orchestration wrapper: call the existing per-step Newton driver
(`det_run_HG!` for 1D, the user's 2D run loop) and inject a Bayesian
L→E→L round trip every `remap_every` steps. The wrapper does NOT
modify the per-step driver source; M3-4 owns those files.

This is a SCAFFOLD entry point. The 2D solver (`det_step_2d_berry_HG!`)
does not yet have a public run-driver in dfmm; M3-5's tests build the
remap loop directly. This stub is provided as documentation of the
intended integration shape and as the M3-6 hand-off site.

# Arguments

- `remap_every::Union{Int, Nothing}` — when set to `K`, run the round
  trip every K steps. `nothing` disables the remap; the wrapper
  becomes a passthrough.
- `remap_state` — a `BayesianRemapState` from M3-5.
- `lag_mesh`, `lag_fields` — the Lagrangian state.

The bit-exact 1D path is preserved when `remap_every === nothing`;
this is the regression contract for the M1+M2+M3 test suite.
"""
function det_run_with_remap_HG!(remap_state::BayesianRemapState,
                                  lag_mesh, lag_fields;
                                  remap_every::Union{Int, Nothing} = nothing,
                                  step::Int = 0,
                                  backend::Symbol = :float,
                                  fields::Union{Nothing, Tuple} = nothing)
    if remap_every === nothing
        return remap_state
    end
    if step > 0 && step % remap_every == 0
        remap_round_trip!(remap_state, lag_mesh, lag_fields;
                          backend = backend, fields = fields)
    end
    return remap_state
end
export det_run_with_remap_HG!

# --- Phase M3-4 Phase 2 API: Tier-C IC bridge + primitive recovery -------
# Bridges primitive `(ρ, u_x, u_y, P)` cell-averages onto the M3-3 12-field
# Cholesky-sector state `(x_a, u_a, α_a, β_a, θ_R, s, Pp, Q)` consumed by
# `det_step_2d_berry_HG!`, plus the inverse `primitive_recovery_2d` used
# for the C.1 / C.2 / C.3 acceptance gates. Cold-limit, isotropic IC
# convention: α = 1, β = 0, θ_R = 0, Pp = Q = 0; s solved from the EOS
# `Mvv(1/ρ, s) = P/ρ` via `s_from_pressure_density`. See
# `reference/notes_M3_4_tier_c_consistency.md` §"Pre-Tier-C handoff items".
export s_from_pressure_density,
       cholesky_sector_state_from_primitive,
       primitive_recovery_2d, primitive_recovery_2d_per_cell
export tier_c_sod_full_ic, tier_c_cold_sinusoid_full_ic,
       tier_c_plane_wave_full_ic

# --- Phase M3-6 Phase 1b API: D.1 KH IC factory --------------------------
# `tier_d_kh_ic` (primitive view) and `tier_d_kh_ic_full` (full Cholesky-
# sector field set) factories for the D.1 Kelvin-Helmholtz falsifier
# (M3-6 Phase 1c calibration). Sheared base flow `u_1(y) = U_jet ·
# tanh((y - y_0)/w)` overlaid with an antisymmetric tilt-mode
# perturbation `δβ_12 = -δβ_21` in the off-diagonal Cholesky factors.
# See `reference/notes_M3_6_phase1b_kh_ic_realizability.md`.
export tier_d_kh_ic, tier_d_kh_ic_full

# --- Phase M3-6 Phase 2 API: D.4 Zel'dovich pancake IC factory -----------
# `tier_d_zeldovich_pancake_ic` (full Cholesky-sector field set) factory
# for the D.4 Zel'dovich pancake collapse falsifier (M3-6 Phase 2). The
# central novel test of methods paper §10.5 D.4: per-axis γ correctly
# identifies the pancake-collapse direction. Cosmological reference test.
# See `reference/notes_M3_6_phase2_D4_zeldovich.md`.
export tier_d_zeldovich_pancake_ic

# --- Phase M3-7 prep API: 3D field set + 3D Cholesky decomposition --------
# Scaffolding for the M3-7 (3D extension) milestone. Adds:
#   • `DetField3D{T}` — the 3D analog of `DetField2D{T}` carrying the
#     13-dof Newton unknown set `(x_a, u_a, α_a, β_a)_{a=1,2,3} +
#     (θ_12, θ_13, θ_23) + s` per leaf cell. (Off-diagonal β and post-
#     Newton Pp/Q sectors are deferred per M3-3a Q3 default + M3-7
#     design note §4.4.)
#   • `cholesky_decompose_3d`, `cholesky_recompose_3d` — 3×3 SPD ↔
#     (α, θ) round-trip via eigendecomposition (intrinsic Cardan ZYX
#     Euler-angle convention; reduces byte-equally to the 2D
#     `cholesky_recompose_2d` on the dimension-lift slice).
#   • `gamma_per_axis_3d` — per-axis γ diagnostic in the 3D principal-
#     axis frame.
#   • `rotation_matrix_3d` — helper to build the SO(3) rotation matrix
#     from three Euler angles in the chosen convention.
#
# This phase does NOT write the 3D EL residual or any 3D scientific
# drivers — those are M3-7a (halo smoke + field set), M3-7b (residual),
# M3-7c (Berry coupling), M3-7d (per-axis γ AMR), M3-7e (Tier C/D).
# The new file `src/cholesky_DD_3d.jl` is intentionally separate from
# `src/cholesky_DD.jl` to avoid conflict with M3-6 Phase 3 work in
# parallel. See `reference/notes_M3_7_prep_3d_scaffolding.md`.
include("cholesky_DD_3d.jl")
export DetField3D
export cholesky_decompose_3d, cholesky_recompose_3d, gamma_per_axis_3d
export rotation_matrix_3d

# --- Phase M3-7a API: 3D Cholesky-sector field-set allocator + helpers ----
# `allocate_cholesky_3d_fields(mesh::HierarchicalMesh{3})` builds the 16-
# named-field `PolynomialFieldSet` over `n_cells(mesh)` mirroring the M3-3a
# 2D allocator pattern; `read_detfield_3d` / `write_detfield_3d!` round-trip
# a `DetField3D` against it bit-exactly. Lives in `src/setups_2d.jl` (the
# dimension-generic Tier-C IC factory file). M3-7b will consume these
# helpers when assembling the native HG-side 3D EL residual.
# See `reference/notes_M3_7a_3d_halo_allocator.md`.
export allocate_cholesky_3d_fields, read_detfield_3d, write_detfield_3d!

# --- Phase M3-7b API: native HG-side 3D EL residual + Newton driver --------
# 3D analog of M3-3b's `cholesky_el_residual_2D!` + `det_step_2d_HG!`.
# 15-dof per leaf cell `(x_a, u_a, α_a, β_a)_{a=1,2,3} + (θ_12, θ_13, θ_23)`;
# face-neighbor stencil expanded to 6 faces; periodic-coordinate wrap
# generalised to 3 axes; θ-rows trivial-driven (Berry coupling lands in
# M3-7c). Newton sparsity is `cell_adjacency_sparsity ⊗ ones(15, 15)`.
# See `reference/notes_M3_7_3d_extension.md` §3 + `reference/notes_M3_7b_native_3d_residual.md`.
export cholesky_el_residual_3D!, cholesky_el_residual_3D
export pack_state_3d, unpack_state_3d!,
       build_face_neighbor_tables_3d, build_periodic_wrap_tables_3d,
       build_residual_aux_3D
export det_step_3d_HG!

# --- Phase M3-7c API: SO(3) Berry coupling integration on the 3D path ------
# 3D analog of M3-3c's `cholesky_el_residual_2D_berry!` +
# `det_step_2d_berry_HG!`. Same 15-dof packing as M3-7b
# (`(x_a, u_a, α_a, β_a)_{a=1,2,3} + (θ_12, θ_13, θ_23)`); the θ_{ab}
# are now genuine Newton unknowns coupled to the per-axis (α_a, β_a)
# blocks via the verified SO(3) Berry kinetic 1-form
# `(1/3) Σ_{a<b} (α_a^3 β_b − α_b^3 β_a) dθ_{ab}` from `src/berry.jl`.
# The closed-form per-pair `∂H_rot/∂θ_{ab}` lives in
# `src/cholesky_DD_3d.jl::h_rot_partial_dtheta_3d` (verification
# artefact for §7.4; the discrete EL residual encodes the per-axis
# Berry-modified Hamilton equations directly so the H_rot solvability
# is structurally guaranteed at every Newton iterate).
# See `reference/notes_M3_7_3d_extension.md` §4 + §7 and
# `reference/notes_M3_7c_3d_berry_integration.md`.
export cholesky_el_residual_3D_berry!, cholesky_el_residual_3D_berry
export det_step_3d_berry_HG!
export h_rot_partial_dtheta_3d, h_rot_kernel_orthogonality_residual_3d

# --- Phase M3-7d API: per-axis γ + AMR + realizability per-axis (3D) -------
# Per-axis γ diagnostic on the 3D Cholesky-sector field set, per-axis
# action-AMR indicator on `HierarchicalMesh{3}`, per-axis realizability
# projection, and the HG `step_with_amr!`-driven AMR loop wired through
# `register_refinement_listener!`. The 3D analog of M3-3d (2D); generalises
# the per-axis decomposition pattern from `(γ_1, γ_2)` to `(γ_1, γ_2, γ_3)`
# selectivity. See `reference/notes_M3_7d_3d_per_axis_gamma_amr.md`.
export gamma_per_axis_3d_field, gamma_per_axis_3d_diag
export realizability_project_3d!, ProjectionStats3D
export action_error_indicator_3d_per_axis,
       register_field_set_on_refine_3d!,
       step_with_amr_3d!

# --- Phase M3-6 Phase 3 API: 2D substrate (tracers + stoch + γ-diag) -----
# Three deliverables, each scoped to extend a 1D substrate to 2D:
#  (a) `TracerMeshHG2D` — per-species per-cell passive scalars on a
#      `HierarchicalMesh{2}` + 14-named-field 2D Cholesky-sector field
#      set, with refine/coarsen mass conservation via
#      `register_tracers_on_refine_2d!`. Pure-Lagrangian byte-exact
#      preservation (Phase 11 + M2-2 invariants on the 2D path).
#  (b) `inject_vg_noise_HG_2d!` — per-axis VG stochastic injection on
#      the 2D field set, with explicit `axes` selectivity (axis-1
#      injection leaves axis-2 fields byte-equal). Honours the M3-6
#      Phase 1b 4-component β-cone via `realizability_project_2d!`.
#  (c) `gamma_per_axis_2d_per_species_field` — per-species wrapper
#      over `gamma_per_axis_2d_field` for D.7 dust-trap and D.10 ISM-
#      tracer per-species γ diagnostics.
# See `reference/notes_M3_6_phase3_2d_substrate.md`.
export TracerMeshHG2D, advect_tracers_HG_2d!,
       set_species!, species_index, n_species, n_cells_2d,
       register_tracers_on_refine_2d!
export inject_vg_noise_HG_2d!, InjectionDiagnostics2D
export gamma_per_axis_2d_per_species, gamma_per_axis_2d_per_species_field

# --- Phase M3-6 Phase 4 API: D.7 dust-traps IC factory -------------------
# `tier_d_dust_trap_ic_full` (Taylor-Green vortex + 2-species gas/dust
# tracer mesh) factory for the D.7 dust-trapping in vortices falsifier
# (M3-6 Phase 4). Methods paper §10.5 D.7. Builds on Phase 1b 4-comp
# realizability + Phase 3 2D substrate.
# See `reference/notes_M3_6_phase4_D7_dust_traps.md`.
export tier_d_dust_trap_ic_full

# --- Phase M3-6 Phase 5 API: D.10 ISM-tracers IC factory -----------------
# `tier_d_ism_tracers_ic_full` (KH-style sheared base flow + N≥3 species
# `TracerMeshHG2D` with phase-stratified concentration profiles) factory
# for the D.10 ISM-like 2D multi-tracer fidelity falsifier (M3-6 Phase 5).
# Methods paper §10.5 D.10. Builds on Phase 1b KH IC + Phase 3 substrate.
# See `reference/notes_M3_6_phase5_D10_ism_tracers.md`.
export tier_d_ism_tracers_ic_full

# --- Phase M3-7e API: 3D Tier-C/D drivers + ICs --------------------------
# 3D analogs of M3-4 Phase 2's 2D Tier-C IC factories + the M3-6 Phase 2
# D.4 Zel'dovich pancake. Cold-limit IC bridge for 3D
# (`cholesky_sector_state_from_primitive_3d`); 3D primitive recovery
# helpers; full-IC factories for C.1 / C.2 / C.3 (3D Sod / cold sinusoid /
# plane wave) and the cosmological D.4 3D Zel'dovich pancake. See
# `reference/notes_M3_7e_3d_tier_cd_drivers.md`.
export cholesky_sector_state_from_primitive_3d,
       primitive_recovery_3d, primitive_recovery_3d_per_cell
export tier_c_sod_3d_full_ic,
       tier_c_cold_sinusoid_3d_full_ic,
       tier_c_plane_wave_3d_full_ic,
       tier_d_zeldovich_pancake_3d_ic_full

# --- Phase M3-8 Phase a API: Tier-E stress-test IC factories -------------
# Methods paper §10.6 Tier E. Three "extreme regime" stress tests:
#   • E.1 high-Mach 2D shocks (`tier_e_high_mach_shock_ic_full`) —
#     Mach 5/10 Sod-style; acceptance is graceful failure (no NaN).
#   • E.2 severe shell-crossing (`tier_e_severe_shell_crossing_ic_full`)
#     — superposition of two-axis Zel'dovich at A=0.7; tests
#     realizability projection against compression cascade.
#   • E.3 very low Knudsen (`tier_e_low_knudsen_ic_full`) — small-τ BGK
#     relaxation regime; tests the implicit Newton's stiff-τ limit.
# See `reference/notes_M3_8a_tier_e_gpu_prep.md`.
export tier_e_high_mach_shock_ic_full,
       tier_e_severe_shell_crossing_ic_full,
       tier_e_low_knudsen_ic_full

# --- Phase M3-8 Phase b API: matrix-free Newton-Krylov drivers ------------
# Matrix-free Newton-Krylov variants of the dense / sparse-Jacobian
# Newton drivers `det_step_2d_HG!`, `det_step_2d_berry_HG!`, and
# `det_step_3d_berry_HG!`. Same residual functions, same bit-equality
# contract on zero-strain ICs (≤ 1e-12 abs); rel-error ≤ 1e-10 on
# active-strain configurations due to GMRES inner-tolerance round-off.
# Drops the SparseMatrixCSC Jacobian construction (M3-8a audit blocker
# #2) and the ForwardDiff-Dual coloring (blocker #1) — both are GPU-
# unfriendly. The matrix-free path is the algorithm-side prerequisite
# for the M3-8c full Metal port; once HG-side `Backend`-parameterized
# `PolynomialFieldSet` lands upstream, the same Newton-Krylov outer
# loop runs unchanged on `MtlArray` / `CuArray` / `ROCArray` storage.
# See `reference/notes_M3_8b_metal_gpu_port.md`.
include("newton_step_matrix_free.jl")
export det_step_2d_HG_matrix_free!,
       det_step_2d_berry_HG_matrix_free!,
       det_step_3d_berry_HG_matrix_free!

# --- Phase M4-2 API: 3D KH 21-dof residual + driver + IC factory --------
# 3D analog of M4 Phase 1 (2D closed-loop β_off ↔ β_a). Lifts the
# off-diagonal Cholesky-sector to six β_{ab} slots (one per pair (a, b))
# with full closed-loop H_back per pair. Used by the M4 Phase 2 D.1 3D
# Kelvin-Helmholtz falsifier driver.
# See `reference/notes_M4_phase2_3d_kh_falsifier.md`.
export cholesky_el_residual_3D_berry_kh!, cholesky_el_residual_3D_berry_kh
export pack_state_3d_kh, unpack_state_3d_kh!, build_residual_aux_3D_kh
export det_step_3d_berry_kh_HG!
export allocate_cholesky_3d_kh_fields, tier_d_kh_3d_ic_full

end # module
