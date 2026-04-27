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

# --- Phase M3-3c API: 2D EL residual + Newton WITH Berry coupling ---------
# Promotes θ_R from a fixed IC value to a Newton unknown. Adds the closed-
# form Berry partials (`src/berry.jl::berry_partials_2d`) into the
# per-axis residual rows; adds a 9th `F^θ_R` row encoding the kinematic-
# equation evolution (trivial drive in M3-3c — off-diagonal velocity
# gradients enter at M3-3d/M3-6). The Newton system grows from 8N to 9N.
# See `reference/notes_M3_3_2d_cholesky_berry.md` §4 + §6 and
# `reference/notes_M3_3c_berry_integration.md`.
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

end # module
