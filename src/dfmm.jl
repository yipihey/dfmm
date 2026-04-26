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
export allocate_detfield_HG, periodic_simplicial_mesh_1D
export read_detfield, write_detfield!
export det_step_HG!, det_run_HG!
export total_mass_HG, total_momentum_HG, total_energy_HG
export segment_density_HG, segment_length_HG
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

end # module
