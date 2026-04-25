module dfmm

# 2D Julia implementation of the unified dfmm framework.
#
# Design corpus:
#   HANDOFF.md
#   specs/01_methods_paper.pdf
#   design/04_action_note_v3_FINAL.pdf
#   specs/05_julia_ecosystem_survey.md
#
# Milestone 1 scope: a 1D Julia implementation reproducing the
# regression target at py-1d/. See HANDOFF.md "Milestone 1 plan".

# --- Track C/F infrastructure layer (Milestone 1, Agent C) -----------------
# Pure infrastructure modules: EOS, diagnostics, I/O, plotting, calibration.
# Track A (integrator) and Track B (regression tests) consume these.
include("eos.jl")
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

end # module
