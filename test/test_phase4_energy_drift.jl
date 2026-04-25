# test_phase4_energy_drift.jl
#
# Tier B.1 long-time energy-drift regression test.
#
# Acceptance (methods paper §10.3 B.1): the variational integrator
# preserves total energy to |ΔE|/|E_0| < 10⁻⁸ over 10⁵ timesteps on a
# smooth periodic acoustic-wave problem.
#
# Strategy. The full 10⁵-step run lives in `experiments/B1_energy_drift.jl`
# (~50 s wall time at the production resolution N=8, dt=1e-6). The
# regression test here runs a *short* version (10⁴ steps) with the
# same setup; same per-step physics, ~6 s wall time. Asserts the
# methods-paper bound and the bounded-oscillation conservation laws.
#
# At N=8, dt=1e-6, 10⁴ steps (t_end=0.01) the observed drift is
# ~7e-11 — three orders of magnitude below the 1e-8 acceptance bound.
# At the full 10⁵ steps (t_end=0.1) the drift is ~6e-9, with a small
# monotone secular component (5e-9 over the run) on top of bounded
# oscillation (4e-10 amplitude); see `reference/notes_phase4_energy_drift.md`
# for the full diagnosis.

using Test
using dfmm

include(joinpath(@__DIR__, "..", "experiments", "B1_energy_drift.jl"))

@testset "Phase 4: long-time energy drift (short version)" begin
    # Short version: 10⁴ steps at N=8, dt=1e-6.
    N = 8
    n_steps = 10_000
    dt = 1e-6
    ε = 1e-4
    result = run_b1_drift(; N = N, n_steps = n_steps, dt = dt, ε = ε,
                          log_stride = 100,
                          out_h5 = nothing, out_png = nothing,
                          quick = false)

    # Mass: round-off (Δm are labels, not state).
    @test result.Δm_max < 1e-12

    # Momentum: ε sets the velocity scale, but the IC has zero net
    # momentum. The midpoint-rule discretely conserves Σ m̄ u to
    # Newton-tolerance round-off.
    @test result.Δp_max < 1e-12

    # Energy at this short horizon: ~7e-11 observed; assert clear of
    # 1e-8 acceptance with margin.
    @test result.rel_E_max < 1e-8

    # Sanity diagnostics: αmax should remain bounded, γ² should not
    # cross the realizability boundary at this horizon.
    @test result.αmax < 1.05
    @test result.γ²min > 0.0

    @info "Phase 4 B.1 (short version)" rel_E_max=result.rel_E_max αmax=result.αmax βmax=result.βmax wall_seconds=result.wall
end
