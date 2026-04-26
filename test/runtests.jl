using dfmm
using Test

@testset verbose = true "dfmm" begin
    @testset verbose = true "Phase 1: zero-strain" begin
        include("test_phase1_zero_strain.jl")
    end
    @testset verbose = true "Phase 1: uniform-strain" begin
        include("test_phase1_uniform_strain.jl")
    end
    @testset verbose = true "Phase 1: symplectic" begin
        include("test_phase1_symplectic.jl")
    end
    @testset verbose = true "Phase 2" begin
        @testset "Phase 2: mass" begin
            include("test_phase2_mass.jl")
        end
        @testset "Phase 2: momentum" begin
            include("test_phase2_momentum.jl")
        end
        @testset "Phase 2: free streaming" begin
            include("test_phase2_free_streaming.jl")
        end
        @testset "Phase 2: acoustic" begin
            include("test_phase2_acoustic.jl")
        end
    end
    @testset verbose = true "Phase 3" begin
        # Tier B.2 cold-limit reduction — the central unification test
        # of the methods paper. Order: Zel'dovich pre-crossing match,
        # then Hessian-degeneracy diagnostic (also writes the
        # `reference/figs/phase3_hessian_degen.png` plot), then energy /
        # mass / momentum conservation.
        @testset "Phase 3: Zel'dovich pre-crossing" begin
            include("test_phase3_zeldovich.jl")
        end
        @testset "Phase 3: Hessian degeneracy" begin
            include("test_phase3_hessian_degen.jl")
        end
        @testset "Phase 3: energy / mass / momentum drift" begin
            include("test_phase3_energy_drift.jl")
        end
    end
    @testset verbose = true "Phase 4: energy drift" begin
        include("test_phase4_energy_drift.jl")
    end
    @testset verbose = true "Phase 5: Sod" begin
        # Tier A.1 Sod regression vs py-1d golden. Adds the deviatoric
        # P_⊥ sector with BGK relaxation and asserts L∞ rel error < 0.05
        # on (rho, u, Pxx, Pp) at t_end = 0.2, τ = 1e-3 (warm regime).
        include("test_phase5_sod_regression.jl")
    end
    @testset verbose = true "Phase 5b: tensor-q" begin
        # Opt-in artificial viscosity (Kuropatenko / vNR) added to the
        # variational integrator. Default `q_kind = :none` reproduces
        # Phase 5 bit-equally; with `q_kind = :vNR_linear_quadratic`
        # the Sod L∞ rel error drops below 0.10 on (ρ, Pxx, Pp).
        include("test_phase5b_artificial_viscosity.jl")
    end
    @testset verbose = true "Phase 6: cold-sinusoid τ-scan" begin
        # Tier A.2 cold-sinusoid τ-scan. Generalizes the Phase-3
        # single-τ Zel'dovich match across six τ decades (10⁻³…10⁷),
        # asserts pre-crossing density vs Zel'dovich to L∞ rel < 1e-3
        # per τ, γ stays at machine-noise scale, conservation invariants
        # hold, and renders the headline 6-panel τ-scan plot
        # (`reference/figs/A2_cold_sinusoid_tauscan.png`). The post-
        # crossing golden match is `@test_skip`'d pending Phase-5b
        # shock-capturing — see `reference/notes_phase6_cold_sinusoid.md`.
        include("test_phase6_cold_sinusoid_scan.jl")
    end
    @testset verbose = true "Phase 7: steady shock" begin
        # Tier A.3 steady-shock Mach scan + heat-flux Lagrange
        # multiplier. Asserts:
        #  (a) The closed-form heat-flux BGK primitives
        #      (`heat_flux_bgk_step`, `heat_flux_step`) match the
        #      bit-equal exponential decay py-1d uses.
        #  (b) `setup_steady_shock` builds an IC whose downstream
        #      state matches analytical Rankine-Hugoniot to 3
        #      decimal places (methods paper §10.2 A.3) for
        #      M1 ∈ {1.5, 2, 3, 5, 10}.
        #  (c) `Mesh1D` and `DetField` accept the new `bc` and `Q`
        #      fields without breaking Phase 1–6 paths.
        #  (d) `det_step!` advances `Q` via BGK (exponential decay
        #      to <1e-10 over 10 steps on a smooth mesh).
        #  (e) Short-horizon (t = 0.005, ~30 steps at N = 80)
        #      preservation of the downstream R-H plateau by the
        #      variational integrator + Phase-5b artificial
        #      viscosity for M1 = 3. The long-horizon golden match
        #      is gated on a future shock-capturing extension —
        #      see `reference/notes_phase7_steady_shock.md`.
        include("test_phase7_steady_shock.jl")
    end
    @testset "eos" begin
        include("test_eos.jl")
    end
    @testset "diagnostics" begin
        include("test_diagnostics.jl")
    end
    @testset "io" begin
        include("test_io.jl")
    end
    @testset "calibration" begin
        include("test_calibration.jl")
    end
    @testset "plotting" begin
        include("test_plotting.jl")
    end
    @testset "Track D — Stochastic dressing primitives" begin
        include("test_vg_sampler.jl")
        include("test_burst_stats.jl")
        include("test_self_consistency.jl")
    end
    @testset verbose=true "Phase 8: stochastic injection" begin
        # Phase 8: variance-gamma noise post-Newton injection +
        # Phase 9 / Tier B.4 burst-stats self-consistency monitor.
        # See `reference/notes_phase8_stochastic_injection.md` and
        # `experiments/B4_compression_bursts.jl`.
        include("test_phase8_stochastic_injection.jl")
    end
    @testset verbose=true "Phase M2-3: realizability projection" begin
        # M2-3: closes M1 Open #4 — the long-time stochastic
        # realizability instability. Adds `realizability_project!` to
        # the post-Newton injection so wave-pool runs reach 10⁴+ steps
        # under production calibration. See
        # `reference/notes_M2_3_realizability.md` and
        # `experiments/M2_3_long_time_stochastic.jl`.
        include("test_phase_M2_3_realizability.jl")
    end
    @testset "setups" begin
        include("test_setups.jl")
    end
    @testset "regression scaffold" begin
        include("test_regression_scaffold.jl")
    end
    @testset verbose = true "Phase 11: passive tracer" begin
        # Tier B.5 — passive scalar advection: bit-exact tracer
        # preservation in pure-Lagrangian regions, multi-tracer
        # support, and decade-level fidelity vs a reference Eulerian
        # upwind scheme. See `reference/notes_phase11_passive_tracer.md`.
        include("test_phase11_tracer_advection.jl")
    end
    @testset verbose=true "Phase M2-2: multi-tracer wave-pool" begin
        # Tier B.6 — multi-tracer fidelity in 1D wave-pool turbulence
        # with Phase-8 stochastic injection enabled. Verifies that
        # Phase-11's bit-exact tracer-preservation property holds
        # even when stochastic noise is active (the noise mutates
        # ρu, P_xx, P_⊥, s — never the tracer matrix). See
        # `reference/notes_M2_2_multitracer.md`.
        include("test_phase_M2_2_multitracer.jl")
    end
    @testset verbose = true "Phase M2-1: action-based AMR" begin
        # Tier B.3 — 1D action-based AMR vs gradient-based AMR on
        # the off-center blast. Verifies refine/coarsen primitives
        # (mass / momentum / tracer conservation), the action-error
        # and gradient indicators, and the headline 20-50% cell-
        # count reduction at matched L² accuracy.
        # See `reference/notes_M2_1_amr.md` and
        # `experiments/B3_action_amr.jl`.
        include("test_phase_M2_1_amr.jl")
    end
    @testset verbose = true "Cross-phase smoke (defensive integration)" begin
        # Catches inter-phase regressions that per-phase tests miss
        # — e.g. struct extension without updating mutation call sites.
        # See test/test_integration_all_phases.jl docstring.
        include("test_integration_all_phases.jl")
    end
    @testset verbose = true "Phase M3-0: HG + R3D integration" begin
        # Foundation phase for the M3 (multi-D) refactor. Confirms HG
        # and R3D import cleanly and the HG-substrate-based Phase-1
        # Cholesky-sector path is bit-exactly equivalent to M1's
        # `Mesh1D`+`Segment` path. See
        # `reference/notes_M3_0_hg_integration.md` and
        # `reference/MILESTONE_3_PLAN.md` Phase M3-0.
        include("test_M3_0_smoke.jl")
        include("test_M3_0_parity_1D.jl")
    end
    @testset verbose = true "Phase M3-1: Phase 2 + 5 + 5b on HG" begin
        # Phase-2 (bulk + entropy + multi-segment periodic mesh),
        # Phase-5 (deviatoric P_⊥ via post-Newton BGK), and Phase-5b
        # (opt-in Kuropatenko / vNR tensor-q) ports to the HG
        # substrate. The HG-side driver delegates to M1's `det_step!`
        # for bit-exact parity. See
        # `reference/notes_M3_1_phase2_5_5b_port.md` and
        # `reference/MILESTONE_3_PLAN.md` Phase M3-1.
        @testset "Phase 2 (HG): mass" begin
            include("test_M3_1_phase2_mass_HG.jl")
        end
        @testset "Phase 2 (HG): momentum" begin
            include("test_M3_1_phase2_momentum_HG.jl")
        end
        @testset "Phase 2 (HG): free streaming" begin
            include("test_M3_1_phase2_free_streaming_HG.jl")
        end
        @testset "Phase 2 (HG): acoustic" begin
            include("test_M3_1_phase2_acoustic_HG.jl")
        end
        @testset "Phase 5 (HG): Sod regression" begin
            include("test_M3_1_phase5_sod_HG.jl")
        end
        @testset "Phase 5b (HG): q_kind=:none bit-equality" begin
            include("test_M3_1_phase5b_qnone_bit_equal.jl")
        end
        @testset "Phase 5b (HG): q_kind=:vNR Sod" begin
            include("test_M3_1_phase5b_qvnr_sod.jl")
        end
        @testset "Phase 5b (HG): compute_q formula" begin
            include("test_M3_1_phase5b_qformula_unit.jl")
        end
    end
    @testset verbose = true "Phase M3-prep: Berry-connection stencils" begin
        # Tier-1 building blocks for M3-3 (2D Cholesky + Berry connection).
        # Pure-functional pre-compute of the verified symbolic forms
        #   Θ_rot^(2D) = (1/3)(α_1³ β_2 − α_2³ β_1) dθ_R
        #   Θ_rot^(3D) = (1/3) Σ_{a<b}(α_a³ β_b − α_b³ β_a) dθ_{ab}
        #   θ_offdiag  = -(1/2)(α_1² α_2 dβ_{21} + α_1 α_2² dβ_{12}).
        # Cross-checked against `scripts/verify_berry_connection*.py`.
        # See `reference/notes_M3_prep_berry_stencil.md` for the API
        # and `src/berry.jl` for the implementation.
        include("test_M3_prep_berry_stencil.jl")
    end
    @testset verbose = true "Phase M3-2: Phase 7/8/11 + M2 on HG" begin
        # Phase 7 (heat-flux Q + steady shock + inflow/outflow), Phase 8
        # (variance-gamma stochastic injection), Phase 11 (passive
        # tracers), and M2-1 (action-AMR) + M2-3 (realizability
        # projection) sub-phases ported onto the HG substrate. All
        # wrappers delegate to their M1 counterparts through the
        # `cache_mesh` shim for bit-exact parity. See
        # `reference/notes_M3_2_phase7811_m2_port.md` and
        # `reference/MILESTONE_3_PLAN.md` Phase M3-2.
        @testset "Phase 7 (HG): steady-shock + heat flux Q" begin
            include("test_M3_2_phase7_steady_shock_HG.jl")
        end
        @testset "Phase 8 (HG): stochastic injection" begin
            include("test_M3_2_phase8_stochastic_HG.jl")
        end
        @testset "Phase 11 (HG): tracer advection" begin
            include("test_M3_2_phase11_tracer_HG.jl")
        end
        @testset "M2-2 (HG): multi-tracer wave-pool" begin
            include("test_M3_2_M2_2_multitracer_HG.jl")
        end
        @testset "M2-1 (HG): action-based AMR" begin
            include("test_M3_2_M2_1_amr_HG.jl")
        end
        @testset "M2-3 (HG): realizability projection" begin
            include("test_M3_2_M2_3_realizability_HG.jl")
        end
    end
end
