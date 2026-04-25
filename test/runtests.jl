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
    @testset "setups" begin
        include("test_setups.jl")
    end
    @testset "regression scaffold" begin
        include("test_regression_scaffold.jl")
    end
end
