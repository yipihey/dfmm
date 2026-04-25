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
