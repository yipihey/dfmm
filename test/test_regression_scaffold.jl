using dfmm
using Test

@testset "load_tier_a_golden: schema for all six goldens" begin
    for name in dfmm.GOLDEN_NAMES
        # Skip silently if the golden isn't present (e.g. CI fresh checkout
        # before `make goldens`); the test_setups.jl suite already covers
        # the round-trip through py-1d at t=0.
        path = try
            dfmm.golden_path(name)
        catch
            continue
        end
        isfile(path) || continue

        g = load_tier_a_golden(name)

        @testset "$(name)" begin
            @test g.name === name
            @test isa(g.params, AbstractDict)
            @test haskey(g.params, "target")
            @test g.params["target"] == String(name)

            @test isa(g.x, Vector{Float64})
            @test isa(g.t, Vector{Float64})
            @test all(isfinite, g.x)
            @test all(isfinite, g.t)
            @test issorted(g.t)
            @test g.t[1] == 0.0
            N = length(g.x)
            n_snap = length(g.t)
            @test n_snap >= 2

            # /U layout
            if name in dfmm.SINGLE_FLUID_NAMES
                @test size(g.U) == (N, 8, n_snap)
                @test isa(g.fields, NamedTuple)
                @test haskey(g.fields, :rho)
                @test size(g.fields.rho) == (N, n_snap)
                @test all(g.fields.rho[:, 1] .> 0)
            else
                @test size(g.U) == (N, 16, n_snap)
                @test haskey(g.fields, :A)
                @test haskey(g.fields, :B)
                @test size(g.fields.A.rho) == (N, n_snap)
                @test size(g.fields.B.rho) == (N, n_snap)
                @test all(g.fields.A.rho[:, 1] .> 0)
                @test all(g.fields.B.rho[:, 1] .> 0)
            end

            # Conserved-state finiteness at t=0
            @test all(isfinite, view(g.U, :, :, 1))

            # nsteps positive integer
            @test isa(g.nsteps, Integer)
            @test g.nsteps >= 1
        end
    end
end

# Tier-A integrator-vs-golden comparisons land in Phase 5+ as
# test/test_A1_sod_regression.jl and friends. Those tests will reuse
# load_tier_a_golden via the same loader plumbing exercised above.
