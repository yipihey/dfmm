using Test
using dfmm

@testset "io: HDF5 round-trip preserves NamedTuple of arrays/scalars" begin
    state = (
        rho = [1.0, 0.9, 0.5],
        u = [0.0, 0.1, -0.2],
        Pxx = [1.0 0.5; 0.5 1.0],
        n_cells = 3,
        gamma_law = 1.6666666666666667,
    )
    mktempdir() do dir
        path = joinpath(dir, "snap.h5")
        dfmm.save_state(path, state)
        round = dfmm.load_state(path)

        # All keys present (order may differ — NamedTuple comparison
        # is by symbol set on both sides).
        @test Set(keys(round)) == Set(keys(state))
        @test round.rho == state.rho
        @test round.u == state.u
        @test round.Pxx == state.Pxx
        @test round.n_cells == state.n_cells
        @test round.gamma_law == state.gamma_law
    end
end

@testset "io: JLD2 round-trip" begin
    state = (
        alpha = collect(range(0.1, 1.0; length = 5)),
        beta = zeros(5),
        gamma = fill(0.3, 5),
        meta = "sod",
    )
    mktempdir() do dir
        path = joinpath(dir, "snap.jld2")
        dfmm.save_state(path, state)
        round = dfmm.load_state(path)
        @test Set(keys(round)) == Set(keys(state))
        @test round.alpha == state.alpha
        @test round.beta == state.beta
        @test round.gamma == state.gamma
        @test round.meta == state.meta
    end
end

@testset "io: byte-exact float round-trip" begin
    # Exact equality on Float64 arrays after round-trip.
    state = (data = [0.123456789012345, -1.0e-12, 1.0e+200, 1.0],)
    mktempdir() do dir
        path = joinpath(dir, "exact.h5")
        dfmm.save_state(path, state)
        round = dfmm.load_state(path)
        # === for byte-level comparison.
        @test all(round.data .=== state.data)

        path2 = joinpath(dir, "exact.jld2")
        dfmm.save_state(path2, state)
        round2 = dfmm.load_state(path2)
        @test all(round2.data .=== state.data)
    end
end

@testset "io: unsupported extension errors" begin
    @test_throws ArgumentError dfmm.save_state(
        "/tmp/foo.txt", (a = [1.0],))
    @test_throws ArgumentError dfmm.load_state("/tmp/foo.txt")
end
