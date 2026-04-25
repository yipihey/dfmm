using Test
using dfmm
using CairoMakie

@testset "plotting: helpers return Figure objects" begin
    N = 32
    grid = collect(range(0.0, 1.0; length = N))
    field = sin.(2π .* grid)

    fig = dfmm.plot_profile(grid, field; ax_label = "rho")
    @test fig isa Figure

    alpha_arr = abs.(randn(N)) .+ 0.1
    beta_arr = randn(N)
    fig2 = dfmm.plot_phase_portrait(alpha_arr, beta_arr)
    @test fig2 isa Figure

    gamma_arr = abs.(randn(N))
    det_arr = -gamma_arr .^ 2  # placeholder shape
    fig3 = dfmm.plot_diagnostics(grid, gamma_arr, det_arr)
    @test fig3 isa Figure
end

@testset "plotting: length mismatches raise" begin
    grid = collect(0:0.1:1.0)
    @test_throws AssertionError dfmm.plot_profile(grid, [1.0, 2.0])
    @test_throws AssertionError dfmm.plot_phase_portrait([1.0], [1.0, 2.0])
    @test_throws AssertionError dfmm.plot_diagnostics(grid, [1.0], [1.0])
end
