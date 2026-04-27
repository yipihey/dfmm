# test_plotting_unit.jl
#
# Beyond-crash smoke tests for the CairoMakie helpers in
# `src/plotting.jl`. The existing `test_plotting.jl` confirms each
# helper returns a `Figure`; this file additionally verifies:
#
#   * Figures save to PNG via the full CairoMakie pipeline (catches
#     issues that pure construction misses).
#   * Empty-input edge cases work (zero-length matched vectors).
#   * Saved files have non-trivial size, indicating a real render.
#
# Axis-label inspection is intentionally avoided because the Makie
# Block / Axis API path varies across versions; the file size check
# is a robust proxy for "rendering succeeded end-to-end".

using Test
using dfmm
using CairoMakie

@testset "plot_profile: PNG save round-trip succeeds" begin
    grid = collect(range(0.0, 1.0; length = 16))
    field = sin.(2π .* grid)
    fig = dfmm.plot_profile(grid, field)
    @test fig isa Figure
    mktempdir() do dir
        path = joinpath(dir, "profile.png")
        CairoMakie.save(path, fig)
        @test isfile(path)
        @test filesize(path) > 100
    end
end

@testset "plot_profile: ax_label keyword accepted (smoke)" begin
    # No-crash with a custom y-label.
    grid = collect(range(0.0, 1.0; length = 8))
    field = grid .^ 2
    fig = dfmm.plot_profile(grid, field; ax_label = "u_velocity")
    @test fig isa Figure
end

@testset "plot_phase_portrait: PNG save round-trip succeeds" begin
    α = abs.(randn(16)) .+ 0.1
    β = randn(16)
    fig = dfmm.plot_phase_portrait(α, β)
    @test fig isa Figure
    mktempdir() do dir
        path = joinpath(dir, "phase.png")
        CairoMakie.save(path, fig)
        @test isfile(path)
        @test filesize(path) > 100
    end
end

@testset "plot_phase_portrait: empty matched-length input does not crash" begin
    fig = dfmm.plot_phase_portrait(Float64[], Float64[])
    @test fig isa Figure
end

@testset "plot_diagnostics: 2-panel layout PNG save round-trip" begin
    grid = collect(range(0.0, 1.0; length = 16))
    gamma_arr = abs.(randn(16))
    det_arr = -gamma_arr .^ 2
    fig = dfmm.plot_diagnostics(grid, gamma_arr, det_arr)
    @test fig isa Figure
    mktempdir() do dir
        path = joinpath(dir, "diagnostics.png")
        CairoMakie.save(path, fig)
        @test isfile(path)
        @test filesize(path) > 100
    end
end

@testset "plot_diagnostics: lengths must match across all three inputs" begin
    grid = collect(range(0.0, 1.0; length = 8))
    @test_throws AssertionError dfmm.plot_diagnostics(
        grid, [1.0, 2.0], grid)
    @test_throws AssertionError dfmm.plot_diagnostics(
        grid, grid, [1.0])
end
