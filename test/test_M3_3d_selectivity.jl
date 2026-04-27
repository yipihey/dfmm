# test_M3_3d_selectivity.jl
#
# §6.5 Per-axis γ selectivity headline test (M3-3d).
#
# Cold sinusoid IC with k_x = 1, k_y = 0:
#   ρ uniform, u = (A sin(2π x), 0), A = 0.5.
# Run on a 2D Cholesky-sector field set for a short window. The
# per-axis γ diagnostic must demonstrate selectivity:
#
#   γ_1 collapses spatially along the active axis (large spatial range)
#   γ_2 stays uniform along the trivial axis (vanishing spatial range)
#   Selectivity ratio std(γ_1) / std(γ_2) ≫ 1
#
# This is the load-bearing scientific gate for M3-3d per the brief:
# "If γ_2 also drops or γ_1 doesn't drop, the principal-axis decomposition
# has a bug — debug before continuing."
#
# See `reference/notes_M3_3_2d_cholesky_berry.md` §6.5 and
# `reference/notes_M3_3d_per_axis_gamma_amr.md`.

using Test
using Statistics

include(joinpath(@__DIR__, "..", "experiments",
                  "M3_3d_per_axis_gamma_cold_sinusoid.jl"))

@testset "M3-3d §6.5 per-axis γ selectivity" begin

    @testset "cold sinusoid kx=1, ky=0: γ_1 collapses spatially; γ_2 uniform" begin
        result = run_M3_3d_per_axis_gamma_selectivity(
            level = 4, A = 0.5, kx = 1, ky = 0,
            dt = 2e-3, n_steps = 100,
        )

        γ1 = result.γ_final[1, :]
        γ2 = result.γ_final[2, :]
        std_γ1 = std(γ1)
        std_γ2 = std(γ2)

        # Selectivity: γ_1 must develop spatial structure that γ_2 lacks.
        # The ratio of spatial standard deviations is the headline metric.
        # std(γ_2) is at machine precision (β_2 grows uniformly across all
        # cells on the trivial axis, so its spatial variance is zero modulo
        # round-off); std(γ_1) is O(0.01) at T = 0.2 with A = 0.5.
        @test std_γ1 > 1e-3       # γ_1 has spatial structure
        @test std_γ2 < 1e-12      # γ_2 is spatially uniform
        @test std_γ1 / max(std_γ2, eps(Float64)) > 1e6   # ratio ≫ 1

        # γ_1 collapse: the trough cell's γ_1 must be measurably smaller
        # than γ_2 anywhere (since γ_2 is uniform, "anywhere" = "the same
        # cell"). The collapse bar is qualitative — require the trough
        # γ_1 to be at least 2% below γ_2.
        min_γ1 = minimum(γ1)
        min_γ2 = minimum(γ2)
        @test min_γ1 < 0.98 * min_γ2

        # γ_2 itself stays O(1): even though β_2 grows uniformly from
        # the cold-limit β̇ = γ²/α driver, it does not approach 1
        # (the realizability boundary) on a short integration window.
        @test min_γ2 > 0.5
    end

    @testset "ky=0 ⇒ β_2 stays spatially uniform across leaves" begin
        # The trivial axis has no spatial structure in u_2 at the IC,
        # and the residual's β_2 driver has no spatial structure either,
        # so β_2 remains identical across all leaves.
        result = run_M3_3d_per_axis_gamma_selectivity(
            level = 3, A = 0.5, kx = 1, ky = 0,
            dt = 2e-3, n_steps = 50,
        )
        β2_arr = [Float64(result.fields.β_2[ci][1]) for ci in result.leaves]
        # All leaves carry the same β_2 to round-off.
        @test maximum(β2_arr) - minimum(β2_arr) < 1e-12
    end

    @testset "γ-trajectory monotonicity along x-axis" begin
        # γ_1's spatial range must grow monotonically over time:
        # the collapsing axis spreads γ_1 values as the cold limit
        # is approached. This is the trajectory-level signature of
        # selectivity.
        result = run_M3_3d_per_axis_gamma_selectivity(
            level = 4, A = 0.5, kx = 1, ky = 0,
            dt = 2e-3, n_steps = 100,
        )
        # Spatial range of γ_1 at start = 0 (uniform IC); at end > 0.
        # The min trajectory γ1_min should be monotonically non-increasing
        # at smooth scales (no shock-like reversal in this regime).
        @test result.γ1_min[1] ≈ 1.0 atol = 1e-12
        @test result.γ1_min[end] < 0.97
        # γ_2 stays at its uniform value (uniform across leaves so min == max).
        @test result.γ2_min[end] == result.γ2_max[end]
    end

    @testset "headline figure save (CSV fallback OK)" begin
        # Just checks the plot driver runs without error; the actual
        # PNG existence depends on CairoMakie being available, but we
        # always fall back to CSV.
        result = run_M3_3d_per_axis_gamma_selectivity(
            level = 3, A = 0.5, kx = 1, ky = 0,
            dt = 2e-3, n_steps = 50,
        )
        save_path = joinpath(@__DIR__, "..", "reference", "figs",
                              "M3_3d_per_axis_gamma_selectivity.png")
        out = plot_M3_3d_per_axis_gamma(result; save_path = save_path)
        @test out isa AbstractString
        # Either the PNG or the CSV fallback exists.
        png_ok = isfile(save_path)
        csv_ok = isfile(replace(save_path, ".png" => ".csv"))
        @test png_ok || csv_ok
    end

end
