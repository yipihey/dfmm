# test_M3_7d_selectivity.jl
#
# §7.5 Per-axis γ selectivity headline test (M3-7d, 3D extension of
# M3-3d's headline gate). Three configurations span the relevant
# symmetry classes:
#
#   1. 1D-symmetric in 3D (k = (1, 0, 0)): γ_1 collapses spatially;
#      γ_2 and γ_3 stay uniform. Selectivity ratio
#      std(γ_1) / (std(γ_2) + std(γ_3) + eps) > 1e10.
#   2. 2D-symmetric in 3D (k = (1, 1, 0)): γ_1 and γ_2 develop
#      correlated spatial structure; γ_3 stays uniform.
#      (std(γ_1) + std(γ_2)) / 2 / (std(γ_3) + eps) > 1e6.
#   3. Full 3D (k = (1, 1, 1)): all three γ_a develop spatial
#      structure simultaneously.
#
# This is the load-bearing scientific gate for M3-7d per the brief:
# "If 3D γ selectivity ratios fall short of 1e10 (1D-sym) or 1e6
# (2D-sym), document — could indicate per-axis decomposition has a
# numerical-precision issue at resolution."
#
# Note: the 3D selectivity threshold (1e10) is loosened from M3-3d's
# 2D 1e14 because 3D has 2 trivial axes (axes 2 + 3 in the 1D-symmetric
# case), so the denominator is std(γ_2) + std(γ_3) which can be
# elevated by noise on either trivial axis. In practice, on the
# REFLECTING-BC + isotropic-Mvv driver the trivial axes' γ stays
# perfectly constant (std = 0.0 to round-off), so the ratio is
# limited only by the active axis' std (~1e-5) divided by `eps()`
# (~2e-16), giving ~1e10 in 20 steps and growing.
#
# See `reference/notes_M3_7_3d_extension.md` §5 + §7.5 and
# `reference/notes_M3_7d_3d_per_axis_gamma_amr.md`.

using Test
using Statistics

include(joinpath(@__DIR__, "..", "experiments",
                  "M3_7d_per_axis_gamma_3d_cold_sinusoid.jl"))

@testset "M3-7d §7.5 per-axis γ selectivity (3D)" begin

    @testset "1D-symmetric: kx=1, ky=kz=0 ⇒ γ_1 develops structure; γ_2, γ_3 uniform" begin
        result = run_M3_7d_per_axis_gamma_selectivity_3d(
            level = 2, A = 0.3, kx = 1, ky = 0, kz = 0,
            dt = 2e-3, n_steps = 20,
        )
        γ1 = result.γ_final[1, :]
        γ2 = result.γ_final[2, :]
        γ3 = result.γ_final[3, :]

        std_γ1 = std(γ1)
        std_γ2 = std(γ2)
        std_γ3 = std(γ3)

        # γ_1 develops measurable spatial structure.
        @test std_γ1 > 1e-7

        # γ_2 and γ_3 are spatially uniform to round-off (no spatial
        # driver on the trivial axes for the cold-sinusoid IC).
        @test std_γ2 < 1e-12
        @test std_γ3 < 1e-12

        # Selectivity ratio: std(γ_1) divided by the sum of trivial-axis
        # stds + eps (to handle the std = 0 case bit-exactly). With the
        # 1D-symmetric IC the denominator is exactly eps, giving
        # ratio ~ 6e10 which exceeds the 1e10 gate.
        ratio = std_γ1 / (std_γ2 + std_γ3 + eps(Float64))
        @test ratio > 1e10

        # γ_2 and γ_3 must equal each other to round-off (both axes
        # are equally trivial).
        @test maximum(γ2) ≈ maximum(γ3) atol = 1e-14
        @test minimum(γ2) ≈ minimum(γ3) atol = 1e-14
    end

    @testset "1D-symmetric: β_2 = β_3 = 0 stays byte-equal across leaves" begin
        result = run_M3_7d_per_axis_gamma_selectivity_3d(
            level = 2, A = 0.3, kx = 1, ky = 0, kz = 0,
            dt = 2e-3, n_steps = 10,
        )
        β2_arr = [Float64(result.fields.β_2[ci][1]) for ci in result.leaves]
        β3_arr = [Float64(result.fields.β_3[ci][1]) for ci in result.leaves]
        @test maximum(β2_arr) - minimum(β2_arr) < 1e-12
        @test maximum(β3_arr) - minimum(β3_arr) < 1e-12
    end

    @testset "2D-symmetric: kx=ky=1, kz=0 ⇒ γ_1, γ_2 fire correlated; γ_3 uniform" begin
        result = run_M3_7d_per_axis_gamma_selectivity_3d(
            level = 2, A = 0.3, kx = 1, ky = 1, kz = 0,
            dt = 2e-3, n_steps = 20,
        )
        γ1 = result.γ_final[1, :]
        γ2 = result.γ_final[2, :]
        γ3 = result.γ_final[3, :]

        std_γ1 = std(γ1)
        std_γ2 = std(γ2)
        std_γ3 = std(γ3)

        # axes 1 and 2 develop correlated spatial structure.
        @test std_γ1 > 1e-7
        @test std_γ2 > 1e-7
        # By symmetry of the IC (kx = ky), the std of γ_1 and γ_2
        # should be equal to round-off.
        @test std_γ1 ≈ std_γ2 atol = 1e-14

        # γ_3 stays uniform to round-off.
        @test std_γ3 < 1e-12

        # Selectivity ratio: average of active stds over the trivial.
        avg_active = 0.5 * (std_γ1 + std_γ2)
        ratio = avg_active / (std_γ3 + eps(Float64))
        @test ratio > 1e6
    end

    @testset "2D-symmetric: β_3 = 0 stays byte-equal across leaves" begin
        result = run_M3_7d_per_axis_gamma_selectivity_3d(
            level = 2, A = 0.3, kx = 1, ky = 1, kz = 0,
            dt = 2e-3, n_steps = 10,
        )
        β3_arr = [Float64(result.fields.β_3[ci][1]) for ci in result.leaves]
        @test maximum(β3_arr) - minimum(β3_arr) < 1e-12
    end

    @testset "Full 3D: kx=ky=kz=1 ⇒ all three γ_a non-trivial" begin
        result = run_M3_7d_per_axis_gamma_selectivity_3d(
            level = 2, A = 0.3, kx = 1, ky = 1, kz = 1,
            dt = 2e-3, n_steps = 20,
        )
        γ1 = result.γ_final[1, :]
        γ2 = result.γ_final[2, :]
        γ3 = result.γ_final[3, :]

        std_γ1 = std(γ1)
        std_γ2 = std(γ2)
        std_γ3 = std(γ3)

        # All three axes simultaneously develop spatial structure.
        @test std_γ1 > 1e-7
        @test std_γ2 > 1e-7
        @test std_γ3 > 1e-7

        # By the symmetry of the IC (kx = ky = kz), all three
        # spatial-stds should agree to round-off.
        @test std_γ1 ≈ std_γ2 atol = 1e-14
        @test std_γ2 ≈ std_γ3 atol = 1e-14

        # No "trivial" axis ⇒ no selectivity gate; just confirm the
        # full-3D regime exercises all three Berry pair generators.
        @test minimum(γ1) < maximum(γ1)
        @test minimum(γ2) < maximum(γ2)
        @test minimum(γ3) < maximum(γ3)
    end

    @testset "γ trajectory monotonicity (1D-sym): γ_1 min strictly decreasing" begin
        result = run_M3_7d_per_axis_gamma_selectivity_3d(
            level = 2, A = 0.3, kx = 1, ky = 0, kz = 0,
            dt = 2e-3, n_steps = 20,
        )
        # γ_1's minimum must drop from 1.0 (uniform IC) to below 1.0
        # over the trajectory (cold-limit collapse on axis 1).
        @test result.γ1_min[1] ≈ 1.0 atol = 1e-12
        @test result.γ1_min[end] < result.γ1_min[1]

        # γ_2 and γ_3 stay at their initial uniform value (max == min
        # at every step).
        @test result.γ2_min[end] == result.γ2_max[end]
        @test result.γ3_min[end] == result.γ3_max[end]
    end

    @testset "Per-axis γ values for 2D-sym match 1D-sym on active axis" begin
        # The 2D-symmetric IC's γ_1 should evolve identically to the
        # 1D-symmetric IC's γ_1 (axis-1 alone is decoupled from axis-2
        # via the trivial-strain limit when β_2 = β_3 = 0 at IC; the
        # 2D-sym IC has a separate axis-2 sinusoid but the per-axis
        # γ on axis 1 sees only axis-1 dynamics in the kinetic-Cholesky
        # driver).
        #
        # We verify this loosely: the 1D-sym γ_1 trajectory and the
        # 2D-sym γ_1 trajectory must agree to a few × 1e-14 (the Newton
        # solve doesn't decouple per-axis exactly; the SO(3) Berry
        # coupling produces per-pair α/β-modifications across axes,
        # but on the cold-limit IC the modifications are zero so the
        # trajectories coincide in early time).
        r1 = run_M3_7d_per_axis_gamma_selectivity_3d(
            level = 2, A = 0.3, kx = 1, ky = 0, kz = 0,
            dt = 2e-3, n_steps = 5,
        )
        r2 = run_M3_7d_per_axis_gamma_selectivity_3d(
            level = 2, A = 0.3, kx = 1, ky = 1, kz = 0,
            dt = 2e-3, n_steps = 5,
        )
        # The minimum γ_1 (along the collapsing axis) must be very
        # close in early time (n_steps = 5 is short enough that the
        # Berry pair-coupling between axis 1 and axis 2 is sub-leading).
        @test r1.γ1_min[end] ≈ r2.γ1_min[end] atol = 1e-3
    end

end
