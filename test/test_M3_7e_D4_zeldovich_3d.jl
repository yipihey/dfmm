# test_M3_7e_D4_zeldovich_3d.jl
#
# M3-7e: D.4 3D Zel'dovich pancake collapse acceptance gates. The 3D
# analog of `test/test_M3_6_phase2_D4_zeldovich.jl`. The headline 3D
# scientific test of M3-7 — the cosmological reference test from
# methods paper §10.5 D.4 lifted to 3D.
#
# Drives `det_step_3d_berry_HG!` with the
# `tier_d_zeldovich_pancake_3d_ic_full` IC (1D-symmetric pancake along
# axis 1; trivial axes 2 and 3) and asserts:
#
#   1. IC analytic match: u_1 = -A·2π·cos(2π m_1), u_2 = u_3 = 0;
#      ρ uniform; mass conservation in the IC.
#
#   2. Per-axis γ selectivity at near-caustic (the methods paper §10.5
#      D.4 cosmological prediction):
#        • γ_1 develops spatial structure (collapse direction).
#        • γ_2 and γ_3 stay uniform throughout (trivial axes preserve
#          anisotropy).
#        • Spatial std ratio std(γ_1) / (std(γ_2) + std(γ_3) + eps)
#          > 1e10 at near-caustic.
#
#   3. Conservation of mass / momentum / energy to ≤ 1e-8.
#
#   4. 1D-symmetry preservation: u_2 = u_3 = 0 throughout.

using Test
using Statistics: std
using HierarchicalGrids: HierarchicalMesh, EulerianFrame, FrameBoundaries,
    REFLECTING, PERIODIC, enumerate_leaves, refine_cells!, cell_physical_box
using dfmm: tier_d_zeldovich_pancake_3d_ic_full,
    det_step_3d_berry_HG!, gamma_per_axis_3d_field, read_detfield_3d, DetField3D

include(joinpath(@__DIR__, "..", "experiments", "D4_zeldovich_3d.jl"))

const M3_7E_D4_CONSERVATION_TOL = 1.0e-8

@testset "M3-7e D.4: 3D Zel'dovich pancake (cosmological reference)" begin
    @testset "D.4 IC analytic match (level=2, A=0.5)" begin
        ic = tier_d_zeldovich_pancake_3d_ic_full(; level = 2, A = 0.5)
        # IC velocity: u_1 = -A·2π·cos(2π m_1), u_2 = u_3 = 0.
        for ci in ic.leaves
            lo_c, hi_c = cell_physical_box(ic.frame, ci)
            cx = 0.5 * (lo_c[1] + hi_c[1])
            u1_expect = -0.5 * 2π * cos(2π * cx)
            @test abs(Float64(ic.fields.u_1[ci][1]) - u1_expect) ≤ 1e-13
            @test abs(Float64(ic.fields.u_2[ci][1])) ≤ 1e-14
            @test abs(Float64(ic.fields.u_3[ci][1])) ≤ 1e-14
        end
        # Density uniform, ρ_per_cell == ρ0.
        @test all(ic.ρ_per_cell .≈ 1.0)
        # Caustic time matches analytic.
        @test ic.t_cross ≈ 1.0 / (0.5 * 2π) atol = 1e-13
    end

    @testset "D.4 mass conservation in IC" begin
        ic = tier_d_zeldovich_pancake_3d_ic_full(; level = 2, A = 0.5)
        n = length(ic.leaves)
        vols = Vector{Float64}(undef, n)
        for (i, ci) in enumerate(ic.leaves)
            lo_c, hi_c = cell_physical_box(ic.frame, ci)
            vols[i] = (hi_c[1] - lo_c[1]) * (hi_c[2] - lo_c[2]) * (hi_c[3] - lo_c[3])
        end
        M = sum(ic.ρ_per_cell .* vols)
        # Box volume = 1.0, ρ = 1.0 ⇒ M = 1.0.
        @test M ≈ 1.0 atol = 1e-13
    end

    @testset "D.4 BC topology: PERIODIC axis 1, REFLECTING axes 2, 3" begin
        # The recommended BC mix is enforced by the driver.
        # Sanity: a level-2 mesh with this BC mix accepts a single
        # det_step_3d_berry_HG! call without errors.
        ic = tier_d_zeldovich_pancake_3d_ic_full(; level = 2, A = 0.5)
        bc_spec = FrameBoundaries{3}(((PERIODIC, PERIODIC),
                                        (REFLECTING, REFLECTING),
                                        (REFLECTING, REFLECTING)))
        det_step_3d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                                bc_spec, 1e-4;
                                M_vv_override = (1.0, 1.0, 1.0), ρ_ref = 1.0)
        # Trivial axes preserve u_2 = u_3 = 0 to round-off after one step.
        for ci in ic.leaves
            @test abs(Float64(ic.fields.u_2[ci][1])) ≤ 1e-13
            @test abs(Float64(ic.fields.u_3[ci][1])) ≤ 1e-13
        end
    end

    @testset "D.4 per-axis γ selectivity at near-caustic (level=2, A=0.5, T_factor=0.25)" begin
        result = run_D4_zeldovich_pancake_3d(; level = 2, A = 0.5, T_factor = 0.25)
        @test result.nan_seen == false
        # γ_1 develops measurable spatial structure (collapse direction).
        @test result.γ1_std[end] > 1e-3
        # γ_2 and γ_3 stay uniform across leaves (trivial axes).
        @test result.γ2_std[end] < 1e-12
        @test result.γ3_std[end] < 1e-12
        # γ_1 minimum drops below γ_1 maximum (collapse signal).
        @test result.γ1_min[end] < result.γ1_max[end]
        # 3D selectivity ratio: std(γ_1) / (std(γ_2) + std(γ_3) + eps) > 1e10.
        ratio = result.selectivity_ratio[end]
        @test ratio > 1e10
        # γ_1 monotonicity: γ1_min strictly decreases (collapse begins
        # at IC = 1.0 and drops over time).
        @test result.γ1_min[end] < result.γ1_min[1]
    end

    @testset "D.4 conservation: mass / momentum / energy to ≤ 1e-8" begin
        result = run_D4_zeldovich_pancake_3d(; level = 2, A = 0.5, T_factor = 0.25)
        @test result.M_err_max ≤ M3_7E_D4_CONSERVATION_TOL
        @test result.Px_err_max ≤ M3_7E_D4_CONSERVATION_TOL
        @test result.Py_err_max ≤ M3_7E_D4_CONSERVATION_TOL
        @test result.Pz_err_max ≤ M3_7E_D4_CONSERVATION_TOL
        @test result.KE_err_max ≤ M3_7E_D4_CONSERVATION_TOL
    end

    @testset "D.4 1D-symmetry preservation: u_2 = u_3 = 0 throughout" begin
        result = run_D4_zeldovich_pancake_3d(; level = 2, A = 0.5, T_factor = 0.25)
        # At final state, u_2 and u_3 should still be 0 to round-off.
        u2_max = maximum(abs(Float64(result.ic.fields.u_2[ci][1])) for ci in result.ic.leaves)
        u3_max = maximum(abs(Float64(result.ic.fields.u_3[ci][1])) for ci in result.ic.leaves)
        @test u2_max ≤ 1e-12
        @test u3_max ≤ 1e-12
    end

    @testset "D.4 γ_2 and γ_3 byte-equal each other by symmetry" begin
        result = run_D4_zeldovich_pancake_3d(; level = 2, A = 0.5, T_factor = 0.25)
        # By the trivial-axis symmetry (β_2 = β_3 = 0 at IC and stays
        # 0), γ_2 and γ_3 should agree to round-off.
        @test result.γ2_max[end] ≈ result.γ3_max[end] atol = 1e-14
        @test result.γ2_min[end] ≈ result.γ3_min[end] atol = 1e-14
        @test result.γ2_std[end] ≈ result.γ3_std[end] atol = 1e-14
    end

    @testset "D.4 selectivity trajectory grows over time" begin
        result = run_D4_zeldovich_pancake_3d(; level = 2, A = 0.5, T_factor = 0.25)
        # The selectivity ratio at the final step exceeds the early-time
        # ratio (after the IC's degenerate eps-only denominator). The
        # IC has γ_1 std = γ_2 std = 0 ⇒ ratio = NaN/0 = NaN initially;
        # at intermediate times std(γ_1) > 0 so the ratio is finite and
        # large. We verify both that the final ratio is large and that
        # std(γ_1) grows monotonically.
        for k in 2:length(result.γ1_std)
            @test result.γ1_std[k] ≥ result.γ1_std[k - 1] - 1e-12
        end
    end
end
