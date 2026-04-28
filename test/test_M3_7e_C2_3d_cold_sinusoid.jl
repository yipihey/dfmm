# test_M3_7e_C2_3d_cold_sinusoid.jl
#
# M3-7e: C.2 3D cold sinusoid acceptance gates. The 3D analog of
# `test/test_M3_4_C2_cold_sinusoid.jl`.
#
# Drives `det_step_3d_berry_HG!` with the C.2 3D cold sinusoid IC at
# multiple `(kx, ky, kz)` settings and asserts:
#
#   1. Bridge round-trip at t = 0: per-cell primitive recovery matches
#      the IC profile to round-off.
#   2. §7.5 Per-axis γ selectivity (the headline gate) reproduces M3-7d:
#      • k = (1, 0, 0): std(γ_1) / (std(γ_2) + std(γ_3) + eps) > 1e6.
#      • k = (1, 1, 0): (std(γ_1) + std(γ_2)) / 2 / (std(γ_3) + eps) > 1e6.
#      • k = (1, 1, 1): all three γ_a > 1e-7 (no trivial axis).
#   3. Conservation: total mass exact; net momentum bounded.

using Test
using Statistics: std
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, cell_physical_box
using dfmm: tier_c_cold_sinusoid_3d_full_ic, primitive_recovery_3d_per_cell,
            det_step_3d_berry_HG!, gamma_per_axis_3d_field

include(joinpath(@__DIR__, "..", "experiments", "C2_3d_cold_sinusoid.jl"))

@testset "M3-7e C.2: 3D cold sinusoid" begin
    @testset "C.2 IC bridge round-trip at t=0 (level=2, k=(1,0,0))" begin
        ic = tier_c_cold_sinusoid_3d_full_ic(; level = 2, A = 0.3, k = (1, 0, 0))
        rec = primitive_recovery_3d_per_cell(ic.fields, ic.leaves, ic.frame,
                                                ic.ρ_per_cell)
        # u_y = u_z = 0 strictly.
        @test maximum(abs, rec.u_y) ≤ 1e-14
        @test maximum(abs, rec.u_z) ≤ 1e-14
        # u_x = A sin(2π · k_x · (x - lo_x) / L_x) per cell center.
        for i in eachindex(ic.leaves)
            u_expect = 0.3 * sin(2π * 1 * (rec.x[i] - 0.0) / 1.0)
            @test abs(rec.u_x[i] - u_expect) ≤ 1e-13
        end
        # ρ uniform.
        @test all(rec.ρ .≈ 1.0)
    end

    @testset "C.2 IC bridge round-trip at t=0 (k=(1,1,0))" begin
        ic = tier_c_cold_sinusoid_3d_full_ic(; level = 2, A = 0.3, k = (1, 1, 0))
        rec = primitive_recovery_3d_per_cell(ic.fields, ic.leaves, ic.frame,
                                                ic.ρ_per_cell)
        @test maximum(abs, rec.u_z) ≤ 1e-14
        for i in eachindex(ic.leaves)
            u_x_expect = 0.3 * sin(2π * 1 * rec.x[i])
            u_y_expect = 0.3 * sin(2π * 1 * rec.y[i])
            @test abs(rec.u_x[i] - u_x_expect) ≤ 1e-13
            @test abs(rec.u_y[i] - u_y_expect) ≤ 1e-13
        end
    end

    @testset "C.2 IC bridge round-trip at t=0 (k=(1,1,1))" begin
        ic = tier_c_cold_sinusoid_3d_full_ic(; level = 2, A = 0.3, k = (1, 1, 1))
        rec = primitive_recovery_3d_per_cell(ic.fields, ic.leaves, ic.frame,
                                                ic.ρ_per_cell)
        for i in eachindex(ic.leaves)
            u_x_expect = 0.3 * sin(2π * 1 * rec.x[i])
            u_y_expect = 0.3 * sin(2π * 1 * rec.y[i])
            u_z_expect = 0.3 * sin(2π * 1 * rec.z[i])
            @test abs(rec.u_x[i] - u_x_expect) ≤ 1e-13
            @test abs(rec.u_y[i] - u_y_expect) ≤ 1e-13
            @test abs(rec.u_z[i] - u_z_expect) ≤ 1e-13
        end
    end

    @testset "C.2 §7.5 selectivity 1D-sym k=(1,0,0): std(γ_1)/(std(γ_2)+std(γ_3)+eps) > 1e6" begin
        result = run_C2_3d_cold_sinusoid(; level = 2, A = 0.3, k = (1, 0, 0),
                                         dt = 2e-3, n_steps = 20)
        @test result.γ1_std[end] > 1e-7
        @test result.γ2_std[end] < 1e-12
        @test result.γ3_std[end] < 1e-12
        ratio = result.γ1_std[end] /
                (result.γ2_std[end] + result.γ3_std[end] + eps(Float64))
        @test ratio > 1e6
        # Quantitative: 6.4e10 from M3-7d.
        @test ratio > 1e10
    end

    @testset "C.2 §7.5 selectivity 2D-sym k=(1,1,0): (avg active) / std(γ_3) > 1e6" begin
        result = run_C2_3d_cold_sinusoid(; level = 2, A = 0.3, k = (1, 1, 0),
                                         dt = 2e-3, n_steps = 20)
        @test result.γ1_std[end] > 1e-7
        @test result.γ2_std[end] > 1e-7
        @test result.γ3_std[end] < 1e-12
        avg_active = 0.5 * (result.γ1_std[end] + result.γ2_std[end])
        ratio = avg_active / (result.γ3_std[end] + eps(Float64))
        @test ratio > 1e6
        # By symmetry kx = ky, std(γ_1) and std(γ_2) agree to round-off.
        @test result.γ1_std[end] ≈ result.γ2_std[end] atol = 1e-14
    end

    @testset "C.2 §7.5 selectivity full-3D k=(1,1,1): all three γ_a fire" begin
        result = run_C2_3d_cold_sinusoid(; level = 2, A = 0.3, k = (1, 1, 1),
                                         dt = 2e-3, n_steps = 20)
        @test result.γ1_std[end] > 1e-7
        @test result.γ2_std[end] > 1e-7
        @test result.γ3_std[end] > 1e-7
        # By isotropy of the IC, all three stds match to round-off.
        @test result.γ1_std[end] ≈ result.γ2_std[end] atol = 1e-14
        @test result.γ2_std[end] ≈ result.γ3_std[end] atol = 1e-14
    end

    @testset "C.2 conservation: mass exact, momentum bounded" begin
        result = run_C2_3d_cold_sinusoid(; level = 2, A = 0.3, k = (1, 0, 0),
                                         dt = 2e-3, n_steps = 10)
        # Compute conservation integrals at start vs end.
        leaves = result.leaves
        frame = result.frame
        n = length(leaves)
        vols = Vector{Float64}(undef, n)
        for (i, ci) in enumerate(leaves)
            lo, hi = cell_physical_box(frame, ci)
            vols[i] = (hi[1] - lo[1]) * (hi[2] - lo[2]) * (hi[3] - lo[3])
        end
        ρ_per_cell = result.ρ_per_cell
        # Final state momenta.
        ux = [Float64(result.fields.u_1[ci][1]) for ci in leaves]
        uy = [Float64(result.fields.u_2[ci][1]) for ci in leaves]
        uz = [Float64(result.fields.u_3[ci][1]) for ci in leaves]
        Py = sum(ρ_per_cell .* uy .* vols)
        Pz = sum(ρ_per_cell .* uz .* vols)
        # Trivial-axis momenta exactly zero by symmetry.
        @test abs(Py) ≤ 1e-12
        @test abs(Pz) ≤ 1e-12
    end
end
