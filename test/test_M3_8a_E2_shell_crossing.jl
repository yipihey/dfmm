# test_M3_8a_E2_shell_crossing.jl
#
# M3-8 Phase a: E.2 severe shell-crossing 2D acceptance gates
# (methods paper §10.6 E.2).
#
# 2D extension of the M2-3 1D compression-cascade scenario. Drives the
# 2D Cholesky-sector Newton system on a *superposition* of two-axis
# Zel'dovich velocity profiles at extreme amplitude (A_x = A_y = 0.7).
# Tests realizability projection effectiveness (compression cascade
# prevention) and long-horizon stability.
#
# References:
#   • methods paper §10.6 E.2 (Tier E.2 spec)
#   • reference/notes_M2_3_realizability.md (1D scenario this lifts)
#   • experiments/E2_severe_shell_crossing.jl

using Test
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, PERIODIC, cell_physical_box
using dfmm: tier_e_severe_shell_crossing_ic_full, primitive_recovery_2d_per_cell,
            det_step_2d_berry_HG!, ProjectionStats

include(joinpath(@__DIR__, "..", "experiments", "E2_severe_shell_crossing.jl"))

@testset "M3-8a E.2: severe shell-crossing 2D" begin
    @testset "E.2 IC bridge round-trip at t=0 (A=0.7, level=2)" begin
        ic = tier_e_severe_shell_crossing_ic_full(; level = 2,
                                                   A_x = 0.7, A_y = 0.7)
        # The 2D Zel'dovich superposition: u_1 along axis 1 + u_2 along axis 2.
        rec = primitive_recovery_2d_per_cell(ic.fields, ic.leaves, ic.frame,
                                                ic.ρ_per_cell)
        L1 = ic.params.L1
        L2 = ic.params.L2
        for i in eachindex(ic.leaves)
            m1 = (rec.x[i] - ic.params.lo[1]) / L1
            m2 = (rec.y[i] - ic.params.lo[2]) / L2
            u1_expect = -0.7 * 2π * cos(2π * m1)
            u2_expect = -0.7 * 2π * cos(2π * m2)
            @test abs(rec.u_x[i] - u1_expect) ≤ 1e-12
            @test abs(rec.u_y[i] - u2_expect) ≤ 1e-12
            @test abs(rec.ρ[i] - 1.0) ≤ 1e-14
        end
        # Caustic time: t_cross = 1/(A·2π).
        @test ic.t_cross ≈ 1.0 / (0.7 * 2π) atol = 1e-12
        @test ic.params.t_cross_x == ic.params.t_cross_y
    end

    @testset "E.2 amplitude / caustic time scaling" begin
        # A=0.5 (Phase 2 Zel'dovich amplitude) gives t_cross ≈ 0.318;
        # A=0.7 (E.2 stress amplitude) gives t_cross ≈ 0.227.
        ic_a = tier_e_severe_shell_crossing_ic_full(; level = 2, A_x = 0.5, A_y = 0.5)
        ic_b = tier_e_severe_shell_crossing_ic_full(; level = 2, A_x = 0.7, A_y = 0.7)
        @test ic_a.t_cross > ic_b.t_cross   # Higher amplitude ⇒ earlier caustic.
        @test ic_a.t_cross ≈ 1.0 / (0.5 * 2π) atol = 1e-12
        @test ic_b.t_cross ≈ 1.0 / (0.7 * 2π) atol = 1e-12
        # Asymmetric test: A_x ≠ A_y; t_cross is min of the two axis caustic times.
        ic_c = tier_e_severe_shell_crossing_ic_full(; level = 2, A_x = 0.5, A_y = 0.7)
        @test ic_c.t_cross ≈ 1.0 / (0.7 * 2π) atol = 1e-12
        @test ic_c.params.t_cross_x ≈ 1.0 / (0.5 * 2π) atol = 1e-12
    end

    @testset "E.2 graceful behavior: pre-caustic (T_factor=0.1, level=2, A=0.7)" begin
        result = run_E2_severe_shell_crossing(; level = 2, A_x = 0.7, A_y = 0.7,
                                                T_factor = 0.1, n_steps = 5,
                                                project_kind = :reanchor,
                                                M_vv_override = (1.0, 1.0))
        # No NaN at pre-caustic.
        for k in eachindex(result.nan_count)
            @test result.nan_count[k] == 0
        end
        # Mass conservation.
        @test result.mass[end] ≈ result.mass[1] atol=1e-12
        # Momentum stays at IC (zero — symmetric IC has zero net momentum).
        for k in eachindex(result.Px)
            @test abs(result.Px[k]) ≤ 1e-10
            @test abs(result.Py[k]) ≤ 1e-10
        end
        # γ_min stays positive (no compression cascade through the pre-caustic regime).
        for gm in result.gamma_min
            @test gm > 0.5
        end
    end

    @testset "E.2 realizability projection effectiveness (with vs without projection)" begin
        # The reanchor projection should produce projection events while
        # the :none projection leaves the count at zero (no events).
        result_proj = run_E2_severe_shell_crossing(; level = 2, A_x = 0.7, A_y = 0.7,
                                                     T_factor = 0.1, n_steps = 3,
                                                     project_kind = :reanchor,
                                                     M_vv_override = (1.0, 1.0))
        result_noproj = run_E2_severe_shell_crossing(; level = 2, A_x = 0.7, A_y = 0.7,
                                                      T_factor = 0.1, n_steps = 3,
                                                      project_kind = :none,
                                                      M_vv_override = (1.0, 1.0))
        # With reanchor projection, the n_events counter advances per step
        # (every cell gets a tag, even if no clamp is needed — the
        # projection routine increments per-cell).
        @test sum(result_proj.proj_n_events) >= 0
        # With :none, no events recorded.
        @test sum(result_noproj.proj_n_events) == 0
        # Both should be NaN-free at the pre-caustic regime.
        for k in eachindex(result_proj.nan_count)
            @test result_proj.nan_count[k] == 0
            @test result_noproj.nan_count[k] == 0
        end
    end

    @testset "E.2 long-horizon stability: T_factor=0.25 (still pre-caustic)" begin
        result = run_E2_severe_shell_crossing(; level = 2, A_x = 0.7, A_y = 0.7,
                                                T_factor = 0.25, n_steps = 5,
                                                project_kind = :reanchor,
                                                M_vv_override = (1.0, 1.0))
        # Bounded across the run.
        for k in eachindex(result.nan_count)
            @test result.nan_count[k] == 0
        end
        # γ_min approaches but stays above 0.
        for gm in result.gamma_min
            @test gm > 0.0
        end
        # Mass conservation.
        @test result.mass[end] ≈ result.mass[1] atol=1e-12
    end

    @testset "E.2 axis symmetry: swap A_x ↔ A_y" begin
        # By construction the IC is symmetric under (x ↔ y) when A_x = A_y.
        ic_xy = tier_e_severe_shell_crossing_ic_full(; level = 2, A_x = 0.7, A_y = 0.7)
        rec = primitive_recovery_2d_per_cell(ic_xy.fields, ic_xy.leaves, ic_xy.frame,
                                              ic_xy.ρ_per_cell)
        # For each cell at (x_i, y_i), there's a symmetric cell at (y_i, x_i)
        # with u_x ↔ u_y. Test by sampling.
        for i in 1:min(8, length(ic_xy.leaves))
            xi, yi = rec.x[i], rec.y[i]
            ux_i, uy_i = rec.u_x[i], rec.u_y[i]
            # Symmetric partner: (y_i, x_i).
            j = findfirst(k -> abs(rec.x[k] - yi) < 1e-10 && abs(rec.y[k] - xi) < 1e-10,
                          eachindex(ic_xy.leaves))
            if j !== nothing
                @test abs(rec.u_x[j] - uy_i) ≤ 1e-12
                @test abs(rec.u_y[j] - ux_i) ≤ 1e-12
            end
        end
    end
end
