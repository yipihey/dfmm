# test_M3_3d_realizability_per_axis.jl
#
# M3-3d unit tests for `realizability_project_2d!`. Three blocks:
#
#   1. Project on a 2D field with M_vv > headroom·max_a β²: no-op.
#   2. Project on a 2D field with M_vv < headroom·max_a β²: s rises so
#      M_vv_post ≥ headroom·max_a β².
#   3. Off-diagonal-coupling-free: per-axis projection on a 1D-symmetric
#      configuration (β_2 = 0) reproduces the M2-3 1D
#      `realizability_project!` `M_vv_target` to ≤ 1e-12.
#
# See `reference/notes_M2_3_realizability.md` (1D design) and
# `reference/notes_M3_3d_per_axis_gamma_amr.md`.

using Test
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves
using dfmm: realizability_project_2d!, ProjectionStats,
            allocate_cholesky_2d_fields, write_detfield_2d!,
            DetField2D, Mvv

@testset "M3-3d per-axis realizability projection" begin

    @testset "no-op when M_vv ≥ headroom·max_a β²" begin
        mesh = HierarchicalMesh{2}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))   # one level
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_2d_fields(mesh)

        # Choose s so Mvv(1, s) is well above 1.05 · max(β₁², β₂²) = 1.05·0.04.
        # Mvv(1, 0) = exp(0) = 1.0 ≫ 0.042 ⇒ no-op.
        for ci in leaves
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (0.20, 0.10),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        s_pre = [Float64(fields.s[ci][1]) for ci in leaves]
        Pp_pre = [Float64(fields.Pp[ci][1]) for ci in leaves]
        stats = ProjectionStats()
        realizability_project_2d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05,
                                   Mvv_floor = 1e-2,
                                   stats = stats)
        for (k, ci) in enumerate(leaves)
            @test fields.s[ci][1] == s_pre[k]
            @test fields.Pp[ci][1] == Pp_pre[k]
        end
        @test stats.n_steps == 1
        @test stats.n_events == 0
    end

    @testset "fires when M_vv < headroom·max_a β²" begin
        mesh = HierarchicalMesh{2}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_2d_fields(mesh)

        # Set s so Mvv(1, s) is BELOW the realizability target.
        # Choose β = (0.95, 0.40); max β² = 0.9025; headroom = 1.05 ⇒
        # target_rel ≈ 0.948. Pick s = log(0.50) so Mvv = 0.50 < target.
        s0 = log(0.50)
        for ci in leaves
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (0.95, 0.40),
                            0.0, s0, 0.5, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        stats = ProjectionStats()
        realizability_project_2d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05,
                                   Mvv_floor = 1e-2,
                                   stats = stats)
        @test stats.n_steps == 1
        @test stats.n_events == length(leaves)

        # Post-projection: Mvv should reach target.
        target = 1.05 * 0.95^2
        for ci in leaves
            s_new = Float64(fields.s[ci][1])
            Mvv_new = Mvv(1.0, s_new)
            @test Mvv_new ≈ target atol = 1e-12
        end
    end

    @testset "no-op variant: project_kind = :none" begin
        mesh = HierarchicalMesh{2}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_2d_fields(mesh)

        # Pre-state with deliberately violating β.
        s0 = log(0.10)
        for ci in leaves
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (0.95, 0.40),
                            0.0, s0, 0.5, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        s_pre = [Float64(fields.s[ci][1]) for ci in leaves]
        stats = ProjectionStats()
        realizability_project_2d!(fields, leaves;
                                   project_kind = :none,
                                   stats = stats)
        @test stats.n_events == 0
        @test stats.n_steps == 1
        for (k, ci) in enumerate(leaves)
            @test fields.s[ci][1] == s_pre[k]
        end
    end

    @testset "1D-symmetric reduction (β_2 = 0) matches 1D M_vv_target" begin
        # On a 1D-symmetric 2D state with β_2 = 0, the per-axis 2D
        # target reduces to `headroom · β_1²` — identical to the 1D
        # `realizability_project!` `M_vv_target_rel`.
        mesh = HierarchicalMesh{2}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_2d_fields(mesh)

        β1 = 0.30
        s0 = log(0.05)   # Mvv = 0.05, target_rel = 1.05 · 0.09 = 0.0945
        for ci in leaves
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (β1, 0.0),
                            0.0, s0, 0.5, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        realizability_project_2d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05,
                                   Mvv_floor = 1e-3)
        target_1d = 1.05 * β1^2
        for ci in leaves
            Mvv_post = Mvv(1.0, Float64(fields.s[ci][1]))
            @test Mvv_post ≈ target_1d atol = 1e-12
        end
    end

end
