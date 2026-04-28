# test_M3_7d_realizability_3d.jl
#
# M3-7d unit tests for `realizability_project_3d!`. Mirrors the M3-3d
# 2D realizability test pattern, lifted to three axes. Five blocks:
#
#   1. No-op when `M_vv ≥ headroom · max_a β_a²` for all three axes.
#   2. Fires when `M_vv < headroom · max_a β_a²`: s rises so all
#      three axes satisfy `M_vv,aa ≥ headroom · β_a²`.
#   3. `:none` variant is a strict no-op (M3-3d 2D + M2-3 1D parity
#      mirror).
#   4. 2D-symmetric reduction (β_3 = 0): the 3D per-axis target
#      reproduces the M3-3d 2D `M_vv_target = headroom · max_a(β_1², β_2²)`
#      to ≤ 1e-12. Verified by running both projections on parallel
#      states and asserting bit-equal s_post values.
#   5. ProjectionStats3D records events (n_events, n_floor_events,
#      n_steps, total_dE_inj, Mvv_min_pre/post).
#
# See `reference/notes_M3_7_3d_extension.md` §4.4 + §7.5 and
# `reference/notes_M3_7d_3d_per_axis_gamma_amr.md`.

using Test
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves
using dfmm: realizability_project_3d!, realizability_project_2d!,
            ProjectionStats3D, ProjectionStats,
            allocate_cholesky_3d_fields, write_detfield_3d!,
            allocate_cholesky_2d_fields, write_detfield_2d!,
            DetField3D, DetField2D, Mvv
import dfmm

@testset "M3-7d per-axis realizability projection (3D)" begin

    @testset "no-op when M_vv ≥ headroom·max_a β_a²" begin
        mesh = HierarchicalMesh{3}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))   # 8 leaves
        leaves = enumerate_leaves(mesh)
        @test length(leaves) == 8
        fields = allocate_cholesky_3d_fields(mesh)

        # Mvv(1, 0) = 1.0; max β² = 0.04 (β_1 = 0.20). 1.05 · 0.04 = 0.042.
        # 1.0 ≫ 0.042 ⇒ no-op.
        for ci in leaves
            v = DetField3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                            (1.0, 1.0, 1.0), (0.20, 0.10, 0.05),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_3d!(fields, ci, v)
        end

        s_pre = [Float64(fields.s[ci][1]) for ci in leaves]
        stats = ProjectionStats3D()
        realizability_project_3d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05,
                                   Mvv_floor = 1e-2,
                                   stats = stats)
        for (k, ci) in enumerate(leaves)
            @test fields.s[ci][1] == s_pre[k]
        end
        @test stats.n_steps == 1
        @test stats.n_events == 0
        @test stats.n_floor_events == 0
        @test stats.total_dE_inj == 0.0
    end

    @testset "fires when M_vv < headroom·max_a β_a² (β_1 binding)" begin
        mesh = HierarchicalMesh{3}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_3d_fields(mesh)

        # β = (0.95, 0.40, 0.30); max β² = 0.9025; target = 1.05 · 0.9025 ≈ 0.948.
        # Pick s = log(0.50) so Mvv = 0.50 < target.
        s0 = log(0.50)
        for ci in leaves
            v = DetField3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                            (1.0, 1.0, 1.0), (0.95, 0.40, 0.30),
                            0.0, 0.0, 0.0, s0)
            write_detfield_3d!(fields, ci, v)
        end

        stats = ProjectionStats3D()
        realizability_project_3d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05,
                                   Mvv_floor = 1e-2,
                                   stats = stats)
        @test stats.n_steps == 1
        @test stats.n_events == length(leaves)
        @test stats.n_floor_events == 0   # the relative target is the binding constraint

        target = 1.05 * 0.95^2
        for ci in leaves
            s_new = Float64(fields.s[ci][1])
            Mvv_new = Mvv(1.0, s_new)
            @test Mvv_new ≈ target atol = 1e-12
        end
        @test stats.Mvv_min_post ≈ target atol = 1e-12
        @test stats.total_dE_inj > 0.0
    end

    @testset "fires with β_3 binding (axis 3 dominant)" begin
        mesh = HierarchicalMesh{3}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_3d_fields(mesh)

        # β = (0.10, 0.20, 0.85); max β² = 0.7225 (axis 3); target = 1.05 · 0.7225.
        s0 = log(0.30)
        for ci in leaves
            v = DetField3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                            (1.0, 1.0, 1.0), (0.10, 0.20, 0.85),
                            0.0, 0.0, 0.0, s0)
            write_detfield_3d!(fields, ci, v)
        end

        stats = ProjectionStats3D()
        realizability_project_3d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05,
                                   Mvv_floor = 1e-2,
                                   stats = stats)
        @test stats.n_events == length(leaves)
        target = 1.05 * 0.85^2
        for ci in leaves
            Mvv_new = Mvv(1.0, Float64(fields.s[ci][1]))
            @test Mvv_new ≈ target atol = 1e-12
        end
    end

    @testset "Mvv_floor branch (β all small)" begin
        # When all β are small but Mvv_pre is also small, the absolute
        # floor binds (rather than the relative headroom).
        mesh = HierarchicalMesh{3}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_3d_fields(mesh)

        # β = (0.01, 0.005, 0.001); max β² = 1e-4 ⇒ rel target = 1.05e-4
        # < Mvv_floor = 1e-2. Set Mvv_pre = exp(log(1e-3)) = 1e-3 < 1e-2 ⇒
        # the floor binds.
        s0 = log(1e-3)
        for ci in leaves
            v = DetField3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                            (1.0, 1.0, 1.0), (0.01, 0.005, 0.001),
                            0.0, 0.0, 0.0, s0)
            write_detfield_3d!(fields, ci, v)
        end

        stats = ProjectionStats3D()
        realizability_project_3d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05,
                                   Mvv_floor = 1e-2,
                                   stats = stats)
        @test stats.n_events == length(leaves)
        @test stats.n_floor_events == length(leaves)   # all floor-bound
        for ci in leaves
            Mvv_new = Mvv(1.0, Float64(fields.s[ci][1]))
            @test Mvv_new ≈ 1e-2 atol = 1e-12
        end
    end

    @testset "no-op variant: project_kind = :none" begin
        mesh = HierarchicalMesh{3}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_3d_fields(mesh)

        s0 = log(0.10)
        for ci in leaves
            v = DetField3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                            (1.0, 1.0, 1.0), (0.95, 0.40, 0.30),
                            0.0, 0.0, 0.0, s0)
            write_detfield_3d!(fields, ci, v)
        end
        s_pre = [Float64(fields.s[ci][1]) for ci in leaves]
        stats = ProjectionStats3D()
        realizability_project_3d!(fields, leaves;
                                   project_kind = :none,
                                   stats = stats)
        @test stats.n_events == 0
        @test stats.n_steps == 1
        @test stats.total_dE_inj == 0.0
        for (k, ci) in enumerate(leaves)
            @test fields.s[ci][1] == s_pre[k]
        end
    end

    @testset "2D-symmetric reduction (β_3 = 0) matches 2D target" begin
        # Build parallel 2D and 3D meshes at the same depth; populate
        # with mirror states (same β_1, β_2; β_3 = 0 in 3D); run both
        # projections and assert bit-equal s_post.
        mesh3 = HierarchicalMesh{3}(; balanced = true)
        refine_cells!(mesh3, enumerate_leaves(mesh3))
        leaves3 = enumerate_leaves(mesh3)
        fields3 = allocate_cholesky_3d_fields(mesh3)

        mesh2 = HierarchicalMesh{2}(; balanced = true)
        refine_cells!(mesh2, enumerate_leaves(mesh2))
        leaves2 = enumerate_leaves(mesh2)
        fields2 = allocate_cholesky_2d_fields(mesh2)

        β1, β2 = 0.85, 0.40   # max β² = 0.7225 ⇒ rel target = 1.05 · 0.7225
        s0 = log(0.30)

        for ci in leaves3
            v = DetField3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                            (1.0, 1.0, 1.0), (β1, β2, 0.0),
                            0.0, 0.0, 0.0, s0)
            write_detfield_3d!(fields3, ci, v)
        end
        for ci in leaves2
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (β1, β2),
                            0.0, s0, 0.0, 0.0)
            write_detfield_2d!(fields2, ci, v)
        end

        realizability_project_3d!(fields3, leaves3;
                                   project_kind = :reanchor,
                                   headroom = 1.05,
                                   Mvv_floor = 1e-3)
        realizability_project_2d!(fields2, leaves2;
                                   project_kind = :reanchor,
                                   headroom = 1.05,
                                   Mvv_floor = 1e-3)

        # Both must converge to the same s_post (Mvv_target = 1.05 · 0.85²).
        target = 1.05 * β1 * β1
        for ci in leaves3
            Mvv3 = Mvv(1.0, Float64(fields3.s[ci][1]))
            @test Mvv3 ≈ target atol = 1e-12
        end
        for ci in leaves2
            Mvv2 = Mvv(1.0, Float64(fields2.s[ci][1]))
            @test Mvv2 ≈ target atol = 1e-12
        end

        # Cross-check: every 3D leaf's s value must equal every 2D
        # leaf's s value (they all started from the same s0 with the
        # same β_1, β_2 ⇒ the projection produces a constant field).
        s_3d = [Float64(fields3.s[ci][1]) for ci in leaves3]
        s_2d = [Float64(fields2.s[ci][1]) for ci in leaves2]
        @test all(s_3d .== s_3d[1])  # all identical (deterministic)
        @test all(s_2d .== s_2d[1])
        @test s_3d[1] ≈ s_2d[1] atol = 1e-14   # 3D ⊂ 2D byte-equal at β_3 = 0
    end

    @testset "1D-symmetric reduction (β_2 = β_3 = 0) matches 1D target" begin
        # 3D ⊂ 1D dimension lift at the projection level. With
        # β_2 = β_3 = 0, the per-axis target collapses to headroom · β_1².
        mesh = HierarchicalMesh{3}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_3d_fields(mesh)

        β1 = 0.30
        s0 = log(0.05)
        for ci in leaves
            v = DetField3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                            (1.0, 1.0, 1.0), (β1, 0.0, 0.0),
                            0.0, 0.0, 0.0, s0)
            write_detfield_3d!(fields, ci, v)
        end
        realizability_project_3d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05,
                                   Mvv_floor = 1e-3)
        target_1d = 1.05 * β1^2
        for ci in leaves
            Mvv_post = Mvv(1.0, Float64(fields.s[ci][1]))
            @test Mvv_post ≈ target_1d atol = 1e-12
        end
    end

    @testset "ProjectionStats3D reset!" begin
        stats = ProjectionStats3D()
        stats.n_steps = 5
        stats.n_events = 3
        stats.n_floor_events = 1
        stats.total_dE_inj = 1.5
        stats.Mvv_min_pre = 0.1
        stats.Mvv_min_post = 0.2
        # Need to call by qualified name since `reset!` is non-exported.
        dfmm.reset!(stats)
        @test stats.n_steps == 0
        @test stats.n_events == 0
        @test stats.n_floor_events == 0
        @test stats.total_dE_inj == 0.0
        @test stats.Mvv_min_pre == Inf
        @test stats.Mvv_min_post == Inf
    end

end
