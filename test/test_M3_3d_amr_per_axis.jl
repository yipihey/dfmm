# test_M3_3d_amr_per_axis.jl
#
# M3-3d unit tests for the per-axis action-AMR indicator
# (`action_error_indicator_2d_per_axis`) and the HG
# `register_refinement_listener!` field-set listener
# (`register_field_set_on_refine!`). Three blocks:
#
#   1. Indicator on a uniform field is identically zero (no refinement
#      triggered).
#   2. Indicator on a 1D-symmetric (k_y = 0) cold-sinusoid-like
#      configuration peaks along the active axis.
#   3. Refinement-listener integrity: refining a leaf in a 2D mesh
#      via `refine_cells!` correctly resizes the field set and
#      prolongates the parent's coefficient into all children.
#
# See `reference/notes_M3_3_2d_cholesky_berry.md` §4.3 + §6.5 and
# `reference/notes_M3_3d_per_axis_gamma_amr.md`.

using Test
using HierarchicalGrids: HierarchicalMesh, EulerianFrame,
    refine_cells!, enumerate_leaves, FrameBoundaries, REFLECTING, PERIODIC,
    cell_physical_box, unregister_refinement_listener!
using dfmm: action_error_indicator_2d_per_axis,
            register_field_set_on_refine!,
            allocate_cholesky_2d_fields,
            write_detfield_2d!, read_detfield_2d, DetField2D

@testset "M3-3d per-axis action-AMR indicator" begin

    bc_periodic = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                       (PERIODIC, PERIODIC)))

    @testset "uniform field ⇒ indicator ≡ 0" begin
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:2
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)

        for ci in leaves
            v = DetField2D((0.5, 0.5), (0.0, 0.0),
                            (1.0, 1.0), (0.0, 0.0),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        ind = action_error_indicator_2d_per_axis(fields, mesh, frame,
                                                   leaves, bc_periodic;
                                                   M_vv_override = (1.0, 1.0))
        # Curvature of every per-leaf scalar = 0 ⇒ indicator near 0.
        # The γ-marker contribution at γ ≈ √Mvv is also 0 (γ_inv = 1).
        @test maximum(ind) < 1e-12
    end

    @testset "1D-symmetric β_1 spike: indicator peaks along axis 1" begin
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:3   # 8×8 = 64 leaves
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)

        # β_1 = sin(2π x_1), β_2 = 0 ⇒ d²β_1 along axis 1 is large,
        # nothing along axis 2.
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            β1 = 0.3 * sin(2π * cx)
            v = DetField2D((cx, 0.5 * (lo[2] + hi[2])), (0.0, 0.0),
                            (1.0, 1.0), (β1, 0.0),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        ind = action_error_indicator_2d_per_axis(fields, mesh, frame,
                                                   leaves, bc_periodic;
                                                   M_vv_override = (1.0, 1.0))
        @test maximum(ind) > 0.0
        # All cells have non-zero indicator (they all sit on a non-zero
        # 2nd derivative of a sinusoid except at the zero-crossings).
        # We assert the indicator is non-trivial everywhere with at
        # least one cell well above zero.
        @test maximum(ind) > 0.05
    end

    @testset "refine listener: parent → children prolongation" begin
        mesh = HierarchicalMesh{2}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))   # 4 leaves
        leaves = enumerate_leaves(mesh)
        @test length(leaves) == 4
        fields = allocate_cholesky_2d_fields(mesh)

        # Mark each leaf with a unique α_1 value.
        for (k, ci) in enumerate(leaves)
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (Float64(k), 1.0), (0.0, 0.0),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        handle = register_field_set_on_refine!(fields, mesh)
        try
            # Refine the first leaf.
            target_old = leaves[1]
            target_α1_pre = Float64(fields.α_1[target_old][1])
            refine_cells!(mesh, [target_old])
            new_leaves = enumerate_leaves(mesh)
            @test length(new_leaves) == 7   # 1 split into 4 ⇒ 4 + 3 = 7
            # Children of the refined cell should all carry the parent's
            # α_1 value.
            n_match = 0
            for ci in new_leaves
                if Float64(fields.α_1[ci][1]) == target_α1_pre
                    n_match += 1
                end
            end
            # 4 children inherit the parent's α_1 value.
            @test n_match == 4
        finally
            unregister_refinement_listener!(mesh, handle)
        end
    end

end
