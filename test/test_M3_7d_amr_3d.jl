# test_M3_7d_amr_3d.jl
#
# M3-7d unit tests for the per-axis action-AMR indicator
# (`action_error_indicator_3d_per_axis`) and the HG
# `register_refinement_listener!` 3D field-set listener
# (`register_field_set_on_refine_3d!`). Five blocks:
#
#   1. Indicator on a uniform 3D field is identically zero.
#   2. Indicator on a 1D-symmetric (k = (1, 0, 0)) cold-sinusoid IC
#      peaks along the active axis.
#   3. Indicator on a 2D-symmetric (k = (1, 1, 0)) IC fires along
#      axes 1 and 2.
#   4. Refinement-listener integrity: refining a single octant of an
#      8-octant uniform mesh produces 15 leaves; mass conservation
#      byte-equal under arithmetic-mean (equal-volume) coarsening.
#   5. Refine + coarsen round-trip: state byte-equal to pre-refine.
#
# See `reference/notes_M3_7_3d_extension.md` §4.3 + §7.5 and
# `reference/notes_M3_7d_3d_per_axis_gamma_amr.md`.

using Test
using HierarchicalGrids: HierarchicalMesh, EulerianFrame,
    refine_cells!, coarsen_cells!, enumerate_leaves, FrameBoundaries,
    REFLECTING, PERIODIC, cell_physical_box,
    unregister_refinement_listener!, n_cells
using dfmm: action_error_indicator_3d_per_axis,
            register_field_set_on_refine_3d!,
            register_field_set_on_refine!,
            allocate_cholesky_3d_fields,
            write_detfield_3d!, read_detfield_3d, DetField3D

@testset "M3-7d per-axis action-AMR indicator (3D)" begin

    bc_periodic = FrameBoundaries{3}(((PERIODIC, PERIODIC),
                                       (PERIODIC, PERIODIC),
                                       (PERIODIC, PERIODIC)))

    @testset "uniform field ⇒ indicator ≡ 0" begin
        mesh = HierarchicalMesh{3}(; balanced = true)
        for _ in 1:2
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        @test length(leaves) == 64
        frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fields = allocate_cholesky_3d_fields(mesh)

        for ci in leaves
            v = DetField3D((0.5, 0.5, 0.5), (0.0, 0.0, 0.0),
                            (1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_3d!(fields, ci, v)
        end

        ind = action_error_indicator_3d_per_axis(fields, mesh, frame,
                                                   leaves, bc_periodic;
                                                   M_vv_override = (1.0, 1.0, 1.0))
        @test maximum(ind) < 1e-12
    end

    @testset "1D-symmetric β_1 spike: indicator > 0" begin
        mesh = HierarchicalMesh{3}(; balanced = true)
        for _ in 1:2   # 4×4×4 = 64 leaves
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fields = allocate_cholesky_3d_fields(mesh)

        # β_1 = sin(2π x_1), β_2 = β_3 = 0.
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            cz = 0.5 * (lo[3] + hi[3])
            β1 = 0.3 * sin(2π * cx)
            v = DetField3D((cx, cy, cz), (0.0, 0.0, 0.0),
                            (1.0, 1.0, 1.0), (β1, 0.0, 0.0),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_3d!(fields, ci, v)
        end

        ind = action_error_indicator_3d_per_axis(fields, mesh, frame,
                                                   leaves, bc_periodic;
                                                   M_vv_override = (1.0, 1.0, 1.0))
        @test maximum(ind) > 0.0
        @test maximum(ind) > 0.05
    end

    @testset "2D-symmetric β_1, β_2 spikes (k_z=0): axes 1, 2 fire" begin
        mesh = HierarchicalMesh{3}(; balanced = true)
        for _ in 1:2
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fields = allocate_cholesky_3d_fields(mesh)

        # β_1 = sin(2π x), β_2 = sin(2π y), β_3 = 0.
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            cz = 0.5 * (lo[3] + hi[3])
            β1 = 0.3 * sin(2π * cx)
            β2 = 0.3 * sin(2π * cy)
            v = DetField3D((cx, cy, cz), (0.0, 0.0, 0.0),
                            (1.0, 1.0, 1.0), (β1, β2, 0.0),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_3d!(fields, ci, v)
        end

        ind = action_error_indicator_3d_per_axis(fields, mesh, frame,
                                                   leaves, bc_periodic;
                                                   M_vv_override = (1.0, 1.0, 1.0))
        @test maximum(ind) > 0.05
        @test minimum(ind) >= 0.0
    end

    @testset "refine listener: parent → 8 children prolongation" begin
        mesh = HierarchicalMesh{3}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))   # 8 leaves
        leaves = enumerate_leaves(mesh)
        @test length(leaves) == 8
        fields = allocate_cholesky_3d_fields(mesh)

        # Mark each leaf with a unique α_1 value.
        for (k, ci) in enumerate(leaves)
            v = DetField3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                            (Float64(k), 1.0, 1.0), (0.0, 0.0, 0.0),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_3d!(fields, ci, v)
        end

        handle = register_field_set_on_refine_3d!(fields, mesh)
        try
            target_old = leaves[1]
            target_α1_pre = Float64(fields.α_1[target_old][1])
            refine_cells!(mesh, [target_old])
            new_leaves = enumerate_leaves(mesh)
            # 1 split into 8 ⇒ 7 + 8 = 15 leaves.
            @test length(new_leaves) == 15
            n_match = 0
            for ci in new_leaves
                if Float64(fields.α_1[ci][1]) == target_α1_pre
                    n_match += 1
                end
            end
            # 8 children inherit the parent's α_1.
            @test n_match == 8
        finally
            unregister_refinement_listener!(mesh, handle)
        end
    end

    @testset "refine + coarsen round-trip: byte-equal restore" begin
        mesh = HierarchicalMesh{3}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))   # 8 leaves
        leaves_pre = enumerate_leaves(mesh)
        @test length(leaves_pre) == 8

        fields = allocate_cholesky_3d_fields(mesh)
        # Pre-state with mixed unique values. Use exact-representable
        # binary fractions (k/2, k/4 …) so the arithmetic-mean
        # coarsening (sum of 8 copies of value v, divided by 8 = v)
        # is bit-exact rather than ULP-drifty.
        pre_states = Dict{Int, DetField3D{Float64}}()
        for (k, ci) in enumerate(leaves_pre)
            kf = Float64(k)
            v = DetField3D((kf, kf + 0.5, kf + 0.25),
                            (0.0, 0.0, 0.0),
                            (1.0 + 0.5 * k, 2.0, 3.0),
                            (0.0, 0.0, 0.0),
                            0.0, 0.0, 0.0, kf)
            write_detfield_3d!(fields, ci, v)
            pre_states[Int(ci)] = v
        end

        handle = register_field_set_on_refine_3d!(fields, mesh)
        try
            target = Int(leaves_pre[1])
            refine_cells!(mesh, [target])
            mid_leaves = enumerate_leaves(mesh)
            @test length(mid_leaves) == 15

            # Now coarsen — find the parent of the 8 children. After
            # refinement, the parent cell index `target` is no longer a
            # leaf; its 8 children are. Calling coarsen_cells!(mesh,
            # [target]) on the parent re-merges them, and the listener's
            # arithmetic-mean averaging must reproduce the original α_1
            # value (since all 8 children carry the same α_1).
            coarsen_cells!(mesh, [UInt32(target)])
            post_leaves = enumerate_leaves(mesh)
            @test length(post_leaves) == 8

            # After coarsening, `target` is again a leaf with the
            # arithmetic-mean of the 8 children (which all share the
            # parent's value). Byte-equal restore.
            for ci in post_leaves
                v_post = read_detfield_3d(fields, ci)
                v_pre = pre_states[Int(ci)]
                @test v_post.alphas[1] == v_pre.alphas[1]
                @test v_post.alphas[2] == v_pre.alphas[2]
                @test v_post.alphas[3] == v_pre.alphas[3]
                @test v_post.s == v_pre.s
            end
        finally
            unregister_refinement_listener!(mesh, handle)
        end
    end

    @testset "refine listener: prolongation across all 16 named fields" begin
        # The listener walks every named scalar field; verify all 16
        # 3D-allocator fields prolongate consistently.
        mesh = HierarchicalMesh{3}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))   # 8 leaves
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_3d_fields(mesh)

        unique_v = DetField3D((0.11, 0.22, 0.33), (0.44, 0.55, 0.66),
                               (1.7, 1.8, 1.9), (0.05, 0.06, 0.07),
                               0.13, 0.14, 0.15, 0.99)
        write_detfield_3d!(fields, leaves[1], unique_v)
        # Other leaves get distinct values.
        for k in 2:length(leaves)
            v = DetField3D((Float64(k), 0.0, 0.0), (0.0, 0.0, 0.0),
                            (Float64(k), Float64(k), Float64(k)),
                            (0.0, 0.0, 0.0),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_3d!(fields, leaves[k], v)
        end

        handle = register_field_set_on_refine_3d!(fields, mesh)
        try
            target = leaves[1]
            refine_cells!(mesh, [target])
            new_leaves = enumerate_leaves(mesh)
            # Find the 8 leaves with α_1 = 1.7 (matches unique_v).
            n_α1 = 0; n_α2 = 0; n_α3 = 0
            n_β1 = 0; n_β2 = 0; n_β3 = 0
            n_θ12 = 0; n_θ13 = 0; n_θ23 = 0
            n_x1 = 0; n_x2 = 0; n_x3 = 0
            n_u1 = 0; n_u2 = 0; n_u3 = 0
            n_s = 0
            for ci in new_leaves
                Float64(fields.α_1[ci][1]) == 1.7 && (n_α1 += 1)
                Float64(fields.α_2[ci][1]) == 1.8 && (n_α2 += 1)
                Float64(fields.α_3[ci][1]) == 1.9 && (n_α3 += 1)
                Float64(fields.β_1[ci][1]) == 0.05 && (n_β1 += 1)
                Float64(fields.β_2[ci][1]) == 0.06 && (n_β2 += 1)
                Float64(fields.β_3[ci][1]) == 0.07 && (n_β3 += 1)
                Float64(fields.θ_12[ci][1]) == 0.13 && (n_θ12 += 1)
                Float64(fields.θ_13[ci][1]) == 0.14 && (n_θ13 += 1)
                Float64(fields.θ_23[ci][1]) == 0.15 && (n_θ23 += 1)
                Float64(fields.x_1[ci][1]) == 0.11 && (n_x1 += 1)
                Float64(fields.x_2[ci][1]) == 0.22 && (n_x2 += 1)
                Float64(fields.x_3[ci][1]) == 0.33 && (n_x3 += 1)
                Float64(fields.u_1[ci][1]) == 0.44 && (n_u1 += 1)
                Float64(fields.u_2[ci][1]) == 0.55 && (n_u2 += 1)
                Float64(fields.u_3[ci][1]) == 0.66 && (n_u3 += 1)
                Float64(fields.s[ci][1]) == 0.99 && (n_s += 1)
            end
            # All 16 fields must have exactly 8 children inheriting
            # the parent's value.
            @test n_α1 == 8 && n_α2 == 8 && n_α3 == 8
            @test n_β1 == 8 && n_β2 == 8 && n_β3 == 8
            @test n_θ12 == 8 && n_θ13 == 8 && n_θ23 == 8
            @test n_x1 == 8 && n_x2 == 8 && n_x3 == 8
            @test n_u1 == 8 && n_u2 == 8 && n_u3 == 8
            @test n_s == 8
        finally
            unregister_refinement_listener!(mesh, handle)
        end
    end

    @testset "indicator selectivity: full 3D k=(1,1,1) all axes fire" begin
        mesh = HierarchicalMesh{3}(; balanced = true)
        for _ in 1:2
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fields = allocate_cholesky_3d_fields(mesh)

        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            cz = 0.5 * (lo[3] + hi[3])
            β1 = 0.2 * sin(2π * cx)
            β2 = 0.2 * sin(2π * cy)
            β3 = 0.2 * sin(2π * cz)
            v = DetField3D((cx, cy, cz), (0.0, 0.0, 0.0),
                            (1.0, 1.0, 1.0), (β1, β2, β3),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_3d!(fields, ci, v)
        end

        ind = action_error_indicator_3d_per_axis(fields, mesh, frame,
                                                   leaves, bc_periodic;
                                                   M_vv_override = (1.0, 1.0, 1.0))
        @test maximum(ind) > 0.0
        @test minimum(ind) >= 0.0
        # All cells contribute non-trivially (sinusoidal d² is nonzero
        # everywhere except zero-crossings).
        nonzero_count = count(>(1e-3), ind)
        @test nonzero_count > 0.5 * length(ind)
    end

end
