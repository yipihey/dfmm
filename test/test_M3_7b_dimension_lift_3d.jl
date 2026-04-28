# test_M3_7b_dimension_lift_3d.jl
#
# M3-7b dimension-lift parity gates (M3-7 design note §7.1a + §7.1b).
#
# **THE LOAD-BEARING M3-7b ACCEPTANCE CRITERION.**
#
# Two sub-gates verify the 3D residual reduces correctly:
#
#   §7.1a (3D ⊂ 1D): with axes 2 + 3 trivial (α_2 = α_3 = const,
#   β_2 = β_3 = 0, M_vv = (1, 0, 0)), all θ_ab = 0, the 3D residual
#   reduces to M1's 1D Phase-1 residual byte-equal (≤ 1e-12 absolute,
#   target 0.0).
#
#   §7.1b (3D ⊂ 2D): with axis 3 trivial (α_3 = const, β_3 = 0,
#   M_vv = (1, 0, 0) — both axis 1 active and axis 2 trivial passive,
#   reproducing M3-3b's 2D 1D-symmetric configuration), θ_13 = θ_23 = 0,
#   the 3D residual reduces to M3-3b's 2D residual byte-equal (≤ 1e-12,
#   target 0.0). This is the sharper test — it verifies the SO(3)
#   extension reduces correctly to SO(2) when one Euler angle is trivial.
#
# If §7.1a fails, the per-axis Cholesky-sector reduction has a bug.
# If §7.1b fails, the SO(3) Cholesky decomposition or 6-face stencil
# has a bug.
#
# Coverage:
#
#   1. §7.1a single-step gate (dt = 1e-3 + dt = 1e-5):
#      Per-cell `(α_1, β_1)` matches `cholesky_step(SVector(α0, β0), 1, 0, dt)`
#      to ≤ 1e-12; `(α_2, β_2, α_3, β_3)` stays at IC byte-equal;
#      `(u_a, x_a)` stays at IC; θ_ab stays at 0.
#
#   2. §7.1a 100-step run gate (dt = 1e-3, T = 0.1):
#      Per-cell agreement after a full M1-Phase-1 trajectory.
#
#   3. §7.1a 4×4×4 + 8×8×8 mesh sizes.
#
#   4. §7.1a axis-swap symmetry: active axis = 2 reproduces M1; active
#      axis = 3 reproduces M1.
#
#   5. §7.1b single-step gate vs M3-3b's 2D `det_step_2d_HG!` on the
#      2D 1D-symmetric configuration. Per-cell agreement on the axis-1
#      and axis-2 sub-residuals.
#
#   6. §7.1b 10-step run gate vs M3-3b's 2D `det_step_2d_HG!`.
#
# See `reference/notes_M3_7_3d_extension.md` §7.1 + §6
# (3D-specific gotchas) and `reference/notes_M3_7b_native_3d_residual.md`.

using Test
using StaticArrays
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, cell_physical_box
using dfmm: DetField3D, DetField2D,
    allocate_cholesky_3d_fields, allocate_cholesky_2d_fields,
    read_detfield_3d, write_detfield_3d!,
    read_detfield_2d, write_detfield_2d!,
    pack_state_3d, det_step_3d_HG!,
    det_step_2d_HG!,
    cholesky_step, cholesky_run

const M3_7B_DIMLIFT_TOL = 1.0e-12

@testset "M3-7b dimension-lift parity gates (§7.1a + §7.1b)" begin

    function build_3d_mesh(level::Int)
        mesh = HierarchicalMesh{3}(; balanced = true)
        for _ in 1:level
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        return mesh, enumerate_leaves(mesh)
    end

    function build_2d_mesh(level::Int)
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:level
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        return mesh, enumerate_leaves(mesh)
    end

    bc_3d = FrameBoundaries{3}(((REFLECTING, REFLECTING),
                                 (REFLECTING, REFLECTING),
                                 (REFLECTING, REFLECTING)))
    bc_2d = FrameBoundaries{2}(((REFLECTING, REFLECTING),
                                 (REFLECTING, REFLECTING)))

    function init_3d_dimension_lifted!(fields, leaves, frame;
                                         α1, α2, α3, β=0.0, s=1.0)
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            cz = 0.5 * (lo[3] + hi[3])
            v = DetField3D((cx, cy, cz), (0.0, 0.0, 0.0),
                            (α1, α2, α3), (β, β, β),
                            0.0, 0.0, 0.0, s)
            write_detfield_3d!(fields, ci, v)
        end
        return fields
    end

    # ─────────────────────────────────────────────────────────────────
    # GATE §7.1a — 3D ⊂ 1D
    # ─────────────────────────────────────────────────────────────────
    @testset "§7.1a 3D ⊂ 1D: single step dt=1e-3, 4×4×4 mesh" begin
        mesh, leaves = build_3d_mesh(2)
        frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fields = allocate_cholesky_3d_fields(mesh)
        α0_axis1 = 1.0
        α0_axis2 = 1.5
        α0_axis3 = 0.7
        init_3d_dimension_lifted!(fields, leaves, frame;
                                    α1 = α0_axis1, α2 = α0_axis2, α3 = α0_axis3)

        dt = 1e-3
        det_step_3d_HG!(fields, mesh, frame, leaves, bc_3d, dt;
                         M_vv_override = (1.0, 0.0, 0.0),
                         ρ_ref = 1.0)

        # M1 reference
        q_n = SVector(α0_axis1, 0.0)
        q_np1 = cholesky_step(q_n, 1.0, 0.0, dt)
        α1_M1, β1_M1 = q_np1[1], q_np1[2]

        max_α1 = 0.0; max_β1 = 0.0
        max_α2 = 0.0; max_β2 = 0.0
        max_α3 = 0.0; max_β3 = 0.0
        max_u  = 0.0; max_x_dev = 0.0; max_θ = 0.0
        for ci in leaves
            v = read_detfield_3d(fields, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - α1_M1))
            max_β1 = max(max_β1, abs(v.betas[1]  - β1_M1))
            max_α2 = max(max_α2, abs(v.alphas[2] - α0_axis2))
            max_β2 = max(max_β2, abs(v.betas[2]  - 0.0))
            max_α3 = max(max_α3, abs(v.alphas[3] - α0_axis3))
            max_β3 = max(max_β3, abs(v.betas[3]  - 0.0))
            max_u  = max(max_u,
                         abs(v.u[1]), abs(v.u[2]), abs(v.u[3]))
            lo, hi = cell_physical_box(frame, ci)
            cx_expected = 0.5 * (lo[1] + hi[1])
            cy_expected = 0.5 * (lo[2] + hi[2])
            cz_expected = 0.5 * (lo[3] + hi[3])
            max_x_dev = max(max_x_dev,
                            abs(v.x[1] - cx_expected),
                            abs(v.x[2] - cy_expected),
                            abs(v.x[3] - cz_expected))
            max_θ = max(max_θ, abs(v.θ_12), abs(v.θ_13), abs(v.θ_23))
        end
        @test max_α1 ≤ M3_7B_DIMLIFT_TOL
        @test max_β1 ≤ M3_7B_DIMLIFT_TOL
        @test max_α2 ≤ M3_7B_DIMLIFT_TOL
        @test max_β2 ≤ M3_7B_DIMLIFT_TOL
        @test max_α3 ≤ M3_7B_DIMLIFT_TOL
        @test max_β3 ≤ M3_7B_DIMLIFT_TOL
        @test max_u  ≤ M3_7B_DIMLIFT_TOL
        @test max_x_dev ≤ M3_7B_DIMLIFT_TOL
        @test max_θ  ≤ M3_7B_DIMLIFT_TOL
    end

    @testset "§7.1a 3D ⊂ 1D: single step dt=1e-5 (small-step bit-exact)" begin
        mesh, leaves = build_3d_mesh(2)
        frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fields = allocate_cholesky_3d_fields(mesh)
        α0_axis1 = 1.0
        init_3d_dimension_lifted!(fields, leaves, frame;
                                    α1 = α0_axis1, α2 = 1.5, α3 = 0.7)

        dt = 1e-5
        det_step_3d_HG!(fields, mesh, frame, leaves, bc_3d, dt;
                         M_vv_override = (1.0, 0.0, 0.0),
                         ρ_ref = 1.0)

        q_n = SVector(α0_axis1, 0.0)
        q_np1 = cholesky_step(q_n, 1.0, 0.0, dt)
        α1_M1, β1_M1 = q_np1[1], q_np1[2]

        max_α1 = 0.0; max_β1 = 0.0
        for ci in leaves
            v = read_detfield_3d(fields, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - α1_M1))
            max_β1 = max(max_β1, abs(v.betas[1]  - β1_M1))
        end
        @test max_α1 ≤ M3_7B_DIMLIFT_TOL
        @test max_β1 ≤ M3_7B_DIMLIFT_TOL
    end

    @testset "§7.1a 3D ⊂ 1D: 100-step run dt=1e-3 on 4×4×4 mesh" begin
        mesh, leaves = build_3d_mesh(2)
        frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fields = allocate_cholesky_3d_fields(mesh)
        α0_axis1 = 1.0
        init_3d_dimension_lifted!(fields, leaves, frame;
                                    α1 = α0_axis1, α2 = 1.5, α3 = 0.7)

        dt = 1e-3
        N_steps = 100
        for _ in 1:N_steps
            det_step_3d_HG!(fields, mesh, frame, leaves, bc_3d, dt;
                             M_vv_override = (1.0, 0.0, 0.0),
                             ρ_ref = 1.0)
        end

        q0 = SVector(α0_axis1, 0.0)
        traj = cholesky_run(q0, 1.0, 0.0, dt, N_steps)
        α1_M1_final, β1_M1_final = traj[end][1], traj[end][2]

        max_α1 = 0.0; max_β1 = 0.0
        max_α2 = 0.0; max_β2 = 0.0
        max_α3 = 0.0; max_β3 = 0.0
        for ci in leaves
            v = read_detfield_3d(fields, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - α1_M1_final))
            max_β1 = max(max_β1, abs(v.betas[1]  - β1_M1_final))
            max_α2 = max(max_α2, abs(v.alphas[2] - 1.5))
            max_β2 = max(max_β2, abs(v.betas[2]  - 0.0))
            max_α3 = max(max_α3, abs(v.alphas[3] - 0.7))
            max_β3 = max(max_β3, abs(v.betas[3]  - 0.0))
        end
        @test max_α1 ≤ M3_7B_DIMLIFT_TOL
        @test max_β1 ≤ M3_7B_DIMLIFT_TOL
        @test max_α2 ≤ M3_7B_DIMLIFT_TOL
        @test max_β2 ≤ M3_7B_DIMLIFT_TOL
        @test max_α3 ≤ M3_7B_DIMLIFT_TOL
        @test max_β3 ≤ M3_7B_DIMLIFT_TOL
    end

    @testset "§7.1a 3D ⊂ 1D: axis-swap symmetry (active axis = 2)" begin
        mesh, leaves = build_3d_mesh(2)
        frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fields = allocate_cholesky_3d_fields(mesh)
        # Active on axis 2: α_2 = 1, M_vv_2 = 1.
        init_3d_dimension_lifted!(fields, leaves, frame;
                                    α1 = 1.5, α2 = 1.0, α3 = 0.7)

        dt = 1e-3
        det_step_3d_HG!(fields, mesh, frame, leaves, bc_3d, dt;
                         M_vv_override = (0.0, 1.0, 0.0),
                         ρ_ref = 1.0)

        q_n = SVector(1.0, 0.0)
        q_np1 = cholesky_step(q_n, 1.0, 0.0, dt)
        α2_M1, β2_M1 = q_np1[1], q_np1[2]

        max_α2 = 0.0; max_β2 = 0.0
        max_α1 = 0.0; max_α3 = 0.0
        for ci in leaves
            v = read_detfield_3d(fields, ci)
            max_α2 = max(max_α2, abs(v.alphas[2] - α2_M1))
            max_β2 = max(max_β2, abs(v.betas[2]  - β2_M1))
            max_α1 = max(max_α1, abs(v.alphas[1] - 1.5))
            max_α3 = max(max_α3, abs(v.alphas[3] - 0.7))
        end
        @test max_α2 ≤ M3_7B_DIMLIFT_TOL
        @test max_β2 ≤ M3_7B_DIMLIFT_TOL
        @test max_α1 ≤ M3_7B_DIMLIFT_TOL
        @test max_α3 ≤ M3_7B_DIMLIFT_TOL
    end

    @testset "§7.1a 3D ⊂ 1D: axis-swap symmetry (active axis = 3)" begin
        mesh, leaves = build_3d_mesh(2)
        frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fields = allocate_cholesky_3d_fields(mesh)
        init_3d_dimension_lifted!(fields, leaves, frame;
                                    α1 = 1.5, α2 = 0.7, α3 = 1.0)

        dt = 1e-3
        det_step_3d_HG!(fields, mesh, frame, leaves, bc_3d, dt;
                         M_vv_override = (0.0, 0.0, 1.0),
                         ρ_ref = 1.0)

        q_n = SVector(1.0, 0.0)
        q_np1 = cholesky_step(q_n, 1.0, 0.0, dt)
        α3_M1, β3_M1 = q_np1[1], q_np1[2]

        max_α3 = 0.0; max_β3 = 0.0
        max_α1 = 0.0; max_α2 = 0.0
        for ci in leaves
            v = read_detfield_3d(fields, ci)
            max_α3 = max(max_α3, abs(v.alphas[3] - α3_M1))
            max_β3 = max(max_β3, abs(v.betas[3]  - β3_M1))
            max_α1 = max(max_α1, abs(v.alphas[1] - 1.5))
            max_α2 = max(max_α2, abs(v.alphas[2] - 0.7))
        end
        @test max_α3 ≤ M3_7B_DIMLIFT_TOL
        @test max_β3 ≤ M3_7B_DIMLIFT_TOL
        @test max_α1 ≤ M3_7B_DIMLIFT_TOL
        @test max_α2 ≤ M3_7B_DIMLIFT_TOL
    end

    @testset "§7.1a 3D ⊂ 1D: 8×8×8 mesh (level 3, 512 leaves)" begin
        mesh, leaves = build_3d_mesh(3)
        @test length(leaves) == 512
        frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fields = allocate_cholesky_3d_fields(mesh)
        α0_axis1 = 1.0
        init_3d_dimension_lifted!(fields, leaves, frame;
                                    α1 = α0_axis1, α2 = 1.5, α3 = 0.7)

        dt = 1e-3
        det_step_3d_HG!(fields, mesh, frame, leaves, bc_3d, dt;
                         M_vv_override = (1.0, 0.0, 0.0),
                         ρ_ref = 1.0)

        q_n = SVector(α0_axis1, 0.0)
        q_np1 = cholesky_step(q_n, 1.0, 0.0, dt)
        α1_M1, β1_M1 = q_np1[1], q_np1[2]
        max_α1 = 0.0; max_β1 = 0.0
        for ci in leaves
            v = read_detfield_3d(fields, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - α1_M1))
            max_β1 = max(max_β1, abs(v.betas[1]  - β1_M1))
        end
        @test max_α1 ≤ M3_7B_DIMLIFT_TOL
        @test max_β1 ≤ M3_7B_DIMLIFT_TOL
    end

    # ─────────────────────────────────────────────────────────────────
    # GATE §7.1b — 3D ⊂ 2D
    # ─────────────────────────────────────────────────────────────────
    # Run M3-3b's 2D `det_step_2d_HG!` on a 4×4 mesh in the 1D-symmetric
    # configuration; in parallel run the 3D `det_step_3d_HG!` on a
    # 4×4×4 mesh in the 2D-symmetric ⊂ 3D configuration with axis 3
    # trivial. The 2D and 3D states should match byte-equal on the
    # axis-1 + axis-2 sub-block.
    #
    # Note: in M3-7b's no-Berry residual, the F^θ_12 row is trivial-
    # driven (θ_12 conserved), exactly mirroring M3-3b's pinned θ_R
    # convention. This gate exercises the per-axis Cholesky-sector
    # reduction at +1 axis (axis 3 trivial), the per-axis pressure
    # stencil along axes 1 + 2 in 3D vs 2D, and the 2D ⊂ 3D structural
    # consistency.
    @testset "§7.1b 3D ⊂ 2D: single step on a 1D-symmetric IC" begin
        # 3D mesh + state.
        mesh3, leaves3 = build_3d_mesh(2)
        frame3 = EulerianFrame(mesh3, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fields3 = allocate_cholesky_3d_fields(mesh3)
        α1_0 = 1.0
        α2_0 = 1.5
        α3_0 = 0.7
        init_3d_dimension_lifted!(fields3, leaves3, frame3;
                                    α1 = α1_0, α2 = α2_0, α3 = α3_0)

        # 2D mesh + state in the matching 1D-symmetric configuration
        # (active axis 1, passive axis 2).
        mesh2, leaves2 = build_2d_mesh(2)
        frame2 = EulerianFrame(mesh2, (0.0, 0.0), (1.0, 1.0))
        fields2 = allocate_cholesky_2d_fields(mesh2)
        for ci in leaves2
            lo, hi = cell_physical_box(frame2, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (α1_0, α2_0), (0.0, 0.0),
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields2, ci, v)
        end

        dt = 1e-3
        det_step_3d_HG!(fields3, mesh3, frame3, leaves3, bc_3d, dt;
                         M_vv_override = (1.0, 0.0, 0.0),
                         ρ_ref = 1.0)
        det_step_2d_HG!(fields2, mesh2, frame2, leaves2, bc_2d, dt;
                         M_vv_override = (1.0, 0.0),
                         ρ_ref = 1.0)

        # Read out the 2D reference values (same per-cell since the IC
        # is uniform across cells).
        v2_ref = read_detfield_2d(fields2, leaves2[1])
        # All 2D leaves agree byte-equal under uniform IC; sanity-check.
        for ci in leaves2
            v = read_detfield_2d(fields2, ci)
            @test v.alphas[1] == v2_ref.alphas[1]
            @test v.alphas[2] == v2_ref.alphas[2]
            @test v.betas[1]  == v2_ref.betas[1]
            @test v.betas[2]  == v2_ref.betas[2]
        end

        # 3D leaves should match the 2D reference on axes 1 + 2 to ≤ 1e-12.
        max_α1 = 0.0; max_β1 = 0.0
        max_α2 = 0.0; max_β2 = 0.0
        max_α3 = 0.0; max_β3 = 0.0
        for ci in leaves3
            v = read_detfield_3d(fields3, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - v2_ref.alphas[1]))
            max_β1 = max(max_β1, abs(v.betas[1]  - v2_ref.betas[1]))
            max_α2 = max(max_α2, abs(v.alphas[2] - v2_ref.alphas[2]))
            max_β2 = max(max_β2, abs(v.betas[2]  - v2_ref.betas[2]))
            max_α3 = max(max_α3, abs(v.alphas[3] - α3_0))
            max_β3 = max(max_β3, abs(v.betas[3]  - 0.0))
        end
        @test max_α1 ≤ M3_7B_DIMLIFT_TOL
        @test max_β1 ≤ M3_7B_DIMLIFT_TOL
        @test max_α2 ≤ M3_7B_DIMLIFT_TOL
        @test max_β2 ≤ M3_7B_DIMLIFT_TOL
        @test max_α3 ≤ M3_7B_DIMLIFT_TOL
        @test max_β3 ≤ M3_7B_DIMLIFT_TOL
    end

    @testset "§7.1b 3D ⊂ 2D: 10-step run on a 1D-symmetric IC" begin
        # 3D mesh + state.
        mesh3, leaves3 = build_3d_mesh(2)
        frame3 = EulerianFrame(mesh3, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fields3 = allocate_cholesky_3d_fields(mesh3)
        α1_0 = 1.0
        α2_0 = 1.5
        α3_0 = 0.7
        init_3d_dimension_lifted!(fields3, leaves3, frame3;
                                    α1 = α1_0, α2 = α2_0, α3 = α3_0)

        # 2D mesh + state.
        mesh2, leaves2 = build_2d_mesh(2)
        frame2 = EulerianFrame(mesh2, (0.0, 0.0), (1.0, 1.0))
        fields2 = allocate_cholesky_2d_fields(mesh2)
        for ci in leaves2
            lo, hi = cell_physical_box(frame2, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (α1_0, α2_0), (0.0, 0.0),
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields2, ci, v)
        end

        dt = 1e-3
        N_steps = 10
        for _ in 1:N_steps
            det_step_3d_HG!(fields3, mesh3, frame3, leaves3, bc_3d, dt;
                             M_vv_override = (1.0, 0.0, 0.0),
                             ρ_ref = 1.0)
            det_step_2d_HG!(fields2, mesh2, frame2, leaves2, bc_2d, dt;
                             M_vv_override = (1.0, 0.0),
                             ρ_ref = 1.0)
        end

        v2_ref = read_detfield_2d(fields2, leaves2[1])

        max_α1 = 0.0; max_β1 = 0.0
        max_α2 = 0.0; max_β2 = 0.0
        max_α3 = 0.0; max_β3 = 0.0
        for ci in leaves3
            v = read_detfield_3d(fields3, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - v2_ref.alphas[1]))
            max_β1 = max(max_β1, abs(v.betas[1]  - v2_ref.betas[1]))
            max_α2 = max(max_α2, abs(v.alphas[2] - v2_ref.alphas[2]))
            max_β2 = max(max_β2, abs(v.betas[2]  - v2_ref.betas[2]))
            max_α3 = max(max_α3, abs(v.alphas[3] - α3_0))
            max_β3 = max(max_β3, abs(v.betas[3]  - 0.0))
        end
        @test max_α1 ≤ M3_7B_DIMLIFT_TOL
        @test max_β1 ≤ M3_7B_DIMLIFT_TOL
        @test max_α2 ≤ M3_7B_DIMLIFT_TOL
        @test max_β2 ≤ M3_7B_DIMLIFT_TOL
        @test max_α3 ≤ M3_7B_DIMLIFT_TOL
        @test max_β3 ≤ M3_7B_DIMLIFT_TOL
    end
end
