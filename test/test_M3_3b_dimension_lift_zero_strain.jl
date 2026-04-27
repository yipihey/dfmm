# test_M3_3b_dimension_lift_zero_strain.jl
#
# M3-3b dimension-lift parity gate (§6.1 of the design note,
# `reference/notes_M3_3_2d_cholesky_berry.md`).
#
# **The single most important M3-3b acceptance criterion.**
#
# Construct a 2D 1D-symmetric configuration that reduces axis-1 to M1's
# Phase-1 zero-strain test (`α_1=1, β_1=0, M_vv_1=1, divu=0`; closed-form
# `α(t) = √(1+t²), β(t) = t/√(1+t²)`) while pinning axis 2 to its trivial
# fixed point (`α_2 = const, β_2 = 0, M_vv_2 = 0`). Step the 2D solver
# alongside M1's `cholesky_step` per cell and assert per-cell agreement
# in the axis-1 sub-residual to ≤ 1e-12 absolute (the §6.1 tolerance).
#
# Coverage:
#
#   1. Single-step gate (dt = 1e-3, 1e-5):
#      Per-cell `(α_1, β_1)` matches `cholesky_step(SVector(α0, β0), 1, 0, dt)`
#      to bit-exact 0.0; `(α_2, β_2)` stays at `(α_2_0, 0)` byte-equally;
#      `(u_1, u_2)` stays at 0; `(x_1, x_2)` stays at the cell center.
#
#   2. 100-step run gate (dt = 1e-3, T = 0.1):
#      Same per-cell agreement after a full M1-Phase-1 trajectory.
#
#   3. 4×4 + 8×8 mesh sizes:
#      Verify the gate holds across two mesh resolutions (level 2 → 16
#      leaves; level 3 → 64 leaves).
#
#   4. REFLECTING BCs — the only kind currently supported by the M3-3b
#      pressure stencil for non-trivial neighbour configurations. The
#      cold-limit fixed point on axis 2 means face pressures are
#      identically zero along axis 2, so the dimension-lift gate is
#      independent of the BC choice on axis 2; we test REFLECTING here
#      and PERIODIC in the basic zero-strain test.
#
# If any of the gate assertions fail, the 2D residual has a per-axis
# coupling bug (likely in either the per-axis Cholesky-sector reduction
# or the per-axis pressure stencil). Stop and debug before continuing.

using Test
using StaticArrays
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, cell_physical_box
using dfmm: DetField2D, allocate_cholesky_2d_fields,
    read_detfield_2d, write_detfield_2d!,
    pack_state_2d, det_step_2d_HG!,
    cholesky_step, cholesky_run

const M3_3B_DIMLIFT_TOL = 1.0e-12

@testset "M3-3b dimension-lift parity gate (§6.1)" begin

    function build_mesh_and_leaves(level::Int)
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:level
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        return mesh, enumerate_leaves(mesh)
    end

    function init_dimension_lifted!(fields, leaves, frame; α1, α2, β=0.0, s=1.0)
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (α1, α2), (β, β),  # β_1 = β_2 = β
                            0.0, s, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        return fields
    end

    bc_spec = FrameBoundaries{2}(((REFLECTING, REFLECTING),
                                   (REFLECTING, REFLECTING)))

    # ─────────────────────────────────────────────────────────────────
    # Block 1: single-step gate at dt = 1e-3 on a 4×4 mesh
    # ─────────────────────────────────────────────────────────────────
    @testset "single step dt=1e-3 on 4×4 mesh" begin
        mesh, leaves = build_mesh_and_leaves(2)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)
        α0_axis1 = 1.0
        α0_axis2 = 1.5
        β0 = 0.0
        init_dimension_lifted!(fields, leaves, frame;
                               α1 = α0_axis1, α2 = α0_axis2, β = β0)

        dt = 1e-3
        det_step_2d_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                         M_vv_override = (1.0, 0.0),
                         ρ_ref = 1.0)

        # M1 reference per-cell
        q_n = SVector(α0_axis1, β0)
        q_np1 = cholesky_step(q_n, 1.0, 0.0, dt)
        α1_M1, β1_M1 = q_np1[1], q_np1[2]

        max_α1 = 0.0; max_β1 = 0.0
        max_α2 = 0.0; max_β2 = 0.0
        max_u  = 0.0; max_x_dev = 0.0
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - α1_M1))
            max_β1 = max(max_β1, abs(v.betas[1]  - β1_M1))
            max_α2 = max(max_α2, abs(v.alphas[2] - α0_axis2))
            max_β2 = max(max_β2, abs(v.betas[2]  - 0.0))
            max_u  = max(max_u, abs(v.u[1]), abs(v.u[2]))
            lo, hi = cell_physical_box(frame, ci)
            cx_expected = 0.5 * (lo[1] + hi[1])
            cy_expected = 0.5 * (lo[2] + hi[2])
            max_x_dev = max(max_x_dev,
                            abs(v.x[1] - cx_expected),
                            abs(v.x[2] - cy_expected))
        end
        # Bit-exact equality on the active axis is the strongest gate.
        # We allow a 1e-12 tolerance per the §6.1 contract; in
        # practice the M3-3b residual achieves 0.0.
        @test max_α1 ≤ M3_3B_DIMLIFT_TOL
        @test max_β1 ≤ M3_3B_DIMLIFT_TOL
        @test max_α2 ≤ M3_3B_DIMLIFT_TOL
        @test max_β2 ≤ M3_3B_DIMLIFT_TOL
        @test max_u  ≤ M3_3B_DIMLIFT_TOL
        @test max_x_dev ≤ M3_3B_DIMLIFT_TOL
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 2: single step at very small dt = 1e-5
    # ─────────────────────────────────────────────────────────────────
    @testset "single step dt=1e-5 (small-step bit-exact)" begin
        mesh, leaves = build_mesh_and_leaves(2)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)
        α0_axis1 = 1.0
        α0_axis2 = 1.5
        init_dimension_lifted!(fields, leaves, frame;
                               α1 = α0_axis1, α2 = α0_axis2)

        dt = 1e-5
        det_step_2d_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                         M_vv_override = (1.0, 0.0),
                         ρ_ref = 1.0)

        q_n = SVector(α0_axis1, 0.0)
        q_np1 = cholesky_step(q_n, 1.0, 0.0, dt)
        α1_M1, β1_M1 = q_np1[1], q_np1[2]

        max_α1 = 0.0; max_β1 = 0.0
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - α1_M1))
            max_β1 = max(max_β1, abs(v.betas[1]  - β1_M1))
        end
        @test max_α1 ≤ M3_3B_DIMLIFT_TOL
        @test max_β1 ≤ M3_3B_DIMLIFT_TOL
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 3: 100-step run on a 4×4 mesh (M1 Phase-1 trajectory)
    # ─────────────────────────────────────────────────────────────────
    @testset "100-step run dt=1e-3 (M1 Phase-1 trajectory match)" begin
        mesh, leaves = build_mesh_and_leaves(2)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)
        α0_axis1 = 1.0
        α0_axis2 = 1.5
        init_dimension_lifted!(fields, leaves, frame;
                               α1 = α0_axis1, α2 = α0_axis2)

        dt = 1e-3
        N_steps = 100
        for _ in 1:N_steps
            det_step_2d_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                             M_vv_override = (1.0, 0.0),
                             ρ_ref = 1.0)
        end

        # M1 reference
        q0 = SVector(α0_axis1, 0.0)
        traj = cholesky_run(q0, 1.0, 0.0, dt, N_steps)
        α1_M1_final, β1_M1_final = traj[end][1], traj[end][2]

        max_α1 = 0.0; max_β1 = 0.0
        max_α2 = 0.0; max_β2 = 0.0
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - α1_M1_final))
            max_β1 = max(max_β1, abs(v.betas[1]  - β1_M1_final))
            max_α2 = max(max_α2, abs(v.alphas[2] - α0_axis2))
            max_β2 = max(max_β2, abs(v.betas[2]  - 0.0))
        end
        @test max_α1 ≤ M3_3B_DIMLIFT_TOL
        @test max_β1 ≤ M3_3B_DIMLIFT_TOL
        @test max_α2 ≤ M3_3B_DIMLIFT_TOL
        @test max_β2 ≤ M3_3B_DIMLIFT_TOL
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 4: 8×8 mesh (level 3, 64 leaves)
    # ─────────────────────────────────────────────────────────────────
    @testset "8×8 mesh (level 3): single step + 10-step run" begin
        mesh, leaves = build_mesh_and_leaves(3)
        @test length(leaves) == 64
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)
        α0_axis1 = 1.0
        α0_axis2 = 1.5
        init_dimension_lifted!(fields, leaves, frame;
                               α1 = α0_axis1, α2 = α0_axis2)

        dt = 1e-3
        det_step_2d_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                         M_vv_override = (1.0, 0.0),
                         ρ_ref = 1.0)

        q_n = SVector(α0_axis1, 0.0)
        q_np1 = cholesky_step(q_n, 1.0, 0.0, dt)
        α1_M1, β1_M1 = q_np1[1], q_np1[2]
        max_α1 = 0.0; max_β1 = 0.0
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - α1_M1))
            max_β1 = max(max_β1, abs(v.betas[1]  - β1_M1))
        end
        @test max_α1 ≤ M3_3B_DIMLIFT_TOL
        @test max_β1 ≤ M3_3B_DIMLIFT_TOL

        # 10-step extension
        for _ in 1:9
            det_step_2d_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                             M_vv_override = (1.0, 0.0),
                             ρ_ref = 1.0)
        end
        traj = cholesky_run(q_n, 1.0, 0.0, dt, 10)
        α1_M1_final, β1_M1_final = traj[end][1], traj[end][2]
        max_α1 = 0.0; max_β1 = 0.0
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - α1_M1_final))
            max_β1 = max(max_β1, abs(v.betas[1]  - β1_M1_final))
        end
        @test max_α1 ≤ M3_3B_DIMLIFT_TOL
        @test max_β1 ≤ M3_3B_DIMLIFT_TOL
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 5: dimension-lift commutes with axis-2 vs axis-1 swap
    # ─────────────────────────────────────────────────────────────────
    # If we configure the active axis as axis 2 instead of axis 1
    # (M_vv_override = (0, 1) and the active α/β on axis 2), the
    # axis-2 sub-residual should reduce to M1 bit-exactly. This tests
    # that the per-axis lift is structurally symmetric in axis index.
    @testset "axis-swap symmetry: active axis = 2" begin
        mesh, leaves = build_mesh_and_leaves(2)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)

        # Active on axis 2: α_2 = 1, β_2 = 0, M_vv_2 = 1.
        # Trivial on axis 1: α_1 = 1.5, β_1 = 0, M_vv_1 = 0.
        α0_axis1 = 1.5  # passive
        α0_axis2 = 1.0  # active
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (α0_axis1, α0_axis2), (0.0, 0.0),
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        dt = 1e-3
        det_step_2d_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                         M_vv_override = (0.0, 1.0),
                         ρ_ref = 1.0)

        q_n = SVector(α0_axis2, 0.0)
        q_np1 = cholesky_step(q_n, 1.0, 0.0, dt)
        α2_M1, β2_M1 = q_np1[1], q_np1[2]

        max_α2 = 0.0; max_β2 = 0.0
        max_α1 = 0.0; max_β1 = 0.0
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            max_α2 = max(max_α2, abs(v.alphas[2] - α2_M1))
            max_β2 = max(max_β2, abs(v.betas[2]  - β2_M1))
            max_α1 = max(max_α1, abs(v.alphas[1] - α0_axis1))
            max_β1 = max(max_β1, abs(v.betas[1]  - 0.0))
        end
        @test max_α2 ≤ M3_3B_DIMLIFT_TOL
        @test max_β2 ≤ M3_3B_DIMLIFT_TOL
        @test max_α1 ≤ M3_3B_DIMLIFT_TOL
        @test max_β1 ≤ M3_3B_DIMLIFT_TOL
    end
end
