# test_M3_6_phase0_offdiag_dimension_lift.jl
#
# §Dimension-lift gate (M3-6 Phase 0, CRITICAL).
#
# At β_12 = β_21 = 0 (the M3-3c configuration), the new 11-dof residual
# must reduce to the existing 9-dof M3-3c residual byte-equal. This is
# the load-bearing 2D dimension-lift gate of M3-6 Phase 0.
#
# Additionally: verify that across a multi-step driver run (the M3-3c
# Newton driver with the 11-dof residual), `β_12` and `β_21` stay at
# zero to ≤ 1e-12 absolute. The trivial-drive rows
#   F^β_12 = (β_12_np1 − β_12_n)/dt
#   F^β_21 = (β_21_np1 − β_21_n)/dt
# pin them at their IC value of zero for every step.
#
# If this gate fails, the M3-6 Phase 0 residual extension has a bug
# (sign, factor, or off-by-one in the Hamilton-equation derivation).

using Test
using StaticArrays
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, cell_physical_box
using dfmm: DetField2D, allocate_cholesky_2d_fields,
    read_detfield_2d, write_detfield_2d!,
    pack_state_2d_berry, det_step_2d_berry_HG!,
    cholesky_step, cholesky_run

const M3_6_DIMLIFT_TOL = 1.0e-12

@testset "M3-6 Phase 0 §Dimension-lift gate (β_12=β_21=0 ⇒ M3-3c parity)" begin

    function build_mesh_and_leaves(level::Int)
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:level
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        return mesh, enumerate_leaves(mesh)
    end

    bc_spec = FrameBoundaries{2}(((REFLECTING, REFLECTING),
                                   (REFLECTING, REFLECTING)))

    @testset "single step: β_12=β_21=0 IC ⇒ M1 parity (4×4 mesh)" begin
        mesh, leaves = build_mesh_and_leaves(2)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)

        α0_axis1 = 1.0
        α0_axis2 = 1.5
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (α0_axis1, α0_axis2), (0.0, 0.0),
                            (0.0, 0.0),     # explicit zero off-diag β IC
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        dt = 1e-3
        det_step_2d_berry_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                               M_vv_override = (1.0, 0.0), ρ_ref = 1.0)

        # M1 reference (axis 1 active, axis 2 trivial).
        q_n = SVector(α0_axis1, 0.0)
        q_np1 = cholesky_step(q_n, 1.0, 0.0, dt)
        α1_M1, β1_M1 = q_np1[1], q_np1[2]

        max_α1 = 0.0; max_β1 = 0.0
        max_α2 = 0.0; max_β2 = 0.0
        max_β12 = 0.0; max_β21 = 0.0
        max_θR = 0.0
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - α1_M1))
            max_β1 = max(max_β1, abs(v.betas[1]  - β1_M1))
            max_α2 = max(max_α2, abs(v.alphas[2] - α0_axis2))
            max_β2 = max(max_β2, abs(v.betas[2]  - 0.0))
            max_β12 = max(max_β12, abs(v.betas_off[1]))
            max_β21 = max(max_β21, abs(v.betas_off[2]))
            max_θR = max(max_θR, abs(v.θ_R))
        end
        @test max_α1 ≤ M3_6_DIMLIFT_TOL
        @test max_β1 ≤ M3_6_DIMLIFT_TOL
        @test max_α2 ≤ M3_6_DIMLIFT_TOL
        @test max_β2 ≤ M3_6_DIMLIFT_TOL
        # M3-6 Phase 0 critical gate: off-diag β stays at zero.
        @test max_β12 ≤ M3_6_DIMLIFT_TOL
        @test max_β21 ≤ M3_6_DIMLIFT_TOL
        @test max_θR ≤ M3_6_DIMLIFT_TOL
    end

    @testset "100-step run: β_12=β_21 stay at 0 throughout the trajectory" begin
        mesh, leaves = build_mesh_and_leaves(2)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)

        α0_axis1 = 1.0
        α0_axis2 = 1.5
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (α0_axis1, α0_axis2), (0.0, 0.0),
                            (0.0, 0.0),
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        dt = 1e-3
        N_steps = 100
        for _ in 1:N_steps
            det_step_2d_berry_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                                   M_vv_override = (1.0, 0.0), ρ_ref = 1.0)
        end

        q0 = SVector(α0_axis1, 0.0)
        traj = cholesky_run(q0, 1.0, 0.0, dt, N_steps)
        α1_M1_final, β1_M1_final = traj[end][1], traj[end][2]

        max_α1 = 0.0; max_β1 = 0.0
        max_α2 = 0.0; max_β2 = 0.0
        max_β12 = 0.0; max_β21 = 0.0
        max_θR = 0.0
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - α1_M1_final))
            max_β1 = max(max_β1, abs(v.betas[1]  - β1_M1_final))
            max_α2 = max(max_α2, abs(v.alphas[2] - α0_axis2))
            max_β2 = max(max_β2, abs(v.betas[2]  - 0.0))
            max_β12 = max(max_β12, abs(v.betas_off[1]))
            max_β21 = max(max_β21, abs(v.betas_off[2]))
            max_θR = max(max_θR, abs(v.θ_R))
        end
        @test max_α1 ≤ M3_6_DIMLIFT_TOL
        @test max_β1 ≤ M3_6_DIMLIFT_TOL
        @test max_α2 ≤ M3_6_DIMLIFT_TOL
        @test max_β2 ≤ M3_6_DIMLIFT_TOL
        @test max_β12 ≤ M3_6_DIMLIFT_TOL
        @test max_β21 ≤ M3_6_DIMLIFT_TOL
        @test max_θR ≤ M3_6_DIMLIFT_TOL
    end

    @testset "non-trivial active β_1 IC: β_12=β_21 still pinned at 0" begin
        # Stronger gate: even if β_1 is non-zero (genuine M1-style
        # Cholesky-sector dynamics on axis 1), the M3-6 Phase 0
        # off-diag pair stays at zero across the step. The new
        # F^β_a couplings to β̇_12, β̇_21 pick up zero β̇_12, β̇_21
        # from the trivial-drive rows ⇒ no perturbation.
        mesh, leaves = build_mesh_and_leaves(2)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)

        α0_axis1 = 1.2
        α0_axis2 = 0.9
        β0_axis1 = 0.15
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (α0_axis1, α0_axis2), (β0_axis1, 0.0),
                            (0.0, 0.0),
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        dt = 1e-3
        for _ in 1:20
            det_step_2d_berry_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                                   M_vv_override = (1.0, 0.5), ρ_ref = 1.0)
        end

        max_β12 = 0.0; max_β21 = 0.0
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            max_β12 = max(max_β12, abs(v.betas_off[1]))
            max_β21 = max(max_β21, abs(v.betas_off[2]))
        end
        @test max_β12 ≤ M3_6_DIMLIFT_TOL
        @test max_β21 ≤ M3_6_DIMLIFT_TOL
    end

    @testset "8×8 mesh (level 3, 64 leaves): β_12=β_21=0 stays at 0" begin
        mesh, leaves = build_mesh_and_leaves(3)
        @test length(leaves) == 64
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)

        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (1.0, 1.5), (0.0, 0.0),
                            (0.0, 0.0),
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        dt = 1e-3
        for _ in 1:10
            det_step_2d_berry_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                                   M_vv_override = (1.0, 0.0), ρ_ref = 1.0)
        end

        max_β12 = 0.0; max_β21 = 0.0
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            max_β12 = max(max_β12, abs(v.betas_off[1]))
            max_β21 = max(max_β21, abs(v.betas_off[2]))
        end
        @test max_β12 ≤ M3_6_DIMLIFT_TOL
        @test max_β21 ≤ M3_6_DIMLIFT_TOL
    end

    @testset "round-trip read/write preserves off-diag β bit-exactly" begin
        # Write a non-zero off-diag β to fields, read back, verify
        # bit-exact round-trip. This is a structural gate on the field
        # set / DetField2D contract.
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:2
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)

        for (j, ci) in enumerate(leaves)
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            β12 = 0.001 * j
            β21 = -0.0007 * j
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (1.0, 1.0), (0.0, 0.0),
                            (β12, β21),
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        for (j, ci) in enumerate(leaves)
            v = read_detfield_2d(fields, ci)
            @test v.betas_off[1] == 0.001 * j
            @test v.betas_off[2] == -0.0007 * j
        end
    end
end
