# test_M3_3c_dimension_lift_with_berry.jl
#
# M3-3c re-verification of the dimension-lift parity gate (§6.1 of the
# design note `reference/notes_M3_3_2d_cholesky_berry.md`).
#
# Critical M3-3c gate: with Berry coupling integrated into the residual
# AND θ_R promoted to a Newton unknown, the 2D code on a 1D-symmetric
# configuration MUST still reproduce M1's 1D bit-exact results to
# ≤ 1e-12 absolute. The Berry term must vanish identically on the
# 1D-symmetric slice (β_2 = 0, θ_R = 0, α_2 = const) — that's the
# sharp test that the integration is correct.
#
# If this gate fails, the Berry integration has a sign/factor bug.
# Stop and debug before continuing.

using Test
using StaticArrays
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, cell_physical_box
using dfmm: DetField2D, allocate_cholesky_2d_fields,
    read_detfield_2d, write_detfield_2d!,
    pack_state_2d_berry, det_step_2d_berry_HG!,
    cholesky_step, cholesky_run

const M3_3C_DIMLIFT_TOL = 1.0e-12

@testset "M3-3c dimension-lift gate (§6.1 with Berry)" begin

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
                            (α1, α2), (β, β),
                            0.0, s, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        return fields
    end

    bc_spec = FrameBoundaries{2}(((REFLECTING, REFLECTING),
                                   (REFLECTING, REFLECTING)))

    @testset "single step dt=1e-3 on 4×4 mesh: Berry vanishes ⇒ M1 parity" begin
        mesh, leaves = build_mesh_and_leaves(2)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)
        α0_axis1 = 1.0
        α0_axis2 = 1.5
        init_dimension_lifted!(fields, leaves, frame;
                               α1 = α0_axis1, α2 = α0_axis2)

        dt = 1e-3
        det_step_2d_berry_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                               M_vv_override = (1.0, 0.0),
                               ρ_ref = 1.0)

        # M1 reference
        q_n = SVector(α0_axis1, 0.0)
        q_np1 = cholesky_step(q_n, 1.0, 0.0, dt)
        α1_M1, β1_M1 = q_np1[1], q_np1[2]

        max_α1 = 0.0; max_β1 = 0.0
        max_α2 = 0.0; max_β2 = 0.0
        max_θR = 0.0
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - α1_M1))
            max_β1 = max(max_β1, abs(v.betas[1]  - β1_M1))
            max_α2 = max(max_α2, abs(v.alphas[2] - α0_axis2))
            max_β2 = max(max_β2, abs(v.betas[2]  - 0.0))
            max_θR = max(max_θR, abs(v.θ_R))
        end
        @test max_α1 ≤ M3_3C_DIMLIFT_TOL
        @test max_β1 ≤ M3_3C_DIMLIFT_TOL
        @test max_α2 ≤ M3_3C_DIMLIFT_TOL
        @test max_β2 ≤ M3_3C_DIMLIFT_TOL
        # θ_R must stay at 0 — the trivial F^θ_R row pins it.
        @test max_θR ≤ M3_3C_DIMLIFT_TOL
    end

    @testset "100-step run dt=1e-3: Berry ≡ 0 throughout the trajectory" begin
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
            det_step_2d_berry_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                                   M_vv_override = (1.0, 0.0),
                                   ρ_ref = 1.0)
        end

        q0 = SVector(α0_axis1, 0.0)
        traj = cholesky_run(q0, 1.0, 0.0, dt, N_steps)
        α1_M1_final, β1_M1_final = traj[end][1], traj[end][2]

        max_α1 = 0.0; max_β1 = 0.0
        max_α2 = 0.0; max_β2 = 0.0
        max_θR = 0.0
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - α1_M1_final))
            max_β1 = max(max_β1, abs(v.betas[1]  - β1_M1_final))
            max_α2 = max(max_α2, abs(v.alphas[2] - α0_axis2))
            max_β2 = max(max_β2, abs(v.betas[2]  - 0.0))
            max_θR = max(max_θR, abs(v.θ_R))
        end
        @test max_α1 ≤ M3_3C_DIMLIFT_TOL
        @test max_β1 ≤ M3_3C_DIMLIFT_TOL
        @test max_α2 ≤ M3_3C_DIMLIFT_TOL
        @test max_β2 ≤ M3_3C_DIMLIFT_TOL
        @test max_θR ≤ M3_3C_DIMLIFT_TOL
    end

    @testset "8×8 mesh (level 3, 64 leaves): single step + 10-step" begin
        mesh, leaves = build_mesh_and_leaves(3)
        @test length(leaves) == 64
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)
        α0_axis1 = 1.0
        α0_axis2 = 1.5
        init_dimension_lifted!(fields, leaves, frame;
                               α1 = α0_axis1, α2 = α0_axis2)

        dt = 1e-3
        det_step_2d_berry_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                               M_vv_override = (1.0, 0.0),
                               ρ_ref = 1.0)

        q_n = SVector(α0_axis1, 0.0)
        q_np1 = cholesky_step(q_n, 1.0, 0.0, dt)
        α1_M1, β1_M1 = q_np1[1], q_np1[2]
        max_α1 = 0.0; max_β1 = 0.0; max_θR = 0.0
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - α1_M1))
            max_β1 = max(max_β1, abs(v.betas[1]  - β1_M1))
            max_θR = max(max_θR, abs(v.θ_R))
        end
        @test max_α1 ≤ M3_3C_DIMLIFT_TOL
        @test max_β1 ≤ M3_3C_DIMLIFT_TOL
        @test max_θR ≤ M3_3C_DIMLIFT_TOL

        # 10-step extension
        for _ in 1:9
            det_step_2d_berry_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                                   M_vv_override = (1.0, 0.0),
                                   ρ_ref = 1.0)
        end
        traj = cholesky_run(q_n, 1.0, 0.0, dt, 10)
        α1_M1_final, β1_M1_final = traj[end][1], traj[end][2]
        max_α1 = 0.0; max_β1 = 0.0; max_θR = 0.0
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - α1_M1_final))
            max_β1 = max(max_β1, abs(v.betas[1]  - β1_M1_final))
            max_θR = max(max_θR, abs(v.θ_R))
        end
        @test max_α1 ≤ M3_3C_DIMLIFT_TOL
        @test max_β1 ≤ M3_3C_DIMLIFT_TOL
        @test max_θR ≤ M3_3C_DIMLIFT_TOL
    end

    @testset "axis-swap symmetry: active axis = 2" begin
        mesh, leaves = build_mesh_and_leaves(2)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)

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
        det_step_2d_berry_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                               M_vv_override = (0.0, 1.0),
                               ρ_ref = 1.0)

        q_n = SVector(α0_axis2, 0.0)
        q_np1 = cholesky_step(q_n, 1.0, 0.0, dt)
        α2_M1, β2_M1 = q_np1[1], q_np1[2]

        max_α2 = 0.0; max_β2 = 0.0
        max_α1 = 0.0; max_β1 = 0.0
        max_θR = 0.0
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            max_α2 = max(max_α2, abs(v.alphas[2] - α2_M1))
            max_β2 = max(max_β2, abs(v.betas[2]  - β2_M1))
            max_α1 = max(max_α1, abs(v.alphas[1] - α0_axis1))
            max_β1 = max(max_β1, abs(v.betas[1]  - 0.0))
            max_θR = max(max_θR, abs(v.θ_R))
        end
        @test max_α2 ≤ M3_3C_DIMLIFT_TOL
        @test max_β2 ≤ M3_3C_DIMLIFT_TOL
        @test max_α1 ≤ M3_3C_DIMLIFT_TOL
        @test max_β1 ≤ M3_3C_DIMLIFT_TOL
        @test max_θR ≤ M3_3C_DIMLIFT_TOL
    end
end
