# test_M3_7c_dimension_lift_3d_with_berry.jl
#
# §7.1a + §7.1b dimension-lift parity gates (M3-7c, with SO(3) Berry).
#
# **THE LOAD-BEARING M3-7c ACCEPTANCE CRITERION.**
#
# This is the 3D Berry-aware version of M3-7b's dimension-lift gate. With
# Berry coupling now in the residual + (θ_12, θ_13, θ_23) as Newton
# unknowns, the test is sharper than M3-7b's:
#
#   §7.1a 3D ⊂ 1D: at θ_12 = θ_13 = θ_23 = 0 + 1D-symmetric base flow
#   (axes 2, 3 trivial: β_2 = β_3 = 0, M_vv = (1, 0, 0)), the 3D
#   Berry-aware residual must reduce to M1's 1D Phase-1 residual byte-
#   equal (≤ 1e-12, target 0.0). Structural guarantee: every Berry α-
#   modification term has factor `θ̇_{ab}`, and every Berry β-modification
#   term has factor `β̄_b`; on the 1D-symmetric slice both are zero, so
#   Berry vanishes multiplicatively.
#
#   §7.1b 3D ⊂ 2D: at θ_13 = θ_23 = 0 + 2D-symmetric base flow (axis 3
#   trivial: β_3 = 0, α_3 = const), the 3D Berry-aware residual must
#   reduce to **M3-3c's 2D Berry residual** byte-equal (≤ 1e-12,
#   target 0.0). Structural guarantee: pair-(1,3) and pair-(2,3)
#   contributions vanish via β_3 = 0 / θ̇_13 = θ̇_23 = 0 (their rows
#   pin them); pair-(1,2) Berry block matches M3-3c's 2D form via
#   CHECK 3b of `notes_M3_prep_3D_berry_verification.md`.
#
# **If §7.1b fails byte-equal, the SO(3) Berry coupling has a sign /
# factor bug** — the 3D Berry must reduce to SO(2) when one Euler angle
# is trivial.
#
# Coverage:
#
#   1. §7.1a: single-step + 100-step 3D ⊂ 1D parity vs M1 `cholesky_step`
#   2. §7.1a: axis-swap symmetry across active axis = 1, 2, 3
#   3. §7.1b: single-step + 10-step 3D ⊂ 2D parity vs M3-3c
#      `det_step_2d_berry_HG!` on a 1D-symmetric 2D IC
#   4. §7.1b: axis-3-trivial 2D-symmetric IC with non-zero θ_12 — Berry
#      coupling active in both 2D and 3D paths; 3D path reduces to 2D
#      byte-equal

using Test
using StaticArrays
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, cell_physical_box
using dfmm: DetField3D, DetField2D,
    allocate_cholesky_3d_fields, allocate_cholesky_2d_fields,
    read_detfield_3d, write_detfield_3d!,
    read_detfield_2d, write_detfield_2d!,
    pack_state_3d, det_step_3d_berry_HG!,
    det_step_2d_berry_HG!,
    cholesky_step, cholesky_run

const M3_7C_DIMLIFT_TOL = 1.0e-12

@testset "M3-7c dimension-lift parity gates (§7.1a + §7.1b WITH Berry)" begin

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
                                         α1, α2, α3, β=0.0, s=1.0,
                                         θ12=0.0, θ13=0.0, θ23=0.0)
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            cz = 0.5 * (lo[3] + hi[3])
            v = DetField3D((cx, cy, cz), (0.0, 0.0, 0.0),
                            (α1, α2, α3), (β, β, β),
                            θ12, θ13, θ23, s)
            write_detfield_3d!(fields, ci, v)
        end
        return fields
    end

    # ─────────────────────────────────────────────────────────────────
    # GATE §7.1a — 3D ⊂ 1D (Berry vanishes structurally on 1D-symmetric)
    # ─────────────────────────────────────────────────────────────────
    @testset "§7.1a 3D-Berry ⊂ 1D: single step (4×4×4 mesh)" begin
        mesh, leaves = build_3d_mesh(2)
        frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fields = allocate_cholesky_3d_fields(mesh)
        α0_axis1 = 1.0
        α0_axis2 = 1.5
        α0_axis3 = 0.7
        init_3d_dimension_lifted!(fields, leaves, frame;
                                    α1 = α0_axis1, α2 = α0_axis2, α3 = α0_axis3)

        dt = 1e-3
        det_step_3d_berry_HG!(fields, mesh, frame, leaves, bc_3d, dt;
                               M_vv_override = (1.0, 0.0, 0.0),
                               ρ_ref = 1.0)

        # M1 reference for axis 1.
        q_n = SVector(α0_axis1, 0.0)
        q_np1 = cholesky_step(q_n, 1.0, 0.0, dt)
        α1_M1, β1_M1 = q_np1[1], q_np1[2]

        max_α1 = 0.0; max_β1 = 0.0
        max_α2 = 0.0; max_β2 = 0.0
        max_α3 = 0.0; max_β3 = 0.0
        max_θ12 = 0.0; max_θ13 = 0.0; max_θ23 = 0.0
        for ci in leaves
            v = read_detfield_3d(fields, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - α1_M1))
            max_β1 = max(max_β1, abs(v.betas[1]  - β1_M1))
            max_α2 = max(max_α2, abs(v.alphas[2] - α0_axis2))
            max_β2 = max(max_β2, abs(v.betas[2]  - 0.0))
            max_α3 = max(max_α3, abs(v.alphas[3] - α0_axis3))
            max_β3 = max(max_β3, abs(v.betas[3]  - 0.0))
            max_θ12 = max(max_θ12, abs(v.θ_12))
            max_θ13 = max(max_θ13, abs(v.θ_13))
            max_θ23 = max(max_θ23, abs(v.θ_23))
        end
        @test max_α1 ≤ M3_7C_DIMLIFT_TOL
        @test max_β1 ≤ M3_7C_DIMLIFT_TOL
        @test max_α2 ≤ M3_7C_DIMLIFT_TOL
        @test max_β2 ≤ M3_7C_DIMLIFT_TOL
        @test max_α3 ≤ M3_7C_DIMLIFT_TOL
        @test max_β3 ≤ M3_7C_DIMLIFT_TOL
        @test max_θ12 ≤ M3_7C_DIMLIFT_TOL
        @test max_θ13 ≤ M3_7C_DIMLIFT_TOL
        @test max_θ23 ≤ M3_7C_DIMLIFT_TOL
    end

    @testset "§7.1a 3D-Berry ⊂ 1D: 100-step run dt=1e-3" begin
        mesh, leaves = build_3d_mesh(2)
        frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fields = allocate_cholesky_3d_fields(mesh)
        α0_axis1 = 1.0
        init_3d_dimension_lifted!(fields, leaves, frame;
                                    α1 = α0_axis1, α2 = 1.5, α3 = 0.7)

        dt = 1e-3
        N_steps = 100
        for _ in 1:N_steps
            det_step_3d_berry_HG!(fields, mesh, frame, leaves, bc_3d, dt;
                                   M_vv_override = (1.0, 0.0, 0.0),
                                   ρ_ref = 1.0)
        end

        q0 = SVector(α0_axis1, 0.0)
        traj = cholesky_run(q0, 1.0, 0.0, dt, N_steps)
        α1_M1_final, β1_M1_final = traj[end][1], traj[end][2]

        max_α1 = 0.0; max_β1 = 0.0
        max_α2 = 0.0; max_β2 = 0.0
        max_α3 = 0.0; max_β3 = 0.0
        max_θ = 0.0
        for ci in leaves
            v = read_detfield_3d(fields, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - α1_M1_final))
            max_β1 = max(max_β1, abs(v.betas[1]  - β1_M1_final))
            max_α2 = max(max_α2, abs(v.alphas[2] - 1.5))
            max_β2 = max(max_β2, abs(v.betas[2]  - 0.0))
            max_α3 = max(max_α3, abs(v.alphas[3] - 0.7))
            max_β3 = max(max_β3, abs(v.betas[3]  - 0.0))
            max_θ = max(max_θ, abs(v.θ_12), abs(v.θ_13), abs(v.θ_23))
        end
        @test max_α1 ≤ M3_7C_DIMLIFT_TOL
        @test max_β1 ≤ M3_7C_DIMLIFT_TOL
        @test max_α2 ≤ M3_7C_DIMLIFT_TOL
        @test max_β2 ≤ M3_7C_DIMLIFT_TOL
        @test max_α3 ≤ M3_7C_DIMLIFT_TOL
        @test max_β3 ≤ M3_7C_DIMLIFT_TOL
        @test max_θ  ≤ M3_7C_DIMLIFT_TOL
    end

    @testset "§7.1a 3D-Berry ⊂ 1D: axis-swap symmetry (active axis = 2)" begin
        mesh, leaves = build_3d_mesh(2)
        frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fields = allocate_cholesky_3d_fields(mesh)
        init_3d_dimension_lifted!(fields, leaves, frame;
                                    α1 = 1.5, α2 = 1.0, α3 = 0.7)

        dt = 1e-3
        det_step_3d_berry_HG!(fields, mesh, frame, leaves, bc_3d, dt;
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
        @test max_α2 ≤ M3_7C_DIMLIFT_TOL
        @test max_β2 ≤ M3_7C_DIMLIFT_TOL
        @test max_α1 ≤ M3_7C_DIMLIFT_TOL
        @test max_α3 ≤ M3_7C_DIMLIFT_TOL
    end

    @testset "§7.1a 3D-Berry ⊂ 1D: axis-swap symmetry (active axis = 3)" begin
        mesh, leaves = build_3d_mesh(2)
        frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fields = allocate_cholesky_3d_fields(mesh)
        init_3d_dimension_lifted!(fields, leaves, frame;
                                    α1 = 1.5, α2 = 0.7, α3 = 1.0)

        dt = 1e-3
        det_step_3d_berry_HG!(fields, mesh, frame, leaves, bc_3d, dt;
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
        @test max_α3 ≤ M3_7C_DIMLIFT_TOL
        @test max_β3 ≤ M3_7C_DIMLIFT_TOL
        @test max_α1 ≤ M3_7C_DIMLIFT_TOL
        @test max_α2 ≤ M3_7C_DIMLIFT_TOL
    end

    # ─────────────────────────────────────────────────────────────────
    # GATE §7.1b — 3D ⊂ 2D (the sharper test: SO(3) reduces to SO(2))
    # ─────────────────────────────────────────────────────────────────
    # Run M3-3c's 2D `det_step_2d_berry_HG!` on a 4×4 mesh in the
    # 1D-symmetric configuration; in parallel run the 3D
    # `det_step_3d_berry_HG!` on a 4×4×4 mesh in the 2D-symmetric ⊂ 3D
    # configuration with axis 3 trivial. The 2D and 3D states should
    # match byte-equal on the axis-1 + axis-2 + θ_12 sub-block.
    @testset "§7.1b 3D-Berry ⊂ 2D-Berry: single step on 1D-symmetric IC" begin
        # 3D mesh + state.
        mesh3, leaves3 = build_3d_mesh(2)
        frame3 = EulerianFrame(mesh3, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fields3 = allocate_cholesky_3d_fields(mesh3)
        α1_0 = 1.0
        α2_0 = 1.5
        α3_0 = 0.7
        init_3d_dimension_lifted!(fields3, leaves3, frame3;
                                    α1 = α1_0, α2 = α2_0, α3 = α3_0)

        # 2D mesh + state in the matching 1D-symmetric configuration.
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
        det_step_3d_berry_HG!(fields3, mesh3, frame3, leaves3, bc_3d, dt;
                               M_vv_override = (1.0, 0.0, 0.0),
                               ρ_ref = 1.0)
        det_step_2d_berry_HG!(fields2, mesh2, frame2, leaves2, bc_2d, dt;
                               M_vv_override = (1.0, 0.0),
                               ρ_ref = 1.0)

        v2_ref = read_detfield_2d(fields2, leaves2[1])
        # Sanity: 2D leaves agree byte-equal under uniform IC.
        for ci in leaves2
            v = read_detfield_2d(fields2, ci)
            @test v.alphas[1] == v2_ref.alphas[1]
            @test v.alphas[2] == v2_ref.alphas[2]
            @test v.betas[1]  == v2_ref.betas[1]
            @test v.betas[2]  == v2_ref.betas[2]
        end

        # 3D leaves should match the 2D Berry reference on axes 1 + 2.
        max_α1 = 0.0; max_β1 = 0.0
        max_α2 = 0.0; max_β2 = 0.0
        max_α3 = 0.0; max_β3 = 0.0
        max_θ12 = 0.0; max_θ13 = 0.0; max_θ23 = 0.0
        for ci in leaves3
            v = read_detfield_3d(fields3, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - v2_ref.alphas[1]))
            max_β1 = max(max_β1, abs(v.betas[1]  - v2_ref.betas[1]))
            max_α2 = max(max_α2, abs(v.alphas[2] - v2_ref.alphas[2]))
            max_β2 = max(max_β2, abs(v.betas[2]  - v2_ref.betas[2]))
            max_α3 = max(max_α3, abs(v.alphas[3] - α3_0))
            max_β3 = max(max_β3, abs(v.betas[3]  - 0.0))
            # θ_12 should match the 2D θ_R (both are pinned at 0 on the
            # 1D-symmetric IC); θ_13, θ_23 should stay at 0.
            max_θ12 = max(max_θ12, abs(v.θ_12 - v2_ref.θ_R))
            max_θ13 = max(max_θ13, abs(v.θ_13))
            max_θ23 = max(max_θ23, abs(v.θ_23))
        end
        @test max_α1 ≤ M3_7C_DIMLIFT_TOL
        @test max_β1 ≤ M3_7C_DIMLIFT_TOL
        @test max_α2 ≤ M3_7C_DIMLIFT_TOL
        @test max_β2 ≤ M3_7C_DIMLIFT_TOL
        @test max_α3 ≤ M3_7C_DIMLIFT_TOL
        @test max_β3 ≤ M3_7C_DIMLIFT_TOL
        @test max_θ12 ≤ M3_7C_DIMLIFT_TOL
        @test max_θ13 ≤ M3_7C_DIMLIFT_TOL
        @test max_θ23 ≤ M3_7C_DIMLIFT_TOL
    end

    @testset "§7.1b 3D-Berry ⊂ 2D-Berry: 10-step run on 1D-symmetric IC" begin
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
            det_step_3d_berry_HG!(fields3, mesh3, frame3, leaves3, bc_3d, dt;
                                   M_vv_override = (1.0, 0.0, 0.0),
                                   ρ_ref = 1.0)
            det_step_2d_berry_HG!(fields2, mesh2, frame2, leaves2, bc_2d, dt;
                                   M_vv_override = (1.0, 0.0),
                                   ρ_ref = 1.0)
        end

        v2_ref = read_detfield_2d(fields2, leaves2[1])

        max_α1 = 0.0; max_β1 = 0.0
        max_α2 = 0.0; max_β2 = 0.0
        max_α3 = 0.0; max_β3 = 0.0
        max_θ = 0.0
        for ci in leaves3
            v = read_detfield_3d(fields3, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - v2_ref.alphas[1]))
            max_β1 = max(max_β1, abs(v.betas[1]  - v2_ref.betas[1]))
            max_α2 = max(max_α2, abs(v.alphas[2] - v2_ref.alphas[2]))
            max_β2 = max(max_β2, abs(v.betas[2]  - v2_ref.betas[2]))
            max_α3 = max(max_α3, abs(v.alphas[3] - α3_0))
            max_β3 = max(max_β3, abs(v.betas[3]  - 0.0))
            max_θ = max(max_θ, abs(v.θ_12), abs(v.θ_13), abs(v.θ_23))
        end
        @test max_α1 ≤ M3_7C_DIMLIFT_TOL
        @test max_β1 ≤ M3_7C_DIMLIFT_TOL
        @test max_α2 ≤ M3_7C_DIMLIFT_TOL
        @test max_β2 ≤ M3_7C_DIMLIFT_TOL
        @test max_α3 ≤ M3_7C_DIMLIFT_TOL
        @test max_β3 ≤ M3_7C_DIMLIFT_TOL
        @test max_θ  ≤ M3_7C_DIMLIFT_TOL
    end

    # ─────────────────────────────────────────────────────────────────
    # SO(3) ⊃ SO(2): 2D-symmetric IC with non-zero β + non-zero θ_12
    # ─────────────────────────────────────────────────────────────────
    # The sharpest 3D ⊂ 2D test: a non-trivial 2D-Berry IC (β_1 ≠ 0,
    # β_2 ≠ 0, θ_R ≠ 0) lifted to 3D with axis 3 trivial. Both paths
    # exercise non-trivial Berry α/β-modifications via pair (1, 2);
    # the 3D path's pair-(1, 3) and pair-(2, 3) blocks vanish via
    # β_3 = 0 / θ̇_13 = θ̇_23 = 0. Byte-equal parity proves the SO(3) →
    # SO(2) reduction is correct.
    @testset "§7.1b 3D-Berry ⊂ 2D-Berry: non-trivial Berry IC (β + θ_12)" begin
        mesh3, leaves3 = build_3d_mesh(2)
        frame3 = EulerianFrame(mesh3, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fields3 = allocate_cholesky_3d_fields(mesh3)
        α1_0 = 1.2; α2_0 = 0.9; α3_0 = 0.5
        β1_0 = 0.07; β2_0 = -0.05
        θ12_0 = 0.04
        # 3D init: axis 3 trivial (β_3 = 0, θ_13 = θ_23 = 0).
        for ci in leaves3
            lo, hi = cell_physical_box(frame3, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            cz = 0.5 * (lo[3] + hi[3])
            v = DetField3D((cx, cy, cz), (0.0, 0.0, 0.0),
                            (α1_0, α2_0, α3_0), (β1_0, β2_0, 0.0),
                            θ12_0, 0.0, 0.0, 1.0)
            write_detfield_3d!(fields3, ci, v)
        end

        mesh2, leaves2 = build_2d_mesh(2)
        frame2 = EulerianFrame(mesh2, (0.0, 0.0), (1.0, 1.0))
        fields2 = allocate_cholesky_2d_fields(mesh2)
        for ci in leaves2
            lo, hi = cell_physical_box(frame2, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (α1_0, α2_0), (β1_0, β2_0),
                            θ12_0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields2, ci, v)
        end

        dt = 1e-3
        det_step_3d_berry_HG!(fields3, mesh3, frame3, leaves3, bc_3d, dt;
                               M_vv_override = (1.0, 1.0, 0.0),
                               ρ_ref = 1.0)
        det_step_2d_berry_HG!(fields2, mesh2, frame2, leaves2, bc_2d, dt;
                               M_vv_override = (1.0, 1.0),
                               ρ_ref = 1.0)

        v2_ref = read_detfield_2d(fields2, leaves2[1])

        max_α1 = 0.0; max_β1 = 0.0
        max_α2 = 0.0; max_β2 = 0.0
        max_α3 = 0.0; max_β3 = 0.0
        max_θ12 = 0.0; max_θ13 = 0.0; max_θ23 = 0.0
        for ci in leaves3
            v = read_detfield_3d(fields3, ci)
            max_α1 = max(max_α1, abs(v.alphas[1] - v2_ref.alphas[1]))
            max_β1 = max(max_β1, abs(v.betas[1]  - v2_ref.betas[1]))
            max_α2 = max(max_α2, abs(v.alphas[2] - v2_ref.alphas[2]))
            max_β2 = max(max_β2, abs(v.betas[2]  - v2_ref.betas[2]))
            max_α3 = max(max_α3, abs(v.alphas[3] - α3_0))
            max_β3 = max(max_β3, abs(v.betas[3]  - 0.0))
            max_θ12 = max(max_θ12, abs(v.θ_12 - v2_ref.θ_R))
            max_θ13 = max(max_θ13, abs(v.θ_13))
            max_θ23 = max(max_θ23, abs(v.θ_23))
        end
        @test max_α1 ≤ M3_7C_DIMLIFT_TOL
        @test max_β1 ≤ M3_7C_DIMLIFT_TOL
        @test max_α2 ≤ M3_7C_DIMLIFT_TOL
        @test max_β2 ≤ M3_7C_DIMLIFT_TOL
        @test max_α3 ≤ M3_7C_DIMLIFT_TOL
        @test max_β3 ≤ M3_7C_DIMLIFT_TOL
        @test max_θ12 ≤ M3_7C_DIMLIFT_TOL
        @test max_θ13 ≤ M3_7C_DIMLIFT_TOL
        @test max_θ23 ≤ M3_7C_DIMLIFT_TOL
    end
end
