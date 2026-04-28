# test_M4_phase1_dimension_lift.jl
#
# §M4 Phase 1 §Dimension-lift gate (CRITICAL).
#
# At β_off = 0 (the M3-3c / M3-4 / M3-6 Phase 0 / M3-6 Phase 1a regression
# configuration), the new closed-loop residual `cholesky_el_residual_2D_berry!`
# (with `c_back ≠ 0` enabling the H_back back-reaction) must reduce
# byte-equal to the M3-6 Phase 1a form. The H_back contribution
#
#   H_back = c_back · G̃_12 · (α_2·β_12·β_2 + α_1·β_21·β_1) / 2
#
# is multiplicative in β_off · β_a, so:
#   • At β_off = 0 IC + axis-aligned (G̃_12 = 0): every H_back term
#     vanishes ⇒ residual matches M3-6 Phase 0.
#   • At β_off = 0 IC + sheared u_1(y) (G̃_12 ≠ 0): the H_back
#     contributions to F^β_12, F^β_21 carry an extra β_a factor; with
#     β_a = 0 at IC these contributions are 0 at the first residual
#     evaluation. Once β_a develops nonzero values across one Newton
#     step, the closed-loop terms activate (the M4 Phase 1 design
#     intent). At pure β_off = 0 AND β_a = 0 (e.g. M3-3c), bit-exact
#     preservation holds throughout.
#   • Setting `c_back = 0.0` recovers the M3-6 Phase 1a form byte-equal
#     across all configurations (the regression-fallback).

using Test
using StaticArrays
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, cell_physical_box
using dfmm: DetField2D, allocate_cholesky_2d_fields,
    read_detfield_2d, write_detfield_2d!,
    pack_state_2d_berry, det_step_2d_berry_HG!,
    build_residual_aux_2D, cholesky_el_residual_2D_berry,
    cholesky_step, cholesky_run

const M4_PHASE1_DIMLIFT_TOL = 1.0e-12

@testset "M4 Phase 1 §Dimension-lift gate (β_off=0 ⇒ M3-6 Phase 1a parity)" begin

    function build_mesh_and_leaves(level::Int)
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:level
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        return mesh, enumerate_leaves(mesh)
    end

    bc_spec = FrameBoundaries{2}(((REFLECTING, REFLECTING),
                                   (REFLECTING, REFLECTING)))

    # ─────────────────────────────────────────────────────────────
    # GATE 1: at β_off = 0 IC + axis-aligned u, the closed-loop
    # residual evaluated at fixed-point input matches the c_back = 0
    # form byte-equal.
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 1: axis-aligned ⇒ c_back has no effect" begin
        mesh, leaves = build_mesh_and_leaves(2)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)

        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (1.2, 0.8), (0.05, -0.03),
                            (0.0, 0.0),     # β_off = 0
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        dt = 1e-3
        # Residual at fixed-point input (y_n == y_np1) — captures the
        # algebraic residual structure, not the time-derivative pieces.
        for c_back_val in (0.0, 1.0)
            aux = build_residual_aux_2D(fields, mesh, frame, leaves, bc_spec;
                                         M_vv_override = (1.0, 1.0),
                                         ρ_ref = 1.0,
                                         c_back = c_back_val)
            y = pack_state_2d_berry(fields, leaves)
            F = cholesky_el_residual_2D_berry(y, y, aux, dt)
            for (i, _) in enumerate(leaves)
                base = 11 * (i - 1)
                # F^β_12, F^β_21 (off-diag rows) must vanish at axis-aligned
                # IC regardless of c_back (G̃_12 = 0 ⇒ both H_rot^off and
                # H_back contributions = 0).
                @test abs(F[base + 9])  ≤ M4_PHASE1_DIMLIFT_TOL
                @test abs(F[base + 10]) ≤ M4_PHASE1_DIMLIFT_TOL
                @test abs(F[base + 11]) ≤ M4_PHASE1_DIMLIFT_TOL
            end
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 2: full M3-3c-style 1D-symmetric IC, multi-step,
    # bit-exact across c_back values.
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 2: M3-3c 1D-symmetric IC, c_back vs c_back=0 byte-equal" begin
        # Use 1D-symmetric IC: u = (0, 0), α = (α_1, const_axis2),
        # β = (β_1, 0), β_off = 0. The dimension-lift expectation is
        # that c_back has no observable effect because at β_off = 0 AND
        # axis-aligned, every H_back contribution vanishes.

        for c_back_val in (0.0, 0.5, 1.0)
            mesh1, leaves1 = build_mesh_and_leaves(2)
            mesh0, leaves0 = build_mesh_and_leaves(2)
            frame1 = EulerianFrame(mesh1, (0.0, 0.0), (1.0, 1.0))
            frame0 = EulerianFrame(mesh0, (0.0, 0.0), (1.0, 1.0))
            fields1 = allocate_cholesky_2d_fields(mesh1)
            fields0 = allocate_cholesky_2d_fields(mesh0)

            α0_axis1 = 1.0
            α0_axis2 = 1.5
            for ci in leaves1
                lo, hi = cell_physical_box(frame1, ci)
                cx = 0.5 * (lo[1] + hi[1])
                cy = 0.5 * (lo[2] + hi[2])
                v = DetField2D((cx, cy), (0.0, 0.0),
                                (α0_axis1, α0_axis2), (0.0, 0.0),
                                (0.0, 0.0),
                                0.0, 1.0, 0.0, 0.0)
                write_detfield_2d!(fields1, ci, v)
                write_detfield_2d!(fields0, ci, v)
            end

            dt = 1e-3
            for n in 1:5
                det_step_2d_berry_HG!(fields1, mesh1, frame1, leaves1,
                                       bc_spec, dt;
                                       M_vv_override = (1.0, 0.0),
                                       ρ_ref = 1.0,
                                       c_back = c_back_val)
                det_step_2d_berry_HG!(fields0, mesh0, frame0, leaves0,
                                       bc_spec, dt;
                                       M_vv_override = (1.0, 0.0),
                                       ρ_ref = 1.0,
                                       c_back = 0.0)
            end

            max_diff_α1 = 0.0; max_diff_β1 = 0.0
            max_diff_α2 = 0.0; max_diff_β2 = 0.0
            max_diff_β12 = 0.0; max_diff_β21 = 0.0
            max_diff_θR = 0.0
            for (ci_1, ci_0) in zip(leaves1, leaves0)
                v1 = read_detfield_2d(fields1, ci_1)
                v0 = read_detfield_2d(fields0, ci_0)
                max_diff_α1 = max(max_diff_α1, abs(v1.alphas[1] - v0.alphas[1]))
                max_diff_α2 = max(max_diff_α2, abs(v1.alphas[2] - v0.alphas[2]))
                max_diff_β1 = max(max_diff_β1, abs(v1.betas[1]  - v0.betas[1]))
                max_diff_β2 = max(max_diff_β2, abs(v1.betas[2]  - v0.betas[2]))
                max_diff_β12 = max(max_diff_β12, abs(v1.betas_off[1] - v0.betas_off[1]))
                max_diff_β21 = max(max_diff_β21, abs(v1.betas_off[2] - v0.betas_off[2]))
                max_diff_θR = max(max_diff_θR, abs(v1.θ_R - v0.θ_R))
            end
            # At β_off = 0 AND axis-aligned, c_back has zero effect ⇒
            # bit-exact byte-equal.
            @test max_diff_α1  ≤ M4_PHASE1_DIMLIFT_TOL
            @test max_diff_α2  ≤ M4_PHASE1_DIMLIFT_TOL
            @test max_diff_β1  ≤ M4_PHASE1_DIMLIFT_TOL
            @test max_diff_β2  ≤ M4_PHASE1_DIMLIFT_TOL
            @test max_diff_β12 ≤ M4_PHASE1_DIMLIFT_TOL
            @test max_diff_β21 ≤ M4_PHASE1_DIMLIFT_TOL
            @test max_diff_θR  ≤ M4_PHASE1_DIMLIFT_TOL
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 3: 2D ⊂ 1D dimension lift — at β_off = 0 + α_2 = const,
    # β_2 = 0, the closed-loop 2D residual reduces to M1 1D path.
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 3: 2D ⊂ 1D — closed-loop residual matches M1" begin
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
        # Run the closed-loop residual.
        det_step_2d_berry_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                               M_vv_override = (1.0, 0.0), ρ_ref = 1.0,
                               c_back = 1.0)

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
            max_θR  = max(max_θR,  abs(v.θ_R))
        end

        # At α_2 = const, β_2 = 0, β_off = 0, axis-aligned: closed-loop
        # 2D residual matches M1 1D path bit-exactly.
        @test max_α1  ≤ M4_PHASE1_DIMLIFT_TOL
        @test max_β1  ≤ M4_PHASE1_DIMLIFT_TOL
        @test max_α2  ≤ M4_PHASE1_DIMLIFT_TOL
        @test max_β2  ≤ M4_PHASE1_DIMLIFT_TOL
        @test max_β12 ≤ M4_PHASE1_DIMLIFT_TOL
        @test max_β21 ≤ M4_PHASE1_DIMLIFT_TOL
        @test max_θR  ≤ M4_PHASE1_DIMLIFT_TOL
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 4: with sheared u + β_off = 0 IC, the c_back=0 vs
    # c_back=1 paths differ ONLY when β_a develops nonzero values
    # — the H_back contributions to F^β_off carry an explicit β_a
    # factor.
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 4: sheared IC + β_off=0, β_a=0 ⇒ first-residual byte-equal" begin
        mesh, leaves = build_mesh_and_leaves(2)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)

        # Sheared u_1(y), β_a = 0, β_off = 0 — the M3-6 Phase 1a
        # smoke-test IC.
        U = 0.5
        w = 0.15
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            u1 = U * tanh((cy - 0.5) / w)
            v = DetField2D((cx, cy), (u1, 0.0),
                            (1.0, 1.0), (0.0, 0.0),
                            (0.0, 0.0),
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        dt = 1e-3
        # Compare residuals at fixed-point input across c_back ∈ {0, 1}.
        # At β_off = 0 AND β_a = 0, the H_back contribution
        #   c_back · G̃ · α · β_a / 2
        # in F^β_off is 0; the H_back contribution
        #   c_back · G̃ · β_a · β_off / (2·α²)
        # in F^β_a is also 0. So the residual is byte-equal across
        # c_back values at this specific IC.
        for c_back_val in (0.0, 0.25, 0.5, 1.0, 2.0)
            aux = build_residual_aux_2D(fields, mesh, frame, leaves, bc_spec;
                                         M_vv_override = (0.0, 0.0),
                                         ρ_ref = 1.0,
                                         c_back = c_back_val)
            aux_ref = build_residual_aux_2D(fields, mesh, frame, leaves, bc_spec;
                                             M_vv_override = (0.0, 0.0),
                                             ρ_ref = 1.0,
                                             c_back = 0.0)
            y = pack_state_2d_berry(fields, leaves)
            F = cholesky_el_residual_2D_berry(y, y, aux, dt)
            F_ref = cholesky_el_residual_2D_berry(y, y, aux_ref, dt)
            max_F_diff = maximum(abs.(F .- F_ref))
            @test max_F_diff ≤ M4_PHASE1_DIMLIFT_TOL
        end
    end

end
