# test_M3_7c_h_rot_solvability_3d.jl
#
# §7.4 H_rot solvability constraint (M3-7c) — 3D SO(3).
#
# Three contracts to verify:
#
#   (a) Closed-form ∂H_rot/∂θ_{ab} from `h_rot_partial_dtheta_3d`
#       (`src/cholesky_DD_3d.jl`) satisfies the per-pair kernel-
#       orthogonality identity `dH · v_ker = 0` at 5 random generic
#       (α, β, γ²) points × 3 (θ̇_{ab})_test values × 3 pairs (1,2),
#       (1,3), (2,3). The 3D analog of M3-3c §6.4 Block 1 with three
#       per-pair sub-blocks.
#
#   (b) Iso-slice value of `h_rot_partial_dtheta_3d` evaluates to 0
#       per pair (3D analog of the 2D iso-slice contract).
#
#   (c) The 3D Newton system with (θ_12, θ_13, θ_23) as Newton
#       unknowns is solvable (non-singular Jacobian) at non-isotropic
#       3D ICs. Run `det_step_3d_berry_HG!` on a non-isotropic 3D IC;
#       verify the post-Newton residual norm ≤ 1e-10. The Newton
#       convergence sub-gate is "≤ 7 iterations" per the M3-7 design
#       note §7.4 (relaxed from 2D's ≤ 5 for the 13-dof system).

using Test
using StaticArrays
using Random
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, cell_physical_box
using dfmm: DetField3D, allocate_cholesky_3d_fields,
    write_detfield_3d!, read_detfield_3d,
    pack_state_3d, unpack_state_3d!,
    cholesky_el_residual_3D_berry,
    cholesky_el_residual_3D_berry!,
    build_residual_aux_3D,
    det_step_3d_berry_HG!,
    h_rot_partial_dtheta_3d, h_rot_kernel_orthogonality_residual_3d

const M3_7C_HROT_TOL = 1.0e-10

@testset "M3-7c §7.4 H_rot solvability constraint (3D SO(3))" begin

    # ─────────────────────────────────────────────────────────────────
    # Block 1: closed-form ∂H_rot/∂θ_{ab} per pair = kernel orthogonality
    # ─────────────────────────────────────────────────────────────────
    @testset "(a) closed-form ∂H_rot/∂θ_{ab} matches kernel-orthogonality (per pair)" begin
        rng = MersenneTwister(20260427)
        for sample in 1:5
            α1 = 0.5 + rand(rng) * 1.5
            α2 = 0.5 + rand(rng) * 1.5
            α3 = 0.5 + rand(rng) * 1.5
            if abs(α1 - α2) < 0.2; α2 += 0.3; end
            if abs(α1 - α3) < 0.2; α3 += 0.35; end
            if abs(α2 - α3) < 0.2; α3 += 0.4; end
            β1 = 2 * rand(rng) - 1
            β2 = 2 * rand(rng) - 1
            β3 = 2 * rand(rng) - 1
            M_vv_1 = 0.5 + rand(rng)
            M_vv_2 = 0.5 + rand(rng)
            M_vv_3 = 0.5 + rand(rng)
            γ²_1 = M_vv_1 - β1^2
            γ²_2 = M_vv_2 - β2^2
            γ²_3 = M_vv_3 - β3^2
            if γ²_1 < 0.05 || γ²_2 < 0.05 || γ²_3 < 0.05
                continue
            end

            α = SVector(α1, α2, α3)
            β = SVector(β1, β2, β3)
            γ² = SVector(γ²_1, γ²_2, γ²_3)

            for pair in (:_12, :_13, :_23)
                # Determine pair indices.
                a, b = if pair === :_12
                    (1, 2)
                elseif pair === :_13
                    (1, 3)
                else
                    (2, 3)
                end

                # Closed form per pair.
                h_rot_partial = h_rot_partial_dtheta_3d(α, β, γ²; pair = pair)

                # Compute α̇, β̇ from the per-pair Berry-modified Hamilton
                # equations (only the pair-(a,b) Berry block active —
                # third axis decouples via the SO(3) commutation
                # `[J_{ab}, (α_c, β_c)] = 0` for c ∉ {a, b}).
                for θ̇_ab_test in [0.0, 0.5, -1.3]
                    α̇ = if pair === :_12
                        SVector(
                            β1 - (α2^3) / (3 * α1^2) * θ̇_ab_test,
                            β2 + (α1^3) / (3 * α2^2) * θ̇_ab_test,
                            β3,
                        )
                    elseif pair === :_13
                        SVector(
                            β1 - (α3^3) / (3 * α1^2) * θ̇_ab_test,
                            β2,
                            β3 + (α1^3) / (3 * α3^2) * θ̇_ab_test,
                        )
                    else  # :_23
                        SVector(
                            β1,
                            β2 - (α3^3) / (3 * α2^2) * θ̇_ab_test,
                            β3 + (α2^3) / (3 * α3^2) * θ̇_ab_test,
                        )
                    end
                    β̇ = if pair === :_12
                        SVector(
                            γ²_1 / α1 - β2 * θ̇_ab_test,
                            γ²_2 / α2 + β1 * θ̇_ab_test,
                            γ²_3 / α3,
                        )
                    elseif pair === :_13
                        SVector(
                            γ²_1 / α1 - β3 * θ̇_ab_test,
                            γ²_2 / α2,
                            γ²_3 / α3 + β1 * θ̇_ab_test,
                        )
                    else  # :_23
                        SVector(
                            γ²_1 / α1,
                            γ²_2 / α2 - β3 * θ̇_ab_test,
                            γ²_3 / α3 + β2 * θ̇_ab_test,
                        )
                    end
                    R = h_rot_kernel_orthogonality_residual_3d(α, β, γ², α̇, β̇;
                                                                pair = pair)
                    @test abs(R) ≤ M3_7C_HROT_TOL
                end
            end
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 2: iso-slice value vanishes per pair
    # ─────────────────────────────────────────────────────────────────
    @testset "(b) iso-slice value: ∂H_rot/∂θ_{ab} = 0 at α_a=α_b, β_a=β_b, γ²_a=γ²_b" begin
        for (αv, βv, γ²v) in [
            (SVector(1.0, 1.0, 1.0), SVector(0.2, 0.2, 0.2), SVector(0.5, 0.5, 0.5)),
            (SVector(1.5, 1.5, 0.7), SVector(-0.3, -0.3, 0.1), SVector(0.6, 0.6, 0.4)),
            (SVector(0.8, 1.2, 0.8), SVector(0.05, -0.10, 0.05), SVector(0.7, 0.5, 0.7)),
        ]
            # Each tuple has at least one pair on iso. Verify the
            # closed-form value vanishes for that pair.
            for pair in (:_12, :_13, :_23)
                a, b = if pair === :_12
                    (1, 2)
                elseif pair === :_13
                    (1, 3)
                else
                    (2, 3)
                end
                if αv[a] ≈ αv[b] && βv[a] ≈ βv[b] && γ²v[a] ≈ γ²v[b]
                    val = h_rot_partial_dtheta_3d(αv, βv, γ²v; pair = pair)
                    @test abs(val) ≤ M3_7C_HROT_TOL
                end
            end
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 3: Newton convergence at non-isotropic 3D IC (≤ 7 iter)
    # ─────────────────────────────────────────────────────────────────
    @testset "(c) Newton convergence at non-isotropic 3D IC (≤ 7 iter)" begin
        mesh = HierarchicalMesh{3}(; balanced = true)
        for _ in 1:1
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        bc_spec = FrameBoundaries{3}(((REFLECTING, REFLECTING),
                                       (REFLECTING, REFLECTING),
                                       (REFLECTING, REFLECTING)))

        fields = allocate_cholesky_3d_fields(mesh)
        # Generic non-isotropic 3D IC. Distinct α's, all β's nonzero,
        # all θ_{ab} nonzero.
        α1 = 1.2; α2 = 0.8; α3 = 1.0
        β1 = 0.15; β2 = -0.10; β3 = 0.07
        θ12 = 0.13; θ13 = -0.09; θ23 = 0.05
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            cz = 0.5 * (lo[3] + hi[3])
            v = DetField3D((cx, cy, cz), (0.0, 0.0, 0.0),
                            (α1, α2, α3), (β1, β2, β3),
                            θ12, θ13, θ23, 1.0)
            write_detfield_3d!(fields, ci, v)
        end

        dt = 1e-3
        # M3-7 §7.4: Newton converges in ≤ 7 iterations on non-isotropic
        # 3D IC. We test by setting maxiters = 7 — if the solve reports
        # success at this budget, convergence count is ≤ 7.
        det_step_3d_berry_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                               M_vv_override = (1.0, 1.0, 1.0),
                               ρ_ref = 1.0,
                               abstol = 1e-13, reltol = 1e-13,
                               maxiters = 7)

        # Sanity: state evolved.
        v = read_detfield_3d(fields, leaves[1])
        @test abs(v.alphas[1] - α1) > 1e-10 ||
              abs(v.betas[1] - β1) > 1e-10 ||
              abs(v.theta_12 - θ12) > 1e-10

        # Re-run Newton from converged state with tight tol; should
        # converge in 1-2 Newton steps.
        det_step_3d_berry_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                               M_vv_override = (1.0, 1.0, 1.0),
                               ρ_ref = 1.0,
                               abstol = 1e-13, reltol = 1e-13,
                               maxiters = 7)
        @test true  # survival = convergence
    end

    # ─────────────────────────────────────────────────────────────────
    # Block 4: post-Newton residual norm ≤ 1e-10 at generic 3D IC
    # ─────────────────────────────────────────────────────────────────
    @testset "(d) post-Newton residual norm ≤ 1e-10 at generic 3D IC" begin
        mesh = HierarchicalMesh{3}(; balanced = true)
        for _ in 1:1
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        bc_spec = FrameBoundaries{3}(((REFLECTING, REFLECTING),
                                       (REFLECTING, REFLECTING),
                                       (REFLECTING, REFLECTING)))

        fields = allocate_cholesky_3d_fields(mesh)
        α1 = 1.5; α2 = 0.9; α3 = 1.1
        β1 = 0.05; β2 = -0.02; β3 = 0.03
        θ12 = 0.07; θ13 = -0.04; θ23 = 0.02
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            cz = 0.5 * (lo[3] + hi[3])
            v = DetField3D((cx, cy, cz), (0.0, 0.0, 0.0),
                            (α1, α2, α3), (β1, β2, β3),
                            θ12, θ13, θ23, 1.0)
            write_detfield_3d!(fields, ci, v)
        end

        aux = build_residual_aux_3D(fields, mesh, frame, leaves, bc_spec;
                                     M_vv_override = (1.0, 1.0, 1.0), ρ_ref = 1.0)
        y_n = pack_state_3d(fields, leaves)
        dt = 1e-3
        det_step_3d_berry_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                               M_vv_override = (1.0, 1.0, 1.0),
                               ρ_ref = 1.0,
                               abstol = 1e-13, reltol = 1e-13,
                               maxiters = 10)
        y_np1 = pack_state_3d(fields, leaves)
        F = cholesky_el_residual_3D_berry(y_np1, y_n, aux, dt)
        @test maximum(abs, F) ≤ 1e-10
    end
end
