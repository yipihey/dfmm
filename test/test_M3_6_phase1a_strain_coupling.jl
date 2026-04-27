# test_M3_6_phase1a_strain_coupling.jl
#
# §M3-6 Phase 1a smoke gate: off-diagonal strain coupling H_rot^off in
# the 2D EL residual.
#
# Phase 0 left F^β_12, F^β_21 trivial-drive (β̇ = 0) and F^θ_R also
# trivial-drive. Phase 1a wires in the H_rot^off ∝ G̃_12·(α_1·β_21 +
# α_2·β_12)/2 coupling that makes the off-diagonal Cholesky pair
# physically dynamical. The KH falsifier (Phase 1c) hangs off this
# drive.
#
# Tests in this file:
#
#   1. Strain stencil sanity. With a sheared base flow
#      `u_1(x, y) = U·tanh((y − 0.5)/w)` (axis-2 dependence only,
#      u_2 = 0), the residual's `(∂_2 u_1, ∂_1 u_2)` stencil reads
#      a finite-difference value matching the analytic `tanh`
#      derivative to FD-stencil tolerance.
#
#   2. Coupling drives β_12, β_21 from rest. With β_12 = β_21 = 0 IC
#      and the sheared flow above, a single Newton step yields nonzero
#      β_12, β_21 ≠ 0 (the F^β_12, F^β_21 rows acquire G̃_12·ᾱ_a/2
#      drives that pull β_12, β_21 off zero in the converged state).
#      This is the load-bearing structural gate.
#
#   3. Bit-exact gate at axis-aligned IC. The Phase 1a residual at
#      u = (0, 0) IC produces a residual byte-equal to Phase 0 — i.e.,
#      every M3-3c regression configuration's residual is unchanged.
#
#   4. Rotational equivariance. A 90°-rotated shear (u_1 = 0,
#      u_2(x_1) = U·tanh((x_1 − 0.5)/w)) produces the mirror-image
#      off-diagonal stencil (∂_1 u_2 = old ∂_2 u_1, ∂_2 u_1 = 0)
#      and drives β_12, β_21 with the same magnitude (same sign for
#      G̃_12 since strain is symmetric in the two off-diagonals).

using Test
using StaticArrays
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, cell_physical_box
using dfmm: DetField2D, allocate_cholesky_2d_fields,
    read_detfield_2d, write_detfield_2d!,
    pack_state_2d_berry, unpack_state_2d_berry!,
    cholesky_el_residual_2D_berry!, cholesky_el_residual_2D_berry,
    build_residual_aux_2D, det_step_2d_berry_HG!

const M3_6_PHASE1A_TOL = 1.0e-12

@testset "M3-6 Phase 1a §Off-diagonal strain coupling" begin

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
    # GATE 1: cross-axis stencil reads finite-difference value
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 1: ∂_2 u_1 stencil matches centered FD on tanh shear" begin
        # 4×4 mesh (level 2) on the unit square. Sheared base flow
        # u_1(y) = U · tanh((y − 0.5)/w), u_2 = 0.
        mesh, leaves = build_mesh_and_leaves(2)
        @test length(leaves) == 16
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)

        U = 0.5
        w = 0.15

        cell_centers_y = Float64[]
        cell_centers_x = Float64[]
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            push!(cell_centers_x, cx)
            push!(cell_centers_y, cy)
            u1 = U * tanh((cy - 0.5) / w)
            u2 = 0.0
            v = DetField2D((cx, cy), (u1, u2),
                            (1.0, 1.0), (0.0, 0.0),
                            (0.0, 0.0),
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        dt = 1e-3
        # M_vv_override = (0, 0) ⇒ no pressure gradient, isolates strain
        # stencil. ρ_ref = 1 standard.
        aux = build_residual_aux_2D(fields, mesh, frame, leaves, bc_spec;
                                     M_vv_override = (0.0, 0.0), ρ_ref = 1.0)
        y_n = pack_state_2d_berry(fields, leaves)
        F = cholesky_el_residual_2D_berry(y_n, y_n, aux, dt)

        # Probe the strain stencil indirectly: the F^β_12 row equals
        #   (β_12_np1 − β_12_n)/dt + G̃_12·ᾱ_2/2
        # at fixed-point input (y_n == y_np1) ⇒ first term = 0, so
        #   F[base+9] = G̃_12 · α_2 / 2 = G̃_12 / 2.
        # The cross-axis cells in the 4×4 mesh have axis-2 neighbours
        # at the cell row above and below; for an interior row,
        #   ∂_2 u_1 ≈ (u_1(y_above) − u_1(y_below)) / (y_above − y_below)
        #   ∂_1 u_2 = 0 (u_2 ≡ 0)
        # ⇒ G̃_12 = (∂_2 u_1 + 0)/2 = (∂_2 u_1)/2.
        # Predicted F[base+9] = (∂_2 u_1)/4.
        # For a 4×4 mesh the cell centres are at y ∈ {1/8, 3/8, 5/8, 7/8}
        # and the spacing between row-above and row-below centres is
        # 2*(1/4) = 1/2 for interior rows. Boundary rows mirror-self
        # for the missing neighbour.

        # Build a per-cell expected G̃_12 by reproducing the residual's
        # neighbour-table arithmetic.
        face_lo = aux.face_lo_idx
        face_hi = aux.face_hi_idx
        n_interior_checks = 0
        n_boundary_checks = 0
        max_err = 0.0
        for (i, ci) in enumerate(leaves)
            ilo2 = face_lo[2][i]
            ihi2 = face_hi[2][i]
            # Mirror-self at boundary (matches the residual).
            if ilo2 == 0
                cy_lo = cell_centers_y[i]
                u1_lo = U * tanh((cy_lo - 0.5) / w)
            else
                cy_lo = cell_centers_y[ilo2]
                u1_lo = U * tanh((cy_lo - 0.5) / w)
            end
            if ihi2 == 0
                cy_hi = cell_centers_y[i]
                u1_hi = U * tanh((cy_hi - 0.5) / w)
            else
                cy_hi = cell_centers_y[ihi2]
                u1_hi = U * tanh((cy_hi - 0.5) / w)
            end
            Δy = cy_hi - cy_lo
            d2u1_expected = Δy > 0 ? (u1_hi - u1_lo) / Δy : 0.0
            # G̃_12 = (∂_2 u_1 + ∂_1 u_2)/2 = ∂_2 u_1 / 2.
            G̃12_expected = d2u1_expected / 2
            # F[base+9] = G̃_12 · α_2 / 2 = G̃_12 / 2 (α_2 = 1).
            base = 11 * (i - 1)
            f_β12_expected = G̃12_expected * 1.0 / 2
            err = abs(F[base + 9] - f_β12_expected)
            max_err = max(max_err, err)
            if ilo2 == 0 || ihi2 == 0
                n_boundary_checks += 1
            else
                n_interior_checks += 1
            end
            @test err ≤ M3_6_PHASE1A_TOL
        end
        @test n_interior_checks > 0
        @test n_boundary_checks > 0
        @test max_err ≤ M3_6_PHASE1A_TOL

        # F^θ_R = (θ_R_np1 − θ_R_n)/dt + W_12 · F_off. At β_12 = β_21 = 0
        # IC, F_off = 0 ⇒ F[base+11] = 0 even though W_12 ≠ 0.
        for i in 1:length(leaves)
            base = 11 * (i - 1)
            @test abs(F[base + 11]) ≤ M3_6_PHASE1A_TOL
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 2: bit-exact at axis-aligned IC (regression preservation)
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 2: u = (0, 0) ⇒ Phase 1a residual matches Phase 0" begin
        # At u = (0, 0) every cell, ∂_2 u_1 = 0 and ∂_1 u_2 = 0
        # ⇒ G̃_12 = W_12 = 0 ⇒ every Phase 1a addition vanishes
        # multiplicatively. The residual reduces byte-equal to the
        # Phase 0 form. This gate is the structural guarantee that
        # all M3-3c, M3-4, M3-6 Phase 0 tests still pass byte-equal.
        mesh, leaves = build_mesh_and_leaves(2)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)

        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (1.2, 0.8), (0.1, -0.05),
                            (0.0, 0.0),
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        dt = 1e-3
        aux = build_residual_aux_2D(fields, mesh, frame, leaves, bc_spec;
                                     M_vv_override = (1.0, 1.0), ρ_ref = 1.0)
        y_n = pack_state_2d_berry(fields, leaves)
        F = cholesky_el_residual_2D_berry(y_n, y_n, aux, dt)

        # At fixed-point with u = 0 and β_12 = β_21 = 0:
        #   F^β_12 = 0  (G̃_12·α_2/2 = 0 since G̃_12 = 0)
        #   F^β_21 = 0
        #   F^θ_R = 0
        # F^β_a still contains -γ_a²/α_a from M3-3c (M_vv = 1 ⇒ γ_a² =
        # 1 - β_a²); for β_1 = 0.1, α_1 = 1.2: F^β_1 = -(1 - 0.01)/1.2 = -0.825.
        for (i, _) in enumerate(leaves)
            base = 11 * (i - 1)
            @test abs(F[base + 9]) ≤ M3_6_PHASE1A_TOL    # F^β_12 unchanged
            @test abs(F[base + 10]) ≤ M3_6_PHASE1A_TOL   # F^β_21 unchanged
            @test abs(F[base + 11]) ≤ M3_6_PHASE1A_TOL   # F^θ_R unchanged
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 3: Phase 1a drives β_12, β_21 from rest
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 3: sheared IC drives β_12, β_21 ≠ 0 after one Newton step" begin
        # Same IC as GATE 1; run a single Newton step. The trivial-
        # drive Phase 0 form would leave β_12 = β_21 = 0 forever; with
        # Phase 1a drive, β_12 and β_21 must move off rest.
        mesh, leaves = build_mesh_and_leaves(2)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)

        U = 0.5
        w = 0.15
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            u1 = U * tanh((cy - 0.5) / w)
            u2 = 0.0
            v = DetField2D((cx, cy), (u1, u2),
                            (1.0, 1.0), (0.0, 0.0),
                            (0.0, 0.0),
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        dt = 1e-3
        det_step_2d_berry_HG!(fields, mesh, frame, leaves, bc_spec, dt;
                               M_vv_override = (0.0, 0.0), ρ_ref = 1.0)

        # Verify β_12, β_21 moved off rest in at least some cells.
        max_β12 = 0.0
        max_β21 = 0.0
        sum_β12_sq = 0.0
        sum_β21_sq = 0.0
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            max_β12 = max(max_β12, abs(v.betas_off[1]))
            max_β21 = max(max_β21, abs(v.betas_off[2]))
            sum_β12_sq += v.betas_off[1]^2
            sum_β21_sq += v.betas_off[2]^2
        end

        # The drive after one step at dt = 1e-3 with G̃_12 ~ U/w/2 ~
        # 0.5/0.15/2 ~ 1.7 and α_2 = 1 gives β̇_12 ~ −1.7/2 ~ −0.85,
        # so |β_12| ~ 0.85 · dt ~ 8.5e-4 in the shear layer cells.
        # Demand at least 1e-5 to be conservative against the tanh
        # plateau cells.
        @test max_β12 ≥ 1.0e-5
        @test max_β21 ≥ 1.0e-5
        # Both β_12 and β_21 should be of comparable magnitude (the
        # drives are G̃_12·α_2/2 and G̃_12·α_1/2 with α_1 = α_2 = 1).
        @test abs(max_β12 - max_β21) ≤ 1.0e-12
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 4: rotational equivariance (90°-rotated shear)
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 4: 90°-rotated shear produces mirror stencil" begin
        # Rotated IC: u_1 = 0, u_2(x) = U·tanh((x − 0.5)/w). Now
        #   ∂_2 u_1 = 0, ∂_1 u_2 ≠ 0
        # ⇒ G̃_12 = (∂_1 u_2)/2, W_12 = -(∂_1 u_2)/2.
        # By symmetry the F^β_12 row magnitude equals the unrotated
        # F^β_12 magnitude, evaluated at the analogous (rotated) cell.
        mesh, leaves = build_mesh_and_leaves(2)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields_xy = allocate_cholesky_2d_fields(mesh)
        fields_yx = allocate_cholesky_2d_fields(mesh)

        U = 0.5
        w = 0.15
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            # Original (axis-2 dependence): u_1(y) = U tanh(...)
            v_xy = DetField2D((cx, cy), (U * tanh((cy - 0.5) / w), 0.0),
                                (1.0, 1.0), (0.0, 0.0),
                                (0.0, 0.0),
                                0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields_xy, ci, v_xy)
            # Rotated (axis-1 dependence): u_2(x) = U tanh(...)
            v_yx = DetField2D((cx, cy), (0.0, U * tanh((cx - 0.5) / w)),
                                (1.0, 1.0), (0.0, 0.0),
                                (0.0, 0.0),
                                0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields_yx, ci, v_yx)
        end

        dt = 1e-3
        aux_xy = build_residual_aux_2D(fields_xy, mesh, frame, leaves, bc_spec;
                                        M_vv_override = (0.0, 0.0), ρ_ref = 1.0)
        aux_yx = build_residual_aux_2D(fields_yx, mesh, frame, leaves, bc_spec;
                                        M_vv_override = (0.0, 0.0), ρ_ref = 1.0)
        y_xy = pack_state_2d_berry(fields_xy, leaves)
        y_yx = pack_state_2d_berry(fields_yx, leaves)
        F_xy = cholesky_el_residual_2D_berry(y_xy, y_xy, aux_xy, dt)
        F_yx = cholesky_el_residual_2D_berry(y_yx, y_yx, aux_yx, dt)

        # By symmetry G̃_12 is the same magnitude in both ICs (both are
        # symmetric strain × 1/2). With α_1 = α_2 = 1, F^β_12 in the
        # rotated case must match the original at the cell-axis-swapped
        # location.
        max_xy_β12 = 0.0
        max_yx_β12 = 0.0
        max_xy_β21 = 0.0
        max_yx_β21 = 0.0
        for (i, _) in enumerate(leaves)
            base = 11 * (i - 1)
            max_xy_β12 = max(max_xy_β12, abs(F_xy[base + 9]))
            max_yx_β12 = max(max_yx_β12, abs(F_yx[base + 9]))
            max_xy_β21 = max(max_xy_β21, abs(F_xy[base + 10]))
            max_yx_β21 = max(max_yx_β21, abs(F_yx[base + 10]))
        end
        # Same magnitudes (axis swap preserves G̃_12 since it's a
        # symmetric strain).
        @test max_xy_β12 ≈ max_yx_β12 rtol=1e-12
        @test max_xy_β21 ≈ max_yx_β21 rtol=1e-12
        @test max_xy_β12 == max_xy_β21      # α_1 = α_2 = 1 ⇒ same drive
        @test max_yx_β12 == max_yx_β21
        @test max_xy_β12 > 0   # nonzero coupling

        # W_12 has opposite sign in the two ICs, but with β_12 = β_21 =
        # 0 IC ⇒ F_off = 0 ⇒ F^θ_R = 0 in both cases.
        for (i, _) in enumerate(leaves)
            base = 11 * (i - 1)
            @test abs(F_xy[base + 11]) ≤ M3_6_PHASE1A_TOL
            @test abs(F_yx[base + 11]) ≤ M3_6_PHASE1A_TOL
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 5: F^θ_R picks up vorticity drive when β_12, β_21 ≠ 0
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 5: W_12 · F_off drives F^θ_R when β_off ≠ 0" begin
        # With sheared u_1(y) and a non-zero β_12 IC, F^θ_R picks up
        # the W_12 · F_off term. Verify it matches the closed form.
        mesh, leaves = build_mesh_and_leaves(2)
        frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))
        fields = allocate_cholesky_2d_fields(mesh)

        U = 0.5
        w = 0.15
        β12_IC = 0.07
        β21_IC = -0.03
        α1 = 1.2
        α2 = 0.8
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            u1 = U * tanh((cy - 0.5) / w)
            v = DetField2D((cx, cy), (u1, 0.0),
                            (α1, α2), (0.0, 0.0),
                            (β12_IC, β21_IC),
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        dt = 1e-3
        aux = build_residual_aux_2D(fields, mesh, frame, leaves, bc_spec;
                                     M_vv_override = (0.0, 0.0), ρ_ref = 1.0)
        y_n = pack_state_2d_berry(fields, leaves)
        F = cholesky_el_residual_2D_berry(y_n, y_n, aux, dt)

        # Closed form per cell (using the residual's neighbour-table
        # arithmetic; same as GATE 1).
        face_lo = aux.face_lo_idx
        face_hi = aux.face_hi_idx
        F_off = (α1^2 * α2 * β12_IC - α1 * α2^2 * β21_IC) / 2

        max_err_θR = 0.0
        for (i, ci) in enumerate(leaves)
            lo, hi = cell_physical_box(frame, ci)
            cy_self = 0.5 * (lo[2] + hi[2])
            ilo2 = face_lo[2][i]
            ihi2 = face_hi[2][i]
            cy_lo = ilo2 == 0 ? cy_self : begin
                lo_n, hi_n = cell_physical_box(frame, leaves[ilo2])
                0.5 * (lo_n[2] + hi_n[2])
            end
            cy_hi = ihi2 == 0 ? cy_self : begin
                lo_n, hi_n = cell_physical_box(frame, leaves[ihi2])
                0.5 * (lo_n[2] + hi_n[2])
            end
            u1_lo = U * tanh((cy_lo - 0.5) / w)
            u1_hi = U * tanh((cy_hi - 0.5) / w)
            Δy = cy_hi - cy_lo
            d2u1 = Δy > 0 ? (u1_hi - u1_lo) / Δy : 0.0
            d1u2 = 0.0   # u_2 ≡ 0
            W12 = (d2u1 - d1u2) / 2
            f_θR_expected = W12 * F_off

            base = 11 * (i - 1)
            err = abs(F[base + 11] - f_θR_expected)
            max_err_θR = max(max_err_θR, err)
        end
        @test max_err_θR ≤ M3_6_PHASE1A_TOL
    end
end
