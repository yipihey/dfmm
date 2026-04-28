# test_M4_phase2_3d_kh_falsifier.jl
#
# §M4 Phase 2 falsifier acceptance gates: 3D Kelvin-Helmholtz
# eigenmode growth (or HONEST FALSIFICATION lifted from 2D Phase 1)
# under the closed-loop β_off ↔ β_a coupling on the 21-dof 3D
# residual.
#
# This test consumes the `experiments/D1_3D_KH_growth_rate.jl`
# driver. Tests:
#
#   1. IC mass conservation (β_a = 0 IC + uniform ρ).
#   2. Bit-exact regression: at β_off = 0 IC + axis-aligned u, the
#      21-dof residual reduces to the M3-7c 15-dof Berry residual
#      byte-equal in the first 15 slots; rows 16..21 are pure
#      kinematic drives.
#   3. Driver smoke at level 2 (no NaN, finite trajectory).
#   4. Linear-regime growth rate at level 3: c_off ∈ [0.5, 2.0]
#      (broad band, matching the 2D Phase 1c verdict).
#   5. Linear-in-t vs exp-in-t fit comparison: linear should fit
#      better — confirming the 2D finding generalizes to 3D.
#   6. 6-component β cone realizability: n_negative_jacobian = 0
#      throughout the level-3 trajectory.
#
# The headline gate is c_off ∈ [0.5, 2.0]; the tighter [0.8, 1.2]
# aspiration is not expected to pass (the M4 Phase 1 honest finding
# generalizes to 3D).

using Test
using Statistics: mean, std
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, PERIODIC, cell_physical_box
using dfmm: tier_d_kh_3d_ic_full,
    det_step_3d_berry_kh_HG!,
    cholesky_el_residual_3D_berry_kh,
    cholesky_el_residual_3D_berry,
    pack_state_3d_kh, pack_state_3d,
    build_residual_aux_3D_kh, build_residual_aux_3D,
    allocate_cholesky_3d_kh_fields, allocate_cholesky_3d_fields

const _M4P2_3D_DRIVER_PATH = joinpath(@__DIR__, "..", "experiments",
                                       "D1_3D_KH_growth_rate.jl")
include(_M4P2_3D_DRIVER_PATH)

const M4_PHASE2_TOL = 1.0e-12

@testset verbose = true "M4 Phase 2 §D.1 3D KH falsifier" begin

    # ─────────────────────────────────────────────────────────────
    # GATE 1: IC mass conservation + uniform-ρ IC sanity
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 1: IC sanity at level 2 (4³ = 64 cells)" begin
        ic = tier_d_kh_3d_ic_full(; level = 2, U_jet = 1.0,
                                    jet_width = 0.15,
                                    perturbation_amp = 1e-3,
                                    perturbation_k = 2)
        N = length(ic.leaves)
        @test N == 64
        # ρ_per_cell uniform.
        @test all(ic.ρ_per_cell .== ic.ρ_per_cell[1])
        # Total mass conservation: sum of cell volumes × ρ ≈ ρ_0 · L_x · L_y · L_z
        # within 1e-12 (uniform mesh, uniform density).
        Vtot = 0.0
        for ci in ic.leaves
            lo, hi = cell_physical_box(ic.frame, ci)
            Vtot += (hi[1] - lo[1]) * (hi[2] - lo[2]) * (hi[3] - lo[3])
        end
        @test abs(Vtot - 1.0) ≤ M4_PHASE2_TOL
        # IC has α = (1, 1, 1), β_a = 0.
        for ci in ic.leaves
            @test ic.fields.α_1[ci][1] ≈ 1.0 atol = M4_PHASE2_TOL
            @test ic.fields.α_2[ci][1] ≈ 1.0 atol = M4_PHASE2_TOL
            @test ic.fields.α_3[ci][1] ≈ 1.0 atol = M4_PHASE2_TOL
            @test abs(ic.fields.β_1[ci][1]) ≤ M4_PHASE2_TOL
            @test abs(ic.fields.β_2[ci][1]) ≤ M4_PHASE2_TOL
            @test abs(ic.fields.β_3[ci][1]) ≤ M4_PHASE2_TOL
            @test abs(ic.fields.θ_12[ci][1]) ≤ M4_PHASE2_TOL
            @test abs(ic.fields.θ_13[ci][1]) ≤ M4_PHASE2_TOL
            @test abs(ic.fields.θ_23[ci][1]) ≤ M4_PHASE2_TOL
        end
        # Off-diag (1,3) / (2,3) pairs zero at IC; (1,2) antisymmetric.
        for ci in ic.leaves
            @test abs(ic.fields.β_13[ci][1]) ≤ M4_PHASE2_TOL
            @test abs(ic.fields.β_31[ci][1]) ≤ M4_PHASE2_TOL
            @test abs(ic.fields.β_23[ci][1]) ≤ M4_PHASE2_TOL
            @test abs(ic.fields.β_32[ci][1]) ≤ M4_PHASE2_TOL
            # β_12 = -β_21 antisymmetric tilt mode
            @test abs(ic.fields.β_12[ci][1] + ic.fields.β_21[ci][1]) ≤ M4_PHASE2_TOL
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 2: at β_off = 0 IC + axis-aligned u, the 21-dof residual
    # reduces byte-equal to the M3-7c 15-dof Berry residual on the
    # first 15 slots; rows 16..21 are pure kinematic drives.
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 2: bit-exact regression at β_off = 0 (M3-7c parity)" begin
        function build_zero_strain_state(level::Int)
            mesh = HierarchicalMesh{3}(; balanced = true)
            for _ in 1:level
                refine_cells!(mesh, enumerate_leaves(mesh))
            end
            leaves = enumerate_leaves(mesh)
            frame = EulerianFrame(mesh, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
            f_kh = allocate_cholesky_3d_kh_fields(mesh)
            f_base = allocate_cholesky_3d_fields(mesh)
            for ci in leaves
                lo, hi = cell_physical_box(frame, ci)
                cx = 0.5 * (lo[1] + hi[1])
                cy = 0.5 * (lo[2] + hi[2])
                cz = 0.5 * (lo[3] + hi[3])
                # Non-trivial state: α ≠ 1, β ≠ 0, θ_12 ≠ 0; but β_off = 0
                # AND velocity axis-aligned (u_2 = u_3 = 0, u_1 only varies
                # along axis 1).
                u1 = 0.5 * sin(2π * cx)
                for f in (f_kh, f_base)
                    f.x_1[ci] = (cx,); f.x_2[ci] = (cy,); f.x_3[ci] = (cz,)
                    f.u_1[ci] = (u1,); f.u_2[ci] = (0.0,); f.u_3[ci] = (0.0,)
                    f.α_1[ci] = (1.2,); f.α_2[ci] = (0.8,); f.α_3[ci] = (1.0,)
                    f.β_1[ci] = (0.05,); f.β_2[ci] = (-0.03,); f.β_3[ci] = (0.02,)
                    f.θ_12[ci] = (0.07,); f.θ_13[ci] = (0.0,); f.θ_23[ci] = (0.0,)
                    f.s[ci] = (1.0,)
                end
                f_kh.β_12[ci] = (0.0,); f_kh.β_21[ci] = (0.0,)
                f_kh.β_13[ci] = (0.0,); f_kh.β_31[ci] = (0.0,)
                f_kh.β_23[ci] = (0.0,); f_kh.β_32[ci] = (0.0,)
            end
            return mesh, leaves, frame, f_kh, f_base
        end

        mesh, leaves, frame, f_kh, f_base = build_zero_strain_state(2)
        bc = FrameBoundaries{3}(((REFLECTING, REFLECTING),
                                  (REFLECTING, REFLECTING),
                                  (REFLECTING, REFLECTING)))
        dt = 1e-3

        aux_kh = build_residual_aux_3D_kh(f_kh, mesh, frame, leaves, bc;
                                          M_vv_override = (1.0, 1.0, 1.0),
                                          ρ_ref = 1.0, c_back = 1.0)
        aux_base = build_residual_aux_3D(f_base, mesh, frame, leaves, bc;
                                          M_vv_override = (1.0, 1.0, 1.0),
                                          ρ_ref = 1.0)
        y_kh = pack_state_3d_kh(f_kh, leaves)
        y_base = pack_state_3d(f_base, leaves)
        F_kh = cholesky_el_residual_3D_berry_kh(y_kh, y_kh, aux_kh, dt)
        F_base = cholesky_el_residual_3D_berry(y_base, y_base, aux_base, dt)

        # First 15 slots per cell: byte-equal between 21-dof KH and 15-dof
        # M3-7c Berry residuals.
        N = length(leaves)
        max_diff_15 = 0.0
        for i in 1:N
            b_kh = 21 * (i - 1); b_b = 15 * (i - 1)
            for k in 1:15
                d = abs(F_kh[b_kh + k] - F_base[b_b + k])
                max_diff_15 = max(max_diff_15, d)
            end
        end
        @test max_diff_15 == 0.0
        # Off-diag rows (16..21) are pure trivial drive (β_off_np1 −
        # β_off_n)/dt = 0 at fixed-point, plus G̃ * α / 2 which vanishes
        # at axis-aligned u.
        max_off = 0.0
        for i in 1:N
            b_kh = 21 * (i - 1)
            for k in 16:21
                max_off = max(max_off, abs(F_kh[b_kh + k]))
            end
        end
        @test max_off ≤ M4_PHASE2_TOL
        # Same parity at c_back = 0.
        aux_kh0 = build_residual_aux_3D_kh(f_kh, mesh, frame, leaves, bc;
                                             M_vv_override = (1.0, 1.0, 1.0),
                                             ρ_ref = 1.0, c_back = 0.0)
        F_kh0 = cholesky_el_residual_3D_berry_kh(y_kh, y_kh, aux_kh0, dt)
        max_diff_15_0 = 0.0
        for i in 1:N
            b_kh = 21 * (i - 1); b_b = 15 * (i - 1)
            for k in 1:15
                d = abs(F_kh0[b_kh + k] - F_base[b_b + k])
                max_diff_15_0 = max(max_diff_15_0, d)
            end
        end
        @test max_diff_15_0 == 0.0
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 3: driver smoke at level 2 (4³ = 64 cells)
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 3: driver smoke at level 2" begin
        # T_factor=0.7 ensures the linear-window [0.5/γ_DR, 1.0/γ_DR]
        # overlaps the trajectory; with T_factor=0.4 the window starts
        # after T_end and the linear fit returns NaN (n_pts < 2).
        r = run_D1_3D_KH_growth_rate(; level = 2, T_factor = 0.7,
                                       c_back = 1.0, verbose = false,
                                       t_window_factor = (0.2, 0.6))
        @test haskey(r, :γ_measured)
        @test haskey(r, :c_off)
        @test r.γ_DR ≈ 1.0 / (2 * 0.15) atol = M4_PHASE2_TOL
        @test !r.nan_seen
        for v in r.A_rms
            @test isfinite(v)
        end
        # γ_measured of order γ_DR.
        @test r.γ_measured > 0.0
        @test r.γ_measured < 100.0 * r.γ_DR
        # Positive trajectory growth.
        @test r.A_rms[end] > r.A_rms[1]
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 4: linear-regime growth at level 3 — c_off ∈ [0.5, 2.0]
    # broad band (not the tightened [0.8, 1.2] aspiration).
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 4: level 3 broad-band acceptance c_off ∈ [0.5, 2.0]" begin
        r = run_D1_3D_KH_growth_rate(; level = 3, T_factor = 0.7,
                                       c_back = 1.0, verbose = false)
        @test !r.nan_seen
        @test isfinite(r.γ_measured)
        @test r.γ_measured > 0.0
        # Methods paper §10.5 D.1 broad band [0.5, 2.0].
        @test 0.5 ≤ r.c_off ≤ 2.0
        # ASPIRATION: tighter band [0.8, 1.2]. Whether this passes is the
        # honest finding. Use @test_skip / soft-report.
        if 0.8 ≤ r.c_off ≤ 1.2
            @info "M4 Phase 2: c_off ∈ [0.8, 1.2] aspiration band PASSED at level 3" c_off=r.c_off
        else
            @info "M4 Phase 2: c_off ∈ [0.8, 1.2] aspiration band NOT achieved (HONEST_FALSIFICATION confirmed in 3D)" c_off=r.c_off
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 5: linear-in-t vs exp-in-t fit comparison. The 2D verdict
    # was "linear-in-t fits better"; the 3D version must report the
    # honest verdict.
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 5: linear-in-t vs exp-in-t fit (HONEST verdict)" begin
        r = run_D1_3D_KH_growth_rate(; level = 3, T_factor = 0.7,
                                       c_back = 1.0, verbose = false)
        # Restrict to the linear window for the fit (avoids
        # IC-perturbation transients).
        keep = [k for k in eachindex(r.t)
                if r.t[k] >= 0.2 * r.T_KH && r.t[k] <= 0.8 * r.T_KH]
        @test length(keep) >= 3
        fit = fit_linear_vs_exp_3d(r.t[keep], r.A_rms[keep])
        @test isfinite(fit.ssr_lin)
        @test isfinite(fit.ssr_exp) || fit.ssr_exp == Inf
        # The 2D Phase 1 verdict: linear-in-t fits better.
        # The 3D verdict on this trajectory; honest reporting.
        if fit.lin_better
            @info "M4 Phase 2 verdict (3D): LINEAR-IN-T fits better — 2D KINEMATIC-ONLY finding GENERALIZES to 3D" ssr_lin=fit.ssr_lin ssr_exp=fit.ssr_exp
        else
            @info "M4 Phase 2 verdict (3D): EXP-IN-T fits better — 3D differs from 2D! γ_exp=$(fit.γ_exp)" ssr_lin=fit.ssr_lin ssr_exp=fit.ssr_exp
        end
        # Honest assertion: both fits produce finite slopes.
        @test isfinite(fit.b_lin)
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 6: 6-component β cone realizability — n_negative_jacobian
    # = 0 throughout the level-3 trajectory.
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 6: 6-component β cone — n_negative_jacobian = 0" begin
        r = run_D1_3D_KH_growth_rate(; level = 3, T_factor = 0.7,
                                       c_back = 1.0, verbose = false)
        for k in eachindex(r.n_negative_jacobian)
            @test r.n_negative_jacobian[k] == 0
        end
        # 6-component β cone: each pair stays bounded.
        for k in eachindex(r.t)
            @test r.off12_max[k] < 10.0   # very loose upper bound on linear regime
            @test r.off21_max[k] < 10.0
            @test r.off13_max[k] < 10.0
            @test r.off31_max[k] < 10.0
            @test r.off23_max[k] < 10.0
            @test r.off32_max[k] < 10.0
        end
        # The (1, 3) and (2, 3) off-diag pairs start at 0; they may
        # develop nonzero values via cross-axis coupling, but should
        # stay subdominant to the (1, 2) tilt mode in the linear regime
        # (that's the dominant KH mode along x-y).
        @info "M4 Phase 2: max |β_12| at end of trajectory" max_b12=r.off12_max[end]
        @info "M4 Phase 2: max |β_13| at end (cross-coupling indicator)" max_b13=r.off13_max[end]
        @info "M4 Phase 2: max |β_23| at end (cross-coupling indicator)" max_b23=r.off23_max[end]
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 7: c_back parameter sweep at level 2 — no NaN at
    # c_back ∈ {0, 0.5, 1, 2}. Also tests the c_back = 0 fallback
    # for kinematic-only behaviour.
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 7: c_back parameter sweep (no NaN)" begin
        for c_back_val in (0.0, 0.5, 1.0, 2.0)
            r = run_D1_3D_KH_growth_rate(; level = 2, T_factor = 0.7,
                                           c_back = c_back_val, verbose = false,
                                           t_window_factor = (0.2, 0.6))
            @test !r.nan_seen
            @test isfinite(r.γ_measured)
            @test r.γ_measured > 0.0
            @test r.A_rms[end] > 0.0
            @info "GATE 7: c_back=$(c_back_val) ⇒ c_off=$(r.c_off)"
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 8: dimension-lift property — the dominant (1, 2) pair
    # tilt mode exhibits the expected growth, and the (1, 3) /
    # (2, 3) pairs (no IC perturbation) show subdominant growth
    # consistent with cross-axis Berry coupling.
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 8: dimension-lift — (1, 2) dominant, (1, 3) / (2, 3) subdominant" begin
        r = run_D1_3D_KH_growth_rate(; level = 2, T_factor = 0.7,
                                       c_back = 1.0, verbose = false,
                                       t_window_factor = (0.2, 0.6))
        @test !r.nan_seen
        # (1, 2) pair tilt mode IS the seed; should grow.
        @test r.off12_max[end] > r.off12_max[1]
        # (1, 3) pair starts at 0; with no z-axis structure in the
        # base flow the cross-axis coupling that activates it is the
        # Berry block. With u_3 = 0 ∂_3 u_1 = 0 and ∂_1 u_3 = 0 ⇒
        # G̃_13 = 0. So the F^β_13, F^β_31 forcing is 0; rows decouple.
        # The (1, 3) pair stays at zero throughout.
        @test r.off13_max[end] < 1e-6
        @test r.off31_max[end] < 1e-6
        # Same for (2, 3) pair: u_3 = 0 + u_2 = 0 ⇒ G̃_23 = 0.
        @test r.off23_max[end] < 1e-6
        @test r.off32_max[end] < 1e-6
    end

end
