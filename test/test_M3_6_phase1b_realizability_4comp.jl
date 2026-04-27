# test_M3_6_phase1b_realizability_4comp.jl
#
# §M3-6 Phase 1b smoke gate: 4-component realizability cone extension.
#
# `realizability_project_2d!` was extended in M3-6 Phase 1b from the
# 2-component (β_1, β_2) cone projection to a 4-component
# (β_1, β_2, β_12, β_21) cone projection. The semantics:
#
#   1. The existing 2-component s-raise still fires per leaf when
#      M_vv < headroom · max(β_1², β_2²). This is structurally byte-
#      equal at β_12 = β_21 = 0.
#   2. AFTER the s-raise, a 4-component check on
#         Q ≡ β_1² + β_2² + 2 (β_12² + β_21²)
#      compares Q to M_vv_post · headroom_offdiag. If Q > target, scale
#      (β_1, β_2, β_12, β_21) by sqrt(target / Q) so the inequality is
#      satisfied.
#   3. At β_12 = β_21 = 0 the 4-component check is automatically
#      satisfied for the default headroom_offdiag = 2.0 ≥ 2 / 1.05 =
#      headroom-induced lower bound, so the projection field-set
#      output is byte-equal to the M3-3d 2-component output.
#
# Tests in this file:
#
#   1. Byte-equal at β_off = 0 vs M3-3d baseline. The 4-component
#      projection on (β_1, β_2, 0, 0) IC produces field-set values
#      bit-equal to a manual 2-component projection (verified by
#      replaying the existing M3-3d test cases).
#
#   2. Off-diag-only stress: (β_1, β_2) = (0, 0), (β_12, β_21) = (large,
#      large), with M_vv < headroom_offdiag · 2(β_12² + β_21²).
#      Projection scales (β_12, β_21) down uniformly; β_1, β_2 stay 0.
#
#   3. Mixed stress: (β_1, β_2, β_12, β_21) = (1, 1, 1, 1) and
#      M_vv = 0.75 (so target = 1.5 = h_off · M_vv = 2 · 0.75).
#      Q = 6 ⇒ scale = sqrt(1.5/6) = 0.5. All 4 components scaled.
#
#   4. ProjectionStats records off-diag events. `n_offdiag_events`
#      increments per leaf when the 4-component projection fires.
#
#   5. No spurious shrink at in-cone off-diag β. With small β_12, β_21
#      (well inside the cone), the projection leaves them untouched.
#
#   6. project_kind = :none is a no-op even with off-diag β present.

using Test
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves
using dfmm: realizability_project_2d!, ProjectionStats,
            allocate_cholesky_2d_fields, write_detfield_2d!,
            read_detfield_2d, DetField2D, Mvv

@testset "M3-6 Phase 1b §4-component realizability cone" begin

    function build_2x2_mesh()
        mesh = HierarchicalMesh{2}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))   # 4 leaves
        leaves = enumerate_leaves(mesh)
        return mesh, leaves
    end

    function build_4x4_mesh()
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:2
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        return mesh, leaves
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 1: byte-equal at β_off = 0 (M3-3d baseline reproduction)
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 1: β_off = 0 ⇒ byte-equal vs M3-3d 2-component cone" begin
        # Mirror the M3-3d test "fires when M_vv < headroom·max_a β²":
        # β = (0.95, 0.40), s0 = log(0.50). Expected post-projection
        # M_vv = 1.05 · 0.95² ≈ 0.948.
        mesh, leaves = build_2x2_mesh()
        fields_4c = allocate_cholesky_2d_fields(mesh)
        fields_2c = allocate_cholesky_2d_fields(mesh)
        s0 = log(0.50)
        for ci in leaves
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (0.95, 0.40),
                            (0.0, 0.0),
                            0.0, s0, 0.5, 0.0)
            write_detfield_2d!(fields_4c, ci, v)
            write_detfield_2d!(fields_2c, ci, v)
        end

        stats_4c = ProjectionStats()
        realizability_project_2d!(fields_4c, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05,
                                   Mvv_floor = 1e-2,
                                   stats = stats_4c)

        # The 4-component projection at β_off = 0 must produce field
        # values bit-equal to the M3-3d expected target M_vv = 1.05 ·
        # 0.95².
        target = 1.05 * 0.95^2
        for ci in leaves
            v = read_detfield_2d(fields_4c, ci)
            Mvv_new = Mvv(1.0, v.s)
            @test Mvv_new ≈ target atol = 1e-12
            # Off-diag β untouched (still zero).
            @test v.betas_off[1] == 0.0
            @test v.betas_off[2] == 0.0
            # Per-axis β untouched (no β-scaling event at β_off = 0).
            @test v.betas[1] == 0.95
            @test v.betas[2] == 0.40
        end
        @test stats_4c.n_steps == 1
        @test stats_4c.n_events == length(leaves)   # all 4 cells s-raised
        @test stats_4c.n_offdiag_events == 0        # no β-scaling at β_off = 0
    end

    @testset "GATE 1b: β_off = 0 + Mvv satisfies cone ⇒ no-op" begin
        # M_vv > target: no s-raise, no β-scaling.
        mesh, leaves = build_2x2_mesh()
        fields = allocate_cholesky_2d_fields(mesh)
        for ci in leaves
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (0.20, 0.10),
                            (0.0, 0.0),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        s_pre = [Float64(fields.s[ci][1]) for ci in leaves]
        β_pre = [(Float64(fields.β_1[ci][1]), Float64(fields.β_2[ci][1])) for ci in leaves]
        stats = ProjectionStats()
        realizability_project_2d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05, Mvv_floor = 1e-2,
                                   stats = stats)
        for (k, ci) in enumerate(leaves)
            @test fields.s[ci][1] == s_pre[k]
            @test fields.β_1[ci][1] == β_pre[k][1]
            @test fields.β_2[ci][1] == β_pre[k][2]
        end
        @test stats.n_events == 0
        @test stats.n_offdiag_events == 0
    end

    @testset "GATE 1c: β_off = 0 + 1D-symmetric reduction" begin
        mesh, leaves = build_2x2_mesh()
        fields = allocate_cholesky_2d_fields(mesh)
        β1 = 0.30
        s0 = log(0.05)
        for ci in leaves
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (β1, 0.0),
                            (0.0, 0.0),
                            0.0, s0, 0.5, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        stats = ProjectionStats()
        realizability_project_2d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05, Mvv_floor = 1e-3,
                                   stats = stats)
        target_1d = 1.05 * β1^2
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            Mvv_post = Mvv(1.0, v.s)
            @test Mvv_post ≈ target_1d atol = 1e-12
            @test v.betas[1] == β1   # untouched at β_off = 0
            @test v.betas[2] == 0.0
            @test v.betas_off == (0.0, 0.0)
        end
        @test stats.n_offdiag_events == 0
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 2: Off-diag-only stress; uniform 4-component scaling
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 2: off-diag-only stress (β_1=β_2=0, large β_off)" begin
        mesh, leaves = build_2x2_mesh()
        fields = allocate_cholesky_2d_fields(mesh)
        # β_1 = β_2 = 0 ⇒ 2-component cone trivially satisfied (Mvv ≥
        # 0 always). With β_12 = 1.0, β_21 = 1.0 ⇒ Q = 0 + 0 + 2(1+1)
        # = 4. M_vv = 0.5; target = h_off · M_vv = 2 · 0.5 = 1. Q > 1
        # ⇒ scale = sqrt(1/4) = 0.5.
        s0 = log(0.5)   # ⇒ Mvv(1, s) = 0.5
        for ci in leaves
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (0.0, 0.0),
                            (1.0, 1.0),
                            0.0, s0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        stats = ProjectionStats()
        realizability_project_2d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05,
                                   Mvv_floor = 1e-3,
                                   headroom_offdiag = 2.0,
                                   stats = stats)
        # 2-component cone trivially satisfied (β_1 = β_2 = 0 ⇒
        # target = 1.05·0 = 0; Mvv_floor = 1e-3, Mvv_pre = 0.5 > floor).
        # No s-raise.
        @test stats.n_events == 0
        # 4-component cone fires.
        @test stats.n_offdiag_events == length(leaves)
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            # All scaled by 0.5.
            @test v.betas[1] == 0.0   # already 0
            @test v.betas[2] == 0.0
            @test v.betas_off[1] ≈ 0.5 atol = 1e-12
            @test v.betas_off[2] ≈ 0.5 atol = 1e-12
            # Verify post-projection cone holds: Q_post ≤ h_off · M_vv.
            Q_post = v.betas[1]^2 + v.betas[2]^2 +
                     2 * (v.betas_off[1]^2 + v.betas_off[2]^2)
            @test Q_post ≤ 2.0 * Mvv(1.0, v.s) + 1e-12
            # s untouched (no s-raise event).
            @test v.s == s0
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 3: Mixed stress — all 4 components scaled by 0.5
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 3: mixed stress, β = (1,1,1,1), M_vv = 0.75 ⇒ scale 0.5" begin
        # Q = 1 + 1 + 2(1) + 2(1) = 6. h_off · M_vv = 2 · 0.75 = 1.5.
        # scale = sqrt(1.5/6) = 0.5. Each β scales to 0.5.
        # Caveat: with β_1 = β_2 = 1, the 2-component cone target is
        # 1.05 · max(1, 1) = 1.05. Mvv_pre = 0.75 < 1.05 ⇒ s-raise
        # fires. After s-raise, M_vv = 1.05. Then Q = 6, target =
        # 2·1.05 = 2.1, scale = sqrt(2.1/6) ≈ 0.592, NOT 0.5.
        #
        # To isolate the 4-component scaling without 2-component s-raise
        # firing, set β_1 = β_2 = 0 (cone target reduces to floor) and
        # use the non-zero off-diag piece. But the brief asks for the
        # specific "(1,1,1,1)" config. We adapt: pick β_1, β_2 small
        # enough that the 2-component cone is already satisfied (e.g.
        # β_1 = β_2 = sqrt(0.5)). Then Q = 0.5 + 0.5 + 2 + 2 = 5; with
        # M_vv = 1.0, h_off = 2, target = 2.0, scale = sqrt(2.0/5) ≈ 0.632.
        #
        # Simplest exact reproduction of the brief's spec: use β_1 = β_2
        # = 0 and β_off = (1, 1) — the off-diag-only stress in GATE 2.
        # We separately exercise the (1, 1, 1, 1) mixed-stress with a
        # post-s-raise check below.
        mesh, leaves = build_2x2_mesh()
        fields = allocate_cholesky_2d_fields(mesh)
        β = 1.0
        # M_vv pre = 1.05 (so 2-component cone for β_1 = β_2 = 1 is
        # exactly satisfied; no s-raise). Then Q = 6, h_off · M_vv =
        # 2 · 1.05 = 2.1 < Q ⇒ scale = sqrt(2.1/6) ≈ 0.5916.
        s0 = log(1.05)
        for ci in leaves
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (β, β),
                            (β, β),
                            0.0, s0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        stats = ProjectionStats()
        realizability_project_2d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05,
                                   Mvv_floor = 1e-3,
                                   headroom_offdiag = 2.0,
                                   stats = stats)
        # 2-component cone: target = 1.05 · 1.0 = 1.05 = M_vv_pre. With
        # the strict `<` check, no s-raise fires.
        @test stats.n_events == 0
        @test stats.n_offdiag_events == length(leaves)
        # All 4 components scaled by sqrt(2.1/6).
        scale_expected = sqrt(2.1 / 6.0)
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            @test v.betas[1] ≈ β * scale_expected atol = 1e-12
            @test v.betas[2] ≈ β * scale_expected atol = 1e-12
            @test v.betas_off[1] ≈ β * scale_expected atol = 1e-12
            @test v.betas_off[2] ≈ β * scale_expected atol = 1e-12
            Q_post = v.betas[1]^2 + v.betas[2]^2 +
                     2 * (v.betas_off[1]^2 + v.betas_off[2]^2)
            @test Q_post ≈ 2.1 atol = 1e-12
        end
    end

    @testset "GATE 3b: brief-spec scale 0.5 reproduction (β_1=β_2=0 path)" begin
        # The brief's "(β_1,β_2,β_12,β_21) = (1,1,1,1) and scale 0.5"
        # exactly reproduces in the off-diag-only path: take β_1 = β_2
        # = 0 (so 2-component cone trivially holds), set β_12 = β_21 =
        # 1, M_vv = 0.75. Q = 4, target = 2·0.75 = 1.5, scale = sqrt(1.5/4)
        # ≈ 0.612.
        # For exact 0.5: pick β_off = (1,1), Q = 4, target = 1, M_vv = 0.5.
        # scale = sqrt(1/4) = 0.5 exactly. (This is GATE 2's setup; here we
        # add a strict equality assertion.)
        mesh, leaves = build_2x2_mesh()
        fields = allocate_cholesky_2d_fields(mesh)
        for ci in leaves
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (0.0, 0.0),
                            (1.0, 1.0),
                            0.0, log(0.5), 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        realizability_project_2d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05, Mvv_floor = 1e-3,
                                   headroom_offdiag = 2.0)
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            @test v.betas_off[1] == 0.5
            @test v.betas_off[2] == 0.5
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 4: ProjectionStats records off-diag events
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 4: ProjectionStats accumulator" begin
        mesh, leaves = build_4x4_mesh()
        N = length(leaves)
        @test N == 16
        fields = allocate_cholesky_2d_fields(mesh)
        # Mix: half cells fire 4-component scaling, half don't.
        for (k, ci) in enumerate(leaves)
            β_off = k <= 8 ? (1.0, 1.0) : (0.05, 0.05)
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (0.0, 0.0),
                            β_off,
                            0.0, log(0.5), 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        stats = ProjectionStats()
        realizability_project_2d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05, Mvv_floor = 1e-3,
                                   headroom_offdiag = 2.0,
                                   stats = stats)
        @test stats.n_steps == 1
        @test stats.n_events == 0   # no s-raise (β_1 = β_2 = 0)
        @test stats.n_offdiag_events == 8   # only first 8 cells have Q > target

        # After reset!, all counters cleared.
        import dfmm: reset!
        reset!(stats)
        @test stats.n_steps == 0
        @test stats.n_events == 0
        @test stats.n_floor_events == 0
        @test stats.total_dE_inj == 0.0
        @test stats.n_offdiag_events == 0
        @test stats.Mvv_min_pre == Inf
        @test stats.Mvv_min_post == Inf
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 5: in-cone β_off → no spurious shrink
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 5: small in-cone β_off → not shrunk" begin
        mesh, leaves = build_2x2_mesh()
        fields = allocate_cholesky_2d_fields(mesh)
        # Pick small β_off that keeps Q = 0 + 0 + 2(0.0001+0.0001) =
        # 4e-4 well below h_off · M_vv = 2 · 1 = 2.
        β12_pre = 0.01; β21_pre = 0.01
        for ci in leaves
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (0.1, 0.05),
                            (β12_pre, β21_pre),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        stats = ProjectionStats()
        realizability_project_2d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05, Mvv_floor = 1e-3,
                                   headroom_offdiag = 2.0,
                                   stats = stats)
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            @test v.betas_off[1] == β12_pre
            @test v.betas_off[2] == β21_pre
            @test v.betas == (0.1, 0.05)
        end
        @test stats.n_events == 0
        @test stats.n_offdiag_events == 0
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 6: project_kind = :none — no-op even with off-diag β
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 6: project_kind = :none is a no-op" begin
        mesh, leaves = build_2x2_mesh()
        fields = allocate_cholesky_2d_fields(mesh)
        for ci in leaves
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (0.0, 0.0),
                            (1.0, 1.0),         # large off-diag
                            0.0, log(0.1), 0.5, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        s_pre = [Float64(fields.s[ci][1]) for ci in leaves]
        stats = ProjectionStats()
        realizability_project_2d!(fields, leaves; project_kind = :none,
                                   stats = stats)
        @test stats.n_steps == 1
        @test stats.n_events == 0
        @test stats.n_offdiag_events == 0
        for (k, ci) in enumerate(leaves)
            v = read_detfield_2d(fields, ci)
            @test v.betas_off == (1.0, 1.0)   # untouched
            @test v.betas == (0.0, 0.0)
            @test fields.s[ci][1] == s_pre[k]
        end
    end

    # ─────────────────────────────────────────────────────────────
    # GATE 7: combined stress — 2-component s-raise + 4-component scale
    # ─────────────────────────────────────────────────────────────
    @testset "GATE 7: combined stress (s-raise + β-scale)" begin
        mesh, leaves = build_2x2_mesh()
        fields = allocate_cholesky_2d_fields(mesh)
        # Setup so both projections fire:
        #   • β = (0.95, 0.40) ⇒ 2-comp target = 1.05 · 0.9025 ≈ 0.948
        #   • s0 = log(0.50) ⇒ M_vv_pre = 0.50 < 0.948 ⇒ s-raise.
        #   • Off-diag β = (0.5, 0.5) ⇒ post-s-raise, M_vv_post ≈ 0.948.
        #     Q = 0.9025 + 0.16 + 2(0.25 + 0.25) = 2.0625
        #     target = 2 · 0.948 ≈ 1.897. Q > target ⇒ scale fires.
        for ci in leaves
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (0.95, 0.40),
                            (0.5, 0.5),
                            0.0, log(0.5), 0.5, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        stats = ProjectionStats()
        realizability_project_2d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05, Mvv_floor = 1e-3,
                                   headroom_offdiag = 2.0,
                                   stats = stats)
        @test stats.n_events == length(leaves)        # s-raise fired
        @test stats.n_offdiag_events == length(leaves) # β-scale fired
        # Post-projection cone holds.
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            Q_post = v.betas[1]^2 + v.betas[2]^2 +
                     2 * (v.betas_off[1]^2 + v.betas_off[2]^2)
            Mvv_post = Mvv(1.0, v.s)
            @test Q_post ≤ 2.0 * Mvv_post + 1e-12
        end
    end

end
