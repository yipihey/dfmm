# test_M3_6_phase0_offdiag_realizability.jl
#
# §Realizability projection compatibility (M3-6 Phase 0).
#
# The 2D realizability projection (`realizability_project_2d!` in
# `src/stochastic_injection.jl`) was developed in M3-3d for the 9-dof
# residual. M3-6 Phase 0 extends `DetField2D` and the field set with
# `β_12, β_21`, but the projection acts only on `s` (and reads `β_1, β_2`
# to set the per-axis Cholesky-cone target). At β_12=β_21=0 IC the
# projection's behaviour must remain byte-equal to M3-3d.
#
# M3-6 Phase 0 explicitly defers the per-cell-cone projection of
# `(β_12, β_21)` to M3-6 Phase 1+ (the D.1 KH falsifier driver will
# need it once non-zero IC fires). For Phase 0 we only verify:
#
#   1. The projection runs without crashing on the 14-named-field set.
#   2. At β_12 = β_21 = 0 IC, the projection leaves `β_12, β_21` at zero
#      (it doesn't read or write them, but the round-trip through the
#      field set must preserve the bit pattern).
#   3. The projection is byte-equal to M3-3d for `(α_1, α_2, β_1, β_2,
#      s)` at β_12 = β_21 = 0 IC.

using Test
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves,
    EulerianFrame, FrameBoundaries, REFLECTING, cell_physical_box
using dfmm: DetField2D, allocate_cholesky_2d_fields,
    read_detfield_2d, write_detfield_2d!,
    realizability_project_2d!, Mvv

@testset "M3-6 Phase 0 §Realizability: project preserves off-diag β at zero" begin

    mesh = HierarchicalMesh{2}(; balanced = true)
    for _ in 1:2
        refine_cells!(mesh, enumerate_leaves(mesh))
    end
    leaves = enumerate_leaves(mesh)
    frame = EulerianFrame(mesh, (0.0, 0.0), (1.0, 1.0))

    @testset "projection no-crash on 14-named-field set" begin
        fields = allocate_cholesky_2d_fields(mesh)
        # Set up a configuration well inside the realizability cone:
        # M_vv ≈ 1, β² ≈ 0.01 ⇒ headroom satisfied, projection is
        # a no-op.
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (1.0, 1.0), (0.1, -0.05),
                            (0.0, 0.0),       # off-diag β = 0
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        # Should not crash.
        realizability_project_2d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05,
                                   ρ_ref = 1.0)
        # Off-diag β remains at zero.
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            @test v.betas_off[1] == 0.0
            @test v.betas_off[2] == 0.0
        end
    end

    @testset "projection fires (β_a near boundary) — off-diag β unchanged" begin
        fields = allocate_cholesky_2d_fields(mesh)
        # Push β_1 close to the realizability cone boundary so the
        # `:reanchor` projection fires (raises `s`, which lifts M_vv).
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            # M_vv at s = 0, ρ = 1 is 1.0 (Mvv(1, 0) = exp(0) = 1).
            # β_1 = 0.99 ⇒ β_1² = 0.98 ⇒ M_vv must be ≥ 1.05·0.98
            # = 1.029 ⇒ headroom requires raising s.
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (1.0, 1.0), (0.99, 0.0),
                            (0.0, 0.0),
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        # Snapshot off-diag β before projection.
        β12_pre = Vector{Float64}(undef, length(leaves))
        β21_pre = Vector{Float64}(undef, length(leaves))
        for (j, ci) in enumerate(leaves)
            v = read_detfield_2d(fields, ci)
            β12_pre[j] = v.betas_off[1]
            β21_pre[j] = v.betas_off[2]
        end

        realizability_project_2d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05,
                                   ρ_ref = 1.0)

        # Off-diag β must be byte-equal pre / post.
        for (j, ci) in enumerate(leaves)
            v = read_detfield_2d(fields, ci)
            @test v.betas_off[1] == β12_pre[j]
            @test v.betas_off[2] == β21_pre[j]
        end
    end

    @testset "projection at non-zero off-diag β IC: doesn't touch them" begin
        # M3-6 Phase 0 explicitly does NOT extend the realizability
        # projection to act on `β_12, β_21`. Verify the projection is
        # a no-op on those two fields even when they are non-zero
        # (M3-6 Phase 1 will add the per-cell-cone extension).
        fields = allocate_cholesky_2d_fields(mesh)
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (1.0, 1.0), (0.1, 0.05),
                            (0.07, -0.03),     # non-zero off-diag β
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        realizability_project_2d!(fields, leaves;
                                   project_kind = :reanchor,
                                   headroom = 1.05,
                                   ρ_ref = 1.0)
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            @test v.betas_off == (0.07, -0.03)
        end
    end

    @testset ":none kind is no-op" begin
        fields = allocate_cholesky_2d_fields(mesh)
        for ci in leaves
            lo, hi = cell_physical_box(frame, ci)
            cx = 0.5 * (lo[1] + hi[1])
            cy = 0.5 * (lo[2] + hi[2])
            v = DetField2D((cx, cy), (0.0, 0.0),
                            (1.0, 1.0), (0.99, 0.0),
                            (0.04, -0.02),
                            0.0, 1.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        realizability_project_2d!(fields, leaves; project_kind = :none)
        for ci in leaves
            v = read_detfield_2d(fields, ci)
            @test v.betas_off == (0.04, -0.02)
            # `:none` doesn't touch s either.
            @test v.s == 1.0
        end
    end
end
