# test_M3_3d_gamma_per_axis_diag.jl
#
# M3-3d unit tests for the per-axis γ diagnostic on the 2D Cholesky-sector
# field set. Three blocks:
#
#   1. Math primitive `gamma_per_axis_2d(β, Mvv_diag)` (M1-style,
#      γ²_a = M_vv,aa − β_a²) reduces to the 1D `gamma_from_state`
#      on each axis independently.
#   2. Field-walking helper `gamma_per_axis_2d_field(fields, leaves;
#      M_vv_override)` returns the right per-axis γ.
#   3. `gamma_per_axis_2d_diag` is a non-trivial alias of
#      `gamma_per_axis_2d_field` (forwarded kwargs).
#
# See `reference/notes_M3_3_2d_cholesky_berry.md` §4.3 and
# `reference/notes_M3_3d_per_axis_gamma_amr.md`.

using Test
using StaticArrays
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves
using dfmm: gamma_per_axis_2d, gamma_per_axis_2d_field, gamma_per_axis_2d_diag,
            allocate_cholesky_2d_fields, write_detfield_2d!,
            DetField2D, gamma_from_state

@testset "M3-3d per-axis γ diagnostic" begin

    @testset "math primitive: γ²_a = M_vv,aa − β_a² reduces per-axis to M1" begin
        # Five sample points; cross-check each axis against M1's
        # gamma_from_state(M_vv, β).
        samples = [
            (β = SVector(0.10, 0.05), Mvv = SVector(0.40, 0.30)),
            (β = SVector(0.30, 0.30), Mvv = SVector(0.50, 0.50)),
            (β = SVector(0.0, 0.20),  Mvv = SVector(0.10, 0.20)),
            (β = SVector(0.99, 0.0),  Mvv = SVector(1.00, 1.00)),
            (β = SVector(-0.20, 0.40), Mvv = SVector(0.50, 0.40)),
        ]
        for (β, Mvv) in samples
            γ = gamma_per_axis_2d(β, Mvv)
            for a in 1:2
                # M1's gamma_from_state expects (M_vv, β) in that
                # order; the per-axis version mirrors this.
                γ_M1 = gamma_from_state(Mvv[a], β[a])
                @test γ[a] ≈ γ_M1 atol = 1e-14
            end
        end
    end

    @testset "math primitive: realizability floor at γ² < 0" begin
        β = SVector(0.95, 1.10)   # axis 2 violates β² > Mvv
        Mvv = SVector(1.00, 1.00)
        γ = gamma_per_axis_2d(β, Mvv)
        @test γ[1] ≈ sqrt(1.0 - 0.95^2) atol = 1e-14
        @test γ[2] == 0.0   # floored
    end

    @testset "field-walking helper: M_vv_override path" begin
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:2
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_2d_fields(mesh)

        # Write a known per-leaf state.
        for (k, ci) in enumerate(leaves)
            β1 = 0.1 * k
            β2 = 0.05 * k
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.5), (β1, β2),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        γ_mat = gamma_per_axis_2d_field(fields, leaves;
                                         M_vv_override = (1.0, 0.5))
        @test size(γ_mat) == (2, length(leaves))
        for (k, _) in enumerate(leaves)
            β1 = 0.1 * k
            β2 = 0.05 * k
            γ1_expected = sqrt(max(1.0 - β1^2, 0.0))
            γ2_expected = sqrt(max(0.5 - β2^2, 0.0))
            @test γ_mat[1, k] ≈ γ1_expected atol = 1e-14
            @test γ_mat[2, k] ≈ γ2_expected atol = 1e-14
        end
    end

    @testset "field-walking helper: EOS path (J=1/ρ_ref, s=0)" begin
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:1
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_2d_fields(mesh)

        for ci in leaves
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (0.0, 0.0),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        γ_mat = gamma_per_axis_2d_field(fields, leaves; ρ_ref = 1.0)
        # At β = 0, s = 0, γ²_a = Mvv(1, 0) — both axes equal.
        for k in 1:length(leaves)
            @test γ_mat[1, k] ≈ γ_mat[2, k] atol = 1e-14
            @test γ_mat[1, k] > 0
        end
    end

    @testset "alias: gamma_per_axis_2d_diag forwards to field-walking helper" begin
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:1
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_2d_fields(mesh)
        for ci in leaves
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (0.2, 0.1),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end
        γ_a = gamma_per_axis_2d_field(fields, leaves;
                                       M_vv_override = (0.6, 0.4))
        γ_b = gamma_per_axis_2d_diag(fields, leaves;
                                      M_vv_override = (0.6, 0.4))
        @test γ_a == γ_b
    end

end
