# test_M3_7d_gamma_per_axis_diag_3d.jl
#
# M3-7d unit tests for the per-axis γ diagnostic on the 3D Cholesky-sector
# field set. Three blocks:
#
#   1. Math primitive `gamma_per_axis_3d(β, Mvv_diag)` (M1-style,
#      γ²_a = M_vv,aa − β_a²) reduces to the 1D `gamma_from_state`
#      on each axis independently.
#   2. Field-walking helper `gamma_per_axis_3d_field(fields, leaves;
#      M_vv_override)` returns the right per-axis γ.
#   3. `gamma_per_axis_3d_diag` is a non-trivial alias of
#      `gamma_per_axis_3d_field` (forwarded kwargs).
#
# See `reference/notes_M3_7_3d_extension.md` §4.3 and
# `reference/notes_M3_7d_3d_per_axis_gamma_amr.md`.

using Test
using StaticArrays
using HierarchicalGrids: HierarchicalMesh, refine_cells!, enumerate_leaves
using dfmm: gamma_per_axis_3d, gamma_per_axis_3d_field, gamma_per_axis_3d_diag,
            allocate_cholesky_3d_fields, write_detfield_3d!,
            DetField3D, gamma_from_state

@testset "M3-7d per-axis γ diagnostic (3D)" begin

    @testset "math primitive: γ²_a = M_vv,aa − β_a² reduces per-axis to M1" begin
        # Five sample points; cross-check each axis against M1's
        # gamma_from_state(M_vv, β).
        samples = [
            (β = SVector(0.10, 0.05, 0.02), Mvv = SVector(0.40, 0.30, 0.20)),
            (β = SVector(0.30, 0.30, 0.30), Mvv = SVector(0.50, 0.50, 0.50)),
            (β = SVector(0.0, 0.20, 0.40),  Mvv = SVector(0.10, 0.20, 0.50)),
            (β = SVector(0.99, 0.0, 0.0),   Mvv = SVector(1.00, 1.00, 1.00)),
            (β = SVector(-0.20, 0.40, -0.30), Mvv = SVector(0.50, 0.40, 0.30)),
        ]
        for (β, Mvv) in samples
            γ = gamma_per_axis_3d(β, Mvv)
            for a in 1:3
                # M1's gamma_from_state expects (M_vv, β) in that
                # order; the per-axis 3D version mirrors this.
                γ_M1 = gamma_from_state(Mvv[a], β[a])
                @test γ[a] ≈ γ_M1 atol = 1e-14
            end
        end
    end

    @testset "math primitive: realizability floor at γ² < 0" begin
        β = SVector(0.95, 1.10, 1.50)   # axes 2 and 3 violate β² > Mvv
        Mvv = SVector(1.00, 1.00, 1.00)
        γ = gamma_per_axis_3d(β, Mvv)
        @test γ[1] ≈ sqrt(1.0 - 0.95^2) atol = 1e-14
        @test γ[2] == 0.0   # floored
        @test γ[3] == 0.0   # floored
    end

    @testset "field-walking helper: M_vv_override path" begin
        mesh = HierarchicalMesh{3}(; balanced = true)
        for _ in 1:2
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_3d_fields(mesh)

        # Write a known per-leaf state.
        for (k, ci) in enumerate(leaves)
            β1 = 0.05 * k
            β2 = 0.04 * k
            β3 = 0.03 * k
            v = DetField3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                            (1.0, 1.5, 2.0), (β1, β2, β3),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_3d!(fields, ci, v)
        end

        γ_mat = gamma_per_axis_3d_field(fields, leaves;
                                         M_vv_override = (1.0, 0.5, 0.25))
        @test size(γ_mat) == (3, length(leaves))
        for (k, _) in enumerate(leaves)
            β1 = 0.05 * k
            β2 = 0.04 * k
            β3 = 0.03 * k
            γ1_expected = sqrt(max(1.0 - β1^2, 0.0))
            γ2_expected = sqrt(max(0.5 - β2^2, 0.0))
            γ3_expected = sqrt(max(0.25 - β3^2, 0.0))
            @test γ_mat[1, k] ≈ γ1_expected atol = 1e-14
            @test γ_mat[2, k] ≈ γ2_expected atol = 1e-14
            @test γ_mat[3, k] ≈ γ3_expected atol = 1e-14
        end
    end

    @testset "field-walking helper: EOS path (J=1/ρ_ref, s=0)" begin
        mesh = HierarchicalMesh{3}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_3d_fields(mesh)

        for ci in leaves
            v = DetField3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                            (1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_3d!(fields, ci, v)
        end

        γ_mat = gamma_per_axis_3d_field(fields, leaves; ρ_ref = 1.0)
        # At β = 0, s = 0, γ²_a = Mvv(1, 0) — all three axes equal.
        for k in 1:length(leaves)
            @test γ_mat[1, k] ≈ γ_mat[2, k] atol = 1e-14
            @test γ_mat[2, k] ≈ γ_mat[3, k] atol = 1e-14
            @test γ_mat[1, k] > 0
        end
    end

    @testset "alias: gamma_per_axis_3d_diag forwards to field-walking helper" begin
        mesh = HierarchicalMesh{3}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_3d_fields(mesh)
        for ci in leaves
            v = DetField3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                            (1.0, 1.0, 1.0), (0.2, 0.1, 0.05),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_3d!(fields, ci, v)
        end
        γ_a = gamma_per_axis_3d_field(fields, leaves;
                                       M_vv_override = (0.6, 0.4, 0.3))
        γ_b = gamma_per_axis_3d_diag(fields, leaves;
                                      M_vv_override = (0.6, 0.4, 0.3))
        @test γ_a == γ_b
    end

    @testset "1D-symmetric reduction: β_2 = β_3 = 0 ⇒ γ_2, γ_3 byte-equal" begin
        mesh = HierarchicalMesh{3}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_3d_fields(mesh)
        for (k, ci) in enumerate(leaves)
            β1 = 0.1 * k
            v = DetField3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                            (1.0, 1.0, 1.0), (β1, 0.0, 0.0),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_3d!(fields, ci, v)
        end
        γ_mat = gamma_per_axis_3d_field(fields, leaves;
                                         M_vv_override = (1.0, 1.0, 1.0))
        # axes 2 and 3 trivial: γ_2 = γ_3 = 1.0 for every leaf.
        for k in 1:length(leaves)
            @test γ_mat[2, k] == 1.0
            @test γ_mat[3, k] == 1.0
        end
        # axis 1 carries the spatial structure.
        @test maximum(γ_mat[1, :]) > minimum(γ_mat[1, :])
    end

end
