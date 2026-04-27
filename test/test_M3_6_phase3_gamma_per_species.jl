# test_M3_6_phase3_gamma_per_species.jl
#
# M3-6 Phase 3 (c): unit tests for the per-species per-axis γ
# diagnostic — `gamma_per_axis_2d_per_species` math primitive (in
# `src/cholesky_DD.jl`) and `gamma_per_axis_2d_per_species_field`
# field walker (in `src/newton_step_HG_M3_2.jl`).
#
# Three blocks:
#
#   1. Math primitive: `gamma_per_axis_2d_per_species(β, M_vv_per_species)`
#      reduces byte-equally to `gamma_per_axis_2d(β, M_vv_per_species[1])`
#      for the single-species case.
#   2. Math primitive: per-axis floor `γ²_a < 0` ⇒ `γ_a = 0` per species.
#   3. Multi-species independence: each species γ computed
#      independently; e.g., dust species `M_vv = 0` ⇒ `γ_a = 0`
#      regardless of β; gas species evaluated against EOS Mvv.
#   4. Field walker: single-species reduces to
#      `gamma_per_axis_2d_field` (byte-equal); multi-species shapes
#      and contents check out.
#
# See `reference/notes_M3_6_phase3_2d_substrate.md`.

using Test
using StaticArrays
using HierarchicalGrids: HierarchicalMesh,
    refine_cells!, enumerate_leaves
using dfmm: gamma_per_axis_2d, gamma_per_axis_2d_field,
            gamma_per_axis_2d_per_species,
            gamma_per_axis_2d_per_species_field,
            allocate_cholesky_2d_fields,
            write_detfield_2d!, DetField2D

@testset "M3-6 Phase 3 (c): per-species per-axis γ diagnostic" begin

    @testset "math primitive: single-species reduces to gamma_per_axis_2d" begin
        samples = [
            (β = SVector(0.10, 0.05),  Mvv = SVector(0.40, 0.30)),
            (β = SVector(0.30, 0.30),  Mvv = SVector(0.50, 0.50)),
            (β = SVector(0.0,  0.20),  Mvv = SVector(0.10, 0.20)),
            (β = SVector(0.99, 0.0),   Mvv = SVector(1.00, 1.00)),
            (β = SVector(-0.20, 0.40), Mvv = SVector(0.50, 0.40)),
        ]
        for (β, Mvv) in samples
            γ_single = gamma_per_axis_2d(β, Mvv)
            # n_species = 1 multi-species path:
            γ_multi = gamma_per_axis_2d_per_species(β, [Mvv])
            @test size(γ_multi) == (1, 2)
            @test γ_multi[1, 1] == γ_single[1]
            @test γ_multi[1, 2] == γ_single[2]
        end
    end

    @testset "math primitive: per-axis floor γ²<0 ⇒ γ=0 per species" begin
        β = SVector(0.95, 1.10)   # axis 2 violates β² > Mvv
        Mvv1 = SVector(1.00, 1.00)
        Mvv2 = SVector(2.00, 1.50)
        γ = gamma_per_axis_2d_per_species(β, [Mvv1, Mvv2])
        @test size(γ) == (2, 2)
        # Species 1: axis 1 OK, axis 2 floored.
        @test γ[1, 1] ≈ sqrt(1.0 - 0.95^2) atol = 1e-14
        @test γ[1, 2] == 0.0   # floored
        # Species 2: axis 1 OK, axis 2 OK.
        @test γ[2, 1] ≈ sqrt(2.0 - 0.95^2) atol = 1e-14
        @test γ[2, 2] ≈ sqrt(1.5 - 1.10^2) atol = 1e-14
    end

    @testset "math primitive: dust + gas multi-species independence" begin
        β = SVector(0.30, 0.20)
        # Dust species: pressureless ⇒ M_vv = (0, 0) ⇒ γ²_a = -β²_a < 0
        # ⇒ γ_a = 0 (floor) per axis.
        # Gas species: warm M_vv = (0.5, 0.5) ⇒ γ_a = √(0.5 - β²).
        Mvv_per_species = [SVector(0.0, 0.0), SVector(0.5, 0.5)]
        γ = gamma_per_axis_2d_per_species(β, Mvv_per_species)
        @test γ[1, 1] == 0.0
        @test γ[1, 2] == 0.0
        @test γ[2, 1] ≈ sqrt(0.5 - 0.30^2) atol = 1e-14
        @test γ[2, 2] ≈ sqrt(0.5 - 0.20^2) atol = 1e-14
    end

    @testset "math primitive: 3 species + assertion" begin
        # 3-species ICs, compare against per-axis primitive directly.
        β = SVector(0.10, 0.40)
        Mvv_per_species = [
            SVector(0.5, 0.5),    # gas
            SVector(0.0, 0.0),    # cold dust
            SVector(2.0, 1.5),    # hot ISM
        ]
        γ = gamma_per_axis_2d_per_species(β, Mvv_per_species)
        @test size(γ) == (3, 2)
        for k in 1:3
            γ_ref = gamma_per_axis_2d(β, Mvv_per_species[k])
            @test γ[k, 1] == γ_ref[1]
            @test γ[k, 2] == γ_ref[2]
        end
    end

    @testset "field walker: single-species reduces byte-equal to gamma_per_axis_2d_field" begin
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:2
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_2d_fields(mesh)
        for (k, ci) in enumerate(leaves)
            β1 = 0.1 * k
            β2 = 0.05 * k
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (β1, β2),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        Mvv_override = (1.0, 0.5)
        γ_single = gamma_per_axis_2d_field(fields, leaves;
                                             M_vv_override = Mvv_override)
        γ_multi = gamma_per_axis_2d_per_species_field(
            fields, leaves;
            M_vv_override_per_species = [Mvv_override],
            n_species = 1,
        )
        @test size(γ_multi) == (1, 2, length(leaves))
        for i in 1:length(leaves)
            @test γ_multi[1, 1, i] == γ_single[1, i]
            @test γ_multi[1, 2, i] == γ_single[2, i]
        end
    end

    @testset "field walker: multi-species independence (3 species)" begin
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:2
            refine_cells!(mesh, enumerate_leaves(mesh))
        end
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_2d_fields(mesh)
        for (k, ci) in enumerate(leaves)
            β1 = 0.1 * k
            β2 = 0.05 * k
            v = DetField2D((0.0, 0.0), (0.0, 0.0),
                            (1.0, 1.0), (β1, β2),
                            0.0, 0.0, 0.0, 0.0)
            write_detfield_2d!(fields, ci, v)
        end

        # 3 species: gas (Mvv=1), cold dust (Mvv=0), hot ISM (Mvv=2).
        Mvv_overrides = [(1.0, 1.0), (0.0, 0.0), (2.0, 2.0)]
        γ = gamma_per_axis_2d_per_species_field(
            fields, leaves;
            M_vv_override_per_species = Mvv_overrides,
            n_species = 3,
        )
        @test size(γ) == (3, 2, length(leaves))

        # Species 2 (dust): all γ values are 0 (any positive β
        # violates the realizability cone since Mvv_dust = 0).
        for i in 1:length(leaves)
            β1 = 0.1 * i
            β2 = 0.05 * i
            if β1 > 0
                @test γ[2, 1, i] == 0.0
            end
            if β2 > 0
                @test γ[2, 2, i] == 0.0
            end
        end

        # Species 1 (gas, Mvv=1): γ_a = √(1 - β_a²).
        for i in 1:length(leaves)
            β1 = 0.1 * i
            β2 = 0.05 * i
            @test γ[1, 1, i] ≈ sqrt(max(1.0 - β1^2, 0.0)) atol = 1e-14
            @test γ[1, 2, i] ≈ sqrt(max(1.0 - β2^2, 0.0)) atol = 1e-14
        end

        # Species 3 (hot ISM, Mvv=2): γ_a = √(2 - β_a²).
        for i in 1:length(leaves)
            β1 = 0.1 * i
            β2 = 0.05 * i
            @test γ[3, 1, i] ≈ sqrt(max(2.0 - β1^2, 0.0)) atol = 1e-14
            @test γ[3, 2, i] ≈ sqrt(max(2.0 - β2^2, 0.0)) atol = 1e-14
        end
    end

    @testset "field walker: nothing override → EOS-derived M_vv" begin
        # Without override, all species use the EOS Mvv(J, s); each
        # species gets the same γ matrix (since M_vv = Mvv(1/ρ, s)
        # is shared across species in this configuration).
        mesh = HierarchicalMesh{2}(; balanced = true)
        for _ in 1:1   # 4 leaves
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

        γ_single = gamma_per_axis_2d_field(fields, leaves)
        γ_multi = gamma_per_axis_2d_per_species_field(
            fields, leaves;
            M_vv_override_per_species = nothing,
            n_species = 2,
        )
        @test size(γ_multi) == (2, 2, length(leaves))
        for i in 1:length(leaves)
            for k in 1:2
                @test γ_multi[k, 1, i] == γ_single[1, i]
                @test γ_multi[k, 2, i] == γ_single[2, i]
            end
        end
    end

    @testset "field walker: input validation" begin
        mesh = HierarchicalMesh{2}(; balanced = true)
        refine_cells!(mesh, enumerate_leaves(mesh))
        leaves = enumerate_leaves(mesh)
        fields = allocate_cholesky_2d_fields(mesh)

        # n_species = 0 ⇒ assertion error.
        @test_throws AssertionError gamma_per_axis_2d_per_species_field(
            fields, leaves; n_species = 0)

        # M_vv_override_per_species length mismatch.
        @test_throws AssertionError gamma_per_axis_2d_per_species_field(
            fields, leaves;
            M_vv_override_per_species = [(1.0, 1.0)],
            n_species = 2)
    end

end
