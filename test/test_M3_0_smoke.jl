# test_M3_0_smoke.jl
#
# Phase M3-0 smoke test: verify the new HG + R3D dependencies load and
# the core HG primitives dfmm consumes are reachable.
#
# This is the dependency-side gate; the per-trajectory bit-equality
# checks live in `test_M3_0_parity_1D.jl`. If both tests pass, the
# foundation phase is solid and M3-1 can begin.

using Test
using HierarchicalGrids
using R3D
using dfmm

@testset "M3-0 smoke: HG + R3D import + 1D primitives" begin
    @testset "dependency imports succeed" begin
        # `using` in the file header would have failed if these didn't
        # load; the tests here are sentinels to record the gate
        # explicitly in the test summary.
        @test isdefined(@__MODULE__, :HierarchicalGrids)
        @test isdefined(@__MODULE__, :R3D)
        @test isdefined(@__MODULE__, :dfmm)
    end

    @testset "HG 1D simplicial mesh constructs" begin
        # 2-cell, 3-vertex periodic-by-construction-not interval.
        mesh = uniform_simplicial_mesh_1D(2, 1.0)
        @test n_simplices(mesh) == 2
        @test n_vertices(mesh) == 3
        @test simplex_volume(mesh, 1) ≈ 0.5
        @test simplex_volume(mesh, 2) ≈ 0.5
    end

    @testset "HG order-0 PolynomialFieldSet for (α, β) carries values" begin
        fields = allocate_chfield_HG(3)
        fields.alpha[1] = (1.0,);  fields.beta[1] = (0.0,)
        fields.alpha[2] = (2.0,);  fields.beta[2] = (-0.5,)
        fields.alpha[3] = (1.5,);  fields.beta[3] = (0.25,)
        @test fields.alpha[1][1] == 1.0
        @test fields.beta[2][1] == -0.5
        @test fields.alpha[3][1] == 1.5
    end

    @testset "single-cell HG run reaches Phase-1 closed-form to truncation order" begin
        # Independent of the M1 bit-equality tests: just sanity-check
        # that the HG-side driver hits the expected Phase-1 trajectory
        # at the implicit-midpoint truncation tolerance.
        mesh   = single_cell_simplicial_mesh_1D()
        fields = allocate_chfield_HG(1)
        fields.alpha[1] = (1.0,)
        fields.beta[1]  = (0.0,)

        M_vv      = 1.0
        divu_half = 0.0
        Δt        = 1e-3
        N         = 100

        cholesky_run_HG!(mesh, fields, M_vv, divu_half, Δt, N)

        t = N * Δt
        α_exact = sqrt(1 + t^2)
        β_exact = t / sqrt(1 + t^2)
        @test isapprox(fields.alpha[1][1], α_exact; atol = 1e-5)
        @test isapprox(fields.beta[1][1],  β_exact; atol = 1e-5)
    end

    @testset "ChFieldND dimension-generic tag type" begin
        c1 = ChFieldND(1.0, 0.5)
        # Both HG and dfmm export `spatial_dimension`; disambiguate.
        @test dfmm.spatial_dimension(c1) == 1
        @test c1.alphas == (1.0,)
        @test c1.betas  == (0.5,)
    end
end
