# test_M3_2b_swap6_sparsity_HG.jl
#
# M3-2b Swap 6 — HG-aware Newton-Jacobian sparsity helper.
#
# Verifies that `det_jac_sparsity_HG(mesh::SimplicialMesh{1, T})`
# produces a `SparseMatrixCSC` byte-equal to M1's hand-rolled
# `det_jac_sparsity(N)` for the periodic 1D case. This is the parity
# gate for Swap 6 (the M3-3 native EL residual will consume the
# HG-aware path; in M3-2b the cache-mesh-driven Newton solve continues
# to use M1's helper, but the HG hook must already be byte-equal).

using Test
using dfmm
using SparseArrays: SparseMatrixCSC, nnz, findnz

@testset "M3-2b Swap 6: HG sparsity matches M1 on N=16 periodic mesh" begin
    N = 16
    positions = collect((0:N-1) .* (1.0 / N))
    velocities = zeros(N)
    αs = fill(0.02, N)
    βs = zeros(N)
    ss = zeros(N)
    Δm = fill(1.0 / N, N)

    mesh_HG = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                     Δm = Δm, L_box = 1.0,
                                     bc = :periodic)

    # M1 hand-rolled sparsity vs HG-aware sparsity.
    sp_M1 = dfmm.det_jac_sparsity(N)
    sp_HG = det_jac_sparsity_HG(mesh_HG)

    # Same shape.
    @test size(sp_M1) == size(sp_HG)
    @test size(sp_M1) == (4N, 4N)

    # Same nnz count.
    @test nnz(sp_M1) == nnz(sp_HG)

    # Same nonzero positions, byte-for-byte.
    rM1, cM1, vM1 = findnz(sp_M1)
    rHG, cHG, vHG = findnz(sp_HG)
    @test rM1 == rHG
    @test cM1 == cHG

    # Materialise to dense Bool arrays and check exact equality.
    @test Matrix(sp_M1 .!= 0) == Matrix(sp_HG .!= 0)
end

@testset "M3-2b Swap 6: HG sparsity tridiagonal-block on N=4" begin
    # Smaller N to exercise the periodic-wrap branch independently.
    N = 4
    positions = collect((0:N-1) .* (1.0 / N))
    velocities = zeros(N)
    αs = fill(0.02, N)
    βs = zeros(N)
    ss = zeros(N)
    Δm = fill(1.0 / N, N)
    mesh_HG = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                     Δm = Δm, L_box = 1.0,
                                     bc = :periodic)
    sp_M1 = dfmm.det_jac_sparsity(N)
    sp_HG = det_jac_sparsity_HG(mesh_HG)
    @test Matrix(sp_M1 .!= 0) == Matrix(sp_HG .!= 0)
end
