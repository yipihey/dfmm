# test_M3_1_phase5b_qnone_bit_equal.jl
#
# Phase M3-1 Phase-5b sub-phase, block 1: bit-equality of `q_kind =
# :none` on the HG path with the bare Phase-5 path. Mirrors M1's
# `test_phase5b_artificial_viscosity.jl` block 2.
#
# Coverage assertions (~96 element-wise comparisons):
#   (a) Two HG meshes evolved identically with default kwargs vs
#       explicit `q_kind = :none` produce byte-identical state.
#   (b) HG `q_kind = :none` matches M1 `q_kind = :none` byte-identically.
#   (c) HG `q_kind = :none` matches M1 (no q_kind) byte-identically
#       (default kwarg pass-through is correct).

using Test
using Random
using dfmm

@testset "M3-1 Phase-5b (HG): q_kind=:none bit-equality" begin
    Random.seed!(20260425)
    N = 8
    L_box = 1.0
    dx = L_box / N
    Γ = 5.0 / 3.0
    positions = collect((0:N-1) .* dx)
    velocities = 0.05 .* sin.(2π .* positions ./ L_box)
    αs = fill(0.02, N)
    βs = zeros(N)
    ss = zeros(N)
    Δm = fill(1.0 / N, N)
    Pps = fill(1.0, N)

    # Two HG meshes, two M1 meshes — four trajectories total.
    mesh_HG_def = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                        Δm = Δm, Pps = Pps, L_box = L_box,
                                        bc = :periodic)
    mesh_HG_qn  = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                        Δm = Δm, Pps = Pps, L_box = L_box,
                                        bc = :periodic)
    mesh_M1_def = dfmm.Mesh1D(positions, velocities, αs, βs, ss;
                              Δm = Δm, Pps = Pps, L_box = L_box,
                              periodic = true)
    mesh_M1_qn  = dfmm.Mesh1D(positions, velocities, αs, βs, ss;
                              Δm = Δm, Pps = Pps, L_box = L_box,
                              periodic = true)

    dt = 1e-3
    # Two-step run.
    det_step_HG!(mesh_HG_def, dt; tau = 1e-3)
    det_step_HG!(mesh_HG_def, dt; tau = 1e-3)
    det_step_HG!(mesh_HG_qn,  dt; tau = 1e-3, q_kind = :none)
    det_step_HG!(mesh_HG_qn,  dt; tau = 1e-3, q_kind = :none)
    dfmm.det_step!(mesh_M1_def, dt; tau = 1e-3)
    dfmm.det_step!(mesh_M1_def, dt; tau = 1e-3)
    dfmm.det_step!(mesh_M1_qn,  dt; tau = 1e-3, q_kind = :none)
    dfmm.det_step!(mesh_M1_qn,  dt; tau = 1e-3, q_kind = :none)

    # (a) HG default-kwargs == HG explicit q_kind=:none, bit-by-bit.
    for j in 1:N
        sa = read_detfield(mesh_HG_def.fields, j)
        sb = read_detfield(mesh_HG_qn.fields,  j)
        @test sa.x  == sb.x
        @test sa.u  == sb.u
        @test sa.α  == sb.α
        @test sa.β  == sb.β
        @test sa.s  == sb.s
        @test sa.Pp == sb.Pp
    end
    # (b) HG q_kind=:none == M1 q_kind=:none, bit-by-bit.
    for j in 1:N
        sH = read_detfield(mesh_HG_qn.fields, j)
        sM = mesh_M1_qn.segments[j].state
        @test sH.x  == sM.x
        @test sH.u  == sM.u
        @test sH.α  == sM.α
        @test sH.β  == sM.β
        @test sH.s  == sM.s
        @test sH.Pp == sM.Pp
    end
    # (c) HG default-kwargs == M1 default-kwargs, bit-by-bit.
    for j in 1:N
        sH = read_detfield(mesh_HG_def.fields, j)
        sM = mesh_M1_def.segments[j].state
        @test sH.x  == sM.x
        @test sH.u  == sM.u
        @test sH.α  == sM.α
        @test sH.β  == sM.β
        @test sH.s  == sM.s
        @test sH.Pp == sM.Pp
    end
end
