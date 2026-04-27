# test_M3_3e_4_realizability_native_vs_cache.jl
#
# Phase M3-3e-4 defensive cross-check: the native
# `realizability_project_HG!` (M3-3e-4) must produce **byte-equal**
# state to running M1's `realizability_project!` on a parallel
# `Mesh1D`. Per-cell projection has no inter-cell coupling, so the
# lift is mechanical and bit-equality should be trivial; this test
# captures that assumption explicitly so any future drift surfaces.
#
# Coverage:
#   Block 1 (single-call, boundary cell): N = 16 IC with one cell
#     forced near the realizability boundary. Per-cell `(s, Pp)` and
#     `ProjectionStats` byte-equal to M1.
#   Block 2 (single-call, multi-event): N = 32 IC with several cells
#     near the boundary at varying β to exercise the floor branch.
#   Block 3 (repeated calls, idempotent): 5 successive
#     `realizability_project_HG!` calls vs 5 calls of M1's; per-call
#     cumulative `ProjectionStats` and final state byte-equal.
#   Block 4 (Mvv_floor branch): IC with very small β driving the
#     `Mvv_target = Mvv_floor` branch (n_floor_events > 0).

using Test
using dfmm
using dfmm: Mesh1D, n_segments, DetField,
            realizability_project!, realizability_project_HG!,
            ProjectionStats, DetMeshHG_from_arrays

# Helper: build matched IC pair for M1 + HG.
function _build_pair_M3_3e_4(positions, velocities, αs, βs, ss; Δm, L_box,
                              Pps = nothing, Qs = nothing, bc = :periodic)
    periodic = bc == :periodic
    mesh_M1 = Mesh1D(positions, velocities, αs, βs, ss;
                     Δm = Δm, Pps = Pps, Qs = Qs, L_box = L_box,
                     periodic = periodic, bc = bc)
    mesh_HG = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                     Δm = Δm, Pps = Pps, Qs = Qs,
                                     L_box = L_box, bc = bc)
    return mesh_M1, mesh_HG
end

# Helper: per-cell `(α, β, x, u, s, Pp, Q)` byte-equal assertion.
function _assert_state_equal_M3_3e_4(mesh_HG, mesh_M1)
    N = n_segments(mesh_M1)
    fs = mesh_HG.fields
    for j in 1:N
        seg = mesh_M1.segments[j].state
        @test fs.x[j][1]     === seg.x
        @test fs.u[j][1]     === seg.u
        @test fs.alpha[j][1] === seg.α
        @test fs.beta[j][1]  === seg.β
        @test fs.s[j][1]     === seg.s
        @test fs.Pp[j][1]    === seg.Pp
        @test fs.Q[j][1]     === seg.Q
    end
end

function _assert_stats_equal(ps_HG, ps_M1)
    @test ps_HG.n_steps        == ps_M1.n_steps
    @test ps_HG.n_events       == ps_M1.n_events
    @test ps_HG.n_floor_events == ps_M1.n_floor_events
    @test ps_HG.total_dE_inj   == ps_M1.total_dE_inj
    @test ps_HG.Mvv_min_pre    == ps_M1.Mvv_min_pre
    @test ps_HG.Mvv_min_post   == ps_M1.Mvv_min_post
end

@testset "M3-3e-4 realizability native vs cache_mesh: single-call boundary cell" begin
    # Block 1 — single boundary cell triggers projection.
    N = 16
    L = 1.0
    Δx = L / N
    Δm_vec = fill(Δx, N)
    positions = [Δx * (j - 1) for j in 1:N]
    velocities = zeros(N)
    αs = fill(0.2, N)
    βs = zeros(Float64, N)
    βs[8] = 0.30
    s_default = log(0.5) + (2.0/3.0) * log(1.0)
    ss = fill(s_default, N)
    ss[8] = log(0.05) + (2.0/3.0) * log(1.0)
    Pps = fill(0.5, N)

    mesh_M1, mesh_HG = _build_pair_M3_3e_4(positions, velocities, αs, βs, ss;
                                            Δm = Δm_vec, Pps = Pps, L_box = L)
    ps_M1 = ProjectionStats()
    ps_HG = ProjectionStats()
    realizability_project!(mesh_M1; kind = :reanchor, headroom = 1.05,
                           Mvv_floor = 1e-3, pressure_floor = 1e-8,
                           stats = ps_M1)
    realizability_project_HG!(mesh_HG; kind = :reanchor, headroom = 1.05,
                              Mvv_floor = 1e-3, pressure_floor = 1e-8,
                              stats = ps_HG)
    _assert_state_equal_M3_3e_4(mesh_HG, mesh_M1)
    _assert_stats_equal(ps_HG, ps_M1)
    @test ps_HG.n_events >= 1
end

@testset "M3-3e-4 realizability native vs cache_mesh: multi-event" begin
    # Block 2 — N = 32 IC with several boundary-violating cells at
    # varying β. Ensures the projection-event loop touches multiple
    # cells, not just the corner case.
    N = 32
    L = 1.0
    Δx = L / N
    Δm_vec = fill(Δx, N)
    positions = [Δx * (j - 1) for j in 1:N]
    velocities = [0.1 * sinpi(2 * positions[j] / L) for j in 1:N]
    αs = fill(0.2, N)
    βs = zeros(Float64, N)
    # Multiple cells deliberately near/violating the cone.
    βs[5]  = 0.30
    βs[12] = 0.20
    βs[20] = 0.40
    βs[28] = 0.10
    s_default = log(0.5) + (2.0/3.0) * log(1.0)
    ss = fill(s_default, N)
    ss[5]  = log(0.05) + (2.0/3.0) * log(1.0)
    ss[12] = log(0.02) + (2.0/3.0) * log(1.0)
    ss[20] = log(0.08) + (2.0/3.0) * log(1.0)
    ss[28] = log(0.005) + (2.0/3.0) * log(1.0)
    Pps = fill(0.5, N)

    mesh_M1, mesh_HG = _build_pair_M3_3e_4(positions, velocities, αs, βs, ss;
                                            Δm = Δm_vec, Pps = Pps, L_box = L)
    ps_M1 = ProjectionStats()
    ps_HG = ProjectionStats()
    realizability_project!(mesh_M1; kind = :reanchor, headroom = 1.05,
                           Mvv_floor = 1e-3, pressure_floor = 1e-8,
                           stats = ps_M1)
    realizability_project_HG!(mesh_HG; kind = :reanchor, headroom = 1.05,
                              Mvv_floor = 1e-3, pressure_floor = 1e-8,
                              stats = ps_HG)
    _assert_state_equal_M3_3e_4(mesh_HG, mesh_M1)
    _assert_stats_equal(ps_HG, ps_M1)
    @test ps_HG.n_events >= 2
end

@testset "M3-3e-4 realizability native vs cache_mesh: repeated calls idempotent" begin
    # Block 3 — call the projection 5× successively. After the first
    # call all cells are inside the cone; subsequent calls should be
    # no-ops at byte-equality. ProjectionStats accumulates.
    N = 16
    L = 1.0
    Δx = L / N
    Δm_vec = fill(Δx, N)
    positions = [Δx * (j - 1) for j in 1:N]
    velocities = zeros(N)
    αs = fill(0.2, N)
    βs = zeros(Float64, N)
    βs[8] = 0.30
    s_default = log(0.5) + (2.0/3.0) * log(1.0)
    ss = fill(s_default, N)
    ss[8] = log(0.10) + (2.0/3.0) * log(1.0)
    Pps = fill(0.5, N)

    mesh_M1, mesh_HG = _build_pair_M3_3e_4(positions, velocities, αs, βs, ss;
                                            Δm = Δm_vec, Pps = Pps, L_box = L)
    ps_M1 = ProjectionStats()
    ps_HG = ProjectionStats()
    for _ in 1:5
        realizability_project!(mesh_M1; kind = :reanchor, headroom = 1.05,
                               Mvv_floor = 1e-3, pressure_floor = 1e-8,
                               stats = ps_M1)
        realizability_project_HG!(mesh_HG; kind = :reanchor, headroom = 1.05,
                                  Mvv_floor = 1e-3, pressure_floor = 1e-8,
                                  stats = ps_HG)
    end
    _assert_state_equal_M3_3e_4(mesh_HG, mesh_M1)
    _assert_stats_equal(ps_HG, ps_M1)
    @test ps_HG.n_steps == 5
end

@testset "M3-3e-4 realizability native vs cache_mesh: Mvv_floor branch" begin
    # Block 4 — IC with very small β where the Mvv_floor (absolute
    # safety floor) becomes the binding constraint, not the relative
    # headroom·β². n_floor_events > 0 on this configuration.
    N = 16
    L = 1.0
    Δx = L / N
    Δm_vec = fill(Δx, N)
    positions = [Δx * (j - 1) for j in 1:N]
    velocities = zeros(N)
    αs = fill(0.2, N)
    βs = fill(1e-4, N)   # very small β so Mvv_floor binds
    s_default = log(1e-5) + (2.0/3.0) * log(1.0)  # M_vv well below floor
    ss = fill(s_default, N)
    Pps = fill(0.5, N)

    mesh_M1, mesh_HG = _build_pair_M3_3e_4(positions, velocities, αs, βs, ss;
                                            Δm = Δm_vec, Pps = Pps, L_box = L)
    ps_M1 = ProjectionStats()
    ps_HG = ProjectionStats()
    realizability_project!(mesh_M1; kind = :reanchor, headroom = 1.05,
                           Mvv_floor = 1e-2, pressure_floor = 1e-8,
                           stats = ps_M1)
    realizability_project_HG!(mesh_HG; kind = :reanchor, headroom = 1.05,
                              Mvv_floor = 1e-2, pressure_floor = 1e-8,
                              stats = ps_HG)
    _assert_state_equal_M3_3e_4(mesh_HG, mesh_M1)
    _assert_stats_equal(ps_HG, ps_M1)
    @test ps_HG.n_floor_events >= 1
end

@testset "M3-3e-4 realizability native vs cache_mesh: :none fast-path" begin
    # Block 5 — kind = :none: only `n_steps` should be incremented;
    # all per-cell state untouched and byte-equal.
    N = 16
    L = 1.0
    Δx = L / N
    Δm_vec = fill(Δx, N)
    positions = [Δx * (j - 1) for j in 1:N]
    velocities = [0.1 * sinpi(2 * positions[j] / L) for j in 1:N]
    αs = fill(0.2, N)
    βs = fill(0.05, N)
    s_default = log(0.5) + (2.0/3.0) * log(1.0)
    ss = fill(s_default, N)
    Pps = fill(0.5, N)

    mesh_M1, mesh_HG = _build_pair_M3_3e_4(positions, velocities, αs, βs, ss;
                                            Δm = Δm_vec, Pps = Pps, L_box = L)
    ps_M1 = ProjectionStats()
    ps_HG = ProjectionStats()
    for _ in 1:3
        realizability_project!(mesh_M1; kind = :none, stats = ps_M1)
        realizability_project_HG!(mesh_HG; kind = :none, stats = ps_HG)
    end
    _assert_state_equal_M3_3e_4(mesh_HG, mesh_M1)
    _assert_stats_equal(ps_HG, ps_M1)
    @test ps_HG.n_steps == 3
    @test ps_HG.n_events == 0
end
