# test_M3_2_M2_3_realizability_HG.jl
#
# Phase M3-2 Block 4b — M2-3 realizability projection on the HG
# substrate. Mirrors `test_phase_M2_3_realizability.jl` for the
# core invariants. The HG-side `realizability_project_HG!` delegates
# to M1's `realizability_project!` through the cache mesh; the
# bit-exact-parity contract is therefore automatic.
#
# Coverage:
#   1. Synthetic boundary cell: pre-projection violates realizability;
#      post-projection M_vv ≥ headroom · β² for every cell.
#   2. Mass + momentum exactness across projection events.
#   3. Bit-equality with M1 path on a smooth IC (no-event regime).
#   4. Realizability invariant after projection.
#   5. ProjectionStats round-trip.

using Test
using Random: MersenneTwister
using dfmm
using dfmm: total_mass_HG, total_momentum_HG,
            DetMeshHG_from_arrays, n_segments,
            realizability_project_HG!, ProjectionStats,
            NoiseInjectionParams, det_run_stochastic!,
            det_run_stochastic_HG!,
            Mesh1D, Mvv, segment_density, read_detfield
import dfmm: reset!

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

"Build a mesh with a single cell forced near the realizability boundary."
function _build_boundary_mesh_HG(; N::Int = 16, L::Float64 = 1.0,
                                 ρ0::Float64 = 1.0, j_target::Int = 8,
                                 β_big::Float64 = 0.3,
                                 Mvv_target::Float64 = 0.10)
    Δx = L / N
    Δm_vec = fill(ρ0 * Δx, N)
    positions = collect(0.0:Δx:(L - Δx))
    velocities = zeros(N)
    αs = fill(0.2, N)
    βs = zeros(N)
    βs[j_target] = β_big
    J0 = 1.0 / ρ0
    s_default = log(0.5) + (2.0/3.0) * log(J0)
    ss = fill(s_default, N)
    ss[j_target] = log(Mvv_target) + (2.0/3.0) * log(J0)
    Pps = fill(0.5, N)
    return DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                  Δm = Δm_vec, Pps = Pps, L_box = L,
                                  bc = :periodic)
end

# ─────────────────────────────────────────────────────────────────────
# 1. Synthetic boundary cell triggers projection
# ─────────────────────────────────────────────────────────────────────

@testset "M3-2 M2-3 (HG): synthetic boundary cell triggers projection" begin
    mesh = _build_boundary_mesh_HG(N = 16, β_big = 0.30, Mvv_target = 0.05)
    j = 8
    # Pre-projection: target cell violates realizability.
    sj = read_detfield(mesh.fields, j)
    ρ_j = mesh.Δm[j] / (1.0 / 16)   # uniform spacing
    Mvv_pre = Mvv(1.0/ρ_j, sj.s)
    @test Mvv_pre < sj.β^2
    @test Mvv_pre < 1.05 * sj.β^2

    stats = ProjectionStats()
    realizability_project_HG!(mesh; kind = :reanchor,
                              headroom = 1.05, Mvv_floor = 1e-3,
                              pressure_floor = 1e-8, stats = stats)

    # Post-projection: every cell satisfies M_vv ≥ headroom · β².
    for k in 1:length(mesh.Δm)
        ρ_k = mesh.Δm[k] / (1.0 / 16)
        sk = read_detfield(mesh.fields, k)
        Mvv_k = Mvv(1.0/ρ_k, sk.s)
        @test Mvv_k >= 1.05 * sk.β^2 * (1 - 1e-12)
    end
    @test stats.n_events >= 1
    @test stats.n_steps == 1
end

# ─────────────────────────────────────────────────────────────────────
# 2. Mass + momentum exactness
# ─────────────────────────────────────────────────────────────────────

@testset "M3-2 M2-3 (HG): projection conserves mass and momentum exactly" begin
    mesh = _build_boundary_mesh_HG(N = 16, β_big = 0.30, Mvv_target = 0.10)
    M0 = total_mass_HG(mesh)
    p0 = total_momentum_HG(mesh)

    stats = ProjectionStats()
    for _ in 1:5
        realizability_project_HG!(mesh; kind = :reanchor,
                                   headroom = 1.05, Mvv_floor = 1e-3,
                                   pressure_floor = 1e-8, stats = stats)
    end

    @test total_mass_HG(mesh) == M0
    @test total_momentum_HG(mesh) == p0
    @test stats.n_steps == 5
end

# ─────────────────────────────────────────────────────────────────────
# 3. Bit-equality with M1 path on smooth IC (no-event regime)
# ─────────────────────────────────────────────────────────────────────

@testset "M3-2 M2-3 (HG): bit-equality with M1 path (smooth IC)" begin
    N = 16
    L = 1.0
    Δx = L / N
    Δm_vec = fill(Δx, N)
    positions = collect(0.0:Δx:(L - Δx))
    velocities = [0.2 * sin(2π * x / L) for x in positions]
    αs = fill(0.2, N)
    βs = zeros(N)
    J0 = 1.0
    s_uniform = log(0.5) + (2.0/3.0) * log(J0)
    ss = fill(s_uniform, N)
    Pps = fill(0.5, N)

    mesh_M1 = Mesh1D(positions, velocities, αs, βs, ss;
                     Δm = Δm_vec, Pps = Pps, L_box = L, periodic = true)
    mesh_HG = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                     Δm = Δm_vec, Pps = Pps, L_box = L,
                                     bc = :periodic)
    dt = 1e-3
    n_steps = 8

    p_reanchor = NoiseInjectionParams(C_A = 0.05, C_B = 0.10, λ = 1.6,
                                      θ_factor = 1.0/1.6,
                                      project_kind = :reanchor,
                                      Mvv_floor = 1e-3,
                                      realizability_headroom = 1.05)

    rng_M1 = MersenneTwister(123)
    rng_HG = MersenneTwister(123)
    ps_M1 = ProjectionStats()
    ps_HG = ProjectionStats()
    det_run_stochastic!(mesh_M1, dt, n_steps; params = p_reanchor,
                        rng = rng_M1, proj_stats = ps_M1)
    det_run_stochastic_HG!(mesh_HG, dt, n_steps; params = p_reanchor,
                           rng = rng_HG, proj_stats = ps_HG)

    @test ps_M1.n_events == ps_HG.n_events
    @test ps_M1.n_steps  == ps_HG.n_steps

    for j in 1:N
        sM = mesh_M1.segments[j].state
        sH = read_detfield(mesh_HG.fields, j)
        @test sM.x  == sH.x
        @test sM.u  == sH.u
        @test sM.α  == sH.α
        @test sM.β  == sH.β
        @test sM.s  == sH.s
        @test sM.Pp == sH.Pp
    end
end

# ─────────────────────────────────────────────────────────────────────
# 4. Realizability invariant
# ─────────────────────────────────────────────────────────────────────

@testset "M3-2 M2-3 (HG): post-projection realizability invariant" begin
    mesh = _build_boundary_mesh_HG(N = 16, β_big = 0.30, Mvv_target = 0.05)
    stats = ProjectionStats()
    realizability_project_HG!(mesh; kind = :reanchor,
                              headroom = 1.05, Mvv_floor = 1e-3,
                              pressure_floor = 1e-8, stats = stats)

    for j in 1:length(mesh.Δm)
        ρ_j = mesh.Δm[j] / (1.0 / 16)
        sj = read_detfield(mesh.fields, j)
        Mvv_j = Mvv(1.0/ρ_j, sj.s)
        @test Mvv_j >= 1e-3 * (1 - 1e-12)
        @test Mvv_j >= 1.05 * sj.β^2 * (1 - 1e-12)
    end
end

# ─────────────────────────────────────────────────────────────────────
# 5. ProjectionStats round-trip
# ─────────────────────────────────────────────────────────────────────

@testset "M3-2 M2-3 (HG): ProjectionStats round-trip" begin
    stats = ProjectionStats()
    @test stats.n_steps == 0
    @test stats.n_events == 0

    mesh = _build_boundary_mesh_HG(N = 16, β_big = 0.30, Mvv_target = 0.05)
    for _ in 1:3
        realizability_project_HG!(mesh; kind = :reanchor,
                                   headroom = 1.05, Mvv_floor = 1e-3,
                                   pressure_floor = 1e-8, stats = stats)
    end
    @test stats.n_steps == 3
    @test stats.n_events >= 1

    reset!(stats)
    @test stats.n_steps == 0
    @test stats.n_events == 0
    @test stats.total_dE_inj == 0.0
end
