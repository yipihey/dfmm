# test_M3_3e_2_stochastic_native_vs_cache.jl
#
# Phase M3-3e-2 defensive cross-check: the native
# `inject_vg_noise_HG_native!` / `det_run_stochastic_HG_native!` path
# (M3-3e-2) must produce **byte-equal** state to running M1's
# `inject_vg_noise!` / `det_run_stochastic!` on a parallel `Mesh1D`.
#
# This is the direct test that the cache_mesh-driven Phase-8 path
# (retired in M3-3e-2) and the native path remain bit-for-bit
# equivalent on the stochastic injection sector.
#
# RNG sequencing gate: with identical seeds the per-cell `1:N` order
# of `rand_variance_gamma!` draws produces byte-identical `eta` arrays,
# and downstream per-cell scalar arithmetic preserves bit-exactness.
#
# Coverage:
#   Block 1 (1-step parity): N = 32 sinusoidal IC, project_kind = :none.
#   Block 2 (K = 10 steps):  N = 32 sinusoidal IC, project_kind = :none.
#   Block 3 (K = 10 steps + reanchor): N = 16 boundary-cell IC,
#                                       project_kind = :reanchor.
#   Block 4 (K = 10 steps + tau + q): N = 32, full-feature stack.

using Test
using Random: MersenneTwister
using dfmm
using dfmm: Mesh1D, n_segments, DetField,
            det_run!, det_step!,
            inject_vg_noise!, det_run_stochastic!,
            inject_vg_noise_HG!, det_run_stochastic_HG!,
            NoiseInjectionParams, InjectionDiagnostics,
            ProjectionStats,
            DetMeshHG_from_arrays, det_step_HG!, det_run_HG!,
            read_detfield

# Helper: build matched IC pair for M1 + HG.
function _build_pair(positions, velocities, αs, βs, ss; Δm, L_box,
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

# Helper: assert per-cell `(α, β, x, u, s, Pp, Q)` byte-equal.
function _assert_state_equal(mesh_HG, mesh_M1)
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

@testset "M3-3e-2 stochastic native vs cache_mesh: 1-step parity" begin
    # Block 1 — 1 inject step on a sinusoidal IC.
    N = 32
    L = 1.0
    Δx = L / N
    Δm_vec = fill(Δx, N)
    positions = [Δx * (j - 1) for j in 1:N]
    velocities = [0.3 * sinpi(2 * positions[j] / L) for j in 1:N]
    αs = fill(0.2, N)
    βs = zeros(Float64, N)
    s_uniform = log(0.5) + (2.0 / 3.0) * log(1.0)
    ss = fill(s_uniform, N)
    Pps = fill(0.5, N)
    mesh_M1, mesh_HG = _build_pair(positions, velocities, αs, βs, ss;
                                    Δm = Δm_vec, Pps = Pps, L_box = L)
    rng_M1 = MersenneTwister(7)
    rng_HG = MersenneTwister(7)
    params = NoiseInjectionParams(C_A = 0.3, C_B = 0.5,
                                   λ = 1.6, θ_factor = 1.0/1.6,
                                   project_kind = :none)
    dt = 5e-4
    inject_vg_noise!(mesh_M1, dt; params = params, rng = rng_M1)
    inject_vg_noise_HG!(mesh_HG, dt; params = params, rng = rng_HG)
    _assert_state_equal(mesh_HG, mesh_M1)
end

@testset "M3-3e-2 stochastic native vs cache_mesh: K = 10 steps" begin
    # Block 2 — det_run_stochastic, 10 steps, no projection, periodic.
    N = 32
    L = 1.0
    Δx = L / N
    Δm_vec = fill(Δx, N)
    positions = [Δx * (j - 1) for j in 1:N]
    velocities = [0.4 * sinpi(2 * positions[j] / L) for j in 1:N]
    αs = fill(0.2, N)
    βs = zeros(Float64, N)
    s_uniform = log(0.5) + (2.0 / 3.0) * log(1.0)
    ss = fill(s_uniform, N)
    Pps = fill(0.5, N)
    mesh_M1, mesh_HG = _build_pair(positions, velocities, αs, βs, ss;
                                    Δm = Δm_vec, Pps = Pps, L_box = L)
    rng_M1 = MersenneTwister(101)
    rng_HG = MersenneTwister(101)
    params = NoiseInjectionParams(C_A = 0.3, C_B = 0.5,
                                   λ = 1.5, θ_factor = 1.0/1.5,
                                   project_kind = :none)
    dt = 5e-4
    n_steps = 10
    det_run_stochastic!(mesh_M1, dt, n_steps; params = params, rng = rng_M1)
    det_run_stochastic_HG!(mesh_HG, dt, n_steps; params = params, rng = rng_HG)
    _assert_state_equal(mesh_HG, mesh_M1)
end

@testset "M3-3e-2 stochastic native vs cache_mesh: K = 10 + reanchor" begin
    # Block 3 — det_run_stochastic with project_kind = :reanchor and a
    # boundary-cell IC that triggers projection events.
    N = 16
    L = 1.0
    Δx = L / N
    Δm_vec = fill(Δx, N)
    positions = [Δx * (j - 1) for j in 1:N]
    velocities = [0.2 * sinpi(2 * positions[j] / L) for j in 1:N]
    αs = fill(0.2, N)
    βs = zeros(Float64, N)
    βs[8] = 0.30
    s_default = log(0.5) + (2.0/3.0) * log(1.0)
    ss = fill(s_default, N)
    ss[8] = log(0.05) + (2.0/3.0) * log(1.0)
    Pps = fill(0.5, N)
    mesh_M1, mesh_HG = _build_pair(positions, velocities, αs, βs, ss;
                                    Δm = Δm_vec, Pps = Pps, L_box = L)
    rng_M1 = MersenneTwister(2026)
    rng_HG = MersenneTwister(2026)
    params = NoiseInjectionParams(C_A = 0.05, C_B = 0.10,
                                   λ = 1.6, θ_factor = 1.0/1.6,
                                   project_kind = :reanchor,
                                   Mvv_floor = 1e-3,
                                   realizability_headroom = 1.05)
    ps_M1 = ProjectionStats()
    ps_HG = ProjectionStats()
    dt = 1e-3
    n_steps = 10
    det_run_stochastic!(mesh_M1, dt, n_steps; params = params,
                        rng = rng_M1, proj_stats = ps_M1)
    det_run_stochastic_HG!(mesh_HG, dt, n_steps; params = params,
                           rng = rng_HG, proj_stats = ps_HG)
    @test ps_M1.n_steps  == ps_HG.n_steps
    @test ps_M1.n_events == ps_HG.n_events
    @test ps_M1.n_floor_events == ps_HG.n_floor_events
    @test ps_M1.total_dE_inj == ps_HG.total_dE_inj
    _assert_state_equal(mesh_HG, mesh_M1)
end

@testset "M3-3e-2 stochastic native vs cache_mesh: K = 10 + τ + q" begin
    # Block 4 — full-feature: BGK τ relaxation + q-dissipation +
    # stochastic noise. Combines M3-3e-1's deterministic native step
    # with M3-3e-2's stochastic native injection.
    N = 32
    L = 1.0
    Δx = L / N
    Δm_vec = fill(Δx, N)
    positions = [Δx * (j - 1) for j in 1:N]
    velocities = [0.05 * sinpi(2 * positions[j] / L) for j in 1:N]
    αs = fill(1.0, N)
    βs = fill(0.05, N)
    ss = fill(0.0, N)
    Pps = fill(1.0, N)
    Qs  = [0.01 * cospi(2 * (j - 0.5) / N) for j in 1:N]
    mesh_M1, mesh_HG = _build_pair(positions, velocities, αs, βs, ss;
                                    Δm = Δm_vec, Pps = Pps, Qs = Qs,
                                    L_box = L)
    rng_M1 = MersenneTwister(31337)
    rng_HG = MersenneTwister(31337)
    params = NoiseInjectionParams(C_A = 0.2, C_B = 0.3,
                                   λ = 1.5, θ_factor = 1.0/1.5,
                                   project_kind = :none)
    dt = 5e-4
    n_steps = 10
    det_run_stochastic!(mesh_M1, dt, n_steps; params = params,
                        rng = rng_M1, tau = 0.1,
                        q_kind = :vNR_linear_quadratic)
    det_run_stochastic_HG!(mesh_HG, dt, n_steps; params = params,
                           rng = rng_HG, tau = 0.1,
                           q_kind = :vNR_linear_quadratic)
    _assert_state_equal(mesh_HG, mesh_M1)
end
