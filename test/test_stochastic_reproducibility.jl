# test_stochastic_reproducibility.jl
#
# Reproducibility regression: a stochastic Phase-8 run with a
# fixed-seed `MersenneTwister` should produce bit-identical state
# across reruns. This is a defensive guard for refactors that
# accidentally introduce a non-deterministic code path (e.g. a
# `Dict` traversal, a `rand()` call without an explicit RNG, or
# parallel reductions in non-deterministic order).
#
# Two replicas of the same IC, same dt, same params, same seeded RNG:
#   * `inject_vg_noise!` produces bit-equal δ(ρu) diagnostics.
#   * `det_run_stochastic!` over 10 steps produces bit-equal final
#     mesh state.
#
# This test runs in O(seconds) on a 32-cell wave-pool surrogate; it
# guards the property that all stochastic paths take their RNG state
# from the explicit `rng` argument (no global RNG dependency).

using Test
using dfmm
using Random

# Build a small wave-pool-style mesh: 32 cells, periodic, isotropic IC,
# small density perturbation to seed the wave field.
function _build_wavepool(N::Int, L::Float64; pert::Float64 = 0.0,
                         seed::Int = 0)
    rng = MersenneTwister(seed)
    Δm = fill(L / N, N)
    positions = [(L / N) * (j - 1) for j in 1:N]
    velocities = pert == 0 ? zeros(N) : pert * randn(rng, N)
    αs = fill(0.5, N)
    βs = zeros(N)
    ss = zeros(N)
    return Mesh1D(positions, velocities, αs, βs, ss;
                  Δm = Δm, L_box = L, periodic = true)
end

@testset "reproducibility: inject_vg_noise! bit-equal across reseeded reruns" begin
    N, L = 32, 1.0
    dt = 1e-3
    params = NoiseInjectionParams(project_kind = :none)
    seed = 314159

    mesh_a = _build_wavepool(N, L; pert = 1e-3, seed = 1)
    mesh_b = _build_wavepool(N, L; pert = 1e-3, seed = 1)

    rng_a = MersenneTwister(seed)
    rng_b = MersenneTwister(seed)
    _, diag_a = inject_vg_noise!(mesh_a, dt; params = params, rng = rng_a)
    _, diag_b = inject_vg_noise!(mesh_b, dt; params = params, rng = rng_b)

    # Diagnostics bit-equal.
    @test diag_a.eta == diag_b.eta
    @test diag_a.delta_rhou == diag_b.delta_rhou
    @test diag_a.delta_rhou_drift == diag_b.delta_rhou_drift
    @test diag_a.delta_rhou_noise == diag_b.delta_rhou_noise
    @test diag_a.divu == diag_b.divu

    # Mesh state bit-equal.
    for j in 1:N
        @test mesh_a.segments[j].state.u  === mesh_b.segments[j].state.u
        @test mesh_a.segments[j].state.s  === mesh_b.segments[j].state.s
        @test mesh_a.segments[j].state.Pp === mesh_b.segments[j].state.Pp
    end
    # p_half bit-equal.
    @test mesh_a.p_half == mesh_b.p_half
end

@testset "reproducibility: 10 steps of inject_vg_noise! are bit-equal" begin
    # Multi-step probe: sequencing of RNG calls and (RNG-independent)
    # state evolution should be deterministic.
    N, L = 32, 1.0
    dt = 1e-3
    params = NoiseInjectionParams(project_kind = :none)
    seed = 12345

    mesh_a = _build_wavepool(N, L; pert = 1e-3, seed = 7)
    mesh_b = _build_wavepool(N, L; pert = 1e-3, seed = 7)

    rng_a = MersenneTwister(seed)
    rng_b = MersenneTwister(seed)
    for _ in 1:10
        inject_vg_noise!(mesh_a, dt; params = params, rng = rng_a)
        inject_vg_noise!(mesh_b, dt; params = params, rng = rng_b)
    end

    for j in 1:N
        @test mesh_a.segments[j].state.x === mesh_b.segments[j].state.x
        @test mesh_a.segments[j].state.u === mesh_b.segments[j].state.u
        @test mesh_a.segments[j].state.α === mesh_b.segments[j].state.α
        @test mesh_a.segments[j].state.β === mesh_b.segments[j].state.β
        @test mesh_a.segments[j].state.s === mesh_b.segments[j].state.s
        @test mesh_a.segments[j].state.Pp === mesh_b.segments[j].state.Pp
        @test mesh_a.segments[j].state.Q === mesh_b.segments[j].state.Q
    end
end

@testset "reproducibility: different seeds produce different draws" begin
    # Sanity: the test above should fail if both reruns *always* gave
    # identical results (e.g. if a global RNG was being used). Pin
    # that distinct seeds → distinct results.
    N, L = 32, 1.0
    dt = 1e-3
    params = NoiseInjectionParams(project_kind = :none)

    mesh_a = _build_wavepool(N, L; pert = 1e-3, seed = 7)
    mesh_b = _build_wavepool(N, L; pert = 1e-3, seed = 7)

    rng_a = MersenneTwister(1)
    rng_b = MersenneTwister(2)
    _, diag_a = inject_vg_noise!(mesh_a, dt; params = params, rng = rng_a)
    _, diag_b = inject_vg_noise!(mesh_b, dt; params = params, rng = rng_b)
    # Different seeds ⇒ at least one entry of η differs.
    @test diag_a.eta != diag_b.eta
end

@testset "reproducibility: rand_variance_gamma! is RNG-deterministic" begin
    # Cross-check that the underlying VG sampler is itself reproducible.
    # This pins the contract that `inject_vg_noise!` builds on.
    out_a = zeros(64)
    out_b = zeros(64)
    rng_a = MersenneTwister(2024)
    rng_b = MersenneTwister(2024)
    rand_variance_gamma!(rng_a, out_a, 6.667, 0.15)
    rand_variance_gamma!(rng_b, out_b, 6.667, 0.15)
    @test out_a == out_b
end
