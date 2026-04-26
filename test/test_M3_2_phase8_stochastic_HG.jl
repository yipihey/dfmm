# test_M3_2_phase8_stochastic_HG.jl
#
# Phase M3-2 Block 2 — variance-gamma stochastic injection on the HG
# substrate. Mirrors `test_phase8_stochastic_injection.jl` for the
# subset of tests that exercise the stochastic-injection wrappers.
# The HG path delegates to M1's `inject_vg_noise!` /
# `det_run_stochastic!` through the cache mesh, so identical RNG seed
# and params yield byte-identical state.
#
# Coverage:
#   1. Bit-exact parity vs M1: identical seed → identical post-step
#      state for every cell over a 5-step run with C_A, C_B != 0.
#   2. Zero-noise gate: with C_A = C_B = 0 the stochastic path equals
#      `det_run_HG!` to round-off (same gate the M1 Phase 8 test 2 uses).
#   3. Mass exactness across stochastic injection events on the HG mesh.
#   4. Diagnostics-vector parity: per-cell `delta_rhou`, `eta`, `divu`
#      values match between M1 and HG paths to 0.0 absolute.

using Test
using Random: MersenneTwister
using dfmm
using dfmm: Mesh1D, n_segments, segment_density,
            DetField, total_mass, total_momentum,
            det_run!, det_step!,
            inject_vg_noise!, det_run_stochastic!,
            inject_vg_noise_HG!, det_run_stochastic_HG!,
            NoiseInjectionParams, InjectionDiagnostics,
            DetMeshHG_from_arrays, det_step_HG!, det_run_HG!,
            total_mass_HG, read_detfield

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

function _build_sinusoidal_pair(; N::Int = 32, L::Float64 = 1.0,
                                 A::Float64 = 0.4,
                                 ρ0::Float64 = 1.0,
                                 P0::Float64 = 0.5,
                                 σx::Float64 = 0.2)
    Δx = L / N
    Δm_vec = fill(ρ0 * Δx, N)
    positions = collect(0.0:Δx:(L - Δx))
    velocities = [A * sin(2π * x / L) for x in positions]
    αs = fill(σx, N)
    βs = zeros(Float64, N)
    J0 = 1.0 / ρ0
    s_uniform = log(P0 / ρ0) + (2.0 / 3.0) * log(J0)
    ss = fill(s_uniform, N)
    Pps = fill(P0, N)
    mesh_M1 = Mesh1D(positions, velocities, αs, βs, ss; Δm = Δm_vec,
                     Pps = Pps, L_box = L, periodic = true)
    mesh_HG = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                     Δm = Δm_vec, Pps = Pps, L_box = L,
                                     bc = :periodic)
    return mesh_M1, mesh_HG
end

# ─────────────────────────────────────────────────────────────────────
# 1. Bit-exact parity vs M1 with identical seed
# ─────────────────────────────────────────────────────────────────────

@testset "M3-2 Phase 8 (HG): bit-exact parity vs M1 (stochastic noise)" begin
    mesh_M1, mesh_HG = _build_sinusoidal_pair(N = 32, A = 0.4)
    rng_M1 = MersenneTwister(42)
    rng_HG = MersenneTwister(42)
    params = NoiseInjectionParams(C_A = 0.3, C_B = 0.5,
                                   λ = 1.5, θ_factor = 1.0/1.5,
                                   project_kind = :none)
    dt = 5e-4
    n_steps = 5

    det_run_stochastic!(mesh_M1, dt, n_steps; params = params, rng = rng_M1)
    det_run_stochastic_HG!(mesh_HG, dt, n_steps; params = params, rng = rng_HG)

    max_err = 0.0
    for j in 1:n_segments(mesh_M1)
        sM = mesh_M1.segments[j].state
        sH = read_detfield(mesh_HG.fields, j)
        max_err = max(max_err,
                      abs(sM.x  - sH.x),
                      abs(sM.u  - sH.u),
                      abs(sM.α  - sH.α),
                      abs(sM.β  - sH.β),
                      abs(sM.s  - sH.s),
                      abs(sM.Pp - sH.Pp),
                      abs(sM.Q  - sH.Q))
    end
    @test max_err < 5e-13
    @test max_err == 0.0
end

# ─────────────────────────────────────────────────────────────────────
# 2. Zero-noise reduces to det_run_HG!
# ─────────────────────────────────────────────────────────────────────

@testset "M3-2 Phase 8 (HG): zero-noise reduces to det_run_HG!" begin
    mesh_HG_a = _build_sinusoidal_pair(N = 16, A = 0.2)[2]
    mesh_HG_b = _build_sinusoidal_pair(N = 16, A = 0.2)[2]
    dt = 1e-3
    n = 5
    rng = MersenneTwister(0)

    det_run_HG!(mesh_HG_a, dt, n)
    p0 = NoiseInjectionParams(C_A = 0.0, C_B = 0.0,
                               λ = 1.6, θ_factor = 1.0/1.6,
                               project_kind = :none)
    det_run_stochastic_HG!(mesh_HG_b, dt, n; params = p0, rng = rng)

    N = length(mesh_HG_a.Δm)
    for j in 1:N
        sa = read_detfield(mesh_HG_a.fields, j)
        sb = read_detfield(mesh_HG_b.fields, j)
        @test isapprox(sa.x, sb.x; atol = 1e-12)
        @test isapprox(sa.u, sb.u; atol = 1e-12)
        @test isapprox(sa.α, sb.α; atol = 1e-12)
        @test isapprox(sa.β, sb.β; atol = 1e-12)
        @test isapprox(sa.s, sb.s; atol = 1e-12)
        @test isapprox(sa.Pp, sb.Pp; atol = 1e-12)
    end
end

# ─────────────────────────────────────────────────────────────────────
# 3. Mass exactness
# ─────────────────────────────────────────────────────────────────────

@testset "M3-2 Phase 8 (HG): mass exact under stochastic injection" begin
    _, mesh_HG = _build_sinusoidal_pair(N = 32, A = 0.4)
    M0 = total_mass_HG(mesh_HG)
    rng = MersenneTwister(7)
    params = NoiseInjectionParams(C_A = 0.3, C_B = 0.5,
                                   λ = 1.5, θ_factor = 1.0/1.5,
                                   project_kind = :none)
    dt = 5e-4
    n_steps = 20
    det_run_stochastic_HG!(mesh_HG, dt, n_steps; params = params, rng = rng)
    @test total_mass_HG(mesh_HG) == M0  # bit-exact (Δm is a label)
end

# ─────────────────────────────────────────────────────────────────────
# 4. Per-step diagnostics parity (delta_rhou, eta, divu)
# ─────────────────────────────────────────────────────────────────────

@testset "M3-2 Phase 8 (HG): inject_vg_noise_HG! diagnostics parity" begin
    mesh_M1, mesh_HG = _build_sinusoidal_pair(N = 16, A = 0.3)
    params = NoiseInjectionParams(C_A = 0.3, C_B = 0.5,
                                   λ = 1.6, θ_factor = 1.0/1.6,
                                   project_kind = :none)
    rng_M1 = MersenneTwister(11)
    rng_HG = MersenneTwister(11)
    diag_M1 = InjectionDiagnostics(n_segments(mesh_M1))
    diag_HG = InjectionDiagnostics(length(mesh_HG.Δm))
    dt = 1e-3
    inject_vg_noise!(mesh_M1, dt; params = params, rng = rng_M1, diag = diag_M1)
    inject_vg_noise_HG!(mesh_HG, dt; params = params, rng = rng_HG, diag = diag_HG)

    # Diagnostics: per-cell δ(ρu), eta, divu match bit-for-bit.
    for j in 1:n_segments(mesh_M1)
        @test diag_M1.divu[j]              == diag_HG.divu[j]
        @test diag_M1.eta[j]               == diag_HG.eta[j]
        @test diag_M1.delta_rhou[j]        == diag_HG.delta_rhou[j]
        @test diag_M1.delta_rhou_drift[j]  == diag_HG.delta_rhou_drift[j]
        @test diag_M1.delta_rhou_noise[j]  == diag_HG.delta_rhou_noise[j]
        @test diag_M1.delta_KE_vol[j]      == diag_HG.delta_KE_vol[j]
        @test diag_M1.compressive[j]       == diag_HG.compressive[j]
    end

    # Per-cell state matches.
    for j in 1:n_segments(mesh_M1)
        sM = mesh_M1.segments[j].state
        sH = read_detfield(mesh_HG.fields, j)
        @test sM.u == sH.u
        @test sM.s == sH.s
        @test sM.Pp == sH.Pp
    end
end
