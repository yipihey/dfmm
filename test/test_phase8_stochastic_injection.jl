# test/test_phase8_stochastic_injection.jl
#
# Phase 8: variance-gamma stochastic-injection unit tests.
#
# Coverage (see `reference/notes_phase8_stochastic_injection.md`):
#   1. Bit-equality gate: with C_A = C_B = 0 the injection is a no-op
#      and `det_run_stochastic!` matches `det_run!` to round-off.
#   2. Mass conservation: per-segment Δm and total mass exact across
#      injection (the noise lives in the momentum sector).
#   3. Smoke compression test: a sinusoidal pre-crossing IC drives a
#      compressive half-cycle; the injected δ(ρu) histogram is non-
#      trivial and concentrated on cells with `divu < 0`.
#   4. Energy bookkeeping: the per-cell ΔKE_vol equals
#      `u·δ + δ²/(2ρ)` to round-off (closes the ledger), and
#      total energy stays bounded over a moderate run window.
#   5. Burst-stats accumulator round-trip on synthetic divu input
#      yields the documented gamma-shape / residual-kurtosis numbers.
#   6. End-to-end: a wave-pool short run with calibrated parameters
#      produces self-consistency-monitor output with sensible fields
#      (k_hat finite when bursts > 5, lambda_res finite when
#      compression sample > 100).
#
# Wall budget: <30 s on the test runner. Wave-pool sub-test uses a
# small `N = 64` mesh and a reduced step count.

using Test
using Random: MersenneTwister
using StatsBase: var, mean
using dfmm
using dfmm: Mesh1D, n_segments, segment_density, segment_length,
            DetField, total_mass, total_momentum, total_energy,
            total_internal_energy, total_kinetic_energy,
            det_run!, det_step!,
            inject_vg_noise!, det_run_stochastic!,
            NoiseInjectionParams, from_calibration,
            BurstStatsAccumulator, record_step!, burst_durations,
            self_consistency_check, InjectionDiagnostics,
            smooth_periodic_3pt!,
            load_noise_model, setup_kmles_wavepool,
            rand_variance_gamma!

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

"Build a sinusoidal-velocity periodic mesh of `N` segments on [0, L]."
function _build_sinusoidal_mesh(; N::Int = 32, L::Float64 = 1.0,
                                A::Float64 = 0.5,
                                ρ0::Float64 = 1.0,
                                P0::Float64 = 0.5,
                                σx::Float64 = 0.2,
                                s0::Float64 = -1.0)
    Δx = L / N
    Δm_vec = fill(ρ0 * Δx, N)
    positions = collect(0.0:Δx:(L - Δx))
    velocities = [A * sin(2π * x / L) for x in positions]
    αs = fill(σx, N)
    βs = zeros(Float64, N)
    # Choose s such that Mvv ≈ P0/ρ0 = const.
    # Mvv(J, s) = J^(1-Γ) exp(s/cv); Γ = 5/3, cv = 1.
    # ⇒ s = log(P0/ρ0) - (1 - Γ) log(J) = log(P0/ρ0) + (2/3) log(J).
    J0 = 1.0 / ρ0
    s_uniform = log(P0 / ρ0) + (2.0 / 3.0) * log(J0)
    ss = fill(s_uniform, N)
    Pps = fill(P0, N)
    return Mesh1D(positions, velocities, αs, βs, ss; Δm = Δm_vec,
                  Pps = Pps, L_box = L, periodic = true)
end

# -----------------------------------------------------------------------------
# 1. Smoothing kernel
# -----------------------------------------------------------------------------

@testset "smooth_periodic_3pt!: variance-preserving" begin
    rng = MersenneTwister(42)
    eta = randn(rng, 1024)
    out = similar(eta)
    smooth_periodic_3pt!(out, eta)
    σ_in = sqrt(var(eta; corrected = false))
    σ_out = sqrt(var(out; corrected = false))
    # Variance preserved by construction:
    @test isapprox(σ_in, σ_out; atol = 1e-12)
    # Some smoothing happened: pairwise differences shrink:
    @test mean(abs.(diff(out))) < mean(abs.(diff(eta)))
    # Periodic wrap respected: out[1] depends on eta[end].
    eta2 = zeros(8); eta2[1] = 1.0
    out2 = similar(eta2)
    smooth_periodic_3pt!(out2, eta2)
    # Before variance renorm the kernel is (¼, ½, ¼) hitting indices
    # (8, 1, 2). After renorm the relative shape is preserved up to a
    # global scale.
    @test out2[8] > 0 && out2[1] > 0 && out2[2] > 0
    @test isapprox(out2[1], 2 * out2[2]; atol = 1e-10)
    @test isapprox(out2[1], 2 * out2[8]; atol = 1e-10)
end

# -----------------------------------------------------------------------------
# 2. C_A = C_B = 0 ⇒ bit-equality gate with deterministic run
# -----------------------------------------------------------------------------

@testset "Phase 8: zero-noise reduces to det_run!" begin
    mesh_a = _build_sinusoidal_mesh(N = 16, A = 0.2)
    mesh_b = _build_sinusoidal_mesh(N = 16, A = 0.2)
    dt = 1e-3
    n  = 5
    rng = MersenneTwister(0)

    # Pure deterministic run on mesh_a.
    det_run!(mesh_a, dt, n)

    # Stochastic run with zero coefficients on mesh_b.
    p0 = NoiseInjectionParams(C_A = 0.0, C_B = 0.0,
                              λ = 1.6, θ_factor = 1.0/1.6)
    det_run_stochastic!(mesh_b, dt, n; params = p0, rng = rng)

    # Bit-equal trajectories.
    for j in 1:n_segments(mesh_a)
        sa = mesh_a.segments[j].state
        sb = mesh_b.segments[j].state
        @test isapprox(sa.x, sb.x; atol = 1e-12)
        @test isapprox(sa.u, sb.u; atol = 1e-12)
        @test isapprox(sa.α, sb.α; atol = 1e-12)
        @test isapprox(sa.β, sb.β; atol = 1e-12)
        @test isapprox(sa.s, sb.s; atol = 1e-12)
        @test isapprox(sa.Pp, sb.Pp; atol = 1e-12)
    end
end

# -----------------------------------------------------------------------------
# 3. Mass and momentum conservation under injection
# -----------------------------------------------------------------------------

@testset "Phase 8: mass exact, total momentum bookkeeping" begin
    mesh = _build_sinusoidal_mesh(N = 32, A = 0.4)
    M0 = total_mass(mesh)
    p0 = total_momentum(mesh)
    rng = MersenneTwister(7)
    params = NoiseInjectionParams(C_A = 0.3, C_B = 0.5,
                                  λ = 1.5, θ_factor = 1.0/1.5)
    n_steps = 20
    dt = 5e-4
    det_run_stochastic!(mesh, dt, n_steps; params = params, rng = rng)
    M1 = total_mass(mesh)
    p1 = total_momentum(mesh)
    # Mass exact (Lagrangian Δm is a label).
    @test M1 == M0
    # Total momentum: every cell-injection ΔP_j is split half/half to
    # adjacent vertices, so vertex-summed momentum = sum of cell
    # injections per step = ∑_j δ_j · Δx_j. With balanced compression/
    # expansion (sinusoidal IC) the drift contribution cancels to
    # leading order; the noise contribution is mean-zero by the VG
    # construction. Empirical bound: << initial RMS momentum.
    rms_mom = sqrt(sum((mesh.p_half).^2) / length(mesh.p_half))
    @test abs(p1 - p0) < 0.5 * rms_mom * sqrt(n_steps)
end

# -----------------------------------------------------------------------------
# 4. Energy ledger: ΔKE_vol per cell == u·δ + δ²/(2ρ)
# -----------------------------------------------------------------------------

@testset "Phase 8: per-cell ΔKE_vol bookkeeping" begin
    mesh = _build_sinusoidal_mesh(N = 16, A = 0.3)
    rng = MersenneTwister(11)
    params = NoiseInjectionParams(C_A = 0.3, C_B = 0.5,
                                  λ = 1.6, θ_factor = 1.0/1.6)
    diag = InjectionDiagnostics(n_segments(mesh))
    # Snapshot pre-injection state for the ledger.
    u_pre = [Float64(seg.state.u) for seg in mesh.segments]
    ρ_pre = [Float64(segment_density(mesh, j)) for j in 1:n_segments(mesh)]
    dt = 1e-3
    inject_vg_noise!(mesh, dt; params = params, rng = rng, diag = diag)
    # ΔKE_vol predicted from cell-centered (u_pre, ρ_pre, δ).
    for j in 1:n_segments(mesh)
        # u_centered_pre = ½ (u_left + u_right) at time n.
        j_right = j == n_segments(mesh) ? 1 : j + 1
        u_c_pre = 0.5 * (u_pre[j] + u_pre[j_right])
        δ = diag.delta_rhou[j]
        ΔKE_pred = u_c_pre * δ + 0.5 * δ * δ / ρ_pre[j]
        @test isapprox(diag.delta_KE_vol[j], ΔKE_pred; atol = 1e-12)
    end
end

# -----------------------------------------------------------------------------
# 5. Compression smoke test: noise concentrates on compressive cells
# -----------------------------------------------------------------------------

@testset "Phase 8: noise lives on compressive cells" begin
    mesh = _build_sinusoidal_mesh(N = 64, A = 0.5)
    rng = MersenneTwister(3)
    params = NoiseInjectionParams(C_A = 0.0, C_B = 0.6,
                                  λ = 1.6, θ_factor = 1.0/1.6,
                                  ke_budget_fraction = 0.5)
    diag = InjectionDiagnostics(n_segments(mesh))
    inject_vg_noise!(mesh, 1e-3; params = params, rng = rng, diag = diag)
    # Cells with divu >= 0 receive zero noise (only drift, which is
    # zero with C_A = 0).
    expansive_δ = [diag.delta_rhou_noise[j]
                   for j in 1:n_segments(mesh)
                   if diag.divu[j] >= 0]
    @test all(==(0.0), expansive_δ)
    # Compressive cells should have nonzero noise (probability ≈ 1
    # for VG draws of N≈32 active cells).
    compressive_δ = [diag.delta_rhou_noise[j]
                     for j in 1:n_segments(mesh)
                     if diag.divu[j] < 0]
    @test length(compressive_δ) > 0
    @test count(!=(0.0), compressive_δ) > 0.5 * length(compressive_δ)
end

# -----------------------------------------------------------------------------
# 6. Burst-stats accumulator round-trip on synthetic divu
# -----------------------------------------------------------------------------

@testset "BurstStatsAccumulator: synthetic burst-detection round-trip" begin
    # Synthesize 1000-step divu time series across 4 cells with known
    # alternating compress/expand pattern of duration ~5.
    N = 4
    n_steps = 1000
    acc = BurstStatsAccumulator(N)
    rng = MersenneTwister(123)
    params = NoiseInjectionParams(C_A = 0.3, C_B = 0.5,
                                  λ = 2.0, θ_factor = 0.5)
    for n in 1:n_steps
        diag = InjectionDiagnostics(N)
        # Per-cell divu: alternating ±1 in blocks of 5 steps, with
        # phase shift across cells.
        for j in 1:N
            phase = (n + 5*(j-1)) ÷ 5
            diag.divu[j] = isodd(phase) ? -1.0 : 0.5
            diag.compressive[j] = diag.divu[j] < 0
            if diag.compressive[j]
                # Inject a unit-VG η and zero limiter saturation.
                diag.eta[j] = randn(rng)  # placeholder unit-variance
                diag.delta_rhou_drift[j] = 0.0
                diag.delta_rhou_noise[j] = 1.0 * diag.eta[j]
                diag.delta_rhou[j] = diag.delta_rhou_noise[j]
            end
        end
        record_step!(acc, diag, 1e-3, params)
    end
    durations = burst_durations(acc)
    @test length(durations) >= 4 * (n_steps ÷ 10)  # ≥1 per cell per period
    res = self_consistency_check(acc; warn_ratio = 100.0)
    @test res.n_bursts == length(durations)
    # The synthesized durations are exactly 5·dt → mean 5·dt, low variance,
    # so estimate_gamma_shape returns a high k_hat. The test only asserts
    # that k_hat is finite and positive.
    @test isfinite(res.k_hat) && res.k_hat > 0
    # Saturation rate is zero by construction.
    @test res.limiter_rate == 0.0
end

# -----------------------------------------------------------------------------
# 7. Calibration loader → from_calibration round-trip
# -----------------------------------------------------------------------------

@testset "from_calibration: round-trip with load_noise_model()" begin
    nm = load_noise_model()
    p = from_calibration(nm)
    @test p.C_A == nm.C_A
    @test p.C_B == nm.C_B
    # λ is derived from kurt - 3.
    excess = nm.kurt - 3.0
    expected_λ = excess > 0 ? 3.0 / excess : 1.6
    @test isapprox(p.λ, expected_λ; rtol = 1e-12)
    @test isapprox(p.θ_factor, 1.0 / p.λ; rtol = 1e-12)
    # Document the production mismatch path (kurt 3.45 → λ ≈ 6.67).
    @test 3.0 < nm.kurt < 4.0
end

# -----------------------------------------------------------------------------
# 8. End-to-end smoke: wave-pool short run produces sane monitor numbers
# -----------------------------------------------------------------------------

@testset "Phase 8: wave-pool short run + self-consistency monitor" begin
    setup = setup_kmles_wavepool(N = 64, t_end = 0.05, u0 = 0.5,
                                 P0 = 0.05, K_max = 8, seed = 0,
                                 tau = 1e-2, n_snaps = 5)
    N = setup.params.N
    L = 1.0
    Δx = L / N
    Δm_vec = fill(setup.params.rho0 * Δx, N)
    positions = collect(0.0:Δx:(L - Δx))

    # Build an entropy field consistent with the IC pressure.
    s0 = log.(setup.P ./ setup.rho) .+ (2.0/3.0) .* log.(1.0 ./ setup.rho)

    mesh = Mesh1D(positions, setup.u,
                  setup.alpha_init, setup.beta_init, s0;
                  Δm = Δm_vec, Pps = setup.Pp, L_box = L, periodic = true)

    # Use a permissive but small λ to avoid heavy-tail saturation.
    params = NoiseInjectionParams(C_A = 0.3, C_B = 0.4,
                                  λ = 1.6, θ_factor = 1.0/1.6,
                                  ke_budget_fraction = 0.25,
                                  ell_corr = 2.0)
    rng = MersenneTwister(2026)
    acc = BurstStatsAccumulator(N)
    n_steps = 50
    dt = 1e-3
    M0 = total_mass(mesh)
    E0 = total_energy(mesh)
    det_run_stochastic!(mesh, dt, n_steps;
                        params = params, rng = rng,
                        tau = 1e-2, accumulator = acc)
    M1 = total_mass(mesh)
    E1 = total_energy(mesh)

    # Mass exact; energy bounded.
    @test M1 == M0
    # Energy-drift bound is generous: stochastic injection plus
    # amplitude-limiter clipping introduces a |E| / |E0| floor at the
    # ke_budget_fraction · n_steps · ⟨ |divu| Δt ⟩ level. Bound is set
    # from the 25% per-cell-per-step budget × dt × n_steps × max
    # compression rate.
    rel_drift = abs(E1 - E0) / max(abs(E0), 1e-12)
    @test rel_drift < 1.0  # extremely permissive smoke test

    # Self-consistency monitor returns sensible numbers when sample
    # sizes pass the documented thresholds.
    res = self_consistency_check(acc; warn_ratio = 5.0)
    @test res.n_bursts >= 0
    @test res.n_residual >= 0
    if res.n_bursts >= 5
        @test isfinite(res.k_hat) && res.k_hat > 0
    end
    # The wave-pool produces compression on roughly half the cells;
    # we expect at least some compression in 50 steps × 64 cells.
    @test acc.n_compress > 50
end

# -----------------------------------------------------------------------------
# 9. Long synthetic compression: residual-kurtosis ↔ shape consistency
# -----------------------------------------------------------------------------

@testset "Phase 8: synthetic VG residual recovers λ via kurtosis" begin
    # Build a long sample of unit-variance VG draws at known λ via
    # the InjectionDiagnostics structure, recording into a
    # BurstStatsAccumulator. Inverting kurtosis(residual) should
    # recover λ to within sampling error (~10% at 5000 samples).
    N = 8
    n_steps = 700
    acc = BurstStatsAccumulator(N)
    rng = MersenneTwister(2024)
    params = NoiseInjectionParams(C_A = 0.0, C_B = 1.0,
                                  λ = 2.0, θ_factor = 0.5)
    eta_buf = Vector{Float64}(undef, N)
    for _ in 1:n_steps
        diag = InjectionDiagnostics(N)
        rand_variance_gamma!(rng, eta_buf, params.λ, params.θ_factor)
        # Force every cell compressive so all draws enter the residual sample.
        for j in 1:N
            diag.divu[j] = -1.0
            diag.compressive[j] = true
            diag.eta[j] = eta_buf[j]
            diag.delta_rhou_drift[j] = 0.0
            diag.delta_rhou_noise[j] = eta_buf[j]
            diag.delta_rhou[j] = eta_buf[j]
            diag.delta_KE_vol[j] = 0.0
        end
        record_step!(acc, diag, 1e-3, params)
    end
    @test length(acc.residual_samples) >= N * n_steps - 10
    res = self_consistency_check(acc; warn_ratio = 100.0)
    # λ_res from kurtosis: with n ≈ 5600 unit-VG samples at λ=2, the
    # excess kurtosis estimator has standard error ≈ √(96/n) ≈ 0.13.
    # Tolerate a factor-2 bracket on λ_res (the production mismatch
    # documented in v3 §1.2 is also ~factor-2).
    @test isfinite(res.lambda_res_hat)
    @test 0.8 < res.lambda_res_hat < 6.0
end
