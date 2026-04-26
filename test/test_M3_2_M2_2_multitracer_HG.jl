# test_M3_2_M2_2_multitracer_HG.jl
#
# Phase M3-2 Block 3 (companion) — multi-tracer fidelity in 1D wave-pool
# turbulence with stochastic injection enabled, on the HG substrate.
# Mirrors `test_phase_M2_2_multitracer.jl` for the bit-exact-tracer-
# preservation claims.

using Test
using Random: MersenneTwister
using dfmm
using dfmm: total_mass, total_mass_HG, det_step!, inject_vg_noise!,
            inject_vg_noise_HG!, det_step_HG!,
            NoiseInjectionParams, from_calibration, InjectionDiagnostics,
            load_noise_model, setup_kmles_wavepool, n_segments,
            DetMeshHG_from_arrays

# Build a periodic wave-pool mesh on the HG substrate.
function _build_wavepool_mesh_HG(; N::Int = 64, u0::Float64 = 0.3,
                                  P0::Float64 = 1.0, K_max::Int = 8,
                                  seed::Int = 42)
    setup = setup_kmles_wavepool(N = N, t_end = 1.0,
                                 u0 = u0, P0 = P0, K_max = K_max,
                                 seed = seed, n_snaps = 5)
    L = 1.0
    Δx = L / N
    Δm_vec = fill(setup.params.rho0 * Δx, N)
    positions = collect(0.0:Δx:(L - Δx))
    s0 = log.(setup.P ./ setup.rho) .+ (2.0/3.0) .* log.(1.0 ./ setup.rho)
    return DetMeshHG_from_arrays(positions, setup.u,
                                  setup.alpha_init, setup.beta_init, s0;
                                  Δm = Δm_vec, Pps = setup.Pp,
                                  L_box = L, bc = :periodic)
end

@testset "M3-2 M2-2 (HG): bit-exact tracer matrix under stochastic injection" begin
    N = 64
    n_steps = 150
    dt = 1e-3

    mesh_HG = _build_wavepool_mesh_HG(; N = N, seed = 2026)
    M_total = total_mass_HG(mesh_HG)

    positions = [0.2, 0.35, 0.5, 0.65, 0.85]
    K = length(positions)
    names = [Symbol("step_", k) for k in 1:K]
    tm = TracerMeshHG(mesh_HG; n_tracers = K, names = names)
    for (k, p) in enumerate(positions)
        set_tracer!(tm, names[k], m -> m < p * M_total ? 1.0 : 0.0)
    end
    tracers_initial = copy(tm.tm.tracers)
    matrix_id_before = objectid(tm.tm.tracers)

    nm = load_noise_model()
    p_default = from_calibration(nm)
    params = NoiseInjectionParams(
        C_A = p_default.C_A,
        C_B = 0.5 * p_default.C_B,
        λ = p_default.λ, θ_factor = p_default.θ_factor,
        ke_budget_fraction = p_default.ke_budget_fraction,
        ell_corr = p_default.ell_corr,
        pressure_floor = p_default.pressure_floor,
    )
    rng = MersenneTwister(2026)
    diag = InjectionDiagnostics(N)

    for n in 1:n_steps
        det_step_HG!(mesh_HG, dt; tau = 1e-2,
                     q_kind = :vNR_linear_quadratic)
        inject_vg_noise_HG!(mesh_HG, dt; params = params, rng = rng,
                            diag = diag)
        advect_tracers_HG!(tm, dt)
        all(isfinite, diag.divu) || break
    end

    # The tracer matrix is the same object and is bit-equal.
    @test objectid(tm.tm.tracers) == matrix_id_before
    @test tm.tm.tracers == tracers_initial
    @test maximum(abs.(tm.tm.tracers .- tracers_initial)) === 0.0
end

@testset "M3-2 M2-2 (HG): bit-exact parity vs M1 (tracer matrix + state)" begin
    N = 32
    n_steps = 50
    dt = 1e-3

    setup = setup_kmles_wavepool(N = N, t_end = 1.0,
                                 u0 = 0.3, P0 = 1.0, K_max = 8,
                                 seed = 99, n_snaps = 5)
    L = 1.0
    Δx = L / N
    Δm_vec = fill(setup.params.rho0 * Δx, N)
    positions = collect(0.0:Δx:(L - Δx))
    s0 = log.(setup.P ./ setup.rho) .+ (2.0/3.0) .* log.(1.0 ./ setup.rho)

    mesh_M1 = dfmm.Mesh1D(positions, setup.u,
                          setup.alpha_init, setup.beta_init, s0;
                          Δm = Δm_vec, Pps = setup.Pp, L_box = L,
                          periodic = true)
    mesh_HG = DetMeshHG_from_arrays(positions, setup.u,
                                     setup.alpha_init, setup.beta_init, s0;
                                     Δm = Δm_vec, Pps = setup.Pp,
                                     L_box = L, bc = :periodic)

    M_total_M1 = total_mass(mesh_M1)
    M_total_HG = total_mass_HG(mesh_HG)
    tm_M1 = TracerMesh(mesh_M1; n_tracers = 3,
                       names = [:a, :b, :c])
    tm_HG = TracerMeshHG(mesh_HG; n_tracers = 3,
                         names = [:a, :b, :c])
    for (k, p) in enumerate([0.25, 0.5, 0.75])
        set_tracer!(tm_M1, [:a, :b, :c][k], m -> m < p * M_total_M1 ? 1.0 : 0.0)
        set_tracer!(tm_HG, [:a, :b, :c][k], m -> m < p * M_total_HG ? 1.0 : 0.0)
    end

    nm = load_noise_model()
    p_default = from_calibration(nm)
    params = NoiseInjectionParams(
        C_A = p_default.C_A,
        C_B = 0.5 * p_default.C_B,
        λ = p_default.λ, θ_factor = p_default.θ_factor,
        ke_budget_fraction = p_default.ke_budget_fraction,
        ell_corr = p_default.ell_corr,
        pressure_floor = p_default.pressure_floor,
    )

    rng_M1 = MersenneTwister(99)
    rng_HG = MersenneTwister(99)
    diag_M1 = InjectionDiagnostics(N)
    diag_HG = InjectionDiagnostics(N)

    for n in 1:n_steps
        dfmm.det_step!(mesh_M1, dt; tau = 1e-2,
                       q_kind = :vNR_linear_quadratic)
        det_step_HG!(mesh_HG, dt; tau = 1e-2,
                     q_kind = :vNR_linear_quadratic)
        inject_vg_noise!(mesh_M1, dt; params = params, rng = rng_M1,
                         diag = diag_M1)
        inject_vg_noise_HG!(mesh_HG, dt; params = params, rng = rng_HG,
                            diag = diag_HG)
        advect_tracers!(tm_M1, dt)
        advect_tracers_HG!(tm_HG, dt)
        all(isfinite, diag_M1.divu) || break
        all(isfinite, diag_HG.divu) || break
    end

    # Tracer matrices identical (neither path mutates them).
    @test tm_M1.tracers == tm_HG.tm.tracers

    # Per-cell fluid state bit-exact.
    max_err = 0.0
    for j in 1:N
        sM = mesh_M1.segments[j].state
        sH = read_detfield(mesh_HG.fields, j)
        max_err = max(max_err,
                      abs(sM.x  - sH.x),
                      abs(sM.u  - sH.u),
                      abs(sM.α  - sH.α),
                      abs(sM.β  - sH.β),
                      abs(sM.s  - sH.s),
                      abs(sM.Pp - sH.Pp))
    end
    @test max_err == 0.0
end
