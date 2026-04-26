# test_M3_1_phase2_acoustic_HG.jl
#
# Phase M3-1 verification gate (Phase-2 sub-phase): linearised acoustic
# wave dispersion on the HG-substrate driver, mirroring M1's
# `test_phase2_acoustic.jl`. Numerical sound speed must match the
# analytical c_s = √Γ to 5%, and the run must bit-equal M1's path
# step-by-step.

using Test
using dfmm

function _cos_projection_HG(mesh_HG, k_m)
    N = length(mesh_HG.Δm)
    Δm_val = mesh_HG.Δm[1]
    s = 0.0
    for j in 1:N
        ρ_j = segment_density_HG(mesh_HG, j)
        m_j = (j - 0.5) * Δm_val
        s += ρ_j * cos(k_m * m_j)
    end
    return s / N
end

@testset "M3-1 Phase-2 (HG): acoustic wave dispersion" begin
    N = 64
    L = 1.0
    ρ0 = 1.0
    s0 = 0.0
    α0 = 1.0
    β0 = 0.0
    Δx = L / N
    Δm_val = ρ0 * Δx
    Δm = fill(Δm_val, N)

    M_total = sum(Δm)
    k_m = 2π / M_total

    ε = 1e-4
    positions = zeros(N)
    cum_x = 0.0
    for j in 1:N
        m_center = (j - 0.5) * Δm_val
        ρ_j = ρ0 * (1 + ε * cos(k_m * m_center))
        positions[j] = cum_x
        cum_x += Δm_val / ρ_j
    end
    L_box = cum_x

    velocities = zeros(N)
    αs = fill(α0, N)
    βs = fill(β0, N)
    ss = fill(s0, N)

    mesh_HG = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                     Δm = Δm, L_box = L_box, bc = :periodic)
    mesh_M1 = dfmm.Mesh1D(positions, velocities, αs, βs, ss;
                          Δm = Δm, L_box = L_box, periodic = true)

    Γ = 5.0 / 3.0
    c_s_x = sqrt(Γ)

    dt = 1e-3
    N_steps = 1500
    times = Float64[0.0]
    cos_proj = Float64[_cos_projection_HG(mesh_HG, k_m)]

    max_parity_err = 0.0
    for n in 1:N_steps
        det_step_HG!(mesh_HG, dt)
        dfmm.det_step!(mesh_M1, dt)
        push!(times, n * dt)
        push!(cos_proj, _cos_projection_HG(mesh_HG, k_m))
        # Bit-exact parity sample: every 100 steps.
        if n % 100 == 0
            for j in 1:N
                sM = mesh_M1.segments[j].state
                sH = read_detfield(mesh_HG.fields, j)
                max_parity_err = max(max_parity_err,
                                     abs(sM.x - sH.x),
                                     abs(sM.u - sH.u),
                                     abs(sM.α - sH.α),
                                     abs(sM.β - sH.β))
            end
        end
    end

    deviation = cos_proj .- (sum(cos_proj) / length(cos_proj))
    crossings = Float64[]
    for i in 2:length(deviation)
        if deviation[i-1] * deviation[i] < 0
            t_cross = times[i-1] + (times[i] - times[i-1]) *
                      abs(deviation[i-1]) / (abs(deviation[i-1]) + abs(deviation[i]))
            push!(crossings, t_cross)
        end
    end
    @test length(crossings) ≥ 4

    half_periods = diff(crossings)
    T_period = 2 * (sum(half_periods) / length(half_periods))
    ω_num = 2π / T_period
    c_s_num = ω_num / (2π / L_box)

    @info "M3-1 acoustic wave (HG)" c_s_num c_s_x ratio = c_s_num / c_s_x parity_err = max_parity_err

    @test abs(c_s_num - c_s_x) / c_s_x < 0.05

    @test isapprox(total_mass_HG(mesh_HG), sum(Δm); atol = 100 * eps())
    @test abs(total_momentum_HG(mesh_HG)) < 1e-10

    # Bit-exact parity vs M1.
    @test max_parity_err < 5e-13
    @test max_parity_err == 0.0
end
