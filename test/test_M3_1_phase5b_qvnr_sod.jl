# test_M3_1_phase5b_qvnr_sod.jl
#
# Phase M3-1 Phase-5b sub-phase, block 2: Sod regression with
# `q_kind = :vNR_linear_quadratic` on the HG substrate. Mirrors M1's
# `test_phase5b_artificial_viscosity.jl` block 3. The HG path
# delegates to M1's `det_step!` so the L∞ rel errors against py-1d's
# Sod golden must match M1's bounds within bit-exact tolerance.

using Test
using dfmm

include(joinpath(@__DIR__, "..", "experiments", "A1_sod.jl"))
include(joinpath(@__DIR__, "test_M3_1_phase5_sod_HG.jl_helpers.jl"))

@testset "M3-1 Phase-5b (HG): Sod with q=:vNR_linear_quadratic — N=100" begin
    N = 100
    t_end = 0.2
    tau   = 1e-3
    sigma_x0 = 0.02
    cfl = 0.3
    c_q_quad = 2.0
    c_q_lin  = 1.0

    ic = setup_sod(; N = N, t_end = t_end, sigma_x0 = sigma_x0, tau = tau)
    mesh_M1 = build_sod_mesh(ic; mirror = true)
    mesh_HG = build_sod_mesh_HG(ic; mirror = true)

    Γ = 5.0 / 3.0
    N_seg = dfmm.n_segments(mesh_M1)
    c_s_max = sqrt(Γ * maximum(ic.P) / minimum(ic.rho))
    dx = 2.0 / N_seg
    dt_cfl = cfl * dx / c_s_max
    n_steps = ceil(Int, t_end / dt_cfl)
    dt = t_end / n_steps

    mass0 = dfmm.total_mass(mesh_M1)
    mom0  = dfmm.total_momentum(mesh_M1)
    KE0   = dfmm.total_kinetic_energy(mesh_M1)
    Eint0 = dfmm.total_internal_energy(mesh_M1)
    Etot0 = KE0 + Eint0

    αmax = 0.0; γ²min = Inf
    max_parity_err = 0.0
    t0 = time()
    for n in 1:n_steps
        dfmm.det_step!(mesh_M1, dt; tau = tau,
                        q_kind = :vNR_linear_quadratic,
                        c_q_quad = c_q_quad, c_q_lin = c_q_lin)
        det_step_HG!(mesh_HG, dt; tau = tau,
                      q_kind = :vNR_linear_quadratic,
                      c_q_quad = c_q_quad, c_q_lin = c_q_lin)
        if n % max(1, n_steps ÷ 10) == 0 || n == n_steps
            for j in 1:N_seg
                sM = mesh_M1.segments[j].state
                sH = read_detfield(mesh_HG.fields, j)
                max_parity_err = max(max_parity_err,
                                     abs(sM.x  - sH.x),
                                     abs(sM.u  - sH.u),
                                     abs(sM.α  - sH.α),
                                     abs(sM.β  - sH.β),
                                     abs(sM.s  - sH.s),
                                     abs(sM.Pp - sH.Pp))
            end
        end
    end
    wall = time() - t0

    KE1   = dfmm.total_kinetic_energy(mesh_M1)
    Eint1 = dfmm.total_internal_energy(mesh_M1)
    Etot1 = KE1 + Eint1
    mass_err = abs(dfmm.total_mass(mesh_M1) - mass0) / mass0
    mom_err  = abs(dfmm.total_momentum(mesh_M1) - mom0)
    ΔE_rel   = abs(Etot1 - Etot0) / abs(Etot0)

    profile_HG = extract_eulerian_profiles_HG(mesh_HG)

    golden = dfmm.load_tier_a_golden(:sod)
    N0 = N
    rho_jl = profile_HG.rho[1:N0]
    u_jl   = profile_HG.u[1:N0]
    Pxx_jl = profile_HG.Pxx[1:N0]
    Pp_jl  = profile_HG.Pp[1:N0]
    x_jl   = profile_HG.x[1:N0]

    interp_to(x_target, x_src, y_src) = [begin
        i = searchsortedfirst(x_src, xi)
        if i == 1
            y_src[1]
        elseif i > length(x_src)
            y_src[end]
        else
            x1 = x_src[i-1]; x2 = x_src[i]
            f = (xi - x1) / (x2 - x1)
            (1-f) * y_src[i-1] + f * y_src[i]
        end
    end for xi in x_target]

    rho_g = interp_to(x_jl, golden.x, golden.fields.rho[:, end])
    u_g   = interp_to(x_jl, golden.x, golden.fields.u[:, end])
    Pxx_g = interp_to(x_jl, golden.x, golden.fields.Pxx[:, end])
    Pp_g  = interp_to(x_jl, golden.x, golden.fields.Pp[:, end])

    linf_rel(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1e-12)
    err_rho = linf_rel(rho_jl, rho_g)
    err_u   = linf_rel(u_jl,   u_g)
    err_Pxx = linf_rel(Pxx_jl, Pxx_g)
    err_Pp  = linf_rel(Pp_jl,  Pp_g)

    # Same bounds as M1 Phase-5b.
    @test err_rho < 0.13
    @test err_Pxx < 0.18
    @test err_Pp  < 0.18
    @test err_u   < 0.70

    @test mass_err < 1e-10
    @test mom_err  < 1e-10
    @test ΔE_rel   < 0.10

    # Bit-exact parity vs M1.
    @test max_parity_err < 1e-12
    @test max_parity_err == 0.0

    @info "M3-1 Phase 5b Sod with q=:vNR (HG, N=100, inline)" err_rho err_u err_Pxx err_Pp wall_seconds = wall parity_err = max_parity_err
end
