# test_deviatoric_unit.jl
#
# Unit tests for the Phase-5 BGK relaxation primitives in
# `src/deviatoric.jl`. These were previously exercised only through the
# Sod regression; here we pin the closed-form behaviour of each helper
# in isolation:
#
#   * `deviatoric_bgk_step` — bilinear implicit-midpoint Π update.
#   * `deviatoric_bgk_step_exponential` — exact-exponential Π update
#     (the form py-1d uses; bit-equality requirement).
#   * `pperp_advect_lagrangian` — `D_t (P_⊥/ρ) = 0` transport.
#   * `bgk_relax_pressures` — joint (P_xx, P_⊥) relaxation conserves
#     P_iso = (P_xx + 2 P_⊥)/3.
#   * `pperp_step` — Lagrangian-transport then BGK; τ → ∞ limit
#     reduces to pure transport, τ → 0 drives P_⊥ → P_xx.

using Test
using dfmm

@testset "deviatoric_bgk_step: η = 0 implicit-midpoint exponential decay" begin
    # With η = 0 and small dt/τ the bilinear form should agree with
    # the exact exponential to leading order, and Π should decrease
    # in magnitude monotonically.
    Π_n = 0.5
    τ = 1.0
    dt = 0.01
    Π_np1 = dfmm.deviatoric_bgk_step(Π_n, 0.0, τ, 0.0, dt)
    expected = Π_n * (1 - dt / (2τ)) / (1 + dt / (2τ))
    @test isapprox(Π_np1, expected; atol = 1e-14)
    @test 0 < Π_np1 < Π_n
end

@testset "deviatoric_bgk_step: η ≠ 0 source term enters linearly" begin
    # Adding η · divu to the bilinear ODE shifts Π_np1 by a quantity
    # linear in η · divu. We verify by evaluating at two η values and
    # checking the difference matches the closed-form coefficient.
    Π_n = 0.0
    τ = 1.0
    dt = 0.01
    divu = 0.4
    Π_a = dfmm.deviatoric_bgk_step(Π_n, divu, τ, 0.0, dt)
    Π_b = dfmm.deviatoric_bgk_step(Π_n, divu, τ, 1.0, dt)
    # Closed form: shift = -2 · 1.0 · divu · dt / (1 + dt/(2τ)).
    expected_shift = -2 * 1.0 * divu * dt / (1 + dt / (2τ))
    @test isapprox(Π_b - Π_a, expected_shift; atol = 1e-14)
end

@testset "deviatoric_bgk_step_exponential: matches py-1d's exp(-Δt/τ) form" begin
    Π_n = 0.5
    τ = 1.0
    dt = 0.01
    @test isapprox(
        dfmm.deviatoric_bgk_step_exponential(Π_n, dt, τ),
        Π_n * exp(-dt / τ); atol = 1e-14)

    # Limit Δt → 0 ⇒ Π_np1 → Π_n.
    @test dfmm.deviatoric_bgk_step_exponential(Π_n, 0.0, τ) == Π_n

    # Limit Δt → ∞ ⇒ Π_np1 → 0.
    @test isapprox(dfmm.deviatoric_bgk_step_exponential(Π_n, 1e6, τ),
                   0.0; atol = 1e-12)
end

@testset "deviatoric_bgk_step_exponential: positivity preserved" begin
    # Π_n > 0 ⇒ Π_np1 > 0 always. Π_n < 0 ⇒ Π_np1 < 0 always.
    @test dfmm.deviatoric_bgk_step_exponential(0.7, 0.05, 1.0) > 0
    @test dfmm.deviatoric_bgk_step_exponential(-0.3, 0.05, 1.0) < 0
end

@testset "pperp_advect_lagrangian: D_t(P_⊥/ρ) = 0 transport" begin
    # Compression (ρ_np1 > ρ_n) → P_⊥ scales up by the same factor.
    @test isapprox(
        dfmm.pperp_advect_lagrangian(1.0, 1.0, 2.0),
        2.0; atol = 1e-14)
    # Expansion (ρ_np1 < ρ_n) → P_⊥ scales down.
    @test isapprox(
        dfmm.pperp_advect_lagrangian(2.0, 1.0, 0.5),
        1.0; atol = 1e-14)
    # No density change → P_⊥ unchanged.
    @test dfmm.pperp_advect_lagrangian(0.7, 1.5, 1.5) == 0.7
end

@testset "bgk_relax_pressures: conserves P_iso = (P_xx + 2 P_⊥)/3" begin
    Pxx_n = 1.5
    Pp_n  = 0.6
    P_iso = (Pxx_n + 2 * Pp_n) / 3
    for dt in (0.0, 0.01, 0.1, 1.0, 100.0)
        Pxx_new, Pp_new = dfmm.bgk_relax_pressures(Pxx_n, Pp_n, dt, 1.0)
        P_iso_new = (Pxx_new + 2 * Pp_new) / 3
        @test isapprox(P_iso_new, P_iso; atol = 1e-13)
    end
end

@testset "bgk_relax_pressures: τ → ∞ limit ⇒ no relaxation" begin
    Pxx_n = 1.5
    Pp_n  = 0.6
    Pxx_new, Pp_new = dfmm.bgk_relax_pressures(Pxx_n, Pp_n, 0.01, 1e12)
    @test isapprox(Pxx_new, Pxx_n; atol = 1e-12)
    @test isapprox(Pp_new, Pp_n;  atol = 1e-12)
end

@testset "bgk_relax_pressures: τ → 0 limit ⇒ full isotropization" begin
    Pxx_n = 1.5
    Pp_n  = 0.6
    P_iso = (Pxx_n + 2 * Pp_n) / 3
    Pxx_new, Pp_new = dfmm.bgk_relax_pressures(Pxx_n, Pp_n, 0.01, 1e-12)
    @test isapprox(Pxx_new, P_iso; atol = 1e-12)
    @test isapprox(Pp_new,  P_iso; atol = 1e-12)
end

@testset "bgk_relax_pressures: anisotropy Π = P_xx − P_⊥ decays by exp(-Δt/τ)" begin
    Pxx_n = 1.2
    Pp_n  = 0.4
    Π_n = Pxx_n - Pp_n
    dt, τ = 0.05, 0.2
    Pxx_new, Pp_new = dfmm.bgk_relax_pressures(Pxx_n, Pp_n, dt, τ)
    Π_new = Pxx_new - Pp_new
    @test isapprox(Π_new, Π_n * exp(-dt / τ); atol = 1e-13)
end

@testset "pperp_step: τ → ∞ limit reduces to pure transport" begin
    # Pp_transport = Pp_n · ρ_np1/ρ_n; with τ → ∞ the BGK is off and
    # the result is exactly the transport.
    Pp_n = 0.5
    ρ_n = 1.0
    ρ_np1 = 0.7
    Pxx_np1 = 0.9
    dt, τ = 0.01, 1e12
    Pp_new = dfmm.pperp_step(Pp_n, ρ_n, ρ_np1, Pxx_np1, dt, τ)
    @test isapprox(Pp_new, Pp_n * ρ_np1 / ρ_n; atol = 1e-12)
end

@testset "pperp_step: τ → 0 limit drives P_⊥ → P_xx" begin
    # Instantaneous BGK ⇒ P_⊥ matches P_xx after the step.
    Pp_n = 0.5
    ρ_n = 1.0
    ρ_np1 = 1.0
    Pxx_np1 = 1.3
    Pp_new = dfmm.pperp_step(Pp_n, ρ_n, ρ_np1, Pxx_np1, 0.01, 1e-12)
    @test isapprox(Pp_new, Pxx_np1; atol = 1e-10)
end

@testset "pperp_step: dt = 0 acts as identity (transport only)" begin
    # With dt = 0, the BGK exponential factor is exp(0) = 1 ⇒ no
    # relaxation; the result is the pure transport value.
    Pp_n = 0.7
    ρ_n = 1.0
    ρ_np1 = 0.5
    Pp_new = dfmm.pperp_step(Pp_n, ρ_n, ρ_np1, 1.0, 0.0, 1.0)
    @test isapprox(Pp_new, Pp_n * ρ_np1 / ρ_n; atol = 1e-13)
end
