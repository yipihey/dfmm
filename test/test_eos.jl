using Test
using dfmm

@testset "eos: Mvv closed form and limits" begin
    # Reference point: J=1, s=0 ⇒ Mvv = 1.
    @test dfmm.Mvv(1.0, 0.0) ≈ 1.0

    # Default Gamma=5/3, cv=1: Mvv = J^(1-Gamma) * exp(s).
    let J = 0.5, s = 0.7
        Gamma = 5 / 3
        expected = J^(1 - Gamma) * exp(s)
        @test dfmm.Mvv(J, s) ≈ expected
    end

    # Custom Gamma/cv path.
    let J = 2.0, s = 1.5, Gamma = 1.4, cv = 1.5
        expected = J^(1 - Gamma) * exp(s / cv)
        @test dfmm.Mvv(J, s; Gamma = Gamma, cv = cv) ≈ expected
    end
end

@testset "eos: cold-limit Mvv -> 0 as s -> -inf" begin
    # Sweep entropy down; Mvv should monotonically -> 0 and never NaN.
    last = 1.0
    for s in (-1.0, -10.0, -100.0, -500.0, -1000.0, -1.0e4)
        m = dfmm.Mvv(1.0, s)
        @test isfinite(m)
        @test m ≥ 0.0
        @test m ≤ last + 1.0e-12
        last = m
    end
    # At s = -1e4 (far below the underflow threshold) we hit the exact
    # zero clamp.
    @test dfmm.Mvv(1.0, -1.0e4) == 0.0
end

@testset "eos: realizability + invalid inputs" begin
    # J <= 0 is unphysical: NaN.
    @test isnan(dfmm.Mvv(0.0, 0.0))
    @test isnan(dfmm.Mvv(-0.1, 0.0))

    # Boundary of finite Float64 doesn't crash.
    @test isfinite(dfmm.Mvv(1.0, 100.0))
end

@testset "eos: gamma_from_state" begin
    # Inside cone.
    @test dfmm.gamma_from_state(1.0, 0.5) ≈ sqrt(1.0 - 0.25)
    # On the realizability boundary.
    @test dfmm.gamma_from_state(0.25, 0.5) ≈ 0.0
    # Outside the cone clamps to zero (NOT NaN).
    @test dfmm.gamma_from_state(0.1, 0.5) == 0.0
    # Cold limit.
    @test dfmm.gamma_from_state(0.0, 0.0) == 0.0
end

@testset "eos: pressure round-trip" begin
    rho, P = 1.3, 0.91
    m = dfmm.Mvv_from_pressure(rho, P)
    @test m ≈ P / rho
    @test dfmm.pressure_from_Mvv(rho, m) ≈ P
end
