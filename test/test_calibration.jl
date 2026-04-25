using Test
using dfmm

@testset "calibration: load_noise_model returns expected fields" begin
    cal = dfmm.load_noise_model()
    expected = (:C_A, :C_A_err, :C_B, :C_B_err, :floor, :kurt, :skew, :dt_save)
    @test Set(keys(cal)) == Set(expected)
end

@testset "calibration: all values are Float64" begin
    cal = dfmm.load_noise_model()
    for (name, val) in pairs(cal)
        @test val isa Float64
        @test isfinite(val)
    end
end

@testset "calibration: production values within sanity ranges" begin
    cal = dfmm.load_noise_model()
    # From py-1d/data/noise_model_params.npz: C_A ~ 0.336, C_B ~ 0.548.
    @test 0.0 < cal.C_A < 1.0
    @test 0.0 < cal.C_B < 1.0
    @test 0.0 < cal.C_A_err < cal.C_A   # error smaller than value
    @test 0.0 < cal.C_B_err < cal.C_B
    @test cal.floor ≥ 0.0
    @test 0.0 < cal.dt_save < 1.0
    # Production kurt ~3.45 (Laplace=3, Gaussian=0); should be positive
    # and not crazy.
    @test 0.0 < cal.kurt < 10.0
    # |skew| should be modest.
    @test abs(cal.skew) < 1.0
end

@testset "calibration: explicit-path load works" begin
    path = joinpath(@__DIR__, "..", "py-1d", "data", "noise_model_params.npz")
    cal = dfmm.load_noise_model(path)
    @test cal.C_A > 0.0
end

@testset "calibration: missing file raises" begin
    @test_throws Exception dfmm.load_noise_model(
        "/nonexistent/path/noise_model_params.npz")
end
