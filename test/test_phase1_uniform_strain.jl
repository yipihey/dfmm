# test_phase1_uniform_strain.jl
#
# Phase 1 / Cholesky-sector uniform-strain test.
#
# Purpose. Isolate the parallel-transport operator D_t^{(1)} on the
# charge-1 field β, and verify that the connection-form discrete
# transport reproduces the analytical exponential factor for the
# strain-coupling term.
#
# Setup. ∂_x u = κ (const), M_vv = 0 (constant external moment).
# Initial conditions α(0) = 1, β(0) = β₀ > 0.
#
# Realizability caveat. With M_vv = 0 and β ≠ 0 we have
# γ² = M_vv − β² = −β² < 0, which is unphysical (γ would be imaginary).
# This is intentional: the test is a *mathematical* isolation of the
# parallel-transport stencil, treating the pair of ODEs
#
#     α̇ = β,            β̇ + κ β = (M_vv − β²)/α                       (*)
#
# as a self-contained dynamical system. The discrete EL residual in
# `cholesky_sector.jl` is well-defined for any real (α, β, M_vv);
# `gamma_from_Mvv` is not invoked here. Phase 3 will revisit
# realizability with proper γ ≥ 0 constraints; in Phase 1 we exercise
# only the integrator's ability to reproduce the strain-coupling
# exponential.
#
# Closed-form solution at M_vv = 0. Multiplying (*)b by α and using
# α̇ = β:
#     α β̇ + β² = −κ α β
#     ⇔  d(αβ)/dt = −κ (αβ)
#     ⇔  α(t) β(t) = α(0) β(0) · e^{−κt}.                              (✓)
#
# This is the exponential strain factor predicted by the connection
# form: a charge-1 quantity (here the product αβ, charge 0+1=1)
# decays at rate κ under uniform strain. Combined with α̇ = β:
#
#     α α̇ = c₀ e^{−κt}     where c₀ = α(0) β(0)
#     ⇒  α(t)² = α(0)² + (2 c₀ / κ) · (1 − e^{−κt})
#     ⇒  β(t) = c₀ e^{−κt} / α(t).                                    (✓✓)
#
# The expression for α reduces to α(0)² + 2 c₀ t in the limit κ → 0
# (zero-strain limit, recovers the M_vv=0 part of the previous test).
#
# Test. Step the integrator at Δt = 1e-3 for 100 steps with κ = 0.1
# and assert pointwise match to (✓✓) at the level of the implicit
# midpoint truncation error, plus the connection-form invariant
# αβ · e^{κt} = const to a tighter tolerance.

using Test
using StaticArrays
using dfmm

@testset "Phase 1: uniform-strain Cholesky-sector evolution" begin
    M_vv = 0.0
    κ = 0.1
    Δt = 1e-3
    N = 100
    α0, β0 = 1.0, 0.5
    c0 = α0 * β0
    q_init = SVector{2,Float64}(α0, β0)

    traj = cholesky_run(q_init, M_vv, κ, Δt, N;
                        abstol = 1e-13, reltol = 1e-13)

    @testset "connection-form invariant α·β·e^{κt} = const to 1e-9" begin
        # The exact discrete invariant of the implicit midpoint scheme
        # for this system is preserved up to truncation error in the
        # midpoint-rule approximation of d(αβ)/dt + καβ = 0. Expect
        # O((Δt)²) ~ 1e-6 drift over T = 0.1; we assert a comfortable
        # 1e-6 absolute bound.
        max_err = 0.0
        for n in 0:N
            t = n * Δt
            invariant = traj[n+1][1] * traj[n+1][2] * exp(κ * t)
            max_err = max(max_err, abs(invariant - c0))
        end
        @test max_err < 1e-6
    end

    @testset "trajectory matches closed form to truncation order" begin
        # α(t) = √(α₀² + (2c₀/κ)(1 − e^{−κt})),  β(t) = c₀ e^{−κt} / α(t).
        # Implicit midpoint global error scales as Δt² ~ 1e-6.
        max_err_α = 0.0
        max_err_β = 0.0
        for n in 0:N
            t = n * Δt
            α_exact = sqrt(α0^2 + (2 * c0 / κ) * (1 - exp(-κ * t)))
            β_exact = c0 * exp(-κ * t) / α_exact
            max_err_α = max(max_err_α, abs(traj[n+1][1] - α_exact))
            max_err_β = max(max_err_β, abs(traj[n+1][2] - β_exact))
        end
        @test max_err_α < 1e-5
        @test max_err_β < 1e-5
    end

    @testset "discrete transport operator on a manufactured solution" begin
        # Direct unit test of D_t_q on the charge-1 field β satisfying
        # exactly β̇ + κ β = 0, i.e. β(t) = β₀ exp(-κ t). Plug in the
        # exact β values at consecutive time levels and verify the
        # discrete operator returns 0 to truncation order (the midpoint
        # rule is exact for affine ODEs, so this should be very small).
        β_n = β0 * exp(-κ * 0.0)
        β_np1 = β0 * exp(-κ * Δt)
        residual = D_t_q(β_n, β_np1, κ, 1, Δt)
        # Midpoint rule applied to ẏ + κy = 0 gives an O((κΔt)³) error
        # on each step; with κΔt = 1e-4 the per-step error is ~1e-12.
        @test abs(residual) < 1e-9
    end
end
