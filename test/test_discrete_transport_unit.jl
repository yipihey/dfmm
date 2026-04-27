# test_discrete_transport_unit.jl
#
# Direct unit tests for `D_t_q` in `src/discrete_transport.jl`. The
# operator is the discrete analogue of the GL(d)-bundle covariant time
# derivative
#
#     D_t^{(q)} φ ≈ (φ_{n+1} - φ_n)/Δt + q · (∂_x u)_{n+1/2} · φ̄,
#
# with `φ̄ = (φ_n + φ_{n+1})/2`. It is called from every Phase-1+ Newton
# step (`cholesky_sector.jl`, `eom.jl`); previously verified only via
# integrator-level smoke tests. These tests pin the closed-form for the
# corner cases the EL residual relies on (zero strain, q=0 / q=1
# branches, midpoint-average symmetry, ε-linearisation, and ForwardDiff-
# friendly type pass-through).

using Test
using dfmm: D_t_q

@testset "D_t_q: zero-strain reduces to forward difference" begin
    # divu_half = 0 ⇒ D_t_q = (φ_{n+1} - φ_n)/Δt regardless of q.
    let phi_n = 1.0, phi_np1 = 1.1, dt = 0.01
        @test D_t_q(phi_n, phi_np1, 0.0, 0, dt) ≈ (phi_np1 - phi_n) / dt
        @test D_t_q(phi_n, phi_np1, 0.0, 1, dt) ≈ (phi_np1 - phi_n) / dt
        @test D_t_q(phi_n, phi_np1, 0.0, 2, dt) ≈ (phi_np1 - phi_n) / dt
    end
end

@testset "D_t_q: stationary field with q=0 sees no strain coupling" begin
    # With q=0 the strain term vanishes; with stationary field
    # (phi_np1 == phi_n), D_t_q must be exactly zero.
    let phi = 0.5, divu = 100.0, dt = 0.01
        @test D_t_q(phi, phi, divu, 0, dt) == 0.0
    end
    # With q=1 and stationary field, the result is q · divu · φ̄ = divu · phi.
    let phi = 0.5, divu = 0.1, dt = 0.01
        @test D_t_q(phi, phi, divu, 1, dt) ≈ divu * phi
    end
end

@testset "D_t_q: midpoint-average symmetry under (φ_n, φ_{n+1}) swap" begin
    # The strain coupling uses φ̄ = (φ_n + φ_{n+1})/2, which is symmetric
    # in the two endpoints; the time-derivative term flips sign on swap.
    # Therefore swapping endpoints must flip the sign of the
    # forward-difference term while leaving the strain term unchanged.
    let phi_n = 0.7, phi_np1 = 1.3, divu = 0.4, q = 1, dt = 0.05
        forward  = D_t_q(phi_n, phi_np1, divu, q, dt)
        reverse  = D_t_q(phi_np1, phi_n, divu, q, dt)
        # Reconstruct each piece:
        dphi_dt = (phi_np1 - phi_n) / dt
        strain  = q * divu * (phi_n + phi_np1) / 2
        @test forward ≈ dphi_dt + strain
        @test reverse ≈ -dphi_dt + strain
        @test isapprox(forward + reverse, 2 * strain; atol = 1e-14)
    end
end

@testset "D_t_q: q=1 vs q=0 difference equals exactly divu · φ̄" begin
    # q=1 and q=0 must differ exactly by `divu_half · φ̄` regardless of
    # endpoint values. This is the contract every D_t^{(q)} caller
    # relies on.
    let phi_n = -0.3, phi_np1 = 0.9, divu = 0.7, dt = 0.02
        d0 = D_t_q(phi_n, phi_np1, divu, 0, dt)
        d1 = D_t_q(phi_n, phi_np1, divu, 1, dt)
        phi_bar = (phi_n + phi_np1) / 2
        @test isapprox(d1 - d0, divu * phi_bar; atol = 1e-14)
    end
end

@testset "D_t_q: linearity in (φ_n, φ_{n+1}) with strain held fixed" begin
    # Linear in (φ_n, φ_{n+1}) jointly: D(aφ_n + bφ_n', aφ_np1 + bφ_np1')
    # = a·D(φ_n, φ_np1) + b·D(φ_n', φ_np1').
    let phi_n = 1.0, phi_np1 = 1.5, ψ_n = 0.2, ψ_np1 = 0.4
        divu, q, dt = 0.3, 1, 0.02
        a, b = 2.5, -1.7
        lhs = D_t_q(a * phi_n + b * ψ_n, a * phi_np1 + b * ψ_np1,
                    divu, q, dt)
        rhs = a * D_t_q(phi_n, phi_np1, divu, q, dt) +
              b * D_t_q(ψ_n, ψ_np1, divu, q, dt)
        @test isapprox(lhs, rhs; atol = 1e-13)
    end
end

@testset "D_t_q: returns the inferred float type for AD compatibility" begin
    # Phase-1 Newton uses ForwardDiff on the residual; D_t_q must
    # promote to the input element type and not pin to Float64. We
    # check Float32 → Float32 and Float64 → Float64 round-trips.
    let r32 = D_t_q(1.0f0, 1.1f0, 0.1f0, 1, 0.01f0)
        @test r32 isa Float32
    end
    let r64 = D_t_q(1.0, 1.1, 0.1, 1, 0.01)
        @test r64 isa Float64
    end
end

@testset "D_t_q: O(Δt) consistency with continuous time-derivative" begin
    # For a smooth field φ(t) = φ_0 + a·t and constant strain σ,
    # `D_t^{(q)} φ |_{n+1/2} = a + q σ (φ_0 + a·(t_n + Δt/2))`.
    # Our discrete form should match this exactly at midpoint.
    let phi_0 = 1.0, a = 0.5, σ = 0.3, q = 1, t_n = 0.4, dt = 0.1
        phi_n   = phi_0 + a * t_n
        phi_np1 = phi_0 + a * (t_n + dt)
        analytic = a + q * σ * (phi_0 + a * (t_n + dt / 2))
        @test isapprox(D_t_q(phi_n, phi_np1, σ, q, dt), analytic; atol = 1e-14)
    end
end
