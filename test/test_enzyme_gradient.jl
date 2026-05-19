# test_enzyme_gradient.jl
#
# Validates the Enzyme reverse-mode end-to-end gradient against high-order
# central finite differences on the Phase-1 Cholesky-sector integrator.
#
# Two configurations are exercised:
#   • Zero-strain (divu = 0): Phase-1 Hamiltonian-conserving regime.
#   • Uniform compression (divu < 0): non-trivial cross-derivatives on
#     all four input cotangents.
#
# Target tolerance: 1e-6 relative agreement with finite differences,
# slack enough for round-off on the FD side at h = 1e-5.

using dfmm
using StaticArrays: SVector
using Test

@testset "Enzyme end-to-end gradient (Cholesky-sector, IFT pullback)" begin

    # Loss functional: a simple smooth scalar of the terminal state.
    # `(α_N - α*)^2 + (β_N - β*)^2` exercises both components of q̄_N.
    α_target = 0.85
    β_target = -0.15
    loss = q -> (q[1] - α_target)^2 + (q[2] - β_target)^2

    function finite_diff_gradient(q_0, M_vv, divu, dt, N; h = 1e-5)
        # Returns (∂L/∂α_0, ∂L/∂β_0, ∂L/∂M_vv, ∂L/∂divu, ∂L/∂dt) via
        # 4th-order central differences.
        function L_eval(q0_, Mvv_, divu_, dt_)
            traj = cholesky_run(q0_, Mvv_, divu_, dt_, N)
            return loss(traj[end])
        end
        # Helper: 4th-order central diff in a single scalar parameter.
        function cd4(f, x, h)
            return (-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)) / (12h)
        end
        dLα = cd4(α -> L_eval(SVector{2,Float64}(α, q_0[2]), M_vv, divu, dt),
                  q_0[1], h)
        dLβ = cd4(β -> L_eval(SVector{2,Float64}(q_0[1], β), M_vv, divu, dt),
                  q_0[2], h)
        dLM = cd4(M -> L_eval(q_0, M, divu, dt), M_vv, h)
        dLd = cd4(d -> L_eval(q_0, M_vv, d, dt), divu, h)
        dLt = cd4(t -> L_eval(q_0, M_vv, divu, t), dt, h)
        return dLα, dLβ, dLM, dLd, dLt
    end

    @testset "Single step (N = 1, zero strain)" begin
        q_0 = SVector{2,Float64}(1.0, 0.0)
        M_vv, divu, dt, N = 1.0, 0.0, 0.05, 1
        L, q̄_0, M̄, d̄, t̄ = cholesky_run_gradient(loss, q_0, M_vv, divu, dt, N)
        fdα, fdβ, fdM, fdd, fdt = finite_diff_gradient(q_0, M_vv, divu, dt, N)
        # Sanity: forward loss matches.
        traj = cholesky_run(q_0, M_vv, divu, dt, N)
        @test L ≈ loss(traj[end]) rtol = 1e-14
        # Gradient agreement.
        @test q̄_0[1] ≈ fdα atol = 1e-7
        @test q̄_0[2] ≈ fdβ atol = 1e-7
        @test M̄      ≈ fdM atol = 1e-7
        @test d̄      ≈ fdd atol = 1e-7
        @test t̄      ≈ fdt atol = 1e-7
    end

    @testset "Multi-step (N = 4, zero strain)" begin
        q_0 = SVector{2,Float64}(1.0, 0.05)
        M_vv, divu, dt, N = 1.0, 0.0, 0.02, 4
        L, q̄_0, M̄, d̄, t̄ = cholesky_run_gradient(loss, q_0, M_vv, divu, dt, N)
        fdα, fdβ, fdM, fdd, fdt = finite_diff_gradient(q_0, M_vv, divu, dt, N)
        @test q̄_0[1] ≈ fdα rtol = 1e-6 atol = 1e-8
        @test q̄_0[2] ≈ fdβ rtol = 1e-6 atol = 1e-8
        @test M̄      ≈ fdM rtol = 1e-6 atol = 1e-8
        @test d̄      ≈ fdd rtol = 1e-6 atol = 1e-8
        @test t̄      ≈ fdt rtol = 1e-6 atol = 1e-8
    end

    @testset "Multi-step (N = 20, uniform compression)" begin
        q_0 = SVector{2,Float64}(1.0, 0.1)
        M_vv, divu, dt, N = 1.2, -0.3, 0.01, 20
        L, q̄_0, M̄, d̄, t̄ = cholesky_run_gradient(loss, q_0, M_vv, divu, dt, N)
        fdα, fdβ, fdM, fdd, fdt = finite_diff_gradient(q_0, M_vv, divu, dt, N)
        @test q̄_0[1] ≈ fdα rtol = 1e-6 atol = 1e-8
        @test q̄_0[2] ≈ fdβ rtol = 1e-6 atol = 1e-8
        @test M̄      ≈ fdM rtol = 1e-6 atol = 1e-8
        @test d̄      ≈ fdd rtol = 1e-6 atol = 1e-8
        @test t̄      ≈ fdt rtol = 1e-6 atol = 1e-8
    end

    @testset "Per-step pullback matches end-to-end driver (N = 3)" begin
        # Manually walk pullbacks step-by-step and compare to
        # cholesky_run_gradient. Catches accumulator-sign mistakes.
        q_0 = SVector{2,Float64}(1.0, 0.02)
        M_vv, divu, dt, N = 1.0, -0.1, 0.02, 3
        # Forward + collect pullbacks.
        traj = Vector{SVector{2,Float64}}(undef, N + 1)
        pulls = Vector{Function}(undef, N)
        traj[1] = q_0
        for n in 1:N
            traj[n+1], pulls[n] = cholesky_step_pullback(traj[n], M_vv, divu, dt)
        end
        L_manual = loss(traj[end])
        # Loss-gradient seed at q_N.
        q̄ = SVector{2,Float64}(2(traj[end][1] - α_target),
                                2(traj[end][2] - β_target))
        M̄, d̄, t̄ = 0.0, 0.0, 0.0
        for n in N:-1:1
            q̄n, ΔM, Δd, Δt = pulls[n](q̄)
            q̄ = q̄n
            M̄ += ΔM; d̄ += Δd; t̄ += Δt
        end
        # Compare to the high-level driver.
        L_driver, q̄_0, M̄_d, d̄_d, t̄_d =
            cholesky_run_gradient(loss, q_0, M_vv, divu, dt, N)
        @test L_manual ≈ L_driver rtol = 1e-14
        @test q̄[1] ≈ q̄_0[1] rtol = 1e-12
        @test q̄[2] ≈ q̄_0[2] rtol = 1e-12
        @test M̄    ≈ M̄_d    rtol = 1e-12 atol = 1e-14
        @test d̄    ≈ d̄_d    rtol = 1e-12 atol = 1e-14
        @test t̄    ≈ t̄_d    rtol = 1e-12 atol = 1e-14
    end
end

@testset "Enzyme forward-mode JVP (Cholesky-sector, IFT pushforward)" begin

    @testset "Single step: JVP matches central FD" begin
        q_0 = SVector{2,Float64}(1.0, 0.05)
        M_vv, divu, dt = 1.1, -0.2, 0.04
        q̇_0 = SVector{2,Float64}(0.7, -0.3)
        Ṁ, ḋ, ṫ = 0.4, -0.6, 0.5
        _, q̇_1 = cholesky_step_jvp(q_0, M_vv, divu, dt, q̇_0, Ṁ, ḋ, ṫ)
        h = 1e-6
        q_1_plus  = cholesky_step(q_0 + h * q̇_0, M_vv + h * Ṁ,
                                  divu + h * ḋ, dt + h * ṫ)
        q_1_minus = cholesky_step(q_0 - h * q̇_0, M_vv - h * Ṁ,
                                  divu - h * ḋ, dt - h * ṫ)
        q̇_fd = (q_1_plus - q_1_minus) / (2h)
        @test q̇_1[1] ≈ q̇_fd[1] atol = 1e-8
        @test q̇_1[2] ≈ q̇_fd[2] atol = 1e-8
    end

    @testset "Forward-reverse duality (single step)" begin
        # End-to-end identity at one step:
        #   ⟨q̄_np1, q̇_np1⟩ = ⟨q̄_n, q̇_n⟩ + M̄·Ṁ + d̄·ḋ + t̄·ṫ.
        q_n = SVector{2,Float64}(1.0, -0.05)
        M_vv, divu, dt = 1.05, 0.15, 0.03
        q̇_n = SVector{2,Float64}(0.3, 0.9)
        Ṁ, ḋ, ṫ = -0.7, 0.2, 1.1
        _, q̇_np1 = cholesky_step_jvp(q_n, M_vv, divu, dt, q̇_n, Ṁ, ḋ, ṫ)
        q̄_np1 = SVector{2,Float64}(0.42, -1.7)
        _, pullback = cholesky_step_pullback(q_n, M_vv, divu, dt)
        q̄_n, M̄, d̄, t̄ = pullback(q̄_np1)
        lhs = q̄_np1[1] * q̇_np1[1] + q̄_np1[2] * q̇_np1[2]
        rhs = q̄_n[1] * q̇_n[1] + q̄_n[2] * q̇_n[2] + M̄ * Ṁ + d̄ * ḋ + t̄ * ṫ
        @test lhs ≈ rhs rtol = 1e-12
    end

    @testset "Multi-step JVP propagates correctly (N = 15)" begin
        q_0 = SVector{2,Float64}(1.0, 0.04)
        M_vv, divu, dt, N = 1.0, -0.2, 0.01, 15
        q̇_0 = SVector{2,Float64}(0.5, -0.2)
        Ṁ, ḋ, ṫ = 0.3, 0.1, -0.4
        traj, tdot = cholesky_run_jvp(q_0, M_vv, divu, dt, N, q̇_0, Ṁ, ḋ, ṫ)
        h = 1e-6
        traj_p = cholesky_run(q_0 + h*q̇_0, M_vv + h*Ṁ, divu + h*ḋ,
                              dt + h*ṫ, N)
        traj_m = cholesky_run(q_0 - h*q̇_0, M_vv - h*Ṁ, divu - h*ḋ,
                              dt - h*ṫ, N)
        q̇_N_fd = (traj_p[end] - traj_m[end]) / (2h)
        @test tdot[end][1] ≈ q̇_N_fd[1] atol = 1e-7
        @test tdot[end][2] ≈ q̇_N_fd[2] atol = 1e-7
    end
end
