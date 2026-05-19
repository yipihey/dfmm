# test_enzyme_gradient_phase2.jl
#
# Reverse-mode end-to-end gradient validation for the Phase-2
# multi-segment Newton solve. Periodic BC, no q-viscosity.
# Compares against high-order central differences on a scalar
# functional of the terminal state (mean squared deviation from a
# target profile).

using dfmm
using StaticArrays: SVector
using Test

@testset "Enzyme Phase-2 IFT pullback (mesh-level)" begin

    # Small periodic mesh fixture (N = 4 segments). Linear-acoustic
    # initial condition: uniform state at rest with a small velocity
    # perturbation that breaks the trivial-equilibrium.
    function build_state(N::Int; ε = 1e-3)
        Δm = fill(1.0 / N, N)
        # Cumulative positions for a uniform-density mesh on [0, 1].
        positions = collect(range(0.0, length = N, step = 1.0 / N))
        velocities = ε .* [sin(2π * j / N) for j in 0:N-1]
        αs = fill(0.9, N)
        βs = fill(0.0, N)
        ss = fill(0.05, N)
        y = Vector{Float64}(undef, 4N)
        @inbounds for j in 1:N
            y[4*(j-1) + 1] = positions[j]
            y[4*(j-1) + 2] = velocities[j]
            y[4*(j-1) + 3] = αs[j]
            y[4*(j-1) + 4] = βs[j]
        end
        return y, Δm, ss, 1.0
    end

    # Loss: ½ ‖y - y_target‖² — smooth, exercises all 4N output cotangent
    # components.
    function make_loss(N, target_seed)
        rng = target_seed
        ytgt = [0.5 + 0.1 * sin(rng + i) for i in 1:4N]
        return y -> 0.5 * sum((y .- ytgt).^2)
    end

    @testset "Single step (N = 4) — pullback vs central FD" begin
        N = 4
        y_0, Δm, s, L_box = build_state(N)
        dt = 5e-3
        loss = make_loss(N, 0.7)

        # Adjoint gradient.
        L, ȳ_0, Δm̄, s̄, L̄_box, d̄t =
            det_run_gradient(loss, y_0, Δm, s, L_box, dt, 1)

        # Forward simulation to verify loss agrees.
        y_1 = det_step_functional(y_0, Δm, s, L_box, dt)
        @test L ≈ loss(y_1) rtol = 1e-14

        # Central-difference reference.
        function fd_grad_vec(v0, perturb)
            h = 1e-6
            grad = similar(v0)
            for i in eachindex(v0)
                vp = copy(v0); vp[i] += h
                vm = copy(v0); vm[i] -= h
                grad[i] = (perturb(vp) - perturb(vm)) / (2h)
            end
            return grad
        end
        fd_y0 = fd_grad_vec(y_0, v -> loss(det_step_functional(
            v, Δm, s, L_box, dt)))
        fd_Δm = fd_grad_vec(Δm, v -> loss(det_step_functional(
            y_0, v, s, L_box, dt)))
        fd_s  = fd_grad_vec(s,  v -> loss(det_step_functional(
            y_0, Δm, v, L_box, dt)))
        # Scalars.
        h = 1e-6
        fd_L = (loss(det_step_functional(y_0, Δm, s, L_box + h, dt)) -
                loss(det_step_functional(y_0, Δm, s, L_box - h, dt))) / (2h)
        fd_t = (loss(det_step_functional(y_0, Δm, s, L_box, dt + h)) -
                loss(det_step_functional(y_0, Δm, s, L_box, dt - h))) / (2h)

        for i in eachindex(y_0)
            @test ȳ_0[i] ≈ fd_y0[i] atol = 1e-6
        end
        for i in eachindex(Δm)
            @test Δm̄[i] ≈ fd_Δm[i] atol = 1e-6
            @test s̄[i]  ≈ fd_s[i]  atol = 1e-6
        end
        @test L̄_box ≈ fd_L atol = 1e-6
        @test d̄t    ≈ fd_t atol = 1e-6
    end

    @testset "Multi-step (N = 4, n_steps = 4) — pullback vs central FD" begin
        N = 4
        y_0, Δm, s, L_box = build_state(N; ε = 5e-3)
        dt = 2e-3
        n_steps = 4
        loss = make_loss(N, 1.3)

        L, ȳ_0, Δm̄, s̄, L̄_box, d̄t =
            det_run_gradient(loss, y_0, Δm, s, L_box, dt, n_steps)

        function fd_grad_vec(v0, perturb)
            h = 1e-6
            grad = similar(v0)
            for i in eachindex(v0)
                vp = copy(v0); vp[i] += h
                vm = copy(v0); vm[i] -= h
                grad[i] = (perturb(vp) - perturb(vm)) / (2h)
            end
            return grad
        end
        function run_loss(y_, Δm_, s_, L_, dt_)
            y = y_
            for _ in 1:n_steps
                y = det_step_functional(y, Δm_, s_, L_, dt_)
            end
            return loss(y)
        end
        fd_y0 = fd_grad_vec(y_0, v -> run_loss(v, Δm, s, L_box, dt))
        fd_Δm = fd_grad_vec(Δm,  v -> run_loss(y_0, v, s, L_box, dt))
        fd_s  = fd_grad_vec(s,   v -> run_loss(y_0, Δm, v, L_box, dt))
        h = 1e-6
        fd_L = (run_loss(y_0, Δm, s, L_box + h, dt) -
                run_loss(y_0, Δm, s, L_box - h, dt)) / (2h)
        fd_t = (run_loss(y_0, Δm, s, L_box, dt + h) -
                run_loss(y_0, Δm, s, L_box, dt - h)) / (2h)

        for i in eachindex(y_0)
            @test ȳ_0[i] ≈ fd_y0[i] atol = 1e-5
        end
        for i in eachindex(Δm)
            @test Δm̄[i] ≈ fd_Δm[i] atol = 1e-5
            @test s̄[i]  ≈ fd_s[i]  atol = 1e-5
        end
        @test L̄_box ≈ fd_L atol = 1e-5
        @test d̄t    ≈ fd_t atol = 1e-5
    end
end
