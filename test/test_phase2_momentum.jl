# test_phase2_momentum.jl
#
# Phase 2 — translation invariance ⇒ total momentum conservation on
# periodic BC. The discrete EL system has a Noether symmetry under
# x_i → x_i + c (rigid translation of all vertices), since the
# pressures depend only on differences x_{j+1} − x_j and the kinetic
# term is a function of velocities only. The corresponding conserved
# charge is Σ_i m̄_i u_i (= total_momentum), which `det_step!` should
# preserve to round-off.
#
# Setup: small-amplitude perturbation around a uniform background,
# periodic BC, ≥ 100 timesteps.

using Test
using dfmm

@testset "Phase 2: momentum conservation (zero-momentum perturbation)" begin
    N = 32
    L = 1.0
    ρ0 = 1.0
    s0 = 0.0
    α0 = 1.0
    β0 = 0.0
    Δx = L / N
    Δm = fill(ρ0 * Δx, N)
    positions = collect((0:N-1) .* Δx) .+ 1e-4 .* sin.(2π .* (0:N-1) ./ N) .* Δx
    # Anti-symmetric velocity perturbation ⇒ initial total momentum = 0.
    velocities = 1e-3 .* cos.(2π .* (0:N-1) ./ N)
    αs = fill(α0, N)
    βs = fill(β0, N)
    ss = fill(s0, N)

    mesh = Mesh1D(positions, velocities, αs, βs, ss;
                  Δm = Δm, L_box = L, periodic = true)

    p0 = total_momentum(mesh)
    @test abs(p0) < 1e-12   # initial total momentum is round-off small

    dt = 5e-3
    N_steps = 100
    p_history = Float64[]
    for n in 1:N_steps
        det_step!(mesh, dt)
        push!(p_history, total_momentum(mesh))
    end

    # With p0 ≈ 0, assert absolute round-off bound. Per-step pressure
    # gradients are O(ε) and Newton tolerance is 1e-13; the variational
    # symmetry guarantees no secular drift, only round-off accumulation.
    @test maximum(abs.(p_history)) < 1e-12
end

@testset "Phase 2: momentum conservation (nonzero net momentum)" begin
    # Add a uniform velocity offset on top of the perturbation; the
    # total momentum is then m̄ ⋅ N ⋅ u_offset (= u_offset for ρ0=1, L=1).
    # Conservation should hold to relative round-off.
    N = 32
    L = 1.0
    ρ0 = 1.0
    s0 = 0.0
    α0 = 1.0
    β0 = 0.0
    Δx = L / N
    Δm = fill(ρ0 * Δx, N)
    positions = collect((0:N-1) .* Δx) .+ 1e-4 .* sin.(2π .* (0:N-1) ./ N) .* Δx
    u_offset = 0.05
    velocities = u_offset .+ 1e-3 .* cos.(2π .* (0:N-1) ./ N)
    αs = fill(α0, N)
    βs = fill(β0, N)
    ss = fill(s0, N)

    mesh = Mesh1D(positions, velocities, αs, βs, ss;
                  Δm = Δm, L_box = L, periodic = true)

    p0 = total_momentum(mesh)

    dt = 5e-3
    N_steps = 100
    drift = 0.0
    for n in 1:N_steps
        det_step!(mesh, dt)
        drift = max(drift, abs(total_momentum(mesh) - p0) / abs(p0))
    end

    @test drift < 1e-10
end
