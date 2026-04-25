# test_phase2_free_streaming.jl
#
# Phase 2 — cold-limit deterministic free streaming (Zel'dovich-style
# IC, *pre-crossing*). With γ → 0 (here implemented as `s` very
# negative ⇒ M_vv at floor ≈ 1e-12), pressure gradients vanish and the
# bulk EL reduces to ẍ = 0 ⇒ x_i(t) = x_i(0) + u_i(0)·t exactly.
#
# This is *not* the Phase-3 cold-limit test. Phase 3 handles the
# Hessian degeneracy at exact γ = 0 and tests that the integrator
# survives the caustic (multi-stream emergence). Phase 2 stays well
# pre-crossing, with a small-amplitude perturbation, and only
# verifies the ballistic x(t) trajectory.
#
# Stop time. Caustic time for a sinusoidal velocity perturbation
# u_0(m) = A sin(2π m/L) is t_cross = L / (2π A) (the time at which
# the gradient ∂u/∂m equals 1, i.e. neighboring vertices begin to
# overtake). We pick A small and stop well before this.

using Test
using dfmm

@testset "Phase 2: free-streaming (cold limit, pre-crossing)" begin
    N = 16
    L = 1.0
    ρ0 = 1.0
    α0 = 1.0
    β0 = 0.0
    # `s` chosen so that exp(s/cv) ≈ 1e-12 ⇒ M_vv ≈ 1e-12 at J = 1.
    s0 = log(1e-12)   # ≈ -27.63
    Δx = L / N

    # Zel'dovich-style IC: uniform mass, sinusoidal velocity.
    A = 0.01            # small amplitude
    Δm = fill(ρ0 * Δx, N)
    positions = collect((0:N-1) .* Δx)
    # u_0 evaluated at the left vertex of segment i.
    velocities = A .* sin.(2π .* (0:N-1) ./ N)
    αs = fill(α0, N)
    βs = fill(β0, N)
    ss = fill(s0, N)

    mesh = Mesh1D(positions, velocities, αs, βs, ss;
                  Δm = Δm, L_box = L, periodic = true)

    x0 = [seg.state.x for seg in mesh.segments]
    u0 = [seg.state.u for seg in mesh.segments]

    dt = 1e-3
    N_steps = 100
    T_end = N_steps * dt   # = 0.1
    # Stop time: caustic time t_cross = L / (2π A) ≈ 15.9. We're at
    # T_end = 0.1, well below. (Documented stop time.)
    @test T_end < L / (2π * A) / 10   # safety: an order of magnitude below caustic.

    for n in 1:N_steps
        det_step!(mesh, dt)
    end

    # Compare positions to ballistic prediction.
    max_err = 0.0
    for i in 1:N
        x_pred = x0[i] + u0[i] * T_end
        x_now = mesh.segments[i].state.x
        max_err = max(max_err, abs(x_now - x_pred))
    end

    # Tolerance: cold M_vv ~ 1e-12 produces residual pressure gradient
    # of order ρ M_vv / Δx ~ 1.6e-11. Integrated over T_end = 0.1 with
    # the second-order midpoint integrator, the position error is
    # ~ Δp/m · T = M_vv · T = 1e-13. Allow 1e-10 to cover Newton
    # tolerance (1e-13 per step × 100 steps).
    @test max_err < 1e-10

    # Velocities should have stayed essentially constant (cold flow).
    max_du = 0.0
    for i in 1:N
        max_du = max(max_du, abs(mesh.segments[i].state.u - u0[i]))
    end
    @test max_du < 1e-10
end
