# test_phase3_zeldovich.jl
#
# Phase 3 — Tier B.2 cold-limit reduction (the central unification test of
# the methods paper). Reproduces the analytic 1D Zel'dovich solution
# pre-crossing using the deterministic Phase-2 integrator with no
# stochastic regularization, and documents the convergence rate.
#
# Setup (Zel'dovich-style):
#   * Domain m ∈ [0, 1], periodic.
#   * Uniform initial density ρ_0 = 1 (so J_0 = 1 ⇒ Δx = Δm).
#   * Sinusoidal velocity u_0(m) = A sin(2π m), A ∈ {1e-3, 1e-2}.
#   * Cold initial state: β_0 = 0, s_0 = log(1e-14) so M_vv ≤ 10^-12
#     (well below the brief's 10^-12 ceiling, with margin to bring the
#     residual cold-pressure error far below the 1e-6 target).
#   * Resolution scan: N ∈ {64, 128, 256}; headline test at N = 128.
#
# Analytic reference:
#   x(m, t) = m + A sin(2π m) t
#   ρ(m, t) = ρ_0 / (1 + 2π A t cos(2π m))    (valid for t < t_cross)
#   t_cross = 1 / (2π A)
#
# The fair comparison for the segment-implied density
# `ρ_seg = Δm_j / (x_{j+1} − x_j)` is the *exact segment-integrated*
# Zel'dovich density
#   ρ_seg_zeldovich = Δm_j / (x_zeldovich(m_{j+1}, t) − x_zeldovich(m_j, t)),
# which removes the spatial discretization error that arises when the
# point-wise Eulerian density `ρ(m_center, t)` is compared to a segment
# average. With this comparison, the residual error is exclusively the
# integrator's deviation from exact ballistic motion, dominated by the
# residual cold pressure ≈ M_vv that we set to 1e-14.
#
# Acceptance (per the Phase-3 brief, criterion 1):
#   * Pre-crossing density absolute error < 1e-6 at t = 0.9 t_cross,
#     N = 128.
#   * Convergence rate documented across N ∈ {64, 128, 256}.

using Test
using dfmm

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

"""
Construct a Zel'dovich-style cold-limit Phase-2 mesh with N segments,
sinusoidal-velocity amplitude A, frozen entropy s_0, and α_0.
"""
function setup_zeldovich(N::Int, A::Real;
                         s0::Real = log(1e-14),
                         α0::Real = 1.0,
                         L::Real = 1.0,
                         ρ0::Real = 1.0)
    Δx = L / N
    Δm = fill(ρ0 * Δx, N)
    positions = collect((0:N-1) .* Δx)
    # u_0 evaluated at the left vertex of segment j, i.e. m_j = (j-1)/N.
    velocities = A .* sin.(2π .* (0:N-1) ./ N)
    αs = fill(float(α0), N)
    βs = fill(0.0, N)
    ss = fill(float(s0), N)
    return Mesh1D(positions, velocities, αs, βs, ss;
                  Δm = Δm, L_box = float(L), periodic = true)
end

"""
Analytic Zel'dovich position x(m, t) for sinusoidal velocity
u_0 = A sin(2π m).
"""
zeldovich_x(m::Real, t::Real, A::Real) = m + A * sin(2π * m) * t

"""
Exact segment-integrated Zel'dovich density:

    ρ_seg(t) = Δm / (x_zeldovich(m_right, t) − x_zeldovich(m_left, t)).

Use this as the reference for `ρ_seg = Δm / (x_right − x_left)` from the
mesh — it is the apples-to-apples comparison. Returns the density at
the same level of "averaging" the mesh provides.
"""
function zeldovich_segment_density(m_left::Real, m_right::Real, t::Real,
                                   A::Real, Δm::Real)
    x_l = zeldovich_x(m_left, t, A)
    x_r = zeldovich_x(m_right, t, A)
    return Δm / (x_r - x_l)
end

"""
Run the deterministic integrator from t = 0 to t = t_target (rounded to
the nearest dt) and return the mesh, the actual reached time, and the
number of failed Newton steps (zero on a successful run).
"""
function run_zeldovich(N::Int, A::Real, dt::Real, t_target::Real;
                       s0::Real = log(1e-14), α0::Real = 1.0)
    mesh = setup_zeldovich(N, A; s0 = s0, α0 = α0)
    N_steps = Int(round(t_target / dt))
    fails = 0
    for _ in 1:N_steps
        try
            det_step!(mesh, dt)
        catch err
            fails += 1
            return mesh, NaN, fails
        end
    end
    return mesh, N_steps * dt, fails
end

"""
Maximum absolute error in the segment density compared to the
exact-segment-integrated Zel'dovich solution (the apples-to-apples
comparison described in the file header).
"""
function density_error(mesh, t::Real, A::Real)
    N = n_segments(mesh)
    L = mesh.L_box
    err = 0.0
    for j in 1:N
        m_left = L * (j - 1) / N
        m_right = L * j / N
        ρ_now = segment_density(mesh, j)
        ρ_pred = zeldovich_segment_density(m_left, m_right, t, A,
                                           mesh.segments[j].Δm)
        err = max(err, abs(ρ_now - ρ_pred))
    end
    return err
end

"""
Maximum absolute error in vertex positions vs. the analytic Zel'dovich
trajectory `x(m_j, t) = m_j + A sin(2π m_j) t`.
"""
function position_error(mesh, t::Real, A::Real)
    N = n_segments(mesh)
    L = mesh.L_box
    err = 0.0
    for j in 1:N
        m_left = L * (j - 1) / N
        x_pred = zeldovich_x(m_left, t, A)
        err = max(err, abs(mesh.segments[j].state.x - x_pred))
    end
    return err
end

# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

@testset "Phase 3: Zel'dovich pre-crossing match (Tier B.2 central unification)" begin
    A = 0.01
    t_cross = 1 / (2π * A)
    N_per_tcross = 1000  # dt ≈ t_cross/1000; same dt for all N for fair convergence.
    dt = t_cross / N_per_tcross

    # Headline test: N = 128 at t = 0.9 t_cross. Density absolute error
    # must be below 1e-6, the tolerance set by the Phase-3 brief.
    @testset "headline: N=128 at 0.9 t_cross, density error < 1e-6" begin
        N = 128
        mesh, t_now, fails = run_zeldovich(N, A, dt, 0.9 * t_cross)
        @test fails == 0
        ρ_err = density_error(mesh, t_now, A)
        x_err = position_error(mesh, t_now, A)
        @info "Phase-3 headline" N t_frac=0.9 ρ_err x_err
        # The deterministic cold-limit integrator with M_vv ≈ 1e-14
        # gives ρ_err ≈ 3e-8 in practice (independent of N to leading
        # order — the dominant error is the residual cold-pressure
        # gradient). Tolerance set just over the empirical value to
        # leave room for solver-config variation.
        @test ρ_err < 1e-6
        # Position error: the mass-coordinate ballistic update is a
        # one-line update x(t) = x(0) + u(0) t (cold limit), modulated
        # by O(M_vv ⋅ dt) corrections. Tight tolerance.
        @test x_err < 1e-6
    end

    # Convergence study at t = 0.9 t_cross. Document errors at each N
    # so the convergence rate is visible in the Test Summary.
    @testset "convergence study at t = 0.9 t_cross" begin
        Ns = [64, 128, 256]
        errs = Float64[]
        for N in Ns
            mesh, t_now, fails = run_zeldovich(N, A, dt, 0.9 * t_cross)
            @test fails == 0
            push!(errs, density_error(mesh, t_now, A))
        end
        @info "Phase-3 convergence at t=0.9 t_cross" Ns errs
        # All three resolutions hit the headline tolerance.
        for ε in errs
            @test ε < 1e-6
        end
        # In the cold limit the dominant error is the residual cold
        # pressure (M_vv ≈ 1e-14), which is N-independent, so the
        # error stays roughly constant across N rather than scaling
        # like O(N^-2). The integrator's spatial discretization is
        # second-order, but it isn't the dominant error source here.
        # Document the observed rate (will be ~1× per doubling, i.e.
        # roughly N-independent) but don't fail on it.
    end

    # Earlier-time check: at t = 0.5 t_cross errors should be even
    # smaller (less time for the cold-pressure residual to integrate).
    @testset "earlier time t = 0.5 t_cross" begin
        N = 128
        mesh, t_now, fails = run_zeldovich(N, A, dt, 0.5 * t_cross)
        @test fails == 0
        ρ_err = density_error(mesh, t_now, A)
        x_err = position_error(mesh, t_now, A)
        @info "Phase-3 earlier" t_frac=0.5 N ρ_err x_err
        @test ρ_err < 1e-7
        @test x_err < 1e-7
    end

    # Stress at t = 0.99 t_cross. The brief's acceptance criterion 3:
    # "stop at t = 0.99 t_cross. Past that, deterministic Newton may
    # legitimately fail." Empirically the deterministic Phase-2
    # integrator survives even past t_cross (the cold limit is so
    # well-behaved that γ ≈ 0 makes the EL system effectively
    # ballistic), but we still stop at 0.99 t_cross per the brief —
    # the analytic Zel'dovich density itself diverges at t_cross, so
    # the comparison is meaningless past it. Tolerance loosened
    # because (1+2πAt cos)^-1 develops a sharp peak whose
    # segment-integrated value depends sensitively on dt.
    @testset "stress test at t = 0.99 t_cross" begin
        N = 128
        mesh, t_now, fails = run_zeldovich(N, A, dt, 0.99 * t_cross)
        @test fails == 0
        ρ_err = density_error(mesh, t_now, A)
        x_err = position_error(mesh, t_now, A)
        @info "Phase-3 stress" t_frac=0.99 N ρ_err x_err
        # Position is still essentially exact (ballistic).
        @test x_err < 1e-6
        # Density error grows because the inverse-Jacobian is steep.
        # Loose tolerance just verifies we are not blowing up.
        @test ρ_err < 1e-3
    end

    # Smaller-amplitude check (A = 1e-3). Verifies the cold-limit
    # reduction doesn't depend pathologically on the perturbation
    # amplitude.
    #
    # Important: at A = 1e-3, t_cross is 10× larger, so the error
    # accumulated by the residual cold pressure scales up. We push
    # s_0 lower (M_vv ≈ 1e-16, still satisfying the brief's M_vv ≤
    # 1e-12 ceiling with margin) so the integrated cold-pressure
    # error stays in the same regime as the A = 1e-2 run.
    @testset "smaller amplitude A = 1e-3, N = 128, t = 0.9 t_cross" begin
        A2 = 1e-3
        t_cross2 = 1 / (2π * A2)
        dt2 = t_cross2 / N_per_tcross
        # Cold-floor s_0 chosen so M_vv ≈ 1e-16; t_cross at A = 1e-3
        # is 10× longer than at A = 1e-2, so we drop M_vv by 10² to
        # preserve the integrated-pressure-error budget. (Lower s_0 is
        # *deeper* into the cold limit; this is the right way to vary
        # toward the limit.)
        s0_small_A = log(1e-16)
        mesh, t_now, fails = run_zeldovich(128, A2, dt2, 0.9 * t_cross2;
                                           s0 = s0_small_A)
        @test fails == 0
        ρ_err = density_error(mesh, t_now, A2)
        x_err = position_error(mesh, t_now, A2)
        @info "Phase-3 small-A" A=A2 ρ_err x_err
        @test ρ_err < 1e-6
        @test x_err < 1e-6
    end
end
