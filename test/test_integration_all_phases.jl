# test_integration_all_phases.jl
#
# Smoke integration test exercising the *combination* of all dfmm phase
# features simultaneously. Per-phase tests cover their own paths in
# isolation; this file catches *inter-phase* regressions — for example,
# the Phase-7-extends-DetField vs Phase-8-mutates-DetField defect that
# the original parallel-phase merge did not expose because each
# phase's tests skipped the other phase's DOFs.
#
# The test is deliberately a smoke test: it asserts that the combined
# integrator path runs without errors and produces sane state, not that
# the result matches any particular reference. Detailed numerical
# assertions live in the per-phase tests.

using dfmm
using dfmm: Mesh1D, DetField, total_mass, total_momentum, total_energy
using Random
using Test

@testset "Cross-phase smoke: every DetField field, every integrator path" begin
    # ─────────────────────────────────────────────────────────────────
    # Build a mesh with every DetField field non-trivially populated.
    # Phase-1 (α, β), Phase-2 (x, u, s), Phase-5 (Pp deviatoric),
    # Phase-7 (Q heat flux). All present + nonzero. Periodic BCs so
    # we don't entangle Phase-7's inflow/outflow with the smoke check.
    # ─────────────────────────────────────────────────────────────────
    N = 16
    L = 1.0
    Δx = L / N
    positions = collect(range(0.0, L - Δx; length = N))

    # Smooth small-amplitude acoustic perturbation so all paths stay in
    # the well-resolved regime (no shocks, no caustics).
    velocities = [0.01 * sin(2π * x / L) for x in positions]
    αs  = fill(0.5,   N)
    βs  = fill(0.001, N)              # nonzero β  (Phase 1 charge-1 sector)
    ss  = fill(0.0,   N)              # M_vv = 1 at J = 1
    Pps = fill(0.4,   N)              # nonzero Pp (Phase 5 deviatoric)
    Qs  = fill(0.05,  N)              # nonzero Q  (Phase 7 heat flux)
    Δm_vec = fill(Δx, N)

    @testset "Mesh constructor accepts all 7 DetField fields" begin
        mesh = Mesh1D(positions, velocities, αs, βs, ss;
                      Δm = Δm_vec, Pps = Pps, Qs = Qs,
                      L_box = L, periodic = true)
        @test n_segments(mesh) == N
        # All 7 DetField fields populated, none default-NaN
        for j in 1:N
            seg = mesh.segments[j].state
            @test isfinite(seg.x)
            @test isfinite(seg.u)
            @test isfinite(seg.α)
            @test isfinite(seg.β)
            @test isfinite(seg.s)
            @test isfinite(seg.Pp)
            @test isfinite(seg.Q)
            @test seg.Q ≈ 0.05
            @test seg.Pp ≈ 0.4
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Path A: deterministic step with q=:vNR + heat flux + tracers
    # carried alongside. Exercises Phase-2 + 5 + 5b + 7 + 11 jointly.
    # ─────────────────────────────────────────────────────────────────
    @testset "Path A: det_step! with q=:vNR + Q-BGK + tracers" begin
        mesh = Mesh1D(positions, velocities, αs, βs, ss;
                      Δm = Δm_vec, Pps = Pps, Qs = Qs,
                      L_box = L, periodic = true)
        tm = TracerMesh(mesh, ones(2, N), [:step, :sin])
        # Sharp step at m = 0.5
        for j in 1:N
            tm.tracers[1, j] = j ≤ N ÷ 2 ? 0.0 : 1.0
            tm.tracers[2, j] = sin(2π * (j - 0.5) / N)
        end
        tracers0 = copy(tm.tracers)

        dt = 1e-4
        for _ in 1:5
            det_step!(mesh, dt;
                      tau = 1e-3,
                      q_kind = :vNR_linear_quadratic,
                      c_q_quad = 2.0,
                      c_q_lin = 1.0)
            advect_tracers!(tm, dt)
        end

        # All 7 fields still populated and finite after 5 steps
        for j in 1:N
            seg = mesh.segments[j].state
            @test isfinite(seg.x) && isfinite(seg.u) && isfinite(seg.α)
            @test isfinite(seg.β) && isfinite(seg.s) && isfinite(seg.Pp)
            @test isfinite(seg.Q)
        end
        # Tracers are bit-exact (Phase 11 invariant; must hold under
        # every other-phase coupling)
        @test tm.tracers == tracers0
        # Q decays via BGK (every cell): Q(t=5dt) < Q(0) for τ << 5dt
        # τ = 1e-3, dt = 1e-4, 5 steps → t = 5e-4 = 0.5 τ
        # Q(0.5τ)/Q(0) = exp(-0.5) ≈ 0.6
        for j in 1:N
            @test mesh.segments[j].state.Q < 0.05  # has decayed
            @test mesh.segments[j].state.Q > 0.02  # not below predicted
        end
        # Mass conservation (per-segment Δm fixed by construction)
        @test total_mass(mesh) ≈ sum(Δm_vec)
    end

    # ─────────────────────────────────────────────────────────────────
    # Path B: stochastic injection on a mesh with Q+Pp populated.
    # This is the *exact* path that exposed the Phase-7/8 inter-phase
    # regression — Phase 8's inject_vg_noise! mutated DetField{T}(...)
    # via the 6-arg form which no longer existed after Phase 7's
    # struct extension. Asserting the path runs is itself the test.
    # ─────────────────────────────────────────────────────────────────
    @testset "Path B: det_run_stochastic! mutates all 7 DetField fields" begin
        mesh = Mesh1D(positions, velocities, αs, βs, ss;
                      Δm = Δm_vec, Pps = Pps, Qs = Qs,
                      L_box = L, periodic = true)
        rng = MersenneTwister(20260425)
        params = dfmm.NoiseInjectionParams(C_A = 0.1, C_B = 0.1, λ = 1.6)
        dt = 1e-4
        # Run 5 stochastic steps. The DetField{T} mutation in
        # inject_vg_noise! will fail with MethodError on the 6-arg
        # constructor if a future phase extends the struct without
        # updating that call site.
        det_run_stochastic!(mesh, dt, 5; rng = rng, params = params)
        # All 7 fields still populated
        for j in 1:N
            seg = mesh.segments[j].state
            @test isfinite(seg.x) && isfinite(seg.u) && isfinite(seg.α)
            @test isfinite(seg.β) && isfinite(seg.s) && isfinite(seg.Pp)
            @test isfinite(seg.Q)
        end
        # Mass exact
        @test total_mass(mesh) ≈ sum(Δm_vec)
        # Q field carried unchanged through stochastic injection
        # (noise modifies (ρu, P_xx, P_⊥, s); Phase 8 inject does not
        # touch Q). Confirms Phase 8's preservation of the Phase 7
        # field after the inter-phase fix.
        for j in 1:N
            @test mesh.segments[j].state.Q ≈ 0.05  rtol=1e-12
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Path C: arity-defensive — every DetField{T} parametric call site
    # in the codebase passes 7 args. If any future field is added to
    # the struct, this test will fail at the constructor on Path A or B
    # above. Document the property explicitly.
    # ─────────────────────────────────────────────────────────────────
    @testset "DetField struct arity: synced across all phases" begin
        # Float64 7-arg parametric constructor exists
        f = DetField{Float64}(1.0, 0.5, 0.2, 0.001, 0.0, 0.4, 0.05)
        @test f.x == 1.0 && f.u == 0.5 && f.α == 0.2
        @test f.β == 0.001 && f.s == 0.0 && f.Pp == 0.4 && f.Q == 0.05
        # Convenience non-parametric constructors with default-Q exist
        # (used by user-facing test setup code)
        f6 = DetField(1.0, 0.5, 0.2, 0.001, 0.0, 0.4)
        @test f6.Q == 0.0
        f5 = DetField(1.0, 0.5, 0.2, 0.001, 0.0)
        @test isnan(f5.Pp) || f5.Pp == 0.0   # Phase-1/2 sentinel acceptable
        @test f5.Q == 0.0
    end
end
