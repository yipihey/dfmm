# IC sanity tests for the six Tier-A factories in `src/setups.jl`.
#
# These tests are intentionally cheap: shape, type, and conservation
# invariants at t = 0; algebraic identities that the factory must
# preserve. They do *not* re-run py-1d (that's `scripts/make_goldens.jl`,
# whose output is asserted against in the future Phase 5+ regression
# tests once the Julia integrator lands).
#
# # What the regression tests will assert later (Phases 5–13)
#
# Every factory's t = 0 snapshot is also embedded inside the matching
# `reference/golden/<name>.h5`. The Phase 5 regression tests will load
# each golden, compare the t = 0 snapshot to the Julia factory's
# output (asserting bit-equality for non-stochastic ICs, RMS-only for
# the wave-pool which has a random phase draw), and then time-step
# forward in Julia and compare the final-time snapshot.
#
# This file only runs the IC half of that loop — i.e. without
# requiring the goldens to be present. So `Pkg.test()` is green even
# in a fresh checkout where `make goldens` has not yet been run.

using Test
using dfmm

# Convenience: allow round-off slack proportional to N to cover sums.
_eps(N::Integer) = 16 * eps(Float64) * N


@testset "setups — Tier-A initial conditions" begin

    # ----- A.1 Sod ------------------------------------------------------
    @testset "A.1 setup_sod" begin
        ic = setup_sod(N=400)
        @test ic.name == "sod"
        @test length(ic.x) == 400
        @test length(ic.rho) == 400
        @test length(ic.u) == 400
        @test length(ic.P) == 400
        @test eltype(ic.rho) == Float64
        @test eltype(ic.u) == Float64

        # Domain layout: half-shifted periodic-style cell centers on [0, 1].
        @test ic.x[1] ≈ 0.5/400 atol=_eps(400)
        @test ic.x[end] ≈ 1 - 0.5/400 atol=_eps(400)

        # Two-state IC values
        @test ic.rho[1] == 1.0       # left half
        @test ic.rho[end] == 0.125   # right half
        @test ic.P[1] == 1.0
        @test ic.P[end] == 0.1

        # Conservation invariants at t = 0
        @test sum(ic.u .* ic.rho) ≈ 0.0  atol=_eps(400)            # zero net momentum
        @test all(iszero, ic.u)
        @test all(iszero, ic.beta_init)
        @test all(iszero, ic.Q)

        # Cholesky γ from primitives.
        @test all(ic.gamma_init .≈ sqrt.(ic.P ./ ic.rho))
        @test all(ic.alpha_init .== 0.02)

        # Pxx == Pp == P at t = 0 (isotropic Maxwellian).
        @test all(ic.Pxx .== ic.P)
        @test all(ic.Pp .== ic.P)

        # Default save_times = [t_end].
        @test ic.params.t_end == 0.2
        @test ic.params.save_times == [0.2]
    end

    # ----- A.2 cold sinusoid -------------------------------------------
    @testset "A.2 setup_cold_sinusoid — cold-limit γ → 0" begin
        ic = setup_cold_sinusoid(N=400)
        @test ic.name == "cold_sinusoid"
        @test length(ic.rho) == 400
        @test all(ic.rho .== 1.0)             # uniform density
        @test ic.P[1] == 1.0 * 1e-3            # rho0 * T0 (default T0=1e-3)

        # Symmetry: the velocity is sin(2π x) at x_i = (i + 0.5)/N.
        # Sum over a uniform cell-centered grid is exactly zero (odd
        # function with N even or odd; numpy's discrete sum is 0).
        @test abs(sum(ic.u)) < _eps(length(ic.u))   # ≈ 0
        @test abs(sum(ic.u .* ic.rho)) < _eps(length(ic.u))   # zero net momentum

        # γ_init = sqrt(P/ρ) for β=0; T0 = 1e-3 by default.
        @test all(ic.gamma_init .≈ sqrt(1e-3))

        # Cold-limit asymptote: as T0 → 0, γ_init → 0 to machine ε.
        ic_cold = setup_cold_sinusoid(N=128, T0=eps(Float64))
        @test maximum(ic_cold.gamma_init) ≤ sqrt(eps(Float64)) * 1.01

        # The factory must allow setting T0 strictly to 0 — the test
        # then asserts γ_init is exactly 0 (the central B.2 test
        # checkpoint).
        ic_zero = setup_cold_sinusoid(N=64, T0=0.0)
        @test all(iszero, ic_zero.gamma_init)
        @test all(iszero, ic_zero.beta_init)
    end

    # ----- A.3 steady shock --------------------------------------------
    @testset "A.3 setup_steady_shock — Rankine–Hugoniot" begin
        ic = setup_steady_shock(M1=2.0, N=400)
        @test ic.name == "steady_shock"
        @test ic.M1 == 2.0

        # Rankine–Hugoniot at γ = 5/3, M = 2.0: ρ2/ρ1 = 8/3 / (1/3 * 4 + 2)
        # = (8/3)/(10/3) — but per the factory's exact algebra:
        #   ρ2/ρ1 = (γ+1)*M² / ((γ-1)*M² + 2)
        # With γ=5/3, M=2: (8/3 * 4) / (2/3 * 4 + 2) = (32/3) / (14/3) = 32/14
        @test ic.rho2 / ic.rho1 ≈ 32/14 atol=1e-12

        # Pre-shock primitives in left half, post-shock in right half.
        @test ic.rho[1] == ic.rho1
        @test ic.rho[end] == ic.rho2
        @test ic.u[1] == ic.u1
        @test ic.u[end] == ic.u2
        @test ic.P[1] == ic.P1
        @test ic.P[end] == ic.P2

        # Inflow record carries the upstream state (reused by py-1d to
        # drive the inflow Dirichlet BC).
        @test ic.inflow.rho == ic.rho1
        @test ic.inflow.u == ic.u1
        @test ic.inflow.P == ic.P1
        @test ic.inflow.M3 ≈ ic.rho1*ic.u1^3 + 3*ic.u1*ic.P1

        # Net mass should equal ∑ ρ Δm; with Δx = 1/(N-1) the discrete sum
        # is just a regression on the discontinuity location.
        # We assert: mass-fractions split at i=N/2.
        @test ic.rho[1:200] == fill(ic.rho1, 200)
    end

    # ----- A.4 KM-LES wave-pool ----------------------------------------
    @testset "A.4 setup_kmles_wavepool — RMS normalization" begin
        ic = setup_kmles_wavepool(N=256, u0=1.0, P0=0.1, K_max=16, seed=42)
        @test ic.name == "wavepool"
        @test length(ic.u) == 256
        # The factory normalizes u to unit RMS = u0.
        @test sqrt(sum(ic.u.^2)/length(ic.u)) ≈ 1.0 atol=1e-12
        @test all(ic.rho .== 1.0)
        @test all(ic.P .== 0.1)
        @test all(ic.beta_init .== 0.0)
        @test length(ic.phases) == 16

        # Periodic-style cell centers.
        @test ic.x[1] ≈ 0.5/256 atol=_eps(256)

        # Different seed ⇒ different phases ⇒ different u realization.
        ic2 = setup_kmles_wavepool(N=256, seed=43)
        @test ic.phases != ic2.phases
        @test ic.u != ic2.u
        # Same RMS though.
        @test sqrt(sum(ic2.u.^2)/length(ic2.u)) ≈ 1.0 atol=1e-12

        # save_times array matches n_snaps cadence.
        @test length(ic.params.save_times) == ic.params.n_snaps
        @test ic.params.save_times[end] ≈ ic.params.t_end atol=1e-12
    end

    # ----- A.5 two-fluid dust-in-gas -----------------------------------
    @testset "A.5 setup_dust_in_gas — two-species primitives" begin
        ic = setup_dust_in_gas(N=400)
        @test ic.name == "dust_in_gas"
        @test length(ic.rho_d) == 400
        @test length(ic.rho_g) == 400

        # Initial densities.
        @test all(ic.rho_d .== 0.1)              # dust_to_gas default
        @test all(ic.rho_g .== 1.0)              # gas density default

        # Same sinusoidal velocity for both species at t = 0.
        @test ic.u_d == ic.u_g
        @test maximum(abs, ic.u_d) ≤ 1.0  # |sin| ≤ 1 with default A=1.0

        # Per-species pressures.
        @test all(ic.P_d .≈ 0.1 * 1e-5)          # dust: T0_dust = 1e-5
        @test all(ic.P_g .≈ 1.0 * 1e-3)          # gas:  T0_gas  = 1e-3

        # Dust phase-space rank-loss: γ_d much smaller than γ_g (the
        # cold-dust regime).
        @test maximum(ic.gamma_init_d) < maximum(ic.gamma_init_g)

        # Net momentum zero (sin on cell-centered grid).
        @test abs(sum(ic.u_d .* ic.rho_d)) < _eps(400)
        @test abs(sum(ic.u_g .* ic.rho_g)) < _eps(400)

        # save_times has n_snaps points.
        @test length(ic.params.save_times) == ic.params.n_snaps
    end

    # ----- A.6 eion ----------------------------------------------------
    @testset "A.6 setup_eion — uniform two-temperature plasma" begin
        ic = setup_eion(N=32, m_e=0.01, m_i=1.0, Te0=2.0, Ti0=0.5)
        @test ic.name == "eion"
        @test length(ic.rho_e) == 32

        # Spatially uniform.
        @test all(ic.rho_e .== 0.01)
        @test all(ic.rho_i .== 1.0)
        @test all(iszero, ic.u_e)
        @test all(iszero, ic.u_i)

        # Pressure such that T = P*m/ρ = T0 in energy units (py-1d
        # convention: rho_*= n*m, T = P*m/rho ⇒ P = rho*T/m).
        @test all(ic.P_e .≈ 0.01 * 2.0 / 0.01)   # = rho_e * Te0/m_e = 2.0
        @test all(ic.P_i .≈ 1.0  * 0.5 / 1.0)    # = rho_i * Ti0/m_i = 0.5

        # The thermal disequilibrium that drives equilibration:
        @test ic.Te0 != ic.Ti0
        @test ic.Te0 == 2.0
        @test ic.Ti0 == 0.5

        # save_times has n_snaps points.
        @test length(ic.params.save_times) == ic.params.n_snaps
    end

    # ----- Cross-cutting ------------------------------------------------
    @testset "common conservation invariants" begin
        # Every single-fluid factory returns Pxx == P, Pp == P, Q = 0
        # at t = 0 (no anisotropy, no heat flux at the IC).
        for ic in (setup_sod(), setup_cold_sinusoid(),
                   setup_steady_shock(M1=2.0),
                   setup_kmles_wavepool(N=64, n_snaps=4))
            @test all(ic.Pxx .== ic.P)
            @test all(ic.Pp  .== ic.P)
            @test all(iszero, ic.Q)
            @test all(iszero, ic.beta_init)
            @test all(>=(0), ic.gamma_init)
            @test eltype(ic.rho) == Float64
        end
    end

end  # @testset "setups"
