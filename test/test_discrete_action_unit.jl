# test_discrete_action_unit.jl
#
# Direct unit tests for the Phase-2 action discretisation primitives in
# `src/discrete_action.jl`:
#
#   * midpoint_strain        — cell-centered (∂_x u)|_{n+½}
#   * midpoint_J             — midpoint specific volume J|_{n+½}
#   * segment_cholesky_action — per-segment ΔS_Ch contribution
#   * discrete_action_sum    — full per-step action on a periodic mesh
#
# These were previously exercised only via the multi-segment Phase-2
# regression tests; this file lands closed-form unit pins so refactors
# of the underlying quadrature can be caught at the source.

using Test
using dfmm

# Pull the unexported helpers / types directly out of the module.
const _midpoint_strain = dfmm.midpoint_strain
const _midpoint_J      = dfmm.midpoint_J
const _seg_action      = dfmm.segment_cholesky_action
const _action_sum      = dfmm.discrete_action_sum
const _Mvv             = dfmm.Mvv
const _DetField        = dfmm.DetField

@testset "midpoint_strain: uniform translation gives zero strain" begin
    # Both vertices share the same velocity ⇒ ū_left = ū_right ⇒
    # numerator vanishes regardless of the spatial spacing.
    let u = 0.7, x_left_n = 0.0, x_right_n = 1.0,
        x_left_np1 = 0.5, x_right_np1 = 1.5
        @test _midpoint_strain(u, u, u, u,
                               x_left_n, x_right_n,
                               x_left_np1, x_right_np1) == 0.0
    end
end

@testset "midpoint_strain: pure-shear closed form" begin
    # u_right - u_left constant, x_right - x_left constant ⇒ strain =
    # Δu / Δx exactly. Easiest sanity check.
    let u_l = 0.0, u_r = 1.0, x_l = 0.0, x_r = 0.5
        # Stationary positions over the timestep (n == np1).
        @test _midpoint_strain(u_l, u_r, u_l, u_r, x_l, x_r, x_l, x_r) ≈
              (u_r - u_l) / (x_r - x_l)
    end

    # Mixed n / np1 values: the operator averages each endpoint over the
    # timestep and only then takes the spatial difference.
    let u_l_n = 0.0, u_r_n = 1.0, u_l_np1 = 0.2, u_r_np1 = 1.4,
        x_l = 0.0, x_r = 0.5
        ū_left  = (u_l_n + u_l_np1) / 2
        ū_right = (u_r_n + u_r_np1) / 2
        analytic = (ū_right - ū_left) / (x_r - x_l)
        @test isapprox(_midpoint_strain(u_l_n, u_r_n, u_l_np1, u_r_np1,
                                        x_l, x_r, x_l, x_r),
                       analytic; atol = 1e-14)
    end
end

@testset "midpoint_J: matches Δx̄/Δm exactly" begin
    let x_left_n = 0.0, x_right_n = 0.5, x_left_np1 = 0.1, x_right_np1 = 0.7,
        Δm = 0.25
        x̄_left  = (x_left_n + x_left_np1) / 2
        x̄_right = (x_right_n + x_right_np1) / 2
        analytic = (x̄_right - x̄_left) / Δm
        @test isapprox(
            _midpoint_J(x_left_n, x_right_n, x_left_np1, x_right_np1, Δm),
            analytic; atol = 1e-14)
    end
end

@testset "midpoint_J: stationary mesh ⇒ J = Δx/Δm with no time blending" begin
    let x_l = 0.2, x_r = 0.7, Δm = 0.5
        @test _midpoint_J(x_l, x_r, x_l, x_r, Δm) ≈ (x_r - x_l) / Δm
    end
end

@testset "segment_cholesky_action: zero-strain ⇒ Phase-1 closed form" begin
    # When the timestep has no spatial strain (divu_mid = 0) and α, β
    # are constant in time (α_n == α_np1, β_n == β_np1), every term
    # except (Δt/2)·ᾱ²·(M_vv - β̄²) vanishes. This is the Phase-1
    # zero-strain reduction of the action.
    let α = 0.5, β = 0.0, M_vv = 1.0, divu = 0.0, dt = 0.01
        analytic = (dt / 2) * α^2 * (M_vv - β^2)
        @test isapprox(_seg_action(α, β, α, β, M_vv, divu, dt),
                       analytic; atol = 1e-14)
    end
end

@testset "segment_cholesky_action: time-derivative term reproduces -ᾱ³ Δβ /3" begin
    # With divu=0 and zero potential (M_vv = β̄²), the Cholesky action
    # reduces to its Hamilton-Pontryagin time-derivative term.
    let α_n = 0.6, β_n = 0.1, α_np1 = 0.6, β_np1 = 0.4,
        divu = 0.0, dt = 0.05
        β̄ = (β_n + β_np1) / 2
        ᾱ = (α_n + α_np1) / 2
        M_vv = β̄^2  # makes the (M_vv - β̄²) term vanish
        analytic = -(ᾱ^3) / 3 * (β_np1 - β_n)
        @test isapprox(_seg_action(α_n, β_n, α_np1, β_np1, M_vv, divu, dt),
                       analytic; atol = 1e-14)
    end
end

@testset "segment_cholesky_action: matches cholesky_one_step_action" begin
    # Phase-2 segment_cholesky_action is the multi-segment lift of
    # Phase-1 cholesky_one_step_action; with identical inputs the two
    # must match bit-for-bit.
    let α_n = 0.5, β_n = 0.05, α_np1 = 0.51, β_np1 = 0.07,
        M_vv = 1.0, divu = 0.2, dt = 0.01
        a2 = _seg_action(α_n, β_n, α_np1, β_np1, M_vv, divu, dt)
        a1 = dfmm.cholesky_one_step_action(α_n, β_n, α_np1, β_np1,
                                           M_vv, divu, dt)
        @test a2 == a1
    end
end

@testset "discrete_action_sum: 2-segment trivial state has only kinetic+potential" begin
    # Stationary mesh, zero velocity, frozen state across the timestep.
    # Action sum should reduce to (Δt/2) · Σ Δm_j · α_j² · (M_vv_j − β_j²).
    N = 2
    Δm = [0.5, 0.5]
    L = 1.0
    dt = 0.01
    α = [0.5, 0.5]
    β = [0.0, 0.0]
    s = [0.0, 0.0]
    x = [0.0, 0.5]
    u = [0.0, 0.0]
    state_n   = [_DetField(x[j], u[j], α[j], β[j], s[j]) for j in 1:N]
    state_np1 = [_DetField(x[j], u[j], α[j], β[j], s[j]) for j in 1:N]

    # Zero velocity ⇒ midpoint strain is identically zero. With
    # frozen (α, β), each segment contributes (Δt/2)·ᾱ²·(M_vv − β̄²).
    expected = 0.0
    for j in 1:N
        J_j = (j == N ? (x[1] + L) - x[j] : x[j+1] - x[j]) / Δm[j]
        Mvv_j = _Mvv(J_j, s[j])
        expected += Δm[j] * (dt / 2) * α[j]^2 * (Mvv_j - β[j]^2)
    end
    @test isapprox(_action_sum(state_n, state_np1, Δm, L, dt),
                   expected; atol = 1e-13)
end

@testset "discrete_action_sum: kinetic contribution from uniform velocity" begin
    # With α=0 the Cholesky-sector contribution vanishes; only the
    # vertex-kinetic terms survive. For uniform u we expect
    # (Δt/2) · Σ_i m̄_i · u² = (Δt/2) · M · u² where M = Σ Δm.
    N = 4
    Δm = fill(0.25, N)
    L = 1.0
    dt = 0.02
    u_const = 0.3
    x = collect(0.0:0.25:0.75)
    state = [_DetField(x[j], u_const, 0.0, 0.0, 0.0) for j in 1:N]
    M_total = sum(Δm)
    expected = (dt / 2) * M_total * u_const^2
    @test isapprox(_action_sum(state, state, Δm, L, dt),
                   expected; atol = 1e-13)
end

@testset "discrete_action_sum: input-length assertions" begin
    # Defensive: pre-conditions are checked at runtime.
    N = 3
    Δm = fill(0.25, N)
    s_short = [_DetField(0.0, 0.0, 0.5, 0.0, 0.0) for _ in 1:(N - 1)]
    s_full  = [_DetField(0.0, 0.0, 0.5, 0.0, 0.0) for _ in 1:N]
    L, dt = 1.0, 0.01
    @test_throws AssertionError _action_sum(s_full, s_short, Δm, L, dt)
    @test_throws AssertionError _action_sum(s_short, s_full, Δm, L, dt)
    @test_throws AssertionError _action_sum(s_full, s_full, Δm[1:end-1], L, dt)
end

@testset "discrete_action_sum: wrap segment uses x[1] + L_box on the right" begin
    # Build a non-uniform mesh where the wrap-around segment N → 1 has a
    # known specific-volume; the action sum must use `x[1] + L` for the
    # right vertex of segment N. This pins the periodic wrap.
    N = 3
    Δm = [0.5, 0.5, 0.5]
    L = 1.5
    dt = 0.01
    α = fill(0.5, N)
    β = zeros(N)
    s = zeros(N)
    x = [0.0, 0.5, 1.0]            # left vertices; right of seg 3 is 0+L=1.5
    u = zeros(N)
    state = [_DetField(x[j], u[j], α[j], β[j], s[j]) for j in 1:N]

    # Per-segment J: each is exactly 0.5/0.5 = 1.0 ⇒ Mvv(1, 0) = 1.0.
    # Sum reduces to (Δt/2)·N·Δm·α²·(1 − 0) = 3 · 0.5 · 0.01/2 · 0.25 · 1.
    expected = N * Δm[1] * (dt / 2) * α[1]^2 * 1.0
    @test isapprox(_action_sum(state, state, Δm, L, dt),
                   expected; atol = 1e-13)
end
