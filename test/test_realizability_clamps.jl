# test_realizability_clamps.jl
#
# Targeted tests on the realizability clamps in the EOS layer and the
# stochastic projection. These properties were previously verified only
# implicitly via the M2-3 long-run stability gate; we now pin the
# point-wise contracts:
#
#   * `gamma_from_state` is total: clamps to 0 across the realizability
#     boundary and never returns NaN for finite real inputs.
#   * `gamma_from_Mvv` (the typed Phase-1 helper) likewise clamps.
#   * `Mvv` cold-limit underflow is exact zero, not NaN.
#   * `realizability_project!` post-state always satisfies
#     M_vv ≥ headroom · β² **and** M_vv ≥ Mvv_floor.
#   * `realizability_project!` is conservative on mass and momentum
#     (ρ and vertex u untouched).
#   * `realizability_project!(:none)` is a true no-op.

using Test
using dfmm

const _Mvv = dfmm.Mvv

@testset "gamma_from_state: realizability boundary clamps to 0 (no NaN)" begin
    # Inside the cone: γ = sqrt(M_vv − β²).
    @test dfmm.gamma_from_state(1.0, 0.5) ≈ sqrt(0.75)

    # On the cone (M_vv == β²): γ = 0.
    @test dfmm.gamma_from_state(0.25, 0.5) == 0.0

    # Outside the cone (M_vv < β²): γ = 0 (clamped, never NaN).
    @test dfmm.gamma_from_state(0.1, 0.5) == 0.0
    @test !isnan(dfmm.gamma_from_state(0.1, 0.5))

    # Cold limit (M_vv = 0): γ = 0.
    @test dfmm.gamma_from_state(0.0, 0.0) == 0.0
end

@testset "gamma_from_Mvv: type-stable clamp" begin
    # Phase-1's typed helper. Same clamp behaviour, but the result type
    # follows the input (used by ForwardDiff Dual paths).
    @test dfmm.gamma_from_Mvv(1.0, 0.5) ≈ sqrt(1.0 - 0.25)
    @test dfmm.gamma_from_Mvv(0.1, 0.5) == 0.0
    # Float32 round-trip preserves type.
    let g = dfmm.gamma_from_Mvv(1.0f0, 0.5f0)
        @test g isa Float32
        @test isapprox(g, sqrt(1.0f0 - 0.25f0); atol = 1e-7)
    end
end

@testset "Mvv: cold-limit returns exact zero, not NaN" begin
    # The closed form is exp(log_Mvv); for log_Mvv < -700 the result is
    # explicitly clamped to 0.0 to avoid a denormalised flush.
    @test _Mvv(1.0, -1.0e4) == 0.0
    @test !isnan(_Mvv(1.0, -1.0e4))
end

@testset "Mvv: invalid J ≤ 0 returns NaN, not error" begin
    @test isnan(_Mvv(0.0, 0.0))
    @test isnan(_Mvv(-0.5, 0.0))
end

# Helper: make a tiny mesh in a realizability-violating state so the
# projection must fire on every cell.
function _build_violating_mesh(N::Int)
    L = 1.0
    Δm = fill(L / N, N)
    positions = [(L / N) * (j - 1) for j in 1:N]
    velocities = zeros(N)
    αs = fill(0.5, N)
    # M_vv = J^(1-Γ) exp(s) with Γ=5/3, J=1, s=0 ⇒ M_vv = 1.0.
    # Set β = 1.5 so β² = 2.25 > 1.0 ⇒ projection fires on every cell.
    βs = fill(1.5, N)
    ss = zeros(N)
    return Mesh1D(positions, velocities, αs, βs, ss;
                  Δm = Δm, L_box = L, periodic = true)
end

@testset "realizability_project!: post-state satisfies M_vv ≥ headroom · β²" begin
    headroom = 1.05
    Mvv_floor = 1e-6
    mesh = _build_violating_mesh(8)
    dfmm.realizability_project!(mesh; kind = :reanchor,
                                headroom = headroom,
                                Mvv_floor = Mvv_floor)
    for j in 1:n_segments(mesh)
        ρ = segment_density(mesh, j)
        J = 1 / ρ
        Mvv_post = _Mvv(J, mesh.segments[j].state.s)
        β = mesh.segments[j].state.β
        @test Mvv_post + 1e-12 ≥ headroom * β^2
    end
end

@testset "realizability_project!: enforces absolute Mvv_floor" begin
    # All cells with β = 0 have M_vv = 1.0, which is below a floor of 5
    # ⇒ projection must raise every cell's M_vv to 5.
    N = 4
    L = 1.0
    Δm = fill(L / N, N)
    positions = [(L / N) * (j - 1) for j in 1:N]
    velocities = zeros(N)
    αs = fill(0.5, N)
    βs = fill(0.0, N)
    ss = zeros(N)
    mesh = Mesh1D(positions, velocities, αs, βs, ss;
                  Δm = Δm, L_box = L, periodic = true)
    Mvv_floor = 5.0
    dfmm.realizability_project!(mesh; kind = :reanchor,
                                headroom = 1.05,
                                Mvv_floor = Mvv_floor)
    for j in 1:n_segments(mesh)
        ρ = segment_density(mesh, j)
        J = 1 / ρ
        Mvv_post = _Mvv(J, mesh.segments[j].state.s)
        @test Mvv_post + 1e-12 ≥ Mvv_floor
    end
end

@testset "realizability_project!: mass and momentum exactly preserved" begin
    mesh = _build_violating_mesh(8)
    M0 = total_mass(mesh)
    P0 = total_momentum(mesh)
    dfmm.realizability_project!(mesh; kind = :reanchor,
                                headroom = 1.05, Mvv_floor = 1e-6)
    @test total_mass(mesh) == M0
    @test total_momentum(mesh) == P0
end

@testset "realizability_project!: kind=:none is a no-op" begin
    mesh = _build_violating_mesh(8)
    snap_s = [mesh.segments[j].state.s for j in 1:n_segments(mesh)]
    snap_Pp = [mesh.segments[j].state.Pp for j in 1:n_segments(mesh)]
    dfmm.realizability_project!(mesh; kind = :none)
    for j in 1:n_segments(mesh)
        @test mesh.segments[j].state.s == snap_s[j]
        @test mesh.segments[j].state.Pp == snap_Pp[j]
    end
end

@testset "realizability_project!: stats accumulator counts firing events" begin
    mesh = _build_violating_mesh(8)
    stats = dfmm.ProjectionStats()
    # Project once. Every cell crosses the boundary ⇒ n_events == N.
    dfmm.realizability_project!(mesh; kind = :reanchor,
                                headroom = 1.05, Mvv_floor = 1e-6,
                                stats = stats)
    @test stats.n_steps == 1
    @test stats.n_events == 8
end

@testset "realizability_project!: idempotent (project twice ≈ project once)" begin
    # Once γ ≥ headroom · β, a second projection should leave the state
    # essentially unchanged (no further entropy increase).
    mesh = _build_violating_mesh(4)
    dfmm.realizability_project!(mesh; kind = :reanchor,
                                headroom = 1.05, Mvv_floor = 1e-6)
    s_after_one = [mesh.segments[j].state.s for j in 1:n_segments(mesh)]
    dfmm.realizability_project!(mesh; kind = :reanchor,
                                headroom = 1.05, Mvv_floor = 1e-6)
    s_after_two = [mesh.segments[j].state.s for j in 1:n_segments(mesh)]
    for (a, b) in zip(s_after_one, s_after_two)
        @test isapprox(a, b; atol = 1e-13)
    end
end

@testset "realizability_project!: unsupported kind raises" begin
    mesh = _build_violating_mesh(2)
    @test_throws AssertionError dfmm.realizability_project!(mesh;
        kind = :unsupported_variant)
end
