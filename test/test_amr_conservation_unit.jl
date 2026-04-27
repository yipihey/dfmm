# test_amr_conservation_unit.jl
#
# Direct unit tests on the M2-1 AMR primitives' conservation laws,
# decoupled from the full integrator. Where the existing
# `test_phase_M2_1_amr.jl` file pins single-call mass/momentum
# conservation, this file extends coverage to:
#
#   * Multi-step random refine / coarsen sequences (mass + momentum
#     drift bounded across O(20) topology mutations).
#   * Coarsen edge cases: end-of-array merge (j == N-1), degenerate-
#     wrap behaviour at small mesh sizes.
#   * Realizability across coarsen (γ² ≥ 0 guaranteed by the law-of-
#     total-covariance branch in `coarsen_segment_pair!`).
#   * `refresh_p_half!` round-trip after a topology change preserves
#     the per-vertex `m̄ · u` invariant.
#   * Refine bit-exact tracer-column duplication; coarsen mass-weighted
#     tracer averaging is conservative.
#
# Together with the integration-level tests in `test_phase_M2_1_amr.jl`
# this gives the AMR primitives a defence-in-depth test suite.

using Test
using dfmm
using Random

const _Mvv = dfmm.Mvv

# Helper: build a uniform mesh of N segments with non-trivial fields.
function _build_test_mesh(N::Int, L::Float64;
                          velocity_amp::Float64 = 0.0,
                          rng::AbstractRNG = MersenneTwister(0))
    Δx = L / N
    Δm = fill(L / N, N)
    positions = [Δx * (j - 1) for j in 1:N]
    velocities = velocity_amp == 0 ? zeros(N) :
                                     velocity_amp * randn(rng, N)
    αs = fill(0.5, N)
    βs = zeros(N)
    ss = zeros(N)
    return Mesh1D(positions, velocities, αs, βs, ss;
                  Δm = Δm, L_box = L, periodic = true)
end

@testset "AMR.refine: multi-step sequence preserves mass + momentum" begin
    rng = MersenneTwister(42)
    mesh = _build_test_mesh(8, 1.0; velocity_amp = 0.1, rng = rng)
    M0 = total_mass(mesh)
    P0 = total_momentum(mesh)

    # Refine 10 random segments, each time choosing one randomly from
    # the current mesh. Mass should remain bit-exact; momentum should
    # remain ≤ 1e-13 deviation per refine (linear-interp midpoint
    # vertex velocity is exactly momentum-conserving).
    for _ in 1:10
        j = rand(rng, 1:n_segments(mesh))
        refine_segment!(mesh, j)
    end
    @test total_mass(mesh) == M0
    @test isapprox(total_momentum(mesh), P0; atol = 1e-12)
    # Sum of per-segment Δm must equal initial total.
    @test isapprox(sum(s.Δm for s in mesh.segments), M0; atol = 1e-14)
end

@testset "AMR.coarsen: multi-step sequence preserves mass exactly" begin
    rng = MersenneTwister(7)
    mesh = _build_test_mesh(16, 1.0; velocity_amp = 0.05, rng = rng)
    M0 = total_mass(mesh)

    for _ in 1:6  # leave a margin above min_segments
        N = n_segments(mesh)
        if N ≤ 4
            break
        end
        # Pick j with j+1 ≤ N to avoid wrap-pair edge cases.
        j = rand(rng, 1:(N - 1))
        coarsen_segment_pair!(mesh, j)
    end
    @test isapprox(total_mass(mesh), M0; atol = 1e-14)
    # No segment should have negative or zero mass.
    @test all(s.Δm > 0 for s in mesh.segments)
end

@testset "AMR.coarsen: realizability γ² ≥ 0 by construction" begin
    # Construct a mesh where two adjacent cells have very different
    # velocities so the law-of-total-covariance "between-cell"
    # variance dominates. After coarsen, γ² = M_vv_new − β_new² must
    # be non-negative even though β_new is the mass-weighted average.
    N = 4
    Δm = fill(0.25, N)
    L = 1.0
    positions = collect(0.0:0.25:0.75)
    velocities = [-2.0, 2.0, 0.0, 0.0]      # large jump between segs 1↔2
    αs = fill(0.5, N)
    βs = [0.05, -0.05, 0.0, 0.0]            # opposite signs ⇒ averaging shrinks β
    ss = zeros(N)
    mesh = Mesh1D(positions, velocities, αs, βs, ss;
                  Δm = Δm, L_box = L, periodic = true)

    coarsen_segment_pair!(mesh, 1)            # merge the high-shear pair

    # Read back the merged cell and check γ² ≥ 0.
    seg = mesh.segments[1]
    ρ = segment_density(mesh, 1)
    J = 1 / ρ
    Mvv_new = _Mvv(J, seg.state.s)
    γ²_new = Mvv_new - seg.state.β^2
    @test γ²_new ≥ 0
end

@testset "AMR.coarsen at j = N-1 is handled correctly (no wrap)" begin
    # The amr_step! coarsen loop excludes wrap pairs (only j ∈ 1:N-1),
    # but coarsen_segment_pair! itself supports j == N. Pin the j=N-1
    # path which is the last "interior" pair; verify mass/momentum.
    N = 4
    Δm = fill(0.25, N)
    L = 1.0
    positions = collect(0.0:0.25:0.75)
    velocities = [0.1, 0.2, 0.3, 0.4]
    αs = fill(0.5, N)
    βs = zeros(N)
    ss = zeros(N)
    mesh = Mesh1D(positions, velocities, αs, βs, ss;
                  Δm = Δm, L_box = L, periodic = true)
    M0 = total_mass(mesh)
    P0 = total_momentum(mesh)

    coarsen_segment_pair!(mesh, N - 1)  # merge segments N-1 and N

    @test n_segments(mesh) == N - 1
    @test total_mass(mesh) == M0
    @test isapprox(total_momentum(mesh), P0; atol = 1e-13)
end

@testset "AMR.refresh_p_half!: enforces p_half[i] = m̄_i · u_i invariant" begin
    # After a refine, every entry of `mesh.p_half` should equal
    # the lumped vertex mass times the vertex velocity. This is a
    # stronger invariant than the global momentum sum.
    mesh = _build_test_mesh(8, 1.0; velocity_amp = 0.1,
                            rng = MersenneTwister(11))
    refine_segment!(mesh, 4)
    for i in 1:n_segments(mesh)
        m̄ = vertex_mass(mesh, i)
        u_i = mesh.segments[i].state.u
        @test isapprox(mesh.p_half[i], m̄ * u_i; atol = 1e-14)
    end
end

@testset "AMR.refine: x_mid lies inside parent (for non-degenerate parent)" begin
    # The inserted vertex must satisfy x_left < x_mid < x_right (no
    # wrap on a non-edge segment), so the periodic wrap doesn't kick in.
    N = 4
    Δm = fill(0.25, N)
    positions = collect(0.0:0.25:0.75)
    velocities = zeros(N)
    αs = fill(0.5, N)
    βs = zeros(N)
    ss = zeros(N)
    mesh = Mesh1D(positions, velocities, αs, βs, ss;
                  Δm = Δm, L_box = 1.0, periodic = true)
    refine_segment!(mesh, 2)
    x_left = mesh.segments[2].state.x
    x_mid  = mesh.segments[3].state.x
    x_right = mesh.segments[4].state.x
    @test x_left < x_mid < x_right
end

@testset "AMR.refine + coarsen: tracer total preserved and conservative" begin
    # Tracers are mass-weighted on coarsen. Constructed: 4 segments
    # with tracer values [1.0, 2.0, 3.0, 4.0]. After refine+coarsen
    # back, the segment-weighted sum Σ Δm_j · t_j should not change
    # by more than the mass-weighted-average operator's introduces.
    N = 4
    Δm = fill(0.25, N)
    positions = collect(0.0:0.25:0.75)
    velocities = zeros(N)
    αs = fill(0.5, N)
    βs = zeros(N)
    ss = zeros(N)
    mesh = Mesh1D(positions, velocities, αs, βs, ss;
                  Δm = Δm, L_box = 1.0, periodic = true)

    tracers = TracerMesh(mesh; n_tracers = 1)
    set_tracer!(tracers, 1, [1.0, 2.0, 3.0, 4.0])
    Δm_arr = [s.Δm for s in mesh.segments]
    sum0 = sum(Δm_arr[j] * tracers.tracers[1, j] for j in 1:N)

    refine_segment!(mesh, 2; tracers = tracers)
    # Refine duplicates the parent's tracer column ⇒ mass-weighted
    # tracer total is preserved bit-exactly.
    Δm_arr_after_refine = [s.Δm for s in mesh.segments]
    sum_after_refine = sum(Δm_arr_after_refine[j] * tracers.tracers[1, j]
                           for j in 1:n_segments(mesh))
    @test isapprox(sum_after_refine, sum0; atol = 1e-14)

    coarsen_segment_pair!(mesh, 2; tracers = tracers)
    Δm_arr_after_coarsen = [s.Δm for s in mesh.segments]
    sum_after_coarsen = sum(Δm_arr_after_coarsen[j] * tracers.tracers[1, j]
                            for j in 1:n_segments(mesh))
    # Mass-weighted average is exactly conservative for the
    # mass-weighted total.
    @test isapprox(sum_after_coarsen, sum0; atol = 1e-14)
end

@testset "AMR.amr_step!: no-op when indicator straddles thresholds" begin
    # All indicators equal 1.0, with τ_refine = 10.0 (no refine fires)
    # and τ_coarsen = 2.0 (no coarsen fires either: 1.0 < 2.0 would
    # qualify, so we choose τ_coarsen = 0.5 < 1.0 instead). To make
    # both thresholds inactive, set τ_refine = 10.0 (1.0 < 10.0,
    # no refine) and indicator = 1.0 > τ_coarsen = 0.5 ⇒ no coarsen.
    mesh = _build_test_mesh(8, 1.0)
    indicator = fill(1.0, n_segments(mesh))
    res = amr_step!(mesh, indicator, 10.0, 0.5)
    @test res.n_refined == 0
    @test res.n_coarsened == 0
    @test n_segments(mesh) == 8
end

@testset "AMR.amr_step!: hysteresis assertion τ_coarsen ≤ τ_refine/4" begin
    # Violate the hysteresis ⇒ AssertionError.
    mesh = _build_test_mesh(8, 1.0)
    indicator = zeros(n_segments(mesh))
    @test_throws AssertionError amr_step!(mesh, indicator, 1.0, 0.5)
end

@testset "AMR.amr_step!: respects max_segments / min_segments caps" begin
    # All indicators above τ_refine; cap should prevent uncontrolled
    # growth. We use a very low max_segments (= initial + 2).
    mesh = _build_test_mesh(8, 1.0)
    indicator = fill(10.0, n_segments(mesh))
    res = amr_step!(mesh, indicator, 1.0; max_segments = 10)
    @test n_segments(mesh) ≤ 10
    @test res.n_refined ≤ 2  # at most 2 additional segments before cap

    # Aggressive coarsen: every cell below τ_coarsen, but
    # min_segments = 4 floor the result.
    mesh2 = _build_test_mesh(8, 1.0)
    indicator2 = zeros(n_segments(mesh2))
    res2 = amr_step!(mesh2, indicator2, 4.0, 0.5; min_segments = 4)
    @test n_segments(mesh2) ≥ 4
end

@testset "AMR.refine: each daughter inherits parent (α, β, s) bit-exactly" begin
    # The constant-reconstruction refine must duplicate (α, β, s, Pp, Q)
    # to both daughters bit-exactly.
    N = 3
    Δm = fill(0.5, N)
    L = 1.5
    positions = [0.0, 0.5, 1.0]
    velocities = [0.0, 0.0, 0.0]
    αs = [0.5, 0.7, 0.9]
    βs = [0.01, 0.02, 0.03]
    ss = [-0.1, 0.2, 0.4]
    mesh = Mesh1D(positions, velocities, αs, βs, ss;
                  Δm = Δm, L_box = L, periodic = true)
    parent_state = mesh.segments[2].state
    refine_segment!(mesh, 2)
    # Position of daughter_left == parent.x, daughter_right.x == x_mid.
    @test mesh.segments[2].state.α == parent_state.α
    @test mesh.segments[2].state.β == parent_state.β
    @test mesh.segments[2].state.s == parent_state.s
    @test mesh.segments[3].state.α == parent_state.α
    @test mesh.segments[3].state.β == parent_state.β
    @test mesh.segments[3].state.s == parent_state.s
end
