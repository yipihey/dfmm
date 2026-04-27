# test_segment_unit.jl
#
# Direct unit tests on the helpers in `src/segment.jl`. These are
# typically exercised only indirectly via Phase-2/5 regression paths,
# so they need explicit unit pins on the periodic-wrap accessor logic
# and the conservation invariants of the diagnostic helpers
# (`total_mass`, `total_momentum`, `total_energy`,
# `total_internal_energy`, `total_kinetic_energy`).

using Test
using dfmm

const _Mvv = dfmm.Mvv

# Build a uniform mesh with controlled state for closed-form checks.
function _uniform_mesh(N::Int, L::Float64;
                      u_const::Float64 = 0.0,
                      α::Float64 = 0.5,
                      β::Float64 = 0.0,
                      s::Float64 = 0.0)
    Δm = fill(L / N, N)
    positions = [(L / N) * (j - 1) for j in 1:N]
    velocities = fill(u_const, N)
    αs = fill(α, N)
    βs = fill(β, N)
    ss = fill(s, N)
    return Mesh1D(positions, velocities, αs, βs, ss;
                  Δm = Δm, L_box = L, periodic = true)
end

@testset "n_segments: matches segments vector length" begin
    mesh = _uniform_mesh(8, 1.0)
    @test n_segments(mesh) == 8
    @test n_segments(mesh) == length(mesh.segments)
end

@testset "vertex_mass: cyclic average of adjacent Δm" begin
    # Non-uniform Δm so the cyclic average is non-trivial.
    Δm = [0.1, 0.2, 0.3, 0.4]
    L = sum(Δm)
    positions = cumsum([0.0; Δm[1:end-1]])
    mesh = Mesh1D(positions, zeros(4), fill(0.5, 4), zeros(4), zeros(4);
                  Δm = Δm, L_box = L, periodic = true)
    # m̄_i = (Δm_{i-1} + Δm_i)/2 cyclically.
    @test vertex_mass(mesh, 1) ≈ (Δm[4] + Δm[1]) / 2
    @test vertex_mass(mesh, 2) ≈ (Δm[1] + Δm[2]) / 2
    @test vertex_mass(mesh, 3) ≈ (Δm[2] + Δm[3]) / 2
    @test vertex_mass(mesh, 4) ≈ (Δm[3] + Δm[4]) / 2
    # Sum of vertex masses equals total mass.
    M = sum(vertex_mass(mesh, i) for i in 1:n_segments(mesh))
    @test isapprox(M, sum(Δm); atol = 1e-14)
end

@testset "segment_length: periodic wrap on the last segment" begin
    # Uniform mesh: every segment has length L/N including the wrap
    # segment N → 1, which uses positions[1] + L_box for the right vertex.
    L, N = 1.0, 4
    mesh = _uniform_mesh(N, L)
    for j in 1:N
        @test isapprox(segment_length(mesh, j), L / N; atol = 1e-14)
    end

    # Non-uniform: explicit positions, periodic wrap on segment 3.
    positions = [0.0, 0.3, 0.7]
    Δm = [0.3, 0.4, 0.3]
    mesh2 = Mesh1D(positions, zeros(3), fill(0.5, 3), zeros(3), zeros(3);
                   Δm = Δm, L_box = 1.0, periodic = true)
    @test isapprox(segment_length(mesh2, 1), 0.3; atol = 1e-14)
    @test isapprox(segment_length(mesh2, 2), 0.4; atol = 1e-14)
    # Wrap segment: x_right = positions[1] + L_box = 0.0 + 1.0 = 1.0;
    # length = 1.0 − 0.7 = 0.3.
    @test isapprox(segment_length(mesh2, 3), 0.3; atol = 1e-14)
end

@testset "segment_density: ρ_j = Δm_j / Δx_j" begin
    L, N = 1.0, 4
    mesh = _uniform_mesh(N, L)
    for j in 1:N
        ρ = segment_density(mesh, j)
        Δm = mesh.segments[j].Δm
        Δx = segment_length(mesh, j)
        @test ρ ≈ Δm / Δx
        @test ρ > 0
    end
end

@testset "total_mass: invariant equals sum(Δm)" begin
    L, N = 1.5, 5
    mesh = _uniform_mesh(N, L)
    @test total_mass(mesh) == sum(s.Δm for s in mesh.segments)
end

@testset "total_momentum: zero for stationary mesh, m̄·u·N for uniform u" begin
    L, N = 1.0, 6
    mesh_zero = _uniform_mesh(N, L; u_const = 0.0)
    @test total_momentum(mesh_zero) == 0.0

    u_c = 0.4
    mesh_u = _uniform_mesh(N, L; u_const = u_c)
    # Total momentum = Σ m̄_i u_i = u_c · Σ m̄_i = u_c · L (uniform Δm).
    @test isapprox(total_momentum(mesh_u), u_c * L; atol = 1e-14)
end

@testset "total_kinetic_energy: zero for stationary, (1/2) M u² for uniform" begin
    L, N = 1.0, 6
    mesh_zero = _uniform_mesh(N, L)
    @test total_kinetic_energy(mesh_zero) == 0.0

    u_c = 0.3
    mesh_u = _uniform_mesh(N, L; u_const = u_c)
    M = total_mass(mesh_u)
    @test isapprox(total_kinetic_energy(mesh_u), 0.5 * M * u_c^2; atol = 1e-13)
end

@testset "total_energy: kinetic + Cholesky-Hamiltonian split" begin
    # With β = 0 and α small, H_Ch = -½ α² M_vv. For a stationary
    # uniform mesh, total energy reduces to that closed form.
    L, N = 1.0, 4
    α = 0.5
    β = 0.0
    s = 0.0
    mesh = _uniform_mesh(N, L; α = α, β = β, s = s)
    # Each segment has J = (L/N)/(L/N) = 1 ⇒ M_vv(1, 0) = 1.
    M_total = total_mass(mesh)
    expected_H = -0.5 * α^2 * 1.0  # per unit Δm
    expected = expected_H * M_total  # KE = 0
    @test isapprox(total_energy(mesh), expected; atol = 1e-13)
end

@testset "total_internal_energy: (Pxx + 2 Pp)/(2 ρ) per cell, weighted by Δm" begin
    # With β = 0, M_vv(1, 0) = 1. Default Pp from the IC is ρ·M_vv = 1.0
    # so Pxx = Pp = 1.0; trace = 3.0, and IE = Σ Δm · 3 / (2·ρ).
    L, N = 1.0, 4
    mesh = _uniform_mesh(N, L)
    ρ_uniform = 1.0  # (L/N)/(L/N) = 1
    expected = sum(s.Δm * 3.0 / (2 * ρ_uniform) for s in mesh.segments)
    @test isapprox(total_internal_energy(mesh), expected; atol = 1e-13)
end

@testset "Mesh1D constructor: assertion on length mismatches" begin
    # All input vectors must have the same length and length(Δm) must match.
    @test_throws AssertionError Mesh1D([0.0], [0.0], [0.5], [0.0], [0.0];
                                       Δm = [1.0], L_box = 1.0, periodic = true)
    # Mismatched velocity vector length:
    @test_throws AssertionError Mesh1D([0.0, 0.5], [0.0], [0.5, 0.5],
                                       [0.0, 0.0], [0.0, 0.0];
                                       Δm = [0.5, 0.5], L_box = 1.0,
                                       periodic = true)
end

@testset "Mesh1D constructor: bc must be supported symbol" begin
    @test_throws AssertionError Mesh1D([0.0, 0.5], [0.0, 0.0], [0.5, 0.5],
                                       [0.0, 0.0], [0.0, 0.0];
                                       Δm = [0.5, 0.5], L_box = 1.0,
                                       periodic = true, bc = :nonsense)
end

@testset "Mesh1D constructor: non-periodic requires :inflow_outflow bc" begin
    @test_throws AssertionError Mesh1D([0.0, 0.5], [0.0, 0.0], [0.5, 0.5],
                                       [0.0, 0.0], [0.0, 0.0];
                                       Δm = [0.5, 0.5], L_box = 1.0,
                                       periodic = false, bc = :periodic)
end

@testset "single_segment_mesh: Phase-1 ChField path" begin
    α, β, γ = 0.5, 0.1, 0.4
    mesh = single_segment_mesh(α, β, γ; Δm = 1.0, m = 0.0)
    @test n_segments(mesh) == 1
    seg = mesh.segments[1]
    @test seg.state.α == α
    @test seg.state.β == β
    @test seg.state.γ == γ
    @test seg.Δm == 1.0
end

@testset "Mesh1D Pp default initialization: Pp = ρ · M_vv(J, s)" begin
    # When `Pps` is not supplied, the constructor should default Pp on
    # each segment to the isotropic-Maxwellian value ρ · M_vv(J, s).
    L, N = 1.0, 4
    mesh = _uniform_mesh(N, L; α = 0.5, β = 0.0, s = 0.3)
    for j in 1:N
        ρ = segment_density(mesh, j)
        J = 1 / ρ
        Mvv_j = _Mvv(J, mesh.segments[j].state.s)
        @test isapprox(mesh.segments[j].state.Pp, ρ * Mvv_j; atol = 1e-13)
    end
end
