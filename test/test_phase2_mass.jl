# test_phase2_mass.jl
#
# Phase 2 — mass conservation on the multi-segment Lagrangian mesh.
#
# Per-segment Lagrangian mass `Δm_j` is a fixed label, so `total_mass`
# is invariant by construction. This test asserts the obvious — but
# importantly verifies that the integrator does not corrupt the
# segment array (e.g. by reordering, by losing entries on
# unpack_state!, or by reading/writing into the wrong vertex).

using Test
using dfmm

@testset "Phase 2: mass conservation" begin
    # Smooth periodic perturbation; non-trivial dynamics.
    N = 16
    L = 1.0
    ρ0 = 1.0
    s0 = 0.0
    α0 = 1.0
    β0 = 0.0
    Δx = L / N
    Δm = fill(ρ0 * Δx, N)
    positions = collect((0:N-1) .* Δx) .+ 0.001 .* sin.(2π .* (0:N-1) ./ N) .* Δx
    velocities = 0.01 .* cos.(2π .* (0:N-1) ./ N)
    αs = fill(α0, N)
    βs = fill(β0, N)
    ss = fill(s0, N)

    mesh = Mesh1D(positions, velocities, αs, βs, ss;
                  Δm = Δm, L_box = L, periodic = true)

    M0 = total_mass(mesh)
    @test M0 ≈ sum(Δm) atol=100*eps()

    dt = 1e-3
    N_steps = 50
    masses_seen = Float64[]
    for n in 1:N_steps
        det_step!(mesh, dt)
        push!(masses_seen, total_mass(mesh))
    end

    # Mass is a sum of fixed labels — should be bit-stable across steps.
    @test all(abs.(masses_seen .- M0) .<= 100 * eps(Float64))

    # Per-segment masses must be unchanged.
    @test all(seg.Δm == Δm[i] for (i, seg) in enumerate(mesh.segments))
end
