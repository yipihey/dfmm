# test_M3_2_M2_1_amr_HG.jl
#
# Phase M3-2 Block 4a — M2-1 action-AMR ports onto the HG substrate.
# Mirrors `test_phase_M2_1_amr.jl` for the conservation properties of
# `refine_segment_HG!`, `coarsen_segment_pair_HG!`, and `amr_step_HG!`.
# All operate on the cache mesh through the M1 primitives, so the
# conservation properties are inherited bit-exactly.
#
# Coverage:
#   1. refine_segment_HG! conservation: bit-exact mass; exact momentum.
#   2. coarsen_segment_pair_HG! conservation: bit-exact mass; exact
#      momentum.
#   3. refine→coarsen roundtrip restores the mesh state.
#   4. action_error_indicator_HG matches M1's indicator bit-for-bit.
#   5. gradient_indicator_HG matches M1's indicator bit-for-bit.
#   6. amr_step_HG! hysteresis enforced and respects min/max segment
#      caps.

using Test
using dfmm
using dfmm: total_mass_HG, total_momentum_HG, total_mass, total_momentum,
            DetMeshHG_from_arrays, n_segments,
            refine_segment_HG!, coarsen_segment_pair_HG!,
            action_error_indicator_HG, gradient_indicator_HG,
            amr_step_HG!,
            refine_segment!, coarsen_segment_pair!,
            action_error_indicator, gradient_indicator,
            amr_step!, Mesh1D

# ─────────────────────────────────────────────────────────────────────
# 1. refine_segment_HG! — mass + momentum exactness
# ─────────────────────────────────────────────────────────────────────

@testset "M3-2 M2-1 (HG): refine_segment_HG! conservation" begin
    N = 4
    positions = collect(0.0:0.25:0.75)
    velocities = [0.1, 0.3, -0.2, 0.5]
    αs = fill(0.02, N)
    βs = [0.0, 0.05, -0.03, 0.01]
    ss = [0.0, 0.1, -0.1, 0.0]
    Δm = fill(0.25, N)

    mesh_HG = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                     Δm = Δm, L_box = 1.0,
                                     bc = :periodic)
    mass0 = total_mass_HG(mesh_HG)
    mom0  = total_momentum_HG(mesh_HG)

    refine_segment_HG!(mesh_HG, 2)

    # n_simplices grew by 1.
    @test length(mesh_HG.Δm) == 5
    @test total_mass_HG(mesh_HG) == mass0
    Δm_after = mesh_HG.Δm
    @test Δm_after[2] == 0.125
    @test Δm_after[3] == 0.125
    @test isapprox(total_momentum_HG(mesh_HG), mom0; atol = 1e-14)
end

# ─────────────────────────────────────────────────────────────────────
# 2. coarsen_segment_pair_HG! — mass + momentum exactness
# ─────────────────────────────────────────────────────────────────────

@testset "M3-2 M2-1 (HG): coarsen_segment_pair_HG! conservation" begin
    N = 4
    positions = collect(0.0:0.25:0.75)
    velocities = [0.1, 0.3, -0.2, 0.5]
    αs = fill(0.02, N)
    βs = [0.0, 0.05, -0.03, 0.01]
    ss = [0.0, 0.1, -0.1, 0.0]
    Δm = fill(0.25, N)

    mesh_HG = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                     Δm = Δm, L_box = 1.0,
                                     bc = :periodic)
    mass0 = total_mass_HG(mesh_HG)
    mom0  = total_momentum_HG(mesh_HG)

    coarsen_segment_pair_HG!(mesh_HG, 2)

    @test length(mesh_HG.Δm) == 3
    @test total_mass_HG(mesh_HG) == mass0
    @test mesh_HG.Δm[2] == 0.5
    @test isapprox(total_momentum_HG(mesh_HG), mom0; atol = 1e-14)
end

# ─────────────────────────────────────────────────────────────────────
# 3. refine→coarsen roundtrip
# ─────────────────────────────────────────────────────────────────────

@testset "M3-2 M2-1 (HG): refine→coarsen roundtrip" begin
    N = 4
    positions = collect(0.0:0.25:0.75)
    velocities = zeros(N)
    αs = fill(0.02, N)
    βs = zeros(N)
    ss = zeros(N)
    Δm = fill(0.25, N)

    mesh_HG = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                     Δm = Δm, L_box = 1.0,
                                     bc = :periodic)
    refine_segment_HG!(mesh_HG, 2)
    @test length(mesh_HG.Δm) == 5
    coarsen_segment_pair_HG!(mesh_HG, 2)
    @test length(mesh_HG.Δm) == 4
    @test all(isapprox.(mesh_HG.Δm, Δm; atol = 1e-15))
end

# ─────────────────────────────────────────────────────────────────────
# 4. action_error_indicator_HG matches M1 bit-for-bit
# ─────────────────────────────────────────────────────────────────────

@testset "M3-2 M2-1 (HG): action_error_indicator_HG matches M1" begin
    N = 16
    positions = collect((0:N-1) * 1.0 / N)
    ρ = [x < 0.5 ? 1.0 : 0.125 for x in positions]
    P = [x < 0.5 ? 1.0 : 0.1 for x in positions]
    Γ = 5/3
    αs = fill(0.02, N)
    βs = zeros(N)
    ss = log.(P ./ ρ.^Γ)
    dx = 1.0 / N
    Δm = ρ .* dx
    velocities = zeros(N)

    mesh_M1 = Mesh1D(positions, velocities, αs, βs, ss;
                     Δm = Δm, L_box = 1.0, periodic = true)
    mesh_HG = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                     Δm = Δm, L_box = 1.0,
                                     bc = :periodic)
    ai_M1 = action_error_indicator(mesh_M1)
    ai_HG = action_error_indicator_HG(mesh_HG)
    @test ai_M1 == ai_HG
end

# ─────────────────────────────────────────────────────────────────────
# 5. gradient_indicator_HG matches M1 bit-for-bit
# ─────────────────────────────────────────────────────────────────────

@testset "M3-2 M2-1 (HG): gradient_indicator_HG matches M1" begin
    N = 16
    positions = collect((0:N-1) * 1.0 / N)
    ρ = [x < 0.5 ? 1.0 : 0.125 for x in positions]
    P = [x < 0.5 ? 1.0 : 0.1 for x in positions]
    Γ = 5/3
    αs = fill(0.02, N)
    βs = zeros(N)
    ss = log.(P ./ ρ.^Γ)
    dx = 1.0 / N
    Δm = ρ .* dx
    velocities = zeros(N)

    mesh_M1 = Mesh1D(positions, velocities, αs, βs, ss;
                     Δm = Δm, L_box = 1.0, periodic = true)
    mesh_HG = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                     Δm = Δm, L_box = 1.0,
                                     bc = :periodic)
    for field in (:rho, :u, :P)
        gi_M1 = gradient_indicator(mesh_M1; field = field)
        gi_HG = gradient_indicator_HG(mesh_HG; field = field)
        @test gi_M1 == gi_HG
    end
end

# ─────────────────────────────────────────────────────────────────────
# 6. amr_step_HG! hysteresis + caps enforced
# ─────────────────────────────────────────────────────────────────────

@testset "M3-2 M2-1 (HG): amr_step_HG! hysteresis enforced" begin
    N = 16
    positions = collect((0:N-1) * 1.0 / N)
    velocities = zeros(N)
    αs = fill(0.02, N)
    βs = zeros(N)
    ss = zeros(N)
    Δm = fill(1.0 / N, N)

    mesh_HG = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                     Δm = Δm, L_box = 1.0,
                                     bc = :periodic)
    indicator = zeros(N)
    @test_throws AssertionError amr_step_HG!(mesh_HG, indicator, 0.1, 0.1)
    # With τ_coarsen = 0.025 (default) and an all-zeros indicator, every
    # cell is below τ_coarsen, so the coarsen pass fires aggressively
    # until min_segments = 4 is reached. Mirrors M1's amr_step!.
    result = amr_step_HG!(mesh_HG, indicator, 0.1)
    @test result.n_refined == 0
    # Default min_segments = 4 prevents over-coarsening.
    @test length(mesh_HG.Δm) >= 4
end

# ─────────────────────────────────────────────────────────────────────
# 7. amr_step_HG! refines on a step-IC indicator (matches M1 outcome)
# ─────────────────────────────────────────────────────────────────────

@testset "M3-2 M2-1 (HG): amr_step_HG! refines on shock indicator (parity vs M1)" begin
    N = 16
    positions = collect((0:N-1) * 1.0 / N)
    ρ = [x < 0.5 ? 1.0 : 0.125 for x in positions]
    P = [x < 0.5 ? 1.0 : 0.1 for x in positions]
    Γ = 5/3
    αs = fill(0.02, N)
    βs = zeros(N)
    ss = log.(P ./ ρ.^Γ)
    dx = 1.0 / N
    Δm = ρ .* dx
    velocities = zeros(N)

    mesh_M1 = Mesh1D(positions, velocities, αs, βs, ss;
                     Δm = Δm, L_box = 1.0, periodic = true)
    mesh_HG = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                     Δm = Δm, L_box = 1.0,
                                     bc = :periodic)

    ai_M1 = action_error_indicator(mesh_M1)
    ai_HG = action_error_indicator_HG(mesh_HG)
    τ_refine = 0.5 * maximum(ai_M1)

    res_M1 = amr_step!(mesh_M1, ai_M1, τ_refine)
    res_HG = amr_step_HG!(mesh_HG, ai_HG, τ_refine)

    @test res_M1.n_refined  == res_HG.n_refined
    @test res_M1.n_coarsened == res_HG.n_coarsened
    @test n_segments(mesh_M1) == length(mesh_HG.Δm)

    # Per-segment Δm and state agree bit-for-bit.
    for j in 1:length(mesh_HG.Δm)
        @test mesh_M1.segments[j].Δm == mesh_HG.Δm[j]
        sM = mesh_M1.segments[j].state
        sH = read_detfield(mesh_HG.fields, j)
        @test sM.x == sH.x
        @test sM.u == sH.u
        @test sM.α == sH.α
        @test sM.β == sH.β
        @test sM.s == sH.s
    end
end
