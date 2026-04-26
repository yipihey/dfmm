# test_M3_0_parity_1D.jl
#
# Phase M3-0 verification gate: bit-exact parity between M1's
# `Mesh1D`+`Segment`-based Cholesky-sector integrator (Phase 1) and the
# new HG-substrate-based path (`SimplicialMesh{1, T}` + `PolynomialFieldSet`).
#
# The HG-side driver `cholesky_run_HG!` calls into M1's `cholesky_step`
# byte-identically (per `src/eom.jl` and `src/newton_step_HG.jl`),
# so the parity tests below assert **exact zero** difference state-by-state
# in deterministic single-thread runs. The brief allows up to 5e-13 to
# absorb representation-induced ULP-level noise; we report both the
# tightest-feasible bound (0.0 == 0.0) and the looser brief target.
#
# Three tests, mirroring M1 Phase-1's three regression tests:
#
#   1. Zero-strain free evolution (mirror of `test_phase1_zero_strain.jl`).
#   2. Uniform-strain test (mirror of `test_phase1_uniform_strain.jl`).
#   3. Symplectic loop integral (mirror of `test_phase1_symplectic.jl`).
#
# Acceptance: each per-step `(α_M1, β_M1)` matches `(α_HG, β_HG)` to 5e-13.

using Test
using StaticArrays
using dfmm

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

"""
    run_parity_step(α0, β0, M_vv, divu_half, Δt, N)

Run an `N`-step Phase-1 trajectory under both M1 (`cholesky_run`) and
the HG-substrate driver (`cholesky_run_HG!`), recording the per-step
state of each. Returns
`(traj_M1::Vector{SVector{2}}, traj_HG::Vector{SVector{2}})` of
length `N + 1` so the parity assertion can be made step-by-step.
"""
function run_parity_step(α0::Float64, β0::Float64,
                          M_vv::Float64, divu_half::Float64,
                          Δt::Float64, N::Integer)
    # M1 trajectory.
    q0   = SVector{2,Float64}(α0, β0)
    M1   = cholesky_run(q0, M_vv, divu_half, Δt, N;
                        abstol = 1e-13, reltol = 1e-13)

    # HG trajectory: step-by-step so we can record intermediate states.
    mesh   = single_cell_simplicial_mesh_1D()
    fields = allocate_chfield_HG(1)
    fields.alpha[1] = (α0,)
    fields.beta[1]  = (β0,)
    HG = Vector{SVector{2,Float64}}(undef, N + 1)
    HG[1] = SVector{2,Float64}(α0, β0)
    for n in 1:N
        cholesky_step_HG!(mesh, fields, M_vv, divu_half, Δt;
                          abstol = 1e-13, reltol = 1e-13)
        HG[n + 1] = SVector{2,Float64}(fields.alpha[1][1],
                                        fields.beta[1][1])
    end
    return M1, HG
end

"""
    max_per_step_error(traj_M1, traj_HG)

Return the maximum absolute per-step error in (α, β) between two
trajectories. Used as the bit-equality metric.
"""
function max_per_step_error(traj_M1, traj_HG)
    max_err = 0.0
    for n in eachindex(traj_M1)
        max_err = max(max_err,
                      abs(traj_M1[n][1] - traj_HG[n][1]),
                      abs(traj_M1[n][2] - traj_HG[n][2]))
    end
    return max_err
end

# ─────────────────────────────────────────────────────────────────────
# Test 1 — zero-strain free evolution
# ─────────────────────────────────────────────────────────────────────

@testset "M3-0 parity (1D): zero-strain Cholesky-sector evolution" begin
    M_vv      = 1.0
    divu_half = 0.0
    Δt        = 1e-3
    N         = 100
    α0, β0    = 1.0, 0.0

    M1, HG = run_parity_step(α0, β0, M_vv, divu_half, Δt, N)

    # Bit-equality (per the brief: < 5e-13). With shared M1 kernel we
    # expect 0.0 exactly under single-thread evaluation; we assert the
    # brief target to leave room for future representation tweaks.
    err = max_per_step_error(M1, HG)
    @test err < 5e-13
    # Tighten the assertion further for the report.
    @test err == 0.0
end

# ─────────────────────────────────────────────────────────────────────
# Test 2 — uniform-strain Cholesky-sector evolution
# ─────────────────────────────────────────────────────────────────────

@testset "M3-0 parity (1D): uniform-strain Cholesky-sector evolution" begin
    M_vv      = 0.0
    κ         = 0.1
    Δt        = 1e-3
    N         = 100
    α0, β0    = 1.0, 0.5

    M1, HG = run_parity_step(α0, β0, M_vv, κ, Δt, N)

    err = max_per_step_error(M1, HG)
    @test err < 5e-13
    @test err == 0.0
end

# ─────────────────────────────────────────────────────────────────────
# Test 3 — symplectic loop integral preservation
# ─────────────────────────────────────────────────────────────────────

# Same loop_integral helper used by the M1 Phase-1 symplectic test:
# trapezoidal ∮ (α³/3) dβ over a discretely-sampled closed loop.
function _loop_integral_M3(loop)
    K = length(loop)
    A = 0.0
    for k in 1:K
        k_next = (k == K) ? 1 : k + 1
        α_a, β_a = loop[k][1], loop[k][2]
        α_b, β_b = loop[k_next][1], loop[k_next][2]
        A += ((α_a^3 + α_b^3) / 6) * (β_b - β_a)
    end
    return A
end

@testset "M3-0 parity (1D): symplectic-form preservation" begin
    M_vv      = 1.0
    divu_half = 0.0
    Δt        = 1e-3
    N         = 100
    K         = 64
    α_c, β_c  = 1.0, 0.0
    h_α, h_β  = 1e-4, 1e-4

    # Build a K-vertex elliptical loop in (α, β).
    loop_init = Vector{SVector{2,Float64}}(undef, K)
    for k in 1:K
        f = (k - 1) / K
        θ = 2π * f
        α_k = α_c + h_α * cos(θ)
        β_k = β_c + h_β * sin(θ)
        loop_init[k] = SVector{2,Float64}(α_k, β_k)
    end

    # Evolve every loop vertex via the HG path.
    mesh   = uniform_simplicial_mesh_1D(K)
    fields = allocate_chfield_HG(K)
    for k in 1:K
        fields.alpha[k] = (loop_init[k][1],)
        fields.beta[k]  = (loop_init[k][2],)
    end
    cholesky_run_HG!(mesh, fields, M_vv, divu_half, Δt, N;
                     abstol = 1e-13, reltol = 1e-13)
    loop_evol_HG = [SVector{2,Float64}(fields.alpha[k][1],
                                        fields.beta[k][1]) for k in 1:K]

    # Compare against M1's per-vertex evolution.
    loop_evol_M1 = Vector{SVector{2,Float64}}(undef, K)
    for k in 1:K
        traj_k = cholesky_run(loop_init[k], M_vv, divu_half, Δt, N;
                              abstol = 1e-13, reltol = 1e-13)
        loop_evol_M1[k] = traj_k[end]
    end

    # Per-vertex bit-equality.
    max_vertex_err = 0.0
    for k in 1:K
        max_vertex_err = max(max_vertex_err,
                             abs(loop_evol_HG[k][1] - loop_evol_M1[k][1]),
                             abs(loop_evol_HG[k][2] - loop_evol_M1[k][2]))
    end
    @test max_vertex_err < 5e-13
    @test max_vertex_err == 0.0

    # Symplectic invariant: the loop integral preserved under the flow.
    A_init     = _loop_integral_M3(loop_init)
    A_evol_HG  = _loop_integral_M3(loop_evol_HG)
    A_evol_M1  = _loop_integral_M3(loop_evol_M1)

    # M3-0 parity tolerance: bit-exact between HG and M1.
    @test A_evol_HG == A_evol_M1
    # M1 Phase-1 symplectic tolerance: 1e-10 absolute on the loop area.
    @test A_init ≈ A_evol_HG atol = 1e-10
end
