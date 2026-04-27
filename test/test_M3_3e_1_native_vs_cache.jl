# test_M3_3e_1_native_vs_cache.jl
#
# Phase M3-3e-1 defensive cross-check: the native `det_step_HG_native!`
# (M3-3e-1) must produce **byte-equal** state to running M1's
# `det_step!` on the wrapper's cached `Mesh1D` shim. This is the
# direct test that the cache_mesh-driven path and the native path
# remain bit-for-bit equivalent on the deterministic Newton sectors.
#
# When M3-3e-2/3/4 retire other paths and the cache_mesh field stays
# allocated only as transitional storage, this test gates against
# subtle algorithmic drift in the native path. After M3-3e-5 drops
# the cache_mesh field, the test will be retired (it relies on
# directly reading `mesh_HG.cache_mesh`).
#
# Coverage (per §A of `reference/notes_M3_3e_cache_mesh_retirement.md`):
#
#   Block 2 — Phase 2 Newton solve (periodic, no τ, no q):
#             N = 32 acoustic IC, 5 steps.
#   Block 3 — Phase 5b q-dissipation entropy update:
#             N = 64 Sod IC with q_kind=:vNR_linear_quadratic, 3 steps.
#   Block 4 — Phase 5 BGK joint relaxation (β, s, Pp, Q):
#             N = 32 isotropic IC with τ = 0.1, 5 steps.
#   Block 5 — Phase 7 inflow/outflow vertex chain pinning:
#             N = 40 steady-shock IC, 4 steps.

using Test
using dfmm

# Helper: build two parallel meshes (one for native path, one for the
# cache-mesh-driven baseline) from identical IC arrays.
function _build_pair(positions, velocities, αs, βs, ss; Δm, L_box,
                     Pps = nothing, Qs = nothing, bc = :periodic,
                     inflow_state = nothing, outflow_state = nothing,
                     n_pin = 2)
    mesh_native = dfmm.DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                              Δm = Δm, L_box = L_box,
                                              Pps = Pps, Qs = Qs, bc = bc,
                                              inflow_state = inflow_state,
                                              outflow_state = outflow_state,
                                              n_pin = n_pin)
    # The baseline mesh is the cache_mesh built by Mesh1D directly,
    # bypassing the native HG-native path entirely. We then run M1's
    # `det_step!` on it, with kwargs translated from the wrapper.
    periodic = bc == :periodic
    mesh_baseline = dfmm.Mesh1D(positions, velocities, αs, βs, ss;
                                 Δm = Δm, Pps = Pps, Qs = Qs, L_box = L_box,
                                 periodic = periodic, bc = bc)
    return mesh_native, mesh_baseline
end

# Helper: assert byte-equal `(x, u, α, β, s, Pp, Q)` between an
# HG `DetMeshHG` and an M1 `Mesh1D`.
function _assert_state_equal(mesh_HG, mesh_M1)
    N = dfmm.n_cells(mesh_HG)
    fs = mesh_HG.fields
    for j in 1:N
        seg = mesh_M1.segments[j].state
        @test fs.x[j][1]     === seg.x
        @test fs.u[j][1]     === seg.u
        @test fs.alpha[j][1] === seg.α
        @test fs.beta[j][1]  === seg.β
        @test fs.s[j][1]     === seg.s
        @test fs.Pp[j][1]    === seg.Pp
        @test fs.Q[j][1]     === seg.Q
    end
    for j in 1:N
        @test mesh_HG.p_half[j] === mesh_M1.p_half[j]
    end
end

@testset "M3-3e-1 native vs cache_mesh: Phase 2 acoustic" begin
    # Block 2 (Newton solve) coverage. Periodic, no BGK, no q.
    N = 32
    L_box = 1.0
    Δm = fill(L_box / N, N)
    positions = [Δm[1] * (j - 1) for j in 1:N]
    velocities = [0.05 * sinpi(2 * positions[j] / L_box) for j in 1:N]
    αs = fill(1.0, N)
    βs = fill(0.0, N)
    ss = fill(0.0, N)
    mesh_native, mesh_baseline = _build_pair(positions, velocities, αs, βs, ss;
                                              Δm = Δm, L_box = L_box)
    dt = 1e-3
    for _ in 1:5
        dfmm.det_step_HG!(mesh_native, dt)
        dfmm.det_step!(mesh_baseline, dt)
    end
    _assert_state_equal(mesh_native, mesh_baseline)
end

@testset "M3-3e-1 native vs cache_mesh: Phase 5b q-dissipation" begin
    # Block 3 (q_kind = :vNR entropy update). Sod-style IC.
    N = 64
    L_box = 1.0
    Δm = fill(L_box / N, N)
    positions = [Δm[1] * (j - 1) for j in 1:N]
    velocities = fill(0.0, N)
    αs = fill(1.0, N)
    βs = fill(0.0, N)
    # Sod-like entropy jump.
    ss = [j <= N ÷ 2 ? 0.0 : -0.5 * log(8.0) for j in 1:N]
    mesh_native, mesh_baseline = _build_pair(positions, velocities, αs, βs, ss;
                                              Δm = Δm, L_box = L_box)
    dt = 5e-4
    for _ in 1:3
        dfmm.det_step_HG!(mesh_native, dt;
                           q_kind = :vNR_linear_quadratic,
                           c_q_quad = 1.0, c_q_lin = 0.5)
        dfmm.det_step!(mesh_baseline, dt;
                        q_kind = :vNR_linear_quadratic,
                        c_q_quad = 1.0, c_q_lin = 0.5)
    end
    _assert_state_equal(mesh_native, mesh_baseline)
end

@testset "M3-3e-1 native vs cache_mesh: Phase 5 BGK relaxation" begin
    # Block 4 (BGK joint relaxation of (P_xx, P_⊥) + β decay + Q decay).
    N = 32
    L_box = 1.0
    Δm = fill(L_box / N, N)
    positions = [Δm[1] * (j - 1) for j in 1:N]
    # Mild perturbation to drive non-trivial transport.
    velocities = [0.03 * sinpi(2 * positions[j] / L_box) for j in 1:N]
    αs = fill(1.0, N)
    βs = fill(0.05, N)
    ss = fill(0.0, N)
    Pps = [1.0 for _ in 1:N]
    Qs  = [0.01 * cospi(2 * (j - 0.5) / N) for j in 1:N]
    mesh_native, mesh_baseline = _build_pair(positions, velocities, αs, βs, ss;
                                              Δm = Δm, L_box = L_box,
                                              Pps = Pps, Qs = Qs)
    dt = 1e-3
    for _ in 1:5
        dfmm.det_step_HG!(mesh_native, dt; tau = 0.1)
        dfmm.det_step!(mesh_baseline, dt; tau = 0.1)
    end
    _assert_state_equal(mesh_native, mesh_baseline)
end

@testset "M3-3e-1 native vs cache_mesh: Phase 7 inflow/outflow pinning" begin
    # Block 5 (Dirichlet pinning + cumulative-Δx vertex chain).
    N = 40
    L_box = 1.0
    Δm = fill(L_box / N, N)
    positions = [Δm[1] * (j - 1) for j in 1:N]
    velocities = fill(0.5, N)
    αs = fill(1.0, N)
    βs = fill(0.0, N)
    ss = fill(0.0, N)
    Pps = fill(1.0, N)
    inflow_state  = (rho = 1.0, u = 0.5, alpha = 1.0, beta = 0.0,
                      s = 0.0, Pp = 1.0, Q = 0.0)
    outflow_state = (rho = 2.0, u = 0.25, alpha = 1.0, beta = 0.0,
                      s = -0.5 * log(2.0), Pp = 2.0, Q = 0.0)
    mesh_native, mesh_baseline = _build_pair(positions, velocities, αs, βs, ss;
                                              Δm = Δm, L_box = L_box,
                                              Pps = Pps,
                                              bc = :inflow_outflow,
                                              inflow_state = inflow_state,
                                              outflow_state = outflow_state,
                                              n_pin = 2)
    dt = 5e-4
    for _ in 1:4
        dfmm.det_step_HG!(mesh_native, dt;
                           tau = 0.1, q_kind = :vNR_linear_quadratic)
        dfmm.det_step!(mesh_baseline, dt;
                        tau = 0.1, q_kind = :vNR_linear_quadratic,
                        bc = :inflow_outflow,
                        inflow_state = inflow_state,
                        outflow_state = outflow_state, n_pin = 2)
    end
    _assert_state_equal(mesh_native, mesh_baseline)
end
