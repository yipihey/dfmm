# test_M3_3e_3_amr_tracer_native_vs_cache.jl
#
# Phase M3-3e-3 defensive cross-check: the native HG-side AMR primitives
# (`refine_segment_HG!`, `coarsen_segment_pair_HG!`, `amr_step_HG!`) and
# the native `TracerMeshHG` must produce **byte-equal** state to running
# M1's `refine_segment!` / `coarsen_segment_pair!` / `amr_step!` and
# `TracerMesh` on a parallel `Mesh1D` IC.
#
# After M3-3e-5 drops the `cache_mesh` field this test will be retired
# (it directly runs the M1 baseline alongside).
#
# Coverage (per §C and §D of `reference/notes_M3_3e_cache_mesh_retirement.md`):
#
#   Block 1 — refine_segment_HG! + advection roundtrip vs M1:
#             K = 10 refine events at varying j on N = 16 periodic mesh.
#   Block 2 — coarsen_segment_pair_HG! conservation + state parity vs M1:
#             K = 10 coarsen events at varying j on N = 32 periodic mesh.
#   Block 3 — amr_step_HG! interleaved with det_step_HG!:
#             N = 32 Sod IC, 5 (det_step + amr_step) cycles, parity vs M1.
#   Block 4 — Tracer matrix bit-exact preservation across refine + coarsen:
#             3 tracers (step, sin, gauss) on N = 16 mesh, 6 mixed events.

using Test
using dfmm

# Helper: build paired meshes (HG native + M1 baseline) from identical
# IC arrays.
function _build_amr_pair(positions, velocities, αs, βs, ss; Δm, L_box,
                          Pps = nothing, Qs = nothing, bc = :periodic)
    mesh_HG = dfmm.DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                          Δm = Δm, L_box = L_box,
                                          Pps = Pps, Qs = Qs, bc = bc)
    periodic = bc == :periodic
    mesh_M1 = dfmm.Mesh1D(positions, velocities, αs, βs, ss;
                           Δm = Δm, Pps = Pps, Qs = Qs, L_box = L_box,
                           periodic = periodic, bc = bc)
    return mesh_HG, mesh_M1
end

# Assert per-cell bit-equal state (x, u, α, β, s, Pp, Q, Δm).
function _assert_amr_state_equal(mesh_HG, mesh_M1)
    N_HG = length(mesh_HG.Δm)
    N_M1 = dfmm.n_segments(mesh_M1)
    @test N_HG == N_M1
    fs = mesh_HG.fields
    for j in 1:N_HG
        sM = mesh_M1.segments[j].state
        @test mesh_HG.Δm[j] === mesh_M1.segments[j].Δm
        @test fs.x[j][1]     === sM.x
        @test fs.u[j][1]     === sM.u
        @test fs.alpha[j][1] === sM.α
        @test fs.beta[j][1]  === sM.β
        @test fs.s[j][1]     === sM.s
        @test fs.Pp[j][1]    === sM.Pp
        @test fs.Q[j][1]     === sM.Q
    end
end

@testset "M3-3e-3 native vs M1: K = 10 refine events" begin
    # Block 1. Periodic Sod-style IC; refine at varying j to exercise
    # the full splice / vertex-redistribution arithmetic.
    N = 16
    L_box = 1.0
    positions = collect((0:N-1) .* (L_box / N))
    velocities = [0.05 * sinpi(2 * positions[j] / L_box) for j in 1:N]
    Γ = 5/3
    ρ = [x < 0.5 ? 1.0 : 0.125 for x in positions]
    P = [x < 0.5 ? 1.0 : 0.1 for x in positions]
    αs = fill(0.02, N)
    βs = [0.05 * cospi(2 * positions[j]) for j in 1:N]
    ss = log.(P ./ ρ.^Γ)
    Δm = ρ .* (L_box / N)
    Pps = copy(P)

    mesh_HG, mesh_M1 = _build_amr_pair(positions, velocities, αs, βs, ss;
                                        Δm = Δm, L_box = L_box, Pps = Pps)

    refine_indices = [3, 8, 12, 15, 5, 7, 9, 11, 4, 6]
    for j in refine_indices
        N_now = length(mesh_HG.Δm)
        j_clamped = min(j, N_now)  # keep within range as N grows
        dfmm.refine_segment_HG!(mesh_HG, j_clamped)
        dfmm.refine_segment!(mesh_M1, j_clamped)
        _assert_amr_state_equal(mesh_HG, mesh_M1)
    end
end

@testset "M3-3e-3 native vs M1: K = 10 coarsen events" begin
    # Block 2. N = 32 acoustic IC; coarsen at varying j (descending so
    # indices stay valid).
    N = 32
    L_box = 1.0
    positions = collect((0:N-1) .* (L_box / N))
    velocities = [0.03 * sinpi(2 * positions[j] / L_box) for j in 1:N]
    αs = fill(1.0, N)
    βs = [0.02 * sinpi(4 * positions[j]) for j in 1:N]
    ss = fill(0.0, N)
    Δm = fill(L_box / N, N)
    Pps = fill(1.0, N)

    mesh_HG, mesh_M1 = _build_amr_pair(positions, velocities, αs, βs, ss;
                                        Δm = Δm, L_box = L_box, Pps = Pps)

    # Descending coarsen indices (mirror M1's amr_step! ordering).
    coarsen_indices = [28, 24, 20, 16, 12, 8, 4, 25, 18, 10]
    for j in coarsen_indices
        N_now = length(mesh_HG.Δm)
        if N_now < 4
            break
        end
        j_clamped = min(j, N_now - 1)  # leave room for j+1
        if j_clamped < 1
            continue
        end
        dfmm.coarsen_segment_pair_HG!(mesh_HG, j_clamped)
        dfmm.coarsen_segment_pair!(mesh_M1, j_clamped)
        _assert_amr_state_equal(mesh_HG, mesh_M1)
    end
end

@testset "M3-3e-3 native vs M1: amr_step! interleaved with det_step!" begin
    # Block 3. Sod IC with periodic mirror doubling; 5 (det + amr) cycles.
    N = 32
    L_box = 1.0
    positions = collect((0:N-1) .* (L_box / N))
    velocities = zeros(N)
    Γ = 5/3
    ρ = [x < 0.5 ? 1.0 : 0.125 for x in positions]
    P = [x < 0.5 ? 1.0 : 0.1 for x in positions]
    αs = fill(0.02, N)
    βs = zeros(N)
    ss = log.(P ./ ρ.^Γ)
    Δm = ρ .* (L_box / N)
    Pps = copy(P)

    mesh_HG, mesh_M1 = _build_amr_pair(positions, velocities, αs, βs, ss;
                                        Δm = Δm, L_box = L_box, Pps = Pps)

    dt = 5e-4
    for cycle in 1:5
        dfmm.det_step_HG!(mesh_HG, dt)
        dfmm.det_step!(mesh_M1, dt)
        _assert_amr_state_equal(mesh_HG, mesh_M1)

        # Re-evaluate indicator on each path. Bit-equal indicator → same
        # refine/coarsen events.
        ai_HG = dfmm.action_error_indicator_HG(mesh_HG)
        ai_M1 = dfmm.action_error_indicator(mesh_M1)
        @test ai_HG == ai_M1
        τ_refine = max(0.5 * maximum(ai_M1), 1e-8)
        res_HG = dfmm.amr_step_HG!(mesh_HG, ai_HG, τ_refine)
        res_M1 = dfmm.amr_step!(mesh_M1, ai_M1, τ_refine)
        @test res_HG.n_refined  == res_M1.n_refined
        @test res_HG.n_coarsened == res_M1.n_coarsened
        _assert_amr_state_equal(mesh_HG, mesh_M1)
    end
end

@testset "M3-3e-3 native vs M1: TracerMeshHG bit-exact across refine + coarsen" begin
    # Block 4. 3 tracers initialised identically on M1 + HG paths.
    # Apply alternating refine/coarsen events and verify the tracer
    # matrices remain in lock-step (bit-exact on refine, conservative
    # on coarsen — matching M1's behaviour exactly).
    N = 16
    L_box = 1.0
    positions = collect((0:N-1) .* (L_box / N))
    velocities = zeros(N)
    αs = fill(1.0, N)
    βs = zeros(N)
    ss = zeros(N)
    Δm = fill(L_box / N, N)
    Pps = fill(1.0, N)

    mesh_HG, mesh_M1 = _build_amr_pair(positions, velocities, αs, βs, ss;
                                        Δm = Δm, L_box = L_box, Pps = Pps)

    # Use a single mass-weighted scale derived from M1 so the tracer
    # initialisations are bit-identical (avoid generator-order differences
    # between total_mass / total_mass_HG).
    M_total = dfmm.total_mass(mesh_M1)
    tm_HG = dfmm.TracerMeshHG(mesh_HG; n_tracers = 3,
                               names = [:step, :sin, :gauss])
    tm_M1 = dfmm.TracerMesh(mesh_M1; n_tracers = 3,
                             names = [:step, :sin, :gauss])
    set_tracer!(tm_HG, :step, m -> m < 0.5 * M_total ? 1.0 : 0.0)
    set_tracer!(tm_M1, :step, m -> m < 0.5 * M_total ? 1.0 : 0.0)
    set_tracer!(tm_HG, :sin,  m -> sinpi(2 * m / M_total))
    set_tracer!(tm_M1, :sin,  m -> sinpi(2 * m / M_total))
    set_tracer!(tm_HG, :gauss, m -> exp(-((m - 0.5 * M_total) / (0.1 * M_total))^2))
    set_tracer!(tm_M1, :gauss, m -> exp(-((m - 0.5 * M_total) / (0.1 * M_total))^2))

    # Initial parity.
    @test tm_HG.tm.tracers == tm_M1.tracers

    # Mixed events. Refine grows N; coarsen shrinks. Cap the indices so
    # they stay valid across both sides as N evolves identically.
    events = [(:refine, 5), (:refine, 11), (:coarsen, 4), (:refine, 7),
              (:coarsen, 6), (:refine, 9)]
    for (op, j) in events
        N_now = length(mesh_HG.Δm)
        if op === :refine
            j_clamped = min(j, N_now)
            dfmm.refine_segment_HG!(mesh_HG, j_clamped; tracers = tm_HG)
            dfmm.refine_segment!(mesh_M1, j_clamped; tracers = tm_M1)
        else  # :coarsen
            if N_now < 4
                continue
            end
            j_clamped = min(j, N_now - 1)
            j_clamped < 1 && continue
            dfmm.coarsen_segment_pair_HG!(mesh_HG, j_clamped; tracers = tm_HG)
            dfmm.coarsen_segment_pair!(mesh_M1, j_clamped; tracers = tm_M1)
        end
        _assert_amr_state_equal(mesh_HG, mesh_M1)
        @test tm_HG.tm.tracers == tm_M1.tracers
        @test maximum(abs.(tm_HG.tm.tracers .- tm_M1.tracers)) === 0.0
    end
end
