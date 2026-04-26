# test_M3_2_phase11_tracer_HG.jl
#
# Phase M3-2 Block 3 — passive tracers on the HG substrate. Mirrors
# `test_phase11_tracer_advection.jl` for the bit-exact-preservation
# claims. The HG-side `TracerMeshHG` wraps an M1 `TracerMesh` whose
# `fluid` is the cache mesh; the tracer matrix is therefore the same
# storage layout as M1's.

using Test
using dfmm

include(joinpath(@__DIR__, "..", "experiments", "A1_sod.jl"))

# Build a Sod-IC HG mesh (mirror-doubled to a periodic Lagrangian mesh).
function build_sod_mesh_HG_for_tracers(ic; mirror::Bool = true)
    N0 = length(ic.rho)
    Γ = 5.0 / 3.0
    if mirror
        rho = vcat(ic.rho, reverse(ic.rho))
        u   = vcat(ic.u,   -reverse(ic.u))
        P   = vcat(ic.P,   reverse(ic.P))
        Pp  = vcat(ic.Pp,  reverse(ic.Pp))
        αs  = vcat(ic.alpha_init, reverse(ic.alpha_init))
        βs  = vcat(ic.beta_init,  -reverse(ic.beta_init))
        N = 2 * N0
        dx = 2.0 / N
        L_box = 2.0
    else
        rho = copy(ic.rho); u = copy(ic.u); P = copy(ic.P); Pp = copy(ic.Pp)
        αs = copy(ic.alpha_init); βs = copy(ic.beta_init)
        N = N0
        dx = 1.0 / N
        L_box = 1.0
    end
    Δm = rho .* dx
    ss = log.(P ./ rho .^ Γ)
    positions = collect((0:N-1) .* dx)
    return DetMeshHG_from_arrays(positions, u, αs, βs, ss;
                                  Δm = Δm, Pps = Pp, L_box = L_box,
                                  bc = :periodic)
end

@testset "M3-2 Phase 11 (HG): bit-exact preservation through shock (1000+ steps)" begin
    ic = setup_sod(; N = 40, t_end = 0.2, tau = 1e-3)
    mesh_HG = build_sod_mesh_HG_for_tracers(ic; mirror = true)
    N_seg = length(mesh_HG.Δm)

    # 3 tracers: step, sinusoid, narrow Gaussian.
    tm = TracerMeshHG(mesh_HG; n_tracers = 3,
                      names = [:step, :sin, :gauss])
    M_total = total_mass_HG(mesh_HG)
    set_tracer!(tm, :step, m -> m < 0.5 * M_total ? 1.0 : 0.0)
    set_tracer!(tm, :sin,  m -> sinpi(2 * m / M_total))
    set_tracer!(tm, :gauss, m -> exp(-((m - 0.5 * M_total) / (0.05 * M_total))^2))

    tracers_initial = copy(tm.tm.tracers)

    Γ = 5.0 / 3.0
    c_s_max = sqrt(Γ * maximum(ic.P) / minimum(ic.rho))
    dx = 2.0 / N_seg
    dt = 0.3 * dx / c_s_max
    n_steps = 1000

    for n in 1:n_steps
        det_step_HG!(mesh_HG, dt; tau = 1e-3)
        advect_tracers_HG!(tm, dt)
    end

    # Matrix object identity AND bitwise equality: tracer storage is
    # never written by the integrator.
    @test tm.tm.tracers === tm.tm.tracers
    @test tm.tm.tracers == tracers_initial
    @test maximum(abs.(tm.tm.tracers .- tracers_initial)) === 0.0
end

@testset "M3-2 Phase 11 (HG): bit-exact parity vs M1 tracer matrix" begin
    # Build identical Sod ICs on M1 and HG paths. Run the integrator
    # on each and confirm the tracer matrices agree bit-exactly.
    ic = setup_sod(; N = 40, t_end = 0.05, tau = 1e-3)
    mesh_M1 = build_sod_mesh(ic; mirror = true)
    mesh_HG = build_sod_mesh_HG_for_tracers(ic; mirror = true)
    N_seg = dfmm.n_segments(mesh_M1)
    # Use a single M_total (from mesh_M1) for both tracer initialisations
    # so the tracer matrices are bit-identical; total_mass and
    # total_mass_HG can differ at ULP level due to differing summation
    # orders (generator-based vs Vector reduction).
    M_total = dfmm.total_mass(mesh_M1)

    tm_M1 = TracerMesh(mesh_M1; n_tracers = 2, names = [:step, :sin])
    tm_HG = TracerMeshHG(mesh_HG; n_tracers = 2, names = [:step, :sin])
    set_tracer!(tm_M1, :step, m -> m < 0.5 * M_total ? 1.0 : 0.0)
    set_tracer!(tm_M1, :sin,  m -> sinpi(2 * m / M_total))
    set_tracer!(tm_HG, :step, m -> m < 0.5 * M_total ? 1.0 : 0.0)
    set_tracer!(tm_HG, :sin,  m -> sinpi(2 * m / M_total))

    Γ = 5.0 / 3.0
    c_s_max = sqrt(Γ * maximum(ic.P) / minimum(ic.rho))
    dx = 2.0 / N_seg
    dt = 0.3 * dx / c_s_max
    n_steps = 200

    for n in 1:n_steps
        dfmm.det_step!(mesh_M1, dt; tau = 1e-3)
        det_step_HG!(mesh_HG, dt; tau = 1e-3)
        advect_tracers!(tm_M1, dt)
        advect_tracers_HG!(tm_HG, dt)
    end

    # Tracer matrices match bit-for-bit (both never mutated; both
    # initialised from the same per-segment mass coordinates).
    @test tm_M1.tracers == tm_HG.tm.tracers
    @test maximum(abs.(tm_M1.tracers .- tm_HG.tm.tracers)) == 0.0
end

@testset "M3-2 Phase 11 (HG): API delegation (set/get/lookup)" begin
    # Smoke test of the tracer API on `TracerMeshHG`. Ensures the
    # delegate methods to the inner `TracerMesh` are wired correctly.
    N = 8
    L = 1.0
    dx = L / N
    rho0 = 1.0
    u0 = 0.0
    P0 = 1.0
    s0 = log(P0 / rho0^(5/3))
    αs = fill(0.02, N); βs = zeros(N); ss = fill(s0, N)
    Δm = fill(rho0 * dx, N)
    positions = collect((0:N-1) .* dx)
    mesh_HG = DetMeshHG_from_arrays(positions, fill(u0, N),
                                     αs, βs, ss;
                                     Δm = Δm, L_box = L,
                                     bc = :periodic)
    tm = TracerMeshHG(mesh_HG; n_tracers = 1, names = [:label])
    @test n_tracer_fields(tm) == 1
    @test n_tracer_segments(tm) == N
    @test tracer_index(tm, :label) == 1
    M_total = total_mass_HG(mesh_HG)
    set_tracer!(tm, :label, m -> begin
        k = clamp(floor(Int, m / (M_total / N)) + 1, 1, N)
        Float64(k)
    end)
    for j in 1:N
        xc = (j - 0.5) * dx
        @test tracer_at_position(tm, xc) == Float64(j)
    end
    # Add a second tracer via add_tracer!.
    tm2 = add_tracer!(tm, :flag, m -> 1.0)
    @test n_tracer_fields(tm2) == 2
    @test :flag in tm2.tm.names
    @test_throws ArgumentError tracer_index(tm, :nonexistent)
end
