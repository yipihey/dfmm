# test_M3_2b_swap8_bckind_HG.jl
#
# M3-2b Swap 8 — BCKind / FrameBoundaries framework on `DetMeshHG`.
#
# Verifies the constructor populates `bc_spec::FrameBoundaries{1}` and
# the inflow/outflow/n_pin slots correctly, and that the
# `bc_symbol_from_spec` translation matches the legacy `bc::Symbol`.
# The Phase-7 steady-shock test gates the runtime byte-equality.

using Test
using dfmm
using HierarchicalGrids: FrameBoundaries, PERIODIC, INFLOW, OUTFLOW

@testset "M3-2b Swap 8: DetMeshHG carries FrameBoundaries{1} for :periodic" begin
    N = 8
    positions = collect((0:N-1) .* (1.0 / N))
    velocities = zeros(N)
    αs = fill(0.02, N)
    βs = zeros(N)
    ss = zeros(N)
    Δm = fill(1.0 / N, N)
    mesh = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                  Δm = Δm, L_box = 1.0, bc = :periodic)
    @test mesh.bc_spec isa FrameBoundaries{1}
    @test mesh.bc_spec.spec === ((PERIODIC, PERIODIC),)
    @test bc_symbol_from_spec(mesh.bc_spec) === :periodic
    @test mesh.inflow_state === nothing
    @test mesh.outflow_state === nothing
    @test mesh.n_pin == 2
end

@testset "M3-2b Swap 8: DetMeshHG carries FrameBoundaries{1} for :inflow_outflow" begin
    N = 8
    positions = collect((0:N-1) .* (1.0 / N))
    velocities = zeros(N)
    αs = fill(0.02, N)
    βs = zeros(N)
    ss = zeros(N)
    Δm = fill(1.0 / N, N)
    inflow = (rho = 1.0, u = 0.0, alpha = 0.02, beta = 0.0,
              s = 0.0, Pp = 1.0, Q = 0.0)
    outflow = (rho = 4.0, u = 1.0, alpha = 0.02, beta = 0.0,
               s = 0.0, Pp = 4.0, Q = 0.0)
    mesh = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                  Δm = Δm, L_box = 1.0,
                                  bc = :inflow_outflow,
                                  inflow_state = inflow,
                                  outflow_state = outflow,
                                  n_pin = 3)
    @test mesh.bc_spec.spec === ((INFLOW, OUTFLOW),)
    @test bc_symbol_from_spec(mesh.bc_spec) === :inflow_outflow
    @test mesh.inflow_state === inflow
    @test mesh.outflow_state === outflow
    @test mesh.n_pin == 3
end

@testset "M3-2b Swap 8: explicit bc_spec overrides bc symbol" begin
    N = 8
    positions = collect((0:N-1) .* (1.0 / N))
    velocities = zeros(N)
    αs = fill(0.02, N)
    βs = zeros(N)
    ss = zeros(N)
    Δm = fill(1.0 / N, N)
    explicit_spec = FrameBoundaries{1}(((INFLOW, OUTFLOW),))
    # User-supplied bc_spec wins, even when bc = :periodic.
    mesh = DetMeshHG_from_arrays(positions, velocities, αs, βs, ss;
                                  Δm = Δm, L_box = 1.0,
                                  bc = :periodic,
                                  bc_spec = explicit_spec)
    @test mesh.bc_spec === explicit_spec
end
