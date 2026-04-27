# experiments/M3_5_remap_round_trip_demo.jl
#
# Milestone M3-5 smoke driver — demonstrates the Bayesian L→E→L remap
# round-trip on a 16×16 Eulerian background with a sinusoidally-deformed
# Lagrangian triangulation, plotting per-cycle conservation (mass /
# momentum / energy) and Liouville-extrema histories.
#
# Outputs:
#   reference/figs/M3_5_remap_conservation.png — 4-panel summary plot.

using HierarchicalGrids
using HierarchicalGrids: SimplicialMesh, EulerianFrame, MonomialBasis,
    SoA, allocate_polynomial_fields,
    enumerate_leaves, cell_physical_box,
    n_simplices, n_vertices, set_vertex_position!, vertex_position,
    simplex_vertex_positions
using CairoMakie
const HG = HierarchicalGrids

using dfmm

# ---------------------------------------------------------------------------
# Build the 16×16 Eulerian + matched Lagrangian-triangulation setup.
# ---------------------------------------------------------------------------

eul_mesh = dfmm.uniform_eulerian_mesh(2, 4)   # 16×16 = 256 leaves
frame = EulerianFrame(eul_mesh, (0.0, 0.0), (1.0, 1.0))

# Build a triangulated Lagrangian mesh (2 triangles per leaf, vertices
# de-duped across leaf boundaries → conforming).
function triangulate_eulerian(eul_mesh, frame)
    leaves = enumerate_leaves(eul_mesh)
    vertices = NTuple{2, Float64}[]
    vertex_idx = Dict{NTuple{2, Float64}, Int32}()
    function vidx!(v)
        if haskey(vertex_idx, v)
            return vertex_idx[v]
        else
            push!(vertices, v)
            idx = Int32(length(vertices))
            vertex_idx[v] = idx
            return idx
        end
    end
    sv = Int32[]
    for ci in leaves
        lo, hi = cell_physical_box(frame, Int(ci))
        v00 = vidx!((lo[1], lo[2]))
        v10 = vidx!((hi[1], lo[2]))
        v01 = vidx!((lo[1], hi[2]))
        v11 = vidx!((hi[1], hi[2]))
        push!(sv, v00, v10, v11)
        push!(sv, v00, v11, v01)
    end
    sv_mat = reshape(sv, 3, :)
    sn_mat = zeros(Int32, 3, size(sv_mat, 2))
    return SimplicialMesh{2, Float64}(vertices, sv_mat, sn_mat)
end

lag_mesh = triangulate_eulerian(eul_mesh, frame)
n_simp = n_simplices(lag_mesh)
@info "M3-5 demo: built Lagrangian + Eulerian meshes" n_simp n_eul_leaves = length(enumerate_leaves(eul_mesh))

# ---------------------------------------------------------------------------
# Initialize fields (rho, ux, uy, energy proxy) on the Lagrangian side.
# ---------------------------------------------------------------------------
basis = MonomialBasis{2, 0}()
lag_fields = allocate_polynomial_fields(SoA(), basis, n_simp;
                                          rho = Float64,
                                          ux = Float64,
                                          uy = Float64,
                                          E = Float64)
@inbounds for s in 1:n_simp
    verts = simplex_vertex_positions(lag_mesh, s)
    cx = (verts[1][1] + verts[2][1] + verts[3][1]) / 3
    cy = (verts[1][2] + verts[2][2] + verts[3][2]) / 3
    rho = 1.0 + 0.3 * sin(2π * cx)
    ux = 0.2 * cos(2π * cy)
    uy = -0.2 * sin(2π * cx)
    P = 1.0 + 0.1 * cos(2π * (cx + cy))
    E = 0.5 * rho * (ux^2 + uy^2) + P / 0.4
    lag_fields.rho[s] = (rho,)
    lag_fields.ux[s]  = (ux,)
    lag_fields.uy[s]  = (uy,)
    lag_fields.E[s]   = (E,)
end

state = dfmm.BayesianRemapState(eul_mesh, frame, lag_fields)

# ---------------------------------------------------------------------------
# Apply a one-time small Lagrangian deformation, then run 10 round trips
# WITHOUT further mesh motion. With a fixed mesh, the L→E→L round trip
# should preserve mass/momentum/energy to ~1e-12 every cycle (the only
# error source is the L²-projection floating-point round-off).
# ---------------------------------------------------------------------------

# One-time deformation (bump shape; vanishes at boundary).
A_def = 0.005
for i in 1:n_vertices(lag_mesh)
    p = vertex_position(lag_mesh, i)
    bump = p[1] * (1 - p[1]) * p[2] * (1 - p[2])
    δx = A_def * bump * cos(2π * 2 * p[1])
    δy = A_def * bump * sin(2π * 2 * p[2])
    set_vertex_position!(lag_mesh, i, (p[1] + δx, p[2] + δy))
end

n_cycles = 10
mass_history = Float64[]
mom_x_history = Float64[]
mom_y_history = Float64[]
E_history = Float64[]
liou_min_history = Float64[]
liou_max_history = Float64[]
walltime_float = Float64[]

# Initial totals (after one-time deformation)
push!(mass_history, dfmm.total_mass_weighted_lagrangian(lag_mesh, lag_fields, :rho))
push!(mom_x_history, dfmm.total_mass_weighted_lagrangian(lag_mesh, lag_fields, :ux))
push!(mom_y_history, dfmm.total_mass_weighted_lagrangian(lag_mesh, lag_fields, :uy))
push!(E_history, dfmm.total_mass_weighted_lagrangian(lag_mesh, lag_fields, :E))

for cycle in 1:n_cycles
    t0 = time()
    dfmm.remap_round_trip!(state, lag_mesh, lag_fields; backend = :float)
    push!(walltime_float, time() - t0)

    push!(mass_history, dfmm.total_mass_weighted_lagrangian(lag_mesh, lag_fields, :rho))
    push!(mom_x_history, dfmm.total_mass_weighted_lagrangian(lag_mesh, lag_fields, :ux))
    push!(mom_y_history, dfmm.total_mass_weighted_lagrangian(lag_mesh, lag_fields, :uy))
    push!(E_history, dfmm.total_mass_weighted_lagrangian(lag_mesh, lag_fields, :E))

    push!(liou_min_history, state.diagnostics.liouville_min)
    push!(liou_max_history, state.diagnostics.liouville_max)
end

# Also time one :exact round trip on the final state. The exact backend
# can hit the documented HG D=2 0//0 collinear-triangle degeneracy on
# certain deformed configurations (`~/.julia/dev/HierarchicalGrids/docs/src/exact_backend.md`);
# we catch the ArgumentError and report it as the caveat being
# observed in dfmm's actual deformation history.
walltime_exact = try
    t0 = time()
    dfmm.bayesian_remap_l_to_e!(state, lag_mesh, lag_fields; backend = :exact)
    time() - t0
catch err
    @warn "IntExact backend hit documented D=2 0//0 caveat on deformed state" err
    NaN
end

@info "M3-5 demo timings (per round trip)" mean_float = sum(walltime_float) / length(walltime_float) walltime_exact_l_to_e_only = walltime_exact

# ---------------------------------------------------------------------------
# Plot 4-panel summary
# ---------------------------------------------------------------------------
fig = Figure(size = (1100, 800))

cycles = 0:n_cycles

axM = Axis(fig[1, 1], title = "Mass conservation", xlabel = "cycle", ylabel = "Σρ V")
lines!(axM, cycles, mass_history)
hlines!(axM, [mass_history[1]], color = :red, linestyle = :dash)

axP = Axis(fig[1, 2], title = "Momentum conservation",
            xlabel = "cycle", ylabel = "Σ ρu V")
lines!(axP, cycles, mom_x_history, label = "ρ ux")
lines!(axP, cycles, mom_y_history, label = "ρ uy")
axislegend(axP, position = :rt)

axE = Axis(fig[2, 1], title = "Energy conservation",
            xlabel = "cycle", ylabel = "Σ E V")
lines!(axE, cycles, E_history)
hlines!(axE, [E_history[1]], color = :red, linestyle = :dash)

axL = Axis(fig[2, 2], title = "Liouville extrema",
            xlabel = "cycle", ylabel = "proxy = entry.vol / src_phys_vol")
lines!(axL, 1:n_cycles, liou_min_history, label = "min")
lines!(axL, 1:n_cycles, liou_max_history, label = "max")
axislegend(axL, position = :rt)

# Save
out_path = joinpath(@__DIR__, "..", "reference", "figs",
                    "M3_5_remap_conservation.png")
mkpath(dirname(out_path))
save(out_path, fig)
@info "M3-5 demo plot saved" out_path

# ---------------------------------------------------------------------------
# Summary log
# ---------------------------------------------------------------------------
println("\n=== M3-5 Bayesian L↔E remap demo summary ===")
println("Setup: 16×16 Eulerian + 512-triangle Lagrangian (2 per leaf)")
println("n_cycles = $n_cycles, A_def (one-time) = $A_def")
println()
println("Mass conservation (L1 history):")
for (i, M) in enumerate(mass_history)
    rel = abs(M - mass_history[1]) / abs(mass_history[1])
    println("  cycle $(i-1): M = $M  (rel-err vs t=0: $rel)")
end
println()
println("Liouville-extrema histories:")
for c in 1:n_cycles
    println("  cycle $c: min = $(liou_min_history[c]), max = $(liou_max_history[c])")
end
println()
println("Wall time: float L→E→L = $(sum(walltime_float)) s ($n_cycles cycles)")
println("            exact L→E   = $walltime_exact s (single pass on deformed mesh)")
