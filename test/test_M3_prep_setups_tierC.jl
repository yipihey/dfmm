# test_M3_prep_setups_tierC.jl
#
# Tier-C IC factory unit tests, prepping the M3-4 dimension-lift gate
# (methods paper §10.4: 1D-symmetric Sod / cold sinusoid / plane wave).
#
# These tests cover the IC half only — they verify that:
#
#   - The dimension-generic factory `tier_c_*_ic` builds an HG
#     `HierarchicalMesh{D}` of the requested resolution and populates
#     the named cell-average fields `(rho, ux, [uy, [uz,]] P)` with
#     the analytical IC's cell averages.
#   - Mass conservation holds to round-off (`Σ_j ρ_j V_j ≈ ∫ ρ dx`).
#   - For 1D-symmetric configurations, the y-direction is independent
#     to 1e-12 (cells with the same x-coordinate are bit-identical).
#   - Sample primitive values at known points match the analytical IC
#     to round-off (4-6 sample points per factory).
#   - The plane wave is rotationally invariant: at θ = 0, π/2 the
#     factory produces fields related by a 90° rotation of velocity
#     components, to plotting precision.
#   - The plane wave's δρ–δu phase relation matches the right-going
#     acoustic mode `δu = (c/ρ₀) δρ k̂` to round-off in the cell
#     average.
#
# Solver coupling is **not** tested here (that lands at M3-4); these
# are pure setup tests on the field-set initial values.

using Test
using dfmm
using HierarchicalGrids: enumerate_leaves, cell_physical_box, n_cells

# Convenience: read the cell-center primitives at leaf j.
function _cell_state(ic, j::Integer)
    centers = tier_c_cell_centers(ic)
    return (rho = ic.fields.rho[j][1],
            ux  = tier_c_velocity_component(ic.fields, j, 1),
            uy  = ic.params.D >= 2 ?
                    tier_c_velocity_component(ic.fields, j, 2) : 0.0,
            uz  = ic.params.D >= 3 ?
                    tier_c_velocity_component(ic.fields, j, 3) : 0.0,
            P   = ic.fields.P[j][1],
            x   = centers[j])
end


@testset "Tier-C IC factories — M3-prep setup-only tests" begin

    # ----- C.1 1D-symmetric 2D Sod -----------------------------------
    @testset "C.1 tier_c_sod_ic" begin
        ic = tier_c_sod_ic(D = 2, level = 4)
        @test ic.name == "tier_c_sod"
        @test ic.params.D == 2
        @test ic.params.shock_axis == 1
        n_leaves = length(enumerate_leaves(ic.mesh))
        @test n_leaves == 16 * 16

        # Mass conservation: total = 0.5*(ρL + ρR) on default
        # box [0,1]×[0,0.2] with x_split=0.5 ⇒ M = 0.5*(1.0+0.125)*0.2.
        M_analytical = 0.5 * (1.0 + 0.125) * 0.2
        M_numerical = tier_c_total_mass(ic)
        @test M_numerical ≈ M_analytical atol = 1e-12

        # Two-state primitives at far-left and far-right cells.
        c_first = _cell_state(ic, 1)
        c_last  = _cell_state(ic, n_leaves)
        # First leaf in DFS order: somewhere in the bottom-left quadrant
        # of [0, 0.5] × [0, 0.1]. Pick a cell explicitly to avoid
        # over-relying on DFS ordering. We assert by location.
        centers = tier_c_cell_centers(ic)
        # Find a cell deep in the left half (x < 0.4) and one deep in
        # the right half (x > 0.6).
        idx_L = findfirst(c -> c[1] < 0.4 && c[2] < 0.05, centers)
        idx_R = findfirst(c -> c[1] > 0.6 && c[2] < 0.05, centers)
        @test idx_L !== nothing && idx_R !== nothing
        cL = _cell_state(ic, idx_L)
        cR = _cell_state(ic, idx_R)
        # Quadrature weights sum to 1 ± ε, so cell averages of constants
        # come out as `ρ * (1 + ε)`. We assert to round-off, not bitwise.
        @test cL.rho ≈ 1.0   atol = 1e-13
        @test cR.rho ≈ 0.125 atol = 1e-13
        @test cL.P   ≈ 1.0   atol = 1e-13
        @test cR.P   ≈ 0.1   atol = 1e-13
        @test cL.ux == 0.0   # exact: factor of 0.0 short-circuits
        @test cR.ux == 0.0
        @test cL.uy == 0.0
        @test cR.uy == 0.0

        # y-direction independence: cells with the same x-cell-index
        # along shock_axis must be bit-identical to round-off. Group
        # cells by their x-center value.
        by_x = Dict{Float64, Vector{Int}}()
        for (j, c) in enumerate(centers)
            push!(get!(by_x, round(c[1], digits = 12), Int[]), j)
        end
        max_y_dev = 0.0
        for (_, js) in by_x
            length(js) <= 1 && continue
            ref = _cell_state(ic, js[1])
            for j in js[2:end]
                cs = _cell_state(ic, j)
                max_y_dev = max(max_y_dev,
                                abs(cs.rho - ref.rho),
                                abs(cs.P   - ref.P),
                                abs(cs.ux  - ref.ux),
                                abs(cs.uy  - ref.uy))
            end
        end
        @test max_y_dev < 1e-12

        # 1D variant: shock_axis = 1 in D = 1 should match M1's
        # setup_sod two-state IC for cells away from the discontinuity.
        ic1 = tier_c_sod_ic(D = 1, level = 6)  # 64 cells
        c1 = tier_c_cell_centers(ic1)
        # Cell with x_center = 0.25 is firmly in the left half:
        idx = argmin(abs.(getindex.(c1, 1) .- 0.25))
        s = _cell_state(ic1, idx)
        @test s.rho ≈ 1.0 atol = 1e-13
        @test s.P   ≈ 1.0 atol = 1e-13
        # Cell with x_center = 0.75 is firmly in the right half:
        idx = argmin(abs.(getindex.(c1, 1) .- 0.75))
        s = _cell_state(ic1, idx)
        @test s.rho ≈ 0.125 atol = 1e-13
        @test s.P   ≈ 0.1   atol = 1e-13

        # Custom shock axis: shock_axis=2 ⇒ jump along y, x-direction
        # independent.
        ic_y = tier_c_sod_ic(D = 2, level = 4, shock_axis = 2,
                             x_split = 0.1,
                             lo = (0.0, 0.0), hi = (0.2, 1.0))
        cy = tier_c_cell_centers(ic_y)
        idx_lo = findfirst(c -> c[2] < 0.05 && c[1] < 0.1, cy)
        idx_hi = findfirst(c -> c[2] > 0.9  && c[1] < 0.1, cy)
        @test _cell_state(ic_y, idx_lo).rho ≈ 1.0   atol = 1e-13
        @test _cell_state(ic_y, idx_hi).rho ≈ 0.125 atol = 1e-13
    end

    # ----- C.2 2D cold sinusoid --------------------------------------
    @testset "C.2 tier_c_cold_sinusoid_ic" begin
        # 1D-symmetric variant: k = (1, 0). uy must be 0 for every cell.
        ic = tier_c_cold_sinusoid_ic(D = 2, level = 4, A = 0.5,
                                     k = (1, 0))
        @test ic.name == "tier_c_cold_sinusoid"
        @test ic.params.k == (1, 0)
        n = length(enumerate_leaves(ic.mesh))
        # Density and pressure are uniform — cell average of a constant
        # is the constant up to sum-of-quadrature-weights round-off
        # (~1e-15).
        @test maximum(abs(ic.fields.uy[j][1]) for j in 1:n) == 0.0
        @test maximum(abs(ic.fields.rho[j][1] - 1.0) for j in 1:n) < 1e-13
        @test maximum(abs(ic.fields.P[j][1]   - 1.0) for j in 1:n) < 1e-13

        # Mass conservation (uniform density ⇒ M = ρ₀ * box_volume).
        @test tier_c_total_mass(ic) ≈ 1.0 atol = 1e-11

        # y-independence: cells at same x-coordinate have identical ux.
        centers = tier_c_cell_centers(ic)
        by_x = Dict{Float64, Vector{Int}}()
        for (j, c) in enumerate(centers)
            push!(get!(by_x, round(c[1], digits = 12), Int[]), j)
        end
        max_dev = 0.0
        for (_, js) in by_x
            length(js) <= 1 && continue
            ref = ic.fields.ux[js[1]][1]
            for j in js[2:end]
                max_dev = max(max_dev, abs(ic.fields.ux[j][1] - ref))
            end
        end
        @test max_dev < 1e-12

        # ux at sample x-locations matches the analytical cell average
        # (1/Δx) ∫ A sin(2π x) dx = A/(2π Δx) * (cos(2π a) - cos(2π(a+Δx))).
        # The factory's Gauss-Legendre quadrature (default order 3 = 5-th
        # degree exact) is *not* exact for sin(·), so we allow ~1e-9
        # quadrature slack. For tighter comparison we re-build the IC
        # with a higher quadrature order and check the error shrinks.
        leaves_j = enumerate_leaves(ic.mesh)
        ux_err_default = 0.0
        for j in 1:n
            lo_c, hi_c = cell_physical_box(ic.frame, leaves_j[j])
            Δx = hi_c[1] - lo_c[1]
            ux_ana = 0.5 * (cos(2π * lo_c[1]) - cos(2π * hi_c[1])) / (2π) / Δx
            ux_err_default = max(ux_err_default,
                                  abs(ic.fields.ux[j][1] - ux_ana))
        end
        @test ux_err_default < 1e-9   # 3-point Gauss-Legendre on sin(·)

        # Tighter rule: bump quadrature to 8 points → exact to 1e-15.
        ic_hi = tier_c_cold_sinusoid_ic(D = 2, level = 4, A = 0.5,
                                         k = (1, 0), quad_order = 8)
        leaves_hi = enumerate_leaves(ic_hi.mesh)
        ux_err_hi = 0.0
        for j in 1:n
            lo_c, hi_c = cell_physical_box(ic_hi.frame, leaves_hi[j])
            Δx = hi_c[1] - lo_c[1]
            ux_ana = 0.5 * (cos(2π * lo_c[1]) - cos(2π * hi_c[1])) / (2π) / Δx
            ux_err_hi = max(ux_err_hi,
                             abs(ic_hi.fields.ux[j][1] - ux_ana))
        end
        @test ux_err_hi < 1e-13   # quadrature converges with `quad_order`

        # 2D variant: k = (1, 1). Both ux and uy non-trivial.
        ic2 = tier_c_cold_sinusoid_ic(D = 2, level = 4, A = 0.5,
                                      k = (1, 1))
        n2 = length(enumerate_leaves(ic2.mesh))
        @test maximum(abs(ic2.fields.ux[j][1]) for j in 1:n2) > 0
        @test maximum(abs(ic2.fields.uy[j][1]) for j in 1:n2) > 0

        # Net momentum (per-axis) is zero by symmetry of sin on the
        # periodic box; cell-averaged sum should be at round-off.
        for d in 1:2
            sum_u = sum(tier_c_velocity_component(ic2.fields, j, d)
                          for j in 1:n2)
            @test abs(sum_u) < 1e-11
        end

        # Total mass = ρ₀ × box volume.
        @test tier_c_total_mass(ic2) ≈ 1.0 atol = 1e-12

        # 1D variant: D = 1 must match.
        ic1 = tier_c_cold_sinusoid_ic(D = 1, level = 6, A = 0.5,
                                      k = (1,), quad_order = 8)
        n1 = length(enumerate_leaves(ic1.mesh))
        # Same period: ux at x = 1/4 should be A * 1.
        c1 = tier_c_cell_centers(ic1)
        idx = argmin(abs.(getindex.(c1, 1) .- 0.25))
        # At level=6, Δx = 1/64. Analytical cell-average of A sin(2π x)
        # over [a, a+Δx] is (A/(2π Δx))*(cos(2π a) - cos(2π(a+Δx))).
        x_lo = c1[idx][1] - 1/128
        x_hi = c1[idx][1] + 1/128
        ux_ana = 0.5 * (cos(2π*x_lo) - cos(2π*x_hi)) / (2π) / (1/64)
        @test ic1.fields.ux[idx][1] ≈ ux_ana atol = 1e-13
    end

    # ----- C.3 plane wave at angle θ ---------------------------------
    @testset "C.3 tier_c_plane_wave_ic — rotational invariance" begin
        # θ = 0 vs θ = π/2: the field at (x, y) of the θ=0 IC should
        # equal the field at (y, x) of the θ=π/2 IC, after swapping
        # ux ↔ uy. We test this on a 2D level=4 grid.
        ic_x = tier_c_plane_wave_ic(D = 2, level = 4, A = 1e-3,
                                    k_mag = 1, angle = 0.0)
        ic_y = tier_c_plane_wave_ic(D = 2, level = 4, A = 1e-3,
                                    k_mag = 1, angle = π/2)
        n = length(enumerate_leaves(ic_x.mesh))
        @test n == 256
        cx = tier_c_cell_centers(ic_x)
        cy = tier_c_cell_centers(ic_y)

        # At θ = 0, k = (1, 0): δρ depends on x only; uy = 0.
        @test maximum(abs(ic_x.fields.uy[j][1]) for j in 1:n) < 1e-15
        # At θ = π/2, k = (0, 1): δρ depends on y only; ux ≈ 0.
        @test maximum(abs(ic_x.fields.ux[j][1]) for j in 1:n) > 0
        @test maximum(abs(ic_y.fields.ux[j][1]) for j in 1:n) < 1e-15
        @test maximum(abs(ic_y.fields.uy[j][1]) for j in 1:n) > 0

        # The δρ envelope is identical between θ=0 and θ=π/2
        # (same k_mag, same box).
        max_drho_x = maximum(abs(ic_x.fields.rho[j][1] - 1.0)
                                 for j in 1:n)
        max_drho_y = maximum(abs(ic_y.fields.rho[j][1] - 1.0)
                                 for j in 1:n)
        @test max_drho_x ≈ max_drho_y atol = 1e-12

        # Pointwise match: rho at (x, y) under θ=0 equals rho at (y, x)
        # under θ=π/2 (the rotation by 90°). We pair cells by
        # (x_center, y_center) ↔ (y_center, x_center).
        idx_y_for_xy = Dict{NTuple{2, Float64}, Int}()
        for (j, c) in enumerate(cy)
            idx_y_for_xy[(round(c[1], digits=12), round(c[2], digits=12))] = j
        end
        max_pair_err = 0.0
        for (jx, c) in enumerate(cx)
            key = (round(c[2], digits=12), round(c[1], digits=12))
            jy = idx_y_for_xy[key]
            max_pair_err = max(max_pair_err,
                abs(ic_x.fields.rho[jx][1] - ic_y.fields.rho[jy][1]),
                abs(ic_x.fields.P[jx][1]   - ic_y.fields.P[jy][1]),
                abs(ic_x.fields.ux[jx][1]  - ic_y.fields.uy[jy][1]),
                abs(ic_x.fields.uy[jx][1]  - ic_y.fields.ux[jy][1]))
        end
        @test max_pair_err < 1e-13   # pointwise rotational invariance

        # The acoustic-mode amplitude relation: δu_d = (c/ρ₀) δρ k̂_d.
        # In the θ=0 IC, k̂ = (1, 0), so ux = (c/ρ₀) * (rho - ρ₀).
        # We assert this cell-by-cell to round-off.
        c_eff = ic_x.params.c
        ρ0 = ic_x.params.ρ0
        max_phase_err = 0.0
        for j in 1:n
            δρ = ic_x.fields.rho[j][1] - ρ0
            ux = ic_x.fields.ux[j][1]
            max_phase_err = max(max_phase_err, abs(ux - c_eff/ρ0 * δρ))
        end
        @test max_phase_err < 1e-13

        # Mass conservation: the cosine integrates to zero over the
        # periodic box, so total mass = ρ₀ × box volume.
        @test tier_c_total_mass(ic_x) ≈ 1.0 atol = 1e-12

        # Rotational invariance to plotting precision (1e-3) — coarse
        # grid satisfies the brief's gentle target with room to spare.
        @test max_pair_err < 1e-3
    end

    # ----- Plane-wave convergence sample -----------------------------
    @testset "C.3 plane wave — convergence δρ vs L₁ refinement" begin
        # The cell-average of cos(2π x) over [a, a+Δx] equals
        # (sin(2π(a+Δx)) - sin(2π a)) / (2π Δx) and approaches the
        # pointwise cos(2π x_center) as Δx → 0 with O(Δx²) leading-
        # order coefficient (-π²/6 from the Taylor expansion). For our
        # smooth IC the L∞ cell-average-vs-pointwise error decreases
        # like 4^{-level} (= Δx²). We confirm this on three levels.
        errs = Float64[]
        for level in 4:6
            ic = tier_c_plane_wave_ic(D = 2, level = level, A = 1e-3,
                                      k_mag = 1, angle = 0.0)
            n = length(enumerate_leaves(ic.mesh))
            centers = tier_c_cell_centers(ic)
            err = 0.0
            for j in 1:n
                xc = centers[j][1]
                δρ_pt = 1e-3 * cos(2π * xc)
                δρ_cell = ic.fields.rho[j][1] - 1.0
                err = max(err, abs(δρ_cell - δρ_pt))
            end
            push!(errs, err)
        end
        # Convergence ratio (O(Δx²) ⇒ ratio close to 4 per level).
        @test errs[1] / errs[2] > 3.5    # 4^1 with finite-grid slack
        @test errs[2] / errs[3] > 3.5
        # Sanity: errors stay below the brief's plotting precision
        # at level 4 (1e-3) and the per-axis γ scale (1e-5) at level 6.
        # Analytical leading order is `π²/6 * (Δx)² * A` (cosine Taylor
        # expansion). For A = 1e-3, level 4 ⇒ Δx = 1/16 ⇒ ~6.4e-6;
        # level 6 ⇒ Δx = 1/64 ⇒ ~4.0e-7. We assert the latter.
        @test errs[1] < 1e-5
        @test errs[3] < 1e-6
    end

end  # @testset "Tier-C IC factories"
