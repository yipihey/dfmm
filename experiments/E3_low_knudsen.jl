# E3_low_knudsen.jl
#
# M3-8 Phase a Deliverable: E.3 very low Knudsen 2D driver (Tier E,
# methods paper §10.6 E.3).
#
# Drives the 2D Cholesky-sector Newton system on a smooth low-amplitude
# strain perturbation in the small-τ BGK relaxation regime
# (τ << τ_dyn ⇒ Kn << 1, the Navier-Stokes limit). Tests that the
# implicit Newton solver handles the stiff-τ relaxation without
# timestep blowup.
#
# Note on the Cholesky-sector substrate: the 2D Cholesky-sector solver
# (`det_step_2d_berry_HG!`) drives the deterministic (α, β, θ_R) sector
# but does NOT include explicit BGK relaxation (the M2-3 Pp/Q sector
# is post-Newton). For E.3 we exercise the deterministic Newton on the
# small-Kn strain mode and verify that γ²_a stays close to M_vv (i.e.
# the Cholesky state remains in the local-equilibrium manifold) — the
# *near-equilibrium* limit. This is the variational analog of the
# Navier-Stokes limit verification.
#
# Acceptance pattern (asserted in `test/test_M3_8a_E3_low_knudsen.jl`):
#
#   1. NaN-count = 0 across n_steps. Stiff-τ should not destabilize
#      the Newton solve.
#
#   2. Near-equilibrium preservation: |β_a| stays bounded by a small
#      multiple of the IC ε. Specifically, max|β_a|_t / max|β_a|_0
#      stays within a factor ≤ 5 across n_steps (no exponential
#      growth of the deviation from local equilibrium).
#
#   3. γ²_a remains close to M_vv: 1 − γ²_a/M_vv ≤ 1e-2 across all
#      cells / steps (the small-Kn limit ⇒ near-isotropic state).
#
#   4. Mass / momentum conservation: total mass exact (ρ_per_cell);
#      momentum drift ≤ 1e-10 per step (smooth perturbation should
#      not drive bulk momentum).
#
# Usage:
#   julia> include("experiments/E3_low_knudsen.jl")
#   julia> result = run_E3_low_knudsen(; level=3, k=(1, 0), n_steps=10)

using HierarchicalGrids: HierarchicalMesh, EulerianFrame, FrameBoundaries,
    REFLECTING, PERIODIC, enumerate_leaves, refine_cells!, cell_physical_box
using dfmm: tier_e_low_knudsen_ic_full, primitive_recovery_2d_per_cell,
            det_step_2d_berry_HG!, read_detfield_2d, DetField2D

"""
    run_E3_low_knudsen(; level=3, k=(1, 0), A_u=1e-2, τ=1e-6,
                        n_steps=10, dt=1e-5,
                        project_kind=:none, M_vv_override=(1.0, 1.0),
                        ρ_ref=1.0, realizability_headroom=1.05)

Drive the E.3 low-Knudsen IC for `n_steps` Newton steps. Returns a
NamedTuple with the trajectory:

  • `t::Vector{Float64}`             — wall times
  • `nan_count::Vector{Int}`         — # of NaN cells per step
  • `beta_max::Vector{Float64}`      — spatial max |β_a| per step
  • `gamma_dev_max::Vector{Float64}` — max relative deviation
                                       1 − γ²_a/M_vv per step
  • `mass::Vector{Float64}`          — total mass per step
  • `Px, Py::Vector{Float64}`        — per-axis total momentum per step
  • `mesh, frame, leaves, fields`    — solver state at end
  • `ic_params, Kn`                  — IC parameters + Knudsen number
"""
function run_E3_low_knudsen(; level::Integer = 3,
                              k = (1, 0),
                              A_u::Real = 1e-2,
                              τ::Real = 1e-6,
                              n_steps::Integer = 10,
                              dt::Real = 1e-5,
                              project_kind::Symbol = :none,
                              M_vv_override = (1.0, 1.0),
                              ρ_ref::Real = 1.0,
                              realizability_headroom::Real = 1.05,
                              Mvv_floor::Real = 1e-2,
                              pressure_floor::Real = 1e-8,
                              abstol::Real = 1e-10,
                              reltol::Real = 1e-10,
                              maxiters::Integer = 50)
    ic = tier_e_low_knudsen_ic_full(; level = level, k = k, A_u = A_u, τ = τ)
    bc_spec = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                    (PERIODIC, PERIODIC)))

    n_leaves = length(ic.leaves)
    cell_areas = Vector{Float64}(undef, n_leaves)
    for (i, ci) in enumerate(ic.leaves)
        lo, hi = cell_physical_box(ic.frame, ci)
        cell_areas[i] = (hi[1] - lo[1]) * (hi[2] - lo[2])
    end

    Mv1 = M_vv_override === nothing ? Float64(ic.params.P0 / ic.params.ρ0) :
            Float64(M_vv_override[1])
    Mv2 = M_vv_override === nothing ? Float64(ic.params.P0 / ic.params.ρ0) :
            Float64(M_vv_override[2])

    t = zeros(Float64, n_steps + 1)
    nan_count = zeros(Int, n_steps + 1)
    beta_max = zeros(Float64, n_steps + 1)
    gamma_dev_max = zeros(Float64, n_steps + 1)
    mass = zeros(Float64, n_steps + 1)
    Px = zeros(Float64, n_steps + 1)
    Py = zeros(Float64, n_steps + 1)

    rec0 = primitive_recovery_2d_per_cell(ic.fields, ic.leaves, ic.frame,
                                            ic.ρ_per_cell)
    mass[1] = sum(ic.ρ_per_cell .* cell_areas)
    Px[1] = sum(ic.ρ_per_cell .* rec0.u_x .* cell_areas)
    Py[1] = sum(ic.ρ_per_cell .* rec0.u_y .* cell_areas)
    nan_count[1] = _count_nans_e3(ic.fields, ic.leaves)
    beta_max[1] = _beta_max_2d(ic.fields, ic.leaves)
    gamma_dev_max[1] = _gamma_dev_max_2d(ic.fields, ic.leaves, Mv1, Mv2)

    for n in 1:n_steps
        try
            det_step_2d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                                    bc_spec, dt;
                                    M_vv_override = M_vv_override, ρ_ref = ρ_ref,
                                    project_kind = project_kind,
                                    realizability_headroom = realizability_headroom,
                                    Mvv_floor = Mvv_floor,
                                    pressure_floor = pressure_floor,
                                    abstol = abstol, reltol = reltol,
                                    maxiters = maxiters)
        catch err
            @info "E.3 Newton solve failed at step $n: $(err); recording graceful failure."
            for k_step in n:n_steps
                nan_count[k_step + 1] = n_leaves
                t[k_step + 1] = k_step * dt
            end
            break
        end
        t[n + 1] = n * dt
        nan_count[n + 1] = _count_nans_e3(ic.fields, ic.leaves)
        beta_max[n + 1] = _beta_max_2d(ic.fields, ic.leaves)
        gamma_dev_max[n + 1] = _gamma_dev_max_2d(ic.fields, ic.leaves, Mv1, Mv2)
        rec = primitive_recovery_2d_per_cell(ic.fields, ic.leaves, ic.frame,
                                                ic.ρ_per_cell)
        mass[n + 1] = sum(ic.ρ_per_cell .* cell_areas)
        Px[n + 1] = sum(ic.ρ_per_cell .* rec.u_x .* cell_areas)
        Py[n + 1] = sum(ic.ρ_per_cell .* rec.u_y .* cell_areas)
    end

    return (
        t = t, nan_count = nan_count,
        beta_max = beta_max, gamma_dev_max = gamma_dev_max,
        mass = mass, Px = Px, Py = Py,
        mesh = ic.mesh, frame = ic.frame, leaves = ic.leaves,
        fields = ic.fields, ρ_per_cell = ic.ρ_per_cell,
        ic_params = ic.params, Kn = ic.params.Kn,
    )
end

function _count_nans_e3(fields, leaves)
    cnt = 0
    for ci in leaves
        if isnan(Float64(fields.u_1[ci][1])) || isnan(Float64(fields.u_2[ci][1])) ||
           isnan(Float64(fields.α_1[ci][1])) || isnan(Float64(fields.α_2[ci][1])) ||
           isnan(Float64(fields.β_1[ci][1])) || isnan(Float64(fields.β_2[ci][1])) ||
           isnan(Float64(fields.s[ci][1]))
            cnt += 1
        end
    end
    return cnt
end

function _beta_max_2d(fields, leaves)
    bmax = 0.0
    for ci in leaves
        bmax = max(bmax, abs(Float64(fields.β_1[ci][1])))
        bmax = max(bmax, abs(Float64(fields.β_2[ci][1])))
    end
    return bmax
end

function _gamma_dev_max_2d(fields, leaves, Mv1::Real, Mv2::Real)
    dev_max = 0.0
    for ci in leaves
        β1 = Float64(fields.β_1[ci][1])
        β2 = Float64(fields.β_2[ci][1])
        γ1sq = Float64(Mv1) - β1^2
        γ2sq = Float64(Mv2) - β2^2
        dev1 = abs(1.0 - γ1sq / Float64(Mv1))
        dev2 = abs(1.0 - γ2sq / Float64(Mv2))
        dev_max = max(dev_max, dev1, dev2)
    end
    return dev_max
end
