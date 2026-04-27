# D10_ism_multi_tracer.jl
#
# §10.5 D.10 ISM-like 2D multi-tracer fidelity in shocked turbulence —
# M3-6 Phase 5. Closes M3-6 entire.
# (`reference/notes_M3_6_phase5_D10_ism_tracers.md`).
#
# Methods paper §10.5 D.10: "ISM-like 2D problem with metallicity
# tracking. Multi-tracer 2D shocked turbulence; metallicity PDF and
# spatial structure consistent with Lagrangian methods (SPH, AREPO);
# significantly better than Eulerian methods. Methods-paper community-
# impact test."
#
# The dfmm 2D variational scheme tests this via a multi-phase ISM-like
# IC carrying N_species ≥ 3 passive scalars on a sheared KH base flow,
# driven through `det_step_2d_berry_HG!` + `inject_vg_noise_HG_2d!`
# with axes=(1, 2) for the "shocked turbulence" regime. Per step:
# `advect_tracers_HG_2d!` (no-op pure-Lagrangian).
#
# **Headline acceptance:** the tracer matrix at end-time is *byte-equal*
# to the IC tracer matrix even after K stochastic injection +
# deterministic-step iterations. This is the 2D analog of the M2-2
# 1D structural bit-exactness argument. The bit-exactness derives
# from inspection of the write sets:
#
#   * `det_step_2d_berry_HG!`: writes only the 14 fluid Cholesky-sector
#     fields (`x_a, u_a, α_a, β_a, β_12, β_21, θ_R, s, Pp, Q`).
#   * `inject_vg_noise_HG_2d!`: per-axis writes to `(β_a, u_a, s, Pp)`.
#   * Neither writes `tm.tracers`. Therefore `tm.tracers` is byte-stable
#     by construction.
#
# Per step the driver tracks:
#   • per-species per-cell concentration via `tm.tracers`
#   • per-species per-axis γ via `gamma_per_axis_2d_per_species_field`
#   • per-species mass conservation: `Σ_leaves c_k · A_cell`
#   • n_negative_jacobian (gas-only diagnostic; species without M_vv
#     override default to `M_vv_iso = Mvv(J, s)` from the EOS)
#   • conservation invariants (M, Px, Py, KE)
#   • spatial-profile snapshots
#
# Headline plot (4-panel CairoMakie):
#   (A) gas density / velocity at end-time (shocked turbulence map)
#   (B) 3 species concentrations at end-time (overlaid)
#   (C) per-species mass conservation `M_k(t) - M_k(0)` over time
#   (D) per-species γ trajectories
#
# Usage (REPL):
#
#   julia> include("experiments/D10_ism_multi_tracer.jl")
#   julia> result = run_D10_ism_multi_tracer(; level=4)
#
# Or full sweep + plot:
#
#   julia> sweep = run_D10_ism_multi_tracer_sweep(; levels=(4, 5))
#   julia> plot_D10_ism_multi_tracer(sweep;
#              save_path="reference/figs/M3_6_phase5_D10_ism_tracers.png")

using Random: MersenneTwister
using HierarchicalGrids: HierarchicalMesh, EulerianFrame,
    refine_cells!, enumerate_leaves, FrameBoundaries, REFLECTING, PERIODIC,
    cell_physical_box
using dfmm: tier_d_ism_tracers_ic_full, allocate_cholesky_2d_fields,
    write_detfield_2d!, read_detfield_2d, DetField2D,
    det_step_2d_berry_HG!, gamma_per_axis_2d_field,
    gamma_per_axis_2d_per_species_field,
    advect_tracers_HG_2d!, inject_vg_noise_HG_2d!,
    InjectionDiagnostics2D, NoiseInjectionParams,
    n_species, species_index, n_cells_2d,
    ProjectionStats
using Statistics: mean, std

"""
    ism_kh_time(; U_jet=1.0, L=1.0) -> Float64

KH base-flow timescale `t_KH = L / U_jet`.
"""
ism_kh_time(; U_jet::Real = 1.0, L::Real = 1.0) =
    Float64(L) / max(Float64(U_jet), 1e-300)

"""
    cell_areas_ism_2d(frame, leaves) -> Vector{Float64}

Per-leaf cell area `(hi_x - lo_x)·(hi_y - lo_y)`.
"""
function cell_areas_ism_2d(frame, leaves)
    A = Vector{Float64}(undef, length(leaves))
    @inbounds for (i, ci) in enumerate(leaves)
        lo_c, hi_c = cell_physical_box(frame, ci)
        A[i] = (hi_c[1] - lo_c[1]) * (hi_c[2] - lo_c[2])
    end
    return A
end

"""
    species_mass(tm, leaves, areas, k::Integer) -> Float64

Total integrated mass for species `k`: `Σ c_k · A_cell` over leaves.
"""
function species_mass(tm, leaves, areas, k::Integer)
    s = 0.0
    @inbounds for (i, ci) in enumerate(leaves)
        s += tm.tracers[k, ci] * Float64(areas[i])
    end
    return s
end

"""
    species_mass_vector(tm, leaves, areas) -> Vector{Float64}

Total integrated mass for every species. Length `n_species(tm)`.
"""
function species_mass_vector(tm, leaves, areas)
    ns = n_species(tm)
    out = Vector{Float64}(undef, ns)
    for k in 1:ns
        out[k] = species_mass(tm, leaves, areas, k)
    end
    return out
end

"""
    fluid_invariants_ism(fields, leaves, ρ_per_cell, areas) -> NamedTuple

Compute fluid conservation invariants `(M, Px, Py, KE)`.
"""
function fluid_invariants_ism(fields, leaves, ρ_per_cell, areas)
    M = 0.0; Px = 0.0; Py = 0.0; KE = 0.0
    @inbounds for (i, ci) in enumerate(leaves)
        ρ_i = Float64(ρ_per_cell[i])
        A_i = Float64(areas[i])
        ux = Float64(fields.u_1[ci][1])
        uy = Float64(fields.u_2[ci][1])
        M += ρ_i * A_i
        Px += ρ_i * ux * A_i
        Py += ρ_i * uy * A_i
        KE += 0.5 * ρ_i * (ux * ux + uy * uy) * A_i
    end
    return (M = M, Px = Px, Py = Py, KE = KE)
end

"""
    negative_jacobian_count_ism(fields, leaves; M_vv_override, ρ_ref) -> Int

Count cells with γ_a ≤ 1e-12 on either axis using a per-species
`M_vv_override` (gas-fluid baseline). With M3-6 Phase 1b 4-component
projection this stays at 0 throughout stable runs.
"""
function negative_jacobian_count_ism(fields, leaves;
                                       M_vv_override = (1.0, 1.0),
                                       ρ_ref::Real = 1.0)
    γ = gamma_per_axis_2d_field(fields, leaves;
                                  M_vv_override = M_vv_override,
                                  ρ_ref = ρ_ref)
    n_neg = 0
    @inbounds for i in 1:size(γ, 2)
        if γ[1, i] ≤ 1e-12 || γ[2, i] ≤ 1e-12
            n_neg += 1
        end
    end
    return n_neg
end

"""
    per_species_gamma_summary(fields, leaves; n_species_n, M_vv_per_species,
                                ρ_ref) -> NamedTuple

Compute per-species per-axis γ summary (mean, max). Uses the
`gamma_per_axis_2d_per_species_field` walker. Returns a NamedTuple
with per-species `gamma_mean[k]`, `gamma_max[k]` along axis 1, plus
the species-pair separation diagnostic
`max_pair_separation = max_{i,j} |gamma_mean[i] - gamma_mean[j]|`.
"""
function per_species_gamma_summary(fields, leaves;
                                     n_species_n::Integer = 3,
                                     M_vv_per_species = ((1.0, 1.0),
                                                         (2.0, 2.0),
                                                         (4.0, 4.0)),
                                     ρ_ref::Real = 1.0)
    γ = gamma_per_axis_2d_per_species_field(fields, leaves;
                                              M_vv_override_per_species = M_vv_per_species,
                                              ρ_ref = ρ_ref,
                                              n_species = n_species_n)
    # Shape (n_species, 2, N).
    ns = size(γ, 1)
    mean_axis1 = zeros(Float64, ns)
    mean_axis2 = zeros(Float64, ns)
    max_axis1 = zeros(Float64, ns)
    max_axis2 = zeros(Float64, ns)
    for k in 1:ns
        a1 = γ[k, 1, :]
        a2 = γ[k, 2, :]
        mean_axis1[k] = mean(a1)
        mean_axis2[k] = mean(a2)
        max_axis1[k] = maximum(a1)
        max_axis2[k] = maximum(a2)
    end
    # Pairwise separation: max |γ_i - γ_j| / max(|γ_i|, |γ_j|, eps).
    sep = 0.0
    for i in 1:ns, j in (i + 1):ns
        d = abs(mean_axis1[i] - mean_axis1[j])
        ref = max(abs(mean_axis1[i]), abs(mean_axis1[j]), 1e-300)
        rel = d / ref
        sep = max(sep, rel)
    end
    return (
        gamma_mean_axis1 = mean_axis1,
        gamma_mean_axis2 = mean_axis2,
        gamma_max_axis1 = max_axis1,
        gamma_max_axis2 = max_axis2,
        max_pair_separation = sep,
    )
end

"""
    run_D10_ism_multi_tracer(; level=4, n_species=3, U_jet=1.0,
                              jet_width=0.1, ρ0=1.0, P0=1.0,
                              perturbation_amp=1e-3, perturbation_k=2,
                              ε_phase=0.02,
                              T_factor=0.1, T_end=nothing, dt=nothing,
                              project_kind=:reanchor,
                              realizability_headroom=1.05,
                              Mvv_floor=1e-2, pressure_floor=1e-8,
                              M_vv_override=(1.0, 1.0),
                              M_vv_per_species=nothing,
                              ρ_ref=1.0,
                              stochastic=true, seed=20260426,
                              C_A=0.05, C_B=0.05, λ=2.0, θ_factor=0.5,
                              ke_budget_fraction=0.25, ell_corr=2.0,
                              snapshots_at=(0.0, 0.5, 1.0),
                              verbose=false) -> NamedTuple

Drive a single mesh-level D.10 ISM multi-tracer trajectory under
*stochastic injection enabled*. The "shocked turbulence" regime is
provided by `inject_vg_noise_HG_2d!` with `axes = (1, 2)` after every
deterministic Newton step.

The tracer matrix is recorded pre-run (`tracers_ic`) and at every step;
the headline byte-equality assertion is `tm.tracers == tracers_ic` at
the end-of-run (the 2D analog of the M2-2 1D structural argument).

Returns a NamedTuple with full per-step trajectory + diagnostics + IC
handle + the byte-equality flag `tracers_byte_equal_to_ic`.
"""
function run_D10_ism_multi_tracer(; level::Integer = 4,
                                    n_species::Integer = 3,
                                    U_jet::Real = 1.0,
                                    jet_width::Real = 0.1,
                                    ρ0::Real = 1.0, P0::Real = 1.0,
                                    perturbation_amp::Real = 1e-3,
                                    perturbation_k::Integer = 2,
                                    ε_phase::Real = 0.02,
                                    T_factor::Real = 0.1,
                                    T_end::Union{Real,Nothing} = nothing,
                                    dt::Union{Real,Nothing} = nothing,
                                    project_kind::Symbol = :reanchor,
                                    realizability_headroom::Real = 1.05,
                                    Mvv_floor::Real = 1e-2,
                                    pressure_floor::Real = 1e-8,
                                    M_vv_override = (1.0, 1.0),
                                    M_vv_per_species = nothing,
                                    ρ_ref::Real = 1.0,
                                    stochastic::Bool = true,
                                    seed::Integer = 20260426,
                                    C_A::Real = 0.05, C_B::Real = 0.05,
                                    λ::Real = 2.0, θ_factor::Real = 0.5,
                                    ke_budget_fraction::Real = 0.25,
                                    ell_corr::Real = 2.0,
                                    snapshots_at = (0.0, 0.5, 1.0),
                                    verbose::Bool = false)
    t_KH = ism_kh_time(; U_jet = U_jet)
    T_end_val = T_end === nothing ? Float64(T_factor) * t_KH : Float64(T_end)
    if dt === nothing
        Δx = 1.0 / (2^Int(level))
        dt_val = 0.25 * Δx / max(Float64(U_jet), 1e-300)
        # Cap so we always get at least 30 samples.
        dt_val = min(dt_val, T_end_val / 30.0)
    else
        dt_val = Float64(dt)
    end
    n_steps = max(Int(ceil(T_end_val / dt_val)), 1)
    dt_val = T_end_val / n_steps

    # Build IC.
    ic = tier_d_ism_tracers_ic_full(; level = level, n_species = n_species,
                                     U_jet = U_jet, jet_width = jet_width,
                                     ρ0 = ρ0, P0 = P0,
                                     perturbation_amp = perturbation_amp,
                                     perturbation_k = perturbation_k,
                                     ε_phase = ε_phase)
    bc_ism = FrameBoundaries{2}(((PERIODIC, PERIODIC),
                                  (REFLECTING, REFLECTING)))
    areas = cell_areas_ism_2d(ic.frame, ic.leaves)
    L1 = ic.params.L1
    L2 = ic.params.L2
    lo_box = ic.params.lo
    # Note: don't call `n_species(...)` here — the keyword argument
    # `n_species::Integer` shadows the imported accessor function in
    # this scope. Read directly from the tracer matrix shape.
    ns = size(ic.tm.tracers, 1)

    # Default per-species M_vv: cold/warm/hot with growing dispersion.
    if M_vv_per_species === nothing
        # 3-species default; fallback to a uniform M_vv for ns ≠ 3.
        if ns == 3
            M_vv_per_species = ((1.0, 1.0),
                                 (2.0, 2.0),
                                 (4.0, 4.0))
        else
            M_vv_per_species = ntuple(_ -> (1.0, 1.0), ns)
        end
    else
        @assert length(M_vv_per_species) == ns "M_vv_per_species length must == n_species(tm)"
    end

    # Snapshot the IC tracer matrix BEFORE any step (the bit-equality
    # reference). Take a deep copy so subsequent in-place activity
    # (none expected on this matrix) cannot pollute the reference.
    tracers_ic = copy(ic.tm.tracers)

    # Pre-allocate trajectory arrays.
    N = n_steps + 1
    t = zeros(Float64, N)
    M_per_species_traj = zeros(Float64, ns, N)
    M_traj = zeros(Float64, N)
    Px_traj = zeros(Float64, N)
    Py_traj = zeros(Float64, N)
    KE_traj = zeros(Float64, N)
    n_neg_jac = zeros(Int, N)
    gamma_mean_axis1 = zeros(Float64, ns, N)
    gamma_max_axis1 = zeros(Float64, ns, N)
    max_pair_separation = zeros(Float64, N)
    tracers_max_diff = zeros(Float64, N)  # |tm.tracers - tracers_ic|_inf at each step

    # RNG + injection params + diag.
    rng = MersenneTwister(seed)
    inj_params = NoiseInjectionParams(; C_A = Float64(C_A),
                                       C_B = Float64(C_B),
                                       λ = Float64(λ),
                                       θ_factor = Float64(θ_factor),
                                       ke_budget_fraction = Float64(ke_budget_fraction),
                                       ell_corr = Float64(ell_corr),
                                       pressure_floor = Float64(pressure_floor),
                                       project_kind = project_kind,
                                       realizability_headroom = Float64(realizability_headroom),
                                       Mvv_floor = Float64(Mvv_floor))
    inj_diag = InjectionDiagnostics2D(length(ic.leaves))

    # Snapshot bookkeeping.
    snap_times = collect(Float64, snapshots_at)
    snap_indices = Int[]
    for tf in snap_times
        target = tf * T_end_val
        idx = clamp(Int(round(target / dt_val)) + 1, 1, N)
        push!(snap_indices, idx)
    end
    snapshots = Vector{NamedTuple}(undef, length(snap_times))
    snap_taken = falses(length(snap_times))

    function take_snapshot(t_now)
        x_centres = Vector{Float64}(undef, length(ic.leaves))
        y_centres = Vector{Float64}(undef, length(ic.leaves))
        u1_arr = Vector{Float64}(undef, length(ic.leaves))
        u2_arr = Vector{Float64}(undef, length(ic.leaves))
        c_per = [Vector{Float64}(undef, length(ic.leaves)) for _ in 1:ns]
        for (i, ci) in enumerate(ic.leaves)
            lo_c, hi_c = cell_physical_box(ic.frame, ci)
            x_centres[i] = 0.5 * (lo_c[1] + hi_c[1])
            y_centres[i] = 0.5 * (lo_c[2] + hi_c[2])
            u1_arr[i] = Float64(ic.fields.u_1[ci][1])
            u2_arr[i] = Float64(ic.fields.u_2[ci][1])
            for k in 1:ns
                c_per[k][i] = ic.tm.tracers[k, ci]
            end
        end
        γ_gas = gamma_per_axis_2d_field(ic.fields, ic.leaves;
                                          M_vv_override = M_vv_override,
                                          ρ_ref = ρ_ref)
        γ_gas_1 = γ_gas[1, :]
        return (t = t_now, x_centres = x_centres, y_centres = y_centres,
                u_1 = u1_arr, u_2 = u2_arr, c_per_species = c_per,
                γ_gas_1 = γ_gas_1)
    end

    # Initial diagnostics.
    M_per_species_traj[:, 1] = species_mass_vector(ic.tm, ic.leaves, areas)
    cons0 = fluid_invariants_ism(ic.fields, ic.leaves, ic.ρ_per_cell, areas)
    M_traj[1] = cons0.M; Px_traj[1] = cons0.Px
    Py_traj[1] = cons0.Py; KE_traj[1] = cons0.KE
    n_neg_jac[1] = negative_jacobian_count_ism(ic.fields, ic.leaves;
                                                 M_vv_override = M_vv_override,
                                                 ρ_ref = ρ_ref)
    gstats0 = per_species_gamma_summary(ic.fields, ic.leaves;
                                          n_species_n = ns,
                                          M_vv_per_species = M_vv_per_species,
                                          ρ_ref = ρ_ref)
    gamma_mean_axis1[:, 1] = gstats0.gamma_mean_axis1
    gamma_max_axis1[:, 1] = gstats0.gamma_max_axis1
    max_pair_separation[1] = gstats0.max_pair_separation
    tracers_max_diff[1] = maximum(abs.(ic.tm.tracers .- tracers_ic))

    if 1 in snap_indices
        for (k, idx) in enumerate(snap_indices)
            if idx == 1 && !snap_taken[k]
                snapshots[k] = take_snapshot(0.0)
                snap_taken[k] = true
            end
        end
    end

    proj_stats = ProjectionStats()
    nan_seen = false
    wall_t0 = time()
    for n in 1:n_steps
        try
            det_step_2d_berry_HG!(ic.fields, ic.mesh, ic.frame, ic.leaves,
                                    bc_ism, dt_val;
                                    M_vv_override = M_vv_override,
                                    ρ_ref = ρ_ref,
                                    project_kind = project_kind,
                                    realizability_headroom = realizability_headroom,
                                    Mvv_floor = Mvv_floor,
                                    pressure_floor = pressure_floor,
                                    proj_stats = proj_stats)
            advect_tracers_HG_2d!(ic.tm, dt_val)
            if stochastic
                inject_vg_noise_HG_2d!(ic.fields, ic.mesh, ic.frame,
                                         ic.leaves, bc_ism, dt_val;
                                         params = inj_params,
                                         rng = rng,
                                         axes = (1, 2),
                                         ρ_ref = ρ_ref,
                                         diag = inj_diag,
                                         proj_stats = proj_stats)
            end
        catch e
            if verbose
                @warn "Step $n failed: $e"
            end
            nan_seen = true
            break
        end

        t[n + 1] = n * dt_val
        M_per_species_traj[:, n + 1] = species_mass_vector(ic.tm, ic.leaves, areas)
        cons = fluid_invariants_ism(ic.fields, ic.leaves, ic.ρ_per_cell, areas)
        M_traj[n + 1] = cons.M; Px_traj[n + 1] = cons.Px
        Py_traj[n + 1] = cons.Py; KE_traj[n + 1] = cons.KE
        n_neg_jac[n + 1] = negative_jacobian_count_ism(ic.fields, ic.leaves;
                                                         M_vv_override = M_vv_override,
                                                         ρ_ref = ρ_ref)
        gstats = per_species_gamma_summary(ic.fields, ic.leaves;
                                             n_species_n = ns,
                                             M_vv_per_species = M_vv_per_species,
                                             ρ_ref = ρ_ref)
        gamma_mean_axis1[:, n + 1] = gstats.gamma_mean_axis1
        gamma_max_axis1[:, n + 1] = gstats.gamma_max_axis1
        max_pair_separation[n + 1] = gstats.max_pair_separation
        tracers_max_diff[n + 1] = maximum(abs.(ic.tm.tracers .- tracers_ic))

        for (k, idx) in enumerate(snap_indices)
            if idx == n + 1 && !snap_taken[k]
                snapshots[k] = take_snapshot(t[n + 1])
                snap_taken[k] = true
            end
        end

        if !isfinite(M_traj[n + 1])
            nan_seen = true
            break
        end

        if verbose && (n % max(1, n_steps ÷ 10) == 0)
            @info "Step $n / $n_steps: t=$(round(t[n+1]; digits=4))," *
                  " tracers_max_diff=$(tracers_max_diff[n+1])," *
                  " gamma_sep=$(round(max_pair_separation[n+1]; sigdigits=3))"
        end
    end
    wall_t1 = time()
    wall_time_per_step = (wall_t1 - wall_t0) / max(n_steps, 1)

    # Take any missing snapshots at the final completed step.
    for (k, taken) in enumerate(snap_taken)
        if !taken
            snapshots[k] = take_snapshot(t[end])
            snap_taken[k] = true
        end
    end

    # Conservation drifts.
    M_per_species_err_max = zeros(Float64, ns)
    for k in 1:ns
        M_per_species_err_max[k] = maximum(abs.(M_per_species_traj[k, :] .-
                                                M_per_species_traj[k, 1]))
    end
    M_err_max = maximum(abs.(M_traj .- M_traj[1]))
    Px_err_max = maximum(abs.(Px_traj .- Px_traj[1]))
    Py_err_max = maximum(abs.(Py_traj .- Py_traj[1]))
    KE_err_max = maximum(abs.(KE_traj .- KE_traj[1]))

    tracers_byte_equal_to_ic = ic.tm.tracers == tracers_ic
    tracers_max_diff_final = maximum(abs.(ic.tm.tracers .- tracers_ic))

    return (
        t = t,
        M_per_species_traj = M_per_species_traj,
        M_traj = M_traj, Px_traj = Px_traj,
        Py_traj = Py_traj, KE_traj = KE_traj,
        M_per_species_err_max = M_per_species_err_max,
        M_err_max = M_err_max, Px_err_max = Px_err_max,
        Py_err_max = Py_err_max, KE_err_max = KE_err_max,
        n_negative_jacobian = n_neg_jac,
        gamma_mean_axis1 = gamma_mean_axis1,
        gamma_max_axis1 = gamma_max_axis1,
        max_pair_separation = max_pair_separation,
        tracers_ic = tracers_ic,
        tracers_max_diff_traj = tracers_max_diff,
        tracers_max_diff_final = tracers_max_diff_final,
        tracers_byte_equal_to_ic = tracers_byte_equal_to_ic,
        snapshot_times = snap_times,
        snapshot_indices = snap_indices,
        snapshots = snapshots,
        t_KH = t_KH,
        wall_time_per_step = wall_time_per_step,
        nan_seen = nan_seen,
        proj_stats_total = (
            n_steps = proj_stats.n_steps,
            n_events = proj_stats.n_events,
            n_floor_events = proj_stats.n_floor_events,
            n_offdiag_events = proj_stats.n_offdiag_events,
            total_dE_inj = proj_stats.total_dE_inj,
        ),
        ic = ic,
        params = (level = level, n_species = ns,
                   U_jet = Float64(U_jet), jet_width = Float64(jet_width),
                   ρ0 = Float64(ρ0), P0 = Float64(P0),
                   perturbation_amp = Float64(perturbation_amp),
                   perturbation_k = Int(perturbation_k),
                   ε_phase = Float64(ε_phase),
                   dt = dt_val, n_steps = n_steps,
                   T_end = T_end_val, T_factor = Float64(T_factor),
                   project_kind = project_kind,
                   realizability_headroom = Float64(realizability_headroom),
                   Mvv_floor = Float64(Mvv_floor),
                   pressure_floor = Float64(pressure_floor),
                   M_vv_override = M_vv_override,
                   M_vv_per_species = M_vv_per_species,
                   ρ_ref = Float64(ρ_ref),
                   stochastic = stochastic, seed = Int(seed),
                   C_A = Float64(C_A), C_B = Float64(C_B),
                   λ = Float64(λ), θ_factor = Float64(θ_factor),
                   ke_budget_fraction = Float64(ke_budget_fraction),
                   ell_corr = Float64(ell_corr),
                   snapshots_at = Tuple(snap_times),
                   species_names = ic.params.species_names),
    )
end

"""
    run_D10_ism_multi_tracer_sweep(; levels=(4, 5), kwargs...) -> NamedTuple

Run `run_D10_ism_multi_tracer` at multiple refinement levels.
"""
function run_D10_ism_multi_tracer_sweep(; levels = (4, 5),
                                          T_factor::Real = 0.1,
                                          kwargs...)
    results = NamedTuple[]
    for L in levels
        push!(results, run_D10_ism_multi_tracer(; level = L,
                                                T_factor = T_factor,
                                                kwargs...))
    end
    t_KH = isempty(results) ? NaN : results[1].t_KH
    bytes_equal = [r.tracers_byte_equal_to_ic for r in results]
    return (
        levels = Tuple(levels),
        results = results,
        t_KH = t_KH,
        tracers_byte_equal_to_ic_per_level = bytes_equal,
    )
end

"""
    save_D10_ism_multi_tracer_to_h5(sweep, save_path)

Write the mesh-sweep result to HDF5.
"""
function save_D10_ism_multi_tracer_to_h5(sweep, save_path::AbstractString)
    HDF5 = if isdefined(Main, :HDF5)
        getfield(Main, :HDF5)
    else
        Base.require(Main, :HDF5)
        getfield(Main, :HDF5)
    end
    mkpath(dirname(save_path))
    HDF5.h5open(save_path, "w") do f
        f["levels"] = collect(sweep.levels)
        f["t_KH"] = sweep.t_KH
        f["tracers_byte_equal_to_ic_per_level"] =
            Int.(sweep.tracers_byte_equal_to_ic_per_level)
        for (i, L) in enumerate(sweep.levels)
            r = sweep.results[i]
            grp = HDF5.create_group(f, "level_$(L)")
            grp["t"] = r.t
            grp["M_per_species_traj"] = r.M_per_species_traj
            grp["M_traj"] = r.M_traj
            grp["Px_traj"] = r.Px_traj
            grp["Py_traj"] = r.Py_traj
            grp["KE_traj"] = r.KE_traj
            grp["n_negative_jacobian"] = r.n_negative_jacobian
            grp["gamma_mean_axis1"] = r.gamma_mean_axis1
            grp["gamma_max_axis1"] = r.gamma_max_axis1
            grp["max_pair_separation"] = r.max_pair_separation
            grp["tracers_max_diff_traj"] = r.tracers_max_diff_traj
            grp["tracers_max_diff_final"] = r.tracers_max_diff_final
            grp["tracers_byte_equal_to_ic"] =
                Int(r.tracers_byte_equal_to_ic)
            grp["wall_time_per_step"] = r.wall_time_per_step
        end
    end
    return save_path
end

"""
    plot_D10_ism_multi_tracer(sweep; save_path) -> save_path

4-panel CairoMakie headline figure:
  • Panel A: gas |u| at end-time (shocked turbulence map)
  • Panel B: 3 species concentrations at end-time (overlaid)
  • Panel C: per-species mass conservation `M_k(t) - M_k(0)` over time
  • Panel D: per-species γ trajectories
"""
function plot_D10_ism_multi_tracer(sweep; save_path::AbstractString)
    try
        CM = if isdefined(Main, :CairoMakie)
            getfield(Main, :CairoMakie)
        else
            Base.require(Main, :CairoMakie)
            getfield(Main, :CairoMakie)
        end
        i_top = lastindex(sweep.results)
        r_top = sweep.results[i_top]
        L_top = sweep.levels[i_top]
        snap_late = r_top.snapshots[end]
        ns = size(r_top.gamma_mean_axis1, 1)
        species_names = r_top.params.species_names

        xs_unique = sort!(unique(round.(snap_late.x_centres; digits = 12)))
        ys_unique = sort!(unique(round.(snap_late.y_centres; digits = 12)))
        nx = length(xs_unique); ny = length(ys_unique)

        function reshape_to_grid(arr)
            G = fill(NaN, ny, nx)
            for k in eachindex(snap_late.x_centres)
                xi = searchsortedfirst(xs_unique,
                                        round(snap_late.x_centres[k]; digits = 12))
                yi = searchsortedfirst(ys_unique,
                                        round(snap_late.y_centres[k]; digits = 12))
                if 1 ≤ xi ≤ nx && 1 ≤ yi ≤ ny
                    G[yi, xi] = arr[k]
                end
            end
            return G
        end

        u_mag = sqrt.(snap_late.u_1.^2 .+ snap_late.u_2.^2)
        U_grid = reshape_to_grid(u_mag)

        fig = CM.Figure(size = (1200, 900))
        axA = CM.Axis(fig[1, 1];
            title = "A: |u| at t=$(round(snap_late.t; sigdigits=3)) (L=$L_top, shocked turbulence)",
            xlabel = "x_1", ylabel = "x_2")
        CM.heatmap!(axA, xs_unique, ys_unique, transpose(U_grid))

        axB = CM.Axis(fig[1, 2];
            title = "B: $(ns)-species concentration at t=$(round(snap_late.t; sigdigits=3))",
            xlabel = "x_2", ylabel = "concentration")
        for k in 1:ns
            cy_idx = sortperm(snap_late.y_centres)
            CM.lines!(axB, snap_late.y_centres[cy_idx],
                       snap_late.c_per_species[k][cy_idx];
                       label = string(species_names[k]))
        end
        CM.axislegend(axB; position = :rt)

        axC = CM.Axis(fig[2, 1];
            title = "C: per-species mass conservation",
            xlabel = "t", ylabel = "M_k(t) − M_k(0)")
        for (i, L) in enumerate(sweep.levels)
            r = sweep.results[i]
            for k in 1:ns
                CM.lines!(axC, r.t,
                           r.M_per_species_traj[k, :] .-
                           r.M_per_species_traj[k, 1];
                           label = "L=$L, $(species_names[k])")
            end
        end
        CM.axislegend(axC; position = :rt)

        axD = CM.Axis(fig[2, 2];
            title = "D: per-species γ trajectory (axis 1)",
            xlabel = "t", ylabel = "γ mean")
        for k in 1:ns
            CM.lines!(axD, r_top.t, r_top.gamma_mean_axis1[k, :];
                       label = string(species_names[k]))
        end
        CM.axislegend(axD; position = :rt)

        mkpath(dirname(save_path))
        CM.save(save_path, fig)
        return save_path
    catch e
        @warn "Plotting D.10 ISM-tracers figure failed: $(e). Saving CSV instead."
        mkpath(dirname(save_path))
        for (i, L) in enumerate(sweep.levels)
            r = sweep.results[i]
            csv = replace(save_path, ".png" => "_L$(L).csv")
            ns_local = size(r.gamma_mean_axis1, 1)
            open(csv, "w") do f
                # CSV header.
                hdr = "t,M_err"
                for k in 1:ns_local
                    hdr *= ",M_$(k)_err,gamma_mean_$(k)"
                end
                hdr *= ",max_pair_separation,tracers_max_diff,n_neg_jac"
                println(f, hdr)
                for j in eachindex(r.t)
                    line = "$(r.t[j]),$(r.M_traj[j]-r.M_traj[1])"
                    for k in 1:ns_local
                        line *= ",$(r.M_per_species_traj[k, j]-r.M_per_species_traj[k, 1])," *
                                "$(r.gamma_mean_axis1[k, j])"
                    end
                    line *= ",$(r.max_pair_separation[j])," *
                            "$(r.tracers_max_diff_traj[j])," *
                            "$(r.n_negative_jacobian[j])"
                    println(f, line)
                end
            end
        end
        return save_path
    end
end
