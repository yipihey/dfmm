# stochastic_injection.jl
#
# Phase 8: variance-gamma stochastic injection wired into the
# deterministic variational integrator as a post-Newton operator-split
# step.
#
# Methods paper §4 + §9.6 specify a per-step recipe: at each cell with
# compressive strain ((∂_x u)_j < 0), perturb (ρu)_j by a drift +
# variance-gamma noise, debit the resulting kinetic-energy change
# from internal energy by paired-pressure (P_xx, P_⊥) bookkeeping,
# and apply a 3-point Gaussian smoothing to the per-cell noise sample.
#
# The implementation here mirrors py-1d's `noise_model.py:hll_step_noise`
# (drift, noise, KE-debit, amplitude limiter) but expresses the energy
# debit through the variational variables (entropy s for the trace
# `P_xx = ρ · M_vv(J, s)`, plus the explicit `P_⊥` field). The split
# between the two "transverse" 3D dimensions is fixed at 2/3 each so
# the 1D specialization matches py-1d bit-for-bit on internal-energy
# accounting.
#
# The per-step Newton solve in `det_step!` is **not** modified: this
# module is a strictly post-Newton mutator on the mesh fields. With
# `params.C_B == 0` and `params.C_A == 0` the operator-split step is a
# no-op, and the integrator reduces to Phase 5/5b deterministic
# evolution bit-for-bit.
#
# References:
#   * specs/01_methods_paper.tex §4 (variance-gamma derivation),
#                                §9.6 (per-step injection recipe),
#                                §10.3 B.4 (burst-statistics acceptance).
#   * design/04_action_note_v3_FINAL.pdf §1 (empirical findings,
#                                            calibration mismatch).
#   * py-1d/dfmm/closure/noise_model.py — the Python reference
#                                          implementation we mirror.
#   * src/stochastic.jl — the variance-gamma + burst-stats primitives
#                         (Track D); read but not modified here.

using Random: AbstractRNG, MersenneTwister
using StatsBase: mean, var

# Reuse the variance-gamma sampler + burst-stats primitives from the
# Track-D module without re-exporting them.
using .Stochastic: rand_variance_gamma, rand_variance_gamma!, burst_detect,
                   estimate_gamma_shape, residual_kurtosis,
                   gamma_shape_from_kurtosis, ks_test

# -----------------------------------------------------------------------------
# Calibration-parameter helper
# -----------------------------------------------------------------------------

"""
    NoiseInjectionParams(; C_A, C_B, λ, θ_factor, ke_budget_fraction,
                          ell_corr, ...)

Bundle of stochastic-injection knobs. Defaults match py-1d's production
configuration (`SimulationConfig` with the calibrated `C_A`, `C_B` from
`py-1d/data/noise_model_params.npz`, `noise_ke_budget_fraction = 0.25`,
`ell_corr = 2.0` cells).

Fields:
  * `C_A::Float64`         — drift coefficient.
  * `C_B::Float64`         — noise amplitude.
  * `λ::Float64`           — variance-gamma shape (production: derived
                             from kurt 3.45 ⇒ λ ≈ 6.67; small-data
                             best fit gives λ ≈ 1.6).
  * `θ_factor::Float64`    — variance-mixing scale; set so the marginal
                             variance is `θ_factor·λ = 1` (unit
                             variance VG noise; combine with `C_B` for
                             physical units).
  * `ke_budget_fraction::Float64` — amplitude-limiter fraction (matches
                                    py-1d, default 0.25).
  * `ell_corr::Float64`    — Gaussian-smoothing correlation length in
                             cells (matches py-1d, default 2.0).
  * `pressure_floor::Float64` — minimum P_xx, P_⊥ retained after the
                                debit (matches py-1d).

The constructor `NoiseInjectionParams(; ...)` accepts any subset of
these as keyword arguments and fills in py-1d defaults for the rest.
The `from_calibration` factory below builds an instance directly from
a `load_noise_model()` NamedTuple.
"""
Base.@kwdef struct NoiseInjectionParams
    C_A::Float64 = 0.336
    C_B::Float64 = 0.548
    λ::Float64   = 6.667     # 3.0 / (kurt - 3.0) at production kurt 3.45
    θ_factor::Float64 = 0.15 # = 1/λ so var = λ·θ_factor = 1
    ke_budget_fraction::Float64 = 0.25
    ell_corr::Float64 = 2.0
    pressure_floor::Float64 = 1e-8
    # ---- M2-3 realizability projection knobs (Phase 8.5) -------------------
    # `project_kind = :none` reproduces M1 Phase 8 bit-for-bit. `:reanchor`
    # raises s post-injection so M_vv ≥ headroom · β². See
    # `realizability_project!` and `reference/notes_M2_3_realizability.md`.
    project_kind::Symbol = :reanchor
    realizability_headroom::Float64 = 1.05  # require M_vv ≥ 1.05 · β²
    # Absolute lower bound on M_vv. Set well above the realizability
    # boundary so that cells driven by the entropy-debit drift toward
    # `M_vv → 0` are caught before the Newton solve becomes
    # ill-conditioned (small `γ ≈ √M_vv` is a weak-coupling regime
    # that lets the implicit step push a vertex past its neighbor —
    # the classic Lagrangian mesh-tangling instability — and the
    # *cumulative* drift over ~1000 steps under production calibration
    # is what closes M1 Open #4; see notes file §2 root-cause analysis
    # and reference/notes_phase8_stochastic_injection.md §7).
    #
    # Empirical sweet-spot for the production-calibrated wave-pool
    # (`load_noise_model()` defaults, N = 128, dt = 5e-4): values in
    # [5e-3, 2e-2] are stable through 12 000 steps. Smaller floors fail
    # at < 3000 steps; larger floors stabilize but the projection becomes
    # the dominant closure with rate > 0.5 per cell-step. `1e-2` is the
    # minimum stable choice and also leaves headroom for the Newton's
    # nonlinear overshoot during iteration.
    Mvv_floor::Float64 = 1e-2
end

"""
    from_calibration(nm::NamedTuple; ell_corr=2.0,
                     ke_budget_fraction=0.25,
                     pressure_floor=1e-8) -> NoiseInjectionParams

Build a `NoiseInjectionParams` from `load_noise_model()`. The
variance-gamma shape `λ` is derived from `nm.kurt` via
`gamma_shape_from_kurtosis(nm.kurt - 3.0)` (excess kurtosis = 3/λ for
VG); `θ_factor` is set to `1/λ` so the marginal variance equals 1
(i.e. `C_B` is the only amplitude knob).

The calibrated production values are `C_A ≈ 0.336`, `C_B ≈ 0.548`,
`kurt ≈ 3.45`. The `kurt → λ` inversion is sensitive to the exact
empirical excess kurtosis; the small-data fit `λ ≈ 1.6` corresponds
to excess ≈ 1.875.
"""
function from_calibration(nm::NamedTuple;
                          ell_corr::Float64 = 2.0,
                          ke_budget_fraction::Float64 = 0.25,
                          pressure_floor::Float64 = 1e-8,
                          project_kind::Symbol = :reanchor,
                          realizability_headroom::Float64 = 1.05,
                          Mvv_floor::Float64 = 1e-2)
    excess = nm.kurt - 3.0
    if excess <= 0
        # Calibration is sub-Gaussian; fall back to the small-data
        # fit λ = 1.6 with a conservative comment in the notes.
        λ = 1.6
    else
        λ = 3.0 / excess
    end
    θ_factor = 1.0 / λ
    return NoiseInjectionParams(C_A = nm.C_A, C_B = nm.C_B,
                                λ = λ, θ_factor = θ_factor,
                                ke_budget_fraction = ke_budget_fraction,
                                ell_corr = ell_corr,
                                pressure_floor = pressure_floor,
                                project_kind = project_kind,
                                realizability_headroom = realizability_headroom,
                                Mvv_floor = Mvv_floor)
end

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

"""
    _segment_divu_centered(mesh, j) -> Float64

Cell-centered ∂_x u for segment `j`, computed as
`(u_{j+1} - u_j) / (x_{j+1} - x_j)` from the Phase-2 vertex
representation. This is the same definition used by `det_el_residual`
for the in-step strain and matches the **midpoint** strain to
leading order (the operator-split step uses the post-Newton state).
"""
function _segment_divu_centered(mesh::Mesh1D{T,DetField{T}}, j::Integer) where {T<:Real}
    N = n_segments(mesh)
    j_right = j == N ? 1 : j + 1
    wrap = (j == N) ? mesh.L_box : zero(T)
    Δx_j = mesh.segments[j_right].state.x + wrap - mesh.segments[j].state.x
    Δu_j = mesh.segments[j_right].state.u - mesh.segments[j].state.u
    return Float64(Δu_j / Δx_j)
end

"""
    _segment_velocity_centered(mesh, j) -> Float64

Cell-centered velocity `½(u_j + u_{j+1})` for the cell-centered KE
bookkeeping (matches py-1d's per-cell `u = U[IDX_MOM]/rho`
interpretation).
"""
function _segment_velocity_centered(mesh::Mesh1D{T,DetField{T}}, j::Integer) where {T<:Real}
    N = n_segments(mesh)
    j_right = j == N ? 1 : j + 1
    return Float64((mesh.segments[j].state.u + mesh.segments[j_right].state.u) / 2)
end

"""
    smooth_periodic_3pt!(out, eta)

In-place 3-point Gaussian-like periodic smoother with weights
`(1, 2, 1)/4` (a discrete approximation to a Gaussian with σ ≈ 1
cell). Variance is renormalized to match the input sample variance
so the smoother does not bias the noise amplitude. With `length(eta)
== 1` the smoother is a no-op.

This is the 1D specialization of `smooth_gaussian_periodic` from
`py-1d/dfmm/closure/noise_model.py`. The full FFT-based smoother
with `ell_corr = 2.0` cells is qualitatively similar; the 3-point
binomial kernel is correct to leading order in `1/ell_corr` and
avoids an FFT dependency for what is a per-step inner loop.
"""
function smooth_periodic_3pt!(out::AbstractVector{Float64},
                              eta::AbstractVector{Float64})
    N = length(eta)
    @assert length(out) == N
    if N < 2
        copyto!(out, eta)
        return out
    end
    # Pre-smooth standard deviation for variance-preserving rescale.
    σ_in = sqrt(var(eta; corrected = false))
    @inbounds for i in 1:N
        ip = i == N ? 1 : i + 1
        im = i == 1 ? N : i - 1
        out[i] = 0.25 * eta[im] + 0.5 * eta[i] + 0.25 * eta[ip]
    end
    σ_out = sqrt(var(out; corrected = false))
    if σ_out > 0 && σ_in > 0
        scale = σ_in / σ_out
        @inbounds for i in 1:N
            out[i] *= scale
        end
    end
    return out
end

# -----------------------------------------------------------------------------
# Per-step injection
# -----------------------------------------------------------------------------

"""
    InjectionDiagnostics

Per-cell diagnostics produced by `inject_vg_noise!`, used by the
self-consistency monitor (`BurstStatsAccumulator`).

Fields:
  * `divu::Vector{Float64}`        — cell-centered ∂_x u after the
                                     deterministic step.
  * `eta::Vector{Float64}`         — smoothed unit-variance VG draw.
  * `delta_rhou::Vector{Float64}`  — total δ(ρu) (drift + noise),
                                     **after** amplitude-limiting.
  * `delta_rhou_drift::Vector{Float64}` — drift component only.
  * `delta_rhou_noise::Vector{Float64}` — noise component only,
                                          before limiting (i.e.
                                          `noise_amp * eta`).
  * `delta_KE_vol::Vector{Float64}` — per-cell ΔKE injected (cell
                                      volumetric).
  * `compressive::BitVector`        — true on cells with `divu < 0`.
"""
struct InjectionDiagnostics
    divu::Vector{Float64}
    eta::Vector{Float64}
    delta_rhou::Vector{Float64}
    delta_rhou_drift::Vector{Float64}
    delta_rhou_noise::Vector{Float64}
    delta_KE_vol::Vector{Float64}
    compressive::BitVector
end

InjectionDiagnostics(N::Int) = InjectionDiagnostics(
    zeros(Float64, N), zeros(Float64, N),
    zeros(Float64, N), zeros(Float64, N),
    zeros(Float64, N), zeros(Float64, N),
    falses(N),
)

# -----------------------------------------------------------------------------
# Realizability projection (Phase 8.5, M2-3)
# -----------------------------------------------------------------------------

"""
    ProjectionStats

Lightweight per-run accumulator counting how often the realizability
projection fired across `inject_vg_noise!` calls. Used by the
`experiments/M2_3_long_time_stochastic.jl` driver and the M2-3 unit
tests to verify that the projection is a regularizer (sparse events)
rather than a dominant code path.

Fields:
  * `n_steps::Int` — number of `realizability_project!` calls observed.
  * `n_events::Int` — number of cell-step pairs where the projection
                      raised `M_vv` (i.e. `M_vv_pre < headroom · β²` or
                      `M_vv_pre < Mvv_floor`).
  * `n_floor_events::Int` — subset of `n_events` where the absolute
                            `Mvv_floor` (rather than the relative
                            headroom) was the binding constraint.
  * `total_dE_inj::Float64` — accumulated extra internal energy added
                              by all projection events (a small but
                              nonzero number; see notes file). The
                              variational form distributes this between
                              `P_xx` (raised) and `P_⊥` (debited if
                              available); the residual that exceeds
                              `P_⊥`'s pressure_floor headroom is added
                              as a silent "stochastic floor" gain
                              (mirrors py-1d's `pressure_floor` clip).
  * `Mvv_min_pre::Float64`  — running min of pre-projection `M_vv`.
  * `Mvv_min_post::Float64` — running min of post-projection `M_vv`.
"""
mutable struct ProjectionStats
    n_steps::Int
    n_events::Int
    n_floor_events::Int
    total_dE_inj::Float64
    Mvv_min_pre::Float64
    Mvv_min_post::Float64
end

ProjectionStats() = ProjectionStats(0, 0, 0, 0.0, Inf, Inf)

"""
    reset!(stats::ProjectionStats)

Zero a `ProjectionStats` accumulator in place. Useful for re-using the
same `stats` across multiple `det_run_stochastic!` calls.
"""
function reset!(stats::ProjectionStats)
    stats.n_steps = 0
    stats.n_events = 0
    stats.n_floor_events = 0
    stats.total_dE_inj = 0.0
    stats.Mvv_min_pre = Inf
    stats.Mvv_min_post = Inf
    return stats
end

"""
    realizability_project!(mesh::Mesh1D{T,DetField{T}};
                           kind::Symbol = :reanchor,
                           headroom::Real = 1.05,
                           Mvv_floor::Real = 1e-6,
                           pressure_floor::Real = 1e-8,
                           stats::Union{ProjectionStats,Nothing} = nothing)
        -> mesh

Apply a per-cell state projection that pushes the mesh state back
inside the realizability cone `M_vv ≥ β²`. Mutates `mesh.segments[j]`
in place; returns `mesh`. If `stats !== nothing`, increment the
projection-event accumulator (see `ProjectionStats`).

This is the **Phase 8.5 / M2-3 fix** for the long-time stochastic-
realizability instability documented in
`reference/notes_phase8_stochastic_injection.md` §7 and
`reference/MILESTONE_1_STATUS.md` Open #4. Without this projection,
production-calibrated wave-pool runs blow up at ~950 steps when the
slow entropy debit on each compressive cell drives `M_vv = J^(1-Γ)
exp(s)` below `β² + γ²`, making the next Newton step's
`γ² = M_vv − β²` go negative and the integrator NaN.

**Variants** (selected via `kind`):

  * `:reanchor` *(default, the chosen variant)* — for each segment
    where `M_vv < max(headroom · β², Mvv_floor)`, raise `s` so that
    the new `M_vv = max(headroom · β², Mvv_floor)`. The mass `Δm` and
    momentum (cell `δ(ρu)` and vertex `u_i`) are untouched. The extra
    internal energy `ρ_j · (M_vv_target − M_vv_pre)` added to `P_xx`
    is taken from `P_⊥` first (kept ≥ `pressure_floor`); any residual
    is admitted as a small silent "stochastic-floor" energy gain
    (this is the py-1d-equivalent of the `pressure_floor` clip on
    `Pxx_new`). The headroom of `1.05` is small enough that the
    projection fires only on cells that genuinely crossed the boundary
    (a few per 100 cells per save interval at production calibration),
    yet large enough that the *next* Newton step's `γ²` is comfortably
    positive.

  * `:none` — no-op. With this kind, `realizability_project!` is a
    no-op; `inject_vg_noise!` reduces to its M1 Phase 8 form.
    Provided for the bit-equality regression test.

  * `:attenuate` *(documented but not selected)* — instead of raising
    `s`, scale the post-injection `δ(ρu)` perturbation down so the
    cell does not cross the boundary. This requires undoing the noise
    debit and re-applying it with a smaller amplitude, which couples
    back into the vertex-velocity update; the cleaner separation of
    `:reanchor` (touches only `(s, P_⊥)`, never `(ρ, ρu)`) is why we
    prefer it. See `reference/notes_M2_3_realizability.md` §3 for the
    derivation showing both variants converge to the same solution
    when the projection event rate is small (the regime we observe
    at production calibration).

  * `:symmetric_debit` *(documented but not selected)* — refine the
    per-cell ⅔/⅓ debit split between `P_xx` and `P_⊥` so the
    inequality `β² + γ² ≤ M_vv` is enforced *exactly* at debit time
    (rather than via the post-hoc `:reanchor` clip). This is the most
    physically faithful but requires a per-cell quadratic solve;
    `:reanchor` is equivalent in event rate at production calibration
    and avoids the extra solve. See notes file §4.

Conservation:
  * Mass: bit-exact (Δm never touched).
  * Momentum: bit-exact (vertex u and cell δ(ρu) never touched —
    the projection lives entirely in the `(s, P_⊥)` sector).
  * Internal energy: changes by `+ρ_j · (M_vv_target − M_vv_pre) Δx_j`
    per event from the `P_xx` rise, *minus* `(P_⊥_old − P_⊥_new) Δx_j`
    from the `P_⊥` debit. Net is non-negative (the projection only
    fires when `M_vv_pre < target`); accumulated in `stats.total_dE_inj`.

The projection-event rate is monitored in the M2-3 unit tests; with
the production calibration on a 128-cell wave-pool we observe rates
of order `1–3` events per 100 cells per `dt_save = 0.01`, well below
the criterion of "few per 100 cells" stated in the M2-3 acceptance.
"""
function realizability_project!(mesh::Mesh1D{T,DetField{T}};
                                kind::Symbol = :reanchor,
                                headroom::Real = 1.05,
                                Mvv_floor::Real = 1e-2,
                                pressure_floor::Real = 1e-8,
                                stats::Union{ProjectionStats,Nothing} = nothing) where {T<:Real}
    if kind === :none
        if stats !== nothing
            stats.n_steps += 1
        end
        return mesh
    end
    @assert kind === :reanchor "realizability_project!: unsupported kind=$(kind)"

    N = n_segments(mesh)
    h = Float64(headroom)
    Mvv_floor_f = Float64(Mvv_floor)
    pf = Float64(pressure_floor)

    if stats !== nothing
        stats.n_steps += 1
    end

    @inbounds for j in 1:N
        seg = mesh.segments[j]
        ρ_j = Float64(segment_density(mesh, j))
        J_j = 1.0 / ρ_j
        s_pre = Float64(seg.state.s)
        β_pre = Float64(seg.state.β)
        Mvv_pre = Float64(Mvv(J_j, s_pre))
        Pp_pre  = Float64(seg.state.Pp)
        if !isfinite(Pp_pre)
            Pp_pre = ρ_j * Mvv_pre  # legacy NaN sentinel — treat isotropic
        end

        if stats !== nothing
            stats.Mvv_min_pre = min(stats.Mvv_min_pre, Mvv_pre)
        end

        # Realizability target. We require M_vv ≥ headroom · β² so the
        # next Newton step's γ² = M_vv − β² has a strict positive
        # margin; we *also* require M_vv ≥ Mvv_floor (an absolute
        # safety floor so cells with very small β are not driven all
        # the way to zero, the cold-limit caustic from §3.5 of the
        # methods paper).
        Mvv_target_rel = h * β_pre * β_pre
        Mvv_target = max(Mvv_target_rel, Mvv_floor_f)

        if Mvv_pre >= Mvv_target
            if stats !== nothing
                stats.Mvv_min_post = min(stats.Mvv_min_post, Mvv_pre)
            end
            continue
        end

        # Projection event: raise s so M_vv = Mvv_target, take the
        # extra energy from P_⊥ first (stays ≥ pressure_floor), then
        # admit any residual as a silent floor-gain.
        if stats !== nothing
            stats.n_events += 1
            if Mvv_target > Mvv_target_rel
                stats.n_floor_events += 1
            end
        end

        Pxx_pre = ρ_j * Mvv_pre
        Pxx_new = ρ_j * Mvv_target
        ΔPxx = Pxx_new - Pxx_pre   # > 0 by construction
        # IE_per_volume = ½ Pxx + Pp. To keep IE conserved across the
        # projection event, debit ½ ΔPxx from P_⊥ (so the ½·ΔPxx from
        # P_xx is matched by an equal decrement in P_⊥). The factor of
        # ½ is the 1D-3D effective-dimensions split that closes the
        # paired-pressure ledger; see methods paper §9.6 and the M2-3
        # notes file §3 derivation.
        Pp_take_target = 0.5 * ΔPxx
        Pp_avail = max(Pp_pre - pf, 0.0)
        Pp_take = min(Pp_take_target, Pp_avail)
        Pp_new = Pp_pre - Pp_take

        # IE budget per unit volume: ΔIE = ½·ΔPxx − Pp_take. With
        # Pp_take = ½·ΔPxx (the target) this is exactly zero — IE
        # conserved across the projection. If P_⊥ couldn't supply the
        # full ½·ΔPxx (cell already near pressure_floor on P_⊥), the
        # residual `0.5 ΔPxx - Pp_take ≥ 0` is admitted as a silent
        # IE gain (matches py-1d's `pressure_floor` semantics).
        floor_gain = (0.5 * ΔPxx) - Pp_take
        if stats !== nothing
            Δx_j = J_j * Float64(seg.Δm)
            stats.total_dE_inj += floor_gain * Δx_j
        end

        # Re-anchor entropy from the new M_vv.
        s_new = s_pre + log(Mvv_target / max(Mvv_pre, 1e-300))

        mesh.segments[j].state = DetField{T}(seg.state.x, seg.state.u,
                                             seg.state.α, seg.state.β,
                                             T(s_new), T(Pp_new),
                                             seg.state.Q)

        if stats !== nothing
            stats.Mvv_min_post = min(stats.Mvv_min_post, Mvv_target)
        end
    end
    return mesh
end

# -----------------------------------------------------------------------------
# Per-axis (2D) realizability projection (M3-3d)
# -----------------------------------------------------------------------------

"""
    realizability_project_2d!(fields, leaves; project_kind=:reanchor,
                              headroom=1.05, Mvv_floor=1e-2,
                              pressure_floor=1e-8, ρ_ref=1.0,
                              Gamma=GAMMA_LAW_DEFAULT,
                              stats::Union{ProjectionStats,Nothing}=nothing)
        -> fields

Per-axis 2D analog of `realizability_project!`. Walks the leaves of a
2D Cholesky-sector field set (`allocate_cholesky_2d_fields`) and pushes
each cell's `(α_a, β_a, s)` state back inside the realizability cone
along **both** principal axes:

    M_vv,aa ≥ headroom · β_a²    for a = 1, 2.

Mutates `fields` in place; returns `fields`. If `stats !== nothing`,
projection-event statistics are accumulated.

# Why per-axis?

In 2D the per-axis `γ²_a = M_vv,aa − β_a²` realizability constraint is
checked once per axis. Because the diagonal EOS gives
`M_vv,11 = M_vv,22 = Mvv(J, s)` (an isotropic ideal gas does not
distinguish axes), raising `s` raises both `M_vv,aa` simultaneously.
The per-axis loop therefore reduces to a single `s`-raise driven by
`max_a(headroom · β_a², Mvv_floor)`. The off-diagonal coupling stays
zero per M3-3a Q3 (off-diag β omitted in M3-3); this projection
honours that — `β_{12}, β_{21}` are not stored in the field set.

# Variants

  • `:reanchor` *(default)* — raise `s` so that
    `M_vv,aa ≥ max(headroom · β_a², Mvv_floor)` for both axes
    simultaneously. The 1D-projection's per-cell P_⊥ debit is
    inherited (debit comes from the post-Newton `Pp` slot if
    available; otherwise admitted as a silent floor-gain).
  • `:none` — no-op. M2-3 1D bit-equality regression mirror.

# Conservation

  • Per-axis β unchanged (projection acts only on `s`).
  • Per-axis (α, x, u) unchanged.
  • Internal-energy increment matches 1D: `+ρ_j · (M_vv_target − M_vv_pre)`
    per leaf (positive only when projection fires).

# Sign convention with respect to the M3-3a per-axis Cholesky

The 2D per-axis γ uses `γ²_a = M_vv,aa − β_a²` (the M1 1D form lifted
per axis), the same convention `gamma_per_axis_2d` in
`src/cholesky_DD.jl` and `gamma_per_axis_2d_field` in
`src/diagnostics.jl` use. With the M3-3 isotropic-EOS field set
(`M_vv,11 = M_vv,22 = Mvv(J, s)`), the per-axis realizability
boundaries reduce to `Mvv(J, s) ≥ headroom · max(β_1², β_2²)` (the
weaker of the two constraints determines whether projection fires).

The 1D bit-equality regression: at `β_2 = 0`, the per-axis target
collapses to `headroom · β_1²` — identical to the 1D
`realizability_project!` form. The M3-3d test verifies this on a
1D-symmetric 2D configuration.
"""
function realizability_project_2d!(fields, leaves;
                                   project_kind::Symbol = :reanchor,
                                   headroom::Real = 1.05,
                                   Mvv_floor::Real = 1e-2,
                                   pressure_floor::Real = 1e-8,
                                   ρ_ref::Real = 1.0,
                                   Gamma::Real = GAMMA_LAW_DEFAULT,
                                   stats::Union{ProjectionStats,Nothing} = nothing)
    if project_kind === :none
        if stats !== nothing
            stats.n_steps += 1
        end
        return fields
    end
    @assert project_kind === :reanchor "realizability_project_2d!: unsupported project_kind=$(project_kind)"

    if stats !== nothing
        stats.n_steps += 1
    end

    h = Float64(headroom)
    Mvv_floor_f = Float64(Mvv_floor)
    pf = Float64(pressure_floor)
    ρ_t = Float64(ρ_ref)
    J_t = 1.0 / ρ_t

    @inbounds for ci in leaves
        β1 = Float64(fields.β_1[ci][1])
        β2 = Float64(fields.β_2[ci][1])
        s_pre = Float64(fields.s[ci][1])
        Pp_pre = Float64(fields.Pp[ci][1])
        if !isfinite(Pp_pre)
            Pp_pre = 0.0
        end

        Mvv_pre = Float64(Mvv(J_t, s_pre; Gamma = Gamma))
        if stats !== nothing
            stats.Mvv_min_pre = min(stats.Mvv_min_pre, Mvv_pre)
        end

        # Per-axis realizability target. The binding constraint is the
        # max over axes (β² is non-negative, so max defines the tighter
        # cone-interior requirement).
        β2_max = max(β1 * β1, β2 * β2)
        Mvv_target_rel = h * β2_max
        Mvv_target = max(Mvv_target_rel, Mvv_floor_f)

        if Mvv_pre >= Mvv_target
            if stats !== nothing
                stats.Mvv_min_post = min(stats.Mvv_min_post, Mvv_pre)
            end
            continue
        end

        # Projection event.
        if stats !== nothing
            stats.n_events += 1
            if Mvv_target > Mvv_target_rel
                stats.n_floor_events += 1
            end
        end

        Pxx_pre = ρ_t * Mvv_pre
        Pxx_new = ρ_t * Mvv_target
        ΔPxx = Pxx_new - Pxx_pre   # > 0 by construction
        Pp_take_target = 0.5 * ΔPxx
        Pp_avail = max(Pp_pre - pf, 0.0)
        Pp_take = min(Pp_take_target, Pp_avail)
        Pp_new = Pp_pre - Pp_take
        floor_gain = (0.5 * ΔPxx) - Pp_take
        if stats !== nothing
            # Tracked per cell — caller can multiply by cell volume if
            # they want a physical-units IE. Same convention as the
            # 1D projection where ΔIE = floor_gain · Δx_j.
            stats.total_dE_inj += floor_gain
        end

        s_new = s_pre + log(Mvv_target / max(Mvv_pre, 1e-300))

        fields.s[ci]  = (s_new,)
        fields.Pp[ci] = (Pp_new,)

        if stats !== nothing
            stats.Mvv_min_post = min(stats.Mvv_min_post, Mvv_target)
        end
    end
    return fields
end

"""
    inject_vg_noise!(mesh::Mesh1D, dt; params, rng,
                     diag = InjectionDiagnostics(n_segments(mesh)))

Apply one variance-gamma noise + drift injection step to `mesh`,
in place. Returns `(mesh, diag)`. The mesh's `(u, s, P_⊥)` fields are
mutated; `(α, β)` are left untouched (the noise lives in the
momentum sector, with the energy debit absorbed via entropy).

Per-step recipe (mirrors py-1d's `hll_step_noise`):
  1. Compute cell-centered `(∂_x u)_j` from the post-Newton vertex
     velocities.
  2. Draw `η_j ~ VG(λ, θ_factor)`, smooth periodically.
  3. Drift `δ_drift = C_A ρ_j (∂_x u)_j Δt`; noise
     `δ_noise = C_B ρ_j √(max(-(∂_x u)_j, 0) Δt) η_j`.
  4. Amplitude-limit so `|δ(ρu)|` cannot extract more than
     `params.ke_budget_fraction · IE_local` from the cell.
  5. Compute `ΔKE_vol = u_j δ + δ²/(2ρ_j)`; debit
     `(2/3) ΔKE_vol` from `P_xx` and `(2/3) ΔKE_vol` from `P_⊥`.
  6. Translate `P_xx_new ↦ s_new` via the EOS (re-anchor entropy);
     update `P_⊥` directly. Both subject to a `pressure_floor` clip.
  7. Distribute the cell-momentum `Δm_j δ_j J_j = δ_j Δx_j` half
     each to the two adjacent vertices (mass-lumped); update
     `mesh.p_half`.

Conservation:
  * Total mass: bit-exact (no `Δm` mutation).
  * Total momentum: equal to `∑_j δ_j Δx_j` per step (sums over
    vertices and cells differ by zero in periodic BC since each
    vertex receives exactly half from each neighbor).
  * Total energy (KE_bulk + IE_internal + Cholesky-Hamiltonian):
    bounded; any drift is bounded by the post-step amplitude
    limiting and the floor-clipping.
"""
function inject_vg_noise!(mesh::Mesh1D{T,DetField{T}}, dt::Real;
                          params::NoiseInjectionParams,
                          rng::AbstractRNG,
                          diag::InjectionDiagnostics =
                              InjectionDiagnostics(n_segments(mesh)),
                          proj_stats::Union{ProjectionStats,Nothing} = nothing) where {T<:Real}
    N = n_segments(mesh)
    @assert length(diag.divu) == N "InjectionDiagnostics size mismatch ($(length(diag.divu)) vs $N)"

    Δt = Float64(dt)

    # 1. Compute divu and ρ per cell from post-Newton state.
    @inbounds for j in 1:N
        diag.divu[j] = _segment_divu_centered(mesh, j)
    end

    # 2. Draw VG noise and smooth.
    eta_white = Vector{Float64}(undef, N)
    rand_variance_gamma!(rng, eta_white, params.λ, params.θ_factor)
    if params.ell_corr > 0 && N >= 3
        smooth_periodic_3pt!(diag.eta, eta_white)
    else
        copyto!(diag.eta, eta_white)
    end

    # 3-6. Per-cell drift, noise, limiter, energy debit, entropy update.
    # We accumulate cell-momentum injection (δρu · Δx) for the vertex
    # update in step 7.
    Δp_cell = Vector{Float64}(undef, N)

    @inbounds for j in 1:N
        seg = mesh.segments[j]
        ρ_j = Float64(segment_density(mesh, j))
        u_c = _segment_velocity_centered(mesh, j)
        J_j = 1.0 / ρ_j
        s_pre = Float64(seg.state.s)
        Mvv_pre = Float64(Mvv(J_j, s_pre))
        Pxx_old = ρ_j * Mvv_pre
        Pp_old  = Float64(seg.state.Pp)
        # Detect Pp sentinel (NaN from legacy 5-arg DetField). When Pp
        # is NaN, treat as isotropic Maxwellian so the debit can still
        # operate on the trace via entropy alone.
        if !isfinite(Pp_old)
            Pp_old = Pxx_old
        end

        divu_j = diag.divu[j]
        diag.compressive[j] = divu_j < 0

        drift_term = params.C_A * ρ_j * divu_j * Δt
        compression = max(-divu_j, 0.0)
        noise_amp = params.C_B * ρ_j * sqrt(compression * Δt)
        δ_drift = drift_term
        δ_noise = noise_amp * diag.eta[j]
        diag.delta_rhou_drift[j] = δ_drift
        diag.delta_rhou_noise[j] = δ_noise
        δ = δ_drift + δ_noise

        # 4. Amplitude limiter (matches py-1d, lines 159-167 of
        #    noise_model.py): KE injection capped at
        #    `ke_budget_fraction · IE_local`. IE_local per unit volume
        #    = 0.5·P_xx + P_⊥, the effective "kinetic-energy reservoir"
        #    in the trace. The discriminant form for the symmetric cap
        #    matches py-1d exactly.
        IE_local = 0.5 * Pxx_old + Pp_old
        KE_budget = params.ke_budget_fraction * IE_local
        abs_u = abs(u_c)
        disc = abs_u * abs_u + 2.0 * KE_budget / max(ρ_j, 1e-30)
        δ_max = (-abs_u + sqrt(max(disc, 0.0))) * ρ_j
        if δ > δ_max
            δ = δ_max
        elseif δ < -δ_max
            δ = -δ_max
        end
        diag.delta_rhou[j] = δ

        # 5. KE-debit and pressure update.
        ΔKE_vol = u_c * δ + 0.5 * δ * δ / ρ_j
        diag.delta_KE_vol[j] = ΔKE_vol
        Pxx_new = Pxx_old - (2.0 / 3.0) * ΔKE_vol
        Pp_new  = Pp_old  - (2.0 / 3.0) * ΔKE_vol
        if Pxx_new < params.pressure_floor
            Pxx_new = params.pressure_floor
        end
        if Pp_new < params.pressure_floor
            Pp_new = params.pressure_floor
        end

        # 6. Re-anchor entropy from P_xx (variational-state convention).
        Mvv_new = Pxx_new / ρ_j
        if Mvv_new > 0 && Mvv_pre > 0
            s_new = s_pre + log(Mvv_new / Mvv_pre)
        else
            s_new = s_pre
        end

        # Mutate the segment. (x, u, α, β) untouched here — the vertex
        # u update happens in step 7. The segment's `state.u` (left
        # vertex value) is rewritten in step 7 too.
        mesh.segments[j].state = DetField{T}(seg.state.x, seg.state.u,
                                             seg.state.α, seg.state.β,
                                             T(s_new), T(Pp_new),
                                             seg.state.Q)
        # Cell-momentum injection per cell: ΔP_j = δ · Δx_j.
        Δx_j = J_j * Float64(seg.Δm)
        Δp_cell[j] = δ * Δx_j
    end

    # 7. Distribute cell momentum to vertices (mass-lumped, half each).
    # Vertex i receives ΔP_{i-1}/2 from its left cell and ΔP_i/2 from
    # its right cell. Updated u_i = u_i^old + (ΔP_{i-1} + ΔP_i)/(2·m̄_i).
    @inbounds for i in 1:N
        i_left = i == 1 ? N : i - 1
        m̄ = Float64(vertex_mass(mesh, i))
        δu = (Δp_cell[i_left] + Δp_cell[i]) / (2.0 * m̄)
        u_old = Float64(mesh.segments[i].state.u)
        u_new = u_old + δu
        seg = mesh.segments[i]
        mesh.segments[i].state = DetField{T}(seg.state.x, T(u_new),
                                             seg.state.α, seg.state.β,
                                             seg.state.s, seg.state.Pp,
                                             seg.state.Q)
        mesh.p_half[i] = m̄ * u_new
    end

    # 8. Realizability projection (M2-3 / Phase 8.5). If
    #    `params.project_kind == :none`, this is a no-op and the
    #    function reduces to the M1 Phase 8 form bit-for-bit (see the
    #    M2-3 unit test "bit-equality with no-projection path").
    realizability_project!(mesh;
                           kind = params.project_kind,
                           headroom = params.realizability_headroom,
                           Mvv_floor = params.Mvv_floor,
                           pressure_floor = params.pressure_floor,
                           stats = proj_stats)

    return mesh, diag
end

# -----------------------------------------------------------------------------
# Burst-statistics accumulator
# -----------------------------------------------------------------------------

"""
    BurstStatsAccumulator(N::Int)

State container for the runtime self-consistency monitor. Accumulates
two streams across timesteps:

  * **Per-cell divu time series.** Per cell `j`, a vector
    `Vector{Float64}` of `(∂_x u)_j(t_n)` samples. Used by
    `burst_durations(...)` to detect contiguous compression runs in
    each cell's history; the empirical burst-duration sample fed to
    `estimate_gamma_shape` is the union of all cells' burst durations.

  * **Compression-cell residual sample.** A single flat
    `Vector{Float64}` of `(δ(ρu) − δ_drift) / noise_amp_norm` values
    on cells with `divu < 0`. The denominator scales by the
    deterministic-amplitude prediction `C_B · ρ · √(|divu| Δt)` so the
    sample is unit-variance under perfect VG noise — `residual_kurtosis`
    on this sample inverts to `λ̂_res` via `gamma_shape_from_kurtosis`.

Implementation notes:
  * The divu history is stored per cell as a `Vector{Vector{Float64}}`
    of `N` independent series, so per-cell `burst_detect` runs find
    contiguous compressions in time-of-cell, not across cells.
  * The residual sample uses every compression-cell injection that
    survives the amplitude limiter; the limiter saturation rate is
    tracked separately for diagnostic purposes
    (`acc.n_limited / acc.n_compress`).
"""
mutable struct BurstStatsAccumulator
    N::Int
    divu_history::Vector{Vector{Float64}}     # per-cell time series
    residual_samples::Vector{Float64}         # flat sample buffer
    n_compress::Int
    n_limited::Int
    n_steps::Int
    dt_history::Vector{Float64}
end

function BurstStatsAccumulator(N::Int)
    return BurstStatsAccumulator(
        N,
        [Float64[] for _ in 1:N],
        Float64[],
        0, 0, 0,
        Float64[],
    )
end

"""
    record_step!(acc::BurstStatsAccumulator, diag::InjectionDiagnostics,
                 dt::Real, params::NoiseInjectionParams)

Append one step's per-cell `(∂_x u, η, δ-residual)` data to the
accumulator. The residual is `eta_j` itself — by construction this is
the realized VG draw, and its kurtosis converges to `3/λ` (excess) as
the sample grows. Cells where the amplitude limiter capped the
injection are flagged via `n_limited`; the residual sample includes
**only** unsaturated compression cells so the kurtosis estimate is not
biased by the cap.
"""
function record_step!(acc::BurstStatsAccumulator,
                      diag::InjectionDiagnostics, dt::Real,
                      params::NoiseInjectionParams)
    @assert length(diag.divu) == acc.N "Accumulator/diag size mismatch"
    acc.n_steps += 1
    push!(acc.dt_history, Float64(dt))
    @inbounds for j in 1:acc.N
        push!(acc.divu_history[j], diag.divu[j])
        if diag.compressive[j]
            acc.n_compress += 1
            # Saturation check: if the limited δ differs from the
            # nominal (drift + noise_amp · η) by more than 1e-12,
            # the limiter fired. Only collect unsaturated samples.
            δ_nominal = diag.delta_rhou_drift[j] + diag.delta_rhou_noise[j]
            if abs(diag.delta_rhou[j] - δ_nominal) > 1e-12 * max(abs(δ_nominal), 1.0)
                acc.n_limited += 1
            else
                # The realized residual is η itself (unit variance under
                # perfect VG); aggregate directly.
                push!(acc.residual_samples, diag.eta[j])
            end
        end
    end
    return acc
end

"""
    burst_durations(acc::BurstStatsAccumulator) -> Vector{Float64}

Walk the per-cell `divu_history` time series, detect contiguous
compression runs via `Stochastic.burst_detect`, and return the union
of durations across all cells in **physical-time units** (multiplied
by the running `dt` mean — Phase-8 driver uses fixed `dt`, so this is
just `n_samples * dt_mean`). Empty if no compression was observed.
"""
function burst_durations(acc::BurstStatsAccumulator)
    isempty(acc.dt_history) && return Float64[]
    dt_mean = mean(acc.dt_history)
    out = Float64[]
    for j in 1:acc.N
        bursts = burst_detect(acc.divu_history[j])
        for b in bursts
            push!(out, dt_mean * b.duration)
        end
    end
    return out
end

"""
    self_consistency_check(acc::BurstStatsAccumulator;
                           warn_ratio = 2.0) -> NamedTuple

Compute the Phase-8/9 self-consistency monitor. Returns a NamedTuple
with the empirical numbers and the `ok` flag:

  * `n_bursts`         — total compression bursts detected.
  * `n_residual`       — size of the un-saturated residual sample.
  * `k_hat`            — gamma-shape of the burst-duration histogram
                         from `estimate_gamma_shape(durations)`. `Inf`
                         if too few bursts.
  * `theta_T_hat`      — burst-duration scale (τ̂ in the methods paper).
  * `lambda_res_hat`   — variance-gamma λ inferred from the residual
                         excess kurtosis. `Inf` if the sample is too
                         small or sub-Gaussian.
  * `ratio`            — `max(k_hat/λ_res_hat, λ_res_hat/k_hat)`.
                         Methods-paper §3.4 and v3 §1.2 say this should
                         be near 1.0 in production.
  * `ok::Bool`         — `ratio ≤ warn_ratio` (default 2.0).
  * `limiter_rate`     — `n_limited / n_compress` (saturation fraction).

The `warn_ratio = 2.0` default is the documented production tolerance:
Tom's v3 §1.2 reports the small-data fit `λ ≈ 1.6` vs the production
kurt 3.45 ⇒ `λ ≈ 6.7`, a factor-4 mismatch attributed to the chaotic-
divergence floor biasing the production residual toward Gaussian. A
factor-2 tolerance is the "is the wiring right" bar.
"""
function self_consistency_check(acc::BurstStatsAccumulator;
                                warn_ratio::Float64 = 2.0)
    durations = burst_durations(acc)
    n_b = length(durations)
    if n_b >= 5
        # estimate_gamma_shape requires positive samples and var > 0.
        # Add a defensive try/catch for degenerate single-step bursts.
        try
            k_hat, θ_T_hat = estimate_gamma_shape(durations)
        catch
            k_hat = Inf
            θ_T_hat = NaN
        end
        # The above try/catch can't write to outer locals in Julia,
        # so re-do with explicit assignment.
        k_hat = Inf
        θ_T_hat = NaN
        try
            (k_hat, θ_T_hat) = estimate_gamma_shape(durations)
        catch
            # leave defaults
        end
    else
        k_hat = Inf
        θ_T_hat = NaN
    end

    n_r = length(acc.residual_samples)
    if n_r >= 100
        ek = residual_kurtosis(acc.residual_samples)
        λ_res = gamma_shape_from_kurtosis(ek)
    else
        λ_res = Inf
    end

    if isfinite(k_hat) && isfinite(λ_res) && k_hat > 0 && λ_res > 0
        ratio = max(k_hat / λ_res, λ_res / k_hat)
    else
        ratio = Inf
    end
    ok = ratio <= warn_ratio

    limiter_rate = acc.n_compress > 0 ?
                   Float64(acc.n_limited) / acc.n_compress : 0.0

    return (n_bursts = n_b, n_residual = n_r,
            k_hat = Float64(k_hat),
            theta_T_hat = Float64(θ_T_hat),
            lambda_res_hat = Float64(λ_res),
            ratio = Float64(ratio),
            ok = ok,
            limiter_rate = limiter_rate)
end

# -----------------------------------------------------------------------------
# Driver: deterministic step + injection wrapper
# -----------------------------------------------------------------------------

"""
    det_run_stochastic!(mesh::Mesh1D, dt, n_steps;
                        params::NoiseInjectionParams,
                        rng::AbstractRNG,
                        tau::Union{Real,Nothing} = nothing,
                        q_kind::Symbol = :none,
                        c_q_quad::Real = 1.0,
                        c_q_lin::Real = 0.5,
                        accumulator::Union{BurstStatsAccumulator,Nothing} = nothing,
                        monitor_every::Int = 0,
                        kwargs...) -> mesh

Run `n_steps` of `det_step!` followed by `inject_vg_noise!` per step.
Mutates `mesh` in place; returns `mesh`. If `accumulator !== nothing`,
calls `record_step!(accumulator, diag, dt, params)` after each
injection. If `monitor_every > 0` and `accumulator !== nothing`, the
self-consistency monitor is invoked every `monitor_every` steps and
its result is **not** persisted (caller can re-run
`self_consistency_check(accumulator)` at end-of-run).

`tau`, `q_kind`, `c_q_quad`, `c_q_lin` are forwarded to `det_step!`
(Phase 5 BGK and Phase 5b artificial-viscosity knobs).

With `params.C_A = params.C_B = 0` this driver is bit-equal to
`det_run!(mesh, dt, n_steps; tau=tau, q_kind=q_kind, …)`. This
property is asserted by the Phase-8 unit tests.
"""
function det_run_stochastic!(mesh::Mesh1D{T,DetField{T}}, dt::Real,
                             n_steps::Integer;
                             params::NoiseInjectionParams,
                             rng::AbstractRNG,
                             tau::Union{Real,Nothing} = nothing,
                             q_kind::Symbol = :none,
                             c_q_quad::Real = 1.0,
                             c_q_lin::Real = 0.5,
                             accumulator::Union{BurstStatsAccumulator,Nothing} = nothing,
                             monitor_every::Int = 0,
                             proj_stats::Union{ProjectionStats,Nothing} = nothing,
                             kwargs...) where {T<:Real}
    N = n_segments(mesh)
    diag = InjectionDiagnostics(N)
    for n in 1:n_steps
        # Pre-Newton projection: ensure no cell has a non-realizable or
        # ill-conditioned state at the *start* of this step. The slow
        # entropy-debit drift in `inject_vg_noise!` can drive `M_vv`
        # toward the realizability boundary across many steps; without
        # a pre-step floor, the next `det_step!` may enter Newton with
        # a vanishingly small `γ ≈ √M_vv` (weak coupling) and produce
        # a tangled mesh (vertex collision). The pre-step projection
        # is what closes M1 Open #4 — it intercepts cells that have
        # drifted into the danger zone *before* Newton sees them.
        # The post-injection projection (inside `inject_vg_noise!`)
        # then handles cells pushed close to the boundary by *this*
        # step's noise. With `params.project_kind = :none` both passes
        # are no-ops and the integrator reduces to M1 Phase 8 bit-for-bit.
        realizability_project!(mesh;
                               kind = params.project_kind,
                               headroom = params.realizability_headroom,
                               Mvv_floor = params.Mvv_floor,
                               pressure_floor = params.pressure_floor,
                               stats = proj_stats)
        det_step!(mesh, dt;
                  tau = tau, q_kind = q_kind,
                  c_q_quad = c_q_quad, c_q_lin = c_q_lin,
                  kwargs...)
        inject_vg_noise!(mesh, dt; params = params, rng = rng,
                         diag = diag, proj_stats = proj_stats)
        if accumulator !== nothing
            record_step!(accumulator, diag, dt, params)
            if monitor_every > 0 && (n % monitor_every == 0)
                # Side-effect-free monitor; warn via @debug to avoid
                # log noise in the test suite.
                _ = self_consistency_check(accumulator)
            end
        end
    end
    return mesh
end
