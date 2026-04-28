"""
Per-cell closure diagnostics on the Cholesky-sector primitives
`(alpha, beta, gamma)`.

These are pure scalar functions over a single phase-space-fiber state;
they have no dependence on the variational integrator's internal
representation, which keeps them independently unit-testable and lets
Track A's (alpha, beta, gamma) live in any container shape. Use
`StaticArrays.SMatrix` for the explicit 2x2 Hessian so callers can
linear-algebra-it (eigvals, condition number) without allocations.

Reference: design/03_action_note_v2.tex §3.5 (Hessian degeneracy at
gamma -> 0). The Hessian of H_Ch = -1/2 * alpha^2 * gamma^2 on the
(alpha, beta) tangent space, with gamma = sqrt(M_vv - beta^2) treated
as a function of beta, is
    Hess(H_Ch) = [[-gamma^2, 2*alpha*beta],
                  [ 2*alpha*beta, alpha^2 ]],
    det = -alpha^2 * gamma^2 - 4 * alpha^2 * beta^2
                                 ----> -4*alpha^2*beta^2 as gamma->0.
The Hessian becomes rank-1 when *both* gamma=0 and beta=0 — the
phase-space rank-collapse caustic that the variational unification
identifies as the cold-limit singularity.

The realizability marker is the py-1d convention: `gamma > 0` is the
realizability cone interior, with `M_vv >= beta^2` as the boundary
(see `py-1d/dfmm/diagnostics.py` line 46:
`gamma = np.sqrt(np.maximum(Svv - beta**2, 0))`).
"""

using StaticArrays

"""
    gamma_rank_indicator(alpha, beta, gamma) -> Float64

Phase-space rank-loss diagnostic. Returns the dimensionless
"normalized gamma" used by py-1d (`diagnostics.py` line 47:
`g_norm = gamma/sqrt(Svv_safe)`):
    g_rank = gamma / sqrt(M_vv) = gamma / sqrt(beta^2 + gamma^2).

Range: `[0, 1]`. Approaches `0` at the rank-collapse caustic
(cold limit / shell-crossing); approaches `1` for an isotropic warm
state with `beta = 0`. Returns `0.0` when both `beta` and `gamma` are
zero (safe limit).
"""
function gamma_rank_indicator(alpha::Real, beta::Real, gamma::Real)::Float64
    Mvv_loc = Float64(beta)^2 + Float64(gamma)^2
    if Mvv_loc <= 0
        return 0.0
    end
    return Float64(gamma) / sqrt(Mvv_loc)
end

"""
    realizability_marker(alpha, beta, gamma) -> Float64

Scalar that is `0` inside the realizability cone (`M_vv >= beta^2`,
i.e. `gamma^2 + 0 >= 0` is satisfied by construction when gamma is
real) and *positive* when realizability is violated. Concretely:
returns `max(beta^2 - (beta^2 + gamma^2), 0) = max(-gamma^2, 0)`,
which is `0` if `gamma >= 0` and `gamma^2` if a numerically negative
`gamma^2 = M_vv - beta^2` was reached.

Callers operating on raw `(alpha, beta, M_vv)` (where `gamma` has not
yet been clamped to `0`) can pass `gamma = sqrt(complex(M_vv-beta^2))`
real-or-imag form via the alternative
`realizability_marker_from_Mvv` below.
"""
function realizability_marker(alpha::Real, beta::Real, gamma::Real)::Float64
    g = Float64(gamma)
    return g < 0 ? g * g : 0.0
end

"""
    realizability_marker_from_Mvv(beta, Mvv_val) -> Float64

`max(beta^2 - M_vv, 0)`. Fires (positive) when the realizability
inequality `M_vv >= beta^2` is violated. Useful when you want to
diagnose violation directly from the EOS-supplied `M_vv` without
already having clamped to zero in `gamma_from_state`.
"""
function realizability_marker_from_Mvv(beta::Real, Mvv_val::Real)::Float64
    deficit = Float64(beta)^2 - Float64(Mvv_val)
    return max(deficit, 0.0)
end

"""
    hessian_HCh(alpha, beta, gamma) -> SMatrix{2,2,Float64}

The 2x2 Hessian of `H_Ch = -1/2 * alpha^2 * gamma^2` on the
(alpha, beta) tangent space:
    [[-gamma^2,    2*alpha*beta],
     [ 2*alpha*beta, alpha^2     ]].
Per design/03_action_note_v2.tex §3.5, eq. (Hessian).
"""
function hessian_HCh(alpha::Real, beta::Real, gamma::Real)::SMatrix{2, 2, Float64}
    a = Float64(alpha)
    b = Float64(beta)
    g = Float64(gamma)
    return SMatrix{2, 2, Float64}(-g * g, 2 * a * b, 2 * a * b, a * a)
end

"""
    det_hessian_HCh(alpha, beta, gamma) -> Float64

Closed-form determinant of `hessian_HCh`:
    det = -alpha^2 * gamma^2 - 4 * alpha^2 * beta^2.

Limits (matching v2 §3.5):
* `gamma -> 0` ⇒ `det -> -4 * alpha^2 * beta^2`.
* `beta  -> 0` ⇒ `det -> -alpha^2 * gamma^2`.
* both `0` ⇒ `det = 0` (rank-1 Hessian — the caustic).
"""
function det_hessian_HCh(alpha::Real, beta::Real, gamma::Real)::Float64
    a2 = Float64(alpha)^2
    return -a2 * Float64(gamma)^2 - 4 * a2 * Float64(beta)^2
end

# -----------------------------------------------------------------------------
# Per-axis (2D) Cholesky-sector γ diagnostic — M3-3d
# -----------------------------------------------------------------------------

"""
    gamma_per_axis_2d_field(fields, leaves; M_vv_override=nothing,
                             ρ_ref=1.0, Gamma=GAMMA_LAW_DEFAULT)
        -> Matrix{Float64}

Walk the leaves of a 2D Cholesky-sector field set (allocated by
`allocate_cholesky_2d_fields` in `src/setups_2d.jl`) and compute the
per-axis γ diagnostic per leaf:

    γ_a[i] = √max(M_vv,aa − β_a²[i], 0)         for a = 1, 2.

Returns a `2 × N` matrix where `out[:, i]` is the per-axis γ at leaf
`leaves[i]`. The `M_vv_override::NTuple{2, Real}` argument forces a
constant per-axis `M_vv,aa` (used by the M3-3d §6.5 selectivity test);
when `nothing`, `M_vv,aa = Mvv(J, s)` is read from the EOS with
`J = 1/ρ_ref` (isotropic — both axes share the same `Mvv(J, s)`).

# Use cases (§4.3 of the M3-3 design note)

  • Per-axis AMR selectivity (the C.2 collapsing-axis test).
  • Per-axis realizability projection (M2-3 carry-over for D.1 KH).
  • Diagnostic plot for the M3-3d headline figure
    (`reference/figs/M3_3d_per_axis_gamma_selectivity.png`).

The math primitive `gamma_per_axis_2d(β, M_vv_diag)` lives in
`src/cholesky_DD.jl` and operates on `SVector{2, T}` inputs; this
walker is the public field-set-layer wrapper.
"""
function gamma_per_axis_2d_field(fields, leaves;
                                 M_vv_override = nothing,
                                 ρ_ref::Real = 1.0,
                                 Gamma::Real = GAMMA_LAW_DEFAULT)
    N = length(leaves)
    out = Matrix{Float64}(undef, 2, N)
    @inbounds for (i, ci) in enumerate(leaves)
        β1 = Float64(fields.β_1[ci][1])
        β2 = Float64(fields.β_2[ci][1])
        s_i = Float64(fields.s[ci][1])
        if M_vv_override !== nothing
            Mvv1 = Float64(M_vv_override[1])
            Mvv2 = Float64(M_vv_override[2])
        else
            Mvv_iso = Float64(Mvv(1.0 / Float64(ρ_ref), s_i; Gamma = Gamma))
            Mvv1 = Mvv_iso
            Mvv2 = Mvv_iso
        end
        γ1² = Mvv1 - β1 * β1
        γ2² = Mvv2 - β2 * β2
        out[1, i] = sqrt(max(γ1², 0.0))
        out[2, i] = sqrt(max(γ2², 0.0))
    end
    return out
end

"""
    gamma_per_axis_2d_diag(fields, leaves; M_vv_override=nothing,
                            ρ_ref=1.0, Gamma=GAMMA_LAW_DEFAULT)
        -> Matrix{Float64}

Diagnostics-layer accessor alias for `gamma_per_axis_2d_field`. Provided
for symmetry with the 1D scalar diagnostics (`gamma_rank_indicator`,
`realizability_marker`, …) and as the canonical name for HDF5/JLD2
snapshot keys (the I/O layer in `src/io.jl` is generic over named
tuples — callers populate the snapshot's `:gamma_per_axis_2d` entry
from this helper).

Returns a `2 × length(leaves)` matrix; row `a` is `γ_a` across leaves
in the order of `leaves`.
"""
gamma_per_axis_2d_diag(fields, leaves; kwargs...) =
    gamma_per_axis_2d_field(fields, leaves; kwargs...)

# ─────────────────────────────────────────────────────────────────────
# M3-7d: per-axis γ field-walking helper for the 3D Cholesky-sector field set
# ─────────────────────────────────────────────────────────────────────

"""
    gamma_per_axis_3d_field(fields, leaves; M_vv_override=nothing,
                             ρ_ref=1.0, Gamma=GAMMA_LAW_DEFAULT)
        -> Matrix{Float64}

Walk the leaves of a 3D Cholesky-sector field set (allocated by
`allocate_cholesky_3d_fields` in `src/setups_2d.jl`) and compute the
per-axis γ diagnostic per leaf:

    γ_a[i] = √max(M_vv,aa − β_a²[i], 0)         for a = 1, 2, 3.

Returns a `3 × N` matrix where `out[:, i]` is the per-axis γ at leaf
`leaves[i]`. The `M_vv_override::NTuple{3, Real}` argument forces a
constant per-axis `M_vv,aa` (used by the M3-7d §7.5 selectivity test);
when `nothing`, `M_vv,aa = Mvv(J, s)` is read from the EOS with
`J = 1/ρ_ref` (isotropic — all three axes share the same `Mvv(J, s)`).

# Use cases (§4.3 of the M3-7 design note)

  • Per-axis 3D AMR selectivity (the C.2 collapsing-axis test, lifted
    to 3D). With `k = (1, 0, 0)` only γ_1 develops spatial structure;
    with `k = (1, 1, 0)` γ_1, γ_2 do; with `k = (1, 1, 1)` all three.
  • Per-axis 3D realizability projection (`realizability_project_3d!`).
  • Diagnostic plot for the 3D Zel'dovich pancake test (M3-7e).

The math primitive `gamma_per_axis_3d(β, M_vv_diag)` lives in
`src/cholesky_DD_3d.jl` and operates on `SVector{3, T}` inputs; this
walker is the public field-set-layer wrapper (the 3D analog of
`gamma_per_axis_2d_field`).
"""
function gamma_per_axis_3d_field(fields, leaves;
                                  M_vv_override = nothing,
                                  ρ_ref::Real = 1.0,
                                  Gamma::Real = GAMMA_LAW_DEFAULT)
    N = length(leaves)
    out = Matrix{Float64}(undef, 3, N)
    @inbounds for (i, ci) in enumerate(leaves)
        β1 = Float64(fields.β_1[ci][1])
        β2 = Float64(fields.β_2[ci][1])
        β3 = Float64(fields.β_3[ci][1])
        s_i = Float64(fields.s[ci][1])
        if M_vv_override !== nothing
            Mvv1 = Float64(M_vv_override[1])
            Mvv2 = Float64(M_vv_override[2])
            Mvv3 = Float64(M_vv_override[3])
        else
            Mvv_iso = Float64(Mvv(1.0 / Float64(ρ_ref), s_i; Gamma = Gamma))
            Mvv1 = Mvv_iso
            Mvv2 = Mvv_iso
            Mvv3 = Mvv_iso
        end
        γ1² = Mvv1 - β1 * β1
        γ2² = Mvv2 - β2 * β2
        γ3² = Mvv3 - β3 * β3
        out[1, i] = sqrt(max(γ1², 0.0))
        out[2, i] = sqrt(max(γ2², 0.0))
        out[3, i] = sqrt(max(γ3², 0.0))
    end
    return out
end

"""
    gamma_per_axis_3d_diag(fields, leaves; M_vv_override=nothing,
                            ρ_ref=1.0, Gamma=GAMMA_LAW_DEFAULT)
        -> Matrix{Float64}

Diagnostics-layer alias for `gamma_per_axis_3d_field`. Provided for
symmetry with the 1D and 2D scalar diagnostics and as the canonical
name for HDF5/JLD2 snapshot keys (the I/O layer in `src/io.jl` is
generic over named tuples — callers populate the snapshot's
`:gamma_per_axis_3d` entry from this helper).

Returns a `3 × length(leaves)` matrix; row `a` is `γ_a` across leaves
in the order of `leaves`.
"""
gamma_per_axis_3d_diag(fields, leaves; kwargs...) =
    gamma_per_axis_3d_field(fields, leaves; kwargs...)
