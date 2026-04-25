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
