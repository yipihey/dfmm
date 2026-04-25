# artificial_viscosity.jl
#
# Phase 5b — opt-in artificial viscosity (tensor-q, 1D specialisation) for
# the variational integrator.
#
# ──────────────────────────────────────────────────────────────────────
# Why this exists.
# ──────────────────────────────────────────────────────────────────────
#
# The bare variational Lagrangian is non-dissipative and produces
# only ~O(Δx²) numerical viscosity in smooth flow; at shock fronts
# the discrete Euler-Lagrange equations have no mechanism to enforce
# the Rankine-Hugoniot jump conditions exactly. Phase 5's Sod result
# shows the symptom: a 1-cell shock-front offset and a ~15-20 % error
# in the post-shock plateau (`reference/notes_phase5_sod_FAILURE.md`).
#
# The standard fix in Lagrangian hydro is to add an *artificial
# viscous pressure* `q` to the EL momentum equation in compressive
# flow. We implement the **Kuropatenko 1968 / von Neumann-Richtmyer
# 1950** combined linear+quadratic form (Caramana-Shashkov-Whalen
# 1998 §2 reduces to this in 1D):
#
#   q = ρ [ c_q^{(2)} · L² · (∂_x u)² + c_q^{(1)} · L · c_s · |∂_x u| ]
#       (in compression, ∂_x u < 0)
#   q = 0   (in expansion, ∂_x u ≥ 0)
#
# where
#   L   = Δx_seg, the segment's Eulerian length;
#   c_s = √(Γ M_vv), the local sound speed for an ideal gas;
#   c_q^{(2)} ∈ [1, 2]   — quadratic coefficient (default 1.0);
#   c_q^{(1)} ∈ [0, 0.5] — linear coefficient (default 0.5).
#
# The quadratic term (von Neumann-Richtmyer 1950) is the workhorse
# for strong shocks; the linear term (Landshoff 1955) suppresses the
# post-shock oscillations the quadratic-only form leaves behind.
# Caramana et al. recommend the joint linear+quadratic form for
# robust shock capture across Mach regimes.
#
# Effect on the discrete equations (see cholesky_sector.jl):
#   * Momentum residual:  ∂_m P_xx → ∂_m (P_xx + q).
#   * Internal-energy / entropy: q dissipates kinetic energy at rate
#     (q/ρ) (∂_x u). The post-Newton entropy update absorbs this
#     dissipation into s (see newton_step.jl `det_step!`).
#
# This is **opt-in**: callers pass `q_kind = :none` (default) for the
# bare variational integrator, or `q_kind = :vNR_linear_quadratic` to
# turn it on. The variational ideal is q-off; q-on is the standard
# numerical-aid for shock-bearing problems.
#
# References:
#   von Neumann, J. & Richtmyer, R. D. (1950). J. Appl. Phys. 21, 232.
#   Kuropatenko, V. F. (1968). Trans. V. A. Steklov Math. Inst. 74, 49.
#   Caramana, E. J., Shashkov, M. J., & Whalen, P. P. (1998).
#     J. Comput. Phys. 144, 70-97.
#   Landshoff, R. (1955). LANL report LA-1930 (linear-q rationale).

const Q_KIND_NONE                = :none
const Q_KIND_VNR_LINEAR_QUADRATIC = :vNR_linear_quadratic

"""
    compute_q_segment(divu, ρ, c_s, L; c_q_quad = 1.0, c_q_lin = 0.5)

Per-segment 1D Kuropatenko / von Neumann-Richtmyer artificial viscous
pressure:

    q = ρ [ c_q_quad · L² · (div u)² + c_q_lin · L · c_s · |div u| ]
        if div u < 0  (compression)
    q = 0
        if div u ≥ 0  (expansion / smooth flow)

All inputs are scalars at one segment midpoint. Returns a scalar of
the same numeric type, AD-friendly (works under ForwardDiff `Dual`).

Arguments:
- `divu` — `(∂_x u)_{n+1/2}` at the segment midpoint (Eulerian strain).
- `ρ`    — segment density at the midpoint.
- `c_s`  — sound speed at the segment, e.g. `√(Γ M_vv)`. Only enters the
           linear term.
- `L`    — characteristic length scale; here taken to be the segment's
           Eulerian length `Δx = J · Δm`.
- `c_q_quad`, `c_q_lin` — the two q-coefficients. See file header for
   the recommended ranges.

The branch on `divu` is differentiable through ForwardDiff because the
expansion branch returns a typed zero of the right element; only the
non-smooth point at `divu = 0` is a problem in principle, but the
Newton solver evaluates this at finite-precision strains where the
sub-gradient choice is unambiguous in practice. (Same caveat applies
to every Lagrangian-hydro code that uses Wilkins/Caramana q.)
"""
@inline function compute_q_segment(divu::Real, ρ::Real, c_s::Real, L::Real;
                                   c_q_quad::Real = 1.0,
                                   c_q_lin::Real  = 0.5)
    # Promote: ρ, divu, c_s, L can all be ForwardDiff Duals.
    T = promote_type(typeof(divu), typeof(ρ), typeof(c_s), typeof(L),
                     typeof(c_q_quad), typeof(c_q_lin))
    if divu < zero(divu)
        absdu = -divu                         # = |div u| in compression
        quad  = T(c_q_quad) * L^2 * absdu^2
        lin   = T(c_q_lin)  * L * c_s * absdu
        return T(ρ) * (quad + lin)
    else
        return zero(T)
    end
end

"""
    q_kind_supported(q_kind::Symbol) -> Bool

True if `q_kind` is a recognised artificial-viscosity flavour.
"""
q_kind_supported(q_kind::Symbol) =
    q_kind === Q_KIND_NONE || q_kind === Q_KIND_VNR_LINEAR_QUADRATIC

"""
    q_active(q_kind::Symbol) -> Bool

True if `q_kind` is anything other than `:none`. Cheap dispatch
predicate used by `det_el_residual` to skip the q-cost when off.
"""
q_active(q_kind::Symbol) = q_kind !== Q_KIND_NONE
