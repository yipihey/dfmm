"""
Equation of state — `Mvv(J, s)` and helpers.

Provides the velocity-variance closure
    M_vv(J, s) = (rho)^(Gamma - 1) * exp(s / c_v)
              = J^(1 - Gamma) * exp(s / c_v)
for an ideal gas, where `J = 1/rho` is the specific volume (Lagrangian
Jacobian per unit reference mass) and `s` is specific entropy in units
of `c_v`. With `Gamma = 5/3` (monatomic, the dfmm default) and
`c_v = 1`, the cold limit `s -> -infinity` drives `M_vv -> 0` for any
fixed `J`, exactly as required by the action note v2 §3.5 cold-limit
reduction.

Why this form? The methods paper §3 introduces `M_vv(J, s)` as the
EOS-derived velocity variance closing the Cholesky factor through
`gamma = sqrt(M_vv - beta^2)`. The standard adiabat for an ideal gas
in (J, s) coordinates gives `P J^Gamma = exp(s/c_v) * const`; with
`P = rho * M_vv` (kinetic-moment definition) and the chosen
normalization the closed form above falls out. The py-1d reference
implementation tracks `M_vv = P_xx/rho` directly as a state moment
(no separate `(J, s)` decomposition) — this module supplies the
thermodynamic factorization the variational scheme needs in addition.

Units. `J` is specific volume `[length / mass]` in 1D; `s` is
dimensionless (`c_v`-normalized). `M_vv` carries units of
`(velocity)^2`.

Public API:
    Mvv(J, s; Gamma=5/3, cv=1.0)              -> Float64
    gamma_from_state(Mvv_val, beta)           -> Float64
    Mvv_from_pressure(rho, P)                 -> Float64
    pressure_from_Mvv(rho, Mvv_val)           -> Float64
"""

const GAMMA_LAW_DEFAULT = 5.0 / 3.0  # monatomic ideal gas; matches py-1d kinetic-moment closure
const CV_DEFAULT = 1.0               # entropy is tracked in c_v units

"""
    Mvv(J, s; Gamma=5/3, cv=1.0) -> Float64

Velocity-variance closure for an ideal gas:
`M_vv = J^(1 - Gamma) * exp(s / cv)`.

* Cold-limit: `s -> -Inf` ⇒ `M_vv -> 0` (returns `0.0` exactly when the
  exponential underflows; never returns `NaN` for finite `J > 0`).
* Hot-limit: `s -> +Inf` ⇒ `M_vv -> Inf`.
* `J <= 0` is unphysical and returns `NaN` (caller's bug).
"""
function Mvv(J::Real, s::Real; Gamma::Real = GAMMA_LAW_DEFAULT,
             cv::Real = CV_DEFAULT)::Float64
    if !(J > 0)
        return NaN
    end
    # Compose log(M_vv) = (1 - Gamma) * log(J) + s/cv to keep the
    # cold-limit underflow well-behaved.
    log_Mvv = (1 - Gamma) * log(J) + s / cv
    if log_Mvv < -700  # exp(-700) underflows to 0.0 in Float64
        return 0.0
    end
    return exp(log_Mvv)
end

"""
    gamma_from_state(Mvv_val, beta) -> Float64

Cholesky-factor `gamma = sqrt(max(M_vv - beta^2, 0))`.
The `max` clamps to zero across the realizability boundary so the
square root is always real; callers needing a violation marker should
use `realizability_marker(...)` from `diagnostics.jl`.
"""
gamma_from_state(Mvv_val::Real, beta::Real)::Float64 =
    sqrt(max(Float64(Mvv_val) - Float64(beta)^2, 0.0))

"""
    Mvv_from_pressure(rho, P) -> Float64

Kinetic-moment shortcut: `M_vv = P_xx / rho` (py-1d convention,
`schemes/_common.py` and `diagnostics.py` line 43). Returns `Inf`
when `rho == 0`.
"""
Mvv_from_pressure(rho::Real, P::Real)::Float64 =
    rho == 0 ? Inf : Float64(P) / Float64(rho)

"""
    pressure_from_Mvv(rho, Mvv_val) -> Float64

Inverse: `P_xx = rho * M_vv`.
"""
pressure_from_Mvv(rho::Real, Mvv_val::Real)::Float64 =
    Float64(rho) * Float64(Mvv_val)
