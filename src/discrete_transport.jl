# discrete_transport.jl
#
# Discrete covariant derivative on the principal GL(d) bundle, 1D
# specialization. The continuous form (v2 §2.1, eq. eq:Dtq-1D):
#
#   D_t^{(q)} φ = ∂_t φ + q · (∂_x u) · φ.
#
# The discrete approximation per methods-paper §9.5 ("Discrete
# parallel transport") evaluates the field and the strain rate at the
# midpoint of the timestep:
#
#   D_t^{(q)} φ |_{n+1/2} ≈ (φ_{n+1} − φ_n) / Δt
#                          + q · (∂_x u)_{n+1/2}_bar · φ_{n+1/2}_bar,
#
# where _bar denotes the half-step average (φ_n + φ_{n+1})/2 and the
# strain rate is supplied externally for Phase 1.
#
# This file implements the operator. The Cholesky-sector EL residual
# (cholesky_sector.jl) calls into it with q ∈ {0, 1} for α and β.

"""
    D_t_q(phi_n, phi_np1, divu_half, q, dt)

Discrete approximation of the covariant time derivative
`D_t^{(q)}φ = ∂_t φ + q (∂_x u) φ` evaluated at the midpoint
`t_{n+1/2}` of the step (methods-paper §9.5). All arguments are
scalars.

Arguments:
- `phi_n`     — field value at time `t_n`.
- `phi_np1`   — field value at time `t_{n+1}` (the unknown when used
                inside the EL residual).
- `divu_half` — midpoint strain rate `(∂_x u)_{n+1/2}` (supplied
                externally; in Phase 1 a fixed scalar).
- `q`         — integer charge under the velocity strain group
                (`0` for α, `1` for β and γ in 1D).
- `dt`        — timestep `Δt`.

Returns: the midpoint discrete approximation of `D_t^{(q)}φ`. Used as
the LHS of the discrete EL system in `cholesky_sector.jl`.
"""
function D_t_q(phi_n, phi_np1, divu_half, q, dt)
    phi_bar = (phi_n + phi_np1) / 2
    dphi_dt = (phi_np1 - phi_n) / dt
    return dphi_dt + q * divu_half * phi_bar
end
