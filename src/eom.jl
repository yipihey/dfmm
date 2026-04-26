# eom.jl — HierarchicalGrids-based discrete Euler–Lagrange residual.
#
# Phase M3-0 scope: a thin shim that ports M1's Phase-1 Cholesky-sector
# integrator (single autonomous cell) onto an HG `SimplicialMesh{1, T}`
# + `PolynomialFieldSet` substrate. The numerical kernel — the discrete
# EL residual `cholesky_el_residual` — is reused **byte-identically**
# from `cholesky_sector.jl` so the HG-based path must produce
# bit-identical output to M1 modulo storage layout (no algorithmic
# divergence is allowed in M3-0).
#
# Storage convention (1D, Phase-1 / M3-0):
#
#   - `mesh::SimplicialMesh{1, T}` carries `n_simplices` segments. Each
#     simplex is two vertices; `simplex_volume(mesh, j)` is the
#     Lagrangian segment length (later phases use this as `Δm` after
#     the EOS coupling). For M3-0 (autonomous Cholesky cell) the
#     Lagrangian-mass coordinate is irrelevant; we only need a
#     per-cell container.
#
#   - `fields::PolynomialFieldSet` allocated with `MonomialBasis{1, 0}`
#     (one coefficient per cell — i.e. piecewise constant per simplex)
#     and named scalar fields `(alpha, beta)`. In later phases
#     (M3-1/2) this lifts to higher-order polynomial reconstruction.
#
# The discrete EL residual is exactly M1's `cholesky_el_residual`. The
# Newton iteration is exactly M1's `cholesky_step` (NonlinearSolve
# `SimpleNewtonRaphson` with `AutoForwardDiff`). The new wrapper
# `cholesky_step_HG!` simply reads `(α, β)` from the field set, calls
# `cholesky_step`, and writes the result back. Per-cell parallelism is
# wired through HG's `parallel_for_cells`.
#
# This file deliberately does NOT introduce a new EL residual function.
# M3-0's correctness gate is byte-equality with M1; introducing a new
# kernel risks divergence that the parity tests then have to debug.
# The HG-substrate generalization happens in M3-3 (2D Berry coupling),
# where the residual genuinely changes.

using StaticArrays: SVector
using HierarchicalGrids: SimplicialMesh, PolynomialFieldSet,
    n_simplices, n_elements, field_names, basis_of, n_coeffs

"""
    read_alphabeta(fields::PolynomialFieldSet, j::Integer)

Read the `(α, β)` pair stored at simplex `j` of an HG order-0
polynomial field set. Returns an `SVector{2,T}` matching M1's
`cholesky_step` calling convention.

The field set must have the structure produced by
`allocate_polynomial_fields(SoA(), MonomialBasis{1, 0}(), n;
alpha=T, beta=T)`. Each polynomial-view's single coefficient is
read via index `[1]`.
"""
@inline function read_alphabeta(fields::PolynomialFieldSet, j::Integer)
    α = fields.alpha[j][1]
    β = fields.beta[j][1]
    T = promote_type(typeof(α), typeof(β))
    return SVector{2,T}(T(α), T(β))
end

"""
    write_alphabeta!(fields::PolynomialFieldSet, j::Integer, q)

Write the `(α, β)` pair `q::SVector{2,T}` to simplex `j`'s order-0
coefficients in the HG field set. Mirrors `read_alphabeta`.
"""
@inline function write_alphabeta!(fields::PolynomialFieldSet,
                                   j::Integer, q)
    fields.alpha[j] = (q[1],)
    fields.beta[j]  = (q[2],)
    return q
end

"""
    cholesky_el_residual_HG(fields, j, q_np1, q_n, M_vv, divu_half, dt)

Per-simplex discrete Euler–Lagrange residual on an HG-based field
set. Algebraically identical to M1's `cholesky_el_residual`; the only
difference is the calling convention (the HG path receives the
field-set + simplex index for context-aware later phases). In Phase
M3-0 the body is a single delegation to `cholesky_el_residual` so
that bit-equality is automatic.

Arguments:
- `fields::PolynomialFieldSet` — current field set (unused in
  Phase M3-0; reserved for later phases that need neighbour data).
- `j::Integer`                  — simplex index (unused in M3-0;
  retained for the future per-cell EOS-coupling phases).
- `q_np1::SVector{2}`           — unknown `(α_{n+1}, β_{n+1})`.
- `q_n::SVector{2}`             — known `(α_n, β_n)`.
- `M_vv`                        — externally-supplied second moment.
- `divu_half`                   — midpoint strain rate.
- `dt`                          — timestep.

Returns: `SVector{2}` residual matching `cholesky_el_residual`.
"""
@inline function cholesky_el_residual_HG(::PolynomialFieldSet,
                                          ::Integer,
                                          q_np1, q_n,
                                          M_vv, divu_half, dt)
    return cholesky_el_residual(q_np1, q_n, M_vv, divu_half, dt)
end
