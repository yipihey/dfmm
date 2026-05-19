# enzyme_adjoint_HG.jl
#
# Reverse-mode end-to-end gradient for the M3-0 HG Phase-1 driver
# (`cholesky_step_HG!` / `cholesky_run_HG!`). The HG Phase-1 path is
# embarrassingly parallel across cells — each simplex's Newton solve
# uses only the cell's own `(α, β)` plus the shared `(M_vv, divu, dt)`
# scalars — so the adjoint is the per-cell Phase-1 pullback applied
# across all cells, with the three time-constant parameter cotangents
# summed.
#
# This deliverable closes the "HG 1D production driver" item by
# demonstrating that the IFT pullback transparently lifts onto the HG
# storage substrate (`PolynomialFieldSet`) without changing the
# algorithmic structure of the adjoint. The Phase-2/M3-1 HG drivers
# (which do introduce inter-cell coupling) are a routine extension of
# the Phase-2 sparse-adjoint deliverable in `enzyme_adjoint_phase2.jl`.

using HierarchicalGrids: SimplicialMesh, n_simplices, n_elements

"""
    cholesky_run_HG_gradient(loss, mesh, fields, M_vv, divu_half, dt, N_steps; kwargs...)
        -> (loss_value, ᾱ_init, β̄_init, M̄_vv, divū_half, d̄t)

End-to-end reverse-mode gradient of a scalar `loss(αs, βs)` evaluated
on the per-cell final state of the HG Phase-1 driver
`cholesky_run_HG!`. The driver mutates `fields` in place (forward
trajectory), then back-propagates per-cell pullbacks and accumulates
parameter cotangents.

`loss(αs::AbstractVector{Float64}, βs::AbstractVector{Float64}) -> Real`
is differentiated with Enzyme reverse (one pass, marked `Const` so
captures in the closure are tolerated).

Returns:
  • the scalar loss value,
  • `ᾱ_init`, `β̄_init` — per-cell cotangents on the initial state,
  • the three scalar parameter cotangents `(M̄_vv, divū_half, d̄t)`
    summed across cells.
"""
function cholesky_run_HG_gradient(loss::F,
                                   mesh,
                                   fields,
                                   M_vv::Real, divu_half::Real, dt::Real,
                                   N_steps::Integer;
                                   abstol::Real = 1e-13,
                                   reltol::Real = 1e-13,
                                   maxiters::Int = 50) where {F}
    N_cells = n_simplices(mesh)
    @assert n_elements(fields) == N_cells

    M_vv64 = Float64(M_vv); divu64 = Float64(divu_half); dt64 = Float64(dt)

    # Forward: per-cell trajectory of (α, β) pairs.
    trajs = Vector{Vector{SVector{2,Float64}}}(undef, N_cells)
    @inbounds for j in 1:N_cells
        trajs[j] = Vector{SVector{2,Float64}}(undef, N_steps + 1)
        q0 = read_alphabeta(fields, j)
        # Force Float64 storage; `read_alphabeta` returns SVector{2, T}
        # where T may be wider/narrower depending on the field-set
        # element type.
        trajs[j][1] = SVector{2,Float64}(Float64(q0[1]), Float64(q0[2]))
    end
    @inbounds for n in 1:N_steps, j in 1:N_cells
        trajs[j][n+1] = cholesky_step(trajs[j][n], M_vv64, divu64, dt64;
                                       abstol = abstol, reltol = reltol,
                                       maxiters = maxiters)
    end
    # Write final state back into the HG field set so callers can
    # inspect it as if they had called `cholesky_run_HG!` directly.
    @inbounds for j in 1:N_cells
        write_alphabeta!(fields, j, trajs[j][end])
    end

    # Loss evaluated on the flat (αs, βs) arrays.
    αs_final = Float64[trajs[j][end][1] for j in 1:N_cells]
    βs_final = Float64[trajs[j][end][2] for j in 1:N_cells]
    L = loss(αs_final, βs_final)

    # Loss gradient seed via Enzyme reverse on the flat-vector form.
    dα = zero(αs_final); dβ = zero(βs_final)
    Enzyme.autodiff(Reverse, Const(loss), Active,
                    Duplicated(αs_final, dα),
                    Duplicated(βs_final, dβ))
    q̄_final = Vector{SVector{2,Float64}}(undef, N_cells)
    @inbounds for j in 1:N_cells
        q̄_final[j] = SVector{2,Float64}(dα[j], dβ[j])
    end

    # Reverse pass: per-cell chain of IFT pullbacks.
    ᾱ_init = zeros(Float64, N_cells)
    β̄_init = zeros(Float64, N_cells)
    M̄_vv = 0.0; divū = 0.0; d̄t = 0.0
    @inbounds for j in 1:N_cells
        q̄ = q̄_final[j]
        for n in N_steps:-1:1
            q_np1 = trajs[j][n+1]
            q_n   = trajs[j][n]
            J = _jacobian_q(q_np1, q_n, M_vv64, divu64, dt64)
            λ = transpose(J) \ q̄
            _, _, dα_n, dβ_n, dM, dd, dt_g = _residual_vjp_full(
                λ[1], λ[2],
                q_np1[1], q_np1[2],
                q_n[1],   q_n[2],
                M_vv64,   divu64, dt64,
            )
            q̄ = SVector{2,Float64}(-dα_n, -dβ_n)
            M̄_vv += -dM; divū += -dd; d̄t += -dt_g
        end
        ᾱ_init[j] = q̄[1]
        β̄_init[j] = q̄[2]
    end
    return L, ᾱ_init, β̄_init, M̄_vv, divū, d̄t
end
