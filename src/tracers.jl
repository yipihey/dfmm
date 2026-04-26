# tracers.jl
#
# Phase 11 — Tier B.5 passive scalar advection.
#
# A passive scalar T(x, t) satisfying ∂_t T + u · ∇T = 0 is, in the
# variational framework, identified with a function of the parcel
# label m. In the Lagrangian frame the evolution is *trivial*:
# T(m, t) = T_0(m) for all t, since the parcel label itself is
# conserved by definition. (See methods paper §7.)
#
# Concretely: each `Mesh1D` segment carries a fixed Lagrangian mass
# label m and Δm. A tracer field is just a value attached to that
# segment, never written to once initialised. There is no flux
# computation, no slope limiter, no upwinding scheme. In Milestone
# 1's 1D scope the remap is a trivial interval intersection that we
# do not exercise; the deterministic-Lagrangian phase therefore
# produces *literally zero* numerical diffusion of tracers — the
# field array isn't mutated at all.
#
# Track-A separation. Per the Phase-11 brief, the tracer storage
# does NOT extend `DetField` or `Mesh1D`. We carry tracers in a
# parallel `TracerMesh` struct that holds a *reference* to the
# fluid mesh and an `[n_tracers, n_segments]` value matrix. This
# decouples tracer plumbing from the Track-A core (Phases 1-7) and
# avoids merge conflicts with the steady-shock and stochastic-
# injection phases that are landing on `src/segment.jl` and
# `src/newton_step.jl` in parallel.

"""
    TracerMesh{T<:Real, M<:Mesh1D}

A passive-scalar field bundle attached to a Lagrangian mesh.

Fields:
- `fluid::M`           — reference (NOT a copy) to the underlying
                         `Mesh1D`. The tracer does not mutate it.
- `tracers::Matrix{T}` — values of each passive scalar on each
                         segment. Shape `[n_tracers, n_segments]`.
                         Row `k` is the k-th tracer's value array.
- `names::Vector{Symbol}` — symbolic labels for each tracer (length
                         `n_tracers`). Used for human-readable
                         diagnostics; not load-bearing.

The struct is *immutable in semantics* during deterministic
integration: `advect_tracers!` is a no-op. The caller may mutate
`tracers` directly to set initial conditions, but should never
write to it inside an integration loop.
"""
mutable struct TracerMesh{T<:Real, M<:Mesh1D}
    fluid::M
    tracers::Matrix{T}
    names::Vector{Symbol}
end

"""
    TracerMesh(fluid::Mesh1D{T}; n_tracers = 1, names = nothing) -> TracerMesh

Allocate a tracer field bundle of `n_tracers` rows on `fluid`, all
zeros. The caller fills the rows via `set_tracer!` or by direct
mutation of the returned `tm.tracers` matrix.

`names` defaults to `[:tracer1, :tracer2, …]`.
"""
function TracerMesh(fluid::Mesh1D{T,F};
                    n_tracers::Integer = 1,
                    names::Union{Nothing,AbstractVector{Symbol}} = nothing,
                   ) where {T<:Real, F}
    @assert n_tracers ≥ 1 "TracerMesh requires at least one tracer field"
    N = n_segments(fluid)
    values = zeros(T, n_tracers, N)
    if names === nothing
        nms = [Symbol("tracer", k) for k in 1:n_tracers]
    else
        @assert length(names) == n_tracers "names length must match n_tracers"
        nms = collect(Symbol, names)
    end
    return TracerMesh{T, typeof(fluid)}(fluid, values, nms)
end

"""
    n_tracer_fields(tm::TracerMesh) -> Int

Number of passive-scalar fields carried by `tm`.
"""
n_tracer_fields(tm::TracerMesh) = size(tm.tracers, 1)

"""
    n_tracer_segments(tm::TracerMesh) -> Int

Number of segments the tracer fields live on (must match
`n_segments(tm.fluid)`).
"""
n_tracer_segments(tm::TracerMesh) = size(tm.tracers, 2)

"""
    tracer_index(tm::TracerMesh, name::Symbol) -> Int

Look up a tracer's row index by its symbolic name. Raises an
`ArgumentError` if the name is not present.
"""
function tracer_index(tm::TracerMesh, name::Symbol)
    k = findfirst(==(name), tm.names)
    k === nothing && throw(ArgumentError(
        "tracer name $name not found; have $(tm.names)"))
    return k
end

"""
    set_tracer!(tm::TracerMesh, name_or_index, values::AbstractVector)
    set_tracer!(tm::TracerMesh, name_or_index, f::Function)

Initialise a tracer field. The vector form sets row `k` to
`values[1:N]`. The functional form evaluates `f(m_j)` at each
segment's Lagrangian mass-coordinate `m_j` (i.e. the centre label).

Both raise if `name_or_index` is invalid or if the vector length
does not match `n_segments(tm.fluid)`.
"""
function set_tracer!(tm::TracerMesh, k::Integer, values::AbstractVector)
    N = n_tracer_segments(tm)
    @assert length(values) == N "values length $(length(values)) ≠ N=$N"
    @assert 1 ≤ k ≤ n_tracer_fields(tm) "tracer index $k out of range"
    @inbounds for j in 1:N
        tm.tracers[k, j] = values[j]
    end
    return tm
end
set_tracer!(tm::TracerMesh, name::Symbol, values::AbstractVector) =
    set_tracer!(tm, tracer_index(tm, name), values)

function set_tracer!(tm::TracerMesh, k::Integer, f::Function)
    N = n_tracer_segments(tm)
    @assert 1 ≤ k ≤ n_tracer_fields(tm) "tracer index $k out of range"
    @inbounds for j in 1:N
        m_j = tm.fluid.segments[j].m
        tm.tracers[k, j] = f(m_j)
    end
    return tm
end
set_tracer!(tm::TracerMesh, name::Symbol, f::Function) =
    set_tracer!(tm, tracer_index(tm, name), f)

"""
    add_tracer!(tm::TracerMesh, name::Symbol, values_or_f) -> TracerMesh

In-place extension is not supported (the matrix layout is fixed
at construction); instead, return a *new* `TracerMesh` with one
additional row whose values come from `values_or_f`. The fluid
mesh is shared (same reference). Existing rows are copied.

Provided as a convenience; for performance, prefer constructing
the `TracerMesh` with the full `n_tracers` upfront.
"""
function add_tracer!(tm::TracerMesh{T}, name::Symbol, values::AbstractVector) where {T}
    @assert !(name in tm.names) "tracer name $name already exists"
    N = n_tracer_segments(tm)
    @assert length(values) == N
    K = n_tracer_fields(tm) + 1
    new_mat = zeros(T, K, N)
    new_mat[1:end-1, :] = tm.tracers
    @inbounds for j in 1:N
        new_mat[K, j] = T(values[j])
    end
    new_names = vcat(tm.names, name)
    return TracerMesh{T, typeof(tm.fluid)}(tm.fluid, new_mat, new_names)
end
add_tracer!(tm::TracerMesh, name::Symbol, f::Function) = begin
    N = n_tracer_segments(tm)
    vals = [f(tm.fluid.segments[j].m) for j in 1:N]
    add_tracer!(tm, name, vals)
end

"""
    advect_tracers!(tm::TracerMesh, dt::Real) -> TracerMesh

Step the tracer fields forward in time by `dt`.

In a pure-Lagrangian frame (no remap), tracers don't change: each
parcel keeps its value forever. Concretely this function is a
**no-op** — the tracer matrix is never written to. This is the
"exact at the discrete level" property called out in §7 of the
methods paper. The caller is expected to invoke it in lockstep
with `det_step!` for symmetry with future Milestone-2 versions
that introduce remap diffusion through interval intersection on
the Eulerian mesh; in Milestone 1 (1D, no remap) the call has no
effect.

Returns `tm` (unchanged) for chaining.
"""
@inline function advect_tracers!(tm::TracerMesh, dt::Real)
    # Intentionally empty. Tracers in the Lagrangian frame are
    # constant on each parcel; the segment matrix stays bit-stable.
    return tm
end

"""
    tracer_at_position(tm::TracerMesh, x::Real, k::Integer = 1) -> Real

Sample tracer `k` at Eulerian (lab-frame) position `x`. Walks the
mesh segments cyclically until it finds the segment containing
`x` and returns the segment-centred tracer value. For a
discontinuous-IC tracer (e.g. step) this is the natural pixel-
constant interpretation.

Used by the Phase-11 figure / the cross-check vs the Eulerian
reference; not on the integration hot path.

Linear interpolation between neighbouring segments is intentionally
*not* applied: the tracer is a label of the parcel, and the
segment's value applies across the segment's full Eulerian
extent. (This matches the methods-paper view that the tracer is a
function of the parcel label, evaluated piecewise-constant per
segment in any Eulerian projection.)
"""
function tracer_at_position(tm::TracerMesh{T}, x::Real, k::Integer = 1) where {T}
    @assert 1 ≤ k ≤ n_tracer_fields(tm) "tracer index $k out of range"
    mesh = tm.fluid
    N = n_segments(mesh)
    L = mesh.L_box
    # Wrap x into the canonical period [x_left_min, x_left_min + L).
    x_left_1 = mesh.segments[1].state.x
    xq = mod(x - x_left_1, L) + x_left_1
    @inbounds for j in 1:N
        seg = mesh.segments[j]
        x_l = seg.state.x
        j_r = j == N ? 1 : j + 1
        wrap = (j == N) ? L : zero(T)
        x_r = mesh.segments[j_r].state.x + wrap
        # If the segment straddles the period boundary, also accept
        # the wrap-around branch; safe under the canonical wrap above.
        if x_l ≤ xq < x_r
            return tm.tracers[k, j]
        end
    end
    # Fallback: numerical edge case at xq exactly equal to x_left_N+L.
    return tm.tracers[k, N]
end

# ──────────────────────────────────────────────────────────────────────
# Reference Eulerian advection (for the B.5 fidelity comparison)
# ──────────────────────────────────────────────────────────────────────

"""
    eulerian_upwind_advect!(T_field, u_field, dx, dt; periodic = true)
        -> T_field

Reference advection of a cell-centred passive scalar `T_field` on a
*uniform* Eulerian mesh of spacing `dx`, by a cell-centred velocity
`u_field`, with a single forward-Euler upwind step of size `dt`.
This is the canonical first-order scheme that the variational
tracer is benchmarked against.

The scheme:
    T^{n+1}_j = T^n_j − dt * max(u_j, 0) * (T^n_j − T^n_{j-1}) / dx
                       − dt * min(u_j, 0) * (T^n_{j+1} − T^n_j) / dx

For periodic boundaries the indices wrap modulo `N`. The CFL
condition `|u| dt / dx ≤ 1` must be respected by the caller.

Modifies `T_field` in place; returns it.
"""
function eulerian_upwind_advect!(T_field::AbstractVector{R},
                                 u_field::AbstractVector{R},
                                 dx::Real, dt::Real;
                                 periodic::Bool = true) where {R<:Real}
    N = length(T_field)
    @assert length(u_field) == N
    Tn = copy(T_field)
    @inbounds for j in 1:N
        u_j = u_field[j]
        if periodic
            jl = j == 1 ? N : j - 1
            jr = j == N ? 1 : j + 1
        else
            jl = max(j - 1, 1)
            jr = min(j + 1, N)
        end
        flux_lo = max(u_j, zero(R)) * (Tn[j]  - Tn[jl]) / dx
        flux_hi = min(u_j, zero(R)) * (Tn[jr] - Tn[j])  / dx
        T_field[j] = Tn[j] - dt * (flux_lo + flux_hi)
    end
    return T_field
end

"""
    interface_width(T_field, x_centers; threshold_lo = 0.05, threshold_hi = 0.95)
        -> Float64

Measure the spatial width of a tracer interface, defined as the
*span of the transition region* — the contiguous range of cells
where the (min-max-normalised) tracer value lies strictly between
`threshold_lo` and `threshold_hi`. For a perfectly sharp step
T = 0 → 1 the transition region is *empty* (no cell sits in
[0.05, 0.95]) and the function returns `0.0`. For a smeared
profile the width is the distance between the first and last
cell in the transition band; in cell-count units this is
`width / dx`.

The normalisation makes the metric scale-invariant: a step from
0 to 1 and a step from 0 to 1000 give identical widths.

Returns 0.0 if the tracer is constant or if no cell lies inside
the transition band.

Used by the Phase-11 fidelity comparison.
"""
function interface_width(T_field::AbstractVector{R},
                         x_centers::AbstractVector{R};
                         threshold_lo::Real = 0.05,
                         threshold_hi::Real = 0.95) where {R<:Real}
    @assert length(T_field) == length(x_centers)
    Tmin, Tmax = extrema(T_field)
    if Tmax - Tmin < 1e-14
        return zero(R)
    end
    # Normalise to [0, 1] for threshold use.
    Tnorm = (T_field .- Tmin) ./ (Tmax - Tmin)
    # Cells inside the transition band.
    in_band = findall(t -> threshold_lo < t < threshold_hi, Tnorm)
    isempty(in_band) && return zero(R)
    j_lo = first(in_band)
    j_hi = last(in_band)
    if length(x_centers) ≥ 2
        dx_est = abs(x_centers[2] - x_centers[1])
    else
        dx_est = zero(R)
    end
    # Width = span of the transition cells, plus one cell width to
    # account for the fact that even one in-band cell represents
    # a smeared region of width dx (rather than a delta).
    return abs(x_centers[j_hi] - x_centers[j_lo]) + dx_est
end
