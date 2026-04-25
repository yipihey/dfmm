# segment.jl
#
# 1D Lagrangian mass-coordinate mesh.
#
# Phase 1 (now legacy) used a single-cell autonomous integration of the
# Cholesky sector. Phase 2 generalises this to a multi-segment mesh with
# bulk fields `x, u`, Cholesky `(α, β)`, and entropy `s` per segment, on
# a periodic Lagrangian-mass coordinate. Variable layout is documented
# in `reference/notes_phase2_discretization.md`:
#   * Positions and velocities at vertices (one per segment in the
#     periodic case).
#   * `α, β, s` cell-centered.
#
# `Segment` carries a `DetField` whose `(x, u)` fields are the
# *left-vertex* values of the segment. The mesh exposes accessor helpers
# `x_left`, `x_right`, `u_left`, `u_right` for explicit vertex-indexing.

"""
    Segment{T<:Real}

A single Lagrangian mass-coordinate cell.

Fields:
- `m::T`     — Lagrangian mass coordinate of the cell center (label;
               does not evolve).
- `Δm::T`    — Lagrangian mass extent of the cell. Fixed by construction.
- `state::DetField{T}` — full deterministic state (Phase 2). The legacy
   Phase-1 path uses `ChField`-only segments via `single_segment_mesh`;
   that path is preserved for backwards compatibility, but new code
   should construct via `Mesh1D(positions, ...)`.

Phase 1 path: `Segment{T}(m, Δm, ChField{T}(α, β, γ))` — used only by
`single_segment_mesh`. Phase 2 path: `Segment{T}(m, Δm, DetField{T}(...))`
constructed by `Mesh1D(...)`.
"""
mutable struct Segment{T<:Real,F}
    m::T
    Δm::T
    state::F
end

# Phase-1 convenience constructor (preserves the original signature).
Segment{T}(m::T, Δm::T, state::ChField{T}) where {T<:Real} =
    Segment{T,ChField{T}}(m, Δm, state)
Segment{T}(m::T, Δm::T, state::DetField{T}) where {T<:Real} =
    Segment{T,DetField{T}}(m, Δm, state)

"""
    Mesh1D{T<:Real,F}

A 1D periodic Lagrangian mesh of `N` segments. The Lagrangian box
length `L_box` is fixed at construction time and used to map periodic
position differences. Phase 1's single-segment use-case is supported
via `single_segment_mesh`, which produces an `F = ChField{T}` mesh.
Phase 2 constructs `F = DetField{T}` meshes via the keyword constructor.

Fields:
- `segments::Vector{Segment{T,F}}` — ordered list of segments.
- `L_box::T` — Lagrangian (Eulerian) box length used for periodic
  wrap-around of vertex positions. For a uniform initial mesh of
  length `L` this is `L`. (For Phase-1 single-segment meshes, set to
  `Δm` since the single-cell action is autonomous and `L_box` is
  unused.)
- `periodic::Bool` — `true` for periodic BC. Phase 2 only implements
  periodic BC; reflective/transmissive are flagged for later.
- `p_half::Vector{T}` — vertex half-step momenta `p_i^{n+1/2}`. Used
  by the variational-Verlet driver to carry the mid-step velocity
  forward between calls. Initialized from the user-supplied initial
  velocity at construction time and then mutated by `det_step!`.
"""
mutable struct Mesh1D{T<:Real,F}
    segments::Vector{Segment{T,F}}
    L_box::T
    periodic::Bool
    p_half::Vector{T}
end

# Backwards-compatible (Phase-1) constructor: positional Vector of
# segments only. Kept so existing Phase-1 tests still pass.
function Mesh1D(segments::Vector{Segment{T,F}}) where {T<:Real,F}
    L_box = sum(seg.Δm for seg in segments)
    p_half = zeros(T, length(segments))
    return Mesh1D{T,F}(segments, L_box, true, p_half)
end

"""
    n_segments(mesh::Mesh1D)

Number of segments on the mesh.
"""
n_segments(mesh::Mesh1D) = length(mesh.segments)

"""
    single_segment_mesh(α, β, γ; Δm = 1.0, m = 0.0)

Construct a one-segment `Mesh1D` for Phase 1 unit-cell tests.
Backwards-compatibility shim: produces a `ChField`-state mesh.
"""
function single_segment_mesh(α, β, γ; Δm = 1.0, m = 0.0)
    T = promote_type(typeof(α), typeof(β), typeof(γ), typeof(Δm), typeof(m))
    state = ChField{T}(T(α), T(β), T(γ))
    seg = Segment{T,ChField{T}}(T(m), T(Δm), state)
    return Mesh1D{T,ChField{T}}([seg], T(Δm), true, zeros(T, 1))
end

# ──────────────────────────────────────────────────────────────────────
# Phase-2 multi-segment constructor
# ──────────────────────────────────────────────────────────────────────

"""
    Mesh1D(positions, velocities, αs, βs, ss; Δm, periodic = true)

Construct a multi-segment Phase-2 mesh on a periodic Lagrangian box.

Arguments:
- `positions::AbstractVector` — vertex positions
  `x_i, i = 1, …, N`. The right boundary `x_{N+1}` is *implied* by
  periodicity: `x_{N+1} = x_1 + L_box`. The Lagrangian box length is
  `L_box = positions[end] + Δm[end]/ρ_avg − positions[1]`, computed
  internally as the cumulative segment length given that
  `J_j = (x_{j+1} − x_j)/Δm_j`. Concretely, the user supplies the
  `N` left-vertex positions and the `N` segment masses; the box length
  is set so that `x_{N+1} − x_1 = sum_j (right vertex offset)` — see
  the implementation.
- `velocities::AbstractVector` — vertex velocities `u_i, i = 1, …, N`.
- `αs, βs, ss::AbstractVector` — cell-centered `(α_j, β_j, s_j)`.

Keyword:
- `Δm::AbstractVector` (required) — per-segment Lagrangian mass.
  Length `N`.
- `periodic::Bool = true` — only periodic supported in Phase 2.

The Lagrangian box length is computed as
`L_box = sum_j (x_{j+1} − x_j)`; for a non-periodic mesh the user
must therefore pass `N+1` positions, which Phase 2 does not yet
support (raise an error).

Vertex masses `m̄_i = (Δm_{i-1} + Δm_i)/2` are formed cyclically.
The half-step momentum vector is initialised from the supplied
velocities: `p_half[i] = m̄_i * u_i`.
"""
function Mesh1D(
    positions::AbstractVector,
    velocities::AbstractVector,
    αs::AbstractVector,
    βs::AbstractVector,
    ss::AbstractVector;
    Δm::AbstractVector,
    L_box::Union{Real,Nothing} = nothing,
    periodic::Bool = true,
)
    N = length(positions)
    @assert length(velocities) == N
    @assert length(αs) == N
    @assert length(βs) == N
    @assert length(ss) == N
    @assert length(Δm) == N
    @assert N ≥ 2 "Phase-2 multi-segment mesh requires at least 2 segments"
    @assert periodic "Phase 2 implements periodic BC only"

    T = promote_type(eltype(positions), eltype(velocities),
                     eltype(αs), eltype(βs), eltype(ss),
                     eltype(Δm))

    # Box-length determination. The user supplies the N left-vertex
    # positions `x_1, …, x_N`; the right vertex of segment N (the
    # periodic wrap) is `x_1 + L_box`. We accept `L_box` as an
    # explicit keyword for clarity; if omitted we estimate it from the
    # forward differences (assumes the implicit `x_{N+1}` extrapolates
    # the leading-pair spacing).
    if L_box === nothing
        # Use mean of forward differences across the supplied N-1
        # vertex pairs to set the trailing segment length.
        Δx_avg = (positions[end] - positions[1]) / (N - 1)
        L = T(Δx_avg * N)
    else
        L = T(L_box)
    end

    segments = Vector{Segment{T,DetField{T}}}(undef, N)
    cum_m = zero(T)
    for j in 1:N
        m_center = cum_m + T(Δm[j]) / 2
        cum_m += T(Δm[j])
        det = DetField{T}(T(positions[j]), T(velocities[j]),
                          T(αs[j]), T(βs[j]), T(ss[j]))
        segments[j] = Segment{T,DetField{T}}(m_center, T(Δm[j]), det)
    end

    # Half-step momentum initialisation. `m̄_i = (Δm_{i-1} + Δm_i)/2`
    # cyclically.
    p_half = Vector{T}(undef, N)
    for i in 1:N
        i_left = i == 1 ? N : i - 1
        m̄ = (T(Δm[i_left]) + T(Δm[i])) / 2
        p_half[i] = m̄ * T(velocities[i])
    end

    return Mesh1D{T,DetField{T}}(segments, L, periodic, p_half)
end

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

"""
    vertex_mass(mesh::Mesh1D, i::Integer)

Vertex-associated mass `m̄_i = (Δm_{i-1} + Δm_i)/2` (cyclic in `i`).
Used as the lumped mass for the per-vertex kinetic-energy term in the
Phase-2 discrete action.
"""
function vertex_mass(mesh::Mesh1D, i::Integer)
    N = n_segments(mesh)
    i_left = i == 1 ? N : i - 1
    return (mesh.segments[i_left].Δm + mesh.segments[i].Δm) / 2
end

"""
    segment_length(mesh::Mesh1D, j::Integer)

Length of segment `j` in Eulerian (lab) coordinates,
`x_{j+1} − x_j`, with periodic wrap if `j == N`.
"""
function segment_length(mesh::Mesh1D, j::Integer)
    N = n_segments(mesh)
    x_left = mesh.segments[j].state.x
    if j < N
        x_right = mesh.segments[j+1].state.x
    else
        x_right = mesh.segments[1].state.x + mesh.L_box
    end
    return x_right - x_left
end

"""
    segment_density(mesh::Mesh1D, j::Integer)

Eulerian density `ρ_j = Δm_j / (x_{j+1} − x_j)` of segment `j`.
"""
segment_density(mesh::Mesh1D, j::Integer) =
    mesh.segments[j].Δm / segment_length(mesh, j)

"""
    total_mass(mesh::Mesh1D)

Sum of `Δm_j` over all segments. Phase-2 invariant: bit-stable across
timesteps because per-segment `Δm` is a label, not a state variable.
"""
total_mass(mesh::Mesh1D) = sum(seg.Δm for seg in mesh.segments)

"""
    total_momentum(mesh::Mesh1D)

Sum of vertex momenta `Σ_i m̄_i u_i`. Periodic-BC translation
invariance ⇒ exactly conserved by the discrete EL system.
"""
function total_momentum(mesh::Mesh1D)
    T = eltype(mesh.p_half)
    p = zero(T)
    @inbounds for i in 1:n_segments(mesh)
        p += mesh.p_half[i]
    end
    return p
end

"""
    total_energy(mesh::Mesh1D)

Total deterministic energy (kinetic + Cholesky-Hamiltonian)
`E = Σ_i ½ m̄_i u_i² + Σ_j Δm_j H_Ch_j`,
where `H_Ch = −½ α² γ² = −½ α² (M_vv − β²)`. Diagnostic only;
preservation is the topic of Phase 4 (B.1).
"""
function total_energy(mesh::Mesh1D)
    T = eltype(mesh.p_half)
    E = zero(T)
    N = n_segments(mesh)
    @inbounds for i in 1:N
        m̄ = vertex_mass(mesh, i)
        u_i = mesh.p_half[i] / m̄
        E += T(0.5) * m̄ * u_i^2
    end
    @inbounds for j in 1:N
        seg = mesh.segments[j]
        det = seg.state
        # γ² = M_vv − β². Use Track-C EOS to get M_vv from (J, s).
        ρ = segment_density(mesh, j)
        J = T(1) / ρ
        Mvv_j = Mvv(J, det.s)
        γ² = max(Mvv_j - det.β^2, zero(T))
        H_Ch = -T(0.5) * det.α^2 * γ²
        E += seg.Δm * H_Ch
    end
    return E
end
