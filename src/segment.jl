# segment.jl
#
# 1D Lagrangian mass-coordinate mesh (Phase 1 stub).
#
# Phase 1 only exercises a single-cell autonomous integration of the
# Cholesky sector with externally-supplied strain rate; the multi-segment
# infrastructure (segment adjacency, ordering, position evolution) lands
# in Phase 2 with the bulk-coupling work. The minimal type and
# constructor below exist so that downstream files have a stable
# import surface.

"""
    Segment{T<:Real}

A single Lagrangian mass-coordinate cell.

Fields:
- `m::T`     — Lagrangian mass coordinate of the cell center (label;
               does not evolve).
- `Δm::T`    — Lagrangian mass extent of the cell. Fixed by construction.
- `state::ChField{T}` — Cholesky-sector state in the cell.

In Phase 2 this struct grows position `x`, velocity `u`, entropy `s`,
plus the segment-graph adjacency. Phase 1 leaves those out.
"""
mutable struct Segment{T<:Real}
    m::T
    Δm::T
    state::ChField{T}
end

"""
    Mesh1D{T<:Real}

A 1D Lagrangian mesh. Phase 1 uses a single segment; the type exists
to make the API forward-compatible.
"""
struct Mesh1D{T<:Real}
    segments::Vector{Segment{T}}
end

"""
    single_segment_mesh(α, β, γ; Δm = 1.0, m = 0.0)

Construct a one-segment `Mesh1D` for Phase 1 unit-cell tests.
"""
function single_segment_mesh(α, β, γ; Δm = 1.0, m = 0.0)
    T = promote_type(typeof(α), typeof(β), typeof(γ), typeof(Δm), typeof(m))
    state = ChField{T}(T(α), T(β), T(γ))
    seg = Segment{T}(T(m), T(Δm), state)
    return Mesh1D{T}([seg])
end
