"""
CairoMakie plotting helpers.

Each helper builds and returns a `Figure` — *no* display, no save side
effects. The caller saves with `CairoMakie.save("foo.png", fig)`. This
keeps the helpers usable from headless test runs (CI) and lets the
caller compose multiple figures into a layout if desired.
"""

using CairoMakie

"""
    plot_profile(grid, field; ax_label="rho") -> Figure

Line plot of a 1D field over a 1D grid. `grid` and `field` must be
length-matched 1D `AbstractVector`s. The y-axis label is `ax_label`;
the x-axis label is fixed to "x".
"""
function plot_profile(grid::AbstractVector, field::AbstractVector;
                      ax_label::AbstractString = "rho")::Figure
    @assert length(grid) == length(field) "grid and field must be the same length"
    fig = Figure(size = (600, 400))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = ax_label)
    lines!(ax, collect(grid), collect(field))
    return fig
end

"""
    plot_phase_portrait(alpha_arr, beta_arr) -> Figure

Scatter of `(alpha, beta)` per cell — the projection of the
phase-space-fiber state onto the Cholesky-sector tangent plane.
Useful for visualising the unification manifold and the rank-collapse
caustic (where points pile up at `alpha > 0, beta = 0`).
"""
function plot_phase_portrait(alpha_arr::AbstractVector,
                             beta_arr::AbstractVector)::Figure
    @assert length(alpha_arr) == length(beta_arr) "alpha and beta must be same length"
    fig = Figure(size = (500, 500))
    ax = Axis(fig[1, 1], xlabel = "alpha", ylabel = "beta",
              title = "Cholesky phase portrait")
    scatter!(ax, collect(alpha_arr), collect(beta_arr), markersize = 4)
    return fig
end

"""
    plot_diagnostics(grid, gamma_arr, det_hess_arr) -> Figure

Two-panel diagnostic plot: gamma rank indicator over `grid` (top
panel) and `det Hess(H_Ch)` over `grid` (bottom panel). Both panels
share the x-axis. The bottom panel is signed (det should be ≤ 0 in
the warm interior of the realizability cone), so a linear y-axis is
used; users can post-process to log scale if desired.
"""
function plot_diagnostics(grid::AbstractVector,
                          gamma_arr::AbstractVector,
                          det_hess_arr::AbstractVector)::Figure
    @assert length(grid) == length(gamma_arr) == length(det_hess_arr) "grid, gamma_arr, det_hess_arr must all be same length"
    fig = Figure(size = (700, 500))
    ax1 = Axis(fig[1, 1], ylabel = "gamma_rank")
    lines!(ax1, collect(grid), collect(gamma_arr))
    ax2 = Axis(fig[2, 1], xlabel = "x", ylabel = "det Hess H_Ch")
    lines!(ax2, collect(grid), collect(det_hess_arr))
    linkxaxes!(ax1, ax2)
    hidexdecorations!(ax1, grid = false)
    return fig
end
