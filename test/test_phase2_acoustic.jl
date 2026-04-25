# test_phase2_acoustic.jl
#
# Phase 2 — linearized acoustic wave dispersion check.
#
# Setup: uniform background ρ_0 = 1, M_vv,0 = 1, with a small density
# perturbation δρ/ρ_0 ~ 1e-4 sinusoidal in mass coordinate. The
# linearized momentum equation in Lagrangian form is
#   ρ_0 ∂_t u = -∂_x P_xx, with P_xx = ρ M_vv = ρ_0 M_vv,0 (1 + (Γ−1)δρ/ρ_0)
#                                           (using M_vv = ρ^(Γ−1) e^{s/cv})
#               ⇒ δP/δρ = Γ M_vv,0
#               ⇒ c_s² = Γ M_vv,0 = Γ at this normalisation.
# Hence c_s = √Γ = √(5/3) ≈ 1.2910 at unit ρ_0, M_vv,0.
#
# We measure the numerical phase speed by tracking the centre-of-mass
# of the (signed) δρ pattern at multiple times and computing how far
# it advances. Equivalently, we follow the position of the pattern
# zero-crossing.
#
# Tolerance: 5% on c_s at N = 64 (per the milestone plan); the
# midpoint-rule integrator is O(Δx²) accurate so we expect convergence
# at second order.

using Test
using dfmm

# Helper: numerical sound speed by fitting the time the wave packet
# advances by a known fraction of the box. We use a *standing wave*
# initial condition (δρ ∝ cos(kx), u = 0): the standing wave has a
# period T_s = 2π/(c_s k). Half a period later, δρ flips sign. Pick
# the time at which the sign flips (zero crossing of δρ_max) and
# read c_s = π/(k * t_zero). Implementation: monitor the
# spatial-amplitude of cos-projection over time, find first zero.

function _cos_projection(mesh, k, L)
    N = length(mesh.segments)
    ρ = [segment_density(mesh, j) for j in 1:N]
    # mass-coordinate phase: m_j (cell-center mass) = (j - 1/2) * Δm
    Δm = mesh.segments[1].Δm
    # but for k a wavenumber in *Lagrangian* m, project onto cos(k*m).
    s = 0.0
    for j in 1:N
        m_j = (j - 0.5) * Δm
        s += ρ[j] * cos(k * m_j)
    end
    return s / N
end

@testset "Phase 2: acoustic wave dispersion" begin
    N = 64
    L = 1.0
    ρ0 = 1.0
    s0 = 0.0       # M_vv,0 = J^(1-Γ) exp(s/cv) = 1 at J = 1
    α0 = 1.0
    β0 = 0.0
    Δx = L / N
    Δm_val = ρ0 * Δx
    Δm = fill(Δm_val, N)

    # Wavelength = L (one period in the box). k_m = 2π / total_mass.
    M_total = sum(Δm)
    k_m = 2π / M_total

    # Standing-wave IC: ρ(m) = ρ_0 (1 + ε cos(k_m * m)). Build positions
    # such that this density holds.
    ε = 1e-4
    # x_i is the left vertex of segment i; cell j has m ∈ [(j-1)Δm, j Δm]
    # and density ρ_j evaluated at m = (j-1/2)Δm. Then
    # x_{i+1} - x_i = Δm / ρ_i.  Build cumulative.
    positions = zeros(N)
    cum_x = 0.0
    for j in 1:N
        m_center = (j - 0.5) * Δm_val
        ρ_j = ρ0 * (1 + ε * cos(k_m * m_center))
        positions[j] = cum_x
        cum_x += Δm_val / ρ_j
    end
    L_box = cum_x   # actual physical box length given the density profile

    velocities = zeros(N)
    αs = fill(α0, N)
    βs = fill(β0, N)
    ss = fill(s0, N)

    mesh = Mesh1D(positions, velocities, αs, βs, ss;
                  Δm = Δm, L_box = L_box, periodic = true)

    # Analytical sound speed in (m, t) coordinates: c_s_m = c_s · ρ_0 = √Γ
    # (since k_x and k_m relate by k_x = ρ_0 k_m).
    Γ = 5.0/3.0
    c_s_x = sqrt(Γ)               # Eulerian sound speed
    ω_analytical = c_s_x * (2π / L_box)   # angular frequency at k_x = 2π/L

    dt = 1e-3
    N_steps = 1500   # several oscillation periods
    times = Float64[]
    cos_proj = Float64[]

    push!(times, 0.0)
    push!(cos_proj, _cos_projection(mesh, k_m, L_box))

    for n in 1:N_steps
        det_step!(mesh, dt)
        push!(times, n * dt)
        push!(cos_proj, _cos_projection(mesh, k_m, L_box))
    end

    # Fit a sinusoid to cos_proj(t) — take FFT and find the dominant
    # mode. Simpler: estimate frequency from zero-crossings of the
    # *signed deviation* (cos_proj - mean).
    deviation = cos_proj .- (sum(cos_proj) / length(cos_proj))
    crossings = Float64[]
    for i in 2:length(deviation)
        if deviation[i-1] * deviation[i] < 0
            # linear interpolation for zero-crossing time
            t_cross = times[i-1] + (times[i] - times[i-1]) *
                      abs(deviation[i-1]) / (abs(deviation[i-1]) + abs(deviation[i]))
            push!(crossings, t_cross)
        end
    end
    @test length(crossings) ≥ 4   # at least two oscillation periods

    # Average period from successive zero-crossings (should be T/2).
    half_periods = diff(crossings)
    T_period = 2 * (sum(half_periods) / length(half_periods))
    ω_num = 2π / T_period
    c_s_num = ω_num / (2π / L_box)

    @info "acoustic wave" c_s_num c_s_x ratio=c_s_num/c_s_x

    @test abs(c_s_num - c_s_x) / c_s_x < 0.05    # within 5%

    # Also assert basic invariants survived.
    @test isapprox(total_mass(mesh), sum(Δm); atol=100*eps())
    @test abs(total_momentum(mesh)) < 1e-10
end
