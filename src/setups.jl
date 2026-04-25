"""
    setups

Tier-A initial-condition factories for the 1D Julia dfmm port.

Each factory mirrors the corresponding py-1d setup
(`py-1d/dfmm/setups/{sod, sine, shock, wavepool}.py` plus the inline ICs
of `py-1d/experiments/05_two_fluid_dust_gas.py` and `07_eion_equilibration.py`)
and returns primitives `(rho, u, P, ...)` together with the Cholesky-sector
seed fields `alpha_init`, `beta_init`, and the rank diagnostic `gamma_init`.
The naming convention follows the methods paper:

  - `rho`         — mass density (Float64, length N)
  - `u`           — velocity     (Float64, length N)
  - `P`           — isotropic pressure used to seed both Pxx and P_perp
  - `Pxx`, `Pp`   — diagonal pressure components (set equal to P at t=0)
  - `Q`           — heat flux (zero at t=0 for every Tier-A IC)
  - `alpha_init`  — Cholesky alpha-field seed (sigma_x0 in py-1d)
  - `beta_init`   — Cholesky beta-field seed (= 0 for every Tier-A IC)
  - `gamma_init`  — sqrt(max(P/rho - beta^2, 0)); the rank-loss diagnostic
  - `x`           — cell centers on [0, 1] (periodic-style: x_i = (i + 0.5)/N)
  - `params`      — NamedTuple of integration knobs passed to py-1d
                    (`tau`, `cfl`, `t_end`, `bc`, `save_times`, ...) so the
                    Julia-side test and the py-1d golden run share one source
                    of truth.

The factory does not assemble py-1d's 8-field conserved state; that is
py-1d's internal layout. The Julia integrator (Phase 5+) will compute its
own conserved state from these primitives. The HDF5 goldens
(see `scripts/make_goldens.jl`) record py-1d's full 8-field history.

For the two-fluid setups the factory returns the per-species primitives
keyed `_d`/`_g` (dust/gas) or `_e`/`_i` (electron/ion); the `gamma`-EOS is
isotropic Maxwellian so `Pxx_<sp> = Pp_<sp> = P_<sp>` at t = 0.

# References
- methods paper §10.2 (Tier A list)
- `reference/MILESTONE_1_PLAN.md` Phases 5–13
- py-1d/dfmm/setups/{sod,sine,shock,wavepool}.py
- py-1d/experiments/{05,07}_*.py
"""

# We keep this module self-contained (no dependencies on the rest of the
# Julia package yet) so test_setups.jl can include it before the rest of
# the integrator lands in Phases 5+.

using Random

"""
    setup_sod(; N=400, t_end=0.2, sigma_x0=0.02, tau=1e-3, cfl=0.3,
              bc="transmissive")

A.1 — Sod shock tube (methods paper §10.2 A.1, py-1d/dfmm/setups/sod.py).

Domain `[0,1]`, cell centers `x_i = (i + 0.5)/N`. Discontinuity at `x = 0.5`:

    (rho, u, P) = (1.0,   0, 1.0)   for x < 0.5
                  (0.125, 0, 0.1)   for x ≥ 0.5

The `tau` parameter selects the regime in py-1d's
`experiments/01_sod_validation.py` — `tau = 1e-3` is the canonical
"intermediate" run used as the regression target. Returns a NamedTuple
with fields described in the module docstring; conservation invariants:
zero net momentum at t = 0; total mass = `0.5*(1.0 + 0.125)`.
"""
function setup_sod(; N::Int=400, t_end::Float64=0.2, sigma_x0::Float64=0.02,
                   tau::Float64=1e-3, cfl::Float64=0.3,
                   bc::String="transmissive")
    x = collect((0:N-1) .+ 0.5) ./ N
    rho = ifelse.(x .< 0.5, 1.0, 0.125)
    u   = zeros(Float64, N)
    P   = ifelse.(x .< 0.5, 1.0, 0.1)
    Pxx = copy(P); Pp = copy(P)
    Q   = zeros(Float64, N)
    alpha_init = fill(sigma_x0, N)
    beta_init  = zeros(Float64, N)
    Svv = P ./ rho
    gamma_init = sqrt.(max.(Svv .- beta_init .^ 2, 0.0))
    return (
        name = "sod",
        x = x, rho = rho, u = u, P = P,
        Pxx = Pxx, Pp = Pp, Q = Q,
        alpha_init = alpha_init, beta_init = beta_init,
        gamma_init = gamma_init,
        params = (N=N, t_end=t_end, sigma_x0=sigma_x0, tau=tau, cfl=cfl,
                  bc=bc, save_times=Float64[t_end]),
    )
end


"""
    setup_cold_sinusoid(; N=400, t_end=0.6, A=1.0, T0=1e-3, rho0=1.0,
                        sigma_x0=0.02, tau=1e3, cfl=0.3, bc="periodic")

A.2 — Cold sinusoid across shell-crossing (methods paper §10.2 A.2,
py-1d/dfmm/setups/sine.py). Periodic domain `[0, 1]`:

    rho = rho0,   u = A sin(2π x),   P = rho0 * T0

The default `t_end = 0.6` and `A = 1.0` reproduce the caustic location
used in `py-1d/experiments/02_sine_shell_crossing.py`. The default
`tau = 1e3` puts the run in the collisionless regime where the
phase-space rank diagnostic `gamma_init → 0` is the central observable
of the Tier-B.2 cold-limit test (`MILESTONE_1_PLAN.md` Phase 3).

Conservation invariants at t = 0: net momentum exact integer multiple
of zero on the periodic domain (∑ u_i = 0 by symmetry of sin on the
half-shifted grid); total mass = rho0.
"""
function setup_cold_sinusoid(; N::Int=400, t_end::Float64=0.6,
                             A::Float64=1.0, T0::Float64=1e-3,
                             rho0::Float64=1.0, sigma_x0::Float64=0.02,
                             tau::Float64=1e3, cfl::Float64=0.3,
                             bc::String="periodic")
    x = collect((0:N-1) .+ 0.5) ./ N
    rho = fill(rho0, N)
    u   = A .* sin.(2π .* x)
    P   = fill(rho0 * T0, N)
    Pxx = copy(P); Pp = copy(P)
    Q   = zeros(Float64, N)
    alpha_init = fill(sigma_x0, N)
    beta_init  = zeros(Float64, N)
    Svv = P ./ rho
    gamma_init = sqrt.(max.(Svv .- beta_init .^ 2, 0.0))
    return (
        name = "cold_sinusoid",
        x = x, rho = rho, u = u, P = P,
        Pxx = Pxx, Pp = Pp, Q = Q,
        alpha_init = alpha_init, beta_init = beta_init,
        gamma_init = gamma_init,
        params = (N=N, t_end=t_end, A=A, T0=T0, rho0=rho0,
                  sigma_x0=sigma_x0, tau=tau, cfl=cfl, bc=bc,
                  save_times=Float64[t_end]),
    )
end


"""
    setup_steady_shock(; M1=3.0, N=400, t_end=3.0, sigma_x0=0.02,
                       tau=1e-3, cfl=0.3, rho1=1.0, P1=1.0,
                       gamma_eos=5/3, bc="inflow_outflow")

A.3 — Steady shock at upstream Mach `M1` (methods paper §10.2 A.3,
py-1d/dfmm/setups/shock.py + experiments/03_steady_shock.py).

Builds the post-Rankine–Hugoniot two-state IC at `x = 0.5`, with the
upstream state seeded as constant inflow boundary on the left two cells.
The post-shock state is computed from the `gamma_eos = 5/3` Rankine–
Hugoniot relations (matching py-1d's `GAMMA = 5.0/3.0`).

`bc = "inflow_outflow"` is a marker; the actual boundary handling is
done by py-1d's `step_inflow_outflow` (which overwrites cells 0 and 1
each step with the upstream state). The Julia integrator must reproduce
this at the appropriate phase (Phase 7).

Returns the standard NamedTuple plus an `inflow` sub-NamedTuple holding
the upstream primitives `(rho, u, P, alpha, beta, M3)` for the boundary
condition.
"""
function setup_steady_shock(; M1::Float64=3.0, N::Int=400,
                            t_end::Float64=3.0, sigma_x0::Float64=0.02,
                            tau::Float64=1e-3, cfl::Float64=0.3,
                            rho1::Float64=1.0, P1::Float64=1.0,
                            gamma_eos::Float64=5.0/3.0,
                            bc::String="inflow_outflow")
    c_s1 = sqrt(gamma_eos * P1 / rho1)
    u1 = M1 * c_s1
    # Rankine-Hugoniot (matches py-1d/dfmm/setups/shock.py).
    rho_ratio = (gamma_eos + 1) * M1^2 / ((gamma_eos - 1) * M1^2 + 2)
    P_ratio   = (2 * gamma_eos * M1^2 - (gamma_eos - 1)) / (gamma_eos + 1)
    rho2 = rho1 * rho_ratio
    u2   = u1 / rho_ratio
    P2   = P1 * P_ratio

    # Mirror py-1d's grid: linspace(0, 1, N), endpoint=True (note the
    # difference from the periodic setups, which use endpoint=False +
    # half-cell shift).
    x = collect(range(0.0, 1.0; length=N))
    rho = ifelse.(x .< 0.5, rho1, rho2)
    u   = ifelse.(x .< 0.5, u1,   u2)
    P   = ifelse.(x .< 0.5, P1,   P2)
    Pxx = copy(P); Pp = copy(P)
    Q   = zeros(Float64, N)
    alpha_init = fill(sigma_x0, N)
    beta_init  = zeros(Float64, N)
    Svv = P ./ rho
    gamma_init = sqrt.(max.(Svv .- beta_init .^ 2, 0.0))

    inflow = (rho=rho1, u=u1, P=P1, alpha=sigma_x0, beta=0.0,
              M3=rho1 * u1^3 + 3.0 * u1 * P1)

    return (
        name = "steady_shock",
        x = x, rho = rho, u = u, P = P,
        Pxx = Pxx, Pp = Pp, Q = Q,
        alpha_init = alpha_init, beta_init = beta_init,
        gamma_init = gamma_init,
        rho1 = rho1, u1 = u1, P1 = P1,
        rho2 = rho2, u2 = u2, P2 = P2,
        M1 = M1, gamma_eos = gamma_eos,
        inflow = inflow,
        params = (N=N, M1=M1, t_end=t_end, sigma_x0=sigma_x0, tau=tau,
                  cfl=cfl, rho1=rho1, P1=P1, gamma_eos=gamma_eos, bc=bc,
                  save_times=Float64[t_end]),
    )
end


"""
    setup_kmles_wavepool(; N=256, t_end=5.0, u0=1.0, P0=0.1, K_max=16,
                         alpha_pl=5/6, rho0=1.0, sigma_x0=0.02, seed=42,
                         tau=1e-3, cfl=0.3, bc="periodic", n_snaps=250)

A.4 — KM-LES wave-pool (methods paper §10.2 A.4,
py-1d/dfmm/setups/wavepool.py + experiments/10_kmles_wavepool.py).

Broadband random-phase velocity field on periodic `[0, 1]`:

    u(x, 0) = u0 * sum_{k=1}^{K_max} A_k cos(2π k x + φ_k),
        A_k = k^{-alpha_pl}

normalized so that the RMS of the resulting `u` equals `u0`. Phases
`φ_k` are drawn from `numpy.random.default_rng(seed).uniform(0, 2π)`,
which we reproduce bit-exactly in `setup_kmles_wavepool` *only if* the
caller passes the same `seed` to py-1d (the Julia factory generates the
phases via numpy through the IO path; see `scripts/make_goldens.jl`).

The Julia factory itself uses Julia's `MersenneTwister(seed)` to draw
the phases. **Bit-equality with py-1d at t = 0 is therefore not
expected** — both phase distributions are uniform on [0, 2π) but the
RNG streams differ. The IC sanity tests (`test/test_setups.jl`) only
assert spectral shape and RMS; full bit-equality at t = 0 is asserted
in the regression tests by reading the t = 0 snapshot from py-1d's
golden HDF5.

`save_times` in `params` is `range(t_end/n_snaps, t_end, n_snaps)`
matching py-1d's `experiments/10_kmles_wavepool.py` time-series
cadence.
"""
function setup_kmles_wavepool(; N::Int=256, t_end::Float64=5.0,
                              u0::Float64=1.0, P0::Float64=0.1,
                              K_max::Int=16, alpha_pl::Float64=5.0/6.0,
                              rho0::Float64=1.0, sigma_x0::Float64=0.02,
                              seed::Int=42, tau::Float64=1e-3,
                              cfl::Float64=0.3, bc::String="periodic",
                              n_snaps::Int=250)
    rng = MersenneTwister(seed)
    phases = 2π .* rand(rng, K_max)

    x = collect((0:N-1) .+ 0.5) ./ N
    k = 1:K_max
    Ak = (1.0 ./ k) .^ alpha_pl
    u = zeros(Float64, N)
    @inbounds for (ki, Aki, phii) in zip(k, Ak, phases)
        u .+= Aki .* cos.(2π * ki .* x .+ phii)
    end
    rms = sqrt(sum(u .^ 2) / N)
    u .*= u0 / rms

    rho = fill(rho0, N)
    P   = fill(P0, N)
    Pxx = copy(P); Pp = copy(P)
    Q   = zeros(Float64, N)
    alpha_init = fill(sigma_x0, N)
    beta_init  = zeros(Float64, N)
    Svv = P ./ rho
    gamma_init = sqrt.(max.(Svv .- beta_init .^ 2, 0.0))

    save_times = collect(range(t_end / n_snaps, t_end; length=n_snaps))

    return (
        name = "wavepool",
        x = x, rho = rho, u = u, P = P,
        Pxx = Pxx, Pp = Pp, Q = Q,
        alpha_init = alpha_init, beta_init = beta_init,
        gamma_init = gamma_init,
        phases = phases, K_max = K_max, alpha_pl = alpha_pl,
        params = (N=N, t_end=t_end, u0=u0, P0=P0, K_max=K_max,
                  alpha_pl=alpha_pl, rho0=rho0, sigma_x0=sigma_x0,
                  seed=seed, tau=tau, cfl=cfl, bc=bc,
                  save_times=save_times, n_snaps=n_snaps),
    )
end


"""
    setup_dust_in_gas(; N=400, t_end=0.5, A=1.0, T0_gas=1e-3, T0_dust=1e-5,
                      dust_to_gas=0.1, grain_radius=1e-3,
                      tau_dd=1e3, tau_gg=1e-5, sigma_x0=0.02,
                      cfl=0.3, bc="periodic", n_snaps=10)

A.5 — Two-fluid dust-in-gas cold sinusoid (methods paper §10.2 A.5,
py-1d/experiments/05_two_fluid_dust_gas.py — built inline since there is
no dedicated `py-1d/dfmm/setups/dust_in_gas.py`).

Two species share the same `A sin(2π x)` initial velocity on `[0, 1]`;
species **A = dust** (`rho = dust_to_gas`, hot self-`tau`, large
self-`tau` for collisionless dust dynamics) and species **B = gas**
(`rho = 1`, small self-`tau` for Eulerian hydro). The cross-coupling
strength is set by `grain_radius` via the Epstein drag kernel
`tf.kernel_epstein(rho, T_gas, grain_radius, m_grain=1.0)` evaluated by
py-1d at run time.

Default `grain_radius = 1e-3` is the "intermediate" stopping-time
regime in `experiments/05_two_fluid_dust_gas.py`. Override to scan the
three regimes (`1e-5` tightly coupled, `1e-3` intermediate, `1e-1`
decoupled).

Returns per-species primitives keyed `_d` / `_g`.
"""
function setup_dust_in_gas(; N::Int=400, t_end::Float64=0.5,
                           A::Float64=1.0, T0_gas::Float64=1e-3,
                           T0_dust::Float64=1e-5,
                           dust_to_gas::Float64=0.1,
                           grain_radius::Float64=1e-3,
                           tau_dd::Float64=1e3, tau_gg::Float64=1e-5,
                           sigma_x0::Float64=0.02, cfl::Float64=0.3,
                           bc::String="periodic", n_snaps::Int=10)
    x = collect((0:N-1) .+ 0.5) ./ N
    # Dust (species A)
    rho_d = fill(dust_to_gas, N)
    u_d   = A .* sin.(2π .* x)
    P_d   = rho_d .* T0_dust
    # Gas (species B)
    rho_g = fill(1.0, N)
    u_g   = A .* sin.(2π .* x)
    P_g   = rho_g .* T0_gas

    Pxx_d = copy(P_d); Pp_d = copy(P_d)
    Pxx_g = copy(P_g); Pp_g = copy(P_g)
    Q_d   = zeros(Float64, N); Q_g = zeros(Float64, N)

    alpha_init_d = fill(sigma_x0, N); beta_init_d = zeros(Float64, N)
    alpha_init_g = fill(sigma_x0, N); beta_init_g = zeros(Float64, N)
    Svv_d = P_d ./ rho_d; Svv_g = P_g ./ rho_g
    gamma_init_d = sqrt.(max.(Svv_d .- beta_init_d .^ 2, 0.0))
    gamma_init_g = sqrt.(max.(Svv_g .- beta_init_g .^ 2, 0.0))

    save_times = collect(range(t_end / n_snaps, t_end; length=n_snaps))

    return (
        name = "dust_in_gas",
        x = x,
        rho_d = rho_d, u_d = u_d, P_d = P_d, Pxx_d = Pxx_d, Pp_d = Pp_d,
        Q_d = Q_d, alpha_init_d = alpha_init_d, beta_init_d = beta_init_d,
        gamma_init_d = gamma_init_d,
        rho_g = rho_g, u_g = u_g, P_g = P_g, Pxx_g = Pxx_g, Pp_g = Pp_g,
        Q_g = Q_g, alpha_init_g = alpha_init_g, beta_init_g = beta_init_g,
        gamma_init_g = gamma_init_g,
        params = (N=N, t_end=t_end, A=A, T0_gas=T0_gas, T0_dust=T0_dust,
                  dust_to_gas=dust_to_gas, grain_radius=grain_radius,
                  tau_dd=tau_dd, tau_gg=tau_gg, sigma_x0=sigma_x0, cfl=cfl,
                  bc=bc, save_times=save_times, n_snaps=n_snaps),
    )
end


"""
    setup_eion(; N=32, t_end=10.0, m_e=0.01, m_i=1.0, Z_e=1.0, Z_i=1.0,
               lnLambda=10.0, Te0=2.0, Ti0=0.5, sigma_x0=0.02,
               n_snaps=200, bc="periodic")

A.6 — Coulomb plasma equilibration (methods paper §10.2 A.6,
py-1d/experiments/07_eion_equilibration.py — built inline).

Spatially uniform two-temperature plasma. Electron species **A**:
`rho_e = m_e`, `u_e = 0`, `P_e = rho_e * Te0 / m_e` (so that the
temperature in *energy* units is `Te0`). Ion species **B**: `rho_i = m_i`,
`u_i = 0`, `P_i = rho_i * Ti0 / m_i`. No spatial gradients ⇒ no hydro
signal speed; the only dynamics are the Coulomb cross-coupling
collision kernel (`tf.kernel_coulomb`, py-1d).

The default `n_snaps = 200` matches py-1d's history sampling cadence.
The default `N = 32` is the smallest grid that exercises the periodic
discretization; the physics is uniform so any `N ≥ 1` would do.
"""
function setup_eion(; N::Int=32, t_end::Float64=10.0,
                    m_e::Float64=0.01, m_i::Float64=1.0,
                    Z_e::Float64=1.0, Z_i::Float64=1.0,
                    lnLambda::Float64=10.0,
                    Te0::Float64=2.0, Ti0::Float64=0.5,
                    sigma_x0::Float64=0.02, n_snaps::Int=200,
                    bc::String="periodic")
    x = collect((0:N-1) .+ 0.5) ./ N
    # Electrons (species A)
    rho_e = fill(m_e, N)
    u_e   = zeros(Float64, N)
    P_e   = rho_e .* Te0 ./ m_e
    # Ions (species B)
    rho_i = fill(m_i, N)
    u_i   = zeros(Float64, N)
    P_i   = rho_i .* Ti0 ./ m_i

    Pxx_e = copy(P_e); Pp_e = copy(P_e)
    Pxx_i = copy(P_i); Pp_i = copy(P_i)
    Q_e   = zeros(Float64, N); Q_i = zeros(Float64, N)

    alpha_init_e = fill(sigma_x0, N); beta_init_e = zeros(Float64, N)
    alpha_init_i = fill(sigma_x0, N); beta_init_i = zeros(Float64, N)
    Svv_e = P_e ./ rho_e; Svv_i = P_i ./ rho_i
    gamma_init_e = sqrt.(max.(Svv_e .- beta_init_e .^ 2, 0.0))
    gamma_init_i = sqrt.(max.(Svv_i .- beta_init_i .^ 2, 0.0))

    save_times = collect(range(t_end / n_snaps, t_end; length=n_snaps))

    return (
        name = "eion",
        x = x,
        rho_e = rho_e, u_e = u_e, P_e = P_e, Pxx_e = Pxx_e, Pp_e = Pp_e,
        Q_e = Q_e, alpha_init_e = alpha_init_e, beta_init_e = beta_init_e,
        gamma_init_e = gamma_init_e,
        rho_i = rho_i, u_i = u_i, P_i = P_i, Pxx_i = Pxx_i, Pp_i = Pp_i,
        Q_i = Q_i, alpha_init_i = alpha_init_i, beta_init_i = beta_init_i,
        gamma_init_i = gamma_init_i,
        Te0 = Te0, Ti0 = Ti0,
        params = (N=N, t_end=t_end, m_e=m_e, m_i=m_i, Z_e=Z_e, Z_i=Z_i,
                  lnLambda=lnLambda, Te0=Te0, Ti0=Ti0, sigma_x0=sigma_x0,
                  n_snaps=n_snaps, bc=bc, save_times=save_times),
    )
end
