# Tier-A regression-target HDF5 schema

This directory holds the six HDF5 "golden" snapshots used as the
regression target for the dfmm-2d Julia integrator. Each file is the
output of running py-1d (`py-1d/dfmm/...`) at a fixed parameter set
defined by the matching Julia factory in `src/setups.jl`.

## Generation

The goldens are produced by the Julia driver `scripts/make_goldens.jl`,
which shells out to the (only) Python helper added by Track B,
`py-1d/scripts/dump_golden.py`. The Julia setup factories
(`src/setups.jl`) are the single source of truth for parameter
defaults; the driver pulls each factory's `.params` NamedTuple,
encodes it as JSON, and passes it to the Python dumper.

```
make goldens                                        # all six
julia --project=. scripts/make_goldens.jl --target sod   # one
```

## Size policy

Goldens are committed to the repo iff their on-disk size is at most
**5 MiB**. The driver tracks this rule:

- `≤ 5 MiB`: the `.h5` file is committed verbatim.
- `>  5 MiB`: the `.h5` is *not* committed (matched by `.gitignore`);
  the driver writes a sibling `<name>.h5.regen` stub instead, which
  contains the exact CLI invocation needed to rebuild it. CI is
  expected to run `make goldens` before regression tests.

At present (Phase 5 entry) all six goldens fit under the 5 MiB budget
at default resolutions (largest is `wavepool.h5` ≈ 1.8 MiB). The
`.regen` mechanism is in place for future resolution bumps.

## File-level layout (single-fluid: sod, cold_sinusoid, steady_shock, wavepool)

Numpy uses C-order `[n_snap+1, N]` for 2D arrays; HDF5.jl reads them
as Julia column-major `[N, n_snap+1]`. Snapshot index 1 (Julia 1-based)
or 0 (numpy 0-based) is always the t = 0 initial condition.

```
/                                  attrs:
                                     nsteps : int64  (number of HLL steps taken
                                                      to reach t_end)
/params                            group, attrs:
    target            : str         one of {sod, cold_sinusoid,
                                            steady_shock, wavepool,
                                            dust_in_gas, eion}
    dfmm_git_sha      : str         git rev-parse HEAD at generation
    py1d_dfmm_version : str         dfmm.__version__ from py-1d
    N                 : int64       grid resolution
    t_end             : float64     final time
    tau               : float64     BGK relaxation time
    cfl               : float64     CFL number
    sigma_x0          : float64     Cholesky alpha-field seed
    bc                : str         "periodic" | "transmissive" |
                                    "inflow_outflow"
    save_times        : float64[*]  output cadence requested
    ... (per-setup: A, T0, M1, K_max, seed, ...)

/grid/x                            float64[N]
                                     cell centers; for periodic setups
                                     x_i = (i + 0.5)/N, for the
                                     inflow-outflow shock
                                     x = linspace(0, 1, N).
/timesteps                         float64[n_snap+1]
                                     actual snapshot times,
                                     timesteps[1] = 0.0,
                                     timesteps[end] ≈ t_end.
/fields/rho                        float64[n_snap+1, N]   density
/fields/u                          float64[n_snap+1, N]   velocity
/fields/Pxx                        float64[n_snap+1, N]   parallel pressure
/fields/Pp                         float64[n_snap+1, N]   perpendicular pressure
/fields/L1                         float64[n_snap+1, N]   Cholesky L_1 = x
                                                          mean (Lagrangian
                                                          tracer)
/fields/alpha                      float64[n_snap+1, N]   Cholesky α
/fields/beta                       float64[n_snap+1, N]   Cholesky β
/fields/M3                         float64[n_snap+1, N]   third moment
/fields/Q                          float64[n_snap+1, N]   heat flux
                                                          (Q = M3 - ρu^3
                                                          - 3 u P_xx)
/fields/gamma                      float64[n_snap+1, N]   Cholesky γ rank
                                                          diagnostic
                                                          (sqrt(P_xx/ρ - β²))
/fields/P_iso                      float64[n_snap+1, N]   (P_xx + 2 P_⊥)/3

/U                                 float64[n_snap+1, 8, N]
                                     Full conserved state for round-trip
                                     validation. Indexing matches
                                     py-1d's 8-field layout:
                                     0=ρ, 1=ρu, 2=E_xx=ρu²+P_xx,
                                     3=P_⊥, 4=ρL_1, 5=ρα, 6=ρβ,
                                     7=M_3.
```

## File-level layout (two-fluid: dust_in_gas, eion)

Two species share the grid; py-1d's 16-field state is split A=[0:8],
B=[8:16]. For `dust_in_gas` species A is dust, B is gas; for `eion`
species A is electrons, B is ions.

```
/                                  attrs:
                                     nsteps : int64
                                     m_A    : float64    species-A particle mass
                                     m_B    : float64    species-B particle mass
/params                            group, attrs (same scheme as single-fluid;
                                    extras: grain_radius, tau_dd, tau_gg,
                                    m_e, m_i, Z_e, Z_i, lnLambda, Te0,
                                    Ti0, dust_to_gas, T0_gas, T0_dust)
/grid/x                            float64[N]
/timesteps                         float64[n_snap+1]
/fields/A/{rho,u,Pxx,Pp,L1,alpha,beta,M3,Q,Piso,T}
                                   float64[n_snap+1, N]
                                   Per-species primitives. T = Piso * m / ρ
                                   is in energy units (matching py-1d's
                                   `tf.primitives` convention).
/fields/B/{rho,u,Pxx,Pp,L1,alpha,beta,M3,Q,Piso,T}
                                   float64[n_snap+1, N]
/U                                 float64[n_snap+1, 16, N]
                                   Full conserved state.
```

## Mapping back to the Julia factory

Each golden is paired with one factory in `src/setups.jl`:

| golden file                           | factory                  | Tier-A § |
|---------------------------------------|--------------------------|----------|
| `sod.h5`                              | `setup_sod`              | A.1      |
| `cold_sinusoid.h5`                    | `setup_cold_sinusoid`    | A.2      |
| `steady_shock.h5`                     | `setup_steady_shock`     | A.3      |
| `wavepool.h5`                         | `setup_kmles_wavepool`   | A.4      |
| `dust_in_gas.h5`                      | `setup_dust_in_gas`      | A.5      |
| `eion.h5`                             | `setup_eion`             | A.6      |

The `params` attribute group on every file is the verbatim JSON of
the factory's `params` NamedTuple; tests in
`test/test_setups.jl` assert that the factory output (rho, u, P, ...)
matches the t = 0 snapshot inside the golden to round-off, modulo
the documented exception for `wavepool` (random-phase RNG is numpy
on the py-1d side, MersenneTwister on the Julia side; the IC RMS is
asserted instead of bit-equality).

## Stability and refresh

When the py-1d code changes, regenerate goldens with:

```
make goldens
```

The `dfmm_git_sha` attribute on each file lets the regression tests
warn (or fail) when a golden is older than the current HEAD.
