# regression.jl
#
# Tier-A regression-test scaffold: load HDF5 goldens written by Track B's
# `scripts/make_goldens.jl` into Julia NamedTuples for comparison against
# the Julia integrator's output. The schema is documented in
# `reference/golden/SCHEMA.md`.
#
# In Milestone 1 the integrator-vs-golden comparison enters at Phase 5
# (Tier A.1 Sod). This file lands the loader plumbing now so the eventual
# regression tests are a thin wrapper over `load_tier_a_golden`.

using HDF5

"""
    GOLDEN_NAMES

The six Tier-A regression-target names, matching the file stems under
`reference/golden/` and the factories in `src/setups.jl`.
"""
const GOLDEN_NAMES = (:sod, :cold_sinusoid, :steady_shock, :wavepool,
                      :dust_in_gas, :eion)

"""
    SINGLE_FLUID_NAMES

Subset of `GOLDEN_NAMES` whose schema is the single-fluid layout
(`/fields/{rho,u,Pxx,Pp,L1,alpha,beta,M3,Q,gamma,P_iso}` plus `/U` of
shape `[n_snap+1, 8, N]`).
"""
const SINGLE_FLUID_NAMES = (:sod, :cold_sinusoid, :steady_shock, :wavepool)

"""
    TWO_FLUID_NAMES

Subset of `GOLDEN_NAMES` whose schema is the two-species layout
(`/fields/A/{...}`, `/fields/B/{...}` plus `/U` of shape
`[n_snap+1, 16, N]`).
"""
const TWO_FLUID_NAMES = (:dust_in_gas, :eion)

"""
    golden_path(name::Symbol; root = pkgdir(dfmm)) -> String

Resolve the on-disk path of the golden whose stem is `name`. Throws if
neither `<root>/reference/golden/<name>.h5` nor a `.h5.regen` stub
exists; the latter is reported with a helpful "run `make goldens`"
message.
"""
function golden_path(name::Symbol;
                     root::AbstractString = something(pkgdir(dfmm),
                                                      dirname(@__DIR__)))
    h5  = joinpath(root, "reference", "golden", string(name) * ".h5")
    isfile(h5) && return h5
    regen = h5 * ".regen"
    if isfile(regen)
        error("Golden $(name).h5 is not committed (size > 5 MiB); run " *
              "`make goldens` to regenerate. Stub: $regen")
    end
    error("No golden found for $(name); expected $h5")
end

"""
    load_tier_a_golden(name::Symbol; root = pkgdir(dfmm)) -> NamedTuple

Read a Tier-A golden HDF5 file into a NamedTuple. The shape follows the
schema documented in `reference/golden/SCHEMA.md`:

- `name :: Symbol` — the golden stem (`:sod`, `:cold_sinusoid`, ...).
- `params :: Dict{String,Any}` — the verbatim params group, all attrs.
- `x :: Vector{Float64}` — `/grid/x`.
- `t :: Vector{Float64}` — `/timesteps`.
- `fields :: NamedTuple` — single-fluid: per-field `Matrix{Float64}` of
  shape `(N, n_snap+1)` (Julia column-major). Two-fluid: nested
  `(A = (...), B = (...))` NamedTuples.
- `U :: Array{Float64, 3}` — `/U` reshaped to `(N, 8|16, n_snap+1)`.
- `nsteps :: Int64` — total HLL steps taken by py-1d.

Raises if the file is missing (see `golden_path`).
"""
function load_tier_a_golden(name::Symbol;
                            root::AbstractString = something(pkgdir(dfmm),
                                                             dirname(@__DIR__)))
    path = golden_path(name; root = root)
    h5open(path, "r") do f
        nsteps = Int64(attrs(f)["nsteps"])
        params = Dict{String,Any}(k => attrs(f["params"])[k]
                                  for k in keys(attrs(f["params"])))
        x = read(f["grid/x"])
        t = read(f["timesteps"])
        U = read(f["U"])

        fields = if name in SINGLE_FLUID_NAMES
            _read_single_fluid_fields(f["fields"])
        else
            (A = _read_single_fluid_fields(f["fields/A"]),
             B = _read_single_fluid_fields(f["fields/B"]))
        end

        return (; name, params, x, t, fields, U, nsteps)
    end
end

const _SF_FIELDS = (:rho, :u, :Pxx, :Pp, :L1, :alpha, :beta, :M3, :Q, :gamma, :P_iso)
const _TF_FIELDS = (:rho, :u, :Pxx, :Pp, :L1, :alpha, :beta, :M3, :Q, :Piso, :T)

function _read_single_fluid_fields(g::HDF5.Group)
    keys_set = Set(keys(g))
    pairs = Pair{Symbol,Matrix{Float64}}[]
    candidates = :T in keys_set ? _TF_FIELDS : _SF_FIELDS
    for k in candidates
        skey = String(k)
        if skey in keys_set
            push!(pairs, k => read(g[skey]))
        end
    end
    return NamedTuple(pairs)
end
