"""
Calibration loader for the Kramers-Moyal LES noise model.

Reads `py-1d/data/noise_model_params.npz` via `NPZ.jl` and returns a
`NamedTuple` of Float64 scalars. The fields and their physical meaning
(see `py-1d/dfmm/closure/noise_model.py` and the methods paper §9.6):

* `C_A`      — drift coefficient in `delta(rho u) = C_A * rho * du/dx * dt`.
               Production value ~0.336.
* `C_A_err`  — 1-sigma uncertainty on `C_A` from the production fit.
* `C_B`      — noise amplitude in
               `delta(rho u) = C_B * rho * sqrt(max(-du/dx, 0) * dt) * eta`.
               Production value ~0.548.
* `C_B_err`  — 1-sigma uncertainty on `C_B`.
* `floor`    — chaotic-divergence variance floor (subtracted before the
               C_B fit; sets a noise-detectability lower bound).
* `kurt`     — empirical residual kurtosis from the production calibration
               (~3.45; Laplace = 3, Gaussian = 0). Variance-gamma shape
               parameter `lambda` follows from `kurt = 3 + 3/lambda`.
* `skew`     — empirical residual skewness (~-0.22; signed asymmetry of
               the noise residuals).
* `dt_save`  — output cadence used during calibration (s; here 0.01).

Note. The `npz` file does not store `lambda` or `theta` directly; those
are derived in Phase 8 of the milestone plan from the `kurt`/`skew`
moments. They are documented here so Track A/D consumers know to derive
them rather than read them.
"""

using NPZ

"""
    load_noise_model([path]) -> NamedTuple

Load the production calibration scalars. Default `path` is the
checked-in `py-1d/data/noise_model_params.npz`. All fields are returned
as `Float64`; a missing field produces a `KeyError` so the caller knows
the file format has changed.

Returned fields:
    C_A, C_A_err, C_B, C_B_err, floor, kurt, skew, dt_save
"""
function load_noise_model(
    path::AbstractString = joinpath(@__DIR__, "..", "py-1d", "data",
                                    "noise_model_params.npz"),
)::NamedTuple
    raw = NPZ.npzread(path)
    keys_expected = ("C_A", "C_A_err", "C_B", "C_B_err",
                     "floor", "kurt", "skew", "dt_save")
    vals = ntuple(i -> _scalar_float64(raw, keys_expected[i]), length(keys_expected))
    return NamedTuple{Symbol.(keys_expected)}(vals)
end

# `npzread` returns 0-d Arrays for scalar entries; collapse to Float64.
function _scalar_float64(d::Dict, key::AbstractString)::Float64
    if !haskey(d, key)
        throw(KeyError(key))
    end
    v = d[key]
    if v isa AbstractArray
        return Float64(only(v))
    end
    return Float64(v)
end
