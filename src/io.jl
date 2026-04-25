"""
HDF5/JLD2 I/O wrappers for mesh-state snapshots.

Track A's canonical mesh state will eventually be a struct (see
`reference/MILESTONE_1_PLAN.md` "Code organization"). For Milestone 1
the contract here is intentionally loose: a state is a `NamedTuple` of
arrays plus optional scalar metadata. Both `HDF5.jl` and `JLD2.jl` are
declared dependencies in `Project.toml`; this module dispatches on file
extension.

Tentative schema (until `reference/golden/SCHEMA.md` lands):

    /                        # root
        attrs:
            schema_version :: String   # currently "0.1"
            created_by     :: String   # short tag, free text
        datasets (one per array field of the state NamedTuple):
            <field_name> :: Float64 array, shape preserved.
        scalar attributes (one per scalar field):
            <field_name> :: Float64 (or Int, String)

Compatible across both backends: `HDF5` writes datasets+attrs; `JLD2`
serializes the NamedTuple round-trippably and we keep the schema fields
inside it for parity. **Flag:** `reference/golden/SCHEMA.md` does not
yet exist; once Track B publishes it, reconcile this docstring.

Public API:
    save_state(path::AbstractString, state::NamedTuple; kwargs...)
    load_state(path::AbstractString)::NamedTuple

`save_state` writes `state` as datasets/attrs; round-trip equality
(`load_state(path) == state`) is asserted by the round-trip test.
"""

using HDF5
using JLD2

const SCHEMA_VERSION = "0.1"

"""
    save_state(path, state; created_by="dfmm.jl") -> path

Write a NamedTuple of arrays/scalars to `path`. Backend chosen by
extension:
* `.h5` / `.hdf5` ⇒ `HDF5.jl`.
* `.jld2`         ⇒ `JLD2.jl`.

Other extensions raise an `ArgumentError`.
"""
function save_state(path::AbstractString, state::NamedTuple;
                    created_by::AbstractString = "dfmm.jl")::String
    ext = lowercase(splitext(path)[2])
    if ext in (".h5", ".hdf5")
        _save_hdf5(path, state; created_by = created_by)
    elseif ext == ".jld2"
        _save_jld2(path, state; created_by = created_by)
    else
        throw(ArgumentError(
            "save_state: unsupported extension '$(ext)'. " *
            "Use .h5/.hdf5 or .jld2."))
    end
    return String(path)
end

"""
    load_state(path) -> NamedTuple

Inverse of `save_state`. Backend chosen by extension. Returns a
`NamedTuple` whose fields exactly mirror the original state (modulo
schema-version / created-by metadata, which is *not* injected into the
returned tuple unless the original state contained those keys).
"""
function load_state(path::AbstractString)::NamedTuple
    ext = lowercase(splitext(path)[2])
    if ext in (".h5", ".hdf5")
        return _load_hdf5(path)
    elseif ext == ".jld2"
        return _load_jld2(path)
    else
        throw(ArgumentError(
            "load_state: unsupported extension '$(ext)'. " *
            "Use .h5/.hdf5 or .jld2."))
    end
end

# --- HDF5 backend ------------------------------------------------------------

function _save_hdf5(path::AbstractString, state::NamedTuple;
                    created_by::AbstractString)
    HDF5.h5open(path, "w") do f
        HDF5.attrs(f)["schema_version"] = SCHEMA_VERSION
        HDF5.attrs(f)["created_by"] = created_by
        for (name, val) in pairs(state)
            key = String(name)
            if val isa AbstractArray
                f[key] = collect(val)  # materialize to a regular Array
            elseif val isa Union{Number, AbstractString, Bool}
                HDF5.attrs(f)[key] = val
            else
                throw(ArgumentError(
                    "save_state(HDF5): field '$key' has unsupported type " *
                    "$(typeof(val)); only arrays + scalars are supported."))
            end
        end
    end
    return path
end

function _load_hdf5(path::AbstractString)::NamedTuple
    HDF5.h5open(path, "r") do f
        # Datasets: arrays.
        names = Symbol[]
        values = Any[]
        for key in keys(f)
            push!(names, Symbol(key))
            push!(values, read(f[key]))
        end
        # Scalar attributes (skip schema metadata).
        for key in keys(HDF5.attrs(f))
            if key in ("schema_version", "created_by")
                continue
            end
            push!(names, Symbol(key))
            push!(values, HDF5.attrs(f)[key])
        end
        return NamedTuple{Tuple(names)}(Tuple(values))
    end
end

# --- JLD2 backend ------------------------------------------------------------

function _save_jld2(path::AbstractString, state::NamedTuple;
                    created_by::AbstractString)
    JLD2.jldopen(path, "w") do f
        f["__schema_version__"] = SCHEMA_VERSION
        f["__created_by__"] = created_by
        for (name, val) in pairs(state)
            f[String(name)] = val
        end
    end
    return path
end

function _load_jld2(path::AbstractString)::NamedTuple
    raw = JLD2.jldopen(path, "r") do f
        d = Dict{Symbol, Any}()
        for key in keys(f)
            startswith(key, "__") && continue
            d[Symbol(key)] = f[key]
        end
        d
    end
    syms = Tuple(sort(collect(keys(raw))))  # deterministic ordering
    return NamedTuple{syms}(Tuple(raw[s] for s in syms))
end
