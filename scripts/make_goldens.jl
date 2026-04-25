#!/usr/bin/env julia
#
# make_goldens.jl — generate Tier-A regression-target HDF5 goldens
#
# Driver for the six Tier-A initial-conditions defined in
# `src/setups.jl`. Shells out to `py-1d/scripts/dump_golden.py` to run
# the py-1d numba-compiled integrator and write the resulting time
# series as HDF5.
#
# # Implementation choice: shell-out vs. PythonCall.jl
#
# This script uses the **shell-out** path. Rationale:
#
#   - PythonCall.jl + CondaPkg.jl would either need to install its
#     own Python and re-install py-1d's `pip install -e .` chain, or
#     be pointed at `py-1d/.venv/bin/python` via
#     `JULIA_CONDAPKG_BACKEND=Null` + `JULIA_PYTHONCALL_EXE`. Both add
#     Julia-side deps and a CI-time interop layer for what is
#     fundamentally a one-off batch run.
#   - Shell-out keeps the Julia driver dependency-free (only stdlib
#     `Printf` and the already-present `HDF5` are used at the Julia
#     level), while letting py-1d's existing `.venv` handle the
#     Python side.
#   - The brief permits this path provided we add exactly one helper
#     file under py-1d/, which is `py-1d/scripts/dump_golden.py`.
#     That file is the *only* edit Track B makes under py-1d/.
#
# The Julia setup factories (`src/setups.jl`) are kept the source of
# truth for parameter defaults; this driver pulls each factory's
# `params` NamedTuple, JSON-encodes it, and passes it to the Python
# dumper so a single edit to `setup_*` propagates to both sides.
#
# # Usage
#
#     julia --project=. scripts/make_goldens.jl                  # all six
#     julia --project=. scripts/make_goldens.jl --target sod     # one
#     julia --project=. scripts/make_goldens.jl --list           # list targets
#     make goldens                                                # via Makefile
#
# # Size budget
#
# Goldens larger than `MAX_INLINE_BYTES = 5 MB` are not committed; the
# driver writes a `.regen` stub instead containing the exact command
# to regenerate. CI is expected to run `make goldens` before the
# regression tests; large goldens are rebuilt on demand.
#
# # References
# - methods paper §10.2 (Tier-A list)
# - `reference/golden/SCHEMA.md` (HDF5 layout)
# - `reference/MILESTONE_1_PLAN.md` Phases 5–13

using Printf
using HDF5

# Resolve repo root robustly.
const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(REPO_ROOT, "src", "setups.jl"))

# Python interpreter to drive py-1d. Search order:
#   1. $PY1D_PYTHON env var (explicit override).
#   2. <REPO_ROOT>/py-1d/.venv/bin/python (the canonical venv).
#   3. <git common dir>/../py-1d/.venv/bin/python (worktree fallback —
#      the venv lives at the main checkout, not the worktree).
function _resolve_py_venv()
    env = get(ENV, "PY1D_PYTHON", "")
    isfile(env) && return env
    cand1 = joinpath(REPO_ROOT, "py-1d", ".venv", "bin", "python")
    isfile(cand1) && return cand1
    # Walk up to find a sibling main checkout. In the dfmm-2d worktree
    # layout, `.git` is a file pointing at the parent's gitdir; the
    # parent dir has the canonical py-1d/.venv.
    try
        gitfile = joinpath(REPO_ROOT, ".git")
        if isfile(gitfile)
            line = readline(gitfile)
            startswith(line, "gitdir:") || return cand1
            gd = strip(line[length("gitdir:")+1:end])
            # gitdir typically: <main>/.git/worktrees/<name>
            main_git = joinpath(dirname(dirname(gd)))
            main_repo = dirname(main_git)
            cand2 = joinpath(main_repo, "py-1d", ".venv", "bin", "python")
            isfile(cand2) && return cand2
        end
    catch
        # Fall through to cand1; main() will produce the standard error.
    end
    return cand1
end

const PY_VENV = _resolve_py_venv()
const PY_DUMPER = joinpath(REPO_ROOT, "py-1d", "scripts", "dump_golden.py")
const GOLDEN_DIR = joinpath(REPO_ROOT, "reference", "golden")
const MAX_INLINE_BYTES = 5 * 1024 * 1024  # 5 MiB; per the brief

# Targets ↔ factories. Every entry must produce a `.params` NamedTuple
# whose keys are exactly what py-1d/scripts/dump_golden.py expects.
const TARGETS = Dict(
    "sod"           => () -> setup_sod(),
    "cold_sinusoid" => () -> setup_cold_sinusoid(N=400, t_end=0.6,
                                                  tau=1e3),
    "steady_shock"  => () -> setup_steady_shock(M1=3.0, t_end=3.0,
                                                 N=400, tau=1e-3),
    "wavepool"      => () -> setup_kmles_wavepool(N=256, t_end=5.0,
                                                   u0=1.0, P0=0.1,
                                                   K_max=16, seed=42,
                                                   tau=1e-3, n_snaps=50),
    "dust_in_gas"   => () -> setup_dust_in_gas(N=400, t_end=0.5,
                                                grain_radius=1e-3,
                                                n_snaps=10),
    "eion"          => () -> setup_eion(N=32, t_end=10.0, m_e=0.01,
                                         n_snaps=50),
)

const ALL_TARGETS = sort(collect(keys(TARGETS)))

# ---- JSON encoding (tiny, hand-rolled to avoid a JSON3 dep) -------------

_json(x::Bool) = x ? "true" : "false"
_json(x::Integer) = string(x)
_json(x::AbstractFloat) = isfinite(x) ? @sprintf("%.17g", x) : "null"
_json(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
_json(::Nothing) = "null"
_json(x::AbstractVector) = "[" * join(_json.(x), ",") * "]"
_json(x::Symbol) = _json(string(x))
function _json(t::NamedTuple)
    parts = String[]
    for k in keys(t)
        push!(parts, _json(string(k)) * ":" * _json(getfield(t, k)))
    end
    return "{" * join(parts, ",") * "}"
end

# ---- Dumper -------------------------------------------------------------

"""
    run_target(target::String) -> Path

Generate one golden. Writes either `<GOLDEN_DIR>/<target>.h5` or, if
the resulting file would exceed `MAX_INLINE_BYTES`, replaces the
`.h5` with a `.h5.regen` stub describing how to rebuild it.
"""
function run_target(target::String)
    haskey(TARGETS, target) ||
        error("unknown target $(target); choose from $(ALL_TARGETS)")
    factory = TARGETS[target]
    ic = factory()
    params_json = _json(ic.params)

    # Mirror the brief's CI semantics: large goldens are not committed,
    # but we still *generate* them when the user runs `make goldens`.
    out_h5 = joinpath(GOLDEN_DIR, "$(target).h5")
    mkpath(GOLDEN_DIR)

    @printf("[make_goldens] target=%s -> %s\n", target, out_h5)
    cmd = `$(PY_VENV) $(PY_DUMPER) --target $(target) --out $(out_h5) --params $(params_json)`
    run(cmd)

    # If the output is bigger than the inline budget, replace with a stub.
    if filesize(out_h5) > MAX_INLINE_BYTES
        regen_path = out_h5 * ".regen"
        write_regen_stub(regen_path, target, ic.params)
        @printf("[make_goldens]   %s exceeds %.1f MiB; wrote stub %s\n",
                basename(out_h5), MAX_INLINE_BYTES/1024/1024,
                basename(regen_path))
        # Leave the .h5 in place locally (CI/tests may use it) but it
        # is gitignored via the size rule. We do *not* delete it.
    else
        # Drop any stale stub from a prior big run.
        regen_path = out_h5 * ".regen"
        isfile(regen_path) && rm(regen_path)
    end

    @printf("[make_goldens]   %s wrote %.1f KiB\n", target,
            filesize(out_h5) / 1024)
    return out_h5
end

function write_regen_stub(path::String, target::String, params)
    json = _json(params)
    open(path, "w") do io
        println(io, "# Regeneration stub for ", target)
        println(io, "#")
        println(io, "# This golden exceeded the 5 MiB commit budget;")
        println(io, "# regenerate it locally / in CI with one of:")
        println(io)
        println(io, "    julia --project=. scripts/make_goldens.jl --target ",
                target)
        println(io)
        println(io, "or directly:")
        println(io)
        println(io, "    py-1d/.venv/bin/python py-1d/scripts/dump_golden.py \\")
        println(io, "        --target ", target, " \\")
        println(io, "        --out reference/golden/", target, ".h5 \\")
        println(io, "        --params '", json, "'")
        println(io)
        println(io, "# Parameters (canonical, JSON-encoded):")
        println(io, json)
    end
end

# ---- CLI ----------------------------------------------------------------

function parse_args(argv)
    target = nothing
    list = false
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--target"
            target = argv[i+1]; i += 2
        elseif a == "--list"
            list = true; i += 1
        elseif a == "-h" || a == "--help"
            println(stdout, """
                Usage: julia --project=. scripts/make_goldens.jl [opts]

                  --target NAME    generate one of: $(join(ALL_TARGETS, ", "))
                  --list           print the target list and exit
                  -h, --help       show this message

                Without --target, all six goldens are generated.
                """)
            exit(0)
        else
            error("unknown argument $a; --help for usage")
        end
    end
    return (; target, list)
end

function main(argv)
    args = parse_args(argv)
    if args.list
        println.(ALL_TARGETS)
        return 0
    end
    isfile(PY_VENV) || error("py-1d venv missing; expected $(PY_VENV). " *
                              "run `python3 -m venv py-1d/.venv && " *
                              "py-1d/.venv/bin/pip install -e py-1d/[dev] h5py`.")
    isfile(PY_DUMPER) || error("dump_golden.py missing at $(PY_DUMPER)")
    targets = isnothing(args.target) ? ALL_TARGETS : [args.target]
    for t in targets
        run_target(t)
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
