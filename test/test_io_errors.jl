# test_io_errors.jl
#
# Error-path coverage for `src/io.jl`. The existing `test_io.jl` file
# covers the happy round-trip; this file targets:
#
#   * Loading a path that does not exist on either backend.
#   * Loading a malformed file (HDF5 backend reads a truncated file).
#   * Saving an unsupported scalar type (e.g. a complex value, a
#     `Vector{Vector{Float64}}` non-array).
#   * Mixed extension casing (.H5, .Hdf5, .JLD2 should all dispatch).
#   * Empty NamedTuple round-trip on both backends.

using Test
using dfmm

@testset "io: load_state on non-existent path raises" begin
    # HDF5 + JLD2 both raise on a missing file. We don't pin the
    # specific error type (it may be a SystemError, IOError, or
    # backend-specific) — only that some exception is raised.
    @test_throws Exception dfmm.load_state("/nonexistent/path/to/file.h5")
    @test_throws Exception dfmm.load_state("/nonexistent/path/to/file.jld2")
end

@testset "io: case-insensitive extension dispatch" begin
    state = (a = [1.0, 2.0],)
    mktempdir() do dir
        for ext in (".H5", ".HDF5", ".Hdf5", ".JLD2", ".Jld2")
            path = joinpath(dir, "snap" * ext)
            dfmm.save_state(path, state)
            round = dfmm.load_state(path)
            @test round.a == state.a
        end
    end
end

@testset "io: unsupported HDF5 field type ⇒ ArgumentError" begin
    # HDF5 backend rejects fields that are neither AbstractArray nor a
    # supported scalar (Number / AbstractString / Bool). A
    # `Vector{Vector{Float64}}` (jagged) hits the `_save_hdf5` else
    # branch; cleanest probe is a Symbol, which is also rejected.
    state = (s = :hello,)  # Symbol is not in the allowed scalar union
    mktempdir() do dir
        path = joinpath(dir, "bad.h5")
        @test_throws ArgumentError dfmm.save_state(path, state)
    end
end

@testset "io: empty NamedTuple round-trip writes only schema metadata" begin
    state = NamedTuple()
    mktempdir() do dir
        for ext in (".h5", ".jld2")
            path = joinpath(dir, "empty" * ext)
            dfmm.save_state(path, state)
            round = dfmm.load_state(path)
            @test isempty(keys(round))
        end
    end
end

@testset "io: HDF5 load on truncated file raises" begin
    # Write a file with the .h5 extension but invalid binary content.
    # The HDF5 reader should raise on opening (we don't pin the type).
    mktempdir() do dir
        path = joinpath(dir, "truncated.h5")
        write(path, "this is not an hdf5 file")
        @test_throws Exception dfmm.load_state(path)
    end
end

@testset "io: HDF5 load on empty file raises" begin
    mktempdir() do dir
        path = joinpath(dir, "empty_bytes.h5")
        write(path, "")
        @test_throws Exception dfmm.load_state(path)
    end
end

@testset "io: integer / string / bool scalars round-trip on HDF5" begin
    # HDF5 backend writes `Number / AbstractString / Bool` as attrs;
    # exercise each path explicitly.
    state = (
        n_steps = 1000,           # Int
        label = "production",     # String
        is_warm = true,           # Bool
        floats = [0.1, 0.2, 0.3], # Array
    )
    mktempdir() do dir
        path = joinpath(dir, "scalars.h5")
        dfmm.save_state(path, state)
        round = dfmm.load_state(path)
        @test round.n_steps == 1000
        @test round.label == "production"
        @test round.is_warm == true
        @test round.floats == state.floats
    end
end

@testset "io: extension with no dot raises ArgumentError" begin
    # `splitext("nofile")[2]` is `""`, which is not a supported
    # extension. The save / load functions must surface this as
    # ArgumentError.
    @test_throws ArgumentError dfmm.save_state("plain_no_extension",
                                               (a = [1.0],))
    @test_throws ArgumentError dfmm.load_state("plain_no_extension")
end
