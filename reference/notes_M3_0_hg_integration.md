# Phase M3-0 — HierarchicalGrids + r3djl integration notes

**Date:** 2026-04-25.
**Branch:** `m3-0-hg-integration`.
**Status:** complete; bit-exact parity to M1 verified.

This note is the implementation log for Phase M3-0 of
`reference/MILESTONE_3_PLAN.md`. The phase brings
[`HierarchicalGrids.jl`](https://github.com/yipihey/HierarchicalGrids.jl)
(HG, v0.1) and [`r3djl`](https://github.com/yipihey/r3djl) (R3D, v0.1)
into dfmm-2d as deps, lays a thin shim that runs M1's Phase-1
Cholesky-sector integrator on top of HG's
`SimplicialMesh{1, T}` + `PolynomialFieldSet` substrate, and verifies
bit-exact parity (per-step `(α, β)` agree to 0.0 absolute) against
M1's hand-rolled `Mesh1D` + `Segment` baseline.

## Version pins

| Package | Version | Source |
|---|---|---|
| HierarchicalGrids | 0.1.0 | Local fork at `~/.julia/dev/HierarchicalGrids` (see "1D coverage" below). Upstream commit `b964c3351a869fdaba497e8b65c4251d6c5475a9` (Initial commit, 2026-04-26). |
| R3D | 0.1.0 | `https://github.com/yipihey/r3djl` (subdir `R3D.jl`), tree-sha1 `decdca3a8b2b092f3df5600118d98b0fc9d7423e`. |

Both packages are recorded in `Project.toml`'s `[deps]` block with
`[compat]` lower bounds at `0.1.0` (R3D pinned via Manifest tree-sha1
on main). HG is currently `develop`-mounted at
`~/.julia/dev/HierarchicalGrids` because of the BMI2/LLVM workaround
described next; once that patch lands upstream the develop link can be
replaced with a Manifest commit pin.

## HG 1D coverage

Pre-launch checks (per the M3-0 brief and `notes_HG_design_guidance.md`
item 1):

- `SimplicialMesh{1, Float64}` constructs cleanly. The default
  constructor enforces `D >= 1`; topology, vertex positions, and
  reference positions all work in 1D.
- `n_simplices`, `n_vertices`, `simplex_volume`, `simplex_vertex_indices`
  all dispatch correctly on `SimplicialMesh{1, T}`.
- `PolynomialFieldSet` allocated with `MonomialBasis{1, 0}()` (one
  coefficient per cell, i.e. piecewise constant) supports per-element
  `(α, β)` storage with bit-exact write-back through
  `fields.alpha[j] = (val,)`. Confirmed in `test_M3_0_smoke.jl`.

Two HG primitives we did **not** exercise in M3-0 (deferred to M3-1):

- `compute_overlap` for D=1 (interval intersection). 1D overlap is
  trivial closed-form; we'll wire it in M3-3 when the Bayesian remap
  enters.
- `parallel_for_cells` is plumbed into `cholesky_step_HG!` behind a
  `threaded::Bool=false` kwarg, but defaulted off for the parity tests
  so threading-induced reordering doesn't perturb bit-equality.

## Workarounds applied

### BMI2 / LLVM precompile failure on Apple Silicon

Symptom: precompiling `HierarchicalGrids` on aarch64 (Apple M-series)
crashes with

```
LLVM ERROR: Cannot select: intrinsic %llvm.x86.bmi.pdep.32
```

Root cause: `src/BitPrimitives/BitPrimitives.jl` uses runtime `if`
branches around `ccall("llvm.x86.bmi.pdep.32", ...)` rather than
compile-time `@static if`. Even on aarch64, where the runtime `if`
takes the `false` branch, Julia 1.12's LLVM lowering tries to compile
the dead-code `ccall` and aborts.

Workaround: a one-liner patch to BitPrimitives.jl that replaces the
runtime `if` with `@static if`. The patch is applied at
`~/.julia/dev/HierarchicalGrids/src/BitPrimitives/BitPrimitives.jl`;
in the dfmm Manifest, HG is `develop`-mounted at that path. The patch
diff against upstream's initial commit is essentially:

```julia
-const HAS_BMI2 = let
-    if Sys.ARCH == :x86_64 || Sys.ARCH == :i686
-        try
-            test_result = ccall("llvm.x86.bmi.pdep.32", llvmcall, UInt32,
-                                (UInt32, UInt32), UInt32(1), UInt32(1))
-            test_result == UInt32(1)
-        catch
-            false
-        end
-    else
-        false
-    end
-end
+const HAS_BMI2 = @static if Sys.ARCH == :x86_64 || Sys.ARCH == :i686
+    try
+        test_result = ccall("llvm.x86.bmi.pdep.32", llvmcall, UInt32,
+                            (UInt32, UInt32), UInt32(1), UInt32(1))
+        test_result == UInt32(1)
+    catch
+        false
+    end
+else
+    false
+end
```

Plus the `@static` qualifier on each `if HAS_BMI2` branch inside the
`pdep`/`pext` 32/64-bit dispatchers (four call sites total).

Action items:

1. Send this patch to the HG project as a small PR or upstream issue.
2. Once it lands, replace the `develop` mount with an upstream commit
   pin in dfmm's Manifest.

This has been a clean, low-risk patch; HG's behavior on x86 with BMI2
is unchanged because the two forms are operationally equivalent.

### r3djl 1D coverage

R3D supports 2D and 3D natively; 1D would be trivial interval
intersection. R3D's interfaces are not exercised in M3-0 (1D Cholesky
sector has no spatial coupling), so this is deferred. M3-3's Bayesian
remap can either:

- use R3D for 2D/3D and a ~10-line dfmm-internal interval-intersection
  primitive for 1D, or
- contribute the 1D primitive to r3djl upstream.

The HG side has `interval_intersection` and `interval_length` for D=1
overlap geometry already, so the r3djl 1D gap is small.

## File layout (new in M3-0)

| File | Role |
|---|---|
| `src/eom.jl` | HG-aware EL residual wrappers (`cholesky_el_residual_HG`, `read_alphabeta`, `write_alphabeta!`). In M3-0 the residual delegates to M1's `cholesky_el_residual` so bit-equality is automatic. |
| `src/newton_step_HG.jl` | HG-side Newton driver (`cholesky_step_HG!`, `cholesky_run_HG!`) plus 1D mesh constructors (`single_cell_simplicial_mesh_1D`, `uniform_simplicial_mesh_1D`, `allocate_chfield_HG`). Each per-simplex Newton solve calls into M1's `cholesky_step` byte-identically. |
| `src/types.jl` (extension) | `ChFieldND{D, T}` — dimension-generic Cholesky-sector field tag type. Coexists with the legacy 1D `ChField{T}` until M3-2 verifies full M1+M2 parity on the HG path. |
| `src/dfmm.jl` (extension) | New `import HierarchicalGrids; import R3D` block + new `include("eom.jl"); include("newton_step_HG.jl")` + new exports for the M3-0 API. |
| `test/test_M3_0_smoke.jl` | Dependency-import sentinels + HG primitive sanity checks. |
| `test/test_M3_0_parity_1D.jl` | Bit-exact parity tests against M1's Phase-1 baseline. |
| `Project.toml` (extension) | Add HG and R3D to `[deps]` + `[compat]`. |
| `Manifest.toml` (extension) | HG develop-pin (LLVM workaround); R3D Manifest tree-sha1 pin from `https://github.com/yipihey/r3djl`. |

Files explicitly **not modified** in M3-0:

- M1's `src/cholesky_sector.jl`, `src/discrete_transport.jl`,
  `src/newton_step.jl`, `src/segment.jl`, the legacy `ChField{T}` block
  in `src/types.jl`. Untouched so M1's regression baseline remains
  identical.
- All M1 / M2 test files (`test_phase1_*.jl`, `test_phase2_*.jl`, etc.).
- `reference/MILESTONE_*_STATUS.md`, `reference/notes_phase*_*.md`,
  `specs/`, `design/`, `HANDOFF.md`.

## Parity test results

Three tests in `test/test_M3_0_parity_1D.jl`, mirroring M1's three
Phase-1 regression tests:

| Test | Setup | Per-step `max(|Δα|, |Δβ|)` | Brief target |
|---|---|---|---|
| Zero-strain free evolution | `M_vv=1.0`, `divu_half=0.0`, `Δt=1e-3`, `N=100` steps, `α₀=1.0, β₀=0.0` | **0.0** | < 5e-13 |
| Uniform-strain | `M_vv=0.0`, `κ=0.1`, `Δt=1e-3`, `N=100` steps, `α₀=1.0, β₀=0.5` | **0.0** | < 5e-13 |
| Symplectic loop preservation | 64-vertex elliptical loop in (α, β), evolved 100 steps; loop integral $\oint(\alpha^3/3)\,d\beta$ | **0.0** per-vertex; HG and M1 loop integrals agree to **0.0**; absolute drift `1e-10` | per-vertex < 5e-13; loop integral < 1e-10 |

The exact-zero result (rather than ULPs) is achieved because
`cholesky_step_HG!` calls `cholesky_step` from M1 directly — no
algorithmic divergence — so the only difference between the two paths
is the storage shim (which is bit-preserving). Threading is off in
the parity tests; turning it on (`threaded=true`) does not change
exactness in our tests because the per-cell solves are independent
and the writes go to disjoint indices, but it would not be reliable
across all hardware/BLAS combinations and so is gated.

Newton iteration counts (informational): 2-3 iterations per cell per
step on the smooth ICs of all three tests, identical to M1.

## Wall-time benchmark

Single-cell Phase-1 trajectory, 5×1000 implicit-midpoint steps, on
the development laptop (Apple M-series, single thread):

| Path | Time | Per step | Ratio HG / M1 |
|---|---|---|---|
| M1 (`cholesky_run`) | 1.15 ms | 0.230 µs | — |
| HG (`cholesky_run_HG!`) | 1.16 ms | 0.232 µs | **1.011×** |

64-cell parallel-trajectories (3×1000 steps):

| Path | Time | Per (cell·step) | Ratio HG / M1 |
|---|---|---|---|
| M1 (loop over `cholesky_run`) | 43.6 ms | 0.227 µs | — |
| HG (`cholesky_run_HG!`) | 38.2 ms | 0.199 µs | **0.877×** (HG slightly faster — better data locality) |

Both well within the 2× brief budget. The HG path's "indirection
overhead" in this phase is negligible because the field-set storage
is contiguous SoA-backed Float64 vectors.

## Dimension-generic API: `ChFieldND{D, T}`

Added to `src/types.jl`. In 1D, this is a single `(α, β)` pair. The
type is a documentation / dispatch tag in M3-0 — the M1 `cholesky_step`
kernel still uses `SVector{2}` for the bit-equality contract, and
per-cell storage goes through HG's `PolynomialFieldSet` (which is
dimension-generic by construction). When M3-3 lands the 2D Berry
connection, `ChFieldND{2, T}` will carry `(α_a, β_a)_{a=1,2}` plus a
rotation angle `θ_R`; in M3-7 the 3D case adds three `θ_{ab}`.

Naming. The legacy 1D `ChField{T}` is preserved; the new
dimension-generic type is `ChFieldND{D, T}` (rather than overloading
the same name). M3-2 will retire the legacy `ChField` once full
M1+M2 parity is verified.

## Open questions / handoff to M3-1

1. **HG BMI2 patch upstream.** Send the `@static if` patch to the HG
   project; replace dfmm's `develop` pin with an upstream commit pin
   once the fix lands.

2. **r3djl 1D primitive.** Either contribute a 1D interval-overlap
   primitive to r3djl upstream, or implement the ~10-line equivalent
   in dfmm. M3-3 (Bayesian remap) is the first phase that needs it.

3. **Threading verification.** `cholesky_step_HG!` accepts
   `threaded=true` via `parallel_for_cells`. We did not add a parity
   test with threading enabled; if threading reorders cell visits in
   a way that affects floating-point determinism on later phases (when
   neighbour data is read), an `@assert` or per-cell-independence
   property test may be wanted in M3-1.

4. **Reference-positions / Lagrangian-mass coordinate.** The HG
   `SimplicialMesh{1, T}` carries `positions` and `reference_positions`
   per vertex. In M3-0 we use neither; in M3-1 (Phase 2 port) the
   reference position becomes the Lagrangian-mass coordinate `m`,
   and `simplex_volume(mesh, j) = J̄_j Δm_j` becomes the per-cell
   Jacobian times mass. The mapping is straightforward; flag here so
   M3-1's agent doesn't re-discover it.

5. **`ChField{1, T}` vs `ChFieldND{1, T}` aliasing.** Once M3-2 retires
   the legacy 1D `ChField`, consider renaming `ChFieldND{D, T}` to
   `ChField{D, T}` — but only after the legacy users (Phase 1's
   `single_segment_mesh`, Phase-2/5's `Segment{T,DetField{T}}`) have
   migrated to the HG path. Don't rename in M3-0/M3-1.
