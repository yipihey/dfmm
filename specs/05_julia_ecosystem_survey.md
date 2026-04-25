# Julia ecosystem survey for dfmm-2d

**Audience:** Tom Abel and the implementation agents.
**Purpose:** Identify which Julia packages we should reuse vs. build from scratch for dfmm-2d. Honest production-readiness assessment for each.
**Date:** April 25, 2026.

---

## Executive summary

The Julia ecosystem is genuinely well-suited for dfmm-2d. Three things stand out:

1. **Trixi.jl is the structural model to study, not the framework to build on.** It does many of the things we'd want (DG on adaptive quadtree, p4est integration, entropy-stable methods, AMR with multiple indicators, MPI-distributed) and has an active multi-institution development team. *But*: its evolution model is method-of-lines with explicit time integration via SciML's OrdinaryDiffEq. dfmm-2d's variational integrator with implicit Newton + per-cell action error doesn't fit Trixi's mold cleanly. We should **read Trixi's code seriously** to understand what they got right (mesh handling, AMR callbacks, type design), but build a new framework rather than extending Trixi.

2. **The GPU portability story is solved.** KernelAbstractions.jl + CUDA.jl/AMDGPU.jl/Metal.jl/oneAPI.jl gives single-source code targeting all four major GPU vendors plus multithreaded CPU. AcceleratedKernels.jl provides standard parallel primitives (sort, scan, reduce) on top. ParallelStencil.jl + ImplicitGlobalGrid.jl provides multi-GPU stencil computation. **All production-grade.**

3. **There is no Julia-native r3d equivalent for our needs.** The polynomial-moment polyhedral overlap is the one piece of geometric machinery we would have to either wrap (r3d C library), reimplement from scratch, or get along without by approximating with quadrature. **This is the single biggest design decision** for the implementation.

The rest of the stack — Enzyme.jl for AD, NonlinearSolve.jl for Newton, Distributions.jl for stochastic injection, P4est.jl for AMR, Makie.jl + WriteVTK.jl for visualization, StaticArrays.jl for small-tensor performance — is mature and ready.

---

## 1. Time integration

### 1.1 OrdinaryDiffEq.jl / SciML symplectic integrators

**Status:** Production. Maintained by Chris Rackauckas's group.

**Relevant content:** SymplecticEuler, VelocityVerlet, VerletLeapfrog, McAte2/3/4, Ruth3, CandyRoz4, CalvoSanz4, Yoshida6, KahanLi6, SofSpa10. Higher-order symplectic from Gauss-Lobatto-IIIA-IIIB via GeometricIntegratorsDiffEq.jl.

**Honest assessment:** The available symplectic integrators assume **separable Hamiltonian** (i.e., H = T(p) + V(q) with quadratic kinetic energy). dfmm-2d's connection-form action has a non-separable Hamiltonian on the Cholesky sector — H_Ch = -½α²γ² couples α to γ via the EOS, and the symplectic form is *weighted* (ω = α²dα∧dβ rather than dα∧dβ). **None of the off-the-shelf symplectic integrators handle this directly.**

**Recommendation:** Use OrdinaryDiffEq.jl for the bulk position/momentum sector (where standard symplectic methods work). Implement a custom variational integrator for the Cholesky sector using the discrete Hamilton-Pontryagin framework. The Newton solve at each step uses NonlinearSolve.jl (§3 below). This is the correct call regardless: the variational integrator is genuinely new methods work and shouldn't go into a black-box library.

### 1.2 GeometricIntegrators.jl

**Status:** Production. Active development by Michael Kraus at IPP Garching.

**Relevant content:** "Reference implementation for the methods described in: Michael Kraus, *Projected Variational Integrators for Degenerate Lagrangian Systems* (arXiv:1708.07356); Michael Kraus and Tomasz M. Tyranowski, *Variational Integrators for Stochastic Dissipative Hamiltonian Systems* (arXiv:1909.07202)."

**This is directly relevant.** Kraus's "Projected Variational Integrators for Degenerate Lagrangian Systems" addresses exactly the case where the Hessian degenerates — which is our Cholesky-sector situation at γ → 0. And the stochastic variational integrators work is published in J. Comp. Dyn. and addresses degenerate stochastic Hamiltonian systems.

**Recommendation:** **Read these two papers carefully before implementing the variational integrator.** GeometricIntegrators.jl provides reference implementations we can either reuse directly (for the standard cases) or specialize from (for the dfmm-2d connection-form structure). At minimum it's the prior-art baseline.

---

## 2. Adaptive mesh refinement

### 2.1 P4est.jl + P4estTypes.jl

**Status:** Production. Wraps the well-established C library p4est (Burstedde-Wilcox-Ghattas 2011, SIAM J. Sci. Comput.). Used in production by Trixi.jl.

**Relevant content:** Forest-of-quadtrees/octrees with parallel AMR via MPI. Supports 2:1 balanced refinement, distributed partitioning, ghost cells. P4estTypes.jl provides a higher-level Julia type interface; P4est.jl is the lower-level binding.

**Honest assessment:** This is the right choice for the Eulerian-mesh AMR backend. The 2:1 refinement constraint matches our needs (Part IX action-error AMR with hysteresis). MPI parallelism comes for free. The integration with KernelAbstractions for GPU AMR is the only weak point — Trixi.jl's AMR step is currently CPU-only, which is a known performance bottleneck.

**Recommendation:** Use P4est.jl directly for the Eulerian quadtree. Plan for CPU AMR step in Milestone 1; defer GPU AMR to a later milestone unless it becomes critical. For the **Lagrangian** simplicial mesh, this doesn't apply — we'd want a different package (see §6 on mesh handling).

### 2.2 Trixi.jl (study, don't reuse)

**Status:** Production-grade for hyperbolic PDEs on quad/hex meshes with DG. Active development by Schlottke-Lakemper, Gassner, Ranocha, Winters, Chan, Rueda-Ramírez. Substantial publication record (Trixi citation list spans 2021–2026).

**Why not reuse:** Trixi is built around the SciML method-of-lines paradigm: spatial DG discretization → ODE in time → explicit RK integration. It does this well. dfmm-2d's variational integrator with implicit Newton iteration on a non-separable connection-form action doesn't fit this pattern. Trying to extend Trixi to handle our case would require rewriting its time-stepping core.

**What to reuse:** **The architectural patterns.** Trixi's separation of mesh, solver, semidiscretization, callbacks, and AMR controllers is well-designed. The IndicatorHennemannGassner (entropy-based) and IndicatorLöhner (gradient-based) AMR patterns generalize naturally to our action-error indicator. The `ControllerThreeLevel` and `ControllerThreeLevelCombined` patterns map onto our refine/coarsen hysteresis. The compatibility with OrdinaryDiffEqLowStorageRK shows how to integrate with SciML cleanly.

**Recommendation:** Read Trixi's source code carefully — particularly `src/callbacks_step/amr.jl`, `src/solvers/dg_p4est.jl`, and the `IndicatorHennemannGassner` implementation. Adopt the architectural patterns; don't subclass.

---

## 3. Implicit nonlinear solvers (Newton iteration)

### 3.1 NonlinearSolve.jl

**Status:** Production. Pal et al. 2024 paper (`arXiv:2403.16341`) demonstrates state-of-the-art performance vs. PETSc SNES, Sundials KINSOL, and MINPACK.

**Relevant content:** NewtonRaphson with sparse Jacobian, Newton-Krylov with preconditioning, TrustRegion, LevenbergMarquardt, polyalgorithms. Automatic algorithm selection. Static-array specialization for small problems (uses `SimpleNewtonRaphson`, non-allocating, GPU-kernel-compatible). Sparsity detection via SparseConnectivityTracer + SparseMatrixColorings, then colored AD via DifferentiationInterface.jl.

**Honest assessment:** This is exactly what we need. The discrete EL equations from the variational integrator give a sparse system whose sparsity pattern matches the triangle adjacency graph; NonlinearSolve.jl detects and exploits this automatically. For small per-triangle subproblems (if we can split the global Newton into local solves), `SimpleNewtonRaphson` on StaticArrays is non-allocating and runs on GPU.

**Recommendation:** Default tool for the implicit time step. Use the polyalgorithm `FastShortcutNonlinearPolyalg` initially, then specialize to NewtonRaphson with sparse Jacobian once the sparsity structure is stable.

### 3.2 LinearSolve.jl

**Status:** Production. Same SciML group.

**Relevant content:** Unified interface to direct solvers (UMFPACK, KLU, Pardiso) and iterative solvers (GMRES, BiCGStab, Krylov.jl). Automatic algorithm selection by problem structure.

**Recommendation:** Use as the linear-solve backend for NonlinearSolve.jl's Newton step. Default selection is good; for very large 2D problems we'll specialize to GMRES with ILU preconditioning.

---

## 4. Automatic differentiation

### 4.1 Enzyme.jl

**Status:** Production-grade for many use cases, but with known limitations. Active development at MIT (Enzyme is the LLVM-level AD framework; Enzyme.jl wraps it for Julia).

**Relevant content:** LLVM-level reverse-mode AD; meets or exceeds state-of-the-art AD performance. Forward and reverse modes. Supports CUDA.jl with KernelAbstractions integration. SciMLSensitivity.jl uses Enzyme.jl as `EnzymeVJP` — "the fastest VJP whenever applicable."

**Honest assessment from the docs:** "Enzyme.jl currently has low coverage over the Julia programming language, for example restricting the user's defined f function to not do things like require garbage collection or calls to BLAS/LAPACK." This means: write inner loops with mutation, no allocations, no fancy linear algebra. For the kind of inner kernels we'd need to differentiate (action evaluation, Jacobian-vector products), this is acceptable but does shape how we write the code.

For BLAS/LAPACK calls, Enzyme has fallback rules for `gemm`, `gemv`, `dot`, `axpy`, and a few others (with fast derivatives), and a slow fallback for everything else (correct but possibly slow on parallel platforms).

**Recommendation:** Use Enzyme.jl for the action-gradient and Jacobian computation in the variational integrator. Write the inner kernels in Enzyme-compatible style (no allocation, no GC). For the AMR error estimator (which evaluates `S^{p+1}|_cell - S^p|_cell`), Enzyme is overkill — direct evaluation suffices. For sensitivity analysis later (parameter studies, calibration), Enzyme is the right backend.

### 4.2 DifferentiationInterface.jl

**Status:** Production. Dalle & Hill 2026 JMLR paper. Used as the AD layer in NonlinearSolve.jl, Optimization.jl, etc.

**Relevant content:** Common interface across ForwardDiff.jl, ReverseDiff.jl, Zygote.jl, Enzyme.jl, Mooncake.jl, Tracker.jl, FiniteDiff.jl, FiniteDifferences.jl. Sparse differentiation via SparseConnectivityTracer + SparseMatrixColorings.

**Recommendation:** Use as the abstraction layer. The default backend choice can switch between ForwardDiff (small problems) and Enzyme (large problems) via `AutoForwardDiff()` vs `AutoEnzyme()`. The `prepare_*` API allows pre-allocating tapes/caches for repeated evaluation.

### 4.3 ForwardDiff.jl (also keep)

**Status:** Production. Rock-solid.

**Relevant content:** Operator-overloading forward-mode AD via dual numbers. Limited to forward mode but extremely robust — works on essentially any pure-Julia code without restrictions.

**Recommendation:** Use as the fallback when Enzyme has trouble with a particular code path. Use as the default for small per-triangle Jacobians (where forward mode is more efficient than reverse). PolyesterForwardDiff for threaded forward AD on dense small Jacobians.

---

## 5. GPU portability

### 5.1 KernelAbstractions.jl

**Status:** Production. JuliaGPU organization. Used by AcceleratedKernels.jl, Trixi.jl (recently), and Reactant.jl integrations.

**Relevant content:** Vendor-neutral kernel programming via the `@kernel` macro. Single-source code targeting CUDA, ROCm (AMDGPU), oneAPI (Intel), Metal (Apple), and multithreaded CPU. The `Backend` type selects target at launch time.

**Honest assessment:** This is the right choice. The ScienceDirect paper (`Juliana` translator from CUDA.jl to KernelAbstractions, 2025) reports < 7% performance overhead for portability. Apple Metal, AMD ROCm, Intel oneAPI, NVIDIA CUDA all work; multi-threaded CPU fallback is automatic.

**Recommendation:** Write all dfmm-2d compute kernels in KernelAbstractions. Target CPU first for development, switch to GPU for production. The `Tom uses Apple Silicon` consideration argues for Metal.jl as a primary target — works out of the box.

### 5.2 CUDA.jl, AMDGPU.jl, Metal.jl, oneAPI.jl

**Status:** All production. CUDA.jl is the most battle-tested (Linux + Windows, comprehensive vendor library bindings: cuBLAS, cuFFT, cuSOLVER, cuSPARSE, cuDNN, cuTENSOR). AMDGPU.jl is solid for Linux/ROCm. Metal.jl has matured beyond experimental status. oneAPI.jl is the least mature but functional.

**Recommendation:** Use these as backend selectors via KernelAbstractions. Direct usage only when needed for vendor-library calls (e.g., cuFFT for spectral diagnostics in the wave-pool calibration).

### 5.3 ParallelStencil.jl + ImplicitGlobalGrid.jl

**Status:** Production. ETH Zürich. Used in geodynamics, glaciology, electroweak field theory simulations.

**Relevant content:** Architecture-agnostic stencil computations on regular staggered grids. KernelAbstractions backend (recently added) plus native CUDA/AMDGPU. Multi-GPU via ImplicitGlobalGrid.jl with CUDA-aware MPI. Hide-communication-behind-computation via `@hide_communication`. 70% of theoretical peak GPU throughput documented in published benchmarks.

**Honest assessment:** Designed for *regular* staggered grids — so it's a natural fit for the **Eulerian** mesh in dfmm-2d (since each level of the quadtree is a regular grid locally), but **not** for the **Lagrangian** simplicial mesh.

**Recommendation:** Use ParallelStencil for any operations on the regular Eulerian quadtree (within-level stencils, regular-array operations). Don't use for the unstructured Lagrangian side.

### 5.4 AcceleratedKernels.jl

**Status:** Production. JuliaGPU organization. Cross-architecture parallel primitives on top of KernelAbstractions.

**Relevant content:** GPU sort and accumulate (adopted as official AMDGPU algorithms), reduce, foreach, map. Single source, targets all KA backends.

**Recommendation:** Use for the standard primitives we'd otherwise reimplement — segmented reduction across triangles, sort by Hilbert key for cache locality, prefix-scan for cumulative-mass coordinates.

---

## 6. Mesh and geometry handling

### 6.1 Meshes.jl

**Status:** Active development (JuliaGeometry organization). Reasonably stable but the API has evolved.

**Relevant content:** Geometric primitives, meshes (Cartesian, structured, unstructured), domain operations.

**Honest assessment:** Meshes.jl is general-purpose — useful for mesh I/O, simple geometric tests, but doesn't have the specialized data structures we'd need for the moving Lagrangian mesh with high-order curved-element reconstruction.

**Recommendation:** Use Meshes.jl for I/O and standard mesh utilities. Build our own data structure for the Lagrangian mesh frame fields.

### 6.2 Polyhedra.jl + GeometryOps.jl + PolygonAlgorithms.jl

**Status:** All have production-quality implementations of polygon clipping in 2D (Greiner-Hormann, Foster-Hormann, Weiler-Atherton, O'Rourke). 3D polyhedral intersection is much weaker.

**Relevant content:**
- GeometryOps.jl (JuliaGeo): production-grade Foster-Hormann clipping for arbitrary polygons in 2D. Used in geospatial workflows. Active development.
- PolygonAlgorithms.jl (Lior Sinai): O'Rourke for convex, Weiler-Atherton for concave. Documented and tested.
- Polyhedra.jl: H-representation/V-representation polyhedra in arbitrary dimensions. Production for convex computation. Not specialized for fast moment computation.

**Honest assessment for our needs:** None of these compute *polynomial moments over polyhedral overlaps* — the specific operation r3d does. They compute the geometric overlap polygon/polyhedron; we'd then need to integrate polynomial moments over that polygon (Stokes-theorem-based, doable by hand).

**Recommendation for 2D dfmm:** Use GeometryOps.jl for the geometric overlap step; implement polynomial moment integration on top via the divergence theorem (each Bernstein/monomial moment over a polygon is a closed-form expression in vertex positions). This is a few hundred lines of code, dimension-2-only.

### 6.3 r3d (the C library) — the elephant in the room

**Status:** Production C/C++. Powell & Abel 2015 J. Comp. Phys. The reference for exact polynomial moments over arbitrary polyhedral intersections in 2D and 3D.

**Honest assessment:** No Julia-native equivalent exists for the *full* r3d capability. PolyClipper (LLNL) is a C++ reimplementation that drops r3d's hard limits (max vertices, fixed-degree mesh assumptions) but only does zeroth and first moments, not arbitrary polynomial. The Interface Reconstruction Library (IRL) is faster than r3d on some configurations but is also C++.

**Three options for dfmm-2d:**

**Option A: Wrap r3d via CxxWrap.jl or BinaryBuilder.jl.** Build r3d as a binary artifact (`r3d_jll`), wrap with Julia bindings. Pros: production-tested code, full feature set. Cons: foreign-function interface, can't use Enzyme.jl on it directly, debugging is harder.

**Option B: Reimplement in pure Julia.** The 2D version is tractable — a few hundred lines for polygon clipping with arbitrary polynomial moment integration via the divergence theorem. The 3D version is substantially more work (Powell-Abel paper has the algorithm worked out, but the corner cases require care).

**Option C: For 2D only, use GeometryOps.jl for the overlap polygon and add a polynomial-moment-integration module on top.** This avoids reimplementing the geometry algorithms (which are nontrivial — Foster-Hormann robustness corner cases) but builds the moment integration we need.

**Recommendation:** **Option C for dfmm-2d**, with an eye toward Option A if performance demands. The reasoning: GeometryOps.jl is well-tested and handles the geometric corner cases robustly; the polynomial-moment integration is a finite-degree algebra problem that's tractable to implement and verify. For dfmm-3d later, revisit — wrapping r3d may become necessary.

For curved-element overlaps (cubic edges in 2D, the case where the dimension-lift approach we discussed in the design document is needed), Option C still works: GeometryOps.jl handles the linear approximation, and the dimension-lifted polynomial-moment integral is computed in the lifted space using the divergence theorem on flat polytopes — exactly what the design document called for.

### 6.4 GeometricalPredicates.jl

**Status:** Production. JuliaGeometry organization. Robust orientation tests, point-in-polygon, area/volume computation, Peano-Hilbert ordering.

**Recommendation:** Use for the low-level geometric robustness primitives.

### 6.5 HOHQMesh.jl

**Status:** Production. Trixi-framework. Curvilinear quad/hex mesh generation.

**Recommendation:** Use if we need curved-boundary domains. For periodic 2D wave-pool and Sod-like tests, not needed.

---

## 7. Stochastic processes (variance-gamma noise)

### 7.1 Distributions.jl

**Status:** Production. JuliaStats. Comprehensive. Used universally.

**Honest assessment:** Variance-gamma is **not** in Distributions.jl directly. Generalized hyperbolic is in QuadraticFormsMGHyp.jl (which contains variance-gamma as λ > 0 special case) but that package is from 2020 and focused on tail probabilities, not sampling.

**Recommendation:** Implement variance-gamma sampling directly via the Gamma + Normal variance-mixture construction:

```julia
using Distributions
function rand_variance_gamma(rng, λ, θ)
    V = rand(rng, Gamma(λ, θ))
    return sqrt(V) * randn(rng)
end
```

This is the construction from the v3 note (Theorem at lines 293-300). Trivial to implement, easy to vectorize for batch sampling, GPU-compatible. For pdf/cdf evaluation (needed for likelihood-based diagnostics), use the Bessel-function form from v3 eq. (5).

### 7.2 StochasticDiffEq.jl

**Status:** Production. SciML.

**Relevant content:** Stochastic integrators for SDE problems, including Stratonovich and Itô forms.

**Recommendation:** **Don't use this for the per-cell stochastic injection.** The injection is a discrete-time event triggered by the deterministic step's compression diagnostic, not a continuous SDE integration. Reach for StochasticDiffEq.jl only if we later want a continuous-time formulation of the LES closure for theoretical analysis.

---

## 8. Visualization and I/O

### 8.1 Makie.jl

**Status:** Production. Massive ecosystem. CairoMakie (publication-quality 2D), GLMakie (interactive 3D), WGLMakie (web).

**Relevant content:** Interactive plotting, GPU-accelerated rendering, full Makie/Observables reactivity, integration with Trixi.jl, FerriteViz.jl, and Meshes.jl.

**Recommendation:** Default visualization for development and paper figures. CairoMakie for publication PDFs.

### 8.2 WriteVTK.jl

**Status:** Production. JuliaVTK organization.

**Relevant content:** Writes VTK XML files (`.vti`, `.vtu`, `.vtp`, `.pvd` for time series). Compatible with ParaView for everything serious.

**Recommendation:** Use for all simulation output. Time-series via `paraview_collection` blocks. ParaView remains the most capable tool for big 3D fields and complex AMR datasets.

### 8.3 Trixi2Vtk.jl

**Status:** Production. Trixi-framework.

**Relevant content:** Converts Trixi.jl output to VTK.

**Recommendation:** If we follow Trixi.jl's output format conventions (HDF5 with metadata), Trixi2Vtk works for free. Otherwise build our own thin layer over WriteVTK.

### 8.4 HDF5.jl + JLD2.jl

**Status:** Production.

**Recommendation:** HDF5.jl for the main output format (interoperable with Python analysis scripts). JLD2.jl for checkpoint/restart (Julia-native, faster for full-state serialization).

---

## 9. Small-tensor performance

### 9.1 StaticArrays.jl

**Status:** Production. JuliaArrays organization. Foundation for high-performance small-array code in Julia.

**Relevant content:** SVector, SMatrix, SArray (immutable, stack-allocated), MVector, MMatrix, MArray (mutable, still stack-allocated for small sizes). 10×–100× speedups vs. Base.Array for sizes < ~100 elements. Specialized det/inv/eigen/cholesky for fixed dimensions.

**Recommendation:** Critical for dfmm-2d. Every per-cell computation (4×4 Cholesky factor, 2×2 strain rate, 2×2 stress tensor) should use StaticArrays. The Cholesky decomposition for the 4×4 covariance has a closed-form expression specialized for the block structure that should be hand-written for speed.

### 9.2 Tensors.jl

**Status:** Production. Ferrite-FEM organization.

**Relevant content:** Symmetric tensors with realizability constraints, automatic differentiation through tensor operations, common operations (otimes, inner products) at fixed dimension.

**Recommendation:** Useful for the deviatoric stress sector and the principal-axis decomposition. May or may not be needed depending on how clean we can make the StaticArrays version.

---

## 10. FEM-related (for inspiration; we won't reuse directly)

### 10.1 Gridap.jl

**Status:** Production. Comparable performance to FEniCS. Composes with PartitionedArrays.jl for distributed.

**Relevant content:** General FEM framework. Custom variational forms, AD-compatible.

**Honest assessment:** Heavyweight — designed for general PDE solving via classical FEM. Doesn't match our variational moment scheme.

**Recommendation:** Don't use. **Do** read Gridap.jl's design paper (Verdugo & Badia 2022, Computer Physics Communications) — the Julia-native FEM design ideas there inform our own type design.

### 10.2 Ferrite.jl

**Status:** Production. Active. Maintained by the Ferrite-FEM organization.

**Relevant content:** Lower-level FEM toolbox. More flexible than Gridap; users assemble systems explicitly. Good for custom variational integrators.

**Honest assessment:** More aligned with our needs than Gridap for the *Lagrangian* simplicial mesh handling. The polynomial reconstruction within each Lagrangian cell could be done via Ferrite's interpolation framework if we squint hard.

**Recommendation:** Read Ferrite.jl's source for the cell-iterator and value-extraction patterns. Don't depend on it as the runtime; implement our own specialized version.

### 10.3 GalerkinToolkit.jl

**Status:** Active development. Re-write of Gridap's core based on a form compiler.

**Recommendation:** Watch the project; not yet stable enough to depend on for dfmm-2d. The form-compiler approach may eventually be the right way to handle our action functional, but it's premature.

---

## 11. Distributed parallelism

### 11.1 MPI.jl

**Status:** Production. JuliaParallel. Universal for HPC.

**Relevant content:** Standard MPI bindings. Works on essentially every cluster. CUDA-aware MPI supported.

**Recommendation:** Use for inter-node parallelism in production runs.

### 11.2 PartitionedArrays.jl

**Status:** Production. Used by Gridap and GalerkinToolkit.

**Relevant content:** Distributed sparse matrices, distributed vectors, algebraic multigrid preconditioners. Pure-Julia alternative to PETSc.

**Recommendation:** Use if we need distributed linear solves at scale. For Milestone 1–2 (single node, possibly multi-GPU), not needed.

### 11.3 PETSc.jl

**Status:** Production. JuliaParallel. Wraps PETSc with MPI support.

**Recommendation:** Alternative to PartitionedArrays.jl. PETSc has more mature linear solvers (especially for ill-conditioned systems) and AMG preconditioners. Use if PartitionedArrays runs into convergence issues at scale.

---

## 12. Things we'll likely need to write ourselves

After the survey, the following pieces don't have ready-to-use Julia packages and will need new code:

1. **Polynomial-moment integration over polygons** (the moment-extraction half of r3d's functionality in 2D). Use GeometryOps.jl for the overlap polygon, write our own moment integration. Few hundred lines.

2. **The variational integrator for the connection-form Cholesky sector** with weighted symplectic form ω = α²dα∧dβ. Read Kraus's papers on degenerate-Lagrangian variational integrators; specialize for our case. Custom implementation in our package.

3. **Polynomial reconstruction with positivity constraints** (Bernstein-basis certificate for det F > 0; exp-parameterized Cholesky diagonals). Few hundred lines, specialized for triangles.

4. **The Bayesian remap operator** (law of total covariance applied through the geometric overlap). Once we have the polynomial moments, this is straightforward bookkeeping.

5. **The action-error AMR indicator** (∆S_cell from v2 eq. 348-349). Specific to our action functional; few hundred lines on top of P4est.jl.

6. **Variance-gamma random sampling and pdf evaluation** (handful of lines).

7. **The discrete parallel transport operator for charge-q fields** between time slices. Specific to our connection-form structure.

These are all tractable. Items 2 and 3 are the hardest and need the most numerical-analysis care.

---

## 13. The recommended technology stack

Putting this all together, the recommended Milestone-1 stack (1D dfmm-2d code in Julia, regression against existing dfmm Python+Numba) is:

**Core numerics:**
- StaticArrays.jl (small-tensor performance)
- NonlinearSolve.jl + LinearSolve.jl (implicit Newton)
- Enzyme.jl + DifferentiationInterface.jl (AD)
- Distributions.jl + custom variance-gamma (stochastic injection)

**Geometry:**
- GeometryOps.jl + custom polynomial moments (Bayesian remap)
- GeometricalPredicates.jl (robust primitives)

**Time integration:**
- Custom variational integrator (referencing GeometricIntegrators.jl)
- OrdinaryDiffEq.jl symplectic methods for the bulk sector (where applicable)

**I/O and visualization:**
- HDF5.jl + JLD2.jl
- WriteVTK.jl + ParaView
- Makie.jl for development figures

**Mesh (1D milestone):**
- Custom mass-coordinate segment data structure
- (No AMR needed at Milestone 1; stays uniform refinement.)

For Milestone 3 (2D code), add:

**GPU portability:**
- KernelAbstractions.jl + CUDA.jl/Metal.jl/AMDGPU.jl backends
- AcceleratedKernels.jl for primitives

**AMR:**
- P4est.jl for the Eulerian quadtree
- Custom triangle-mesh data structure for the Lagrangian side

**Distributed (Milestone 4 if scaling demands):**
- MPI.jl + ImplicitGlobalGrid.jl for the regular Eulerian sector
- ParallelStencil.jl for stencil ops

This is a tractable stack. Most pieces are mature; the few new pieces (variational integrator, polynomial-moment integration) are exactly the genuine methods work that should be ours rather than a library's.

---

## 14. A note on rusting vs. Julia-rs

The original handoff prompt for the 1D agent suggested Rust as the implementation language (matching Tom's existing dfmm preference and the impress suite tooling). The Julia survey here suggests Julia is at least as well suited.

**Honest comparison for dfmm-2d specifically:**

| Aspect | Rust | Julia |
|---|---|---|
| Variational integrator stencils | Hand-written, ndarray, fast | Hand-written, StaticArrays, fast |
| Implicit Newton solve | Custom or `argmin`, less mature | NonlinearSolve.jl, production |
| Automatic differentiation | `enzyme-rs` (limited) or finite-diff | Enzyme.jl (production for our cases) |
| GPU portability | wgpu, cust, naga (early) | KernelAbstractions.jl (production, all 4 vendors) |
| AMR (quadtree) | Custom or wrap p4est | P4est.jl (production) |
| Polygon clipping with moments | None ready, write custom | GeometryOps.jl + custom moments |
| Visualization | None ready, write custom | Makie.jl + WriteVTK.jl (production) |
| Iteration speed (REPL/Pluto) | Compile-then-run | REVISE-driven hot-reload |
| Production deploy | Static binary, easy | Julia + sysimage, slightly harder |

**Recommendation:** **For dfmm-2d, Julia is the right choice.** The combination of NonlinearSolve.jl + Enzyme.jl + KernelAbstractions.jl + P4est.jl + Makie.jl is genuinely state-of-the-art and would take substantial effort to replicate in Rust. The variational integrator and polynomial-moment integration we'd write ourselves anyway, so no benefit to Rust on that front. Iteration speed during research (REVISE.jl hot-reload, Pluto.jl notebooks) is a real productivity multiplier in the design-and-debug phase.

The Rust ecosystem in the impress suite is a separate matter and remains the right choice for those tools (file-format I/O, CLI tools, performance-critical infrastructure). dfmm-2d is research code where the specific Julia ecosystem advantages dominate.

---

## 15. References

Below is the package list with URLs, alphabetically:

- AcceleratedKernels.jl: https://github.com/JuliaGPU/AcceleratedKernels.jl
- AMDGPU.jl: https://github.com/JuliaGPU/AMDGPU.jl
- CUDA.jl: https://github.com/JuliaGPU/CUDA.jl
- DifferentiationInterface.jl: https://github.com/JuliaDiff/DifferentiationInterface.jl
- DifferentialEquations.jl / OrdinaryDiffEq.jl: https://github.com/SciML/DifferentialEquations.jl
- Distributions.jl: https://github.com/JuliaStats/Distributions.jl
- Enzyme.jl: https://github.com/EnzymeAD/Enzyme.jl
- Ferrite.jl: https://github.com/Ferrite-FEM/Ferrite.jl
- ForwardDiff.jl: https://github.com/JuliaDiff/ForwardDiff.jl
- GalerkinToolkit.jl: https://github.com/gridap/GalerkinToolkit.jl
- GeometricIntegrators.jl: https://github.com/JuliaGNI/GeometricIntegrators.jl
- GeometricalPredicates.jl: https://github.com/JuliaGeometry/GeometricalPredicates.jl
- GeometryOps.jl: https://github.com/JuliaGeo/GeometryOps.jl
- Gridap.jl: https://github.com/gridap/Gridap.jl
- HDF5.jl: https://github.com/JuliaIO/HDF5.jl
- HOHQMesh.jl: https://github.com/trixi-framework/HOHQMesh.jl
- ImplicitGlobalGrid.jl: https://github.com/eth-cscs/ImplicitGlobalGrid.jl
- JLD2.jl: https://github.com/JuliaIO/JLD2.jl
- KernelAbstractions.jl: https://github.com/JuliaGPU/KernelAbstractions.jl
- LinearSolve.jl: https://github.com/SciML/LinearSolve.jl
- Makie.jl: https://github.com/MakieOrg/Makie.jl
- Meshes.jl: https://github.com/JuliaGeometry/Meshes.jl
- Metal.jl: https://github.com/JuliaGPU/Metal.jl
- MPI.jl: https://github.com/JuliaParallel/MPI.jl
- NonlinearSolve.jl: https://github.com/SciML/NonlinearSolve.jl
- oneAPI.jl: https://github.com/JuliaGPU/oneAPI.jl
- P4est.jl: https://github.com/trixi-framework/P4est.jl
- P4estTypes.jl: https://github.com/trixi-framework/P4estTypes.jl
- ParallelStencil.jl: https://github.com/omlins/ParallelStencil.jl
- PartitionedArrays.jl: https://github.com/PartitionedArrays/PartitionedArrays.jl
- PETSc.jl: https://github.com/JuliaParallel/PETSc.jl
- Polyhedra.jl: https://github.com/JuliaPolyhedra/Polyhedra.jl
- PolygonAlgorithms.jl: https://github.com/LiorSinai/PolygonAlgorithms.jl
- QuadraticFormsMGHyp.jl: https://github.com/sleinen/QuadraticFormsMGHyp.jl
- StaticArrays.jl: https://github.com/JuliaArrays/StaticArrays.jl
- Tensors.jl: https://github.com/Ferrite-FEM/Tensors.jl
- Trixi.jl: https://github.com/trixi-framework/Trixi.jl
- WriteVTK.jl: https://github.com/JuliaVTK/WriteVTK.jl
- r3d (C library, candidate for FFI wrap): https://github.com/devonmpowell/r3d

External:
- Kraus, M., "Projected Variational Integrators for Degenerate Lagrangian Systems," arXiv:1708.07356.
- Kraus, M. & Tyranowski, T. M., "Variational Integrators for Stochastic Dissipative Hamiltonian Systems," arXiv:1909.07202, Journal of Computational Dynamics.
- Powell, D. M. & Abel, T., "An exact general remeshing scheme applied to physically conservative voxelization," J. Comput. Phys. 297 (2015) 340–356.
- Pal, A. et al., "NonlinearSolve.jl: High-Performance and Robust Solvers for Systems of Nonlinear Equations in Julia," arXiv:2403.16341.
- Burstedde, C., Wilcox, L. C. & Ghattas, O., "p4est: Scalable Algorithms for Parallel Adaptive Mesh Refinement on Forests of Octrees," SIAM J. Sci. Comput. 33 (2011) 1103–1133.

---

## 16. Honest summary

Six packages are absolutely critical and production-grade for our needs:
**StaticArrays.jl, NonlinearSolve.jl, Enzyme.jl, KernelAbstractions.jl, P4est.jl, Makie.jl + WriteVTK.jl.** Build on these.

Four packages provide useful patterns to study but shouldn't be runtime dependencies:
**Trixi.jl** (architecture for AMR + DG), **GeometricIntegrators.jl** (variational integrator reference), **Gridap.jl/Ferrite.jl** (FEM design patterns).

Three pieces will need custom implementation:
**Polynomial moments over polygons** (using GeometryOps.jl for the geometry), **the connection-form variational integrator** (using GeometricIntegrators.jl as reference), and **action-error AMR indicator** (on top of P4est.jl).

One open question with strategic consequences:
**Wrap r3d via FFI vs. reimplement in Julia for 2D.** Recommendation: pure Julia for 2D (tractable), revisit for 3D.

Julia is the right language for this. The variational integrator and polynomial-moment work would be ours regardless of language; the surrounding ecosystem (Newton, AD, GPU portability, AMR, visualization) is genuinely better in Julia than in Rust or Python for our specific needs.
