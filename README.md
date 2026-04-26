# dfmm — dual frame moment method

Two-stage development of the dfmm scheme in one repository.

|  | Stage | Language | Dimension | Status |
|---|---|---|---|---|
| `py-1d/` | Reference | Python + Numba | 1D | Reproduces the methods paper |
| top level | Unified variational scheme | Julia | 1D → 2D | **Milestones 1 + 2 complete** (2044 tests + 1 deferred) |

## Status snapshot — Milestones 1 + 2

**2044 + 1 deferred tests** on main, full `Pkg.test()` ~3m. **All 1D Tier-A and Tier-B regression items verified.** The methods paper's three central scientific claims — **Tier B.2 cold-limit unification**, **Tier B.5 passive-scalar exactness**, and **Tier B.3 action-based AMR cell-savings** — are all verified.

| Phase | Tier | Headline result |
|---|---|---|
| 1 | — | Cholesky-sector variational integrator, symplectic |
| 2 | — | Bulk + entropy coupling, mass/momentum exact, acoustic 0.64% |
| 3 | **B.2** | **Cold-limit unification verified** (Zel'dovich match 2.57e-8) |
| 4 | B.1 | Energy drift 5.6e-9 over 10⁵ steps (passes literal bound; t¹ secular noted) |
| 5 / 5b | A.1 | Sod qualitative dfmm Fig. 2 match; L∞ ~10-20%, L1 ~3-4%; opt-in tensor-q viscosity available |
| 6 | A.2 | Cold sinusoid τ-scan across 6 decades; ρ_err ~1e-4 vs Zel'dovich |
| 7 | A.3 | Steady shock R-H to ≥3 decimals at $M_1 \in \{1.5, 2, 3, 5, 10\}$ |
| 8 / 9 | B.4 | Variance-gamma noise injection; self-consistency monitor working |
| 11 | **B.5** | **Tracer-exactness verified — L∞ change = 0.0 literally** |
| **M2-1** | **B.3** | **Action-based AMR: 37.8% cell savings vs gradient AMR** |
| **M2-2** | **B.6** | **Multi-tracer wave-pool: bit-exact under stochastic; 25-114× over Eulerian** |
| **M2-3** | **Open #4** | **Realizability projection: wave-pool reaches 12,000 steps (was 1222)** |
| **M2-4** | — | **Methods-paper text corrections applied** (§3.5, §3.2, §3.1, §10.3 B.1) |

See `reference/MILESTONE_1_STATUS.md` for the full synthesis,
`reference/notes_methods_paper_corrections.md` for paper-edit items
surfaced during implementation, and `reference/notes_performance.md`
for timings.

## Layout

- `py-1d/` — the original 1D Python+Numba implementation with its
  paper, experiments, tests, and data. Self-contained; setup with
  `pip install -e py-1d/` and run tests with `pytest py-1d/tests/`.
- `HANDOFF.md` — the original design handoff. Reading order for the
  design corpus and the Milestone-1 plan.
- `specs/` — canonical design (the methods paper) and the Julia
  ecosystem survey that fixes the implementation stack.
- `design/` — the three iterative action notes (v1, v2, v3 FINAL).
  v3 supersedes v2 supersedes v1.
- `reference/` — Milestone-1 implementation notes, status synthesis,
  performance data, paper-correction items, and validation outputs.
  All `notes_phase{N}_*.md` are per-phase implementation journals.
- `reference/golden/` — Tier-A regression-target HDF5 snapshots
  generated from `py-1d/`; schema in `reference/golden/SCHEMA.md`.
- `reference/figs/` — published figures (one per phase deliverable).
- `Project.toml`, `src/`, `test/` — the Julia package `dfmm`.
- `experiments/` — production drivers (one per Tier-A/B test).
- `scripts/` — reproducer utilities (`make_goldens.jl`, etc.).
- `Makefile` — top-level targets (`make goldens` regenerates Tier-A goldens).

## Quick start

```bash
# (1) Reproduce the 1D Python paper (regression target)
cd py-1d
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
make paper
cd ..

# (2) Activate the Julia package and run the Milestone-1 suite
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'

# (3) Reproduce a specific experiment
julia --project=. experiments/A2_cold_sinusoid.jl
julia --project=. experiments/A1_sod_with_q.jl
```

Julia 1.11 or later is recommended.

## Reading order for new agents

1. `HANDOFF.md` (the original design handoff).
2. `reference/MILESTONE_1_STATUS.md` (where things stand).
3. `reference/MILESTONE_1_PLAN.md` (the prospective phase plan; phases 1-11 + 5b done).
4. `reference/notes_methods_paper_corrections.md` (open paper-edit items).
5. `reference/notes_phase{1,2,3,...}_*.md` (per-phase journals as needed).

## License

MIT. See `LICENSE`.
