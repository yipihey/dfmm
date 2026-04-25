# dfmm — dual frame moment method

Two-stage development of the dfmm scheme in one repository.

| | Stage | Language | Dimension | Status |
|---|---|---|---|---|
| `py-1d/` | Reference | Python + Numba | 1D | Reproduces the methods paper |
| top level | Unified variational scheme | Julia | 1D → 2D | Under development (Milestone 1) |

## Layout

- `py-1d/` — the original 1D Python+Numba implementation with its
  paper, experiments, tests, and data. Self-contained; setup with
  `pip install -e py-1d/` and run tests with `pytest py-1d/tests/`.
- `HANDOFF.md` — design handoff README from April 2026. **Read first.**
  It specifies the design corpus reading order and the Milestone-1 plan.
- `specs/` — canonical design (the methods paper) and the Julia
  ecosystem survey that fixes the implementation stack.
- `design/` — the three iterative action notes (v1, v2, v3 FINAL) that
  produced the variational design. v3 supersedes v2 supersedes v1.
- `reference/` — empty; populated by Milestone-1 work with
  implementation notes, performance data, and validation outputs.
- `Project.toml`, `src/`, `test/` — the Julia package `dfmm`
  (working name during development).

## Quick start

```bash
# (1) Reproduce the 1D Python paper (regression target)
cd py-1d
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest tests/
make paper

# (2) Activate the Julia package (top level)
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Julia 1.11 or later is recommended (per the handoff sanity checks);
1.10 is the minimum supported.

## License

MIT. See `LICENSE`.
