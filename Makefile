# Top-level dfmm-2d Makefile
#
# This is intentionally minimal — the bulk of the build/test machinery
# lives in Julia's `Pkg`. The only target here is `goldens`, which
# regenerates the Tier-A regression-target HDF5 files in
# `reference/golden/` by shelling out to py-1d. See
# `reference/golden/SCHEMA.md` and `scripts/make_goldens.jl`.
#
# `py-1d/` has its own Makefile for the Python project; we do not
# shadow it here.

JULIA ?= julia

.PHONY: goldens goldens-clean test help

help:
	@echo "Targets:"
	@echo "  goldens         regenerate all six Tier-A HDF5 goldens"
	@echo "  goldens-clean   remove generated goldens (keep .regen stubs)"
	@echo "  test            run the Julia test suite"

goldens:
	$(JULIA) --project=. scripts/make_goldens.jl

goldens-clean:
	@echo "Removing reference/golden/*.h5"
	@rm -f reference/golden/*.h5

test:
	$(JULIA) --project=. -e 'using Pkg; Pkg.test()'
