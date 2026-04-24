# Top-level Makefile for momentlag
#
# Usage:
#   make install    # install the package in editable mode
#   make test       # run smoke tests
#   make paper      # compile the paper
#   make figures    # regenerate all paper figures (slow, ~20 min)
#   make clean      # remove build artifacts
#
# Individual experiment scripts in experiments/ are invoked by number, e.g.:
#   python experiments/01_sod_validation.py
#
# See experiments/README.md for the full reproduction pipeline.

.PHONY: install test paper figures clean help

help:
	@echo "Common targets:"
	@echo "  install   install momentlag in editable mode"
	@echo "  test      run smoke tests"
	@echo "  paper     compile the paper PDF"
	@echo "  figures   regenerate all paper figures (takes ~20 minutes)"
	@echo "  clean     remove build artifacts"

install:
	pip install -e .

test:
	pytest tests/ -v

paper:
	$(MAKE) -C paper

figures:
	cd experiments && bash run_all.sh

clean:
	rm -rf build dist *.egg-info
	rm -rf momentlag/__pycache__ momentlag/*/__pycache__
	$(MAKE) -C paper clean
