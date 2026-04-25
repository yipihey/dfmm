"""Unit tests for `dfmm.config.SimulationConfig`."""
import numpy as np
import pytest
from dataclasses import replace

from dfmm.config import SimulationConfig


def test_defaults_match_historical_values():
    cfg = SimulationConfig()
    assert cfg.rho_floor == 1e-30
    assert cfg.alpha_floor == 1e-15
    assert cfg.pressure_floor == 1e-8
    assert cfg.realizability_headroom == 0.999
    assert cfg.noise_ke_budget_fraction == 0.25
    assert cfg.cfl == 0.3
    assert cfg.tau == 1e-3
    assert cfg.bc_left == 'periodic' and cfg.bc_right == 'periodic'
    assert cfg.reconstruction == 'first'
    assert cfg.limiter == 'minmod'
    assert cfg.n_ghost == 1  # auto-selected for first-order


def test_muscl_selects_two_ghost_cells():
    cfg = SimulationConfig(reconstruction='muscl')
    assert cfg.n_ghost == 2


def test_explicit_n_ghost_overrides_default():
    cfg = SimulationConfig(n_ghost=3)
    assert cfg.n_ghost == 3


def test_replace_produces_new_instance():
    """Immutability via dataclasses.replace."""
    base = SimulationConfig()
    tight = replace(base, realizability_headroom=0.9)
    assert tight.realizability_headroom == 0.9
    assert base.realizability_headroom == 0.999  # unchanged


def test_invalid_cfl_raises():
    with pytest.raises(ValueError, match="cfl"):
        SimulationConfig(cfl=0)
    with pytest.raises(ValueError, match="cfl"):
        SimulationConfig(cfl=1.5)


def test_invalid_tau_raises():
    with pytest.raises(ValueError, match="tau"):
        SimulationConfig(tau=0)


def test_invalid_reconstruction_raises():
    with pytest.raises(ValueError, match="reconstruction"):
        SimulationConfig(reconstruction='cubic')


def test_invalid_limiter_raises():
    with pytest.raises(ValueError, match="limiter"):
        SimulationConfig(limiter='superbee')  # not yet supported


def test_periodic_must_be_symmetric():
    with pytest.raises(ValueError, match="periodic"):
        SimulationConfig(bc_left='periodic', bc_right='transmissive')
    with pytest.raises(ValueError, match="periodic"):
        SimulationConfig(bc_left='transmissive', bc_right='periodic')


def test_frozen_dataclass_disallows_field_mutation():
    cfg = SimulationConfig()
    with pytest.raises(Exception):
        cfg.cfl = 0.5  # frozen dataclass raises FrozenInstanceError
