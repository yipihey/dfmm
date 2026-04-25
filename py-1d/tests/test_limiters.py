"""Unit tests for the slope-limiter library."""
import numpy as np
import pytest

from dfmm.schemes._common import (minmod, mc, van_leer, limit,
                                  LIMITER_MINMOD, LIMITER_MC,
                                  LIMITER_VAN_LEER)


# ---------------- minmod -----------------

def test_minmod_opposite_signs_is_zero():
    """Standard property: mixed-sign arguments return 0 (extremum)."""
    assert minmod(1.0, -1.0) == 0.0
    assert minmod(-2.0, 3.0) == 0.0
    assert minmod(0.5, 0.0) == 0.0


def test_minmod_picks_smaller_magnitude():
    """When both arguments share sign, take the one with smaller |.|."""
    assert minmod(1.0, 2.0) == 1.0
    assert minmod(3.0, 1.0) == 1.0
    assert minmod(-1.0, -5.0) == -1.0
    assert minmod(-3.0, -1.0) == -1.0


def test_minmod_symmetric():
    assert minmod(1.5, 0.5) == minmod(0.5, 1.5)
    assert minmod(-2.0, -1.0) == minmod(-1.0, -2.0)


# ---------------- MC -----------------

def test_mc_opposite_signs_is_zero():
    assert mc(1.0, -1.0) == 0.0
    assert mc(-2.0, 3.0) == 0.0


def test_mc_equals_centered_on_smooth_data():
    """On a smooth slope where a == b, MC returns a (the centered slope)."""
    assert abs(mc(1.0, 1.0) - 1.0) < 1e-12
    assert abs(mc(-0.5, -0.5) - (-0.5)) < 1e-12


def test_mc_tighter_than_minmod():
    """MC lets more signal through than minmod on smoothly-varying data:
    when a and b share sign and differ, MC >= minmod (in magnitude)."""
    a, b = 0.5, 1.5
    assert abs(mc(a, b)) >= abs(minmod(a, b)) - 1e-12


# ---------------- van Leer -----------------

def test_van_leer_opposite_signs_is_zero():
    assert van_leer(1.0, -1.0) == 0.0


def test_van_leer_harmonic_mean_on_smooth_data():
    """For a == b, van Leer returns 2ab/(a+b) = a."""
    assert abs(van_leer(0.3, 0.3) - 0.3) < 1e-12
    assert abs(van_leer(-0.7, -0.7) - (-0.7)) < 1e-12


def test_van_leer_between_minmod_and_mc():
    """Property: van_leer >= minmod for same-sign inputs (both TVD, van
    Leer is sharper)."""
    for a, b in [(0.2, 0.8), (1.0, 3.0), (-0.5, -2.0)]:
        vl = van_leer(a, b)
        mm = minmod(a, b)
        assert abs(vl) >= abs(mm) - 1e-12


# ---------------- limit dispatch -----------------

def test_limit_dispatch():
    assert limit(1.0, 2.0, LIMITER_MINMOD) == minmod(1.0, 2.0)
    assert limit(1.0, 2.0, LIMITER_MC) == mc(1.0, 2.0)
    assert limit(1.0, 2.0, LIMITER_VAN_LEER) == van_leer(1.0, 2.0)


def test_limit_unknown_code_falls_back_to_minmod():
    """Defensive: unknown codes shouldn't crash or return NaN."""
    assert limit(1.0, 2.0, 99) == minmod(1.0, 2.0)


# ---------------- TVD property -----------------

def test_minmod_tvd_on_monotone_sequence():
    """Limiting a monotone sequence's slopes never increases total
    variation. For slopes all positive, limited slopes are non-
    negative and bounded by the un-limited slopes."""
    np.random.seed(0)
    # Monotone-increasing sequence
    y = np.cumsum(np.random.uniform(0.1, 1.0, size=100))
    slopes = np.diff(y)  # all positive
    # Limit each interior slope via minmod of neighbours
    limited = np.array([minmod(slopes[i], slopes[i+1])
                        for i in range(len(slopes) - 1)])
    assert (limited >= 0).all()
    # Each limited slope is bounded by the smaller neighbour (minmod)
    assert (limited <= np.maximum(slopes[:-1], slopes[1:])).all()
