"""Unit tests for the ghost-cell boundary-condition appliers."""
import numpy as np
import pytest

from dfmm.schemes._common import IDX_RHO, IDX_MOM, IDX_M3
from dfmm.schemes.boundaries import (
    apply_periodic, apply_transmissive, apply_reflective, apply_dirichlet,
    apply_mixed, pad_with_ghosts, unpad_ghosts, bc_code,
    BC_PERIODIC, BC_TRANSMISSIVE, BC_REFLECTIVE, BC_DIRICHLET,
)


def _fresh(N=8, n_ghost=2):
    """Build a ghost-padded state with distinct interior values."""
    n_fields = 8
    U = np.arange(n_fields * N, dtype=float).reshape(n_fields, N) + 1
    return pad_with_ghosts(U, n_ghost), U


def test_pad_and_unpad_roundtrip():
    U = np.arange(8 * 6, dtype=float).reshape(8, 6) + 1
    for n_ghost in (1, 2, 3):
        U_ghost = pad_with_ghosts(U, n_ghost)
        assert U_ghost.shape == (8, 6 + 2 * n_ghost)
        U_back = unpad_ghosts(U_ghost, n_ghost)
        assert np.array_equal(U_back, U)


def test_periodic_copies_opposite_edges():
    U_ghost, U_in = _fresh(N=8, n_ghost=2)
    apply_periodic(U_ghost, n_ghost=2)
    # Left ghost 0 = interior cell N-2 (second-to-last)
    assert np.array_equal(U_ghost[:, 0], U_in[:, 6])
    assert np.array_equal(U_ghost[:, 1], U_in[:, 7])
    # Right ghost 0 = interior cell 0
    assert np.array_equal(U_ghost[:, 10], U_in[:, 0])
    assert np.array_equal(U_ghost[:, 11], U_in[:, 1])


def test_transmissive_copies_nearest_interior():
    U_ghost, U_in = _fresh(N=8, n_ghost=2)
    apply_transmissive(U_ghost, n_ghost=2)
    # All left ghosts == first interior cell
    for g in range(2):
        assert np.array_equal(U_ghost[:, g], U_in[:, 0])
    # All right ghosts == last interior cell
    for g in range(2):
        assert np.array_equal(U_ghost[:, 2 + 8 + g], U_in[:, 7])


def test_reflective_mirrors_even_flips_odd_moments():
    U_ghost, U_in = _fresh(N=8, n_ghost=2)
    apply_reflective(U_ghost, n_ghost=2)
    # Left ghost 1 mirrors interior cell 0; ghost 0 mirrors interior cell 1
    for field in range(8):
        sign = -1 if field in (IDX_MOM, IDX_M3) else 1
        # Left
        assert U_ghost[field, 1] == sign * U_in[field, 0]
        assert U_ghost[field, 0] == sign * U_in[field, 1]
        # Right ghost 0 mirrors interior cell 7 (last); ghost 1 mirrors interior cell 6
        assert U_ghost[field, 10] == sign * U_in[field, 7]
        assert U_ghost[field, 11] == sign * U_in[field, 6]


def test_dirichlet_fills_fixed_state():
    U_ghost, _ = _fresh(N=8, n_ghost=2)
    left_state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    right_state = np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0])
    apply_dirichlet(U_ghost, 2, left_state, right_state)
    for g in range(2):
        assert np.array_equal(U_ghost[:, g], left_state)
        assert np.array_equal(U_ghost[:, 2 + 8 + g], right_state)


def test_apply_mixed_periodic_both():
    U_ghost, U_in = _fresh(N=8, n_ghost=2)
    apply_mixed(U_ghost, n_ghost=2, bc_left='periodic', bc_right='periodic')
    assert np.array_equal(U_ghost[:, 0], U_in[:, 6])
    assert np.array_equal(U_ghost[:, 11], U_in[:, 1])


def test_apply_mixed_transmissive_reflective():
    """Left transmissive, right reflective — each side independent."""
    U_ghost, U_in = _fresh(N=8, n_ghost=2)
    apply_mixed(U_ghost, n_ghost=2,
                bc_left='transmissive', bc_right='reflective')
    # Left ghosts = first interior cell
    for g in range(2):
        assert np.array_equal(U_ghost[:, g], U_in[:, 0])
    # Right ghosts: mirrored with odd-moment flip
    for field in range(8):
        sign = -1 if field in (IDX_MOM, IDX_M3) else 1
        assert U_ghost[field, 10] == sign * U_in[field, 7]
        assert U_ghost[field, 11] == sign * U_in[field, 6]


def test_apply_mixed_rejects_periodic_on_one_side():
    """Periodic on only one side is ill-defined."""
    U_ghost, _ = _fresh()
    with pytest.raises(ValueError):
        apply_mixed(U_ghost, 2, 'periodic', 'transmissive')


def test_bc_code_lookup():
    assert bc_code('periodic') == BC_PERIODIC
    assert bc_code('transmissive') == BC_TRANSMISSIVE
    assert bc_code('reflective') == BC_REFLECTIVE
    assert bc_code('dirichlet') == BC_DIRICHLET
    assert bc_code('PERIODIC') == BC_PERIODIC  # case-insensitive
    with pytest.raises(ValueError):
        bc_code('nonsense')
