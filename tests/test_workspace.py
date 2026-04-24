"""Unit tests for `dfmm.schemes._common.Workspace`."""
import numpy as np
import pytest

from dfmm.schemes._common import Workspace


def test_for_padded_state_shape_matches():
    """Workspace.Unew matches U_ghost; Fleft has one extra column."""
    U = np.zeros((8, 130))  # e.g. N=128 + 2*1 ghost cells
    ws = Workspace.for_padded_state(U)
    assert ws.Unew.shape == U.shape
    assert ws.Fleft.shape == (8, 131)


def test_for_padded_state_preserves_dtype():
    for dt in (np.float64, np.float32):
        U = np.zeros((8, 34), dtype=dt)
        ws = Workspace.for_padded_state(U)
        assert ws.Unew.dtype == dt
        assert ws.Fleft.dtype == dt


def test_workspace_is_mutable_dataclass():
    """Allowing mutation lets callers swap buffers for ping-pong."""
    U = np.zeros((8, 10))
    ws = Workspace.for_padded_state(U)
    new = np.ones_like(ws.Unew)
    ws.Unew = new
    assert ws.Unew is new


def test_scratch_buffers_are_separate_arrays():
    """Unew and Fleft are distinct allocations (not views of each other)."""
    U = np.zeros((8, 10))
    ws = Workspace.for_padded_state(U)
    ws.Unew[0, 0] = 1.0
    ws.Fleft[0, 0] = 2.0
    assert ws.Unew[0, 0] == 1.0
    assert ws.Fleft[0, 0] == 2.0
