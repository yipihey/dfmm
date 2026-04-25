"""Regression snapshot for the wave-pool stepper.

Runs the cholesky periodic HLL stepper for a fixed number of steps on
a fixed seed and checks the resulting state against a committed
checksum. Numerical changes (anything that alters bit-level output)
will trip this test.

If an intentional change to the numerics is made, regenerate the
expected checksum by running the block at the bottom manually and
update `EXPECTED_SHA256`. Include the reason in the commit message so
`git log tests/test_regression_snapshot.py` tells the story.

This test GATES the refactors in the 4-phase roadmap. If a refactor
is supposed to be numerically-equivalent, this test should pass
unchanged; if it's numerically different, update the hash with a
commit explaining why.
"""
import hashlib
import numpy as np
import pytest

from dfmm.setups.wavepool import make_wave_pool_ic
from dfmm.schemes.cholesky import hll_step_periodic, max_signal_speed

EXPECTED_SHA256 = "34a305164a3284a67837eb2204c3189ff8259cdb7793a07879724757075517b9"
EXPECTED_SUM = 331.8295236387366
EXPECTED_MAX_ABS = 3.4648201836921833


def _run_reference():
    """The reference wave-pool protocol. Keep in lockstep with whatever
    produced the checksum in this file — if this protocol changes, the
    checksum must be regenerated."""
    U, x = make_wave_pool_ic(N=128, u0=1.0, P0=0.1, seed=42)
    dx = x[1] - x[0]
    cfl = 0.3
    for _ in range(100):
        smax = max_signal_speed(U)
        dt = cfl * dx / smax
        U = hll_step_periodic(U, dx, dt, tau=1e-3)
    return U


def test_wavepool_bitwise_regression():
    """Bit-identical reproduction after 100 wave-pool steps."""
    U = _run_reference()
    h = hashlib.sha256(U.tobytes()).hexdigest()
    assert h == EXPECTED_SHA256, (
        f"Wave-pool snapshot diverged from {EXPECTED_SHA256} to {h}.\n"
        f"  sum   = {U.sum():.10f}   (expected {EXPECTED_SUM:.10f})\n"
        f"  max|U| = {float(np.max(np.abs(U))):.10f}   (expected {EXPECTED_MAX_ABS:.10f})\n"
        "If this change is intentional, update EXPECTED_SHA256 and this\n"
        "docstring explaining why in the commit message.")


def test_wavepool_summary_stats():
    """Looser check — sum and max |U| — in case floating-point
    reordering is tolerated but gross drift is not."""
    U = _run_reference()
    # Tolerance ~1e-10 matches the conservation-test tolerance.
    assert abs(U.sum() - EXPECTED_SUM) < 1e-8
    assert abs(float(np.max(np.abs(U))) - EXPECTED_MAX_ABS) < 1e-8
