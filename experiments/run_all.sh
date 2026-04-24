#!/usr/bin/env bash
# Run all reproduction experiments in order.
# Total runtime approximately 25-30 minutes (full) or 2-3 minutes (SMALL=1).
set -e

here="$(dirname "$0")"
for script in \
    01_sod_validation.py \
    02_sine_shell_crossing.py \
    03_steady_shock.py \
    04_shock_thickness_scans.py \
    05_two_fluid_dust_gas.py \
    06_maxent_kappa.py \
    07_eion_equilibration.py \
    10_kmles_wavepool.py \
    11_kmles_calibrate.py \
    12_kmles_correlation_scan.py \
    13_kmles_energy_conservation.py \
    14_kmles_ensemble.py ; do
    echo ""
    echo "=== Running $script ==="
    python3 "$here/$script"
done
echo ""
echo "All experiments complete."
