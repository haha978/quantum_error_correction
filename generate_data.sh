#!/usr/bin/env bash
set -euo pipefail

# Use venv python explicitly so it works when run from PowerShell -> bash/WSL.
venv_py="/mnt/c/Users/leomo/OneDrive/Desktop/quantum_error_correction/venv/Scripts/python.exe"

for idx in $(seq 1 20); do
  rounds=$((15 * idx))
  echo "Starting num_rounds=${rounds} (idx=${idx})"
  "$venv_py" syndrome_data_uniform_noise.py --distance 5 --num_rounds "${rounds}" --num_shots 100000
  echo "Finished num_rounds=${rounds}"
done
