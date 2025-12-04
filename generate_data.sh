#!/bin/bash

# Use venv python explicitly so it works when run from PowerShell -> bash/WSL.

for idx in $(seq 1 10); do
  rounds=$((2 * idx))
  echo "Starting num_rounds=${rounds} (idx=${idx})"
  python syndrome_data_bad_qubit.py --distance 5 --num_rounds "${rounds}" --num_shots 100000
  echo "Finished num_rounds=${rounds}"
done
