#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 prepare.py "$@"
julia --startup-file=no --project=. -t 5 experiments/benchmark.jl "$@"
julia --startup-file=no --project=. -t 5 experiments/resampling.jl "$@"
julia --startup-file=no --project=. -t 1 experiments/all_nodes.jl "$@"
julia --startup-file=no --project=. -t 1 experiments/solver_baselines.jl "$@"
python3 report.py
