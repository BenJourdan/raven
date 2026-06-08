#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"$script_dir/profile.sh" oracle-graph
"$script_dir/profile.sh" oracle-coreset
"$script_dir/profile.sh" dummy
"$script_dir/profile.sh" leiden

if [[ "${PROFILE_PLAYGROUND:-0}" == "1" ]]; then
    "$script_dir/profile.sh" playground
fi

# examples:

# This will run the hotspot profiler on the oracle-graph benchmark,
# with 200 repeats, single-threaded Rayon, and 1 trial.
# PROFILE_TOOL=hotspot PROFILE_REPEAT=200 RAYON_NUM_THREADS=1 NUM_TRIALS=1 ./profile.sh leiden
# PROFILE_PLAYGROUND=1 PROFILE_TOOL=hotspot ./profile_all.sh
