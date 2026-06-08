#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage: profile.sh [--tool flamegraph|hotspot|heaptrack] <target> [repeat]

Targets:
  dummy           Query with dummy clustering
  leiden          Query with feature-gated Leiden clustering
  playground      Full in_memory_playground binary run
  oracle-graph    In-memory graph_neighbourhoods batch lookup
  oracle-coreset  In-memory coreset_neighbourhoods batch lookup

Examples:
  ./crates/raven-adapters/flame/profile.sh dummy
  ./crates/raven-adapters/flame/profile.sh playground
  ./crates/raven-adapters/flame/profile.sh --tool hotspot dummy
  PROFILE_REPEAT=3 PROFILE_TOOL=heaptrack ./crates/raven-adapters/flame/profile.sh leiden

For bench targets, the repeat count builds each target fixture once, then
repeats only the profiled lookup/query operation. The playground target runs the
full binary once.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

tool="${PROFILE_TOOL:-flamegraph}"
if [[ "${1:-}" == "--tool" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "--tool requires one of: flamegraph, hotspot, heaptrack" >&2
        exit 2
    fi
    tool="$2"
    shift 2
elif [[ "${1:-}" == "flamegraph" || "${1:-}" == "hotspot" || "${1:-}" == "heaptrack" ]]; then
    tool="$1"
    shift
fi

target="${1:-dummy}"
profile_repeat="${2:-${PROFILE_REPEAT:-1}}"
sample_freq="${FLAMEGRAPH_FREQ:-997}"
perf_call_graph="${PERF_CALL_GRAPH:-fp}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "$script_dir/../../.." && pwd)"
out_dir="$script_dir/flame_out"
post_process_args=()
target_kind="bench"

case "$target" in
    dummy | query-dummy | query_dummy_subset)
        bench_filter="query_dummy_subset"
        output_stem="query_dummy_subset"
        feature_args=()
        ;;
    leiden | query-leiden | query_leiden_subset)
        bench_filter="query_leiden_subset"
        output_stem="query_leiden_subset"
        feature_args=(--features bench-clustering)
        ;;
    playground | in-memory-playground | in_memory_playground)
        target_kind="playground"
        bench_filter="in_memory_playground"
        output_stem="in_memory_playground"
        feature_args=(--features bench-clustering)
        ;;
    oracle-graph | graph | graph_neighbourhoods)
        bench_filter="graph_neighbourhoods"
        output_stem="oracle_graph_neighbourhoods"
        feature_args=()
        ;;
    oracle-coreset | coreset | coreset_neighbourhoods)
        bench_filter="coreset_neighbourhoods"
        output_stem="oracle_coreset_neighbourhoods"
        feature_args=()
        ;;
    *)
        echo "unknown profile target: $target" >&2
        usage
        exit 2
        ;;
esac

mkdir -p "$out_dir"

export CARGO_PROFILE_BENCH_DEBUG="${CARGO_PROFILE_BENCH_DEBUG:-true}"
export CARGO_PROFILE_RELEASE_DEBUG="${CARGO_PROFILE_RELEASE_DEBUG:-true}"
if [[ "${FORCE_FRAME_POINTERS:-1}" == "1" && "${RUSTFLAGS:-}" != *"force-frame-pointers"* ]]; then
    export RUSTFLAGS="${RUSTFLAGS:-} -C force-frame-pointers=yes"
fi

bench_args=(--profile-once "$bench_filter" "$profile_repeat")
target_args=()
if [[ "$target_kind" == "bench" ]]; then
    target_args=("${bench_args[@]}")
fi

build_bench() {
    local cargo_output bench_path

    cargo_output="$(cargo bench \
        -p raven-adapters \
        "${feature_args[@]}" \
        --bench in_memory_query \
        --no-run 2>&1)"
    printf '%s\n' "$cargo_output" >&2

    bench_path="$(
        printf '%s\n' "$cargo_output" \
            | sed -n 's/.*(\(target\/release\/deps\/in_memory_query-[^)]*\)).*/\1/p' \
            | tail -n 1
    )"
    if [[ -n "$bench_path" ]]; then
        printf '%s\n' "$workspace_root/$bench_path"
    fi
}

build_playground() {
    cargo build \
        -p raven-adapters \
        "${feature_args[@]}" \
        --bin in_memory_playground \
        --release >&2

    printf '%s\n' "$workspace_root/target/release/in_memory_playground"
}

find_bench_exe() {
    find target/release/deps \
        -maxdepth 1 \
        -type f \
        -executable \
        -name 'in_memory_query-*' \
        -printf '%T@ %p\n' \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
}

build_target_exe() {
    if [[ "$target_kind" == "playground" ]]; then
        build_playground
    else
        build_bench
    fi
}

find_target_exe() {
    if [[ "$target_kind" == "playground" ]]; then
        if [[ -x target/release/in_memory_playground ]]; then
            printf '%s\n' "target/release/in_memory_playground"
        fi
    else
        find_bench_exe
    fi
}

cd "$workspace_root"
if [[ "$target_kind" == "playground" ]]; then
    echo "Profiling '$bench_filter' with $tool (full binary run)"
else
    echo "Profiling '$bench_filter' with $tool (measured repeat=$profile_repeat)"
fi

if [[ "${FORCE_FRAME_POINTERS:-1}" == "1" ]]; then
    echo "Using frame pointers for cleaner stack unwinding"
fi

case "$tool" in
    flamegraph)
        if ! cargo flamegraph --help >/dev/null 2>&1; then
            echo "cargo-flamegraph is not installed. Try: cargo install flamegraph" >&2
            exit 1
        fi

        if command -v rustfilt >/dev/null 2>&1; then
            post_process_args=(--post-process rustfilt)
        fi

        output="$out_dir/$output_stem.svg"
        echo "Sampling at ${sample_freq}Hz"
        echo "Writing flamegraph to $output"
        if [[ "${#post_process_args[@]}" -eq 0 ]]; then
            echo "Tip: install rustfilt for demangled Rust symbols: cargo install rustfilt"
        fi

        if [[ "$target_kind" == "playground" ]]; then
            cargo flamegraph \
                -p raven-adapters \
                "${feature_args[@]}" \
                --bin in_memory_playground \
                -F "$sample_freq" \
                "${post_process_args[@]}" \
                -o "$output"
        else
            cargo flamegraph \
                -p raven-adapters \
                "${feature_args[@]}" \
                --bench in_memory_query \
                -F "$sample_freq" \
                "${post_process_args[@]}" \
                -o "$output" \
                -- \
                "${target_args[@]}"
        fi
        ;;
    hotspot)
        if ! command -v perf >/dev/null 2>&1; then
            echo "perf is required for Hotspot data capture" >&2
            exit 1
        fi

        bench_exe="$(build_target_exe)"
        if [[ -z "$bench_exe" ]]; then
            bench_exe="$(find_target_exe)"
        fi
        if [[ -z "$bench_exe" ]]; then
            echo "failed to locate executable for profile target '$target'" >&2
            exit 1
        fi

        output="$out_dir/$output_stem.perf.data"
        echo "Sampling at ${sample_freq}Hz with call graph '$perf_call_graph'"
        echo "Writing perf data to $output"
        perf record \
            -F "$sample_freq" \
            --call-graph "$perf_call_graph" \
            -o "$output" \
            -- \
            "$bench_exe" \
            "${target_args[@]}"

        echo "Open with: hotspot $output"
        if [[ "${OPEN_HOTSPOT:-0}" == "1" ]]; then
            if command -v hotspot >/dev/null 2>&1; then
                hotspot "$output" >/dev/null 2>&1 &
            else
                echo "hotspot is not installed; wrote $output" >&2
            fi
        fi
        ;;
    heaptrack)
        if ! command -v heaptrack >/dev/null 2>&1; then
            echo "heaptrack is not installed" >&2
            exit 1
        fi

        bench_exe="$(build_target_exe)"
        if [[ -z "$bench_exe" ]]; then
            bench_exe="$(find_target_exe)"
        fi
        if [[ -z "$bench_exe" ]]; then
            echo "failed to locate executable for profile target '$target'" >&2
            exit 1
        fi

        output="$out_dir/$output_stem.heaptrack"
        echo "Writing heaptrack data to $output"
        heaptrack \
            --output "$output" \
            "$bench_exe" \
            "${target_args[@]}"

        heap_file="$(
            find "$out_dir" \
                -maxdepth 1 \
                -type f \
                -name "$(basename "$output")*" \
                -printf '%T@ %p\n' \
                | sort -nr \
                | head -n 1 \
                | cut -d' ' -f2-
        )"
        if [[ -n "$heap_file" ]]; then
            echo "Open with: heaptrack_gui $heap_file"
            if [[ "${OPEN_HEAPTRACK:-0}" == "1" ]]; then
                if command -v heaptrack_gui >/dev/null 2>&1; then
                    heaptrack_gui "$heap_file" >/dev/null 2>&1 &
                else
                    echo "heaptrack_gui is not installed; wrote $heap_file" >&2
                fi
            fi
        fi
        ;;
    *)
        echo "unknown profiling tool: $tool" >&2
        echo "expected one of: flamegraph, hotspot, heaptrack" >&2
        exit 2
        ;;
esac
