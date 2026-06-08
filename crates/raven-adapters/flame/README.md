# In-Memory Profiling

These scripts profile the `raven-adapters` Criterion bench target and write
outputs to `flame_out/`.

Install the flamegraph tool once if needed:

```bash
cargo install flamegraph
```

For readable Rust symbol names, also install:

```bash
cargo install rustfilt
```

For GUI exploration, install system packages for:

- `hotspot`: opens `perf.data` with an interactive call tree/flamegraph UI.
- `heaptrack` and `heaptrack_gui`: records and explores heap allocation traces.

Run one flamegraph target:

```bash
./crates/raven-adapters/flame/profile.sh dummy
./crates/raven-adapters/flame/profile.sh oracle-graph
./crates/raven-adapters/flame/profile.sh oracle-coreset
./crates/raven-adapters/flame/profile.sh leiden
./crates/raven-adapters/flame/profile.sh playground
```

By default, profile scripts run the selected target once via the bench binary's
`--profile-once` path. Each profiling target builds its workload/fixture once,
then repeats only the lookup or query operation being profiled. This keeps setup
noise from growing with the repeat count while still giving profilers a long
steady-state run. The `playground` target is different: it profiles one full
`in_memory_playground` binary run and ignores the repeat count. Set
`PROFILE_REPEAT` or pass the repeat count as the second argument for bench
targets:

```bash
PROFILE_REPEAT=3 ./crates/raven-adapters/flame/profile.sh dummy
./crates/raven-adapters/flame/profile.sh leiden 3
```

Run Hotspot/perf capture:

```bash
PROFILE_TOOL=hotspot ./crates/raven-adapters/flame/profile.sh dummy
./crates/raven-adapters/flame/profile.sh --tool hotspot leiden
./crates/raven-adapters/flame/profile.sh --tool hotspot playground
hotspot crates/raven-adapters/flame/flame_out/query_dummy_subset.perf.data
```

Open Hotspot automatically after recording:

```bash
OPEN_HOTSPOT=1 PROFILE_TOOL=hotspot ./crates/raven-adapters/flame/profile.sh dummy
```

Run Heaptrack allocation capture:

```bash
PROFILE_TOOL=heaptrack ./crates/raven-adapters/flame/profile.sh dummy
OPEN_HEAPTRACK=1 PROFILE_TOOL=heaptrack ./crates/raven-adapters/flame/profile.sh leiden
PROFILE_TOOL=heaptrack ./crates/raven-adapters/flame/profile.sh playground
```

Heaptrack will print the exact trace file it wrote. Open it with:

```bash
heaptrack_gui crates/raven-adapters/flame/flame_out/query_dummy_subset.heaptrack*
```

Run all profiles:

```bash
./crates/raven-adapters/flame/profile_all.sh
```

`profile_all.sh` skips the full playground by default. Include it explicitly
with:

```bash
PROFILE_PLAYGROUND=1 ./crates/raven-adapters/flame/profile_all.sh
```

Run all profiles with another backend:

```bash
PROFILE_TOOL=hotspot ./crates/raven-adapters/flame/profile_all.sh
PROFILE_TOOL=heaptrack ./crates/raven-adapters/flame/profile_all.sh
```

The scripts default `CARGO_PROFILE_BENCH_DEBUG=true` for better symbol names,
build with `-C force-frame-pointers=yes` for cleaner stack unwinding, pass
the bench binary's `--profile-once` path, and use `rustfilt` automatically when
it is installed. Set `FORCE_FRAME_POINTERS=0` to use your existing `RUSTFLAGS`
unchanged.

Hotspot capture uses `perf record` directly. You can tune the sample rate and
call graph mode:

```bash
FLAMEGRAPH_FREQ=499 PERF_CALL_GRAPH=dwarf PROFILE_TOOL=hotspot \
  ./crates/raven-adapters/flame/profile.sh dummy
```

On Linux, `perf` may require adjusted permissions or `sudo`; if profiling fails
with a permissions error, check
`/proc/sys/kernel/perf_event_paranoid`.
