# embeddenator-contract-bench

Deterministic “contract benchmarks” for `embeddenator` VSA operations.

The goal is to produce stable, comparable benchmark results over time (and across environments), with baseline snapshots checked into the repo.

## Status

Alpha. This crate is primarily for maintainers and CI.

## Running

```bash
# Run the bench binary (not `cargo bench`)
cargo run -p embeddenator-contract-bench --release -- --help
```

## Output

Results are typically written under `bench_results/` and can be compared against baselines in `baselines/`.

## License

MIT
