# embeddenator-contract-bench

Comprehensive benchmark suite for embeddenator VSA operations, providing both deterministic contract benchmarks and Criterion-based performance measurements.

**Independent component** extracted from the Embeddenator monolithic repository. Part of the [Embeddenator workspace](https://github.com/tzervas/embeddenator).

**Repository:** [https://github.com/tzervas/embeddenator-contract-bench](https://github.com/tzervas/embeddenator-contract-bench)

## Overview

This crate provides two types of benchmarks:

1. **Contract Benchmarks** (via binary): Deterministic, reproducible benchmarks with baseline snapshots
2. **Criterion Benchmarks** (via `cargo bench`): Statistical performance analysis with HTML reports

## Status

Alpha. This crate is primarily for maintainers, CI, and performance validation.

## Benchmark Suites

### Criterion Benchmarks

Located in `benches/` directory, these provide detailed statistical analysis:

#### `vsa_ops.rs` - Core VSA Operations
- `sparsevec_ops`: Bundle, bind, cosine operations  
- `reversible_encode_decode`: Encode/decode at various data sizes (64B to 16KB)
- `bundle_modes`: Pairwise vs sum-many vs hybrid bundling strategies

**Run:**
```bash
cargo bench --bench vsa_ops
cargo bench --bench vsa_ops -- "bundle"  # Filter by pattern
```

#### `retrieval_index.rs` - Inverted Index Performance
- `retrieval_index/build`: Index construction at scale (100 to 10K vectors)
- `retrieval_index/query_top_k_*`: Query performance with various k values
- `query_throughput`: Batch query operations
- `index_scaling`: Finalization performance

**Run:**
```bash
cargo bench --bench retrieval_index
cargo bench --bench retrieval_index -- "query"
```

#### `hierarchical_scale.rs` - Hierarchical Bundling at Scale
- `hierarchical_bundling`: 10MB/50MB/100MB with/without sharding
- `bundle_memory_scaling`: Memory usage characteristics
- `ingest_scaling`: Directory ingestion performance

**Run:**
```bash
cargo bench --bench hierarchical_scale
cargo bench --bench hierarchical_scale -- "10MB"  # Specific size
```

#### `query_hierarchical.rs` - Hierarchical Query Performance
- `hierarchical_query_depth`: Performance vs hierarchy depth
- `hierarchical_query_width`: Performance vs hierarchy width
- `beam_width_scaling`: Beam width parameter tuning
- `flat_vs_hierarchical`: Query strategy comparison

**Run:**
```bash
cargo bench --bench query_hierarchical
cargo bench --bench query_hierarchical -- "flat_vs"
```

#### `simd_cosine.rs` - SIMD Acceleration
- `cosine_scalar_vs_simd`: Scalar vs SIMD comparison
- `cosine_synthetic_sparsity`: Controlled sparsity levels
- `cosine_query_workload`: Realistic query scenarios
- `cosine_overlap_patterns`: Various overlap percentages

**Run:**
```bash
cargo bench --bench simd_cosine
cargo bench --bench simd_cosine -- "scalar"
```

#### `io_operations.rs` - I/O Performance
- `file_io`: Read/write at various sizes (1KB to 10MB)
- `batch_io`: Batch file operations
- `directory_ops`: Directory creation
- `serialization`: JSON and bincode serialization

**Run:**
```bash
cargo bench --bench io_operations
```

#### `fs_operations.rs` - EmbrFS Operations
- `embrfs_ingest`: Directory ingestion (flat structures)
- `embrfs_nested`: Nested directory ingestion
- `embrfs_extract`: File extraction performance
- `embrfs_metadata`: Metadata operations
- `tree_traversal`: Tree traversal performance

**Run:**
```bash
cargo bench --bench fs_operations
```

### Contract Benchmarks

**Run the bench binary:**
```bash
cargo run -p embeddenator-contract-bench --release -- --help

# VSA operations
cargo run -p embeddenator-contract-bench --release -- vsa

# Retrieval benchmarks
cargo run -p embeddenator-contract-bench --release -- retrieval --input-dir ./test_data

# Full suite
cargo run -p embeddenator-contract-bench --release -- suite
```

## Running Benchmarks

### All Criterion Benchmarks
```bash
# Run all benchmarks (may take 30+ minutes)
cargo bench

# Run with custom sample size (faster, less precise)
cargo bench -- --sample-size 10
```

### Specific Benchmark Suite
```bash
cargo bench --bench vsa_ops
cargo bench --bench hierarchical_scale
cargo bench --bench query_hierarchical
```

### Filtered Benchmarks
```bash
# Filter by name pattern
cargo bench --bench hierarchical_scale -- "10MB"
cargo bench -- "bundle"

# Multiple filters (OR logic)
cargo bench -- "10MB|bundle"
```

## Benchmark Configuration

### Sample Sizes

Control precision vs speed:
```bash
cargo bench -- --sample-size 10    # Quick (~10 iterations)
cargo bench -- --sample-size 100   # Default (balanced)
cargo bench -- --measurement-time 30  # 30 seconds per benchmark
```

### Output Formats

Criterion generates:
- Console output with statistics
- HTML reports in `target/criterion/`
- Comparison with previous runs

View reports:
```bash
open target/criterion/report/index.html  # macOS
xdg-open target/criterion/report/index.html  # Linux
```

## Baseline Management

### Save Baseline
```bash
cargo bench -- --save-baseline my-baseline
```

### Compare Against Baseline
```bash
cargo bench -- --baseline my-baseline
```

## Performance Insights

### Expected Performance Characteristics

These are order-of-magnitude estimates. **Actual performance varies significantly** based on hardware, vector density, data patterns, and system configuration.

- **Bundle operations**: O(n) with sparsity
- **Bind operations**: O(n) with sparsity
- **Cosine similarity**: O(n) with sparsity
- **Index construction**: O(n log n)
- **Top-k query**: O(k log k + m)
- **Hierarchical bundling**: O(n) with chunking overhead

> Run `cargo bench` to measure performance on your specific system.

### Performance Regression Detection

Criterion automatically detects regressions:
- Green: Within noise threshold
- Yellow: Possible regression
- Red: Significant regression (>10%)

## CI Integration

### GitHub Actions Example
```yaml
name: Benchmarks

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run benchmarks
        run: |
          cargo bench --manifest-path embeddenator-contract-bench/Cargo.toml \
            -- --sample-size 10 --quick
```

## Known Limitations

Some internal VSA substrate types (`BitslicedTritVec`, `BlockSparseTritVec`, `CarrySaveBundle`) and IO envelope APIs are not exposed in the public `embeddenator` API. Benchmarks using these types are temporarily commented out but can be re-enabled when APIs are exposed.

## Output

### Criterion Results
- `target/criterion/`: HTML reports and data
- Console: Statistical summary

### Contract Benchmark Results  
- `bench_results/`: JSON measurements
- `baselines/`: Baseline snapshots

## License

MIT
