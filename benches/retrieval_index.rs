//! Retrieval index benchmark suite
//!
//! Benchmarks for inverted index operations:
//! - Index construction at various scales
//! - Top-k query performance
//! - Query throughput with various corpus sizes

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use embeddenator::{SparseVec, TernaryInvertedIndex};

/// Benchmark retrieval index construction and query performance
fn bench_retrieval_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("retrieval_index");

    // Build a deterministic corpus at various scales
    let corpus_sizes = [100usize, 500, 1_000, 5_000, 10_000];

    for n in corpus_sizes {
        // Benchmark index construction
        group.bench_with_input(BenchmarkId::new("build", n), &n, |bencher, &n| {
            bencher.iter(|| {
                let mut index = TernaryInvertedIndex::new();
                for i in 0..n {
                    let v = SparseVec::from_data(black_box(format!("doc-{i}").as_bytes()));
                    index.add(i, &v);
                }
                index.finalize();
                black_box(index)
            })
        });

        // Build once for query benchmarks
        let mut index = TernaryInvertedIndex::new();
        for i in 0..n {
            let v = SparseVec::from_data(format!("doc-{i}").as_bytes());
            index.add(i, &v);
        }
        index.finalize();

        // Benchmark queries with various k values
        for k in [5, 10, 20, 50] {
            if k > n / 2 {
                continue; // Skip if k is too large for corpus
            }

            let query = SparseVec::from_data(b"doc-123");
            group.bench_with_input(
                BenchmarkId::new(format!("query_top_k_{k}"), n),
                &(n, k),
                |bencher, &(_n, k)| {
                    bencher.iter(|| {
                        let hits = index.query_top_k(black_box(&query), k);
                        black_box(hits)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark query throughput with batched queries
fn bench_query_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_throughput");

    // Build a medium-sized index
    let corpus_size = 5_000;
    let mut index = TernaryInvertedIndex::new();
    for i in 0..corpus_size {
        let v = SparseVec::from_data(format!("doc-{i}").as_bytes());
        index.add(i, &v);
    }
    index.finalize();

    // Prepare multiple queries
    let queries: Vec<SparseVec> = (0..100)
        .map(|i| SparseVec::from_data(format!("query-{i}").as_bytes()))
        .collect();

    group.bench_function("batch_100_queries", |bencher| {
        bencher.iter(|| {
            for query in &queries {
                let hits = index.query_top_k(black_box(query), 20);
                black_box(hits);
            }
        })
    });

    group.finish();
}

/// Benchmark index memory characteristics
fn bench_index_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_scaling");
    group.sample_size(10); // Fewer samples for large operations

    let scales = [1_000, 5_000, 10_000, 20_000];

    for n in scales {
        group.bench_with_input(BenchmarkId::new("finalize", n), &n, |bencher, &n| {
            bencher.iter_with_setup(
                || {
                    let mut index = TernaryInvertedIndex::new();
                    for i in 0..n {
                        let v = SparseVec::from_data(format!("doc-{i}").as_bytes());
                        index.add(i, &v);
                    }
                    index
                },
                |mut index| {
                    index.finalize();
                    black_box(index)
                },
            )
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_retrieval_index,
    bench_query_throughput,
    bench_index_scaling
);
criterion_main!(benches);
