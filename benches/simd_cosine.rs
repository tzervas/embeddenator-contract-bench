//! SIMD cosine similarity benchmark suite
//!
//! Compares scalar vs SIMD implementations across:
//! - Various vector sparsity levels
//! - Different data patterns (identical, similar, different)
//! - Query workload simulations
//! - Synthetic sparsity levels

use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration,
};
use embeddenator::{ReversibleVSAConfig, SparseVec};

/// Benchmark scalar vs SIMD cosine with real data patterns
fn bench_cosine_scalar_vs_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_scalar_vs_simd");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let config = ReversibleVSAConfig::default();

    // Test with various data patterns
    let test_cases = vec![
        (
            "identical",
            b"test data 123".as_slice(),
            b"test data 123".as_slice(),
        ),
        (
            "similar",
            b"test data 123".as_slice(),
            b"test data 124".as_slice(),
        ),
        (
            "different",
            b"test data 123".as_slice(),
            b"completely different".as_slice(),
        ),
        ("short", b"hi".as_slice(), b"hello".as_slice()),
        (
            "medium",
            b"the quick brown fox".as_slice(),
            b"the lazy dog".as_slice(),
        ),
        (
            "long",
            b"the quick brown fox jumps over the lazy dog again and again".as_slice(),
            b"the quick brown fox jumps over the lazy dog one more time".as_slice(),
        ),
    ];

    for (name, data_a, data_b) in test_cases {
        let a = SparseVec::encode_data(data_a, &config, None);
        let b = SparseVec::encode_data(data_b, &config, None);

        // Benchmark scalar version
        group.bench_with_input(
            BenchmarkId::new("scalar", name),
            &(&a, &b),
            |bencher, (a, b)| bencher.iter(|| black_box(a).cosine_scalar(black_box(b))),
        );

        // Benchmark SIMD version if available
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd", name),
            &(&a, &b),
            |bencher, (a, b)| {
                bencher.iter(|| embeddenator::simd_cosine::cosine_simd(black_box(a), black_box(b)))
            },
        );
    }

    group.finish();
}

/// Benchmark cosine with controlled synthetic sparsity
fn bench_cosine_synthetic_sparsity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_synthetic_sparsity");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    // Create synthetic vectors with controlled sparsity
    let sparsity_levels = vec![10, 50, 100, 200, 500, 1000, 2000];

    for sparsity in sparsity_levels {
        // Create two vectors with 50% overlap in indices
        let pos_a: Vec<usize> = (0..sparsity).map(|i| i * 2).collect();
        let neg_a: Vec<usize> = (0..sparsity).map(|i| i * 2 + 1).collect();

        let pos_b: Vec<usize> = (sparsity / 2..sparsity + sparsity / 2)
            .map(|i| i * 2)
            .collect();
        let neg_b: Vec<usize> = (sparsity / 2..sparsity + sparsity / 2)
            .map(|i| i * 2 + 1)
            .collect();

        let a = SparseVec {
            pos: pos_a,
            neg: neg_a,
        };
        let b = SparseVec {
            pos: pos_b,
            neg: neg_b,
        };

        // Benchmark scalar version
        group.bench_with_input(
            BenchmarkId::new("scalar", sparsity),
            &(&a, &b),
            |bencher, (a, b)| bencher.iter(|| black_box(a).cosine_scalar(black_box(b))),
        );

        // Benchmark SIMD version if available
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd", sparsity),
            &(&a, &b),
            |bencher, (a, b)| {
                bencher.iter(|| embeddenator::simd_cosine::cosine_simd(black_box(a), black_box(b)))
            },
        );
    }

    group.finish();
}

/// Benchmark realistic query workload: one query vs many documents
fn bench_cosine_query_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_query_workload");

    let config = ReversibleVSAConfig::default();

    // Simulate a realistic query workload
    let query = SparseVec::encode_data(b"search query: machine learning embeddings", &config, None);

    // Create document corpus
    let documents: Vec<SparseVec> = (0..1000)
        .map(|i| {
            SparseVec::encode_data(
                format!("document {} with various content", i).as_bytes(),
                &config,
                None,
            )
        })
        .collect();

    group.bench_function("scalar_1000_docs", |bencher| {
        bencher.iter(|| {
            for doc in &documents {
                let _score = black_box(&query).cosine_scalar(black_box(doc));
                black_box(_score);
            }
        })
    });

    #[cfg(feature = "simd")]
    group.bench_function("simd_1000_docs", |bencher| {
        bencher.iter(|| {
            for doc in &documents {
                let _score =
                    embeddenator::simd_cosine::cosine_simd(black_box(&query), black_box(doc));
                black_box(_score);
            }
        })
    });

    group.finish();
}

/// Benchmark cosine with varying overlap percentages
fn bench_cosine_overlap_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_overlap_patterns");

    let sparsity = 500;
    let overlap_percentages = vec![0, 25, 50, 75, 100];

    for overlap_pct in overlap_percentages {
        let overlap_count = (sparsity * overlap_pct) / 100;

        // Create vectors with controlled overlap
        let pos_a: Vec<usize> = (0..sparsity).map(|i| i * 2).collect();
        let neg_a: Vec<usize> = (0..sparsity).map(|i| i * 2 + 1).collect();

        let pos_b: Vec<usize> = (0..overlap_count)
            .map(|i| i * 2)
            .chain((overlap_count..sparsity).map(|i| (sparsity + i) * 2))
            .collect();
        let neg_b: Vec<usize> = (0..overlap_count)
            .map(|i| i * 2 + 1)
            .chain((overlap_count..sparsity).map(|i| (sparsity + i) * 2 + 1))
            .collect();

        let a = SparseVec {
            pos: pos_a,
            neg: neg_a,
        };
        let b = SparseVec {
            pos: pos_b,
            neg: neg_b,
        };

        group.bench_with_input(
            BenchmarkId::new("scalar", format!("overlap_{}pct", overlap_pct)),
            &(&a, &b),
            |bencher, (a, b)| bencher.iter(|| black_box(a).cosine_scalar(black_box(b))),
        );

        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd", format!("overlap_{}pct", overlap_pct)),
            &(&a, &b),
            |bencher, (a, b)| {
                bencher.iter(|| embeddenator::simd_cosine::cosine_simd(black_box(a), black_box(b)))
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cosine_scalar_vs_simd,
    bench_cosine_synthetic_sparsity,
    bench_cosine_query_workload,
    bench_cosine_overlap_patterns
);
criterion_main!(benches);
