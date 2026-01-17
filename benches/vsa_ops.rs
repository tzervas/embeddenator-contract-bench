//! VSA operations benchmark suite
//!
//! Comprehensive benchmarks for core VSA operations including:
//! - Bundle, bind, and cosine operations
//! - Encode/decode operations at various data sizes
//! - Bundle mode comparisons (pairwise vs sum-many vs hybrid)
//! - Chain operations and performance characteristics

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use embeddenator::{ReversibleVSAConfig, SparseVec, DIM};

/// Benchmark fundamental SparseVec operations
fn bench_sparsevec_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparsevec_ops");

    // Deterministic vectors for stable benches
    let config = ReversibleVSAConfig::default();
    let a = SparseVec::encode_data(b"alpha", &config, None);
    let b = SparseVec::encode_data(b"beta", &config, None);
    let cvec = SparseVec::encode_data(b"gamma", &config, None);

    group.bench_function("bundle", |bencher| {
        bencher.iter(|| black_box(&a).bundle(black_box(&b)))
    });

    group.bench_function("bind", |bencher| {
        bencher.iter(|| black_box(&a).bind(black_box(&b)))
    });

    group.bench_function("cosine", |bencher| {
        bencher.iter(|| black_box(&a).cosine(black_box(&b)))
    });

    group.bench_function("bundle_chain_8", |bencher| {
        bencher.iter(|| {
            let mut acc = black_box(a.clone());
            for _ in 0..7 {
                acc = acc.bundle(black_box(&b));
            }
            black_box(acc)
        })
    });

    group.bench_function("bind_chain_8", |bencher| {
        bencher.iter(|| {
            let mut acc = black_box(a.clone());
            for _ in 0..7 {
                acc = acc.bind(black_box(&b));
            }
            black_box(acc)
        })
    });

    // Ensure we still exercise a non-trivial cosine shape
    group.bench_function("cosine_chain_mix", |bencher| {
        bencher.iter(|| {
            let mixed = black_box(&a).bundle(black_box(&b)).bind(black_box(&cvec));
            black_box(mixed.cosine(black_box(&a)))
        })
    });

    group.finish();
}

/// Benchmark reversible encode/decode operations at various data sizes
fn bench_reversible_encode_decode(c: &mut Criterion) {
    let config = ReversibleVSAConfig::default();

    let sizes = [64usize, 256, 1024, 4096, 16384];

    let mut group = c.benchmark_group("reversible_encode_decode");
    for size in sizes {
        let data: Vec<u8> = (0..size).map(|i| (i as u8).wrapping_mul(31)).collect();

        group.bench_with_input(BenchmarkId::new("encode", size), &data, |bencher, data| {
            bencher.iter(|| {
                let v = SparseVec::encode_data(
                    black_box(data),
                    black_box(&config),
                    Some("/bench/path"),
                );
                black_box(v)
            })
        });

        let encoded = SparseVec::encode_data(&data, &config, Some("/bench/path"));
        group.bench_with_input(
            BenchmarkId::new("decode", size),
            &encoded,
            |bencher, encoded| {
                bencher.iter(|| {
                    let out = black_box(encoded).decode_data(
                        black_box(&config),
                        Some("/bench/path"),
                        size,
                    );
                    black_box(out)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark different bundling strategies
fn bench_bundle_modes(c: &mut Criterion) {
    let config = ReversibleVSAConfig::default();

    // Sparse inputs (low collision probability)
    let sa = SparseVec::encode_data(b"sparse-a", &config, None);
    let sb = SparseVec::encode_data(b"sparse-b", &config, None);
    let sc = SparseVec::encode_data(b"sparse-c", &config, None);

    // Dense-ish synthetic inputs to trigger packed/associative paths
    let make_dense = |offset: usize| SparseVec {
        pos: (offset..offset + 4000).step_by(2).collect(),
        neg: (offset + 1..offset + 4000).step_by(2).collect(),
    };
    let da = make_dense(0);
    let db = make_dense(500);
    let dc = make_dense(1000);

    // Mid-density synthetic inputs to probe the packed-threshold boundary
    let make_mid = |offset: usize, span: usize| SparseVec {
        pos: (offset..offset + span).step_by(2).collect(),
        neg: (offset + 1..offset + span).step_by(2).collect(),
    };
    let ma_lo = make_mid(0, 1200);
    let mb_lo = make_mid(400, 1200);
    let mc_lo = make_mid(800, 1200);
    let ma_hi = make_mid(0, 1400);
    let mb_hi = make_mid(400, 1400);
    let mc_hi = make_mid(800, 1400);

    let mut group = c.benchmark_group("bundle_modes");

    group.bench_function("pairwise_sparse", |bch| {
        bch.iter(|| {
            let acc = black_box(&sa).bundle(black_box(&sb)).bundle(black_box(&sc));
            black_box(acc)
        })
    });

    group.bench_function("sum_many_sparse", |bch| {
        bch.iter(|| {
            let acc = SparseVec::bundle_sum_many([black_box(&sa), black_box(&sb), black_box(&sc)]);
            black_box(acc)
        })
    });

    group.bench_function("hybrid_sparse", |bch| {
        bch.iter(|| {
            let acc =
                SparseVec::bundle_hybrid_many([black_box(&sa), black_box(&sb), black_box(&sc)]);
            black_box(acc)
        })
    });

    group.bench_function("pairwise_dense", |bch| {
        bch.iter(|| {
            let acc = black_box(&da).bundle(black_box(&db)).bundle(black_box(&dc));
            black_box(acc)
        })
    });

    group.bench_function("sum_many_dense", |bch| {
        bch.iter(|| {
            let acc = SparseVec::bundle_sum_many([black_box(&da), black_box(&db), black_box(&dc)]);
            black_box(acc)
        })
    });

    group.bench_function("hybrid_dense", |bch| {
        bch.iter(|| {
            let acc =
                SparseVec::bundle_hybrid_many([black_box(&da), black_box(&db), black_box(&dc)]);
            black_box(acc)
        })
    });

    group.bench_function("pairwise_mid_low", |bch| {
        bch.iter(|| {
            let acc = black_box(&ma_lo)
                .bundle(black_box(&mb_lo))
                .bundle(black_box(&mc_lo));
            black_box(acc)
        })
    });

    group.bench_function("sum_many_mid_low", |bch| {
        bch.iter(|| {
            let acc = SparseVec::bundle_sum_many([
                black_box(&ma_lo),
                black_box(&mb_lo),
                black_box(&mc_lo),
            ]);
            black_box(acc)
        })
    });

    group.bench_function("hybrid_mid_low", |bch| {
        bch.iter(|| {
            let acc = SparseVec::bundle_hybrid_many([
                black_box(&ma_lo),
                black_box(&mb_lo),
                black_box(&mc_lo),
            ]);
            black_box(acc)
        })
    });

    group.bench_function("pairwise_mid_high", |bch| {
        bch.iter(|| {
            let acc = black_box(&ma_hi)
                .bundle(black_box(&mb_hi))
                .bundle(black_box(&mc_hi));
            black_box(acc)
        })
    });

    group.bench_function("sum_many_mid_high", |bch| {
        bch.iter(|| {
            let acc = SparseVec::bundle_sum_many([
                black_box(&ma_hi),
                black_box(&mb_hi),
                black_box(&mc_hi),
            ]);
            black_box(acc)
        })
    });

    group.bench_function("hybrid_mid_high", |bch| {
        bch.iter(|| {
            let acc = SparseVec::bundle_hybrid_many([
                black_box(&ma_hi),
                black_box(&mb_hi),
                black_box(&mc_hi),
            ]);
            black_box(acc)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sparsevec_ops,
    bench_reversible_encode_decode,
    bench_bundle_modes
);
criterion_main!(benches);
