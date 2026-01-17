//! Hierarchical query benchmark suite
//!
//! Benchmarks for hierarchical query performance:
//! - Performance vs hierarchy depth
//! - Performance vs hierarchy width
//! - Flat vs hierarchical query comparison
//! - Beam width parameter tuning

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use embeddenator::{
    query_hierarchical_codebook, EmbrFS, HierarchicalQueryBounds, ReversibleVSAConfig, SparseVec,
};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use tempfile::TempDir;

/// Create hierarchical test structure with controlled depth and width
fn create_hierarchical_structure(
    dir: &TempDir,
    depth: usize,
    width: usize,
    file_size: usize,
) -> usize {
    let base_path = dir.path();
    let mut total_files = 0;

    fn create_level(
        path: &std::path::Path,
        current_depth: usize,
        max_depth: usize,
        width: usize,
        file_size: usize,
        total_files: &mut usize,
    ) {
        if current_depth >= max_depth {
            return;
        }

        // Create files at this level (limited to avoid storage issues)
        let files_at_level = width.min(5);
        for file_idx in 0..files_at_level {
            let file_path = path.join(format!("file_{:04}.txt", file_idx));
            let mut file = fs::File::create(&file_path).unwrap();

            let content = format!(
                "Depth {} File {} Content: {}\n",
                current_depth,
                file_idx,
                "Sample data. ".repeat(file_size / 20)
            );
            file.write_all(content.as_bytes()).unwrap();
            *total_files += 1;
        }

        // Create subdirectories and recurse (limit branching)
        let subdirs = width.min(3);
        for dir_idx in 0..subdirs {
            let subdir = path.join(format!("dir_{:04}", dir_idx));
            fs::create_dir_all(&subdir).unwrap();
            create_level(
                &subdir,
                current_depth + 1,
                max_depth,
                width,
                file_size,
                total_files,
            );
        }
    }

    create_level(base_path, 0, depth, width, file_size, &mut total_files);
    total_files
}

/// Benchmark hierarchical query performance vs depth
fn bench_hierarchical_query_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("hierarchical_query_depth");
    group.sample_size(10);

    // Test different hierarchy depths with moderate width
    let depth_configs = vec![
        (2, 5, "depth_2_width_5"),
        (3, 5, "depth_3_width_5"),
        (4, 3, "depth_4_width_3"),
    ];

    for (depth, width, label) in depth_configs {
        group.bench_with_input(
            BenchmarkId::new("query_performance", label),
            &(depth, width),
            |bencher, &(depth, width)| {
                let config = ReversibleVSAConfig::default();

                // Setup: create structure and build hierarchical index
                let temp_dir = TempDir::new().unwrap();
                let _total_files = create_hierarchical_structure(&temp_dir, depth, width, 1024);
                let mut fs = EmbrFS::new();
                fs.ingest_directory(temp_dir.path(), false, &config)
                    .unwrap();

                let hierarchical = fs.bundle_hierarchically(500, false, &config).unwrap();

                // Extract codebook
                let codebook: HashMap<usize, SparseVec> = fs
                    .engram
                    .codebook
                    .iter()
                    .map(|(&id, vec)| (id, vec.clone()))
                    .collect();

                // Create query vector
                let query = SparseVec::encode_data(b"Sample data with patterns", &config, None);

                let bounds = HierarchicalQueryBounds {
                    k: 20,
                    candidate_k: 100,
                    beam_width: 10,
                    max_depth: depth,
                    max_expansions: 1000,
                    max_open_engrams: 100,
                    max_open_indices: 50,
                };

                bencher.iter(|| {
                    let results = query_hierarchical_codebook(
                        black_box(&hierarchical),
                        black_box(&codebook),
                        black_box(&query),
                        black_box(&bounds),
                    );
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark hierarchical query performance vs width
fn bench_hierarchical_query_width(c: &mut Criterion) {
    let mut group = c.benchmark_group("hierarchical_query_width");
    group.sample_size(10);

    // Test different hierarchy widths with fixed depth
    let width_configs = vec![
        (2, 5, "depth_2_width_5"),
        (2, 10, "depth_2_width_10"),
        (2, 15, "depth_2_width_15"),
    ];

    for (depth, width, label) in width_configs {
        group.bench_with_input(
            BenchmarkId::new("query_performance", label),
            &(depth, width),
            |bencher, &(depth, width)| {
                let config = ReversibleVSAConfig::default();

                let temp_dir = TempDir::new().unwrap();
                let _total_files = create_hierarchical_structure(&temp_dir, depth, width, 1024);
                let mut fs = EmbrFS::new();
                fs.ingest_directory(temp_dir.path(), false, &config)
                    .unwrap();

                let hierarchical = fs.bundle_hierarchically(500, false, &config).unwrap();
                let codebook: HashMap<usize, SparseVec> = fs
                    .engram
                    .codebook
                    .iter()
                    .map(|(&id, vec)| (id, vec.clone()))
                    .collect();

                let query = SparseVec::encode_data(b"Sample data with patterns", &config, None);
                let bounds = HierarchicalQueryBounds {
                    k: 20,
                    candidate_k: 100,
                    beam_width: 10,
                    max_depth: depth,
                    max_expansions: 1000,
                    max_open_engrams: 100,
                    max_open_indices: 50,
                };

                bencher.iter(|| {
                    let results = query_hierarchical_codebook(
                        black_box(&hierarchical),
                        black_box(&codebook),
                        black_box(&query),
                        black_box(&bounds),
                    );
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark beam width scaling
fn bench_beam_width_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("beam_width_scaling");
    group.sample_size(10);

    let config = ReversibleVSAConfig::default();
    let temp_dir = TempDir::new().unwrap();
    create_hierarchical_structure(&temp_dir, 3, 5, 1024);

    let mut fs = EmbrFS::new();
    fs.ingest_directory(temp_dir.path(), false, &config)
        .unwrap();
    let hierarchical = fs.bundle_hierarchically(500, false, &config).unwrap();

    let codebook: HashMap<usize, SparseVec> = fs
        .engram
        .codebook
        .iter()
        .map(|(&id, vec)| (id, vec.clone()))
        .collect();

    let query = SparseVec::encode_data(b"Sample data with patterns", &config, None);

    for beam_width in [5, 10, 20, 50] {
        group.bench_with_input(
            BenchmarkId::new("beam_width", beam_width),
            &beam_width,
            |bencher, &beam_width| {
                let bounds = HierarchicalQueryBounds {
                    k: 20,
                    candidate_k: 100,
                    beam_width,
                    max_depth: 3,
                    max_expansions: 1000,
                    max_open_engrams: 100,
                    max_open_indices: 50,
                };

                bencher.iter(|| {
                    let results = query_hierarchical_codebook(
                        black_box(&hierarchical),
                        black_box(&codebook),
                        black_box(&query),
                        black_box(&bounds),
                    );
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark flat vs hierarchical query strategies
fn bench_flat_vs_hierarchical(c: &mut Criterion) {
    let mut group = c.benchmark_group("flat_vs_hierarchical");
    group.sample_size(10);

    let config = ReversibleVSAConfig::default();
    let temp_dir = TempDir::new().unwrap();
    create_hierarchical_structure(&temp_dir, 3, 5, 1024);

    let mut fs = EmbrFS::new();
    fs.ingest_directory(temp_dir.path(), false, &config)
        .unwrap();

    // Build hierarchical index
    let hierarchical = fs.bundle_hierarchically(500, false, &config).unwrap();
    let codebook: HashMap<usize, SparseVec> = fs
        .engram
        .codebook
        .iter()
        .map(|(&id, vec)| (id, vec.clone()))
        .collect();

    // Build flat index
    let flat_index = fs.engram.build_codebook_index();

    let query = SparseVec::encode_data(b"Sample data with patterns", &config, None);

    group.bench_function("flat_query", |bencher| {
        bencher.iter(|| {
            let results = flat_index.query_top_k(black_box(&query), 20);
            black_box(results)
        })
    });

    group.bench_function("hierarchical_query", |bencher| {
        let bounds = HierarchicalQueryBounds {
            k: 20,
            candidate_k: 100,
            beam_width: 10,
            max_depth: 3,
            max_expansions: 1000,
            max_open_engrams: 100,
            max_open_indices: 50,
        };

        bencher.iter(|| {
            let results = query_hierarchical_codebook(
                black_box(&hierarchical),
                black_box(&codebook),
                black_box(&query),
                black_box(&bounds),
            );
            black_box(results)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_hierarchical_query_depth,
    bench_hierarchical_query_width,
    bench_beam_width_scaling,
    bench_flat_vs_hierarchical
);
criterion_main!(benches);
