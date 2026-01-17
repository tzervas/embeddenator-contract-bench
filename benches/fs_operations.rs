//! Filesystem operations benchmark suite
//!
//! Benchmarks for EmbrFS-specific operations:
//! - Directory ingestion at various scales
//! - File extraction performance
//! - Metadata operations
//! - Tree traversal performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use embeddenator::{EmbrFS, ReversibleVSAConfig};
use std::fs;
use std::io::Write;
use tempfile::TempDir;

/// Create a test directory structure with files
fn create_test_files(dir: &TempDir, file_count: usize, file_size: usize) {
    let base_path = dir.path();

    for i in 0..file_count {
        let file_path = base_path.join(format!("file_{:04}.txt", i));
        let mut file = fs::File::create(&file_path).unwrap();

        let content = format!(
            "File {} content: {}\n",
            i,
            "Sample data. ".repeat(file_size / 20)
        );
        file.write_all(content.as_bytes()).unwrap();
    }
}

/// Create nested directory structure
fn create_nested_structure(dir: &TempDir, depth: usize, files_per_dir: usize, file_size: usize) {
    fn create_level(
        path: &std::path::Path,
        current_depth: usize,
        max_depth: usize,
        files_per_dir: usize,
        file_size: usize,
    ) {
        if current_depth >= max_depth {
            return;
        }

        // Create files at this level
        for i in 0..files_per_dir {
            let file_path = path.join(format!("file_{:04}.txt", i));
            let mut file = fs::File::create(&file_path).unwrap();
            let content = format!(
                "Depth {} File {} - {}\n",
                current_depth,
                i,
                "Data. ".repeat(file_size / 10)
            );
            file.write_all(content.as_bytes()).unwrap();
        }

        // Create subdirectories
        let subdirs = 3;
        for i in 0..subdirs {
            let subdir = path.join(format!("dir_{:02}", i));
            fs::create_dir_all(&subdir).unwrap();
            create_level(
                &subdir,
                current_depth + 1,
                max_depth,
                files_per_dir,
                file_size,
            );
        }
    }

    create_level(dir.path(), 0, depth, files_per_dir, file_size);
}

/// Benchmark EmbrFS ingest operations
fn bench_embrfs_ingest(c: &mut Criterion) {
    let mut group = c.benchmark_group("embrfs_ingest");
    group.sample_size(10);

    let file_counts = vec![10, 50, 100];
    let file_size = 1024; // 1KB per file

    for count in file_counts {
        group.bench_with_input(
            BenchmarkId::new("flat_structure", count),
            &count,
            |bencher, &count| {
                let config = ReversibleVSAConfig::default();

                bencher.iter_with_setup(
                    || {
                        let temp_dir = TempDir::new().unwrap();
                        create_test_files(&temp_dir, count, file_size);
                        (temp_dir, EmbrFS::new())
                    },
                    |(temp_dir, mut fs)| {
                        fs.ingest_directory(temp_dir.path(), false, &config)
                            .unwrap();
                        black_box(fs)
                    },
                )
            },
        );
    }

    group.finish();
}

/// Benchmark EmbrFS with nested directory structures
fn bench_embrfs_nested(c: &mut Criterion) {
    let mut group = c.benchmark_group("embrfs_nested");
    group.sample_size(10);

    let configs = vec![(2, 5, "depth2_files5"), (3, 3, "depth3_files3")];

    for (depth, files, label) in configs {
        group.bench_with_input(
            BenchmarkId::new("ingest_nested", label),
            &(depth, files),
            |bencher, &(depth, files)| {
                let config = ReversibleVSAConfig::default();

                bencher.iter_with_setup(
                    || {
                        let temp_dir = TempDir::new().unwrap();
                        create_nested_structure(&temp_dir, depth, files, 500);
                        (temp_dir, EmbrFS::new())
                    },
                    |(temp_dir, mut fs)| {
                        fs.ingest_directory(temp_dir.path(), false, &config)
                            .unwrap();
                        black_box(fs)
                    },
                )
            },
        );
    }

    group.finish();
}

/// Benchmark file extraction performance
fn bench_embrfs_extract(c: &mut Criterion) {
    let mut group = c.benchmark_group("embrfs_extract");
    group.sample_size(10);

    let config = ReversibleVSAConfig::default();

    // Setup: create and ingest files
    let temp_dir = TempDir::new().unwrap();
    create_test_files(&temp_dir, 50, 1024);

    let mut fs = EmbrFS::new();
    fs.ingest_directory(temp_dir.path(), false, &config)
        .unwrap();

    // Get a file ID to extract
    if let Some((&file_id, _)) = fs.engram.codebook.iter().next() {
        group.bench_function("extract_single_file", |bencher| {
            bencher.iter(|| {
                let data = fs
                    .engram
                    .codebook
                    .get(&file_id)
                    .unwrap()
                    .decode_data(&config, None, 1024);
                black_box(data)
            })
        });
    }

    group.finish();
}

/// Benchmark metadata operations
fn bench_embrfs_metadata(c: &mut Criterion) {
    let mut group = c.benchmark_group("embrfs_metadata");

    let config = ReversibleVSAConfig::default();

    let temp_dir = TempDir::new().unwrap();
    create_test_files(&temp_dir, 100, 1024);

    let mut fs = EmbrFS::new();
    fs.ingest_directory(temp_dir.path(), false, &config)
        .unwrap();

    group.bench_function("count_files", |bencher| {
        bencher.iter(|| {
            let count = fs.engram.codebook.len();
            black_box(count)
        })
    });

    group.bench_function("list_all_ids", |bencher| {
        bencher.iter(|| {
            let ids: Vec<usize> = fs.engram.codebook.keys().copied().collect();
            black_box(ids)
        })
    });

    group.finish();
}

/// Benchmark tree traversal operations
fn bench_tree_traversal(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_traversal");
    group.sample_size(10);

    let depths = vec![2, 3, 4];

    for depth in depths {
        group.bench_with_input(
            BenchmarkId::new("traverse_depth", depth),
            &depth,
            |bencher, &depth| {
                bencher.iter_with_setup(
                    || {
                        let temp_dir = TempDir::new().unwrap();
                        create_nested_structure(&temp_dir, depth, 3, 500);
                        temp_dir
                    },
                    |temp_dir| {
                        let mut file_count = 0;
                        for entry in walkdir::WalkDir::new(temp_dir.path()) {
                            if let Ok(entry) = entry {
                                if entry.file_type().is_file() {
                                    file_count += 1;
                                }
                            }
                        }
                        black_box(file_count)
                    },
                )
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_embrfs_ingest,
    bench_embrfs_nested,
    bench_embrfs_extract,
    bench_embrfs_metadata,
    bench_tree_traversal
);
criterion_main!(benches);
