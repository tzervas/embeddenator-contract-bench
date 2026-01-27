//! I/O operations benchmark suite
//!
//! Benchmarks for I/O operations:
//! - File reading and writing at various sizes
//! - Compression codec performance
//! - Serialization/deserialization performance
//! - Batch I/O operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::fs;
use std::io::Write;
use tempfile::TempDir;

/// Benchmark file I/O operations at various sizes
fn bench_file_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("file_io");

    let sizes = vec![
        (1024, "1KB"),
        (10 * 1024, "10KB"),
        (100 * 1024, "100KB"),
        (1024 * 1024, "1MB"),
        (10 * 1024 * 1024, "10MB"),
    ];

    for (size, label) in sizes {
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        // Benchmark write
        group.bench_with_input(
            BenchmarkId::new("write", label),
            &size,
            |bencher, &_size| {
                bencher.iter_with_setup(
                    || TempDir::new().unwrap(),
                    |temp_dir| {
                        let file_path = temp_dir.path().join("test_file.bin");
                        let mut file = fs::File::create(&file_path).unwrap();
                        file.write_all(black_box(&data)).unwrap();
                        black_box(file)
                    },
                )
            },
        );

        // Benchmark read
        group.bench_with_input(BenchmarkId::new("read", label), &size, |bencher, &_size| {
            bencher.iter_with_setup(
                || {
                    let temp_dir = TempDir::new().unwrap();
                    let file_path = temp_dir.path().join("test_file.bin");
                    let mut file = fs::File::create(&file_path).unwrap();
                    file.write_all(&data).unwrap();
                    (temp_dir, file_path)
                },
                |(_temp_dir, file_path)| {
                    let content = fs::read(black_box(&file_path)).unwrap();
                    black_box(content)
                },
            )
        });
    }

    group.finish();
}

/// Benchmark batch file operations
fn bench_batch_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_io");
    group.sample_size(10);

    let file_counts = vec![10, 50, 100];
    let file_size = 10 * 1024; // 10KB per file

    for count in file_counts {
        group.bench_with_input(
            BenchmarkId::new("write_batch", count),
            &count,
            |bencher, &count| {
                let data: Vec<u8> = (0..file_size).map(|i| (i % 256) as u8).collect();

                bencher.iter_with_setup(
                    || TempDir::new().unwrap(),
                    |temp_dir| {
                        for i in 0..count {
                            let file_path = temp_dir.path().join(format!("file_{}.bin", i));
                            let mut file = fs::File::create(&file_path).unwrap();
                            file.write_all(black_box(&data)).unwrap();
                        }
                        black_box(temp_dir)
                    },
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("read_batch", count),
            &count,
            |bencher, &count| {
                bencher.iter_with_setup(
                    || {
                        let temp_dir = TempDir::new().unwrap();
                        let data: Vec<u8> = (0..file_size).map(|i| (i % 256) as u8).collect();
                        for i in 0..count {
                            let file_path = temp_dir.path().join(format!("file_{}.bin", i));
                            let mut file = fs::File::create(&file_path).unwrap();
                            file.write_all(&data).unwrap();
                        }
                        temp_dir
                    },
                    |temp_dir| {
                        for i in 0..count {
                            let file_path = temp_dir.path().join(format!("file_{}.bin", i));
                            let content = fs::read(black_box(&file_path)).unwrap();
                            black_box(content);
                        }
                    },
                )
            },
        );
    }

    group.finish();
}

/// Benchmark directory operations
fn bench_directory_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("directory_ops");

    let dir_counts = vec![10, 50, 100];

    for count in dir_counts {
        group.bench_with_input(
            BenchmarkId::new("create_dirs", count),
            &count,
            |bencher, &count| {
                bencher.iter_with_setup(
                    || TempDir::new().unwrap(),
                    |temp_dir| {
                        for i in 0..count {
                            let dir_path = temp_dir.path().join(format!("dir_{}", i));
                            fs::create_dir_all(black_box(&dir_path)).unwrap();
                        }
                        black_box(temp_dir)
                    },
                )
            },
        );
    }

    group.finish();
}

/// Benchmark serialization operations
fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Clone)]
    struct TestData {
        id: usize,
        name: String,
        values: Vec<f64>,
    }

    let sizes = vec![10, 100, 1000];

    for size in sizes {
        let data: Vec<TestData> = (0..size)
            .map(|i| TestData {
                id: i,
                name: format!("item_{}", i),
                values: vec![i as f64; 10],
            })
            .collect();

        // JSON serialization
        group.bench_with_input(
            BenchmarkId::new("json_serialize", size),
            &size,
            |bencher, &_size| {
                bencher.iter(|| {
                    let json = serde_json::to_string(black_box(&data)).unwrap();
                    black_box(json)
                })
            },
        );

        let json = serde_json::to_string(&data).unwrap();
        group.bench_with_input(
            BenchmarkId::new("json_deserialize", size),
            &size,
            |bencher, &_size| {
                bencher.iter(|| {
                    let data: Vec<TestData> = serde_json::from_str(black_box(&json)).unwrap();
                    black_box(data)
                })
            },
        );

        // Bincode serialization
        group.bench_with_input(
            BenchmarkId::new("bincode_serialize", size),
            &size,
            |bencher, &_size| {
                bencher.iter(|| {
                    let bytes = bincode::serialize(black_box(&data)).unwrap();
                    black_box(bytes)
                })
            },
        );

        let bytes = bincode::serialize(&data).unwrap();
        group.bench_with_input(
            BenchmarkId::new("bincode_deserialize", size),
            &size,
            |bencher, &_size| {
                bencher.iter(|| {
                    let data: Vec<TestData> = bincode::deserialize(black_box(&bytes)).unwrap();
                    black_box(data)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_file_io,
    bench_batch_io,
    bench_directory_ops,
    bench_serialization
);
criterion_main!(benches);
