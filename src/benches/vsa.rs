use crate::harness::{measure_fn, BenchConfig};
use crate::schema::Measurement;
use crate::VsaVariant;
use embeddenator::{BitslicedTritVec, BlockSparseTritVec, CarrySaveBundle, PackedTritVec, ReversibleVSAConfig, SparseVec, DIM};
use serde_json::json;
use std::hint::black_box;
use std::io;
use std::path::Path;
use std::time::Instant;

use crate::dataset::DatasetReader;

pub fn run(cfg: &BenchConfig, variant: VsaVariant) -> Vec<Measurement> {
    let warmup = cfg.warmup_iters();
    let iters = cfg.iters();

    let config = ReversibleVSAConfig::default();
    // Deterministic base vectors.
    let a = SparseVec::encode_data(b"alpha", &config, Some("/bench/vsa"));
    let b = SparseVec::encode_data(b"beta", &config, Some("/bench/vsa"));
    let c = SparseVec::encode_data(b"gamma", &config, Some("/bench/vsa"));

    let run_packed = matches!(variant, VsaVariant::All | VsaVariant::Packed);
    let run_bitsliced = matches!(variant, VsaVariant::All | VsaVariant::Bitsliced);
    let run_hybrid = matches!(variant, VsaVariant::All | VsaVariant::Hybrid);
    let run_block_sparse = matches!(variant, VsaVariant::All | VsaVariant::BlockSparse);

    let mut out = Vec::new();

    // SparseVec ops (these dynamically choose packed/hybrid paths depending on features/gates).
    // Always included as they represent the high-level API.
    {
        let m = measure_fn(iters, warmup, || a.bundle(&b));
        out.push(Measurement {
            name: "vsa.sparsevec.bundle".to_string(),
            unit: "ns/iter".to_string(),
            iters: m.iters,
            warmup_iters: m.warmup_iters,
            total_ns: m.total_ns,
            ns_per_iter: m.ns_per_iter,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": DIM}),
        });
    }
    {
        let m = measure_fn(iters, warmup, || a.bind(&b));
        out.push(Measurement {
            name: "vsa.sparsevec.bind".to_string(),
            unit: "ns/iter".to_string(),
            iters: m.iters,
            warmup_iters: m.warmup_iters,
            total_ns: m.total_ns,
            ns_per_iter: m.ns_per_iter,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": DIM}),
        });
    }
    {
        let m = measure_fn(iters, warmup, || a.cosine(&b));
        out.push(Measurement {
            name: "vsa.sparsevec.cosine".to_string(),
            unit: "ns/iter".to_string(),
            iters: m.iters,
            warmup_iters: m.warmup_iters,
            total_ns: m.total_ns,
            ns_per_iter: m.ns_per_iter,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": DIM}),
        });
    }

    // Explicit packed/bitsliced/hybrid substrate benches.
    // These are intended to stay stable even as SparseVec routing changes.
    if run_packed {
        let pa = PackedTritVec::from_sparsevec(&a, DIM);
        let pb = PackedTritVec::from_sparsevec(&b, DIM);

        {
            let m = measure_fn(iters, warmup, || pa.bundle(&pb));
            out.push(Measurement {
                name: "vsa.packed.bundle".to_string(),
                unit: "ns/iter".to_string(),
                iters: m.iters,
                warmup_iters: m.warmup_iters,
                total_ns: m.total_ns,
                ns_per_iter: m.ns_per_iter,
                bytes_processed: None,
                throughput_bytes_per_s: None,
                extra: json!({"dim": DIM}),
            });
        }
        {
            let m = measure_fn(iters, warmup, || pa.bind(&pb));
            out.push(Measurement {
                name: "vsa.packed.bind".to_string(),
                unit: "ns/iter".to_string(),
                iters: m.iters,
                warmup_iters: m.warmup_iters,
                total_ns: m.total_ns,
                ns_per_iter: m.ns_per_iter,
                bytes_processed: None,
                throughput_bytes_per_s: None,
                extra: json!({"dim": DIM}),
            });
        }
        {
            let m = measure_fn(iters, warmup, || pa.dot(&pb));
            out.push(Measurement {
                name: "vsa.packed.dot".to_string(),
                unit: "ns/iter".to_string(),
                iters: m.iters,
                warmup_iters: m.warmup_iters,
                total_ns: m.total_ns,
                ns_per_iter: m.ns_per_iter,
                bytes_processed: None,
                throughput_bytes_per_s: None,
                extra: json!({"dim": DIM}),
            });
        }
    }

    if run_bitsliced {
        let ba = BitslicedTritVec::from_sparse(&a, DIM);
        let bb = BitslicedTritVec::from_sparse(&b, DIM);

    {
        let m = measure_fn(iters, warmup, || ba.bundle_dispatch(&bb));
        out.push(Measurement {
            name: "vsa.bitsliced.bundle".to_string(),
            unit: "ns/iter".to_string(),
            iters: m.iters,
            warmup_iters: m.warmup_iters,
            total_ns: m.total_ns,
            ns_per_iter: m.ns_per_iter,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": DIM}),
        });
    }
    {
        let m = measure_fn(iters, warmup, || ba.bind_dispatch(&bb));
        out.push(Measurement {
            name: "vsa.bitsliced.bind".to_string(),
            unit: "ns/iter".to_string(),
            iters: m.iters,
            warmup_iters: m.warmup_iters,
            total_ns: m.total_ns,
            ns_per_iter: m.ns_per_iter,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": DIM}),
        });
    }
        {
            let m = measure_fn(iters, warmup, || ba.cosine(&bb));
            out.push(Measurement {
                name: "vsa.bitsliced.cosine".to_string(),
                unit: "ns/iter".to_string(),
                iters: m.iters,
                warmup_iters: m.warmup_iters,
                total_ns: m.total_ns,
                ns_per_iter: m.ns_per_iter,
                bytes_processed: None,
                throughput_bytes_per_s: None,
                extra: json!({"dim": DIM}),
            });
        }
    }

    // Hybrid bundling: Carry-save accumulator, then finalize.
    if run_hybrid {
        let ba3 = BitslicedTritVec::from_sparse(&a, DIM);
        let bb3 = BitslicedTritVec::from_sparse(&b, DIM);
        let bc3 = BitslicedTritVec::from_sparse(&c, DIM);

        let m = measure_fn(iters, warmup, || {
            let mut acc = CarrySaveBundle::new(DIM);
            acc.accumulate(&ba3);
            acc.accumulate(&bb3);
            acc.accumulate(&bc3);
            acc.finalize()
        });
        out.push(Measurement {
            name: "vsa.hybrid.carry_save_bundle_3".to_string(),
            unit: "ns/iter".to_string(),
            iters: m.iters,
            warmup_iters: m.warmup_iters,
            total_ns: m.total_ns,
            ns_per_iter: m.ns_per_iter,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": DIM, "n": 3}),
        });
    }

    // Block-sparse benchmarks: optimized for large dimensions with low density.
    // Note: At DIM=10K with typical sparsity, block-sparse may not show advantages.
    // These benchmarks are included for completeness; use large-dimension datasets
    // (100K+) to see the true benefits of block-sparse representation.
    if run_block_sparse {
        let bsa = BlockSparseTritVec::from_sparse(&a, DIM);
        let bsb = BlockSparseTritVec::from_sparse(&b, DIM);
        let bsc = BlockSparseTritVec::from_sparse(&c, DIM);

        {
            let m = measure_fn(iters, warmup, || bsa.bind_dispatch(&bsb));
            out.push(Measurement {
                name: "vsa.blocksparse.bind".to_string(),
                unit: "ns/iter".to_string(),
                iters: m.iters,
                warmup_iters: m.warmup_iters,
                total_ns: m.total_ns,
                ns_per_iter: m.ns_per_iter,
                bytes_processed: None,
                throughput_bytes_per_s: None,
                extra: json!({"dim": DIM, "blocks_a": bsa.block_count(), "blocks_b": bsb.block_count()}),
            });
        }
        {
            let m = measure_fn(iters, warmup, || bsa.bundle_dispatch(&bsb));
            out.push(Measurement {
                name: "vsa.blocksparse.bundle".to_string(),
                unit: "ns/iter".to_string(),
                iters: m.iters,
                warmup_iters: m.warmup_iters,
                total_ns: m.total_ns,
                ns_per_iter: m.ns_per_iter,
                bytes_processed: None,
                throughput_bytes_per_s: None,
                extra: json!({"dim": DIM, "blocks_a": bsa.block_count(), "blocks_b": bsb.block_count()}),
            });
        }
        {
            let m = measure_fn(iters, warmup, || bsa.dot_dispatch(&bsb));
            out.push(Measurement {
                name: "vsa.blocksparse.dot".to_string(),
                unit: "ns/iter".to_string(),
                iters: m.iters,
                warmup_iters: m.warmup_iters,
                total_ns: m.total_ns,
                ns_per_iter: m.ns_per_iter,
                bytes_processed: None,
                throughput_bytes_per_s: None,
                extra: json!({"dim": DIM, "blocks_a": bsa.block_count(), "blocks_b": bsb.block_count()}),
            });
        }
        {
            let m = measure_fn(iters, warmup, || bsa.cosine_dispatch(&bsb));
            out.push(Measurement {
                name: "vsa.blocksparse.cosine".to_string(),
                unit: "ns/iter".to_string(),
                iters: m.iters,
                warmup_iters: m.warmup_iters,
                total_ns: m.total_ns,
                ns_per_iter: m.ns_per_iter,
                bytes_processed: None,
                throughput_bytes_per_s: None,
                extra: json!({"dim": DIM, "blocks_a": bsa.block_count(), "blocks_b": bsb.block_count()}),
            });
        }
        {
            // Bundle-many using block-sparse pairwise reduction
            let vecs = vec![bsa.clone(), bsb.clone(), bsc.clone()];
            let m = measure_fn(iters, warmup, || BlockSparseTritVec::bundle_many(&vecs));
            out.push(Measurement {
                name: "vsa.blocksparse.bundle_many_3".to_string(),
                unit: "ns/iter".to_string(),
                iters: m.iters,
                warmup_iters: m.warmup_iters,
                total_ns: m.total_ns,
                ns_per_iter: m.ns_per_iter,
                bytes_processed: None,
                throughput_bytes_per_s: None,
                extra: json!({"dim": DIM, "n": 3}),
            });
        }
    }

    out
}

fn dataset_ops_for_profile(cfg: &BenchConfig, available: u64) -> u64 {
    match cfg.profile {
        crate::harness::Profile::Quick => available.min(10_000),
        crate::harness::Profile::Full => available,
    }
}

pub fn run_dataset(cfg: &BenchConfig, variant: VsaVariant, dataset_path: &Path) -> io::Result<Vec<Measurement>> {
    let mut reader = DatasetReader::open(dataset_path)?;
    let meta = reader.meta().clone();
    let dim = meta.dimension as usize;

    // We process pairs (a,b) for most ops.
    let available_pairs = meta.count.saturating_sub(1) / 2;
    let pairs = dataset_ops_for_profile(cfg, available_pairs);

    // For hybrid we process triples (a,b,c).
    let available_triples = meta.count.saturating_sub(2) / 3;
    let triples = dataset_ops_for_profile(cfg, available_triples);

    let run_packed = matches!(variant, VsaVariant::All | VsaVariant::Packed);
    let run_bitsliced = matches!(variant, VsaVariant::All | VsaVariant::Bitsliced);
    let run_hybrid = matches!(variant, VsaVariant::All | VsaVariant::Hybrid);
    let run_block_sparse = matches!(variant, VsaVariant::All | VsaVariant::BlockSparse);

    let mut out = Vec::new();

    // --- SparseVec dataset ops (always included) ---
    {
        reader.reset()?;
        let it = reader.by_ref();
        let mut i = 0u64;
        let start = Instant::now();
        while i < pairs {
            let a = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let b = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            black_box(a.bundle(&b));
            i += 1;
        }
        let elapsed = start.elapsed();
        let total_ns = elapsed.as_nanos();
        let denom = pairs.max(1) as f64;
        let ops_per_s = (pairs as f64) / elapsed.as_secs_f64().max(1e-12);
        out.push(Measurement {
            name: "vsa_dataset.sparsevec.bundle".to_string(),
            unit: "ns/op".to_string(),
            iters: pairs,
            warmup_iters: 0,
            total_ns,
            ns_per_iter: (total_ns as f64) / denom,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": dim, "dataset": dataset_path.display().to_string(), "vectors": meta.count, "ops": pairs, "ops_per_s": ops_per_s}),
        });
    }
    {
        reader.reset()?;
        let it = reader.by_ref();
        let mut i = 0u64;
        let start = Instant::now();
        while i < pairs {
            let a = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let b = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            black_box(a.bind(&b));
            i += 1;
        }
        let elapsed = start.elapsed();
        let total_ns = elapsed.as_nanos();
        let denom = pairs.max(1) as f64;
        let ops_per_s = (pairs as f64) / elapsed.as_secs_f64().max(1e-12);
        out.push(Measurement {
            name: "vsa_dataset.sparsevec.bind".to_string(),
            unit: "ns/op".to_string(),
            iters: pairs,
            warmup_iters: 0,
            total_ns,
            ns_per_iter: (total_ns as f64) / denom,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": dim, "dataset": dataset_path.display().to_string(), "vectors": meta.count, "ops": pairs, "ops_per_s": ops_per_s}),
        });
    }
    {
        reader.reset()?;
        let it = reader.by_ref();
        let mut i = 0u64;
        let start = Instant::now();
        while i < pairs {
            let a = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let b = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            black_box(a.cosine(&b));
            i += 1;
        }
        let elapsed = start.elapsed();
        let total_ns = elapsed.as_nanos();
        let denom = pairs.max(1) as f64;
        let ops_per_s = (pairs as f64) / elapsed.as_secs_f64().max(1e-12);
        out.push(Measurement {
            name: "vsa_dataset.sparsevec.cosine".to_string(),
            unit: "ns/op".to_string(),
            iters: pairs,
            warmup_iters: 0,
            total_ns,
            ns_per_iter: (total_ns as f64) / denom,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": dim, "dataset": dataset_path.display().to_string(), "vectors": meta.count, "ops": pairs, "ops_per_s": ops_per_s}),
        });
    }

    // --- Packed dataset ops ---
    if run_packed {
        // bundle
        reader.reset()?;
        let it = reader.by_ref();
        let mut i = 0u64;
        let start = Instant::now();
        while i < pairs {
            let a = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let b = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let pa = PackedTritVec::from_sparsevec(&a, dim);
            let pb = PackedTritVec::from_sparsevec(&b, dim);
            black_box(pa.bundle(&pb));
            i += 1;
        }
        let elapsed = start.elapsed();
        let total_ns = elapsed.as_nanos();
        let denom = pairs.max(1) as f64;
        let ops_per_s = (pairs as f64) / elapsed.as_secs_f64().max(1e-12);
        out.push(Measurement {
            name: "vsa_dataset.packed.bundle".to_string(),
            unit: "ns/op".to_string(),
            iters: pairs,
            warmup_iters: 0,
            total_ns,
            ns_per_iter: (total_ns as f64) / denom,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": dim, "dataset": dataset_path.display().to_string(), "vectors": meta.count, "ops": pairs, "ops_per_s": ops_per_s}),
        });

        // bind
        reader.reset()?;
        let it = reader.by_ref();
        let mut i = 0u64;
        let start = Instant::now();
        while i < pairs {
            let a = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let b = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let pa = PackedTritVec::from_sparsevec(&a, dim);
            let pb = PackedTritVec::from_sparsevec(&b, dim);
            black_box(pa.bind(&pb));
            i += 1;
        }
        let elapsed = start.elapsed();
        let total_ns = elapsed.as_nanos();
        let denom = pairs.max(1) as f64;
        let ops_per_s = (pairs as f64) / elapsed.as_secs_f64().max(1e-12);
        out.push(Measurement {
            name: "vsa_dataset.packed.bind".to_string(),
            unit: "ns/op".to_string(),
            iters: pairs,
            warmup_iters: 0,
            total_ns,
            ns_per_iter: (total_ns as f64) / denom,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": dim, "dataset": dataset_path.display().to_string(), "vectors": meta.count, "ops": pairs, "ops_per_s": ops_per_s}),
        });

        // dot
        reader.reset()?;
        let it = reader.by_ref();
        let mut i = 0u64;
        let start = Instant::now();
        while i < pairs {
            let a = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let b = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let pa = PackedTritVec::from_sparsevec(&a, dim);
            let pb = PackedTritVec::from_sparsevec(&b, dim);
            black_box(pa.dot(&pb));
            i += 1;
        }
        let elapsed = start.elapsed();
        let total_ns = elapsed.as_nanos();
        let denom = pairs.max(1) as f64;
        let ops_per_s = (pairs as f64) / elapsed.as_secs_f64().max(1e-12);
        out.push(Measurement {
            name: "vsa_dataset.packed.dot".to_string(),
            unit: "ns/op".to_string(),
            iters: pairs,
            warmup_iters: 0,
            total_ns,
            ns_per_iter: (total_ns as f64) / denom,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": dim, "dataset": dataset_path.display().to_string(), "vectors": meta.count, "ops": pairs, "ops_per_s": ops_per_s}),
        });
    }

    // --- Bitsliced dataset ops ---
    if run_bitsliced {
        // bundle
        reader.reset()?;
        let it = reader.by_ref();
        let mut i = 0u64;
        let start = Instant::now();
        while i < pairs {
            let a = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let b = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let ba = BitslicedTritVec::from_sparse(&a, dim);
            let bb = BitslicedTritVec::from_sparse(&b, dim);
            black_box(ba.bundle_dispatch(&bb));
            i += 1;
        }
        let elapsed = start.elapsed();
        let total_ns = elapsed.as_nanos();
        let denom = pairs.max(1) as f64;
        let ops_per_s = (pairs as f64) / elapsed.as_secs_f64().max(1e-12);
        out.push(Measurement {
            name: "vsa_dataset.bitsliced.bundle".to_string(),
            unit: "ns/op".to_string(),
            iters: pairs,
            warmup_iters: 0,
            total_ns,
            ns_per_iter: (total_ns as f64) / denom,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": dim, "dataset": dataset_path.display().to_string(), "vectors": meta.count, "ops": pairs, "ops_per_s": ops_per_s}),
        });

        // bind
        reader.reset()?;
        let it = reader.by_ref();
        let mut i = 0u64;
        let start = Instant::now();
        while i < pairs {
            let a = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let b = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let ba = BitslicedTritVec::from_sparse(&a, dim);
            let bb = BitslicedTritVec::from_sparse(&b, dim);
            black_box(ba.bind_dispatch(&bb));
            i += 1;
        }
        let elapsed = start.elapsed();
        let total_ns = elapsed.as_nanos();
        let denom = pairs.max(1) as f64;
        let ops_per_s = (pairs as f64) / elapsed.as_secs_f64().max(1e-12);
        out.push(Measurement {
            name: "vsa_dataset.bitsliced.bind".to_string(),
            unit: "ns/op".to_string(),
            iters: pairs,
            warmup_iters: 0,
            total_ns,
            ns_per_iter: (total_ns as f64) / denom,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": dim, "dataset": dataset_path.display().to_string(), "vectors": meta.count, "ops": pairs, "ops_per_s": ops_per_s}),
        });

        // cosine
        reader.reset()?;
        let it = reader.by_ref();
        let mut i = 0u64;
        let start = Instant::now();
        while i < pairs {
            let a = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let b = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let ba = BitslicedTritVec::from_sparse(&a, dim);
            let bb = BitslicedTritVec::from_sparse(&b, dim);
            black_box(ba.cosine(&bb));
            i += 1;
        }
        let elapsed = start.elapsed();
        let total_ns = elapsed.as_nanos();
        let denom = pairs.max(1) as f64;
        let ops_per_s = (pairs as f64) / elapsed.as_secs_f64().max(1e-12);
        out.push(Measurement {
            name: "vsa_dataset.bitsliced.cosine".to_string(),
            unit: "ns/op".to_string(),
            iters: pairs,
            warmup_iters: 0,
            total_ns,
            ns_per_iter: (total_ns as f64) / denom,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": dim, "dataset": dataset_path.display().to_string(), "vectors": meta.count, "ops": pairs, "ops_per_s": ops_per_s}),
        });
    }

    // --- Hybrid dataset ops ---
    if run_hybrid {
        reader.reset()?;
        let it = reader.by_ref();
        let mut i = 0u64;
        let start = Instant::now();
        while i < triples {
            let a = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let b = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let c = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let ba = BitslicedTritVec::from_sparse(&a, dim);
            let bb = BitslicedTritVec::from_sparse(&b, dim);
            let bc = BitslicedTritVec::from_sparse(&c, dim);
            let mut acc = CarrySaveBundle::new(dim);
            acc.accumulate(&ba);
            acc.accumulate(&bb);
            acc.accumulate(&bc);
            black_box(acc.finalize());
            i += 1;
        }
        let elapsed = start.elapsed();
        let total_ns = elapsed.as_nanos();
        let denom = triples.max(1) as f64;
        let ops_per_s = (triples as f64) / elapsed.as_secs_f64().max(1e-12);
        out.push(Measurement {
            name: "vsa_dataset.hybrid.carry_save_bundle_3".to_string(),
            unit: "ns/op".to_string(),
            iters: triples,
            warmup_iters: 0,
            total_ns,
            ns_per_iter: (total_ns as f64) / denom,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": dim, "dataset": dataset_path.display().to_string(), "vectors": meta.count, "ops": triples, "ops_per_s": ops_per_s, "n": 3}),
        });
    }

    // --- Block-sparse dataset ops ---
    // Block-sparse is optimized for large dimensions (100K+) with low density (<1%).
    // At smaller dimensions, bitsliced may outperform.
    if run_block_sparse {
        // bind
        reader.reset()?;
        let it = reader.by_ref();
        let mut i = 0u64;
        let start = Instant::now();
        while i < pairs {
            let a = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let b = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let bsa = BlockSparseTritVec::from_sparse(&a, dim);
            let bsb = BlockSparseTritVec::from_sparse(&b, dim);
            black_box(bsa.bind_dispatch(&bsb));
            i += 1;
        }
        let elapsed = start.elapsed();
        let total_ns = elapsed.as_nanos();
        let denom = pairs.max(1) as f64;
        let ops_per_s = (pairs as f64) / elapsed.as_secs_f64().max(1e-12);
        out.push(Measurement {
            name: "vsa_dataset.blocksparse.bind".to_string(),
            unit: "ns/op".to_string(),
            iters: pairs,
            warmup_iters: 0,
            total_ns,
            ns_per_iter: (total_ns as f64) / denom,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": dim, "dataset": dataset_path.display().to_string(), "vectors": meta.count, "ops": pairs, "ops_per_s": ops_per_s}),
        });

        // bundle
        reader.reset()?;
        let it = reader.by_ref();
        let mut i = 0u64;
        let start = Instant::now();
        while i < pairs {
            let a = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let b = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let bsa = BlockSparseTritVec::from_sparse(&a, dim);
            let bsb = BlockSparseTritVec::from_sparse(&b, dim);
            black_box(bsa.bundle_dispatch(&bsb));
            i += 1;
        }
        let elapsed = start.elapsed();
        let total_ns = elapsed.as_nanos();
        let denom = pairs.max(1) as f64;
        let ops_per_s = (pairs as f64) / elapsed.as_secs_f64().max(1e-12);
        out.push(Measurement {
            name: "vsa_dataset.blocksparse.bundle".to_string(),
            unit: "ns/op".to_string(),
            iters: pairs,
            warmup_iters: 0,
            total_ns,
            ns_per_iter: (total_ns as f64) / denom,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": dim, "dataset": dataset_path.display().to_string(), "vectors": meta.count, "ops": pairs, "ops_per_s": ops_per_s}),
        });

        // cosine
        reader.reset()?;
        let it = reader.by_ref();
        let mut i = 0u64;
        let start = Instant::now();
        while i < pairs {
            let a = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let b = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let bsa = BlockSparseTritVec::from_sparse(&a, dim);
            let bsb = BlockSparseTritVec::from_sparse(&b, dim);
            black_box(bsa.cosine_dispatch(&bsb));
            i += 1;
        }
        let elapsed = start.elapsed();
        let total_ns = elapsed.as_nanos();
        let denom = pairs.max(1) as f64;
        let ops_per_s = (pairs as f64) / elapsed.as_secs_f64().max(1e-12);
        out.push(Measurement {
            name: "vsa_dataset.blocksparse.cosine".to_string(),
            unit: "ns/op".to_string(),
            iters: pairs,
            warmup_iters: 0,
            total_ns,
            ns_per_iter: (total_ns as f64) / denom,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": dim, "dataset": dataset_path.display().to_string(), "vectors": meta.count, "ops": pairs, "ops_per_s": ops_per_s}),
        });

        // bundle_many (3 vectors)
        reader.reset()?;
        let it = reader.by_ref();
        let mut i = 0u64;
        let start = Instant::now();
        while i < triples {
            let a = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let b = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let c = it.next().transpose()?.ok_or_else(|| io::Error::other("unexpected EOF"))?;
            let bsa = BlockSparseTritVec::from_sparse(&a, dim);
            let bsb = BlockSparseTritVec::from_sparse(&b, dim);
            let bsc = BlockSparseTritVec::from_sparse(&c, dim);
            let vecs = vec![bsa, bsb, bsc];
            black_box(BlockSparseTritVec::bundle_many(&vecs));
            i += 1;
        }
        let elapsed = start.elapsed();
        let total_ns = elapsed.as_nanos();
        let denom = triples.max(1) as f64;
        let ops_per_s = (triples as f64) / elapsed.as_secs_f64().max(1e-12);
        out.push(Measurement {
            name: "vsa_dataset.blocksparse.bundle_many_3".to_string(),
            unit: "ns/op".to_string(),
            iters: triples,
            warmup_iters: 0,
            total_ns,
            ns_per_iter: (total_ns as f64) / denom,
            bytes_processed: None,
            throughput_bytes_per_s: None,
            extra: json!({"dim": dim, "dataset": dataset_path.display().to_string(), "vectors": meta.count, "ops": triples, "ops_per_s": ops_per_s, "n": 3}),
        });
    }

    Ok(out)
}
