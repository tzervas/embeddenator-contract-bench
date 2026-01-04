use crate::harness::{measure_fn, BenchConfig, Profile};
use crate::schema::Measurement;
use embeddenator::EmbrFS;
use embeddenator::retrieval::RerankedResult;
use embeddenator::ReversibleVSAConfig;
use rayon::prelude::*;
use serde_json::json;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::io;

#[derive(Clone, Debug)]
pub struct RetrievalArgs {
    pub input_dir: std::path::PathBuf,
    pub k: usize,
    pub candidate_factor: usize,
    pub queries: Option<usize>,
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

pub fn run(cfg: &BenchConfig, args: &RetrievalArgs) -> io::Result<Vec<Measurement>> {
    if !args.input_dir.is_dir() {
        return Err(io::Error::other("--input-dir must be a directory"));
    }

    let config = ReversibleVSAConfig::default();
    let mut fsys = EmbrFS::new();
    fsys.ingest_directory(&args.input_dir, false, &config)?;

    let engram = &fsys.engram;
    let index = engram.build_codebook_index();

    let mut codebook: Vec<(usize, embeddenator::SparseVec)> = engram
        .codebook
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    codebook.sort_by_key(|(k, _)| *k);

    let chunks = codebook.len();
    if chunks == 0 {
        return Err(io::Error::other("no chunks in codebook"));
    }

    let k = args.k.max(1).min(chunks);
    let candidate_k = (k.saturating_mul(args.candidate_factor)).max(50).min(chunks);

    let queries = match (cfg.profile, args.queries) {
        (_, Some(q)) => q,
        (Profile::Quick, None) => chunks.min(100),
        (Profile::Full, None) => chunks.min(1_000),
    }
    .max(1)
    .min(chunks);

    // Deterministic queries: take first N vectors.
    let query_vecs: Vec<(usize, embeddenator::SparseVec)> = codebook.iter().take(queries).cloned().collect();

    let warmup = cfg.warmup_iters().min(10);
    let iters = 1; // One measured pass over all queries.

    let mut last_stats = json!({});

    let m = measure_fn(iters, warmup, || {
        let mut latencies_ms: Vec<f64> = Vec::with_capacity(queries);
        let mut total_recall_hits: usize = 0;

        for (qid, qv) in &query_vecs {
            let start = std::time::Instant::now();
            let approx: Vec<RerankedResult> =
                engram.query_codebook_with_index(&index, qv, candidate_k, k);
            let elapsed = start.elapsed();
            latencies_ms.push(elapsed.as_secs_f64() * 1000.0);

            // Brute force exact top-k (parallel).
            let mut exact: Vec<(usize, f64)> = codebook
                .par_iter()
                .map(|(cid, cv)| (*cid, qv.cosine(cv)))
                .collect();
            exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            exact.truncate(k);

            let exact_ids: HashSet<usize> = exact.into_iter().map(|(id, _)| id).collect();
            let approx_ids: HashSet<usize> = approx.into_iter().map(|r| r.id).collect();

            let hits = approx_ids.intersection(&exact_ids).count();
            total_recall_hits += hits;

            let _ = qid; // keep deterministic query id in scope
        }

        latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let mean_ms = latencies_ms.iter().sum::<f64>() / (latencies_ms.len().max(1) as f64);
        let total_time_s = latencies_ms.iter().sum::<f64>() / 1000.0;
        let qps = if total_time_s <= 0.0 {
            0.0
        } else {
            (queries as f64) / total_time_s
        };
        let recall = (total_recall_hits as f64) / ((queries * k) as f64);

        last_stats = json!({
            "chunks": chunks,
            "queries": queries,
            "k": k,
            "candidate_k": candidate_k,
            "qps": qps,
            "latency_ms": {
                "p50": quantile(&latencies_ms, 0.50),
                "p95": quantile(&latencies_ms, 0.95),
                "p99": quantile(&latencies_ms, 0.99),
                "mean": mean_ms,
            },
            "recall_at_k": recall,
        });

        Ok::<(), io::Error>(())
    });

    Ok(vec![Measurement {
        name: "retrieval.query_codebook_with_index".to_string(),
        unit: "ns/iter".to_string(),
        iters: m.iters,
        warmup_iters: m.warmup_iters,
        total_ns: m.total_ns,
        ns_per_iter: m.ns_per_iter,
        bytes_processed: None,
        throughput_bytes_per_s: None,
        extra: json!({
            "input_dir": args.input_dir.to_string_lossy().to_string(),
            "stats": last_stats,
        }),
    }])
}
