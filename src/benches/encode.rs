use crate::schema::Measurement;
use crate::{harness::BenchConfig, harness::measure_fn};
use embeddenator::EmbrFS;
use embeddenator::{BinaryWriteOptions, CompressionCodec, PayloadKind, envelope};
use embeddenator::ReversibleVSAConfig;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

#[derive(Clone, Debug)]
pub struct EncodeArgs {
    pub inputs: Vec<PathBuf>,
    pub prefix: Option<String>,
    pub codec: CompressionCodec,
    pub codec_level: Option<i32>,
    pub verify: bool,
}

fn sha256_file(path: &Path) -> io::Result<[u8; 32]> {
    let bytes = fs::read(path)?;
    Ok(Sha256::digest(&bytes).into())
}

fn hex32(d: [u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn collect_files(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if root.is_file() {
        out.push(root.to_path_buf());
        return Ok(out);
    }

    for entry in walkdir::WalkDir::new(root).follow_links(false) {
        let entry = entry?;
        if entry.file_type().is_file() {
            out.push(entry.path().to_path_buf());
        }
    }
    out.sort();
    Ok(out)
}

fn logical_prefix_for_input(input: &Path, explicit: Option<&str>) -> String {
    if let Some(p) = explicit {
        return p.to_string();
    }
    input
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("input")
        .to_string()
}

pub fn run(cfg: &BenchConfig, args: &EncodeArgs) -> io::Result<Vec<Measurement>> {
    if args.inputs.is_empty() {
        return Err(io::Error::other("at least one input is required"));
    }

    let config = ReversibleVSAConfig::default();

    // Precompute raw bytes + hashes (for optional verification).
    let mut raw_bytes: u64 = 0;
    let mut original_hashes: BTreeMap<String, String> = BTreeMap::new();

    for input in &args.inputs {
        let files = collect_files(input)?;
        for f in files {
            raw_bytes += fs::metadata(&f)?.len();
            if args.verify {
                let rel = if input.is_dir() {
                    match f.strip_prefix(input).ok().and_then(|p| p.to_str()) {
                        Some(s) => s.replace('\\', "/"),
                        None => f.to_string_lossy().replace('\\', "/"),
                    }
                } else {
                    f.file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("input.bin")
                        .to_string()
                };

                let key = format!(
                    "{}/{}",
                    logical_prefix_for_input(input, args.prefix.as_deref()),
                    rel
                )
                .trim_end_matches('/')
                .to_string();

                original_hashes.insert(key, hex32(sha256_file(&f)?));
            }
        }
    }

    // Encode/ingest measurement: treat one ingest pass as one iteration.
    let warmup = cfg.warmup_iters().min(5);
    let iters = match cfg.profile {
        crate::harness::Profile::Quick => 3,
        crate::harness::Profile::Full => 10,
    };

    let mut last_sizes = None;
    let mut last_verify = None;

    let m = measure_fn(iters, warmup, || {
        let mut fsys = EmbrFS::new();
        for input in &args.inputs {
            if input.is_dir() {
                let prefix = logical_prefix_for_input(input, args.prefix.as_deref());
                fsys.ingest_directory_with_prefix(input, Some(&prefix), false, &config)?;
            } else {
                let prefix = logical_prefix_for_input(input, args.prefix.as_deref());
                let logical_path = format!(
                    "{}/{}",
                    prefix,
                    input
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("input.bin")
                );
                fsys.ingest_file(input, logical_path, false, &config)?;
            }
        }

        // Size stats after ingest.
        let root_bincode = bincode::serialize(&fsys.engram.root).map_err(io::Error::other)?;
        let codebook_bincode = bincode::serialize(&fsys.engram.codebook).map_err(io::Error::other)?;
        let corrections_bincode = bincode::serialize(&fsys.engram.corrections).map_err(io::Error::other)?;
        let manifest_json = serde_json::to_vec(&fsys.manifest).map_err(io::Error::other)?;

        let denom_bytes = (root_bincode.len() + codebook_bincode.len() + corrections_bincode.len() + manifest_json.len()) as f64;
        let effective_ratio = if denom_bytes <= 0.0 { 0.0 } else { (raw_bytes as f64) / denom_bytes };

        let engram_bincode = bincode::serialize(&fsys.engram).map_err(io::Error::other)?;
        let opts = BinaryWriteOptions {
            codec: args.codec,
            level: args.codec_level,
        };
        let wrapped = envelope::wrap_or_legacy(PayloadKind::EngramBincode, opts, &engram_bincode)?;

        let stats = fsys.correction_stats();

        last_sizes = Some(json!({
            "raw_bytes": raw_bytes,
            "root_bincode_bytes": root_bincode.len(),
            "codebook_bincode_bytes": codebook_bincode.len(),
            "corrections_bincode_bytes": corrections_bincode.len(),
            "manifest_json_bytes": manifest_json.len(),
            "engram_wrapped_bytes": wrapped.len(),
            "effective_ratio_including_corrections": effective_ratio,
            "corrections": {
                "total_chunks": stats.total_chunks,
                "perfect_ratio": stats.perfect_ratio,
                "correction_ratio": stats.correction_ratio,
            }
        }));

        if args.verify {
            let temp = TempDir::new()?;
            let engram_path = temp.path().join("root.engram");
            let manifest_path = temp.path().join("manifest.json");
            let out_dir = temp.path().join("out");

            fsys.save_engram_with_options(&engram_path, opts)?;
            fsys.save_manifest(&manifest_path)?;

            let e = EmbrFS::load_engram(&engram_path)?;
            let m = EmbrFS::load_manifest(&manifest_path)?;
            EmbrFS::extract(&e, &m, &out_dir, false, &config)?;

            let mut mismatches: u64 = 0;
            for (logical_path, expected_hash) in &original_hashes {
                let extracted_path = out_dir.join(logical_path);
                let got_hash = hex32(sha256_file(&extracted_path)?);
                if &got_hash != expected_hash {
                    mismatches += 1;
                }
            }
            last_verify = Some(json!({"ok": mismatches == 0, "mismatches": mismatches}));
        }

        Ok::<(), io::Error>(())
    });

    let sizes = last_sizes.unwrap_or_else(|| json!({}));

    let mut out = Vec::new();
    out.push(Measurement {
        name: "encode.ingest".to_string(),
        unit: "ns/iter".to_string(),
        iters: m.iters,
        warmup_iters: m.warmup_iters,
        total_ns: m.total_ns,
        ns_per_iter: m.ns_per_iter,
        bytes_processed: Some(raw_bytes),
        throughput_bytes_per_s: {
            let total_s = (m.total_ns as f64) / 1e9;
            if total_s <= 0.0 { None } else { Some((raw_bytes as f64) / total_s) }
        },
        extra: json!({
            "inputs": args.inputs.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>(),
            "codec": format!("{:?}", args.codec),
            "codec_level": args.codec_level,
            "sizes": sizes,
            "verify": last_verify,
        }),
    });

    Ok(out)
}
