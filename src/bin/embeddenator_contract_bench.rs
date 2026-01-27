use clap::{Parser, Subcommand, ValueEnum};
use embeddenator_contract_bench::benches;
use embeddenator_contract_bench::dataset::{self, GenerateConfig};
use embeddenator_contract_bench::harness::{BenchConfig, Profile};
use embeddenator_contract_bench::schema::{ContractBenchReport, RunMeta};
use embeddenator_contract_bench::VsaVariant;
use std::fs;
use std::io;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ProfileArg {
    Quick,
    Full,
}

impl From<ProfileArg> for Profile {
    fn from(v: ProfileArg) -> Self {
        match v {
            ProfileArg::Quick => Profile::Quick,
            ProfileArg::Full => Profile::Full,
        }
    }
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Balanced-ternary substrate operations: packed/bitsliced/hybrid + SparseVec.
    Vsa {
        /// Which VSA substrate variant(s) to benchmark.
        #[arg(long, value_enum, default_value_t = VsaVariant::All)]
        variant: VsaVariant,

        /// Optional dataset file for scaled benchmarks.
        ///
        /// If provided, the VSA benches run over the dataset vectors (streamed from disk)
        /// instead of the small fixed "alpha/beta/gamma" microbench inputs.
        #[arg(long, value_name = "FILE")]
        dataset: Option<PathBuf>,
    },

    /// Encode/extract contract metrics (ingest time, size breakdown; optional verify).
    Encode {
        /// Input directory or file. Can be provided multiple times.
        #[arg(short, long, value_name = "PATH", num_args = 1.., action = clap::ArgAction::Append)]
        input: Vec<PathBuf>,

        /// Logical prefix for each input path; defaults to basename.
        #[arg(long, value_name = "PREFIX")]
        prefix: Option<String>,

        /// Engram compression codec (none|zstd|lz4).
        #[arg(long, default_value = "none")]
        codec: String,

        /// Engram compression level (codec-dependent; used for zstd).
        #[arg(long)]
        level: Option<i32>,

        /// Perform an extract + SHA256 verify pass.
        #[arg(long, default_value_t = false)]
        verify: bool,
    },

    /// Retrieval seam metrics (approx QPS/latency + recall@k vs brute force).
    Retrieval {
        #[arg(long, value_name = "DIR")]
        input_dir: PathBuf,

        #[arg(long, default_value_t = 10)]
        k: usize,

        #[arg(long, default_value_t = 10)]
        candidate_factor: usize,

        #[arg(long)]
        queries: Option<usize>,
    },

    /// Run all contract benches.
    Suite {
        #[arg(short, long, value_name = "PATH", num_args = 1.., action = clap::ArgAction::Append)]
        input: Vec<PathBuf>,

        #[arg(long, value_name = "DIR")]
        retrieval_input_dir: Option<PathBuf>,

        #[arg(long, default_value = "none")]
        codec: String,

        #[arg(long)]
        level: Option<i32>,

        #[arg(long, default_value_t = false)]
        verify: bool,

        /// Which VSA substrate variant(s) to benchmark.
        #[arg(long, value_enum, default_value_t = VsaVariant::All)]
        variant: VsaVariant,
    },

    /// Generate a deterministic dataset of SparseVec vectors for scaled benchmarks.
    ///
    /// Creates binary files containing 10K/100K/1M vectors that can be loaded
    /// efficiently for VSA substrate benchmarking.
    GenerateDataset {
        /// Number of vectors to generate.
        #[arg(long, short = 'n', default_value_t = 10_000)]
        count: u64,

        /// Output directory for the generated dataset.
        #[arg(long, short = 'o', value_name = "DIR")]
        output: PathBuf,

        /// Random seed for deterministic generation.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Target sparsity per sign (number of +1 and -1 indices each).
        /// Default is dimension/100 (~1% density).
        #[arg(long)]
        sparsity: Option<usize>,

        /// Vector dimension. Default is 10000.
        #[arg(long, default_value_t = 10_000)]
        dimension: usize,
    },

    /// Show metadata for a generated dataset file.
    DatasetInfo {
        /// Path to the dataset file.
        #[arg(value_name = "FILE")]
        path: PathBuf,
    },
}

#[derive(Parser, Debug)]
#[command(name = "embeddenator-contract-bench")]
#[command(about = "Deterministic-ish contract benchmark runner (JSON output)")]
struct Args {
    #[arg(long, value_enum, default_value_t = ProfileArg::Quick, global = true)]
    profile: ProfileArg,

    #[arg(long, default_value_t = 0, global = true)]
    seed: u64,

    /// Where to write the JSON report. If omitted, prints to stdout.
    #[arg(long, global = true)]
    out: Option<PathBuf>,

    #[command(subcommand)]
    cmd: Command,
}

fn now_utc_rfc3339() -> String {
    // Avoid adding chrono dependency; this is "good enough" for filenames + reports.
    // Format: YYYY-MM-DDTHH:MM:SSZ
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("unix:{secs}")
}

fn git_sha_short() -> Option<String> {
    // Best-effort: read from environment set by CI/build scripts.
    std::env::var("GIT_SHA")
        .ok()
        .or_else(|| std::env::var("GITHUB_SHA").ok())
        .map(|s| s.chars().take(12).collect())
}

// Note: parse_codec temporarily disabled due to unavailable envelope API
/*
fn parse_codec(s: &str) -> io::Result<embeddenator::envelope::CompressionCodec> {
    match s.to_ascii_lowercase().as_str() {
        "none" => Ok(embeddenator::envelope::CompressionCodec::None),
        "zstd" => Ok(embeddenator::envelope::CompressionCodec::Zstd),
        "lz4" => Ok(embeddenator::envelope::CompressionCodec::Lz4),
        _ => Err(io::Error::other(format!("unknown codec: {s} (none|zstd|lz4)"))),
    }
}
*/

/// Format vector count as human-readable suffix (10k, 100k, 1m, etc.)
fn format_count(count: u64) -> String {
    match count {
        n if n >= 1_000_000 && n % 1_000_000 == 0 => format!("{}m", n / 1_000_000),
        n if n >= 1_000 && n % 1_000 == 0 => format!("{}k", n / 1_000),
        n => n.to_string(),
    }
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    let cfg = BenchConfig {
        profile: args.profile.into(),
        seed: args.seed,
    };

    let mut measurements = Vec::new();

    match &args.cmd {
        Command::Vsa { variant, dataset } => {
            if let Some(path) = dataset {
                measurements.extend(benches::vsa::run_dataset(&cfg, *variant, path)?);
            } else {
                measurements.extend(benches::vsa::run(&cfg, *variant));
            }
        }
        Command::Encode {
            input: _,
            prefix: _,
            codec: _,
            level: _,
            verify: _,
        } => {
            // Note: encode benchmarks temporarily disabled due to unavailable IO APIs
            eprintln!("Encode benchmarks are temporarily disabled");
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "encode benchmarks temporarily disabled",
            ));
            /*
            let codec = parse_codec(codec)?;
            let enc_args = benches::encode::EncodeArgs {
                inputs: input.clone(),
                prefix: prefix.clone(),
                codec,
                codec_level: *level,
                verify: *verify,
            };
            measurements.extend(benches::encode::run(&cfg, &enc_args)?);
            */
        }
        Command::Retrieval {
            input_dir,
            k,
            candidate_factor,
            queries,
        } => {
            let r_args = benches::retrieval::RetrievalArgs {
                input_dir: input_dir.clone(),
                k: *k,
                candidate_factor: *candidate_factor,
                queries: *queries,
            };
            measurements.extend(benches::retrieval::run(&cfg, &r_args)?);
        }
        Command::Suite {
            input,
            retrieval_input_dir,
            codec: _,
            level: _,
            verify: _,
            variant,
        } => {
            measurements.extend(benches::vsa::run(&cfg, *variant));

            if !input.is_empty() {
                // Note: encode benchmarks temporarily disabled due to unavailable IO APIs
                eprintln!("Warning: Encode benchmarks are temporarily disabled");
                /*
                let codec = parse_codec(codec)?;
                let enc_args = benches::encode::EncodeArgs {
                    inputs: input.clone(),
                    prefix: None,
                    codec,
                    codec_level: *level,
                    verify: *verify,
                };
                measurements.extend(benches::encode::run(&cfg, &enc_args)?);
                */
            }

            if let Some(dir) = retrieval_input_dir {
                let r_args = benches::retrieval::RetrievalArgs {
                    input_dir: dir.clone(),
                    k: 10,
                    candidate_factor: 10,
                    queries: None,
                };
                measurements.extend(benches::retrieval::run(&cfg, &r_args)?);
            }
        }
        Command::GenerateDataset {
            count,
            output,
            seed,
            sparsity,
            dimension,
        } => {
            let sparsity = sparsity.unwrap_or(dimension / 100);
            let gen_config = GenerateConfig {
                count: *count,
                dimension: *dimension,
                seed: *seed,
                sparsity,
            };

            // Create output directory
            fs::create_dir_all(output)?;

            // Generate filename based on count
            let filename = format!(
                "sparsevec_{}_{}_seed{}.embr",
                format_count(*count),
                dimension,
                seed
            );
            let filepath = output.join(&filename);

            eprintln!(
                "Generating {} vectors (dim={}, sparsity={}, seed={})...",
                count, dimension, sparsity, seed
            );

            let start = std::time::Instant::now();
            // Stream directly to disk to avoid materializing Vec<SparseVec> (RAM spike at 1M+).
            dataset::write_dataset_streaming(&filepath, &gen_config, 4096)?;
            let elapsed = start.elapsed();

            let file_size = fs::metadata(&filepath)?.len();
            eprintln!(
                "Wrote {:.2} MB in {:.2}s ({:.1} MB/s, {:.0} vec/s)",
                file_size as f64 / 1_048_576.0,
                elapsed.as_secs_f64(),
                (file_size as f64 / 1_048_576.0) / elapsed.as_secs_f64(),
                (*count as f64) / elapsed.as_secs_f64()
            );

            eprintln!("\nDataset saved: {}", filepath.display());
            eprintln!("  Vectors: {}", count);
            eprintln!("  Dimension: {}", dimension);
            eprintln!(
                "  Sparsity: {} per sign (~{:.1}% density)",
                sparsity,
                (sparsity * 2) as f64 / *dimension as f64 * 100.0
            );
            eprintln!("  Seed: {}", seed);
            eprintln!("  File size: {:.2} MB", file_size as f64 / 1_048_576.0);

            // Skip normal JSON report for generate-dataset
            return Ok(());
        }
        Command::DatasetInfo { path } => {
            let meta = dataset::read_dataset_meta(path)?;
            eprintln!("Dataset: {}", path.display());
            eprintln!("  Vectors: {}", meta.count);
            eprintln!("  Dimension: {}", meta.dimension);
            eprintln!("  Seed: {}", meta.seed);

            let file_size = fs::metadata(path)?.len();
            eprintln!("  File size: {:.2} MB", file_size as f64 / 1_048_576.0);

            // Skip normal JSON report
            return Ok(());
        }
    }

    let report = ContractBenchReport {
        run: RunMeta {
            schema_version: 1,
            bench_version: env!("CARGO_PKG_VERSION").to_string(),
            profile: cfg.profile.as_str().to_string(),
            seed: cfg.seed,
            timestamp_utc: now_utc_rfc3339(),
            git_sha: git_sha_short(),
        },
        measurements,
    };

    let json = serde_json::to_string_pretty(&report).map_err(io::Error::other)?;
    if let Some(out) = args.out {
        fs::write(out, json)?;
    } else {
        println!("{json}");
    }

    Ok(())
}
