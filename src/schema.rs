use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMeta {
    pub schema_version: u32,
    pub bench_version: String,
    pub profile: String,
    pub seed: u64,
    pub timestamp_utc: String,
    pub git_sha: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    pub name: String,
    pub unit: String,

    pub iters: u64,
    pub warmup_iters: u64,

    pub total_ns: u128,
    pub ns_per_iter: f64,

    pub bytes_processed: Option<u64>,
    pub throughput_bytes_per_s: Option<f64>,

    pub extra: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractBenchReport {
    pub run: RunMeta,
    pub measurements: Vec<Measurement>,
}
