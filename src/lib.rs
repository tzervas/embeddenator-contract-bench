use clap::ValueEnum;

pub mod benches;
pub mod dataset;
pub mod harness;
pub mod schema;

/// VSA substrate variant to benchmark.
#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub enum VsaVariant {
    /// Run all VSA substrate benchmarks (packed, bitsliced, hybrid, block-sparse).
    #[default]
    All,
    /// PackedTritVec substrate only.
    Packed,
    /// BitslicedTritVec substrate only.
    Bitsliced,
    /// CarrySaveBundle (hybrid) substrate only.
    Hybrid,
    /// BlockSparseTritVec substrate only (for large dimensions).
    BlockSparse,
}
