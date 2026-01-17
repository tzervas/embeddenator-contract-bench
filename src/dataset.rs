//! Dataset generation and loading for scaled VSA benchmarks.
//!
//! Generates deterministic `SparseVec` datasets at various scales (10K/100K/1M vectors)
//! for benchmarking VSA substrate performance.
//!
//! # Binary Format
//!
//! ```text
//! Header:
//!   magic: [u8; 8]  = b"EMBR_DST"
//!   version: u32    = 1
//!   count: u64      = number of vectors
//!   dimension: u64  = vector dimension (typically 10000)
//!   seed: u64       = random seed used for generation
//!   reserved: [u8; 32] = zeros (future use)
//!
//! Body (repeated `count` times):
//!   pos_len: u32
//!   pos_indices: [u32; pos_len]
//!   neg_len: u32
//!   neg_indices: [u32; neg_len]
//! ```

use embeddenator::{SparseVec, DIM};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Magic bytes identifying the dataset format.
const MAGIC: &[u8; 8] = b"EMBR_DST";

/// Current format version.
const FORMAT_VERSION: u32 = 1;

/// Header size in bytes.
const HEADER_SIZE: usize = 8 + 4 + 8 + 8 + 8 + 32; // magic + version + count + dim + seed + reserved

/// Dataset metadata from the header.
#[derive(Debug, Clone)]
pub struct DatasetMeta {
    pub count: u64,
    pub dimension: u64,
    pub seed: u64,
}

/// Configuration for dataset generation.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Number of vectors to generate.
    pub count: u64,
    /// Vector dimension (default: DIM = 10000).
    pub dimension: usize,
    /// Random seed for deterministic generation.
    pub seed: u64,
    /// Target sparsity: number of +1 and -1 indices each (~1% of dimension).
    pub sparsity: usize,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            count: 10_000,
            dimension: DIM,
            seed: 42,
            sparsity: DIM / 100, // ~1% density for each sign
        }
    }
}

/// Generate a single deterministic SparseVec given an RNG.
fn generate_sparse_vec(rng: &mut ChaCha8Rng, dimension: usize, sparsity: usize) -> SparseVec {
    let mut indices: Vec<usize> = (0..dimension).collect();
    indices.shuffle(rng);

    let mut pos: Vec<usize> = indices[..sparsity].to_vec();
    let mut neg: Vec<usize> = indices[sparsity..sparsity * 2].to_vec();

    pos.sort_unstable();
    neg.sort_unstable();

    SparseVec { pos, neg }
}

fn per_vector_seed(master_seed: u64, index: usize) -> u64 {
    master_seed
        .wrapping_add(index as u64)
        .wrapping_mul(0x517cc1b727220a95)
}

fn write_header<W: Write>(
    writer: &mut W,
    count: u64,
    dimension: usize,
    seed: u64,
) -> io::Result<()> {
    writer.write_all(MAGIC)?;
    writer.write_all(&FORMAT_VERSION.to_le_bytes())?;
    writer.write_all(&count.to_le_bytes())?;
    writer.write_all(&(dimension as u64).to_le_bytes())?;
    writer.write_all(&seed.to_le_bytes())?;
    writer.write_all(&[0u8; 32])?; // reserved
    Ok(())
}

fn write_vector<W: Write>(writer: &mut W, vec: &SparseVec) -> io::Result<()> {
    writer.write_all(&(vec.pos.len() as u32).to_le_bytes())?;
    for &idx in &vec.pos {
        writer.write_all(&(idx as u32).to_le_bytes())?;
    }

    writer.write_all(&(vec.neg.len() as u32).to_le_bytes())?;
    for &idx in &vec.neg {
        writer.write_all(&(idx as u32).to_le_bytes())?;
    }

    Ok(())
}

/// Generate a dataset of deterministic SparseVec vectors.
///
/// Uses parallel generation with per-thread RNGs derived from the master seed
/// for reproducibility and performance.
pub fn generate_dataset(config: &GenerateConfig) -> Vec<SparseVec> {
    let count = config.count as usize;
    let dimension = config.dimension;
    let sparsity = config.sparsity;
    let seed = config.seed;

    // For reproducibility, we generate sequential indices and use index-derived seeds
    (0..count)
        .into_par_iter()
        .map(|i| {
            // Derive per-vector seed from master seed + index for determinism
            let vec_seed = per_vector_seed(seed, i);
            let mut rng = ChaCha8Rng::seed_from_u64(vec_seed);
            generate_sparse_vec(&mut rng, dimension, sparsity)
        })
        .collect()
}

/// Write a dataset directly to disk without materializing all vectors in memory.
///
/// This is the preferred generator for very large counts (100K/1M+), because peak
/// memory is bounded by `batch_size`.
pub fn write_dataset_streaming<P: AsRef<Path>>(
    path: P,
    config: &GenerateConfig,
    batch_size: usize,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(64 * 1024, file);

    write_header(&mut writer, config.count, config.dimension, config.seed)?;

    let count = config.count as usize;
    let dimension = config.dimension;
    let sparsity = config.sparsity;
    let seed = config.seed;

    let batch_size = batch_size.max(1);
    let mut start = 0usize;
    while start < count {
        let end = (start + batch_size).min(count);

        // Range is an IndexedParallelIterator; collect preserves order.
        let batch: Vec<SparseVec> = (start..end)
            .into_par_iter()
            .map(|i| {
                let vec_seed = per_vector_seed(seed, i);
                let mut rng = ChaCha8Rng::seed_from_u64(vec_seed);
                generate_sparse_vec(&mut rng, dimension, sparsity)
            })
            .collect();

        for v in &batch {
            write_vector(&mut writer, v)?;
        }

        start = end;
    }

    writer.flush()?;
    Ok(())
}

/// Write a dataset to a binary file.
pub fn write_dataset<P: AsRef<Path>>(
    path: P,
    vectors: &[SparseVec],
    config: &GenerateConfig,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(64 * 1024, file);

    write_header(
        &mut writer,
        vectors.len() as u64,
        config.dimension,
        config.seed,
    )?;

    for vec in vectors {
        write_vector(&mut writer, vec)?;
    }

    writer.flush()?;
    Ok(())
}

/// Read dataset metadata from a file header.
pub fn read_dataset_meta<P: AsRef<Path>>(path: P) -> io::Result<DatasetMeta> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid magic bytes: expected {:?}, got {:?}", MAGIC, magic),
        ));
    }

    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    reader.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported format version: {}", version),
        ));
    }

    reader.read_exact(&mut buf8)?;
    let count = u64::from_le_bytes(buf8);

    reader.read_exact(&mut buf8)?;
    let dimension = u64::from_le_bytes(buf8);

    reader.read_exact(&mut buf8)?;
    let seed = u64::from_le_bytes(buf8);

    Ok(DatasetMeta {
        count,
        dimension,
        seed,
    })
}

/// Load a dataset from a binary file.
pub fn load_dataset<P: AsRef<Path>>(path: P) -> io::Result<(DatasetMeta, Vec<SparseVec>)> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(64 * 1024, file);

    // Read header
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid magic bytes: expected {:?}, got {:?}", MAGIC, magic),
        ));
    }

    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    reader.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported format version: {}", version),
        ));
    }

    reader.read_exact(&mut buf8)?;
    let count = u64::from_le_bytes(buf8);

    reader.read_exact(&mut buf8)?;
    let dimension = u64::from_le_bytes(buf8);

    reader.read_exact(&mut buf8)?;
    let seed = u64::from_le_bytes(buf8);

    // Skip reserved bytes
    let mut reserved = [0u8; 32];
    reader.read_exact(&mut reserved)?;

    let meta = DatasetMeta {
        count,
        dimension,
        seed,
    };

    // Read vectors
    let mut vectors = Vec::with_capacity(count as usize);
    for _ in 0..count {
        // Read pos indices
        reader.read_exact(&mut buf4)?;
        let pos_len = u32::from_le_bytes(buf4) as usize;
        let mut pos = Vec::with_capacity(pos_len);
        for _ in 0..pos_len {
            reader.read_exact(&mut buf4)?;
            pos.push(u32::from_le_bytes(buf4) as usize);
        }

        // Read neg indices
        reader.read_exact(&mut buf4)?;
        let neg_len = u32::from_le_bytes(buf4) as usize;
        let mut neg = Vec::with_capacity(neg_len);
        for _ in 0..neg_len {
            reader.read_exact(&mut buf4)?;
            neg.push(u32::from_le_bytes(buf4) as usize);
        }

        vectors.push(SparseVec { pos, neg });
    }

    Ok((meta, vectors))
}

/// Memory-mapped dataset loader for large datasets.
///
/// This allows iterating over vectors without loading the entire dataset into memory.
pub struct DatasetReader {
    meta: DatasetMeta,
    reader: BufReader<File>,
    current_index: u64,
}

impl DatasetReader {
    /// Open a dataset file for streaming reads.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(&path)?;
        let mut reader = BufReader::with_capacity(64 * 1024, file);

        // Read and validate header
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid magic bytes",
            ));
        }

        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported format version: {}", version),
            ));
        }

        reader.read_exact(&mut buf8)?;
        let count = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let dimension = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let seed = u64::from_le_bytes(buf8);

        let mut reserved = [0u8; 32];
        reader.read_exact(&mut reserved)?;

        let meta = DatasetMeta {
            count,
            dimension,
            seed,
        };

        Ok(Self {
            meta,
            reader,
            current_index: 0,
        })
    }

    /// Get dataset metadata.
    pub fn meta(&self) -> &DatasetMeta {
        &self.meta
    }

    /// Read the next vector from the dataset.
    pub fn next_vector(&mut self) -> io::Result<Option<SparseVec>> {
        if self.current_index >= self.meta.count {
            return Ok(None);
        }

        let mut buf4 = [0u8; 4];

        // Read pos indices
        self.reader.read_exact(&mut buf4)?;
        let pos_len = u32::from_le_bytes(buf4) as usize;
        let mut pos = Vec::with_capacity(pos_len);
        for _ in 0..pos_len {
            self.reader.read_exact(&mut buf4)?;
            pos.push(u32::from_le_bytes(buf4) as usize);
        }

        // Read neg indices
        self.reader.read_exact(&mut buf4)?;
        let neg_len = u32::from_le_bytes(buf4) as usize;
        let mut neg = Vec::with_capacity(neg_len);
        for _ in 0..neg_len {
            self.reader.read_exact(&mut buf4)?;
            neg.push(u32::from_le_bytes(buf4) as usize);
        }

        self.current_index += 1;
        Ok(Some(SparseVec { pos, neg }))
    }

    /// Read multiple vectors at once for batch processing.
    pub fn read_batch(&mut self, batch_size: usize) -> io::Result<Vec<SparseVec>> {
        let remaining = (self.meta.count - self.current_index) as usize;
        let to_read = batch_size.min(remaining);
        let mut batch = Vec::with_capacity(to_read);

        for _ in 0..to_read {
            if let Some(vec) = self.next_vector()? {
                batch.push(vec);
            }
        }

        Ok(batch)
    }

    /// Reset reader to the beginning of the dataset.
    pub fn reset(&mut self) -> io::Result<()> {
        use std::io::Seek;
        self.reader
            .seek(std::io::SeekFrom::Start(HEADER_SIZE as u64))?;
        self.current_index = 0;
        Ok(())
    }
}

impl Iterator for DatasetReader {
    type Item = io::Result<SparseVec>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_vector() {
            Ok(Some(vec)) => Some(Ok(vec)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Compute expected file size for a dataset.
pub fn expected_file_size(count: u64, sparsity: usize) -> u64 {
    // Header: 68 bytes
    // Per vector: 4 (pos_len) + sparsity*4 (pos) + 4 (neg_len) + sparsity*4 (neg)
    let per_vector = 4 + (sparsity * 4) + 4 + (sparsity * 4);
    HEADER_SIZE as u64 + count * per_vector as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_generate_deterministic() {
        let config = GenerateConfig {
            count: 100,
            seed: 42,
            ..Default::default()
        };

        let vecs1 = generate_dataset(&config);
        let vecs2 = generate_dataset(&config);

        assert_eq!(vecs1.len(), vecs2.len());
        for (v1, v2) in vecs1.iter().zip(vecs2.iter()) {
            assert_eq!(v1.pos, v2.pos);
            assert_eq!(v1.neg, v2.neg);
        }
    }

    #[test]
    fn test_write_and_read() {
        let config = GenerateConfig {
            count: 50,
            seed: 123,
            ..Default::default()
        };

        let vectors = generate_dataset(&config);
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.embr");

        write_dataset(&path, &vectors, &config).unwrap();
        let (meta, loaded) = load_dataset(&path).unwrap();

        assert_eq!(meta.count, 50);
        assert_eq!(meta.seed, 123);
        assert_eq!(loaded.len(), vectors.len());

        for (orig, loaded) in vectors.iter().zip(loaded.iter()) {
            assert_eq!(orig.pos, loaded.pos);
            assert_eq!(orig.neg, loaded.neg);
        }
    }

    #[test]
    fn test_streaming_reader() {
        let config = GenerateConfig {
            count: 25,
            seed: 999,
            ..Default::default()
        };

        let vectors = generate_dataset(&config);
        let dir = tempdir().unwrap();
        let path = dir.path().join("stream.embr");

        write_dataset(&path, &vectors, &config).unwrap();

        let mut reader = DatasetReader::open(&path).unwrap();
        assert_eq!(reader.meta().count, 25);

        let mut count = 0;
        for result in &mut reader {
            let vec = result.unwrap();
            assert_eq!(vec.pos, vectors[count].pos);
            assert_eq!(vec.neg, vectors[count].neg);
            count += 1;
        }
        assert_eq!(count, 25);
    }

    #[test]
    fn test_batch_reading() {
        let config = GenerateConfig {
            count: 100,
            seed: 456,
            ..Default::default()
        };

        let vectors = generate_dataset(&config);
        let dir = tempdir().unwrap();
        let path = dir.path().join("batch.embr");

        write_dataset(&path, &vectors, &config).unwrap();

        let mut reader = DatasetReader::open(&path).unwrap();
        let batch = reader.read_batch(30).unwrap();
        assert_eq!(batch.len(), 30);

        for (i, vec) in batch.iter().enumerate() {
            assert_eq!(vec.pos, vectors[i].pos);
        }
    }

    #[test]
    fn test_streaming_writer_matches_in_memory() {
        let config = GenerateConfig {
            count: 250,
            seed: 2026,
            ..Default::default()
        };

        let vectors = generate_dataset(&config);
        let dir = tempdir().unwrap();
        let path_mem = dir.path().join("mem.embr");
        let path_stream = dir.path().join("stream.embr");

        write_dataset(&path_mem, &vectors, &config).unwrap();
        write_dataset_streaming(&path_stream, &config, 64).unwrap();

        let (_m1, v1) = load_dataset(&path_mem).unwrap();
        let (_m2, v2) = load_dataset(&path_stream).unwrap();

        assert_eq!(v1.len(), v2.len());
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert_eq!(a.pos, b.pos);
            assert_eq!(a.neg, b.neg);
        }
    }
}
