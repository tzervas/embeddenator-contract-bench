use std::hint::black_box;
use std::time::Instant;

use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Copy, Debug)]
pub enum Profile {
    Quick,
    Full,
}

impl Profile {
    pub fn as_str(&self) -> &'static str {
        match self {
            Profile::Quick => "quick",
            Profile::Full => "full",
        }
    }
}

#[derive(Clone, Debug)]
pub struct BenchConfig {
    pub profile: Profile,
    pub seed: u64,
}

impl BenchConfig {
    pub fn rng(&self) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(self.seed)
    }

    pub fn warmup_iters(&self) -> u64 {
        match self.profile {
            Profile::Quick => 32,
            Profile::Full => 200,
        }
    }

    pub fn iters(&self) -> u64 {
        match self.profile {
            Profile::Quick => 300,
            Profile::Full => 3_000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Measured {
    pub iters: u64,
    pub warmup_iters: u64,
    pub total_ns: u128,
    pub ns_per_iter: f64,
}

pub fn measure_fn<T>(iters: u64, warmup_iters: u64, mut f: impl FnMut() -> T) -> Measured {
    for _ in 0..warmup_iters {
        black_box(f());
    }

    let start = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    let elapsed = start.elapsed();

    let total_ns = elapsed.as_nanos();
    let denom = iters.max(1) as f64;
    let ns_per_iter = (total_ns as f64) / denom;

    Measured {
        iters,
        warmup_iters,
        total_ns,
        ns_per_iter,
    }
}
