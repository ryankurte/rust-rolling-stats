# Rust-Rolling-Stats

The `rolling-stats` library offers rolling statistics calculations (minimum, maximum, mean, standard
deviation) over arbitrary floating point numbers. It uses Welford's Online Algorithm for these
computations.

For more information on the algorithm, visit
[Algorithms for calculating variance](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
on Wikipedia.

## Status

[![GitHub tag](https://img.shields.io/github/tag/ryankurte/rust-rolling-stats.svg)](https://github.com/ryankurte/rust-rolling-stats)
[![Build Status](https://travis-ci.com/ryankurte/rust-rolling-stats.svg?branch=master)](https://travis-ci.com/ryankurte/rust-rolling-stats)
[![Crates.io](https://img.shields.io/crates/v/rolling-stats.svg)](https://crates.io/crates/rolling-stats)
[![Docs.rs](https://docs.rs/rolling-stats/badge.svg)](https://docs.rs/rolling-stats)

## Usage

### Single Thread Example

Below is an example of using `rust-rolling-stats` in a single-threaded context:

```rust
use rolling_stats::Stats;
use rand_distr::{Distribution, Normal};
use rand::SeedableRng;

type T = f64;

const MEAN: T = 0.0;
const STD_DEV: T = 1.0;
const NUM_SAMPLES: usize = 10_000;
const SEED: u64 = 42;

let mut stats: Stats<T> = Stats::new();
let mut rng = rand::rngs::StdRng::seed_from_u64(SEED); // Seed the RNG for reproducibility
let normal = Normal::<T>::new(MEAN, STD_DEV).unwrap();

// Generate random data
let random_data: Vec<T> = (0..NUM_SAMPLES).map(|_x| normal.sample(&mut rng)).collect();

// Update the stats one by one
random_data.iter().for_each(|v| stats.update(*v));

// Print the stats
println!("{}", stats);
// Output: (avg: 0.00, std_dev: 1.00, min: -3.53, max: 4.11, count: 10000)
```

### Multi Thread Example

This example showcases the usage of `rust-rolling-stats` in a multi-threaded context with the help
of the `rayon` crate:

```rust
use rolling_stats::Stats;
use rand_distr::{Distribution, Normal};
use rand::SeedableRng;
use rayon::prelude::*;

type T = f64;

const MEAN: T = 0.0;
const STD_DEV: T = 1.0;
const NUM_SAMPLES: usize = 500_000;
const SEED: u64 = 42;
const CHUNK_SIZE: usize = 1000;

let mut stats: Stats<T> = Stats::new();
let mut rng = rand::rngs::StdRng::seed_from_u64(SEED); // Seed the RNG for reproducibility
let normal = Normal::<T>::new(MEAN, STD_DEV).unwrap();

// Generate random data
let random_data: Vec<T> = (0..NUM_SAMPLES).map(|_x| normal.sample(&mut rng)).collect();

// Update the stats in parallel. New stats objects are created for each chunk of data.
let stats: Vec<Stats<T>> = random_data
    .par_chunks(CHUNK_SIZE) // Multi-threaded parallelization via Rayon
    .map(|chunk| {
        let mut s: Stats<T> = Stats::new();
        chunk.iter().for_each(|v| s.update(*v));
        s
    })
    .collect();

// Check if there's more than one stat object
assert!(stats.len() > 1);

// Accumulate the stats using the reduce method
let merged_stats = stats.into_iter().reduce(|acc, s| acc.merge(&s)).unwrap();

// Print the stats
println!("{}", merged_stats);
// Output: (avg: -0.00, std_dev: 1.00, min: -4.53, max: 4.57, count: 500000)
```

## Feature Flags

The following feature flags are available:

- `std`: Enables the `std` crate. Enabled by default.
- `serde`: Enables serialization and deserialization of the `Stats` struct via the `serde` crate.

## `no_std` Compatibility

This crate is `no_std` compatible, simply disable the default features in your `Cargo.toml`:

```toml
[dependencies]
rolling-stats = { version = "0.7.0", default-features = false }
```

## License

The `rolling-stats` library is dual-licensed under the MIT and Apache License 2.0. By opening a pull
request, you are implicitly agreeing to these licensing terms.
