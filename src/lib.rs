#![no_std]

extern crate alloc;

use core::{
    fmt::{self, Debug},
    ops::AddAssign,
};

use num_traits::{cast::FromPrimitive, float::Float, identities::One, identities::Zero};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A statistics object that continuously calculates min, max, mean, and deviation for tracking time-varying statistics.
/// Utilizes Welford's Online algorithm. More details on the algorithm can be found at:
/// "https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm"
///
///
/// # Example
///
/// ```
/// use rolling_stats::Stats;
/// use rand_distr::{Distribution, Normal};
/// use rand::SeedableRng;
///
/// type T = f64;
///
/// const MEAN: T = 0.0;
/// const STD_DEV: T = 1.0;
/// const NUM_SAMPLES: usize = 10_000;
/// const SEED: u64 = 42;
///
/// let mut stats: Stats<T> = Stats::new();
/// let mut rng = rand::rngs::StdRng::seed_from_u64(SEED); // Seed the RNG for reproducibility
/// let normal = Normal::<T>::new(MEAN, STD_DEV).unwrap();
///
/// // Generate random data
/// let random_data: Vec<T> = (0..NUM_SAMPLES).map(|_x| normal.sample(&mut rng)).collect();
///
/// // Update the stats one by one
/// random_data.iter().for_each(|v| stats.update(*v));
///
/// // Print the stats
/// println!("{}", stats);
/// // Output: (avg: 0.00, std_dev: 1.00, min: -3.53, max: 4.11, count: 10000)
///
/// ```
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default)]
pub struct Stats<T: Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug> {
    /// The smallest value seen so far.
    pub min: T,

    /// The largest value seen so far.
    pub max: T,

    /// The calculated mean (average) of all the values seen so far.
    pub mean: T,

    /// The calculated standard deviation of all the values seen so far.
    pub std_dev: T,

    /// The count of the total values seen.
    pub count: usize,

    /// The square of the mean value. This is an internal value used in the calculation of the standard deviation.
    #[cfg_attr(feature = "serde", serde(skip))]
    mean2: T,
}

/// Implementing the Display trait for the Stats struct to present the statistics in a readable format.
impl<T> fmt::Display for Stats<T>
where
    T: fmt::Display + Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug,
{
    /// Formats the output of the statistics.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision().unwrap_or(2);

        write!(f, "(avg: {:.precision$}, std_dev: {:.precision$}, min: {:.precision$}, max: {:.precision$}, count: {})", self.mean, self.std_dev, self.min, self.max, self.count, precision=precision)
    }
}

impl<T> Stats<T>
where
    T: Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug,
{
    /// Creates a new stats object with all values set to their initial states.
    pub fn new() -> Stats<T> {
        Stats {
            count: 0,
            min: T::infinity(),
            max: T::neg_infinity(),
            mean: T::zero(),
            std_dev: T::zero(),
            mean2: T::zero(),
        }
    }

    /// Updates the stats object with a new value. The statistics are recalculated using the new value.
    pub fn update(&mut self, value: T) {
        // Track min and max
        if value > self.max {
            self.max = value;
        }
        if value < self.min {
            self.min = value;
        }

        // Increment counter
        self.count += 1;
        let count = T::from(self.count).unwrap();

        // Calculate mean
        let delta = value - self.mean;
        self.mean += delta / count;

        // Mean2 used internally for standard deviation calculation
        let delta2 = value - self.mean;
        self.mean2 += delta * delta2;

        // Calculate standard deviation
        if self.count > 1 {
            self.std_dev = (self.mean2 / (count - T::one())).sqrt();
        }
    }

    /// Merges another stats object into new one. This is done by combining the statistics of the two objects
    /// in accordance with the formula provided at:
    /// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    ///
    /// This is useful for combining statistics from multiple threads or processes.
    ///
    /// # Example
    ///
    /// ```
    /// use rolling_stats::Stats;
    /// use rand_distr::{Distribution, Normal};
    /// use rand::SeedableRng;
    /// use rayon::prelude::*;
    ///
    /// type T = f64;
    ///
    /// const MEAN: T = 0.0;
    /// const STD_DEV: T = 1.0;
    /// const NUM_SAMPLES: usize = 500_000;
    /// const SEED: u64 = 42;
    /// const CHUNK_SIZE: usize = 1000;
    ///
    /// let mut stats: Stats<T> = Stats::new();
    /// let mut rng = rand::rngs::StdRng::seed_from_u64(SEED); // Seed the RNG for reproducibility
    /// let normal = Normal::<T>::new(MEAN, STD_DEV).unwrap();
    ///
    /// // Generate random data
    /// let random_data: Vec<T> = (0..NUM_SAMPLES).map(|_x| normal.sample(&mut rng)).collect();
    ///
    /// // Update the stats in parallel. New stats objects are created for each chunk of data.
    /// let stats: Vec<Stats<T>> = random_data
    ///     .par_chunks(CHUNK_SIZE) // Multi-threaded parallelization via Rayon
    ///     .map(|chunk| {
    ///             let mut s: Stats<T> = Stats::new();
    ///             chunk.iter().for_each(|v| s.update(*v));
    ///             s
    ///      })
    ///     .collect();
    ///
    /// // Check if there's more than one stat object
    /// assert!(stats.len() > 1);
    ///
    /// // Accumulate the stats using the reduce method. The last stats object is returned.
    /// let merged_stats = stats.into_iter().reduce(|acc, s| acc.merge(&s)).unwrap();
    ///
    /// // Print the stats
    /// println!("{}", merged_stats);
    ///
    /// // Output: (avg: -0.00, std_dev: 1.00, min: -4.53, max: 4.57, count: 500000)
    ///```
    pub fn merge(&self, other: &Self) -> Self {
        let mut merged = Stats::<T>::new();

        // If both stats objects are empty, return an empty stats object
        if self.count + other.count == 0 {
            return merged;
        }

        // If one of the stats objects is empty, return the other one
        if self.count == 0 {
            return other.clone();
        } else if other.count == 0 {
            return self.clone();
        }

        merged.max = if other.max > self.max {
            other.max
        } else {
            self.max
        };

        merged.min = if other.min < self.min {
            other.min
        } else {
            self.min
        };

        merged.count = self.count + other.count;

        // Convert to T to avoid overflow
        let merged_count = T::from(merged.count).unwrap();
        let self_count = T::from(self.count).unwrap();
        let other_count = T::from(other.count).unwrap();

        let delta = other.mean - self.mean;

        merged.mean = (self.mean * self_count + other.mean * other_count) / merged_count;

        merged.mean2 =
            self.mean2 + other.mean2 + delta * delta * self_count * other_count / merged_count;

        merged.std_dev = (merged.mean2 / (merged_count - T::one())).sqrt();

        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::vec;
    use alloc::vec::Vec;

    use float_cmp::{ApproxEq, ApproxEqUlps};
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};
    use rayon::prelude::*;

    type T = f64;

    #[test]
    fn it_works() {
        let mut s: Stats<f32> = Stats::new();

        let vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        for v in &vals {
            s.update(*v);
        }

        assert_eq!(s.count, vals.len());

        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 5.0);

        assert!(s.mean.approx_eq_ulps(&3.0, 2));
        assert!(s.std_dev.approx_eq_ulps(&1.5811388, 2));
    }

    /// Calculate the mean of a vector of values
    fn calc_mean(vals: &Vec<T>) -> T {
        let sum = vals.iter().fold(T::zero(), |acc, x| acc + *x);

        sum / T::from_usize(vals.len()).unwrap()
    }

    /// Calculate the standard deviation of a vector of values
    fn calc_std_dev(vals: &Vec<T>) -> T {
        let mean = calc_mean(vals);
        let std_dev = (vals
            .iter()
            .fold(T::zero(), |acc, x| acc + (*x - mean).powi(2))
            / T::from_usize(vals.len() - 1).unwrap())
        .sqrt();

        std_dev
    }

    /// Get the maximum value in a vector of values
    fn get_max(vals: &Vec<T>) -> T {
        let mut max = T::min_value();
        for v in vals {
            if *v > max {
                max = *v;
            }
        }
        max
    }

    /// Get the minimum value in a vector of values
    fn get_min(vals: &Vec<T>) -> T {
        let mut min = T::max_value();
        for v in vals {
            if *v < min {
                min = *v;
            }
        }
        min
    }

    #[test]
    fn stats_for_large_random_data() {
        // Define some constants
        const MEAN: T = 2.0;
        const STD_DEV: T = 3.0;
        const SEED: u64 = 42;
        const NUM_SAMPLES: usize = 10_000;

        let mut s: Stats<T> = Stats::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let normal = Normal::<T>::new(MEAN, STD_DEV).unwrap();

        // Generate some random data
        let random_data: Vec<T> = (0..NUM_SAMPLES).map(|_x| normal.sample(&mut rng)).collect();

        // Update the stats
        random_data.iter().for_each(|v| s.update(*v));

        // Calculate the mean using sum/count method
        let mean = calc_mean(&random_data);

        // Check the mean value against the stats' mean value
        assert!(s.mean.approx_eq(mean, (1.0e-13, 2)));

        // Calculate the standard deviation
        let std_dev = calc_std_dev(&random_data);

        // Check the standard deviation against the stats' standard deviation
        assert!(s.std_dev.approx_eq(std_dev, (1.0e-13, 2)));

        // Check the count
        assert_eq!(s.count, random_data.len());

        // Find the max and min values
        let max = get_max(&random_data);
        let min = get_min(&random_data);

        // Check the max and min values
        assert_eq!(s.max, max);
        assert_eq!(s.min, min);
    }

    #[test]
    fn stats_merge() {
        // Define some constants
        const MEAN: T = 2.0;
        const STD_DEV: T = 3.0;
        const SEED: u64 = 42;
        const NUM_SAMPLES: usize = 10_000;

        let mut s: Stats<T> = Stats::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let normal = Normal::<T>::new(MEAN, STD_DEV).unwrap();

        // Generate some random data
        let random_data: Vec<T> = (0..NUM_SAMPLES).map(|_x| normal.sample(&mut rng)).collect();

        // Update the stats
        random_data.iter().for_each(|v| s.update(*v));

        // Calculate the stats using the aggregate method instead of the rolling method
        let mean = calc_mean(&random_data);
        let std_dev = calc_std_dev(&random_data);
        let max = get_max(&random_data);
        let min = get_min(&random_data);

        let chunks_size = 1000;

        let stats: Vec<Stats<T>> = random_data
            .chunks(chunks_size)
            .map(|chunk| {
                let mut s: Stats<T> = Stats::new();
                chunk.iter().for_each(|v| s.update(*v));
                s
            })
            .collect();

        assert_eq!(stats.len(), NUM_SAMPLES / chunks_size);

        // Accumulate the stats
        let merged_stats = stats.into_iter().reduce(|acc, s| acc.merge(&s)).unwrap();

        // Check the stats against the aggregate stats (using sum/count method)
        assert!(merged_stats.mean.approx_eq(mean, (1.0e-13, 2)));
        assert!(merged_stats.std_dev.approx_eq(std_dev, (1.0e-13, 2)));
        assert_eq!(merged_stats.max, max);
        assert_eq!(merged_stats.min, min);
        assert_eq!(merged_stats.count, NUM_SAMPLES);

        // Check the stats against the merged stats object
        assert!(merged_stats.mean.approx_eq(s.mean, (1.0e-13, 2)));
        assert!(merged_stats.std_dev.approx_eq(s.std_dev, (1.0e-13, 2)));
        assert_eq!(merged_stats.max, s.max);
        assert_eq!(merged_stats.min, s.min);
        assert_eq!(merged_stats.count, s.count);

        // Check edge cases

        // Check merging with an empty stats object
        let empty_stats: Stats<T> = Stats::new();
        let merged_stats = s.merge(&empty_stats);
        assert_eq!(merged_stats.count, s.count);

        // Check merging an empty stats object with a non-empty stats object
        let empty_stats: Stats<T> = Stats::new();
        let merged_stats = empty_stats.merge(&s);
        assert_eq!(merged_stats.count, s.count);

        // Check merging two empty stats objects
        let empty_stats_1: Stats<T> = Stats::new();
        let empty_stats_2: Stats<T> = Stats::new();

        let merged_stats = empty_stats_1.merge(&empty_stats_2);
        assert_eq!(merged_stats.count, 0);
    }

    #[test]
    fn stats_merge_parallel() {
        // Define some constants
        const MEAN: T = 2.0;
        const STD_DEV: T = 3.0;
        const SEED: u64 = 42;
        const NUM_SAMPLES: usize = 500_000;

        let mut s: Stats<T> = Stats::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let normal = Normal::<T>::new(MEAN, STD_DEV).unwrap();

        // Generate some random data
        let random_data: Vec<T> = (0..NUM_SAMPLES).map(|_x| normal.sample(&mut rng)).collect();

        // Update the stats
        random_data.iter().for_each(|v| s.update(*v));

        // Calculate the stats using the aggregate method instead of the rolling method
        let mean = calc_mean(&random_data);
        let std_dev = calc_std_dev(&random_data);
        let max = get_max(&random_data);
        let min = get_min(&random_data);

        let chunks_size = 1000;

        let stats: Vec<Stats<T>> = random_data
            .par_chunks(chunks_size) // <--- Parallelization by Rayon
            .map(|chunk| {
                let mut s: Stats<T> = Stats::new();
                chunk.iter().for_each(|v| s.update(*v));
                s
            })
            .collect();

        // There should be more than one stat
        assert!(stats.len() >= NUM_SAMPLES / chunks_size);

        // Accumulate the stats
        let merged_stats = stats.into_iter().reduce(|acc, s| acc.merge(&s)).unwrap();

        // Check the stats against the aggregate stats (using sum/count method)
        assert!(merged_stats.mean.approx_eq(mean, (1.0e-13, 2)));
        assert!(merged_stats.std_dev.approx_eq(std_dev, (1.0e-13, 2)));
        assert_eq!(merged_stats.max, max);
        assert_eq!(merged_stats.min, min);
        assert_eq!(merged_stats.count, NUM_SAMPLES);

        // Check the stats against the merged stats object
        assert!(merged_stats.mean.approx_eq(s.mean, (1.0e-13, 2)));
        assert!(merged_stats.std_dev.approx_eq(s.std_dev, (1.0e-13, 2)));
        assert_eq!(merged_stats.max, s.max);
        assert_eq!(merged_stats.min, s.min);
        assert_eq!(merged_stats.count, s.count);
    }
}
