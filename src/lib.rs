use core::fmt::Debug;
use core::ops::AddAssign;

use num_traits::{cast::FromPrimitive, float::Float, identities::One, identities::Zero};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Stats object calculates continuous min/max/mean/deviation for tracking of time varying statistics.
///
/// See: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm for
/// Details of the underlying algorithm.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct Stats<T: Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug> {
    /// Minimum value
    pub min: T,
    /// Maximum value
    pub max: T,
    /// Mean of sample set
    pub mean: T,
    /// Standard deviation of sample
    pub std_dev: T,

    /// Number of values collected
    #[cfg_attr(feature = "serde", serde(skip))]
    pub count: usize,

    /// Internal mean squared for algo
    #[cfg_attr(feature = "serde", serde(skip))]
    mean2: T,
}

use core::fmt;

impl<T> fmt::Display for Stats<T>
where
    T: fmt::Display + Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision().unwrap_or(2);

        write!(f, "(avg: {:.precision$}, std_dev: {:.precision$}, min: {:.precision$}, max: {:.precision$}, count: {})", self.mean, self.std_dev, self.min, self.max, self.count, precision=precision)
    }
}

impl<T> Stats<T>
where
    T: Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug,
{
    /// Create a new stats object
    pub fn new() -> Stats<T> {
        Stats {
            count: 0,
            min: T::zero(),
            max: T::zero(),
            mean: T::zero(),
            std_dev: T::zero(),
            mean2: T::zero(),
        }
    }

    /// Update the stats object
    pub fn update(&mut self, value: T) {
        // Track min and max
        if value > self.max || self.count == 0 {
            self.max = value;
        }
        if value < self.min || self.count == 0 {
            self.min = value;
        }

        // Increment counter
        self.count += 1;
        let count = T::from_usize(self.count).unwrap();

        // Calculate mean
        let delta: T = value - self.mean;
        self.mean += delta / count;

        // Mean2 used internally for standard deviation calculation
        let delta2: T = value - self.mean;
        self.mean2 += delta * delta2;

        // Calculate standard deviation
        if self.count > 1 {
            self.std_dev = (self.mean2 / (count - T::one())).sqrt();
        }
    }

    /// Merge a set of stats objects for analysis
    /// This performs a weighted averaging across the provided stats object, the output
    /// object should not be updated further.
    pub fn merge<S: Iterator<Item = Stats<T>>>(stats: S) -> Stats<T> {
        let mut merged = Stats::new();

        for s in stats {
            // Track min and max
            if s.max > merged.max || merged.count == 0 {
                merged.max = s.max;
            }
            if s.min < merged.min || merged.count == 0 {
                merged.min = s.min;
            }

            let merged_count = T::from_usize(merged.count).unwrap();
            let s_count = T::from_usize(s.count).unwrap();

            if merged.count > 0 {
                merged.mean =
                    (merged.mean * merged_count + s.mean * s_count) / (merged_count + s_count);
                merged.std_dev = (merged.std_dev * merged_count + s.std_dev * s_count)
                    / (merged_count + s_count);
                merged.count += s.count;
            } else {
                merged.mean = s.mean;
                merged.std_dev = s.std_dev;
                merged.count = s.count;
            }
        }

        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use float_cmp::ApproxEqUlps;

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
}
