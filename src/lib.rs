
use core::fmt::Debug;
use core::ops::AddAssign;

extern crate num_traits;
use num_traits::{float::Float, identities::Zero, identities::One, cast::FromPrimitive};

#[macro_use]
extern crate serde;
use serde::{Serialize, Deserialize};

/// Stats is an object that calculates continuous min/max/mean/deviation for tracking of time varying statistics
/// 
/// 
/// See: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm for the algorithm

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stats<T: Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug> {
    /// Minimum value
    pub min:     T,
    /// Maximum value
    pub max:     T,
    /// Mean of sample set
    pub mean:    T,
    /// Standard deviation of sample
    pub std_dev: T,

    /// Number of values collected
    count: usize,

    /// Internal mean squared for algo
    mean2:   T,
}

impl <T> Stats<T> 
where
    T: Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug,
{   
    /// Create a new rolling-stats object
    pub fn new() -> Stats<T> {
        Stats{count: 0, min: T::zero(), max: T::zero(), mean: T::zero(), std_dev: T::zero(), mean2: T::zero()}
    }

    /// Update the rolling-stats object
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
}


#[cfg(test)]
mod tests {
    use super::*;

    extern crate float_cmp;
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
