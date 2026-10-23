//! Summary statistics of timing samples.

/// Mean, percentiles and extremes of a set of samples, in the unit of the samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    pub mean: f64,
    pub p50: f64,
    pub p99: f64,
    pub min: f64,
    pub max: f64,
}

impl Summary {
    /// Summarises `samples`, or returns `None` for an empty set.
    pub fn of(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        Some(Self {
            mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
            p50: percentile(&sorted, 50.0),
            p99: percentile(&sorted, 99.0),
            min: sorted[0],
            max: sorted[sorted.len() - 1],
        })
    }
}

/// The nearest-rank percentile of an ascending, non-empty slice: the smallest
/// sample that at least `percent` of the samples do not exceed.
fn percentile(sorted: &[f64], percent: f64) -> f64 {
    let rank = (percent / 100.0 * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_has_no_summary() {
        assert_eq!(Summary::of(&[]), None);
    }

    #[test]
    fn single_sample() {
        let s = Summary::of(&[2.5]).unwrap();
        assert_eq!(
            (s.mean, s.p50, s.p99, s.min, s.max),
            (2.5, 2.5, 2.5, 2.5, 2.5)
        );
    }

    #[test]
    fn nearest_rank_percentiles() {
        // 1 to 100: the median is 50 and the 99th percentile 99.
        let samples: Vec<f64> = (1..=100).rev().map(f64::from).collect();
        let s = Summary::of(&samples).unwrap();
        assert_eq!(s.mean, 50.5);
        assert_eq!(s.p50, 50.0);
        assert_eq!(s.p99, 99.0);
        assert_eq!((s.min, s.max), (1.0, 100.0));
    }

    #[test]
    fn a_slow_outlier_shows_in_p99_but_not_in_the_median() {
        let mut samples = vec![1.0; 99];
        samples.push(50.0);
        let s = Summary::of(&samples).unwrap();
        assert_eq!(s.p50, 1.0);
        assert_eq!(s.p99, 1.0); // 99 of 100 samples are at most 1.0
        assert_eq!(s.max, 50.0);

        samples.push(60.0);
        let s = Summary::of(&samples).unwrap();
        assert_eq!(s.p99, 50.0);
    }
}
