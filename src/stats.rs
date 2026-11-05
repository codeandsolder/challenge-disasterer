use std::{
    collections::{BTreeMap, VecDeque},
    time::Instant,
};

use serde::Serialize;

const HISTORY: usize = 256;

#[derive(Debug, Clone, Serialize)]
pub struct Calibration {
    pub gpu_hashes_per_second: Option<f64>,
    pub cpu_hashes_per_second_per_worker: f64,
    pub gpu_adapter: Option<String>,
}

#[derive(Debug)]
pub struct StatsStore {
    started: Instant,
    samples: BTreeMap<u8, VecDeque<f64>>,
    pub solves_total: u64,
    pub solves_gpu: u64,
    pub solves_cpu: u64,
    pub failures: u64,
}

impl Default for StatsStore {
    fn default() -> Self {
        Self {
            started: Instant::now(),
            samples: BTreeMap::new(),
            solves_total: 0,
            solves_gpu: 0,
            solves_cpu: 0,
            failures: 0,
        }
    }
}

impl StatsStore {
    pub fn record(&mut self, difficulty: u8, backend: &str, solve_ms: f64) {
        self.record_generic(backend);
        // A backend's malformed timing must not poison the rolling median.
        if !solve_ms.is_finite() || solve_ms < 0.0 {
            return;
        }
        let samples = self.samples.entry(difficulty).or_default();
        if samples.len() == HISTORY {
            samples.pop_front();
        }
        samples.push_back(solve_ms);
    }

    pub fn record_generic(&mut self, backend: &str) {
        self.solves_total += 1;
        match backend {
            "gpu" => self.solves_gpu += 1,
            "cpu" => self.solves_cpu += 1,
            _ => {}
        }
    }
    pub const fn record_failure(&mut self) {
        self.failures += 1;
    }

    pub fn snapshot(&self) -> GeneralStats {
        let median_solve_ms_by_difficulty = self
            .samples
            .iter()
            .map(|(&difficulty, samples)| {
                let mut values: Vec<_> = samples.iter().copied().collect();
                values.sort_by(f64::total_cmp);
                let middle = values.len() / 2;
                let median = if values.len().is_multiple_of(2) {
                    // Half-distance avoids overflowing when both samples
                    // are large but individually finite.
                    values[middle - 1] + (values[middle] - values[middle - 1]) / 2.0
                } else {
                    values[middle]
                };
                (difficulty, median)
            })
            .collect();
        GeneralStats {
            uptime_ms: self.started.elapsed().as_millis(),
            solves_total: self.solves_total,
            solves_gpu: self.solves_gpu,
            solves_cpu: self.solves_cpu,
            failures: self.failures,
            median_solve_ms_by_difficulty,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct GeneralStats {
    pub uptime_ms: u128,
    pub solves_total: u64,
    pub solves_gpu: u64,
    pub solves_cpu: u64,
    pub failures: u64,
    pub median_solve_ms_by_difficulty: BTreeMap<u8, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_the_statistical_median_for_odd_and_even_samples() {
        let mut stats = StatsStore::default();
        stats.record(6, "cpu", 10.0);
        assert_eq!(stats.snapshot().median_solve_ms_by_difficulty[&6], 10.0);
        stats.record(6, "gpu", 30.0);
        assert_eq!(stats.snapshot().median_solve_ms_by_difficulty[&6], 20.0);
        stats.record(6, "cpu", 20.0);
        assert_eq!(stats.snapshot().median_solve_ms_by_difficulty[&6], 20.0);
        assert_eq!(stats.solves_total, 3);
        assert_eq!(stats.solves_cpu, 2);
        assert_eq!(stats.solves_gpu, 1);
    }

    #[test]
    fn samples_are_bounded_to_the_most_recent_history() -> anyhow::Result<()> {
        let mut stats = StatsStore::default();
        for value in 1..=HISTORY + 4 {
            stats.record(4, "cpu", f64::from(u32::try_from(value)?));
        }
        assert_eq!(stats.samples[&4].len(), HISTORY);
        assert_eq!(stats.snapshot().median_solve_ms_by_difficulty[&4], 132.5);
        Ok(())
    }

    #[test]
    fn malformed_timings_do_not_poison_the_rollup() {
        let mut stats = StatsStore::default();
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            stats.record(6, "cpu", invalid);
        }
        assert!(stats.snapshot().median_solve_ms_by_difficulty.is_empty());
        stats.record(6, "cpu", 12.0);
        assert_eq!(stats.snapshot().median_solve_ms_by_difficulty[&6], 12.0);
        assert_eq!(stats.solves_total, 5);
    }

    #[test]
    fn large_finite_samples_do_not_overflow_the_median() {
        let mut stats = StatsStore::default();
        stats.record(2, "gpu", f64::MAX);
        stats.record(2, "gpu", f64::MAX);
        assert_eq!(stats.snapshot().median_solve_ms_by_difficulty[&2], f64::MAX);
    }
}
