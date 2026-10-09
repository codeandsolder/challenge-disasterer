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
    pub fn record(&mut self, difficulty: u8, backend: &str, ms: f64) {
        self.record_generic(backend);
        let samples = self.samples.entry(difficulty).or_default();
        if samples.len() == HISTORY {
            samples.pop_front();
        }
        samples.push_back(ms);
    }

    pub fn record_generic(&mut self, backend: &str) {
        self.solves_total += 1;
        match backend {
            "gpu" => self.solves_gpu += 1,
            "cpu" => self.solves_cpu += 1,
            _ => {}
        }
    }
    pub fn record_failure(&mut self) {
        self.failures += 1;
    }

    pub fn snapshot(&self) -> GeneralStats {
        let median_solve_ms_by_difficulty = self
            .samples
            .iter()
            .map(|(&difficulty, samples)| {
                let mut values: Vec<_> = samples.iter().copied().collect();
                values.sort_by(f64::total_cmp);
                let median = if values.is_empty() {
                    0.0
                } else {
                    values[values.len() / 2]
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
