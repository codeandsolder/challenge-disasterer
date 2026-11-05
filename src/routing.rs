use std::{fs, path::Path};

use anyhow::Context;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct RouteEntry {
    pub provider: String,
    #[serde(default)]
    pub provider_url: Option<String>,
    #[serde(default)]
    pub adapter: Option<String>,
    pub solver: String,
    #[serde(default)]
    pub implementation: Option<String>,
    pub algorithm: String,
    pub difficulty: u32,
    pub batch_size: usize,
    pub makespan_ms: f64,
    #[serde(default)]
    pub router: bool,
}

#[derive(Debug, Deserialize)]
struct CalibrationFile {
    version: u32,
    entries: Vec<RouteEntry>,
}

#[derive(Debug, Clone)]
pub struct RoutingCalibration {
    entries: Vec<RouteEntry>,
}

impl RoutingCalibration {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("read routing calibration {}", path.display()))?;
        let parsed: CalibrationFile = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse routing calibration {}", path.display()))?;
        anyhow::ensure!(
            parsed.version == 1,
            "unsupported routing calibration version {}",
            parsed.version
        );
        anyhow::ensure!(
            !parsed.entries.is_empty(),
            "routing calibration has no entries"
        );
        for entry in &parsed.entries {
            anyhow::ensure!(
                entry.batch_size > 0,
                "routing calibration has zero batch size"
            );
            anyhow::ensure!(
                entry.makespan_ms.is_finite() && entry.makespan_ms >= 0.0,
                "routing calibration has invalid makespan"
            );
            anyhow::ensure!(
                entry.provider == "local" || entry.provider_url.is_some(),
                "external routing entry has no provider URL"
            );
        }
        Ok(Self {
            entries: parsed.entries,
        })
    }

    pub fn select_candidates(
        &self,
        algorithm: &str,
        difficulty: u32,
        batch_size: usize,
        selected_adapter: Option<&str>,
    ) -> Vec<RouteEntry> {
        if batch_size == 0 {
            return Vec::new();
        }

        // Preserve all compatible routes: once the nearest calibrated batch
        // fails, the caller can try the next closest measured batch.
        let mut candidates = self
            .entries
            .iter()
            .filter(|entry| {
                !entry.router
                    && entry.algorithm == algorithm
                    && entry.difficulty == difficulty
                    && compatible_adapter(entry, selected_adapter)
            })
            .cloned()
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            compare_batch_ratio(left.batch_size, right.batch_size, batch_size)
                .then_with(|| left.makespan_ms.total_cmp(&right.makespan_ms))
                .then_with(|| left.batch_size.cmp(&right.batch_size))
        });
        candidates
    }
}

fn compare_batch_ratio(left: usize, right: usize, target: usize) -> std::cmp::Ordering {
    fn ratio(size: usize, target: usize) -> (u128, u128) {
        let size = size as u128;
        let target = target as u128;
        (size.max(target), size.min(target))
    }
    let (left_num, left_den) = ratio(left, target);
    let (right_num, right_den) = ratio(right, target);
    (left_num * right_den).cmp(&(right_num * left_den))
}

fn compatible_adapter(entry: &RouteEntry, selected_adapter: Option<&str>) -> bool {
    if entry.provider != "local" {
        return entry.provider_url.is_some();
    }
    entry
        .adapter
        .as_deref()
        .is_none_or(|required| selected_adapter.is_some_and(|selected| selected == required))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(
        provider: &str,
        adapter: Option<&str>,
        solver: &str,
        batch: usize,
        ms: f64,
    ) -> RouteEntry {
        RouteEntry {
            provider: provider.to_owned(),
            provider_url: (provider != "local").then(|| "http://solver.example".to_owned()),
            adapter: adapter.map(str::to_owned),
            solver: solver.to_owned(),
            implementation: None,
            algorithm: "hashx".to_owned(),
            difficulty: 20,
            batch_size: batch,
            makespan_ms: ms,
            router: false,
        }
    }

    #[test]
    fn selects_nearest_batch_and_current_adapter() {
        let calibration = RoutingCalibration {
            entries: vec![
                entry("local", None, "anubis-hashx-jit", 10, 50.0),
                entry("local", Some("reno"), "anubis-hashx", 10, 20.0),
                entry("local", Some("rtx"), "anubis-hashx", 10, 2.0),
                entry("local", Some("reno"), "anubis-hashx", 100, 100.0),
            ],
        };
        let candidates = calibration.select_candidates("hashx", 20, 12, Some("reno"));
        assert_eq!(candidates[0].adapter.as_deref(), Some("reno"));
        assert_eq!(candidates[0].batch_size, 10);
        assert_eq!(candidates[0].solver, "anubis-hashx");
        assert_eq!(candidates[1].solver, "anubis-hashx-jit");

        let candidates = calibration.select_candidates("hashx", 20, 40, Some("reno"));
        assert_eq!(candidates[0].batch_size, 100);
    }

    #[test]
    fn tries_other_calibrated_batch_sizes_after_the_nearest() {
        let calibration = RoutingCalibration {
            entries: vec![
                entry("local", None, "small-batch", 1, 1.0),
                entry("local", Some("reno"), "nearest-slower", 10, 20.0),
                entry("remote", None, "next-batch", 100, 4.0),
                entry("local", Some("reno"), "nearest-faster", 10, 3.0),
                entry("local", Some("rtx"), "wrong-adapter", 10, 0.1),
                RouteEntry {
                    router: true,
                    ..entry("local", None, "recursive-router", 10, 0.1)
                },
            ],
        };
        let candidates = calibration.select_candidates("hashx", 20, 12, Some("reno"));
        let solvers: Vec<_> = candidates
            .iter()
            .map(|entry| entry.solver.as_str())
            .collect();
        assert_eq!(
            solvers,
            ["nearest-faster", "nearest-slower", "next-batch", "small-batch"]
        );
    }

    #[test]
    fn equal_relative_distance_prefers_faster_calibrated_route() {
        let calibration = RoutingCalibration {
            entries: vec![
                entry("local", None, "half-batch", 5, 20.0),
                entry("local", None, "double-batch", 20, 5.0),
                entry("local", None, "exact-batch", 10, 50.0),
            ],
        };
        let candidates = calibration.select_candidates("hashx", 20, 10, None);
        let solvers: Vec<_> = candidates
            .iter()
            .map(|entry| entry.solver.as_str())
            .collect();
        assert_eq!(solvers, ["exact-batch", "double-batch", "half-batch"]);
    }

    #[test]
    fn ignores_zero_batch_hints_and_unmatched_routes() {
        let calibration = RoutingCalibration {
            entries: vec![entry("local", None, "hashx-local", 10, 1.0)],
        };
        assert!(
            calibration
                .select_candidates("hashx", 20, 0, None)
                .is_empty()
        );
        assert!(
            calibration
                .select_candidates("sha256", 20, 10, None)
                .is_empty()
        );
        assert!(
            calibration
                .select_candidates("hashx", 21, 10, None)
                .is_empty()
        );
    }

    #[test]
    fn external_candidate_can_win() {
        let calibration = RoutingCalibration {
            entries: vec![
                entry("local", None, "anubis-hashx-jit", 1, 10.0),
                entry("remote", None, "hashx-fast", 1, 3.0),
            ],
        };
        let candidates = calibration.select_candidates("hashx", 20, 1, None);
        assert_eq!(candidates[0].provider, "remote");
        assert_eq!(candidates[1].provider, "local");
    }
}
