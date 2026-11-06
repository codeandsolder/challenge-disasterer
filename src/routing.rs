use std::{collections::BTreeMap, fs, path::Path};

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

        // A batch hint is the number of concurrent independent requests, not
        // a selector for the nearest calibration row. Estimate completion time
        // for each *concrete* backend across its measured concurrency points
        // before comparing routes; otherwise a slow nearby calibration can
        // incorrectly outrank a much faster backend measured at another batch.
        let mut profiles = BTreeMap::new();
        for entry in self.entries.iter().filter(|entry| {
            !entry.router
                && entry.algorithm == algorithm
                && entry.difficulty == difficulty
                && compatible_adapter(entry, selected_adapter)
        }) {
            profiles
                .entry((
                    entry.provider.as_str(),
                    entry.provider_url.as_deref(),
                    entry.adapter.as_deref(),
                    entry.solver.as_str(),
                    entry.implementation.as_deref(),
                ))
                .or_insert_with(Vec::new)
                .push(entry);
        }

        let mut candidates = Vec::with_capacity(profiles.len());
        for rows in profiles.into_values() {
            // The closest row remains the representative for provider/solver
            // identity; its makespan is replaced with the estimate at the
            // requested batch so external timeouts scale with actual work.
            let mut nearest = rows[0];
            for &entry in rows.iter().skip(1) {
                if compare_batch_ratio(entry.batch_size, nearest.batch_size, batch_size)
                    .then_with(|| entry.makespan_ms.total_cmp(&nearest.makespan_ms))
                    .is_lt()
                {
                    nearest = entry;
                }
            }
            let mut candidate = nearest.clone();
            candidate.makespan_ms = estimate_makespan_ms(&rows, batch_size);
            candidates.push(candidate);
        }
        candidates.sort_by(|left, right| {
            left.makespan_ms
                .total_cmp(&right.makespan_ms)
                .then_with(|| left.provider.cmp(&right.provider))
                .then_with(|| left.provider_url.cmp(&right.provider_url))
                .then_with(|| left.adapter.cmp(&right.adapter))
                .then_with(|| left.solver.cmp(&right.solver))
                .then_with(|| left.implementation.cmp(&right.implementation))
        });
        candidates
    }
}

/// Estimate concurrent-batch makespan using each concrete backend's sampled
/// latency curve, not a batch size picked from a different backend.
///
/// Calibrations are means over independently issued `PoW` work, so occasional
/// lucky batches can violate the physical monotonicity of batch makespan.
/// Merge repeated points with an online mean, then conservatively enforce
/// nondecreasing completion time. Interpolate linearly within measured points,
/// bound under-range predictions by the smallest measured batch, and beyond
/// the last point project at least the last observed cost per request.
#[expect(
    clippy::cast_precision_loss,
    reason = "batch sizes are converted to floating point only for approximate time predictions"
)]
fn estimate_makespan_ms(rows: &[&RouteEntry], target: usize) -> f64 {
    let mut by_batch = BTreeMap::<usize, (f64, usize)>::new();
    for entry in rows {
        let (mean, count) = by_batch.entry(entry.batch_size).or_insert((0.0, 0));
        *count += 1;
        *mean += (entry.makespan_ms - *mean) / (*count as f64);
    }

    let mut points = Vec::with_capacity(by_batch.len());
    let mut previous_ms = 0.0_f64;
    for (batch, (ms, _)) in by_batch {
        previous_ms = previous_ms.max(ms);
        points.push((batch, previous_ms));
    }

    let (mut previous_batch, mut previous_ms) = points[0];
    if target <= previous_batch {
        return previous_ms;
    }
    for &(batch, ms) in points.iter().skip(1) {
        if target <= batch {
            let fraction = (target - previous_batch) as f64 / (batch - previous_batch) as f64;
            return finite_makespan(fraction.mul_add(ms - previous_ms, previous_ms));
        }
        previous_batch = batch;
        previous_ms = ms;
    }

    let observed_rate = previous_ms / previous_batch as f64;
    let recent_rate = if points.len() >= 2 {
        let (earlier_batch, earlier_ms) = points[points.len() - 2];
        (previous_ms - earlier_ms) / (previous_batch - earlier_batch) as f64
    } else {
        0.0
    };
    finite_makespan(
        ((target - previous_batch) as f64).mul_add(observed_rate.max(recent_rate), previous_ms),
    )
}

const fn finite_makespan(ms: f64) -> f64 {
    if ms.is_finite() { ms } else { f64::MAX }
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
            [
                "nearest-faster",
                "next-batch",
                "small-batch",
                "nearest-slower"
            ]
        );
    }

    #[test]
    fn attempts_each_concrete_backend_only_once() {
        let calibration = RoutingCalibration {
            entries: vec![
                entry("remote", None, "remote-jit", 1, 1.0),
                entry("remote", None, "remote-jit", 10, 2.0),
                entry("remote", None, "remote-jit", 100, 3.0),
                entry("local", None, "local-jit", 100, 4.0),
            ],
        };
        let candidates = calibration.select_candidates("hashx", 20, 12, None);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].solver, "remote-jit");
        assert_eq!(candidates[0].batch_size, 10);
        assert_eq!(candidates[1].solver, "local-jit");
    }

    #[test]
    fn estimated_latency_overrides_relative_batch_proximity() {
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
        assert_eq!(solvers, ["double-batch", "half-batch", "exact-batch"]);
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
    fn predicted_makespan_beats_nearest_batch() {
        let calibration = RoutingCalibration {
            entries: vec![
                entry("local", None, "near-slow", 10, 50.0),
                entry("remote", None, "distant-fast", 1, 3.0),
                entry("remote", None, "distant-fast", 100, 8.0),
            ],
        };
        let candidates = calibration.select_candidates("hashx", 20, 12, None);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].solver, "distant-fast");
        assert_eq!(candidates[0].batch_size, 100);
        assert!((candidates[0].makespan_ms - (3.0 + 11.0 * 5.0 / 99.0)).abs() < 1e-10);
        assert_eq!(candidates[1].solver, "near-slow");
        assert_eq!(candidates[1].makespan_ms, 60.0);
    }

    #[test]
    fn extrapolates_for_unmeasured_concurrency_without_optimistic_plateaus() {
        let calibration = RoutingCalibration {
            entries: vec![
                entry("remote", None, "fixed", 10, 5.0),
                entry("remote", None, "fixed", 100, 20.0),
                entry("local", None, "linear", 1, 2.0),
            ],
        };
        let under = calibration.select_candidates("hashx", 20, 1, None);
        assert_eq!(under[0].solver, "linear");
        assert_eq!(under[1].makespan_ms, 5.0);
        let over = calibration.select_candidates("hashx", 20, 200, None);
        assert_eq!(over[0].solver, "fixed");
        assert_eq!(over[0].makespan_ms, 40.0);
        assert_eq!(over[1].makespan_ms, 400.0);
    }

    #[test]
    fn repeated_measurements_and_lucky_batches_do_not_make_work_faster() {
        let calibration = RoutingCalibration {
            entries: vec![
                entry("remote", None, "repeat", 1, 20.0),
                entry("remote", None, "repeat", 1, 40.0),
                entry("remote", None, "repeat", 10, 5.0),
                entry("remote", None, "repeat", 100, 10.0),
            ],
        };
        let result = calibration.select_candidates("hashx", 20, 10, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].makespan_ms, 30.0);
        assert_eq!(
            calibration.select_candidates("hashx", 20, 50, None)[0].makespan_ms,
            30.0
        );
    }

    #[test]
    fn handles_large_batches_and_durations_without_arithmetic_overflow() {
        let calibration = RoutingCalibration {
            entries: vec![entry("local", None, "large", 1, f64::MAX)],
        };
        let candidates = calibration.select_candidates("hashx", 20, usize::MAX, None);
        assert_eq!(candidates[0].makespan_ms, f64::MAX);
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
