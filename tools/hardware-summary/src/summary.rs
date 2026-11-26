use std::{collections::BTreeMap, str::FromStr};

use anyhow::{Context, Result, ensure};
use num_traits::ToPrimitive;
use serde::Serialize;
use serde_json::{Number, Value};

use crate::records::{AnubisBenchmark, Measurement, PairedHashx, Run};

#[derive(PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct SolverKey {
    phase: String,
    protocol: String,
    variant: String,
    implementation: String,
    accelerator: String,
    host: String,
    gpu_adapter: Option<String>,
    work_budget: Option<u64>,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct BaselineKey {
    host: String,
    baseline: String,
    protocol: String,
    variant: String,
    implementation: String,
    work_budget: u64,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct ExternalKey {
    host: String,
    baseline: String,
    protocol: String,
    variant: String,
    implementation: String,
    accelerator: String,
    gpu_adapter: Option<String>,
    comparison_scope: String,
    work_budget: u64,
}

#[derive(Serialize)]
struct Stats {
    samples: usize,
    median_work_per_second: f64,
    median_solve_ms: f64,
    median_attempted_work: Number,
}

fn middle<T: Copy>(values: &[T]) -> Result<(T, T)> {
    let last = values
        .len()
        .checked_sub(1)
        .context("empty benchmark group")?;
    let left = *values.get(last / 2).context("missing lower median")?;
    let right = *values
        .get(values.len() / 2)
        .context("missing upper median")?;
    Ok((left, right))
}

fn median_float(mut values: Vec<f64>) -> Result<f64> {
    ensure!(
        values
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0),
        "invalid benchmark measurement"
    );
    values.sort_unstable_by(f64::total_cmp);
    let (left, right) = middle(&values)?;
    Ok(left.midpoint(right))
}

fn exact_work_number(twice: u128) -> Result<Number> {
    let whole = u64::try_from(twice / 2)?;
    if twice.is_multiple_of(2) {
        Ok(Number::from(whole))
    } else {
        Ok(Number::from_str(&format!("{whole}.5"))?)
    }
}

fn statistics(rows: &[Measurement]) -> Result<Stats> {
    let mut work: Vec<_> = rows.iter().map(|row| row.attempted_work).collect();
    work.sort_unstable();
    let (left, right) = middle(&work)?;
    let exact_work_twice = u128::from(left) + u128::from(right);
    let times = rows
        .iter()
        .map(|row| row.solve_ns.to_f64().context("solve time out of range"))
        .collect::<Result<Vec<_>>>()?;
    let rates = rows.iter().map(|row| row.work_per_second).collect();
    Ok(Stats {
        samples: rows.len(),
        median_work_per_second: median_float(rates)?,
        median_solve_ms: median_float(times)? / 1e6,
        median_attempted_work: exact_work_number(exact_work_twice)?,
    })
}

#[derive(Default)]
struct SolverGroup {
    rows: Vec<Measurement>,
    invalid: bool,
}

#[derive(Serialize)]
struct SolverSummary {
    #[serde(flatten)]
    key: SolverKey,
    #[serde(flatten)]
    stats: Stats,
    all_target_conditions_valid: bool,
}

#[derive(Serialize)]
struct GroupSummary<K> {
    #[serde(flatten)]
    key: K,
    #[serde(flatten)]
    stats: Stats,
}

fn groups<K>(grouped: BTreeMap<K, Vec<Measurement>>) -> Result<Vec<GroupSummary<K>>> {
    grouped
        .into_iter()
        .map(|(key, rows)| {
            Ok(GroupSummary {
                key,
                stats: statistics(&rows)?,
            })
        })
        .collect()
}

#[derive(Serialize)]
struct Comparison<'a> {
    host: &'a str,
    protocol: &'a str,
    variant: &'a str,
    legacy_baseline: &'a str,
    legacy_implementation: &'a str,
    work_budget: u64,
    legacy_median_work_per_second: f64,
    matched_current_auto_exact_work_ratio: Option<f64>,
    best_current_exact_work_implementation: Option<&'a str>,
    best_current_exact_work_ratio: Option<f64>,
}

fn ratio(numerator: f64, denominator: f64) -> Result<f64> {
    let ratio = numerator / denominator;
    ensure!(ratio.is_finite(), "non-finite throughput ratio");
    Ok(ratio)
}

fn comparisons<'a>(
    baseline: &'a [GroupSummary<BaselineKey>],
    solver: &'a [SolverSummary],
) -> Result<Vec<Comparison<'a>>> {
    let mut comparisons = Vec::new();
    for legacy in baseline.iter().filter(|row| row.key.baseline != "current") {
        let key = &legacy.key;
        let current = baseline.iter().find(|candidate| {
            candidate.key.baseline == "current"
                && candidate.key.implementation == "auto"
                && candidate.key.host == key.host
                && candidate.key.protocol == key.protocol
                && candidate.key.variant == key.variant
                && candidate.key.work_budget == key.work_budget
        });
        let matched_ratio = current
            .map(|matched| {
                ratio(
                    matched.stats.median_work_per_second,
                    legacy.stats.median_work_per_second,
                )
            })
            .transpose()?;
        let best = solver
            .iter()
            .filter(|candidate| {
                candidate.key.host == key.host
                    && candidate.key.protocol == key.protocol
                    && candidate.key.variant == key.variant
                    && candidate.key.work_budget == Some(key.work_budget)
                    && candidate.key.phase == "throughput"
                    && candidate.key.accelerator == "cpu"
                    && candidate.all_target_conditions_valid
            })
            .max_by(|left, right| {
                left.stats
                    .median_work_per_second
                    .total_cmp(&right.stats.median_work_per_second)
                    .then_with(|| right.key.implementation.cmp(&left.key.implementation))
            });
        comparisons.push(Comparison {
            host: &key.host,
            protocol: &key.protocol,
            variant: &key.variant,
            legacy_baseline: &key.baseline,
            legacy_implementation: &key.implementation,
            work_budget: key.work_budget,
            legacy_median_work_per_second: legacy.stats.median_work_per_second,
            matched_current_auto_exact_work_ratio: matched_ratio,
            best_current_exact_work_implementation: best.map(|row| row.key.implementation.as_str()),
            best_current_exact_work_ratio: best
                .map(|row| {
                    ratio(
                        row.stats.median_work_per_second,
                        legacy.stats.median_work_per_second,
                    )
                })
                .transpose()?,
        });
    }
    Ok(comparisons)
}

#[derive(Default)]
pub struct Collector {
    hosts: BTreeMap<String, Value>,
    solver: BTreeMap<SolverKey, SolverGroup>,
    baseline: BTreeMap<BaselineKey, Vec<Measurement>>,
    external: BTreeMap<ExternalKey, Vec<Measurement>>,
    paired: Vec<PairedHashx>,
    benchmarks: Vec<AnubisBenchmark>,
}

impl Collector {
    pub fn add(&mut self, run: Run) {
        let Run {
            metadata,
            host,
            solver,
            baseline,
            external,
            paired,
            benchmarks,
        } = run;
        self.hosts.insert(host.clone(), metadata);
        self.paired.extend(paired);
        self.benchmarks.extend(benchmarks);
        for row in solver {
            let budget = (row.phase == "throughput").then_some(row.measurement.attempted_work);
            let key = SolverKey {
                phase: row.phase,
                protocol: row.protocol,
                variant: row.variant,
                implementation: row.implementation,
                accelerator: row.accelerator,
                host: host.clone(),
                gpu_adapter: row.gpu_adapter,
                work_budget: budget,
            };
            let group = self.solver.entry(key).or_default();
            group.rows.push(row.measurement);
            group.invalid |= !row.target_condition_ok;
        }
        for row in baseline {
            let key = BaselineKey {
                host: host.clone(),
                baseline: row.baseline,
                protocol: row.protocol,
                variant: row.variant,
                implementation: row.implementation,
                work_budget: row.measurement.attempted_work,
            };
            self.baseline.entry(key).or_default().push(row.measurement);
        }
        for row in external {
            let key = ExternalKey {
                host: host.clone(),
                baseline: row.baseline,
                protocol: row.protocol,
                variant: row.variant,
                implementation: row.implementation,
                accelerator: row.accelerator,
                gpu_adapter: row.gpu_adapter,
                comparison_scope: row.comparison_scope,
                work_budget: row.measurement.attempted_work,
            };
            self.external.entry(key).or_default().push(row.measurement);
        }
    }

    pub fn finish(self) -> Result<Value> {
        let results = self
            .solver
            .into_iter()
            .map(|(key, group)| {
                Ok(SolverSummary {
                    key,
                    stats: statistics(&group.rows)?,
                    all_target_conditions_valid: !group.invalid,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let baseline = groups(self.baseline)?;
        let external = groups(self.external)?;
        let comparisons = comparisons(&baseline, &results)?;
        Ok(serde_json::json!({
            "hosts": self.hosts,
            "results": results,
            "preexisting_baselines": baseline,
            "preexisting_comparison": comparisons,
            "external_baselines": external,
            "paired_hashx": self.paired,
            "anubis_benchmarks": self.benchmarks,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_integer_and_half_integer_medians_above_float_precision() -> Result<()> {
        let a = 9_007_199_254_740_992_u64;
        let b = a + 1;
        let rows = [a, b].map(|attempted_work| Measurement {
            attempted_work,
            solve_ns: 1000,
            work_per_second: 10.0,
        });
        let stats = statistics(&rows)?;
        assert_eq!(
            stats.median_attempted_work.to_string(),
            "9007199254740992.5"
        );
        assert!((stats.median_solve_ms - 0.001).abs() < f64::EPSILON);
        Ok(())
    }

    #[test]
    fn invalid_measurement_is_rejected() {
        let rows = [Measurement {
            attempted_work: 1,
            solve_ns: 1,
            work_per_second: f64::INFINITY,
        }];
        assert!(statistics(&rows).is_err());
    }
}

#[cfg(test)]
mod acceptance {
    use super::*;
    use crate::load;
    use std::path::Path;

    // The frozen legacy reference uses JSON integers for odd medians and
    // binary64 values for even medians. Numeric equality suffices for this
    // small corpus; exact large integer arithmetic is tested separately.
    fn normalize_numbers(value: &mut Value) -> Result<()> {
        match value {
            Value::Number(number) => {
                *number = Number::from_f64(
                    number
                        .as_f64()
                        .context("fixture number not representable")?,
                )
                .context("fixture number not finite")?;
            }
            Value::Array(array) => array.iter_mut().try_for_each(normalize_numbers)?,
            Value::Object(object) => object.values_mut().try_for_each(normalize_numbers)?,
            _ => {}
        }
        Ok(())
    }

    #[test]
    fn all_input_families_match_frozen_python_reference() -> Result<()> {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let mut collector = Collector::default();
        for name in ["run-a", "run-b"] {
            collector.add(load::run(&fixtures.join(name))?);
        }
        let mut actual = collector.finish()?;
        // The new exact-work identity is intentionally exposed for all
        // fixed-work groups and every distinct historical implementation.
        for section in [
            "preexisting_baselines",
            "preexisting_comparison",
            "external_baselines",
        ] {
            for row in actual
                .get_mut(section)
                .and_then(Value::as_array_mut)
                .context("missing summary array")?
            {
                let row = row.as_object_mut().context("summary row not an object")?;
                assert!(row.remove("work_budget").is_some());
                if section == "preexisting_comparison" {
                    assert!(row.remove("legacy_implementation").is_some());
                }
            }
        }
        let mut expected: Value =
            serde_json::from_str(include_str!("../fixtures/python-reference.json"))?;
        normalize_numbers(&mut actual)?;
        normalize_numbers(&mut expected)?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn separate_budgets_and_legacy_implementations_never_collapse() -> Result<()> {
        let mut collector = Collector::default();
        let baseline = [
            ("legacy", "one", 100, 10.0),
            ("legacy", "two", 100, 20.0),
            ("legacy", "one", 200, 40.0),
            ("current", "auto", 100, 40.0),
            ("current", "auto", 200, 80.0),
        ]
        .map(
            |(baseline, implementation, attempted_work, work_per_second)| {
                crate::records::BaselineRow {
                    baseline: baseline.to_owned(),
                    protocol: "anubis".to_owned(),
                    variant: "hashx".to_owned(),
                    implementation: implementation.to_owned(),
                    measurement: Measurement {
                        attempted_work,
                        solve_ns: 1000,
                        work_per_second,
                    },
                }
            },
        )
        .into();
        collector.add(Run {
            host: "host".to_owned(),
            metadata: serde_json::json!({"hostname":"host"}),
            solver: Vec::new(),
            external: Vec::new(),
            paired: Vec::new(),
            benchmarks: Vec::new(),
            baseline,
        });
        let summary = collector.finish()?;
        let comparisons = summary["preexisting_comparison"]
            .as_array()
            .context("missing comparison array")?;
        assert_eq!(comparisons.len(), 3);
        assert_eq!(comparisons[0]["work_budget"], 100);
        assert_eq!(
            comparisons[0]["matched_current_auto_exact_work_ratio"].as_f64(),
            Some(4.0)
        );
        assert_eq!(comparisons[1]["work_budget"], 200);
        assert_eq!(
            comparisons[1]["matched_current_auto_exact_work_ratio"].as_f64(),
            Some(2.0)
        );
        assert_eq!(comparisons[2]["legacy_implementation"], "two");
        Ok(())
    }
}
