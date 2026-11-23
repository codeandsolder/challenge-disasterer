use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Deserialize)]
pub(crate) struct Measurement {
    pub(crate) attempted_work: u64,
    pub(crate) solve_ns: u128,
    pub(crate) work_per_second: f64,
}

fn throughput() -> String {
    String::from("throughput")
}

const fn valid() -> bool {
    true
}

#[derive(Deserialize)]
pub(crate) struct SolverRow {
    #[serde(default = "throughput")]
    pub(crate) phase: String,
    pub(crate) protocol: String,
    pub(crate) variant: String,
    pub(crate) implementation: String,
    pub(crate) accelerator: String,
    pub(crate) gpu_adapter: Option<String>,
    #[serde(default = "valid")]
    pub(crate) target_condition_ok: bool,
    #[serde(flatten)]
    pub(crate) measurement: Measurement,
}

#[derive(Deserialize)]
pub(crate) struct BaselineRow {
    pub(crate) baseline: String,
    pub(crate) protocol: String,
    pub(crate) variant: String,
    pub(crate) implementation: String,
    #[serde(flatten)]
    pub(crate) measurement: Measurement,
}

#[derive(Deserialize)]
pub(crate) struct ExternalRow {
    pub(crate) baseline: String,
    pub(crate) protocol: String,
    pub(crate) variant: String,
    pub(crate) implementation: String,
    pub(crate) accelerator: String,
    pub(crate) gpu_adapter: Option<String>,
    pub(crate) comparison_scope: String,
    #[serde(flatten)]
    pub(crate) measurement: Measurement,
}

#[derive(Deserialize)]
#[serde(tag = "row_kind", rename_all = "lowercase")]
pub(crate) enum AltchaRow {
    Solver(SolverRow),
    External(ExternalRow),
}

#[derive(Serialize)]
pub(crate) struct PairedHashx {
    pub(crate) host: String,
    pub(crate) difficulty: u64,
    pub(crate) implementation: String,
    pub(crate) samples: u64,
    pub(crate) mean_solver_ms: f64,
}

#[derive(Serialize)]
pub(crate) struct AnubisBenchmark {
    pub(crate) host: String,
    pub(crate) algorithm: String,
    pub(crate) solver: String,
    pub(crate) backend: String,
    pub(crate) implementation: Option<String>,
    pub(crate) difficulty: u64,
    pub(crate) samples: u64,
    pub(crate) avg_answer_ms: f64,
    pub(crate) effective_h_s: f64,
    pub(crate) mean_solver_ms: f64,
}

pub(crate) struct Run {
    pub(crate) metadata: Value,
    pub(crate) host: String,
    pub(crate) solver: Vec<SolverRow>,
    pub(crate) baseline: Vec<BaselineRow>,
    pub(crate) external: Vec<ExternalRow>,
    pub(crate) paired: Vec<PairedHashx>,
    pub(crate) benchmarks: Vec<AnubisBenchmark>,
}
