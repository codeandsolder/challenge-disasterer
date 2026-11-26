use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::{Number, Value};

#[derive(Deserialize)]
pub struct Measurement {
    pub attempted_work: u64,
    #[serde(deserialize_with = "exact_u128")]
    pub solve_ns: u128,
    #[serde(deserialize_with = "json_float")]
    pub work_per_second: f64,
}

fn exact_u128<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u128, D::Error> {
    Number::deserialize(deserializer)?
        .to_string()
        .parse()
        .map_err(de::Error::custom)
}

fn json_float<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    Number::deserialize(deserializer)?
        .as_f64()
        .ok_or_else(|| de::Error::custom("measurement is not a finite f64"))
}

fn throughput() -> String {
    String::from("throughput")
}

const fn valid() -> bool {
    true
}

#[derive(Deserialize)]
pub struct SolverRow {
    #[serde(default = "throughput")]
    pub phase: String,
    pub protocol: String,
    pub variant: String,
    pub implementation: String,
    pub accelerator: String,
    pub gpu_adapter: Option<String>,
    #[serde(default = "valid")]
    pub target_condition_ok: bool,
    #[serde(flatten)]
    pub measurement: Measurement,
}

#[derive(Deserialize)]
pub struct BaselineRow {
    pub baseline: String,
    pub protocol: String,
    pub variant: String,
    pub implementation: String,
    #[serde(flatten)]
    pub measurement: Measurement,
}

#[derive(Deserialize)]
pub struct ExternalRow {
    pub baseline: String,
    pub protocol: String,
    pub variant: String,
    pub implementation: String,
    pub accelerator: String,
    pub gpu_adapter: Option<String>,
    pub comparison_scope: String,
    #[serde(flatten)]
    pub measurement: Measurement,
}

#[derive(Deserialize)]
#[serde(tag = "row_kind", rename_all = "lowercase")]
pub enum AltchaRow {
    Solver(SolverRow),
    External(ExternalRow),
}

#[derive(Serialize)]
pub struct PairedHashx {
    pub host: String,
    pub difficulty: u64,
    pub implementation: String,
    pub samples: u64,
    pub mean_solver_ms: f64,
}

#[derive(Serialize)]
pub struct AnubisBenchmark {
    pub host: String,
    pub algorithm: String,
    pub solver: String,
    pub backend: String,
    pub implementation: Option<String>,
    pub difficulty: u64,
    pub samples: u64,
    pub avg_answer_ms: f64,
    pub effective_h_s: f64,
    pub mean_solver_ms: f64,
}

pub struct Run {
    pub metadata: Value,
    pub host: String,
    pub solver: Vec<SolverRow>,
    pub baseline: Vec<BaselineRow>,
    pub external: Vec<ExternalRow>,
    pub paired: Vec<PairedHashx>,
    pub benchmarks: Vec<AnubisBenchmark>,
}
