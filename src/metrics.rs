//! Shared approximate work-count conversion for GPU calibration, telemetry
//! and standalone benchmark output. Conversion is deliberately inexact above
//! the integer precision of f64; these are estimates, never proof predicates.

#[expect(
    clippy::cast_precision_loss,
    reason = "work counts are converted only for approximate timing and throughput metrics"
)]
pub const fn approximate_work_f64(work: u64) -> f64 {
    work as f64
}
