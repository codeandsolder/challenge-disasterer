//! One cancellable GPU admission gate and its protocol-scoped measurements.
//!
//! The gate is shared by all GPU worker families, including their channel
//! submission. Queue waiting is counted before semaphore acquisition, not
//! after a worker receives a job. RAII makes task cancellation observable.

use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex, PoisonError},
    time::Instant,
};

use serde::Serialize;
use tokio::sync::{AcquireError, OwnedSemaphorePermit, Semaphore};

const HISTORY: usize = 256;

#[derive(Clone)]
pub(super) struct GpuAdmission {
    inner: Arc<Inner>,
}

struct Inner {
    slots: Arc<Semaphore>,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    waiting: usize,
    running: usize,
    protocols: BTreeMap<String, Protocol>,
}

#[derive(Default)]
struct Protocol {
    waiting: usize,
    running: usize,
    finished: u64,
    cancelled_waits: u64,
    wait_samples: VecDeque<f64>,
    occupancy_samples: VecDeque<f64>,
}

/// Durations record permit acquisition and occupancy, not GPU kernel timings.
#[derive(Debug, Serialize)]
pub(super) struct GpuAdmissionSnapshot {
    pub(super) waiting_requests: usize,
    pub(super) running_requests: usize,
    pub(super) protocols: BTreeMap<String, ProtocolSnapshot>,
}

#[derive(Debug, Serialize)]
pub(super) struct ProtocolSnapshot {
    waiting_requests: usize,
    running_requests: usize,
    finished_reservations: u64,
    cancelled_waits: u64,
    wait_p50_ms: Option<f64>,
    wait_p95_ms: Option<f64>,
    occupancy_p50_ms: Option<f64>,
    occupancy_p95_ms: Option<f64>,
}

impl GpuAdmission {
    pub(super) fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                slots: Arc::new(Semaphore::new(1)),
                state: Mutex::new(State::default()),
            }),
        }
    }

    /// Join the FIFO admission queue; cancellation drops a waiting reservation.
    pub(super) async fn acquire(&self, protocol: &str) -> Result<GpuPermit, AcquireError> {
        let waiting = Waiting::new(self.inner.clone(), protocol.to_owned());
        let permit = self.inner.slots.clone().acquire_owned().await?;
        Ok(waiting.activate(permit))
    }

    /// Only claim an idle GPU; failure never creates a waiting request.
    pub(super) fn try_acquire(&self, protocol: &str) -> Option<GpuPermit> {
        let permit = self.inner.slots.clone().try_acquire_owned().ok()?;
        let protocol = protocol.to_owned();
        let acquired = Instant::now();
        self.inner.with_state(|state| {
            state.running += 1;
            state.protocols.entry(protocol.clone()).or_default().running += 1;
        });
        Some(GpuPermit {
            _permit: permit,
            inner: self.inner.clone(),
            protocol,
            acquired,
        })
    }

    pub(super) fn snapshot(&self) -> GpuAdmissionSnapshot {
        self.inner.with_state(|state| GpuAdmissionSnapshot {
            waiting_requests: state.waiting,
            running_requests: state.running,
            protocols: state
                .protocols
                .iter()
                .map(|(name, samples)| {
                    (
                        name.clone(),
                        ProtocolSnapshot {
                            waiting_requests: samples.waiting,
                            running_requests: samples.running,
                            finished_reservations: samples.finished,
                            cancelled_waits: samples.cancelled_waits,
                            wait_p50_ms: percentile(&samples.wait_samples, 50),
                            wait_p95_ms: percentile(&samples.wait_samples, 95),
                            occupancy_p50_ms: percentile(&samples.occupancy_samples, 50),
                            occupancy_p95_ms: percentile(&samples.occupancy_samples, 95),
                        },
                    )
                })
                .collect(),
        })
    }
}

impl Inner {
    fn with_state<T>(&self, f: impl FnOnce(&mut State) -> T) -> T {
        // A telemetry snapshot must remain possible even after another
        // thread panics while updating counters.
        let mut guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        f(&mut guard)
    }
}

struct Waiting {
    inner: Arc<Inner>,
    protocol: String,
    entered: Instant,
    pending: bool,
}

impl Waiting {
    fn new(inner: Arc<Inner>, protocol: String) -> Self {
        inner.with_state(|state| {
            state.waiting += 1;
            state.protocols.entry(protocol.clone()).or_default().waiting += 1;
        });
        Self {
            inner,
            protocol,
            entered: Instant::now(),
            pending: true,
        }
    }

    fn activate(mut self, permit: OwnedSemaphorePermit) -> GpuPermit {
        let acquired = Instant::now();
        let wait_ms = acquired.duration_since(self.entered).as_secs_f64() * 1e3;
        self.inner.with_state(|state| {
            state.waiting -= 1;
            state.running += 1;
            let protocol = state.protocols.entry(self.protocol.clone()).or_default();
            protocol.waiting -= 1;
            protocol.running += 1;
            push_sample(&mut protocol.wait_samples, wait_ms);
        });
        self.pending = false;
        GpuPermit {
            _permit: permit,
            inner: self.inner.clone(),
            protocol: self.protocol.clone(),
            acquired,
        }
    }
}

impl Drop for Waiting {
    fn drop(&mut self) {
        if self.pending {
            self.inner.with_state(|state| {
                state.waiting -= 1;
                let protocol = state.protocols.entry(self.protocol.clone()).or_default();
                protocol.waiting -= 1;
                protocol.cancelled_waits += 1;
            });
        }
    }
}

/// Remains owned by the job until its GPU worker releases the global slot.
pub(super) struct GpuPermit {
    _permit: OwnedSemaphorePermit,
    inner: Arc<Inner>,
    protocol: String,
    acquired: Instant,
}

impl Drop for GpuPermit {
    fn drop(&mut self) {
        let elapsed_ms = self.acquired.elapsed().as_secs_f64() * 1e3;
        self.inner.with_state(|state| {
            state.running -= 1;
            let protocol = state.protocols.entry(self.protocol.clone()).or_default();
            protocol.running -= 1;
            protocol.finished += 1;
            push_sample(&mut protocol.occupancy_samples, elapsed_ms);
        });
    }
}

fn push_sample(samples: &mut VecDeque<f64>, value: f64) {
    if samples.len() == HISTORY {
        samples.pop_front();
    }
    samples.push_back(value);
}

fn percentile(samples: &VecDeque<f64>, percentile: usize) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let mut values: Vec<_> = samples.iter().copied().collect();
    values.sort_by(f64::total_cmp);
    let index = values
        .len()
        .saturating_mul(percentile)
        .div_ceil(100)
        .saturating_sub(1);
    values.get(index).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    async fn wait_for_depth(gate: &GpuAdmission, depth: usize) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while gate.snapshot().waiting_requests != depth {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap_or_else(|err| panic!("wait for {depth} GPU queue depth: {err}"));
    }

    #[tokio::test]
    async fn tracks_all_families_and_cancellation() -> anyhow::Result<()> {
        let gate = GpuAdmission::new();
        assert_eq!(gate.snapshot().running_requests, 0);
        let active = gate.acquire("anubis-sha256").await?;
        let a = gate.clone();
        let b = gate.clone();
        let hashx = tokio::spawn(async move { a.acquire("anubis-hashx").await });
        wait_for_depth(&gate, 1).await;
        let friendly = tokio::spawn(async move { b.acquire("protocol:friendlycaptcha").await });
        wait_for_depth(&gate, 2).await;
        let snapshot = gate.snapshot();
        assert_eq!(snapshot.running_requests, 1);
        assert_eq!(snapshot.protocols["anubis-hashx"].waiting_requests, 1);
        assert_eq!(
            snapshot.protocols["protocol:friendlycaptcha"].waiting_requests,
            1
        );
        hashx.abort();
        let _ = hashx.await;
        wait_for_depth(&gate, 1).await;
        assert_eq!(gate.snapshot().protocols["anubis-hashx"].cancelled_waits, 1);
        drop(active);
        let friendly = friendly.await??;
        assert_eq!(gate.snapshot().running_requests, 1);
        drop(friendly);
        let snapshot = gate.snapshot();
        assert_eq!(snapshot.waiting_requests, 0);
        assert_eq!(snapshot.running_requests, 0);
        assert_eq!(snapshot.protocols["anubis-sha256"].finished_reservations, 1);
        assert_eq!(
            snapshot.protocols["protocol:friendlycaptcha"].finished_reservations,
            1
        );
        assert!(
            snapshot.protocols["protocol:friendlycaptcha"]
                .wait_p50_ms
                .is_some()
        );
        Ok(())
    }

    #[tokio::test]
    async fn waiting_requests_receive_fifo_grants() -> anyhow::Result<()> {
        let gate = GpuAdmission::new();
        let blocker = gate.acquire("protocol:cerberus").await?;
        let (order_tx, mut order_rx) = tokio::sync::mpsc::unbounded_channel::<u8>();
        let first_gate = gate.clone();
        let first_sender = order_tx.clone();
        let first = tokio::spawn(async move {
            let _permit = first_gate.acquire("anubis-hashx").await?;
            first_sender.send(1)?;
            Ok::<_, anyhow::Error>(())
        });
        wait_for_depth(&gate, 1).await;
        let second_gate = gate.clone();
        let second = tokio::spawn(async move {
            let _permit = second_gate.acquire("protocol:altcha").await?;
            order_tx.send(2)?;
            Ok::<_, anyhow::Error>(())
        });
        wait_for_depth(&gate, 2).await;
        drop(blocker);
        assert_eq!(order_rx.recv().await, Some(1));
        assert_eq!(order_rx.recv().await, Some(2));
        first.await??;
        second.await??;
        let state = gate.snapshot();
        assert_eq!(state.waiting_requests, 0);
        assert_eq!(state.running_requests, 0);
        Ok(())
    }

    #[tokio::test]
    async fn opportunistic_reservation_does_not_enqueue() -> anyhow::Result<()> {
        let gate = GpuAdmission::new();
        let active = gate
            .try_acquire("anubis-sha256")
            .ok_or(anyhow::anyhow!("idle slot"))?;
        assert!(gate.try_acquire("protocol:altcha").is_none());
        assert_eq!(gate.snapshot().waiting_requests, 0);
        drop(active);
        let altcha = gate
            .try_acquire("protocol:altcha")
            .ok_or(anyhow::anyhow!("freed slot"))?;
        assert_eq!(
            gate.snapshot().protocols["protocol:altcha"].running_requests,
            1
        );
        drop(altcha);
        Ok(())
    }

    #[tokio::test]
    async fn dropped_worker_channel_releases_the_slot() -> anyhow::Result<()> {
        let gate = GpuAdmission::new();
        let permit = gate.acquire("protocol:cerberus").await?;
        let (tx, rx) = tokio::sync::mpsc::channel::<GpuPermit>(1);
        drop(rx);
        let error = tx.send(permit).await;
        drop(error);
        assert_eq!(gate.snapshot().running_requests, 0);
        assert_eq!(
            gate.snapshot().protocols["protocol:cerberus"].finished_reservations,
            1
        );
        Ok(())
    }

    #[tokio::test]
    async fn closed_gate_cancels_wait_without_leaking() -> anyhow::Result<()> {
        let gate = GpuAdmission::new();
        gate.inner.slots.close();
        assert!(gate.acquire("protocol:linear").await.is_err());
        let snapshot = gate.snapshot();
        assert_eq!(snapshot.waiting_requests, 0);
        assert_eq!(snapshot.running_requests, 0);
        assert_eq!(snapshot.protocols["protocol:linear"].cancelled_waits, 1);
        Ok(())
    }
}
