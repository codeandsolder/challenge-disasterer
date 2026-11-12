//! Shared CPU and memory reservations for memory-hard `PoW`, with ALTCHA Argon2id fanout.
use anyhow::{Context, ensure};
use clap::ValueEnum;
use serde_json::Value;
use std::{
    fs,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    time::Duration,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum WorkerPolicy {
    #[default]
    Adaptive,
    Fixed,
    /// Wait briefly for co-arriving KDF jobs, then divide CPU slots fairly
    /// across all outstanding memory-hard requests before reserving workers.
    Contended,
}

const KDF_ADMISSION_WINDOW: Duration = Duration::from_millis(1);

/// Tracks memory-hard jobs from admission through the end of blocking work.
///
/// The synchronous lock makes registering a new first-worker waiter atomic
/// with testing the queue before speculative extra reservations. It is never
/// held across a semaphore await or expensive computation.
#[derive(Default)]
pub(super) struct KdfAdmission {
    state: Arc<Mutex<AdmissionState>>,
}

#[derive(Default)]
struct AdmissionState {
    in_flight: usize,
    waiting_first: usize,
}

/// A panic outside the lock must not make KDF admission counters unusable.
fn lock_admission(state: &Mutex<AdmissionState>) -> MutexGuard<'_, AdmissionState> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

impl KdfAdmission {
    pub(super) fn enter(&self) -> KdfInFlight {
        let mut state = lock_admission(&self.state);
        state.in_flight += 1;
        state.waiting_first += 1;
        KdfInFlight {
            state: Arc::clone(&self.state),
            awaiting_first: true,
        }
    }

    pub(super) fn count(&self) -> usize {
        lock_admission(&self.state).in_flight
    }
}

pub(super) struct KdfInFlight {
    state: Arc<Mutex<AdmissionState>>,
    awaiting_first: bool,
}

impl KdfInFlight {
    /// All waiting and executing KDF requests participate in the split.
    pub(super) fn fair_workers(&self, cpu_slots: usize, desired: usize) -> usize {
        let contenders = lock_admission(&self.state).in_flight.max(1);
        desired.min((cpu_slots / contenders).max(1))
    }

    /// Mark a request admitted only after both its first CPU permit and
    /// first arena of memory have been reserved.
    fn first_worker_reserved(&mut self) {
        if self.awaiting_first {
            let mut state = lock_admission(&self.state);
            state.waiting_first -= 1;
            self.awaiting_first = false;
        }
    }

    /// A bounded admission window makes a synchronized burst visible before
    /// early arrivals take speculative extra worker permits.
    pub(super) async fn coalesce(&self) {
        tokio::time::sleep(KDF_ADMISSION_WINDOW).await;
    }
}

impl Drop for KdfInFlight {
    fn drop(&mut self) {
        let mut state = lock_admission(&self.state);
        if self.awaiting_first {
            state.waiting_first -= 1;
        }
        state.in_flight -= 1;
    }
}

const NS_PER_BLOCK_PASS: u128 = 300;
const NS_PER_EXTRA_WORKER: u128 = 10_000_000;
const CACHE_FRIENDLY_ARENA_KIB: u64 = 1024;

#[derive(Clone, Copy, Debug)]
pub(super) struct Work {
    memory_kib: u64,
    passes: u64,
    prefix_nibbles: usize,
    counter_bits: usize,
}

impl Work {
    /// Malformed or unrelated challenge metadata is left to the protocol parser.
    pub(super) fn from_challenge(solver: &str, challenge: &Value) -> Option<Self> {
        let p = challenge.get("parameters")?;
        if solver != "altcha"
            || !p
                .get("algorithm")?
                .as_str()?
                .eq_ignore_ascii_case("ARGON2ID")
        {
            return None;
        }
        let memory_kib = p.get("memoryCost")?.as_u64()?;
        let passes = p.get("cost")?.as_u64()?;
        let prefix = p.get("keyPrefix")?.as_str()?;
        if memory_kib == 0
            || memory_kib > u64::from(u32::MAX)
            || passes == 0
            || passes > u64::from(u32::MAX)
            || !prefix.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return None;
        }
        let counter_bits = if challenge.get("counterMode").and_then(Value::as_str) == Some("string")
        {
            53
        } else {
            32
        };
        Some(Self {
            memory_kib,
            passes,
            prefix_nibbles: prefix.len(),
            counter_bits,
        })
    }

    pub(super) fn target_workers(
        self,
        configured: usize,
        total_cpu_slots: usize,
        policy: WorkerPolicy,
    ) -> usize {
        if self.prefix_nibbles == 0 || configured <= 1 {
            return 1;
        }
        if policy == WorkerPolicy::Fixed {
            return configured;
        }
        // A prefix longer than the counter domain cannot be interpreted as
        // random-prefix difficulty. Upstream may have precomputed a matching
        // nonce anywhere in the domain; its position is not inferable here.
        let prefix_bits = self.prefix_nibbles.saturating_mul(4);

        // The cache-sized Argon2 workload benefits from full nonce fanout.
        // Larger per-worker arenas saturate shared memory bandwidth earlier.
        // The split is a measured initial policy, not a correctness limit.

        let bandwidth_cap = configured
            .div_ceil(2)
            .max(2)
            .min((total_cpu_slots / 2).max(2))
            .min(configured);
        let worker_cap =
            if policy == WorkerPolicy::Contended || self.memory_kib <= CACHE_FRIENDLY_ARENA_KIB {
                configured.min(total_cpu_slots)
            } else {
                bandwidth_cap
            };

        if prefix_bits > self.counter_bits {
            // Unknown precomputed solution position: choose a fanout based
            // on per-worker memory cost instead of fabricating one attempt.
            return worker_cap;
        }
        let expected_tries = 1_u64 << prefix_bits;
        // Ordinary random-prefix searches admit an estimated work budget.
        let estimated_ns = u128::from(self.memory_kib)
            .saturating_mul(u128::from(self.passes))
            .saturating_mul(u128::from(expected_tries))
            .saturating_mul(NS_PER_BLOCK_PASS);
        let useful_workers = estimated_ns / NS_PER_EXTRA_WORKER + 1;
        worker_cap
            .min(usize::try_from(useful_workers).unwrap_or(usize::MAX))
            .max(1)
    }
}

/// Each permit remains held until all workers have joined in the blocking task.
pub(super) struct Reservation {
    cpu: Vec<OwnedSemaphorePermit>,
    _memory: Vec<OwnedSemaphorePermit>,
}

impl Reservation {
    pub(super) const fn workers(&self) -> usize {
        self.cpu.len()
    }

    pub(super) async fn acquire(
        cpu_slots: &Arc<Semaphore>,
        memory_slots: Option<&Arc<Semaphore>>,
        memory_capacity_mib: Option<usize>,
        memory_bytes: Option<u64>,
        desired: usize,
    ) -> anyhow::Result<Self> {
        Self::acquire_inner(
            cpu_slots,
            memory_slots,
            memory_capacity_mib,
            memory_bytes,
            desired,
            None,
        )
        .await
    }

    /// The base worker goes first. Additional CPU and memory permits are
    /// borrowed only when no admitted KDF request still awaits its base
    /// worker. Cancellation before admission drops the waiter's guard.
    pub(super) async fn acquire_fair(
        cpu_slots: &Arc<Semaphore>,
        memory_slots: Option<&Arc<Semaphore>>,
        memory_capacity_mib: Option<usize>,
        memory_bytes: Option<u64>,
        desired: usize,
        admission: &mut KdfInFlight,
    ) -> anyhow::Result<Self> {
        Self::acquire_inner(
            cpu_slots,
            memory_slots,
            memory_capacity_mib,
            memory_bytes,
            desired,
            Some(admission),
        )
        .await
    }

    async fn acquire_inner(
        cpu_slots: &Arc<Semaphore>,
        memory_slots: Option<&Arc<Semaphore>>,
        memory_capacity_mib: Option<usize>,
        memory_bytes: Option<u64>,
        desired: usize,
        mut admission: Option<&mut KdfInFlight>,
    ) -> anyhow::Result<Self> {
        ensure!(desired > 0, "KDF reservation requires at least one worker");
        let memory_mib = memory_bytes
            .map(|bytes| u32::try_from(bytes.div_ceil(1024 * 1024)))
            .transpose()
            .context("KDF memory requirement exceeds semaphore accounting range")?;
        ensure!(
            memory_mib != Some(0),
            "KDF memory requirement cannot be zero"
        );
        if let (Some(mib), Some(capacity)) = (memory_mib, memory_capacity_mib) {
            ensure!(
                usize::try_from(mib)? <= capacity,
                "KDF requires {mib} MiB per worker, exceeding the configured {capacity} MiB budget"
            );
        }
        // Always acquire memory before CPU and only await the first worker.
        // Extra workers must never wait while holding the first worker's CPU.
        let mut memory = Vec::new();
        if let (Some(slots), Some(mib)) = (memory_slots, memory_mib) {
            memory.push(slots.clone().acquire_many_owned(mib).await?);
        }
        let mut cpu = vec![cpu_slots.clone().acquire_owned().await?];
        if let Some(admission) = &mut admission {
            admission.first_worker_reserved();
        }
        for _ in 1..desired {
            // Hold the short admission lock only during the nonblocking
            // attempt for one additional worker. A waiting request
            // cannot be bypassed by speculative extras, nor can it register
            // between the check and the extra reservation.
            let _fair_first = if let Some(admission) = &admission {
                let gate = lock_admission(&admission.state);
                if gate.waiting_first != 0 {
                    break;
                }
                Some(gate)
            } else {
                None
            };
            let extra_memory = match (memory_slots, memory_mib) {
                (Some(slots), Some(mib)) => match slots.clone().try_acquire_many_owned(mib) {
                    Ok(permit) => Some(permit),
                    Err(_) => break,
                },
                _ => None,
            };
            match cpu_slots.clone().try_acquire_owned() {
                Ok(permit) => {
                    cpu.push(permit);
                    if let Some(permit) = extra_memory {
                        memory.push(permit);
                    }
                }
                Err(_) => break,
            }
        }
        Ok(Self {
            cpu,
            _memory: memory,
        })
    }

    #[cfg(test)]
    pub(super) fn reserved_memory_mib(&self) -> usize {
        self._memory
            .iter()
            .map(OwnedSemaphorePermit::num_permits)
            .sum()
    }
}

/// Linux memory availability including cgroup v2 constraints, if exposed.
/// Other targets require an explicit quota or proceed without a memory cap.
pub(super) fn available_memory_mib() -> Option<usize> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    let available_kib = text.lines().find_map(|line| {
        line.strip_prefix("MemAvailable:")
            .and_then(|value| value.split_whitespace().next())
            .and_then(|value| value.parse::<u64>().ok())
    })?;
    let mut available_bytes = available_kib.saturating_mul(1024);
    if let (Ok(max), Ok(current)) = (
        fs::read_to_string("/sys/fs/cgroup/memory.max"),
        fs::read_to_string("/sys/fs/cgroup/memory.current"),
    ) && let (Ok(max), Ok(current)) = (max.trim().parse::<u64>(), current.trim().parse::<u64>())
    {
        available_bytes = available_bytes.min(max.saturating_sub(current));
    }
    usize::try_from(available_bytes / 1024 / 1024).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn challenge(memory_kib: u64, passes: u64, prefix: &str) -> Value {
        json!({"parameters": {"algorithm":"argon2id", "memoryCost":memory_kib, "cost":passes,"keyPrefix":prefix}})
    }

    #[test]
    fn selection_uses_protocol_and_work_cost() -> anyhow::Result<()> {
        assert!(Work::from_challenge("altcha", &challenge(65_536, 3, "aBc")).is_some());
        assert!(Work::from_challenge("goaway", &challenge(65_536, 3, "ab")).is_none());
        assert!(Work::from_challenge("altcha", &json!({})).is_none());
        assert!(Work::from_challenge("altcha", &challenge(65_536, 3, "not-hex")).is_none());
        let tiny = Work::from_challenge("altcha", &challenge(64, 1, "a"))
            .context("expected Argon2 profile")?;
        let precomputed = Work::from_challenge("altcha", &challenge(64, 1, &"ab".repeat(16)))
            .context("precomputed long-prefix challenge")?;
        let large = Work::from_challenge("altcha", &challenge(65_536, 3, "abc"))
            .context("expected Argon2 profile")?;
        let trivial = Work::from_challenge("altcha", &challenge(65_536, 3, ""))
            .context("expected Argon2 profile")?;
        assert_eq!(tiny.target_workers(8, 8, WorkerPolicy::Adaptive), 1);
        assert_eq!(precomputed.target_workers(8, 8, WorkerPolicy::Adaptive), 8);
        assert_eq!(large.target_workers(8, 8, WorkerPolicy::Adaptive), 4);
        assert_eq!(large.target_workers(8, 16, WorkerPolicy::Adaptive), 4);
        assert_eq!(large.target_workers(2, 8, WorkerPolicy::Adaptive), 2);
        assert_eq!(large.target_workers(4, 8, WorkerPolicy::Adaptive), 2);
        assert_eq!(trivial.target_workers(8, 8, WorkerPolicy::Fixed), 1);
        assert_eq!(large.target_workers(8, 8, WorkerPolicy::Fixed), 8);
        assert_eq!(large.target_workers(8, 8, WorkerPolicy::Contended), 8);
        assert_eq!(tiny.target_workers(8, 8, WorkerPolicy::Contended), 1);
        Ok(())
    }

    #[tokio::test]
    async fn in_flight_tracks_waiting_and_completed_jobs() -> anyhow::Result<()> {
        let admission = KdfAdmission::default();
        let first = admission.enter();
        let second = admission.enter();
        assert_eq!(admission.count(), 2);
        assert_eq!(first.fair_workers(8, 8), 4);
        assert_eq!(second.fair_workers(3, 3), 1);
        drop(first);
        assert_eq!(second.fair_workers(8, 8), 8);
        drop(second);
        assert_eq!(admission.count(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn canceling_a_waiter_releases_its_count_and_memory() -> anyhow::Result<()> {
        let admission = Arc::new(KdfAdmission::default());
        let cpu = Arc::new(Semaphore::new(0));
        let memory = Arc::new(Semaphore::new(16));
        let (registered, ready) = tokio::sync::oneshot::channel();
        let job = tokio::spawn({
            let admission = Arc::clone(&admission);
            let cpu = Arc::clone(&cpu);
            let memory = Arc::clone(&memory);
            async move {
                let _in_flight = admission.enter();
                let _ = registered.send(());
                Reservation::acquire(&cpu, Some(&memory), Some(16), Some(8 * 1024 * 1024), 2).await
            }
        });
        ready.await?;
        assert_eq!(admission.count(), 1);
        job.abort();
        assert!(job.await.is_err());
        assert_eq!(admission.count(), 0);
        assert_eq!(memory.available_permits(), 16);
        Ok(())
    }

    #[tokio::test]
    async fn cancellation_after_spawn_keeps_job_and_reservations_alive() -> anyhow::Result<()> {
        let admission = KdfAdmission::default();
        let cpu = Arc::new(Semaphore::new(2));
        let memory = Arc::new(Semaphore::new(64));
        let guard = admission.enter();
        let reservation =
            Reservation::acquire(&cpu, Some(&memory), Some(64), Some(32 * 1024 * 1024), 2).await?;
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
        let job = tokio::task::spawn_blocking(move || {
            let _guard = guard;
            let _reservation = reservation;
            let _ = started_tx.send(());
            let _ = release_rx.recv();
            drop(_reservation);
            drop(_guard);
            let _ = finished_tx.send(());
        });
        started_rx.await?;
        job.abort();
        assert_eq!(admission.count(), 1);
        assert_eq!(cpu.available_permits(), 0);
        assert_eq!(memory.available_permits(), 0);
        release_tx.send(())?;
        finished_rx.await?;
        // The blocking closure can drop its owned resources only on completion.
        // Waiting for a different blocker's read confirms the worker has exited.
        tokio::task::spawn_blocking(move || {}).await?;
        assert_eq!(admission.count(), 0);
        assert_eq!(cpu.available_permits(), 2);
        assert_eq!(memory.available_permits(), 64);
        Ok(())
    }

    #[tokio::test]
    async fn fair_first_leaves_idle_permits_for_an_existing_waiter() -> anyhow::Result<()> {
        let admission = KdfAdmission::default();
        let cpu = Arc::new(Semaphore::new(4));
        let memory = Arc::new(Semaphore::new(80));

        let mut first = admission.enter();
        let mut second = admission.enter();
        let reserved = Reservation::acquire_fair(
            &cpu,
            Some(&memory),
            Some(80),
            Some(20 * 1024 * 1024),
            4,
            &mut first,
        )
        .await?;
        assert_eq!(reserved.workers(), 1);
        assert_eq!(cpu.available_permits(), 3);
        assert_eq!(memory.available_permits(), 60);

        let next = Reservation::acquire_fair(
            &cpu,
            Some(&memory),
            Some(80),
            Some(20 * 1024 * 1024),
            4,
            &mut second,
        )
        .await?;
        assert_eq!(next.workers(), 3);
        assert_eq!(cpu.available_permits(), 0);
        assert_eq!(memory.available_permits(), 0);
        drop(reserved);
        drop(next);
        drop(first);
        drop(second);
        assert_eq!(admission.count(), 0);
        assert_eq!(cpu.available_permits(), 4);
        assert_eq!(memory.available_permits(), 80);
        Ok(())
    }

    #[tokio::test]
    async fn canceled_memory_waiter_stops_blocking_worker_lending() -> anyhow::Result<()> {
        let admission = Arc::new(KdfAdmission::default());
        let cpu = Arc::new(Semaphore::new(4));
        let memory = Arc::new(Semaphore::new(80));
        // An active memory-heavy job uses one arena. It must not borrow
        // extras while another job is awaiting its first memory arena.
        let mut first = admission.enter();
        let reserved = Reservation::acquire_fair(
            &cpu,
            Some(&memory),
            Some(80),
            Some(40 * 1024 * 1024),
            1,
            &mut first,
        )
        .await?;
        let holding = memory.clone().acquire_many_owned(40).await?;
        let (registered_tx, registered_rx) = tokio::sync::oneshot::channel();
        let waiter = tokio::spawn({
            let admission = Arc::clone(&admission);
            let cpu = Arc::clone(&cpu);
            let memory = Arc::clone(&memory);
            async move {
                let mut second = admission.enter();
                registered_tx
                    .send(())
                    .map_err(|_| anyhow::anyhow!("receiver closed"))?;
                Reservation::acquire_fair(
                    &cpu,
                    Some(&memory),
                    Some(80),
                    Some(40 * 1024 * 1024),
                    2,
                    &mut second,
                )
                .await
            }
        });
        registered_rx.await?;
        // No permit is available for second's first arena. Even though
        // three CPUs are idle, extras cannot bypass the first-worker queue.
        let mut borrowing = admission.enter();
        let fair =
            Reservation::acquire_fair(&cpu, Some(&memory), Some(80), None, 4, &mut borrowing)
                .await?;
        assert_eq!(fair.workers(), 1);
        drop(fair);
        waiter.abort();
        assert!(waiter.await.is_err());
        drop(borrowing);
        drop(holding);
        drop(reserved);
        drop(first);
        assert_eq!(admission.count(), 0);
        assert_eq!(memory.available_permits(), 80);
        assert_eq!(cpu.available_permits(), 4);
        Ok(())
    }

    #[tokio::test]
    async fn memory_and_cpu_are_reserved_for_every_worker() -> anyhow::Result<()> {
        let cpu = Arc::new(Semaphore::new(8));
        let memory = Arc::new(Semaphore::new(192));
        let first =
            Reservation::acquire(&cpu, Some(&memory), Some(192), Some(64 * 1024 * 1024), 8).await?;
        assert_eq!(first.workers(), 3);
        assert_eq!(first.reserved_memory_mib(), 192);
        assert_eq!(cpu.available_permits(), 5);
        drop(first);
        assert_eq!(memory.available_permits(), 192);
        assert_eq!(cpu.available_permits(), 8);
        Ok(())
    }

    #[tokio::test]
    async fn argon2_and_scrypt_share_one_memory_budget() -> anyhow::Result<()> {
        const MIB: u64 = 1024 * 1024;
        let cpu = Arc::new(Semaphore::new(8));
        let memory = Arc::new(Semaphore::new(70));
        // Argon2 consumes 64 MiB; the second requested worker cannot
        // reserve another arena even when CPU slots are free.
        let argon2 = Reservation::acquire(&cpu, Some(&memory), Some(70), Some(64 * MIB), 2).await?;
        assert_eq!(argon2.workers(), 1);
        let scrypt =
            Reservation::acquire(&cpu, Some(&memory), Some(70), Some(4 * MIB + 1), 4).await?;
        assert_eq!(scrypt.workers(), 1);
        assert_eq!(memory.available_permits(), 1);
        assert_eq!(cpu.available_permits(), 6);
        drop(argon2);
        assert_eq!(memory.available_permits(), 65);
        drop(scrypt);
        assert_eq!(memory.available_permits(), 70);
        assert_eq!(cpu.available_permits(), 8);

        assert!(
            Reservation::acquire(&cpu, Some(&memory), Some(70), Some(71 * MIB), 1,)
                .await
                .is_err()
        );
        assert_eq!(memory.available_permits(), 70);
        assert_eq!(cpu.available_permits(), 8);
        Ok(())
    }

    #[tokio::test]
    async fn parallel_requests_can_share_both_budgets() -> anyhow::Result<()> {
        let cpu = Arc::new(Semaphore::new(8));
        let memory = Arc::new(Semaphore::new(512));
        let first =
            Reservation::acquire(&cpu, Some(&memory), Some(512), Some(64 * 1024 * 1024), 4).await?;
        let second =
            Reservation::acquire(&cpu, Some(&memory), Some(512), Some(64 * 1024 * 1024), 4).await?;
        assert_eq!((first.workers(), second.workers()), (4, 4));
        assert_eq!(memory.available_permits(), 0);
        drop(first);
        drop(second);
        assert_eq!(cpu.available_permits(), 8);
        assert_eq!(memory.available_permits(), 512);
        Ok(())
    }
}
