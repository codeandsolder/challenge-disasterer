//! Shared CPU and memory reservations for memory-hard `PoW`, with ALTCHA Argon2id fanout.
use anyhow::{Context, ensure};
use serde_json::Value;
use std::{
    fs,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Tracks memory-hard jobs from admission through completion of blocking work.
/// The guard belongs to the blocking solver, not the cancelable HTTP future.
#[derive(Default)]
pub(super) struct KdfAdmission {
    in_flight: Arc<AtomicUsize>,
}

impl KdfAdmission {
    pub(super) fn enter(&self) -> KdfInFlight {
        self.in_flight.fetch_add(1, Ordering::AcqRel);
        KdfInFlight {
            count: self.in_flight.clone(),
        }
    }

    pub(super) fn count(&self) -> usize {
        self.in_flight.load(Ordering::Acquire)
    }
}

pub(super) struct KdfInFlight {
    count: Arc<AtomicUsize>,
}

impl Drop for KdfInFlight {
    fn drop(&mut self) {
        self.count.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Protocol validation belongs to the solver. Only select how many
/// independent nonce-search workers to request; shared permits enforce limits.
pub(super) fn altcha_argon2_workers(
    solver: &str,
    challenge: &Value,
    configured: usize,
) -> Option<usize> {
    if solver != "altcha"
        || !challenge
            .pointer("/parameters/algorithm")?
            .as_str()?
            .eq_ignore_ascii_case("ARGON2ID")
    {
        return None;
    }
    let prefix = challenge.pointer("/parameters/keyPrefix")?.as_str()?;
    Some(if prefix.is_empty() { 1 } else { configured })
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
        for _ in 1..desired {
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

    #[test]
    fn only_parallelizes_nontrivial_altcha_argon2() {
        let challenge = |algorithm: &str, prefix: &str| json!({"parameters": {"algorithm": algorithm, "keyPrefix": prefix}});
        assert_eq!(
            altcha_argon2_workers("altcha", &challenge("ARGON2ID", "abc"), 8),
            Some(8)
        );
        assert_eq!(
            altcha_argon2_workers("altcha", &challenge("argon2id", ""), 8),
            Some(1)
        );
        assert_eq!(
            altcha_argon2_workers("altcha", &challenge("SHA-256", "abc"), 8),
            None
        );
        assert_eq!(
            altcha_argon2_workers("other", &challenge("ARGON2ID", "abc"), 8),
            None
        );
        assert_eq!(altcha_argon2_workers("altcha", &json!({}), 8), None);
    }

    #[tokio::test]
    async fn in_flight_tracks_waiting_and_completed_jobs() -> anyhow::Result<()> {
        let admission = KdfAdmission::default();
        let first = admission.enter();
        let second = admission.enter();
        assert_eq!(admission.count(), 2);
        drop(first);
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
