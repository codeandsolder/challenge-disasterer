//! Adaptive ALTCHA Argon2id scheduling with CPU and memory reservations.
use anyhow::{Context, ensure};
use clap::ValueEnum;
use serde_json::Value;
use std::{fs, sync::Arc};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub(crate) enum WorkerPolicy {
    #[default]
    Adaptive,
    Fixed,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Work {
    memory_kib: u64,
    passes: u64,
    prefix_nibbles: usize,
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
        Some(Self {
            memory_kib,
            passes,
            prefix_nibbles: prefix.len(),
        })
    }

    pub(super) fn memory_mib(self) -> anyhow::Result<u32> {
        u32::try_from(self.memory_kib.div_ceil(1024))
            .context("Argon2 memoryCost exceeds memory accounting range")
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
        let expected_tries = if self.prefix_nibbles >= 16 {
            u64::MAX
        } else {
            1_u64 << (self.prefix_nibbles * 4)
        };
        // Approximate 300ns/block-pass from observed Zen 2 Argon2 workloads.
        // Require roughly 10ms of serial candidate work per extra worker to
        // amortize thread startup. Saturate for arbitrary ALTCHA prefix lengths.
        const NS_PER_BLOCK_PASS: u128 = 300;
        const NS_PER_EXTRA_WORKER: u128 = 10_000_000;
        let estimated_ns = u128::from(self.memory_kib)
            .saturating_mul(u128::from(self.passes))
            .saturating_mul(u128::from(expected_tries))
            .saturating_mul(NS_PER_BLOCK_PASS);
        let useful_workers = estimated_ns / NS_PER_EXTRA_WORKER + 1;
        // Half of the CPU budget avoids saturating memory bandwidth on the
        // observed dual-channel architecture and leaves CPU headroom.
        let bandwidth_cap = (total_cpu_slots / 2).max(1);
        configured
            .min(bandwidth_cap)
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
    pub(super) fn workers(&self) -> usize {
        self.cpu.len()
    }

    pub(super) async fn acquire(
        cpu_slots: &Arc<Semaphore>,
        memory_slots: Option<&Arc<Semaphore>>,
        memory_capacity_mib: Option<usize>,
        work: Option<Work>,
        desired: usize,
    ) -> anyhow::Result<Self> {
        ensure!(
            desired > 0,
            "Argon2 reservation requires at least one worker"
        );
        let memory_mib = work.map(Work::memory_mib).transpose()?;
        if let (Some(mib), Some(capacity)) = (memory_mib, memory_capacity_mib) {
            ensure!(
                usize::try_from(mib)? <= capacity,
                "ALTCHA Argon2id requires {mib} MiB per worker, exceeding the configured {capacity} MiB budget"
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

    fn challenge(memory_kib: u64, passes: u64, prefix: &str) -> Value {
        json!({"parameters": {"algorithm":"argon2id", "memoryCost":memory_kib, "cost":passes,"keyPrefix":prefix}})
    }

    #[test]
    fn selection_uses_protocol_and_work_cost() -> anyhow::Result<()> {
        assert!(Work::from_challenge("altcha", &challenge(65_536, 3, "aBc")).is_some());
        assert!(Work::from_challenge("goaway", &challenge(65_536, 3, "ab")).is_none());
        assert!(Work::from_challenge("altcha", &json!({})).is_none());
        assert!(Work::from_challenge("altcha", &challenge(65_536, 3, "not-hex")).is_none());
        let tiny = Work::from_challenge("altcha", &challenge(64, 1, "abc"))
            .context("expected Argon2 profile")?;
        let large = Work::from_challenge("altcha", &challenge(65_536, 3, "abc"))
            .context("expected Argon2 profile")?;
        let trivial = Work::from_challenge("altcha", &challenge(65_536, 3, ""))
            .context("expected Argon2 profile")?;
        assert_eq!(tiny.target_workers(8, 8, WorkerPolicy::Adaptive), 1);
        assert_eq!(large.target_workers(8, 8, WorkerPolicy::Adaptive), 4);
        assert_eq!(large.target_workers(8, 16, WorkerPolicy::Adaptive), 8);
        assert_eq!(trivial.target_workers(8, 8, WorkerPolicy::Fixed), 1);
        assert_eq!(large.target_workers(8, 8, WorkerPolicy::Fixed), 8);
        Ok(())
    }

    #[tokio::test]
    async fn memory_and_cpu_are_reserved_for_every_worker() -> anyhow::Result<()> {
        let cpu = Arc::new(Semaphore::new(8));
        let memory = Arc::new(Semaphore::new(192));
        let work = Work::from_challenge("altcha", &challenge(65_536, 3, "aa"));
        let first = Reservation::acquire(&cpu, Some(&memory), Some(192), work, 8).await?;
        assert_eq!(first.workers(), 3);
        assert_eq!(first.reserved_memory_mib(), 192);
        assert_eq!(cpu.available_permits(), 5);
        drop(first);
        assert_eq!(memory.available_permits(), 192);
        assert_eq!(cpu.available_permits(), 8);
        Ok(())
    }

    #[tokio::test]
    async fn parallel_requests_can_share_both_budgets() -> anyhow::Result<()> {
        let cpu = Arc::new(Semaphore::new(8));
        let memory = Arc::new(Semaphore::new(512));
        let work = Work::from_challenge("altcha", &challenge(65_536, 3, "aa"));
        let first = Reservation::acquire(&cpu, Some(&memory), Some(512), work, 4).await?;
        let second = Reservation::acquire(&cpu, Some(&memory), Some(512), work, 4).await?;
        assert_eq!((first.workers(), second.workers()), (4, 4));
        assert_eq!(memory.available_permits(), 0);
        drop(first);
        drop(second);
        assert_eq!(cpu.available_permits(), 8);
        assert_eq!(memory.available_permits(), 512);
        Ok(())
    }
}
