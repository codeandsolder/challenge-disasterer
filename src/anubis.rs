use std::{num::NonZeroU8, time::Instant};

use anyhow::{Context, anyhow};
use pow_buster::{
    DecimalSolver, SingleBlockSolver, compute_mask_anubis,
    message::{DecimalMessage, SingleBlockMessage},
    solver::{SOLVE_TYPE_MASK, Solver},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Deserialize)]
pub struct Challenge {
    pub data: String,
    pub difficulty: u8,
}

impl Challenge {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(1..=16).contains(&self.difficulty) {
            return Err(anyhow!("difficulty must be in 1..=16"));
        }
        if self.data.len() > 4096 {
            return Err(anyhow!("data is too large"));
        }
        Ok(())
    }

    pub fn expected_work(&self) -> u64 {
        16_u64.saturating_pow(u32::from(self.difficulty))
    }

    pub fn gpu_compatible(&self) -> bool {
        self.data.len().is_multiple_of(64)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Answer {
    pub nonce: String,
    pub hash: String,
}

#[derive(Debug, Clone)]
pub struct BackendResult {
    pub answer: Answer,
    pub solve_ms: f64,
}

fn mask(difficulty: u8) -> anyhow::Result<u64> {
    let difficulty = NonZeroU8::new(difficulty).context("difficulty cannot be zero")?;
    Ok(compute_mask_anubis(difficulty))
}

fn hash_hex(words: [u32; 8]) -> String {
    words
        .into_iter()
        .map(|word| format!("{word:08x}"))
        .collect()
}

pub fn verify(challenge: &Challenge, answer: &Answer) -> bool {
    let digest = Sha256::digest(format!("{}{}", challenge.data, answer.nonce).as_bytes());
    let hex = format!("{digest:x}");
    hex == answer.hash && hex.starts_with(&"0".repeat(challenge.difficulty as usize))
}

pub fn solve_cpu(challenge: &Challenge, limit: u64) -> anyhow::Result<BackendResult> {
    challenge.validate()?;
    let message = DecimalMessage::new(challenge.data.as_bytes(), 0)
        .context("unsupported Anubis challenge shape")?;
    let mut solver = DecimalSolver::from(message);
    solver.set_limit(limit);
    let started = Instant::now();
    let (nonce, hash) = solver
        .solve::<SOLVE_TYPE_MASK>(0, mask(challenge.difficulty)?)
        .context("CPU nonce search exhausted configured limit")?;
    let result = BackendResult {
        answer: Answer {
            nonce: nonce.to_string(),
            hash: hash_hex(hash),
        },
        solve_ms: started.elapsed().as_secs_f64() * 1000.0,
    };
    anyhow::ensure!(
        verify(challenge, &result.answer),
        "CPU solver returned an invalid answer"
    );
    Ok(result)
}

pub fn calibrate_cpu(hash_limit: u64) -> anyhow::Result<f64> {
    let started = Instant::now();
    let mut attempted_total = 0_u64;
    let calibration_mask = mask(5)?;
    for sample in 0_u8..8 {
        let mut prefix = vec![b'a'; 128];
        prefix[0] = sample;
        let message = SingleBlockMessage::new(&prefix, 0).context("calibration message rejected")?;
        let mut solver = SingleBlockSolver::from(message);
        solver.set_limit(hash_limit);
        let _ = solver.solve_nonce_only::<SOLVE_TYPE_MASK>(0, calibration_mask);
        attempted_total = attempted_total.saturating_add(solver.get_attempted_nonces());
    }
    let elapsed = started.elapsed().as_secs_f64();
    anyhow::ensure!(attempted_total > 0 && elapsed > 0.0, "invalid CPU calibration sample");
    Ok(attempted_total as f64 / elapsed)
}
