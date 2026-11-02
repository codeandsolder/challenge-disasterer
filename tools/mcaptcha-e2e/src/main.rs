use anyhow::{Context, Result, ensure};
use mcaptcha_pow_sha256::{ConfigBuilder, PoW};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    io::{self, Read},
    time::Instant,
};

#[derive(Deserialize)]
struct ServiceResponse {
    answer: Answer,
    solve_ms: f64,
    implementation: String,
}

#[derive(Deserialize)]
struct Answer {
    nonce: u64,
    result: String,
}

fn proof(nonce: u64, result: &str) -> Result<PoW<String>> {
    serde_json::from_value(serde_json::json!({
        "nonce": nonce,
        "result": result,
        "_spook": null
    }))
    .context("deserialize official PoW type")
}

fn benchmark_fixed(sample: u32, limit: u64) -> Result<()> {
    ensure!(limit > 0, "fixed-work limit must be positive");
    let salt = format!("matrix-salt-{sample:08x}");
    let phrase = "challenge-disasterer-matrix";
    let serialized = bincode::serialize(&phrase).context("serialize fixed mCaptcha phrase")?;
    let prefix = Sha256::new()
        .chain_update(salt.as_bytes())
        .chain_update(&serialized);
    println!("{}", mcaptcha_pow_sha256::dev::score(prefix.clone(), 0));

    let mut sink = 0_u128;
    let started = Instant::now();
    for nonce in 0..limit {
        sink ^= mcaptcha_pow_sha256::dev::score(prefix.clone(), nonce);
    }
    println!("{} {sink}", started.elapsed().as_nanos());
    Ok(())
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.get(1).map(String::as_str) == Some("--benchmark-fixed") {
        let sample = args
            .get(2)
            .context("missing benchmark sample")?
            .parse::<u32>()
            .context("invalid benchmark sample")?;
        let limit = args
            .get(3)
            .context("missing benchmark limit")?
            .parse::<u64>()
            .context("invalid benchmark limit")?;
        return benchmark_fixed(sample, limit);
    }

    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .context("read response")?;
    let response: ServiceResponse =
        serde_json::from_str(&input).context("parse service response")?;
    let expected_implementation =
        std::env::var("CHALLENGE_IMPLEMENTATION").unwrap_or_else(|_| "auto".to_owned());
    ensure!(
        response.implementation == expected_implementation,
        "service silently used a different mCaptcha implementation: expected {expected_implementation}, got {}",
        response.implementation
    );

    let phrase = "challenge-disasterer-mcaptcha-e2e".to_owned();
    let salt = "mcaptcha-e2e-salt-long-enough-and-stable".to_owned();
    let difficulty = 10_000u32;

    let config = ConfigBuilder::default()
        .salt(salt)
        .build()
        .context("build official config")?;
    let work = proof(response.answer.nonce, &response.answer.result)?;

    ensure!(
        config.is_valid_proof(&work, &phrase),
        "official mCaptcha verifier rejected proof"
    );
    ensure!(
        config.is_sufficient_difficulty(&work, difficulty),
        "official mCaptcha verifier rejected difficulty"
    );

    let corrupted = proof(response.answer.nonce, "0")?;
    ensure!(
        !config.is_valid_proof(&corrupted, &phrase),
        "official mCaptcha verifier accepted corrupted proof"
    );

    println!(
        "mCaptcha: PASS implementation={} nonce={} solve_ms={:.3}",
        response.implementation, response.answer.nonce, response.solve_ms
    );
    Ok(())
}
