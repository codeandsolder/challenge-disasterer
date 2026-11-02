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

fn proof(nonce: u64, result: String) -> PoW<String> {
    serde_json::from_value(serde_json::json!({
        "nonce": nonce,
        "result": result,
        "_spook": null
    }))
    .expect("deserialize official PoW type")
}

fn benchmark_fixed(sample: u32, limit: u64) {
    assert!(limit > 0, "fixed-work limit must be positive");
    let salt = format!("matrix-salt-{sample:08x}");
    let phrase = "challenge-disasterer-matrix";
    let serialized = bincode::serialize(&phrase).expect("serialize fixed mCaptcha phrase");
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
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if args.get(1).map(String::as_str) == Some("--benchmark-fixed") {
        let sample = args
            .get(2)
            .expect("missing benchmark sample")
            .parse::<u32>()
            .expect("invalid benchmark sample");
        let limit = args
            .get(3)
            .expect("missing benchmark limit")
            .parse::<u64>()
            .expect("invalid benchmark limit");
        benchmark_fixed(sample, limit);
        return;
    }
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .expect("read response");
    let response: ServiceResponse = serde_json::from_str(&input).expect("parse service response");
    let expected_implementation =
        std::env::var("CHALLENGE_IMPLEMENTATION").unwrap_or_else(|_| "auto".to_owned());
    assert_eq!(
        response.implementation, expected_implementation,
        "service silently used a different mCaptcha implementation"
    );

    let phrase = "challenge-disasterer-mcaptcha-e2e".to_owned();
    let salt = "mcaptcha-e2e-salt-long-enough-and-stable".to_owned();
    let difficulty = 10_000u32;

    let config = ConfigBuilder::default()
        .salt(salt)
        .build()
        .expect("build official config");
    let work = proof(response.answer.nonce, response.answer.result.clone());

    assert!(
        config.is_valid_proof(&work, &phrase),
        "official mCaptcha verifier rejected proof"
    );
    assert!(
        config.is_sufficient_difficulty(&work, difficulty),
        "official mCaptcha verifier rejected difficulty"
    );

    let corrupted = proof(response.answer.nonce, "0".to_owned());
    assert!(
        !config.is_valid_proof(&corrupted, &phrase),
        "official mCaptcha verifier accepted corrupted proof"
    );

    println!(
        "mCaptcha: PASS implementation={} nonce={} solve_ms={:.3}",
        response.implementation, response.answer.nonce, response.solve_ms
    );
}
