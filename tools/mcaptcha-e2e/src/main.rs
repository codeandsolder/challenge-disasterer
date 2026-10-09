use mcaptcha_pow_sha256::{ConfigBuilder, PoW};
use serde::Deserialize;
use std::io::{self, Read};

#[derive(Deserialize)]
struct ServiceResponse {
    answer: Answer,
    solve_ms: f64,
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

fn main() {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .expect("read response");
    let response: ServiceResponse = serde_json::from_str(&input).expect("parse service response");

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
        "mCaptcha: PASS nonce={} solve_ms={:.3}",
        response.answer.nonce, response.solve_ms
    );
}
