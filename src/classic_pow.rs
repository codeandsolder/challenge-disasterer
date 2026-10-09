use std::{num::NonZeroU32, time::Instant};

use anyhow::{Context, anyhow, bail, ensure};
use pow_buster::{
    AltchaSha256Solver, DecimalSolver,
    adapter::{
        capjs::ChallengeDescriptor as CapJsChallenge,
        cerberus::ChallengeDescriptor as CerberusChallenge, goaway::GoAwayConfig,
        mcaptcha::PoWConfig,
    },
    build_mcaptcha_prefix, compute_target_mcaptcha, extract128_be,
    message::{AltchaMessage, DecimalMessage},
    solver::{SOLVE_TYPE_GT, SOLVE_TYPE_MASK, Solver},
};
use serde::Deserialize;
use serde_json::{Value, json};

const DEFAULT_LIMIT: u64 = 4_000_000_000;

#[derive(Debug)]
pub struct Result {
    pub answer: Value,
    pub solve_ms: f64,
}

#[derive(Debug, Deserialize)]
struct McaptchaRequest {
    config: PoWConfig,
    key: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AltchaEnvelope {
    parameters: AltchaParameters,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AltchaParameters {
    algorithm: String,
    nonce: String,
    salt: String,
    cost: NonZeroU32,
    key_length: NonZeroU32,
    key_prefix: String,
}

pub fn solve(kind: &str, challenge: Value) -> anyhow::Result<Result> {
    match kind {
        "cerberus" => solve_cerberus(challenge),
        "goaway" => solve_goaway(challenge),
        "capjs" => solve_capjs(challenge),
        "mcaptcha" => solve_mcaptcha(challenge),
        "altcha" => solve_altcha(challenge),
        _ => bail!("unsupported classic solver: {kind}"),
    }
}

fn solve_cerberus(challenge: Value) -> anyhow::Result<Result> {
    let descriptor: CerberusChallenge =
        serde_json::from_value(challenge).context("invalid Cerberus challenge")?;
    let started = Instant::now();
    let (solution, attempted_nonces) = descriptor.solve_with_limit(DEFAULT_LIMIT);
    let (solution, hash) = solution.context("Cerberus search exhausted configured limit")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let response = encode_words_le(hash);
    Ok(Result {
        answer: json!({
            "solution": solution.to_string(),
            "response": response,
            "attempted_nonces": attempted_nonces,
        }),
        solve_ms,
    })
}

fn solve_goaway(challenge: Value) -> anyhow::Result<Result> {
    let config: GoAwayConfig =
        serde_json::from_value(challenge).context("invalid go-away challenge")?;
    ensure!(
        config.challenge().len() == 64,
        "go-away challenge must be exactly 64 characters"
    );
    let started = Instant::now();
    let (solution, attempted_nonces) = config.solve_with_limit(DEFAULT_LIMIT);
    let (nonce, hash) = solution.context("go-away search exhausted configured limit")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;

    let token = format!("{}{:016x}", config.challenge(), nonce);
    let id = hash
        .into_iter()
        .take(4)
        .flat_map(u32::to_ne_bytes)
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();

    Ok(Result {
        answer: json!({
            "nonce": nonce.to_string(),
            "token": token,
            "id": id,
            "attempted_nonces": attempted_nonces,
        }),
        solve_ms,
    })
}

fn solve_capjs(challenge: Value) -> anyhow::Result<Result> {
    let descriptor: CapJsChallenge =
        serde_json::from_value(challenge).context("invalid Cap.js challenge")?;
    let started = Instant::now();
    let (response, attempted_nonces) = descriptor.solve_with_limit(DEFAULT_LIMIT);
    let response = response.context("Cap.js search exhausted configured limit")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut answer = serde_json::to_value(response).context("serialize Cap.js solution")?;
    if let Value::Object(ref mut object) = answer {
        object.insert(
            "attempted_nonces_total".to_owned(),
            Value::from(attempted_nonces),
        );
    }
    Ok(Result { answer, solve_ms })
}

fn solve_mcaptcha(challenge: Value) -> anyhow::Result<Result> {
    let request: McaptchaRequest =
        serde_json::from_value(challenge).context("invalid mCaptcha request")?;
    let mut prefix = Vec::new();
    build_mcaptcha_prefix(&mut prefix, &request.config.string, &request.config.salt);
    let target = compute_target_mcaptcha(u64::from(request.config.difficulty_factor));
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;

    for bank in 0_u32.. {
        let Some(message) = DecimalMessage::new(&prefix, bank) else {
            break;
        };
        let mut solver = DecimalSolver::from(message);
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        let result = solver.solve::<SOLVE_TYPE_GT>(target, u64::MAX);
        attempted_nonces = attempted_nonces.saturating_add(solver.get_attempted_nonces());
        if let Some((nonce, hash)) = result {
            let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
            return Ok(Result {
                answer: json!({
                    "string": request.config.string,
                    "result": extract128_be(hash).to_string(),
                    "nonce": nonce,
                    "key": request.key,
                    "attempted_nonces": attempted_nonces,
                }),
                solve_ms,
            });
        }
        if attempted_nonces >= DEFAULT_LIMIT {
            break;
        }
    }

    Err(anyhow!("mCaptcha search exhausted configured limit"))
}

fn solve_altcha(challenge: Value) -> anyhow::Result<Result> {
    let envelope: AltchaEnvelope =
        serde_json::from_value(challenge).context("invalid ALTCHA v2 challenge")?;
    let parameters = envelope.parameters;
    ensure!(
        matches!(parameters.algorithm.as_str(), "SHA-256" | "PBKDF2/SHA-256"),
        "unsupported ALTCHA algorithm: {}",
        parameters.algorithm
    );
    let nonce = decode_fixed_16("nonce", &parameters.nonce)?;
    let salt = decode_fixed_16("salt", &parameters.salt)?;
    let (target, mask) = prefix_target(&parameters.key_prefix)?;
    let message = AltchaMessage {
        nonce,
        salt,
        cost: parameters.cost,
        pbkdf2: parameters.algorithm == "PBKDF2/SHA-256",
        key_length: parameters.key_length,
    };
    let mut solver = AltchaSha256Solver::from(message);
    solver.set_limit(DEFAULT_LIMIT);
    let started = Instant::now();
    let (counter, key) = solver
        .solve::<SOLVE_TYPE_MASK>(target, mask)
        .context("ALTCHA search exhausted configured limit")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let key_length = usize::try_from(parameters.key_length.get()).unwrap_or(usize::MAX);
    ensure!(key_length <= 32, "ALTCHA keyLength above 32 is unsupported");
    let derived_key = encode_words_be(key);
    let derived_key = &derived_key[..key_length * 2];

    Ok(Result {
        answer: json!({
            "counter": counter,
            "derivedKey": derived_key,
            "time": solve_ms,
            "attempted_nonces": solver.get_attempted_nonces(),
        }),
        solve_ms,
    })
}

fn decode_fixed_16(label: &str, value: &str) -> anyhow::Result<[u8; 16]> {
    let bytes = hex::decode(value).with_context(|| format!("invalid ALTCHA {label} hex"))?;
    bytes
        .try_into()
        .map_err(|_| anyhow!("ALTCHA {label} must be exactly 16 bytes"))
}

fn prefix_target(prefix: &str) -> anyhow::Result<(u64, u64)> {
    ensure!(
        !prefix.is_empty() && prefix.len().is_multiple_of(2),
        "ALTCHA keyPrefix must be non-empty whole-byte hex"
    );
    let bytes = hex::decode(prefix).context("invalid ALTCHA keyPrefix hex")?;
    ensure!(
        bytes.len() <= 4,
        "ALTCHA keyPrefix longer than 4 bytes needs the 64-bit comparison build"
    );
    let mut first = [0_u8; 8];
    first[..bytes.len()].copy_from_slice(&bytes);
    let target = u64::from_be_bytes(first);
    let bits = bytes.len() * 8;
    let mask = if bits == 0 {
        0
    } else {
        u64::MAX << (64 - bits)
    };
    Ok((target, mask))
}

fn encode_words_le(words: [u32; 8]) -> String {
    words
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn encode_words_be(words: [u32; 8]) -> String {
    words
        .into_iter()
        .flat_map(u32::to_be_bytes)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn altcha_prefix_target_places_bytes_at_hash_front() {
        assert_eq!(
            prefix_target("12ab").unwrap(),
            (0x12ab_0000_0000_0000, 0xffff_0000_0000_0000)
        );
    }

    #[test]
    fn fixed_hex_rejects_wrong_length() {
        assert!(decode_fixed_16("nonce", "00").is_err());
    }

    #[test]
    fn protocol_nonce_json_types_match_wire_contracts() {
        let mcaptcha = solve_mcaptcha(json!({
            "config": {
                "string": "schema-test",
                "difficulty_factor": 1,
                "salt": "schema-test-salt"
            },
            "key": "schema-test"
        }))
        .expect("easy mCaptcha solve");
        assert!(mcaptcha.answer["nonce"].is_u64());

        let goaway = solve_goaway(json!({
            "challenge": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "difficulty": 1
        }))
        .expect("easy go-away solve");
        assert!(goaway.answer["nonce"].is_string());
    }
}
