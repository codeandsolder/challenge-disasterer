//! Protocol correctness, device parity and benchmark economics tests.

use super::*;

use super::cpu::{cap_rsw_input, cap_rsw_solve};
#[cfg(feature = "gpu")]
use super::gpu::{
    run_gpu_altcha_cases, run_gpu_anubis_haphash_cases, run_gpu_capjs_cases,
    run_gpu_cerberus_cases, run_gpu_goaway_case, run_gpu_wasm_mcaptcha_cases,
};

#[test]
fn protocol_selection_defaults_to_all_and_supports_multiple_families() -> Result<()> {
    let all = Args::try_parse_from(["matrix"])?;
    for protocol in [
        Protocol::Anubis,
        Protocol::Haphash,
        Protocol::Mcaptcha,
        Protocol::Capjs,
        Protocol::Linear,
        Protocol::GoAway,
        Protocol::Cerberus,
        Protocol::Altcha,
    ] {
        assert!(all.includes(protocol));
    }
    let selected =
        Args::try_parse_from(["matrix", "--protocol", "go-away", "--protocol", "capjs"])?;
    assert!(selected.includes(Protocol::GoAway));
    assert!(selected.includes(Protocol::Capjs));
    assert!(!selected.includes(Protocol::Anubis));
    assert!(!selected.includes(Protocol::Cerberus));
    let linear = Args::try_parse_from(["matrix", "--protocol", "linear"])?;
    assert!(linear.includes(Protocol::Linear));
    assert!(!linear.has_gpu_work());
    Ok(())
}

#[test]
fn cap_rsw_fixed_vector_matches_external_reference() -> Result<()> {
    let (modulus, x) = cap_rsw_input(0)?;
    let y = cap_rsw_solve(&modulus, &x)?;
    assert_eq!(
        y.to_u64_digits().first().copied().unwrap_or(0),
        0x92a2_5a0a_9175_dee7
    );
    Ok(())
}

/// Requires an actual native GPU. Exact u32 rollover and the maximum
/// JavaScript-safe integer are both covered without traversing the
/// intervening billions of candidate hashes.
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires a Vulkan-capable GPU"]
async fn altcha_gpu_decimal_counters_cover_full_safe_integer_domain() -> Result<()> {
    use std::num::NonZeroU32;

    use challenge_solvers::{
        gpu::{AltchaSha256Solver, GpuContext},
        message::AltchaMessage,
    };

    let gpu = GpuContext::create().await?;
    gpu.warm_up().await?;
    for counter in [
        0_u64,
        u64::from(u32::MAX) - 1,
        u64::from(u32::MAX),
        u64::from(u32::MAX) + 1,
        (1_u64 << 53) - 1,
    ] {
        for pbkdf2 in [false, true] {
            let message = AltchaMessage {
                nonce: [0x9a; 16],
                salt: [0x27; 16],
                cost: NonZeroU32::new(if pbkdf2 { 16 } else { 2 })
                    .context("positive iterations")?,
                pbkdf2,
                key_length: NonZeroU32::new(32).context("positive key length")?,
            };
            let mut solver = AltchaSha256Solver::from(message);
            solver.set_string_counter();
            solver.set_start_nonce(counter);
            solver.set_limit(counter + 1);
            let result = solver
                .solve::<SOLVE_TYPE_MASK>(&gpu, 0, 0)
                .await?
                .context("GPU failed to find a one-candidate full-mask solution")?;
            assert_eq!(result.nonce, counter, "pbkdf2={pbkdf2}");
            assert_eq!(solver.attempted_nonces(), 1);
        }
    }
    Ok(())
}

/// Full 256-bit digest-prefix matching, including half-byte comparisons.
/// A deliberately wrong final nibble must fail even when the first
/// 64 digest bits match. Uses real WGPU on an explicitly selected device.
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires a Vulkan-capable GPU"]
async fn altcha_gpu_full_digest_nibble_prefixes() -> Result<()> {
    use std::num::NonZeroU32;

    use challenge_solvers::{
        gpu::{AltchaSha256Solver, GpuContext},
        message::AltchaMessage,
    };

    let gpu = GpuContext::create().await?;
    gpu.warm_up().await?;

    for pbkdf2 in [false, true] {
        for string_counter in [false, true] {
            let counter = if string_counter {
                u64::from(u32::MAX) + 1
            } else {
                17_u64
            };
            let message = AltchaMessage {
                nonce: [0x9a; 16],
                salt: [0x27; 16],
                cost: NonZeroU32::new(if pbkdf2 { 16 } else { 2 })
                    .context("positive iterations")?,
                pbkdf2,
                key_length: NonZeroU32::new(32).context("positive key length")?,
            };
            let mut reference = AltchaSha256Solver::from(message.clone());
            if string_counter {
                reference.set_string_counter();
            }
            reference.set_start_nonce(counter);
            reference.set_limit(counter + 1);
            let digest = reference
                .solve::<SOLVE_TYPE_MASK>(&gpu, 0, 0)
                .await?
                .context("reference GPU search did not find unconstrained counter")?
                .hash;
            let nibbles = (0..64)
                .map(|index| {
                    u8::try_from((digest[index / 8] >> ((7 - index % 8) * 4)) & 0x0f)
                        .context("SHA256 digest nibble fits in u8")
                })
                .collect::<Result<Vec<_>>>()?;
            for nibble_len in [0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 47, 63, 64] {
                let mut matching = AltchaSha256Solver::from(message.clone());
                if string_counter {
                    matching.set_string_counter();
                }
                matching.set_start_nonce(counter);
                matching.set_limit(counter + 1);
                matching.set_prefix_nibbles(&nibbles[..nibble_len])?;
                let result = matching
                    .solve::<SOLVE_TYPE_MASK>(&gpu, 0, 0)
                    .await?
                    .context("matching digest prefix rejected by GPU")?;
                assert_eq!(result.nonce, counter);
                assert_eq!(result.hash, digest);
                assert_eq!(matching.attempted_nonces(), 1);
                if nibble_len > 0 {
                    let mut wrong = nibbles[..nibble_len].to_vec();
                    if let Some(last) = wrong.last_mut() {
                        *last ^= 1;
                    }
                    matching.set_prefix_nibbles(&wrong)?;
                    assert!(
                        matching
                            .solve::<SOLVE_TYPE_MASK>(&gpu, 0, 0)
                            .await?
                            .is_none(),
                        "GPU accepted a wrong hex prefix length {nibble_len}"
                    );
                }
            }
            let mut malformed = AltchaSha256Solver::from(message);
            assert!(malformed.set_prefix_nibbles(&[16]).is_err());
            assert!(malformed.set_prefix_nibbles(&[0; 65]).is_err());
        }
    }
    Ok(())
}

/// Verifies arbitrary fixed salt/nonce sizes, SHA block/padding boundaries
/// and high decimal counters against an independent CPU SHA256 digest.
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires a real GPU"]
async fn altcha_gpu_general_sha256_prefixes_cross_sha_block_boundaries() -> Result<()> {
    use challenge_solvers::gpu::{AltchaSha256Solver, GpuContext};

    let gpu = GpuContext::create().await?;
    gpu.warm_up().await?;
    for fixed_len in [
        0_usize, 1, 16, 31, 48, 55, 56, 59, 60, 63, 64, 65, 119, 120, 127, 128, 129, 255, 256, 4096,
    ] {
        let salt = vec![0x5b; fixed_len];
        let nonce = vec![0xe1; fixed_len % 19];
        for (string_counter, counter) in [
            (false, 0_u64),
            (false, u64::from(u32::MAX)),
            (true, 63_u64),
            (true, u64::from(u32::MAX) + 1),
            (true, (1_u64 << 53) - 1),
        ] {
            let cost = NonZeroU32::new(3).context("valid SHA iterations")?;
            let mut solver = AltchaSha256Solver::for_sha256_prefix(&salt, &nonce, cost)?;
            if string_counter {
                solver.set_string_counter();
            }
            let suffix = if string_counter {
                counter.to_string().into_bytes()
            } else {
                u32::try_from(counter)?.to_be_bytes().to_vec()
            };
            let mut source = Vec::with_capacity(salt.len() + nonce.len() + suffix.len());
            source.extend_from_slice(&salt);
            source.extend_from_slice(&nonce);
            source.extend_from_slice(&suffix);
            let mut digest: [u8; 32] = Sha256::digest(source).into();
            for _ in 1..cost.get() {
                digest = Sha256::digest(digest).into();
            }
            let words: [u32; 8] = core::array::from_fn(|i| {
                u32::from_be_bytes([
                    digest[i * 4],
                    digest[i * 4 + 1],
                    digest[i * 4 + 2],
                    digest[i * 4 + 3],
                ])
            });
            let nibbles: Vec<u8> = digest
                .iter()
                .flat_map(|byte| [byte >> 4, byte & 0x0f])
                .collect();
            solver.set_prefix_nibbles(&nibbles)?;
            solver.set_start_nonce(counter);
            solver.set_limit(counter + 1);
            let solved = solver
                .solve::<SOLVE_TYPE_MASK>(&gpu, 0, 0)
                .await?
                .context("matching general SHA-256 digest rejected by GPU")?;
            assert_eq!(solved.nonce, counter);
            assert_eq!(solved.hash, words);
            assert_eq!(solver.attempted_nonces(), 1);
            let mut bad = nibbles;
            bad[63] ^= 1;
            solver.set_prefix_nibbles(&bad)?;
            assert!(
                solver.solve::<SOLVE_TYPE_MASK>(&gpu, 0, 0).await?.is_none(),
                "GPU accepted wrong last nibble: fixed_len={fixed_len}, string={string_counter}, counter={counter}"
            );
        }
    }
    Ok(())
}

/// Independent PBKDF2-HMAC-SHA256 oracle for arbitrary salt and password
/// lengths, HMAC key-prehash boundaries, U1 multi-block salt padding and
/// both counter encodings (including exact JS-safe upper bound).
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires a real GPU"]
async fn altcha_gpu_general_pbkdf2_crosses_hmac_and_sha_boundaries() -> Result<()> {
    use challenge_solvers::gpu::{AltchaSha256Solver, GpuContext};

    let gpu = GpuContext::create().await?;
    gpu.warm_up().await?;
    for (salt_len, nonce_len) in [
        (0_usize, 0_usize),
        (1, 1),
        (51, 16),
        (52, 16),
        (55, 16),
        (56, 16),
        (59, 16),
        (60, 16),
        (63, 16),
        (64, 16),
        (115, 16),
        (116, 16),
        (119, 16),
        (120, 16),
        (127, 16),
        (128, 16),
        (1024, 16),
        (1, 48),
        (1, 49),
        (1, 59),
        (1, 60),
        (1, 61),
        (1, 63),
        (1, 64),
        (1, 65),
        (1, 127),
        (1, 128),
        (127, 1),
        (64, 64),
        (4097, 65),
    ] {
        let salt: Vec<u8> = (0..salt_len).map(|i| (i as u8).wrapping_mul(47)).collect();
        let nonce: Vec<u8> = (0..nonce_len).map(|i| (i as u8).wrapping_mul(31)).collect();
        for (string_counter, counter) in [
            (false, 0_u64),
            (false, u64::from(u32::MAX)),
            (true, 0_u64),
            (true, u64::from(u32::MAX) + 1),
            (true, (1_u64 << 53) - 1),
        ] {
            let iterations = if salt_len == 4097 { 1 } else { 3 };
            let cost = NonZeroU32::new(iterations).context("positive iterations")?;
            let mut solver = AltchaSha256Solver::for_pbkdf2(&salt, &nonce, cost)?;
            if string_counter {
                solver.set_string_counter();
            }
            let suffix = if string_counter {
                counter.to_string().into_bytes()
            } else {
                u32::try_from(counter)?.to_be_bytes().to_vec()
            };
            let mut password = nonce.clone();
            password.extend_from_slice(&suffix);
            let mut digest = [0_u8; 32];
            pbkdf2::pbkdf2_hmac::<Sha256>(&password, &salt, iterations, &mut digest);
            let expected: [u32; 8] = core::array::from_fn(|i| {
                u32::from_be_bytes([
                    digest[i * 4],
                    digest[i * 4 + 1],
                    digest[i * 4 + 2],
                    digest[i * 4 + 3],
                ])
            });
            let nibbles: Vec<u8> = digest.iter().flat_map(|b| [b >> 4, b & 0xf]).collect();
            solver.set_start_nonce(counter);
            solver.set_limit(counter + 1);
            solver.set_prefix_nibbles(&nibbles)?;
            let answer = solver
                .solve::<SOLVE_TYPE_MASK>(&gpu, 0, 0)
                .await?
                .context("GPU rejected exact independent PBKDF2 digest")?;
            assert_eq!(
                answer.nonce, counter,
                "salt={salt_len} nonce={nonce_len} string={string_counter}"
            );
            assert_eq!(
                answer.hash, expected,
                "salt={salt_len} nonce={nonce_len} string={string_counter}"
            );
            assert_eq!(solver.attempted_nonces(), 1);
            let mut wrong = nibbles;
            wrong[63] ^= 1;
            solver.set_prefix_nibbles(&wrong)?;
            assert!(
                solver.solve::<SOLVE_TYPE_MASK>(&gpu, 0, 0).await?.is_none(),
                "GPU accepted incorrect final digest nibble: salt={salt_len} nonce={nonce_len} string={string_counter}"
            );
        }
    }
    Ok(())
}

/// Check arbitrary PBKDF2 output block counts, including exact full-key
/// hex prefixes spanning the first SHA256 output block, a second block,
/// and a third/fourth block. Compare against the standard independent
/// PBKDF2 implementation and the complete returned derived key.
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires a real GPU"]
async fn altcha_gpu_pbkdf2_multiblock_derived_key_and_full_prefix() -> Result<()> {
    use challenge_solvers::gpu::{AltchaSha256Solver, GpuContext};

    let gpu = GpuContext::create().await?;
    gpu.warm_up().await?;
    for (salt_len, nonce_len, key_len) in [
        (0_usize, 0_usize, 32_usize),
        (0, 0, 33),
        (1, 60, 33),
        (59, 64, 33),
        (60, 65, 48),
        (63, 16, 64),
        (64, 16, 65),
        (127, 1, 80),
        (128, 128, 96),
        (1024, 65, 128),
    ] {
        let salt: Vec<u8> = (0..salt_len).map(|i| (i as u8).wrapping_mul(23)).collect();
        let nonce: Vec<u8> = (0..nonce_len).map(|i| (i as u8).wrapping_mul(43)).collect();
        for (string_counter, counter) in [
            (false, 0_u64),
            (false, u64::from(u32::MAX)),
            (true, u64::from(u32::MAX) + 1),
            (true, (1_u64 << 53) - 1),
        ] {
            let cost = NonZeroU32::new(3).context("valid iterations")?;
            let mut solver = AltchaSha256Solver::for_pbkdf2(&salt, &nonce, cost)?;
            solver.set_pbkdf2_key_length(key_len)?;
            if string_counter {
                solver.set_string_counter();
            }
            let suffix = if string_counter {
                counter.to_string().into_bytes()
            } else {
                u32::try_from(counter)?.to_be_bytes().to_vec()
            };
            let mut password = nonce.clone();
            password.extend_from_slice(&suffix);
            let mut digest = vec![0_u8; key_len];
            pbkdf2::pbkdf2_hmac::<Sha256>(&password, &salt, cost.get(), &mut digest);
            let first: [u32; 8] = core::array::from_fn(|i| {
                u32::from_be_bytes([
                    digest[i * 4],
                    digest[i * 4 + 1],
                    digest[i * 4 + 2],
                    digest[i * 4 + 3],
                ])
            });
            let nibbles: Vec<u8> = digest.iter().flat_map(|b| [b >> 4, b & 0xf]).collect();
            solver.set_start_nonce(counter);
            solver.set_limit(counter + 1);
            solver.set_prefix_nibbles(&nibbles)?;
            let solved = solver
                .solve::<SOLVE_TYPE_MASK>(&gpu, 0, 0)
                .await?
                .context("GPU rejected matching full PBKDF2 multi-block prefix")?;
            assert_eq!(solved.nonce, counter);
            assert_eq!(solved.hash, first);
            assert_eq!(solver.verified_key(), Some(digest.as_slice()));
            assert_eq!(solver.attempted_nonces(), 1);
            let mut wrong = nibbles;
            *wrong.last_mut().context("non-empty derived key")? ^= 1;
            solver.set_prefix_nibbles(&wrong)?;
            assert!(
                solver.solve::<SOLVE_TYPE_MASK>(&gpu, 0, 0).await?.is_none(),
                "GPU accepted incorrect final nibble: salt={salt_len} nonce={nonce_len} key={key_len} string={string_counter}"
            );
            assert!(solver.verified_key().is_none());
        }
    }
    Ok(())
}

/// Verify both ROMix phases and all p lanes against an independent full
/// RFC 7914 scrypt implementation, including varying N and r.
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires a real GPU"]
async fn scrypt_romix_wgpu_matches_full_rfc7914_cpu_reference() -> Result<()> {
    use challenge_solvers::gpu::{GpuContext, ScryptRomixGpu};

    let gpu = GpuContext::create().await?;
    let romix = ScryptRomixGpu::new(&gpu);
    for (n, r, p) in [
        (2_u32, 1_u32, 1_u32),
        (16, 1, 1),
        (16, 1, 4),
        (16, 2, 3),
        (16, 8, 1),
        (32, 8, 2),
        (64, 1, 8),
        (64, 8, 2),
        (256, 1, 4),
        (256, 8, 1),
        (1024, 1, 2),
        (1024, 8, 1),
    ] {
        let password: Vec<u8> = (0..41)
            .map(|index| (index as u8).wrapping_mul(19))
            .collect();
        let salt: Vec<u8> = (0..(n as usize % 59))
            .map(|index| (index as u8).wrapping_mul(31))
            .collect();
        let params = scrypt::Params::new(u8::try_from(n.ilog2())?, r, p)
            .map_err(|_| anyhow::anyhow!("invalid scrypt test parameter"))?;
        let initial_len = usize::try_from(r)?
            .checked_mul(usize::try_from(p)?)
            .and_then(|rp| rp.checked_mul(128))
            .context("scrypt initial B byte count overflow")?;
        let mut original = vec![0_u8; initial_len];
        pbkdf2::pbkdf2_hmac::<Sha256>(&password, &salt, 1, &mut original);
        let scratch_max_lanes = romix.batch_lanes(n, r)?;
        assert!(scratch_max_lanes > 0);
        let mixed = romix.mix_blocks(&original, n, r).await?;
        assert_eq!(mixed.len(), original.len());
        let mut gpu_derived = [0_u8; 64];
        pbkdf2::pbkdf2_hmac::<Sha256>(&password, &mixed, 1, &mut gpu_derived);
        let mut reference = [0_u8; 64];
        scrypt::scrypt(&password, &salt, &params, &mut reference)
            .map_err(|_| anyhow::anyhow!("reference scrypt rejected key length"))?;
        assert_eq!(
            gpu_derived, reference,
            "scrypt ROMix mismatch for N={n}, r={r}, p={p}"
        );
    }
    Ok(())
}

/// Compare completed end-to-end GPU-offloaded scrypt against the same
/// RFC 7914 CPU KDF, over different batch occupancy and scratch sizes.
/// The GPU time includes host PBKDF2 input and output phases, transfers,
/// and the GPU ROMix, not only compute-shader execution.
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires a real GPU"]
async fn scrypt_romix_wgpu_economics_cross_section() -> Result<()> {
    use challenge_solvers::gpu::{GpuContext, ScryptRomixGpu};

    let gpu = GpuContext::create().await?;
    let romix = ScryptRomixGpu::new(&gpu);
    for (n, r, lanes) in [
        (16_u32, 1_u32, 1_usize),
        (16, 1, 64),
        (16, 8, 1),
        (16, 8, 64),
        (64, 1, 1),
        (64, 1, 64),
        (64, 8, 1),
        (64, 8, 64),
        (256, 1, 1),
        (256, 1, 64),
        (256, 8, 1),
        (256, 8, 64),
        (1024, 1, 1),
        (1024, 1, 64),
        (1024, 8, 1),
        (1024, 8, 64),
    ] {
        let params = scrypt::Params::new(u8::try_from(n.ilog2())?, r, 1)
            .map_err(|_| anyhow::anyhow!("invalid scrypt reference parameters"))?;
        let salt: Vec<u8> = (0..16).map(|i| (i as u8).wrapping_mul(31)).collect();
        let passwords: Vec<Vec<u8>> = (0..lanes)
            .map(|i| format!("challenge-solvers-scrypt-{i}").into_bytes())
            .collect();
        let mut expected = Vec::with_capacity(lanes);
        let start_cpu = Instant::now();
        for password in &passwords {
            let mut output = [0_u8; 32];
            scrypt::scrypt(password, &salt, &params, &mut output)
                .map_err(|_| anyhow::anyhow!("reference scrypt invalid key"))?;
            expected.push(output);
        }
        let cpu_ms = start_cpu.elapsed().as_secs_f64() * 1000.0;

        let start_gpu = Instant::now();
        let block_size = usize::try_from(r)?
            .checked_mul(128)
            .context("ROMix block size overflow")?;
        let mut original = Vec::with_capacity(lanes * block_size);
        for password in &passwords {
            let mut b = vec![0_u8; block_size];
            pbkdf2::pbkdf2_hmac::<Sha256>(password, &salt, 1, &mut b);
            original.extend_from_slice(&b);
        }
        let mixed = romix.mix_blocks(&original, n, r).await?;
        let mut actual = Vec::with_capacity(lanes);
        for (password, chunk) in passwords.iter().zip(mixed.chunks_exact(block_size)) {
            let mut output = [0_u8; 32];
            pbkdf2::pbkdf2_hmac::<Sha256>(password, chunk, 1, &mut output);
            actual.push(output);
        }
        let gpu_ms = start_gpu.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(
            actual, expected,
            "reference GPU scrypt N={n} r={r} lanes={lanes}"
        );
        println!(
            "{}",
            json!({
                "n": n, "r": r, "lanes": lanes, "cpu_ms": cpu_ms,
                "gpu_hybrid_ms": gpu_ms, "speedup": cpu_ms/gpu_ms,
                "scratch_bytes_per_lane": u64::from(n)*u64::from(r)*128,
                "max_gpu_lanes": romix.batch_lanes(n,r)?,
                "verified": true,
            })
        );
    }
    Ok(())
}

/// Interleaved hardware timestamp comparison of merged vec4 and directed
/// `ROMix` revisions, with full `RustCrypto` RFC7914 key verification.
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires a GPU with TIMESTAMP_QUERY"]
async fn scrypt_romix_wgpu_phase_timestamps() -> Result<()> {
    use challenge_solvers::gpu::{GpuContext, ScryptRomixGpu};

    let gpu = GpuContext::create_with_timestamps().await?;
    // Exact merged vec4 shader from challenge-solvers@fc56391.
    // The two implementations use the same context, queue and timestamp clock.
    let baseline = ScryptRomixGpu::new_from_wgsl(
        &gpu,
        include_str!("../fixtures/scrypt_romix_vec4_fc56391.wgsl"),
    );
    let directed = ScryptRomixGpu::new(&gpu);
    let interleaved = ScryptRomixGpu::new_from_wgsl(
        &gpu,
        include_str!("../fixtures/scrypt_romix_interleaved_scratch.wgsl"),
    );
    let tiled32 = ScryptRomixGpu::new_from_wgsl(
        &gpu,
        include_str!("../fixtures/scrypt_romix_tiled_scratch_32.wgsl"),
    );
    let tiled64 = ScryptRomixGpu::new_from_wgsl(
        &gpu,
        include_str!("../fixtures/scrypt_romix_tiled_scratch_64.wgsl"),
    );
    let variants = [
        ("merged-vec4-fc56391", &baseline),
        ("directed", &directed),
        ("interleaved-scratch", &interleaved),
        ("tiled-scratch-32", &tiled32),
        ("tiled-scratch-64", &tiled64),
    ];
    let pair_string = std::env::var("SCRYPT_PROFILE_AB_PAIR")
        .unwrap_or_else(|_| "merged-vec4-fc56391,directed".to_owned());
    let names: Vec<&str> = pair_string.split(',').collect();
    anyhow::ensure!(
        names.len() >= 2
            && names
                .iter()
                .enumerate()
                .all(|(index, name)| !names[..index].contains(name)),
        "SCRYPT_PROFILE_AB_PAIR must specify at least two distinct shader names"
    );
    let pair: Vec<_> = names
        .into_iter()
        .map(|name| {
            variants
                .iter()
                .copied()
                .find(|variant| variant.0 == name)
                .ok_or_else(|| anyhow::anyhow!("unknown scrypt shader variant: {name}"))
        })
        .collect::<Result<_>>()?;
    let trials = std::env::var("SCRYPT_PROFILE_TRIALS")
        .ok()
        .map(|value| value.parse::<u32>())
        .transpose()?
        .unwrap_or(5);
    anyhow::ensure!(trials > 0, "SCRYPT_PROFILE_TRIALS must be positive");
    let salt: Vec<u8> = (0_u8..53).map(|i| i.wrapping_mul(31)).collect();
    let filter = std::env::var("SCRYPT_PROFILE_SHAPES").ok();
    let mut selected = 0;
    for (n, r, lanes) in [
        (64_u32, 1_u32, 1_usize),
        (64, 1, 64),
        (256, 1, 64),
        (256, 1, 256),
        (256, 8, 1),
        (256, 8, 64),
        (256, 8, 256),
        (1024, 8, 64),
        (1024, 8, 256),
    ] {
        let shape = format!("{n}/{r}/{lanes}");
        if filter
            .as_deref()
            .is_some_and(|value| !value.split(',').any(|part| part == shape))
        {
            continue;
        }
        selected += 1;
        let block_bytes = usize::try_from(r)?
            .checked_mul(128)
            .context("scrypt profile block size overflow")?;
        let passwords = (0..lanes)
            .map(|i| format!("scrypt-profile-candidate-{i}").into_bytes())
            .collect::<Vec<Vec<u8>>>();
        let params = scrypt::Params::new(u8::try_from(n.ilog2())?, r, 1)
            .map_err(|_| anyhow::anyhow!("invalid scrypt profiling parameters"))?;
        let mut initial = Vec::with_capacity(lanes * block_bytes);
        let mut expected = Vec::with_capacity(lanes);
        for password in &passwords {
            let mut block = vec![0_u8; block_bytes];
            pbkdf2::pbkdf2_hmac::<Sha256>(password, &salt, 1, &mut block);
            initial.extend_from_slice(&block);
            let mut output = [0_u8; 32];
            scrypt::scrypt(password, &salt, &params, &mut output)
                .map_err(|_| anyhow::anyhow!("independent scrypt profile reference failed"))?;
            expected.push(output);
        }
        // Pre-compile and warm both pipelines before interleaved timing.
        for (_, solver) in &pair {
            let (mixed, _) = solver.mix_blocks_with_timestamps(&initial, n, r).await?;
            assert_eq!(mixed.len(), initial.len());
        }
        for trial in 0..trials {
            // Rotate execution order to keep every implementation equally
            // likely to run first despite thermals and competing GPU work.
            for position in 0..pair.len() {
                let (variant, romix) = pair[(trial as usize + position) % pair.len()];
                let wall_start = Instant::now();
                let (mixed, timestamps) = romix.mix_blocks_with_timestamps(&initial, n, r).await?;
                let wall_ns = wall_start.elapsed().as_nanos();
                assert_eq!(mixed.len(), initial.len());
                let mut actual = Vec::with_capacity(lanes);
                for (password, block) in passwords.iter().zip(mixed.chunks_exact(block_bytes)) {
                    let mut output = [0_u8; 32];
                    pbkdf2::pbkdf2_hmac::<Sha256>(password, block, 1, &mut output);
                    actual.push(output);
                }
                assert_eq!(actual, expected, "ROMix profile: {shape}, {variant}");
                assert!(
                    !timestamps.is_empty(),
                    "GPU produced no ROMix phase timestamps"
                );
                for (batch, timing) in timestamps.iter().enumerate() {
                    println!(
                        "{}",
                        json!({
                            "variant":variant, "adapter":gpu.adapter(),
                            "n":n, "r":r, "lanes":lanes, "trial":trial, "batch":batch,
                            "fill_ticks":timing.fill_ticks,
                            "mix_ticks":timing.mix_ticks,
                            "timestamp_period_ns":timing.timestamp_period_ns,
                            "wall_ns":wall_ns, "verified":true,
                        })
                    );
                }
            }
        }
    }
    assert!(
        selected > 0,
        "SCRYPT_PROFILE_SHAPES selects no profile cases"
    );
    Ok(())
}

/// Compare the entire on-device PBKDF2/ROMix/PBKDF2 pipeline with the
/// independent RFC7914 RustCrypto reference, including multiblock keys,
/// arbitrary salt/password lengths and cross-batch p lanes.
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires a real GPU"]
async fn scrypt_full_wgpu_matches_independent_rfc7914_kdf() -> Result<()> {
    use challenge_solvers::gpu::{GpuContext, ScryptKdfGpu};

    let gpu = GpuContext::create().await?;
    let scrypt_gpu = ScryptKdfGpu::new(&gpu);
    for (n, r, p, password_len, salt_len, key_len) in [
        (2_u32, 1_u32, 1_u32, 0_usize, 0_usize, 1_usize),
        (2, 1, 1, 1, 1, 33),
        (16, 1, 1, 16, 55, 32),
        (16, 1, 4, 64, 56, 64),
        (16, 2, 3, 65, 59, 96),
        (16, 8, 1, 128, 60, 32),
        (64, 1, 8, 31, 119, 33),
        (64, 8, 2, 65, 120, 64),
        (256, 1, 4, 48, 257, 80),
        (1024, 8, 1, 65, 4096, 64),
        (1024, 8, 130, 17, 52, 32),
    ] {
        let password = (0..password_len)
            .map(|i| (i as u8).wrapping_mul(19))
            .collect::<Vec<_>>();
        let salt = (0..salt_len)
            .map(|i| (i as u8).wrapping_mul(43))
            .collect::<Vec<_>>();
        let params = scrypt::Params::new(u8::try_from(n.ilog2())?, r, p)
            .map_err(|_| anyhow::anyhow!("invalid scrypt reference parameters"))?;
        let mut expected = vec![0_u8; key_len];
        scrypt::scrypt(&password, &salt, &params, &mut expected)
            .map_err(|_| anyhow::anyhow!("RustCrypto reference rejected output length"))?;
        let actual = scrypt_gpu
            .derive_key(&password, &salt, n, r, p, key_len)
            .await?;
        assert_eq!(
            actual, expected,
            "GPU full scrypt mismatch N={n},r={r},p={p},password_len={password_len},salt_len={salt_len},key_len={key_len}"
        );
    }
    Ok(())
}

/// Optimized release-profile benchmark: a single scrypt KDF whose
/// p independent ROMix lanes are fully GPU-resident. Unlike the older
/// ROMix-only matrix, both PBKDF2 stages and final key are on device.
/// Exact complete CPU oracle output is checked for every GPU trial.
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires real GPU; run cargo test --release -- --ignored"]
async fn scrypt_full_wgpu_release_economics() -> Result<()> {
    use challenge_solvers::gpu::{GpuContext, ScryptKdfGpu};

    let gpu = GpuContext::create().await?;
    let solver = ScryptKdfGpu::new(&gpu);
    let password: Vec<u8> = (0_u8..65).map(|i| i.wrapping_mul(31)).collect();
    let salt: Vec<u8> = (0_u8..59).map(|i| i.wrapping_mul(23)).collect();
    for (n, r, p) in [
        (16_u32, 1_u32, 1_u32),
        (16, 1, 64),
        (16, 8, 1),
        (16, 8, 64),
        (64, 1, 1),
        (64, 1, 64),
        (64, 8, 1),
        (64, 8, 64),
        (256, 1, 1),
        (256, 1, 64),
        (256, 8, 1),
        (256, 8, 64),
        (1024, 1, 1),
        (1024, 1, 64),
        (1024, 8, 1),
        (1024, 8, 64),
        (1024, 8, 130),
    ] {
        let params = scrypt::Params::new(u8::try_from(n.ilog2())?, r, p)
            .map_err(|_| anyhow::anyhow!("invalid RFC scrypt params"))?;
        let mut reference = [0_u8; 32];
        scrypt::scrypt(&password, &salt, &params, &mut reference)
            .map_err(|_| anyhow::anyhow!("reference scrypt failed"))?;
        let warm = solver.derive_key(&password, &salt, n, r, p, 32).await?;
        assert_eq!(warm, reference, "initial full GPU result mismatch");
        for trial in 0..3_u32 {
            let mut cpu_ns = 0_u128;
            let gpu_ns;
            let mut run_cpu = || -> Result<()> {
                let start = Instant::now();
                let mut actual = [0_u8; 32];
                scrypt::scrypt(&password, &salt, &params, &mut actual)
                    .map_err(|_| anyhow::anyhow!("reference scrypt failed"))?;
                cpu_ns = start.elapsed().as_nanos();
                assert_eq!(actual, reference);
                Ok(())
            };
            if trial.is_multiple_of(2) {
                let start = Instant::now();
                let derived = solver.derive_key(&password, &salt, n, r, p, 32).await?;
                gpu_ns = start.elapsed().as_nanos();
                assert_eq!(derived, reference, "GPU != CPU RFC7914");
                run_cpu()?;
            } else {
                run_cpu()?;
                let start = Instant::now();
                let derived = solver.derive_key(&password, &salt, n, r, p, 32).await?;
                gpu_ns = start.elapsed().as_nanos();
                assert_eq!(derived, reference, "GPU != CPU RFC7914");
            }
            println!(
                "{}",
                json!({
                    "n":n, "r":r, "p":p, "trial":trial,
                    "cpu_ns":cpu_ns, "full_gpu_ns":gpu_ns,
                    "speedup":cpu_ns as f64/gpu_ns as f64,
                    "scratch_bytes_per_lane":u64::from(n)*u64::from(r)*128,
                    "key_length":32, "verified":true
                })
            );
        }
    }
    Ok(())
}

/// The fast Gray-code SHA kernel must never search past the caller's
/// exact budget, including single-candidate and partial-workgroup tails.
/// Both direct Anubis and the fast single-block route use the same kernel.
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires a real GPU"]
async fn sha256_fast_gpu_hard_limits_are_exact() -> Result<()> {
    use challenge_solvers::{
        gpu::{GpuContext, SingleBlockSolver},
        message::SingleBlockMessage,
    };
    let gpu = GpuContext::create().await?;
    let prefix = vec![0x58_u8; 64];
    let difficulty = NonZeroU8::new(1).context("difficulty one is nonzero")?;
    let mask = 0xf000_0000_0000_0000_u64;

    let mut block = [0_u8; 64];
    block[..10].copy_from_slice(b"1000000000");
    block[19] = 0x80;
    block[56..64].copy_from_slice(&(19_u64 * 8).to_be_bytes());
    let template = SingleBlockMessage {
        message: core::array::from_fn(|i| {
            let chunk = &block[i * 4..i * 4 + 4];
            u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
        }),
        prefix_state: [
            0x6a09_e667,
            0xbb67_ae85,
            0x3c6e_f372,
            0xa54f_f53a,
            0x510e_527f,
            0x9b05_688c,
            0x1f83_d9ab,
            0x5be0_cd19,
        ],
        digit_index: 10,
        nonce_addend: 1_000_000_000_000_000_000,
        approx_working_set_count: NonZeroU32::new(1).context("working set")?,
        no_trailing_zeros: false,
    };
    for limit in [0_u64, 1, 2, 7, 127, 128, 129, 8191, 8192, 8193, 16385] {
        let mut direct_expected = false;
        let mut fast_expected = false;
        for counter in 0..limit {
            let decimal = format!("1{counter:018o}");
            let digest = Sha256::digest([prefix.as_slice(), decimal.as_bytes()].concat());
            direct_expected |= digest[0] & 0xf0 == 0;
            let digest = Sha256::digest(decimal.as_bytes());
            fast_expected |= digest[0] & 0xf0 == 0;
        }

        let direct = gpu
            .solve_anubis_with_limit(&prefix, difficulty, limit)
            .await?;
        assert_eq!(
            direct.is_some(),
            direct_expected,
            "direct Anubis limit={limit}"
        );
        if let Some(found) = direct {
            let decimal = found.nonce.to_string();
            anyhow::ensure!(decimal.len() == 19 && decimal.starts_with('1'));
            let counter = u64::from_str_radix(&decimal[1..], 8)?;
            assert!(counter < limit, "direct searched outside limit {limit}");
            let digest = Sha256::digest([prefix.as_slice(), decimal.as_bytes()].concat());
            let actual: Vec<u8> = found
                .hash
                .iter()
                .flat_map(|word| word.to_be_bytes())
                .collect();
            assert_eq!(actual.as_slice(), digest.as_slice());
            assert!(found.dispatched_hashes <= limit);
        }

        let mut fast = SingleBlockSolver::from(template.clone());
        fast.set_limit(limit);
        let found = fast.solve::<SOLVE_TYPE_MASK>(&gpu, 0, mask).await?;
        assert_eq!(
            found.is_some(),
            fast_expected,
            "fast single-block limit={limit}"
        );
        assert!(fast.attempted_nonces() <= limit);
        if let Some(found) = found {
            let decimal = found.nonce.to_string();
            anyhow::ensure!(decimal.len() == 19 && decimal.starts_with('1'));
            let counter = u64::from_str_radix(&decimal[1..], 8)?;
            assert!(
                counter < limit,
                "fast single-block searched outside limit {limit}"
            );
            let digest = Sha256::digest(decimal.as_bytes());
            let actual: Vec<u8> = found
                .hash
                .iter()
                .flat_map(|word| word.to_be_bytes())
                .collect();
            assert_eq!(actual.as_slice(), digest.as_slice());
            assert!(found.dispatched_hashes <= limit);
        }
        println!(
            "{}",
            json!({"limit":limit,"direct_hit":direct_expected,"fast_hit":fast_expected,"verified":true})
        );
    }
    Ok(())
}

/// Measure the real crossover between independent SHA-256 GPU requests
/// (serialized on one shared GPU context) and the same independent requests
/// on one CPU worker or a bounded Rayon CPU pool. All tasks do exact work.
///
/// The measured CPU alternatives include their own solver construction.
/// The GPU includes real queue submissions and readbacks for each request.
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires a physical GPU; use --release for economics"]
async fn sha256_concurrent_cpu_gpu_crossover() -> Result<()> {
    use rayon::prelude::*;

    let threads = std::env::var("SHA256_CROSSOVER_THREADS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(8);
    let request_counts = std::env::var("SHA256_CROSSOVER_REQUEST_COUNTS")
        .unwrap_or_else(|_| "1,2,4,8,16".to_owned());
    let request_counts: Vec<usize> = request_counts
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    anyhow::ensure!(
        !request_counts.is_empty() && request_counts.iter().all(|count| *count > 0),
        "independent-request counts must be positive"
    );
    let max_requests = *request_counts
        .iter()
        .max()
        .context("at least one request count required")?;
    let trials = std::env::var("SHA256_CROSSOVER_TRIALS")
        .ok()
        .map(|value| value.parse::<u32>())
        .transpose()?
        .unwrap_or(5);
    anyhow::ensure!(
        threads > 0 && trials > 0,
        "CPU threads and trials must be positive"
    );
    let budgets = std::env::var("SHA256_CROSSOVER_BUDGETS")
        .unwrap_or_else(|_| "1,64,1024,8192,65536,262144,1048576".to_owned());
    let budgets: Vec<u64> = budgets
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    anyhow::ensure!(
        !budgets.is_empty() && budgets.iter().all(|budget| *budget > 0),
        "crossover candidate budgets must be positive"
    );

    let gpu = challenge_solvers::gpu::GpuContext::create().await?;
    gpu.warm_up().await?;
    let cpu_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;
    let prefixes: Vec<Vec<u8>> = (0..max_requests)
        .map(|request| -> Result<Vec<u8>> {
            let mut prefix = vec![b'a'; 96];
            prefix[..8].copy_from_slice(&u64::try_from(request)?.to_le_bytes());
            Ok(prefix)
        })
        .collect::<Result<_>>()?;

    let cpu_solve = |prefix: &[u8], budget: u64| -> Result<()> {
        let message = DecimalMessage::new(prefix, 0)
            .context("construct matching CPU Anubis decimal challenge")?;
        let mut solver = challenge_solvers::DecimalSolver::from(message);
        solver.set_limit(budget);
        let solution = solver.solve::<SOLVE_TYPE_MASK>(0, u64::MAX);
        anyhow::ensure!(solution.is_none(), "unexpected 64-bit SHA CPU proof");
        anyhow::ensure!(
            solver.attempted_nonces() == budget,
            "CPU did not exhaust exact search budget"
        );
        Ok(())
    };
    let gpu_solve = async |prefix: &[u8], budget: u64| {
        let message = DecimalMessage::new(prefix, 0)
            .context("construct matching GPU Anubis decimal challenge")?;
        let mut solver = challenge_solvers::gpu::DecimalSolver::from(message);
        solver.set_limit(budget);
        let solution = solver.solve::<SOLVE_TYPE_MASK>(&gpu, 0, u64::MAX).await?;
        anyhow::ensure!(solution.is_none(), "unexpected 64-bit SHA GPU proof");
        anyhow::ensure!(
            solver.attempted_nonces() == budget,
            "GPU did not exhaust exact search budget"
        );
        Ok::<(), anyhow::Error>(())
    };

    // Warm actual protocol encoding, CPU SIMD initialization, Rayon
    // workers, GPU pipeline and readback outside timed observations.
    cpu_pool.install(|| {
        prefixes
            .par_iter()
            .try_for_each(|prefix| cpu_solve(prefix, 8192))
    })?;
    for prefix in &prefixes {
        gpu_solve(prefix, 8192).await?;
    }
    let adapter = gpu.adapter();
    for requests in request_counts {
        let prefixes = &prefixes[..requests];
        for &budget in &budgets {
            for trial in 0..trials {
                // Rotating order prevents systematic first-backend advantage.
                for phase in 0..3_u32 {
                    let backend = (trial + phase) % 3;
                    let started = Instant::now();
                    match backend {
                        0 => {
                            for prefix in prefixes {
                                cpu_solve(prefix, budget)?;
                            }
                        }
                        1 => {
                            cpu_pool.install(|| {
                                prefixes
                                    .par_iter()
                                    .try_for_each(|prefix| cpu_solve(prefix, budget))
                            })?;
                        }
                        2 => {
                            for prefix in prefixes {
                                gpu_solve(prefix, budget).await?;
                            }
                        }
                        _ => unreachable!("backend is always in 0..3"),
                    }
                    let elapsed_ns = started.elapsed().as_nanos();
                    println!(
                        "{}",
                        json!({
                            "benchmark": "sha256-independent-requests-crossover-v1",
                            "host": read_trimmed("/proc/sys/kernel/hostname"),
                            "cpu": cpu_model(),
                            "gpu": adapter,
                            "protocol": "anubis",
                            "variant": "decimal",
                            "backend": match backend {
                                0 => "cpu-serial-auto",
                                1 => "cpu-parallel-auto",
                                2 => "gpu-serial-wgpu",
                                _ => unreachable!("backend is always in 0..3"),
                            },
                            "threads": if backend == 1 { threads } else { 1 },
                            "requests": requests,
                            "budget_per_request": budget,
                            "attempted_work": u64::try_from(requests)?.checked_mul(budget).context("total work overflow")?,
                            "trial": trial,
                            "elapsed_ns": elapsed_ns,
                            "verified": true
                        })
                    );
                }
            }
        }
    }
    Ok(())
}

/// Compare equivalent SHA-256 majority formulas across all GPU protocol
/// families with the same device, queue, work budget and live solver paths.
/// Each pair uses identical challenge bytes, rotating execution order.
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires a physical GPU; use --release for throughput comparisons"]
async fn sha256_gpu_majority_interleaved() -> Result<()> {
    use challenge_solvers::gpu::GpuContext;

    let baseline = GpuContext::create().await?;
    let or_majority = baseline.with_alternative_majority();
    let samples = std::env::var("SHA256_MAJORITY_AB_TRIALS")
        .ok()
        .map(|value| value.parse::<u32>())
        .transpose()?
        .unwrap_or(10);
    anyhow::ensure!(samples > 0, "SHA256_MAJORITY_AB_TRIALS must be positive");
    let limit = std::env::var("SHA256_MAJORITY_AB_LIMIT")
        .ok()
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(1_048_576);
    anyhow::ensure!(
        limit > 0 && limit.is_multiple_of(8192),
        "SHA256_MAJORITY_AB_LIMIT must be a positive multiple of 8192"
    );
    anyhow::ensure!(baseline.adapter() == or_majority.adapter());
    baseline.warm_up().await?;
    or_majority.warm_up().await?;
    let mut args = Args {
        samples: 1,
        limit,
        argon2_limit: 64,
        cpu: false,
        gpu: true,
        list_gpu_adapters: false,
        mode: Mode::Throughput,
        commit: String::new(),
        protocol: Vec::new(),
    };
    let host = read_trimmed("/proc/sys/kernel/hostname");
    let cpu = cpu_model();
    let protocols = [
        Protocol::Anubis,
        Protocol::Haphash,
        Protocol::Mcaptcha,
        Protocol::Capjs,
        Protocol::GoAway,
        Protocol::Cerberus,
        Protocol::Altcha,
    ];
    // Warm actual challenge formats, not only the synthetic startup
    // dispatch. Shader compilation and first-use driver work then occur
    // outside the timed pairs.
    args.limit = 8192;
    for (variant, gpu) in [("xor-majority", &baseline), ("or-majority", &or_majority)] {
        let meta = Meta {
            host: host.clone(),
            cpu_model: cpu.clone(),
            commit: format!("warmup-{variant}"),
        };
        for protocol in protocols {
            args.protocol = vec![protocol];
            run_sha256_majority_case(&meta, &args, gpu, protocol, 0).await?;
        }
    }
    args.limit = limit;
    for trial in 0..samples {
        let order = if trial.is_multiple_of(2) {
            [("xor-majority", &baseline), ("or-majority", &or_majority)]
        } else {
            [("or-majority", &or_majority), ("xor-majority", &baseline)]
        };
        for protocol in protocols {
            args.protocol = vec![protocol];
            for (variant, gpu) in order {
                let meta = Meta {
                    host: host.clone(),
                    cpu_model: cpu.clone(),
                    commit: variant.to_owned(),
                };
                run_sha256_majority_case(&meta, &args, gpu, protocol, trial).await?;
            }
        }
    }
    Ok(())
}

#[cfg(feature = "gpu")]
async fn run_sha256_majority_case(
    meta: &Meta,
    args: &Args,
    gpu: &challenge_solvers::gpu::GpuContext,
    protocol: Protocol,
    sample: u32,
) -> Result<()> {
    match protocol {
        Protocol::Anubis => {
            run_gpu_anubis_haphash_cases(meta, args, gpu, gpu.adapter(), sample).await?;
            run_gpu_wasm_mcaptcha_cases(meta, args, gpu, gpu.adapter(), sample).await
        }
        Protocol::Haphash => {
            run_gpu_anubis_haphash_cases(meta, args, gpu, gpu.adapter(), sample).await
        }
        Protocol::Mcaptcha => {
            run_gpu_wasm_mcaptcha_cases(meta, args, gpu, gpu.adapter(), sample).await
        }
        Protocol::Capjs => run_gpu_capjs_cases(meta, args, gpu, gpu.adapter(), sample).await,
        Protocol::GoAway => run_gpu_goaway_case(meta, args, gpu, gpu.adapter(), sample).await,
        Protocol::Cerberus => run_gpu_cerberus_cases(meta, args, gpu, gpu.adapter(), sample).await,
        Protocol::Altcha => run_gpu_altcha_cases(meta, args, gpu, gpu.adapter(), sample).await,
        Protocol::Linear => anyhow::bail!("Linear.pub has no SHA-256 GPU solver"),
    }
}

/// Same-device paired benchmark of complete GPU PBKDF2/ROMix/PBKDF2
/// using the exact merged vec4 shader and the directed candidate shader.
/// Every full derived key is checked against independent RustCrypto scrypt.
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires a real GPU; use --release for performance measurements"]
async fn scrypt_full_kdf_interleaved_ab() -> Result<()> {
    use challenge_solvers::gpu::{GpuContext, ScryptKdfGpu};
    use rayon::prelude::*;

    let cpu_pool = rayon::ThreadPoolBuilder::new().build()?;
    let cpu_threads = cpu_pool.current_num_threads();
    let gpu = GpuContext::create().await?;
    let baseline = ScryptKdfGpu::new_from_romix_wgsl(
        &gpu,
        include_str!("../fixtures/scrypt_romix_vec4_fc56391.wgsl"),
    );
    let directed = ScryptKdfGpu::new(&gpu);
    let interleaved = ScryptKdfGpu::new_from_romix_wgsl(
        &gpu,
        include_str!("../fixtures/scrypt_romix_interleaved_scratch.wgsl"),
    );
    let tiled32 = ScryptKdfGpu::new_from_romix_wgsl(
        &gpu,
        include_str!("../fixtures/scrypt_romix_tiled_scratch_32.wgsl"),
    );
    let tiled64 = ScryptKdfGpu::new_from_romix_wgsl(
        &gpu,
        include_str!("../fixtures/scrypt_romix_tiled_scratch_64.wgsl"),
    );
    let variants = [
        ("merged-vec4-fc56391", &baseline),
        ("directed", &directed),
        ("interleaved-scratch", &interleaved),
        ("tiled-scratch-32", &tiled32),
        ("tiled-scratch-64", &tiled64),
    ];
    let pair_string = std::env::var("SCRYPT_KDF_AB_PAIR")
        .unwrap_or_else(|_| "merged-vec4-fc56391,directed".to_owned());
    let names: Vec<&str> = pair_string.split(',').collect();
    anyhow::ensure!(
        names.len() == 2 && names[0] != names[1],
        "SCRYPT_KDF_AB_PAIR must specify two distinct shader names"
    );
    let pair: Vec<_> = names
        .into_iter()
        .map(|name| {
            variants
                .iter()
                .copied()
                .find(|variant| variant.0 == name)
                .ok_or_else(|| anyhow::anyhow!("unknown scrypt shader variant: {name}"))
        })
        .collect::<Result<_>>()?;
    let salt: Vec<u8> = (0_u8..59).map(|i| i.wrapping_mul(23)).collect();
    let filter = std::env::var("SCRYPT_KDF_AB_SHAPES").ok();
    let mut selected = 0;
    for (n, r, p, count) in [
        (64_u32, 1_u32, 1_u32, 1_usize),
        (64, 1, 1, 64),
        (256, 1, 1, 64),
        (256, 1, 1, 256),
        (256, 8, 1, 1),
        (256, 8, 1, 64),
        (256, 8, 1, 256),
        (1024, 8, 1, 64),
        (256, 8, 4, 8),
    ] {
        let shape = format!("{n}/{r}/{p}/{count}");
        if filter
            .as_deref()
            .is_some_and(|value| !value.split(',').any(|part| part == shape))
        {
            continue;
        }
        selected += 1;
        let passwords: Vec<Vec<u8>> = (0..count)
            .map(|index| {
                (0..(65 + index % 3))
                    .map(|byte| (byte as u8).wrapping_mul(31).wrapping_add(index as u8))
                    .collect()
            })
            .collect();
        let refs: Vec<&[u8]> = passwords.iter().map(Vec::as_slice).collect();
        let params = scrypt::Params::new(u8::try_from(n.ilog2())?, r, p)
            .map_err(|_| anyhow::anyhow!("invalid RFC7914 parameters: {shape}"))?;
        let cpu_start = Instant::now();
        let mut expected = Vec::with_capacity(count);
        for password in &passwords {
            let mut derived = [0_u8; 32];
            scrypt::scrypt(password, &salt, &params, &mut derived)
                .map_err(|_| anyhow::anyhow!("RustCrypto scrypt failed: {shape}"))?;
            expected.push(derived.to_vec());
        }
        let cpu_serial_ns = cpu_start.elapsed().as_nanos();
        for (_, solver) in &pair {
            let warm = solver.derive_keys(&refs, &salt, n, r, p, 32).await?;
            assert_eq!(warm, expected, "full KDF A/B warmup: {shape}");
        }
        for trial in 0..5_u32 {
            // Compare against a warm, concurrently dispatched CPU
            // reference rather than treating serial RustCrypto as the
            // achievable CPU throughput on a multi-core machine.
            let cpu_parallel_start = Instant::now();
            let parallel_reference: Vec<Vec<u8>> = cpu_pool.install(|| {
                passwords
                    .par_iter()
                    .map(|password| -> Result<Vec<u8>> {
                        let mut key = [0_u8; 32];
                        scrypt::scrypt(password, &salt, &params, &mut key)
                            .map_err(|_| anyhow::anyhow!("CPU reference failed: {shape}"))?;
                        Ok(key.to_vec())
                    })
                    .collect::<Result<Vec<_>>>()
            })?;
            let cpu_parallel_ns = cpu_parallel_start.elapsed().as_nanos();
            assert_eq!(
                parallel_reference, expected,
                "CPU parallel reference: {shape}"
            );
            let order = if trial.is_multiple_of(2) {
                [pair[0], pair[1]]
            } else {
                [pair[1], pair[0]]
            };
            for (variant, solver) in order {
                let start = Instant::now();
                let actual = solver.derive_keys(&refs, &salt, n, r, p, 32).await?;
                let gpu_full_kdf_ns = start.elapsed().as_nanos();
                assert_eq!(
                    actual, expected,
                    "full RFC7914 KDF mismatch: {shape}, {variant}, trial={trial}"
                );
                println!(
                    "{}",
                    json!({
                        "variant": variant, "adapter": gpu.adapter(),
                        "n": n, "r": r, "p": p, "candidates": count,
                        "trial": trial, "gpu_full_kdf_ns": gpu_full_kdf_ns,
                        "cpu_serial_reference_ns": cpu_serial_ns,
                        "cpu_parallel_ns": cpu_parallel_ns,
                        "cpu_threads": cpu_threads,
                        "verified": true,
                    })
                );
            }
        }
    }
    assert!(
        selected > 0,
        "SCRYPT_KDF_AB_SHAPES selects no benchmark cases"
    );
    Ok(())
}

/// Complete multi-password GPU KDF compared with independent CPU RFC7914.
/// Separates independent candidate count from scrypt's internal p lanes.
/// Exercise SHA256 HMAC key boundaries, arbitrary salt sizes, and
/// simultaneous per-password output >32 bytes.
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires an actual WGPU device"]
async fn scrypt_batched_passwords_gpu_independent_reference() -> Result<()> {
    use challenge_solvers::gpu::{GpuContext, ScryptKdfGpu};

    let gpu = GpuContext::create().await?;
    let solver = ScryptKdfGpu::new(&gpu);
    for (n, r, p, candidates, key_length, salt_len) in [
        (2_u32, 1_u32, 1_u32, 1_usize, 1_usize, 0_usize),
        (16, 1, 1, 2, 33, 55),
        (16, 1, 4, 17, 64, 56),
        (64, 8, 2, 8, 96, 120),
        (256, 1, 3, 64, 64, 4096),
        (1024, 8, 1, 130, 32, 52),
    ] {
        let salt: Vec<u8> = (0..salt_len).map(|i| (i as u8).wrapping_mul(13)).collect();
        let lengths = [0, 1, 16, 63, 64, 65, 128];
        let passwords: Vec<Vec<u8>> = (0..candidates)
            .map(|candidate| {
                (0..lengths[candidate % lengths.len()])
                    .map(|i| (i as u8).wrapping_mul(23).wrapping_add(candidate as u8))
                    .collect()
            })
            .collect();
        let refs: Vec<&[u8]> = passwords.iter().map(Vec::as_slice).collect();
        let actual = solver
            .derive_keys(&refs, &salt, n, r, p, key_length)
            .await?;
        assert_eq!(actual.len(), candidates);
        let params = scrypt::Params::new(u8::try_from(n.ilog2())?, r, p)
            .map_err(|_| anyhow::anyhow!("invalid RFC7914 reference params"))?;
        for (index, (password, result)) in passwords.iter().zip(actual).enumerate() {
            let mut expected = vec![0_u8; key_length];
            scrypt::scrypt(password, &salt, &params, &mut expected)
                .map_err(|_| anyhow::anyhow!("RustCrypto scrypt rejected parameters"))?;
            assert_eq!(
                result, expected,
                "candidate {index}, N={n}, r={r}, p={p}, count={candidates}"
            );
        }
        assert_eq!(
            solver
                .derive_key(&passwords[0], &salt, n, r, p, key_length)
                .await?,
            solver
                .derive_keys(&[refs[0]], &salt, n, r, p, key_length)
                .await?[0],
        );
        assert!(solver.batch_candidates(salt.len(), n, r, p, key_length)? > 0);
    }
    let empty: Vec<&[u8]> = Vec::new();
    assert!(
        solver
            .derive_keys(&empty, b"", 16, 8, 1, 32)
            .await?
            .is_empty()
    );
    assert!(solver.batch_candidates(16, 3, 8, 1, 32).is_err());
    assert!(solver.batch_candidates(16, 16, 0, 1, 32).is_err());
    assert!(solver.batch_candidates(16, 16, 8, 0, 32).is_err());
    Ok(())
}

/// Optimized release benchmark for independent *candidate passwords*,
/// as opposed to one scrypt password using p internal lanes.
/// Every returned key is compared against RustCrypto CPU scrypt.
#[cfg(feature = "gpu")]
#[tokio::test]
#[ignore = "requires real GPU; run with cargo test --release"]
async fn scrypt_batched_passwords_release_economics() -> Result<()> {
    use challenge_solvers::gpu::{GpuContext, ScryptKdfGpu};

    let gpu = GpuContext::create().await?;
    let solver = ScryptKdfGpu::new(&gpu);
    let salt: Vec<u8> = (0_u8..53).map(|i| i.wrapping_mul(31)).collect();
    for (n, r, count) in [
        (64_u32, 1_u32, 1_usize),
        (64, 1, 64),
        (64, 1, 256),
        (256, 1, 1),
        (256, 1, 64),
        (256, 1, 256),
        (256, 8, 1),
        (256, 8, 64),
        (256, 8, 256),
        (1024, 1, 1),
        (1024, 1, 64),
        (1024, 1, 256),
        (1024, 8, 1),
        (1024, 8, 64),
        (1024, 8, 128),
        (1024, 8, 256),
    ] {
        let passwords: Vec<Vec<u8>> = (0..count)
            .map(|candidate| {
                (0..65)
                    .map(|i| (i as u8).wrapping_mul(17).wrapping_add(candidate as u8))
                    .collect()
            })
            .collect();
        let refs: Vec<&[u8]> = passwords.iter().map(Vec::as_slice).collect();
        let params = scrypt::Params::new(u8::try_from(n.ilog2())?, r, 1)
            .map_err(|_| anyhow::anyhow!("invalid scrypt benchmark params"))?;
        let mut expected = Vec::with_capacity(count);
        for password in &passwords {
            let mut key = [0_u8; 32];
            scrypt::scrypt(password, &salt, &params, &mut key)
                .map_err(|_| anyhow::anyhow!("RustCrypto failed reference key"))?;
            expected.push(key.to_vec());
        }
        let initial = solver.derive_keys(&refs, &salt, n, r, 1, 32).await?;
        assert_eq!(initial, expected);
        for trial in 0..3_u32 {
            let mut cpu_ns = 0_u128;
            let gpu_ns;
            let mut cpu = || -> Result<()> {
                let start = Instant::now();
                let mut derived = Vec::with_capacity(count);
                for password in &passwords {
                    let mut key = [0_u8; 32];
                    scrypt::scrypt(password, &salt, &params, &mut key)
                        .map_err(|_| anyhow::anyhow!("reference scrypt failed"))?;
                    derived.push(key.to_vec());
                }
                cpu_ns = start.elapsed().as_nanos();
                assert_eq!(derived, expected);
                Ok(())
            };
            if trial.is_multiple_of(2) {
                let start = Instant::now();
                let keys = solver.derive_keys(&refs, &salt, n, r, 1, 32).await?;
                gpu_ns = start.elapsed().as_nanos();
                assert_eq!(keys, expected);
                cpu()?;
            } else {
                cpu()?;
                let start = Instant::now();
                let keys = solver.derive_keys(&refs, &salt, n, r, 1, 32).await?;
                gpu_ns = start.elapsed().as_nanos();
                assert_eq!(keys, expected);
            }
            println!(
                "{}",
                json!({
                    "n":n, "r":r, "p":1, "candidates":count, "trial":trial,
                    "cpu_ns":cpu_ns, "gpu_batch_ns":gpu_ns,
                    "speedup":cpu_ns as f64/gpu_ns as f64, "verified":true,
                    "device_batch_capacity":solver.batch_candidates(salt.len(),n,r,1,32)?
                })
            );
        }
    }
    Ok(())
}

#[test]
fn throughput_condition_preserves_comparison_type() {
    assert_eq!(
        benchmark_condition(Mode::Throughput, 123, 456, SOLVE_TYPE_GT),
        (u64::MAX - 1, u64::MAX, SOLVE_TYPE_GT)
    );
    assert_eq!(
        benchmark_condition(Mode::Throughput, 123, 456, SOLVE_TYPE_LT),
        (1, u64::MAX, SOLVE_TYPE_LT)
    );
    assert_eq!(
        benchmark_condition(Mode::Throughput, 123, 456, SOLVE_TYPE_MASK),
        (0, u64::MAX, SOLVE_TYPE_MASK)
    );
}
