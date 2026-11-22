//! Protocol correctness, device parity and benchmark economics tests.

use super::*;

use super::cpu::{cap_rsw_input, cap_rsw_solve};

#[test]
fn protocol_selection_defaults_to_all_and_supports_multiple_families() -> Result<()> {
    let all = Cli::try_parse_from(["matrix"])?;
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
        assert!(all.matrix.includes(protocol));
    }
    let selected = Cli::try_parse_from(["matrix", "--protocol", "go-away", "--protocol", "capjs"])?;
    assert!(selected.matrix.includes(Protocol::GoAway));
    assert!(selected.matrix.includes(Protocol::Capjs));
    assert!(!selected.matrix.includes(Protocol::Anubis));
    assert!(!selected.matrix.includes(Protocol::Cerberus));
    let linear = Cli::try_parse_from(["matrix", "--protocol", "linear"])?;
    assert!(linear.matrix.includes(Protocol::Linear));
    assert!(!linear.matrix.has_gpu_work());
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
            let result = solver
                .solve::<SOLVE_TYPE_MASK>(&gpu, 0, 0)
                .await?
                .context("matching general SHA-256 digest rejected by GPU")?;
            assert_eq!(result.nonce, counter);
            assert_eq!(result.hash, words);
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
        let salt: Vec<u8> = (0..salt_len)
            .map(|i| i.to_le_bytes()[0].wrapping_mul(47))
            .collect();
        let nonce: Vec<u8> = (0..nonce_len)
            .map(|i| i.to_le_bytes()[0].wrapping_mul(31))
            .collect();
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
        let salt: Vec<u8> = (0..salt_len)
            .map(|i| i.to_le_bytes()[0].wrapping_mul(23))
            .collect();
        let nonce: Vec<u8> = (0..nonce_len)
            .map(|i| i.to_le_bytes()[0].wrapping_mul(43))
            .collect();
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
            let result = solver
                .solve::<SOLVE_TYPE_MASK>(&gpu, 0, 0)
                .await?
                .context("GPU rejected matching full PBKDF2 multi-block prefix")?;
            assert_eq!(result.nonce, counter);
            assert_eq!(result.hash, first);
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

/// Verify both `ROMix` phases and all p lanes against an independent full
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
        let password: Vec<u8> = (0_u8..41).map(|index| index.wrapping_mul(19)).collect();
        let salt: Vec<u8> = (0..(n % 59))
            .map(|index| index.to_le_bytes()[0].wrapping_mul(31))
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
/// and the GPU `ROMix`, not only compute-shader execution.
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
        let salt: Vec<u8> = (0_u8..16).map(|i| i.wrapping_mul(31)).collect();
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

        let gpu_started = Instant::now();
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
        let gpu_ms = gpu_started.elapsed().as_secs_f64() * 1000.0;
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

/// Compare the entire on-device PBKDF2/`ROMix`/PBKDF2 pipeline with the
/// independent RFC7914 `RustCrypto` reference, including multiblock keys,
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
            .map(|i| i.to_le_bytes()[0].wrapping_mul(19))
            .collect::<Vec<_>>();
        let salt = (0..salt_len)
            .map(|i| i.to_le_bytes()[0].wrapping_mul(43))
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
/// p independent `ROMix` lanes are fully GPU-resident. Unlike the older
/// `ROMix`-only matrix, both PBKDF2 stages and final key are on device.
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
            let mut reference_duration = std::time::Duration::ZERO;
            let device_duration;
            let mut run_cpu = || -> Result<()> {
                let start = Instant::now();
                let mut actual = [0_u8; 32];
                scrypt::scrypt(&password, &salt, &params, &mut actual)
                    .map_err(|_| anyhow::anyhow!("reference scrypt failed"))?;
                reference_duration = start.elapsed();
                assert_eq!(actual, reference);
                Ok(())
            };
            if trial.is_multiple_of(2) {
                let start = Instant::now();
                let derived = solver.derive_key(&password, &salt, n, r, p, 32).await?;
                device_duration = start.elapsed();
                assert_eq!(derived, reference, "GPU != CPU RFC7914");
                run_cpu()?;
            } else {
                run_cpu()?;
                let start = Instant::now();
                let derived = solver.derive_key(&password, &salt, n, r, p, 32).await?;
                device_duration = start.elapsed();
                assert_eq!(derived, reference, "GPU != CPU RFC7914");
            }
            println!(
                "{}",
                json!({
                    "n":n, "r":r, "p":p, "trial":trial,
                    "cpu_ns":reference_duration.as_nanos(), "full_gpu_ns":device_duration.as_nanos(),
                    "speedup":reference_duration.as_secs_f64()/device_duration.as_secs_f64(),
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
        let salt: Vec<u8> = (0..salt_len)
            .map(|i| i.to_le_bytes()[0].wrapping_mul(13))
            .collect();
        let lengths = [0_usize, 1, 16, 63, 64, 65, 128];
        let passwords: Vec<Vec<u8>> = (0..candidates)
            .map(|candidate| {
                (0..lengths[candidate % lengths.len()])
                    .map(|i| {
                        i.to_le_bytes()[0]
                            .wrapping_mul(23)
                            .wrapping_add(candidate.to_le_bytes()[0])
                    })
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
    assert_eq!(
        solver.derive_keys(&empty, b"", 16, 8, 1, 32).await?,
        Vec::<Vec<u8>>::new()
    );
    assert!(solver.batch_candidates(16, 3, 8, 1, 32).is_err());
    assert!(solver.batch_candidates(16, 16, 0, 1, 32).is_err());
    assert!(solver.batch_candidates(16, 16, 8, 0, 32).is_err());
    Ok(())
}

/// Optimized release benchmark for independent *candidate passwords*,
/// as opposed to one scrypt password using p internal lanes.
/// Every returned key is compared against `RustCrypto` CPU scrypt.
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
                (0_u8..65)
                    .map(|i| i.wrapping_mul(17).wrapping_add(candidate.to_le_bytes()[0]))
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
            let mut reference_duration = std::time::Duration::ZERO;
            let device_duration;
            let mut cpu = || -> Result<()> {
                let start = Instant::now();
                let mut derived = Vec::with_capacity(count);
                for password in &passwords {
                    let mut key = [0_u8; 32];
                    scrypt::scrypt(password, &salt, &params, &mut key)
                        .map_err(|_| anyhow::anyhow!("reference scrypt failed"))?;
                    derived.push(key.to_vec());
                }
                reference_duration = start.elapsed();
                assert_eq!(derived, expected);
                Ok(())
            };
            if trial.is_multiple_of(2) {
                let start = Instant::now();
                let keys = solver.derive_keys(&refs, &salt, n, r, 1, 32).await?;
                device_duration = start.elapsed();
                assert_eq!(keys, expected);
                cpu()?;
            } else {
                cpu()?;
                let start = Instant::now();
                let keys = solver.derive_keys(&refs, &salt, n, r, 1, 32).await?;
                device_duration = start.elapsed();
                assert_eq!(keys, expected);
            }
            println!(
                "{}",
                json!({
                    "n":n, "r":r, "p":1, "candidates":count, "trial":trial,
                    "cpu_ns":reference_duration.as_nanos(), "gpu_batch_ns":device_duration.as_nanos(),
                    "speedup":reference_duration.as_secs_f64()/device_duration.as_secs_f64(), "verified":true,
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
