#![feature(portable_simd)]
#![cfg_attr(feature = "gpu", recursion_limit = "256")]
#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![doc = "Internal protocol `PoW` solver core for challenge-disasterer."]
#![deny(missing_docs)]
#![deny(unsafe_code)]

use core::num::{NonZeroU8, NonZeroU64};

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "json")]
use serde_json as _;

#[cfg(feature = "gpu")]
/// GPU solver backend using native `wgpu` or browser WebGPU.
pub mod gpu;

#[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "wasm32"))]
cfg_if::cfg_if! {
    if #[cfg(feature = "internals")] {
        /// String manipulation functions
        pub mod strings;
    } else {
        mod strings;
    }
}

/// SHA-256 primitives
mod sha256;

/// BLAKE3 primitives
mod blake3;

/// Message builders
pub mod message;

/// Solvers
pub mod solver;

#[cfg(feature = "adapter")]
/// Data structures and adapters for end-to-end `PoW` solving without IO dependencies.
pub mod adapter;

#[cold]
const fn unlikely() {}

pub(crate) const fn low_u32(value: u64) -> u32 {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

pub(crate) const fn high_u32(value: u64) -> u32 {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[expect(
    clippy::inline_always,
    reason = "byte extraction is an inner-loop primitive in architecture-specific solvers"
)]
#[inline(always)]
pub(crate) const fn read_be_word_byte(words: &[u32; 16], byte_index: usize) -> u8 {
    words[byte_index / 4].to_be_bytes()[byte_index % 4]
}

#[expect(
    clippy::inline_always,
    reason = "byte insertion is an inner-loop primitive in solver kernels"
)]
#[inline(always)]
pub(crate) fn write_be_word_byte(words: &mut [u32; 16], byte_index: usize, byte: u8) {
    let shift = (3 - byte_index % 4) * 8;
    let mask = 0xffu32 << shift;
    let word = &mut words[byte_index / 4];
    *word = (*word & !mask) | (u32::from(byte) << shift);
}

#[expect(
    clippy::inline_always,
    reason = "byte extraction is an inner-loop primitive in solver kernels"
)]
#[inline(always)]
pub(crate) const fn read_le_word_byte(words: &[u32; 16], byte_index: usize) -> u8 {
    words[byte_index / 4].to_le_bytes()[byte_index % 4]
}

#[expect(
    clippy::inline_always,
    reason = "byte insertion is an inner-loop primitive in solver kernels"
)]
#[inline(always)]
pub(crate) fn write_le_word_byte(words: &mut [u32; 16], byte_index: usize, byte: u8) {
    let shift = (byte_index % 4) * 8;
    let mask = 0xffu32 << shift;
    let word = &mut words[byte_index / 4];
    *word = (*word & !mask) | (u32::from(byte) << shift);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
cfg_if::cfg_if! {
    if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))] {
        /// Altcha SHA-256 solver
        pub type AltchaSha256Solver = crate::solver::avx512::AltchaSha256Solver;
    } else {
        /// Altcha SHA-256 solver
        pub type AltchaSha256Solver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::AltchaMessage,
            crate::solver::avx512::AltchaSha256Solver,
            crate::solver::SolverRouter<
                crate::solver::avx2::RequiredFeatures,
                crate::message::AltchaMessage,
                crate::solver::avx2::AltchaSha256Solver,
                crate::solver::safe::AltchaSha256Solver,
            >,
        >;
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
cfg_if::cfg_if! {
    if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))] {
        /// Single block solver
        pub type SingleBlockSolver = crate::solver::avx512::SingleBlockSolver;
        /// Double block solver
        pub type DoubleBlockSolver = crate::solver::avx512::DoubleBlockSolver;
        /// Dynamic dispatching Decimal solver
        pub type DecimalSolver = crate::solver::avx512::DecimalSolver;
        /// Binary solver
        pub type BinarySolver = crate::solver::avx512::BinarySolver;
        /// Go Away Solver
        pub type GoAwaySolver = crate::solver::avx512::GoAwaySolver;
        /// Cerberus solver
        pub type CerberusSolver = crate::solver::avx512::CerberusSolver;
        /// Solver name
        pub const SOLVER_NAME: &str = "AVX-512";
    } else if #[cfg(target_feature = "sha")] {
        /// Single block solver
        pub type SingleBlockSolver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::SingleBlockMessage,
            crate::solver::avx512::SingleBlockSolver,
            crate::solver::sha_ni::SingleBlockSolver,
        >;
        /// Double block solver
        pub type DoubleBlockSolver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::DoubleBlockMessage,
            crate::solver::avx512::DoubleBlockSolver,
            crate::solver::sha_ni::DoubleBlockSolver,
        >;
        /// Dynamic dispatching Decimal solver
        pub type DecimalSolver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::DecimalMessage,
            crate::solver::avx512::DecimalSolver,
            crate::solver::sha_ni::DecimalSolver,
        >;
        /// Binary solver
        pub type BinarySolver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::BinaryMessage,
            crate::solver::avx512::BinarySolver,
            crate::solver::SolverRouter<
                crate::solver::avx2::RequiredFeatures,
                crate::message::BinaryMessage,
                crate::solver::avx2::BinarySolver,
                crate::solver::safe::BinarySolver,
            >,
        >;
        /// Go away solver
        pub type GoAwaySolver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::GoAwayMessage,
            crate::solver::avx512::GoAwaySolver,
            crate::solver::sha_ni::GoAwaySolver,
        >;
        /// Cerberus solver
        pub type CerberusSolver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::CerberusMessage,
            crate::solver::avx512::CerberusSolver,
            crate::solver::SolverRouter<
                crate::solver::avx2::RequiredFeatures,
                crate::message::CerberusMessage,
                crate::solver::avx2::CerberusSolver,
                crate::solver::safe::CerberusSolver,
            >,
        >;
        /// Solver name
        pub const SOLVER_NAME: &str = "SHA-NI";
    } else if #[cfg(target_feature = "avx2")] {
        /// Single block solver
        pub type SingleBlockSolver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::SingleBlockMessage,
            crate::solver::avx512::SingleBlockSolver,
            crate::solver::avx2::SingleBlockSolver,
        >;
        /// Double block solver
        pub type DoubleBlockSolver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::DoubleBlockMessage,
            crate::solver::avx512::DoubleBlockSolver,
            crate::solver::avx2::DoubleBlockSolver,
        >;
        /// Dynamic dispatching Decimal solver
        pub type DecimalSolver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::DecimalMessage,
            crate::solver::avx512::DecimalSolver,
            crate::solver::avx2::DecimalSolver,
        >;
        /// Binary solver
        pub type BinarySolver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::BinaryMessage,
            crate::solver::avx512::BinarySolver,
            crate::solver::avx2::BinarySolver,
        >;
        /// Go away solver
        pub type GoAwaySolver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::GoAwayMessage,
            crate::solver::avx512::GoAwaySolver,
            crate::solver::avx2::GoAwaySolver,
        >;
        /// Cerberus solver
        pub type CerberusSolver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::CerberusMessage,
            crate::solver::avx512::CerberusSolver,
            crate::solver::avx2::CerberusSolver,
        >;
        /// Solver name
        pub const SOLVER_NAME: &str = "AVX2";
    } else {
        /// Single block solver
        pub type SingleBlockSolver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::SingleBlockMessage,
            crate::solver::avx512::SingleBlockSolver,
            crate::solver::SolverRouter<
                crate::solver::sha_ni::RequiredFeatures,
                crate::message::SingleBlockMessage,
                crate::solver::sha_ni::SingleBlockSolver,
                crate::solver::SolverRouter<
                    crate::solver::avx2::RequiredFeatures,
                    crate::message::SingleBlockMessage,
                    crate::solver::avx2::SingleBlockSolver,
                    crate::solver::safe::SingleBlockSolver,
                >,
            >,
        >;
        /// Double block solver
        pub type DoubleBlockSolver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::DoubleBlockMessage,
            crate::solver::avx512::DoubleBlockSolver,
            crate::solver::SolverRouter<
                crate::solver::sha_ni::RequiredFeatures,
                crate::message::DoubleBlockMessage,
                crate::solver::sha_ni::DoubleBlockSolver,
                crate::solver::SolverRouter<
                    crate::solver::avx2::RequiredFeatures,
                    crate::message::DoubleBlockMessage,
                    crate::solver::avx2::DoubleBlockSolver,
                    crate::solver::safe::DoubleBlockSolver,
                >,
            >,
        >;
        /// Dynamic dispatching Decimal solver
        pub type DecimalSolver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::DecimalMessage,
            crate::solver::avx512::DecimalSolver,
            crate::solver::SolverRouter<
                crate::solver::sha_ni::RequiredFeatures,
                crate::message::DecimalMessage,
                crate::solver::sha_ni::DecimalSolver,
                crate::solver::SolverRouter<
                    crate::solver::avx2::RequiredFeatures,
                    crate::message::DecimalMessage,
                    crate::solver::avx2::DecimalSolver,
                    crate::solver::safe::DecimalSolver,
                >,
            >,
        >;
        /// Binary solver
        pub type BinarySolver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::BinaryMessage,
            crate::solver::avx512::BinarySolver,
            crate::solver::SolverRouter<
                crate::solver::avx2::RequiredFeatures,
                crate::message::BinaryMessage,
                crate::solver::avx2::BinarySolver,
                crate::solver::safe::BinarySolver,
            >,
        >;
        /// Go away solver
        pub type GoAwaySolver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::GoAwayMessage,
            crate::solver::avx512::GoAwaySolver,
            crate::solver::SolverRouter<
                crate::solver::sha_ni::RequiredFeatures,
                crate::message::GoAwayMessage,
                crate::solver::sha_ni::GoAwaySolver,
                crate::solver::SolverRouter<
                    crate::solver::avx2::RequiredFeatures,
                    crate::message::GoAwayMessage,
                    crate::solver::avx2::GoAwaySolver,
                    crate::solver::safe::GoAwaySolver,
                >,
            >,
        >;
        /// Cerberus solver
        pub type CerberusSolver = crate::solver::SolverRouter<
            crate::solver::avx512::RequiredFeatures,
            crate::message::CerberusMessage,
            crate::solver::avx512::CerberusSolver,
            crate::solver::SolverRouter<
                crate::solver::avx2::RequiredFeatures,
                crate::message::CerberusMessage,
                crate::solver::avx2::CerberusSolver,
                crate::solver::safe::CerberusSolver,
            >,
        >;
        /// Solver name
        pub const SOLVER_NAME: &str = "Fallback";
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
cfg_if::cfg_if! {
    if #[cfg(target_arch = "wasm32")] {
        /// Single block solver
        pub type SingleBlockSolver = crate::solver::simd128::SingleBlockSolver;
        /// Double block solver
        pub type DoubleBlockSolver = crate::solver::simd128::DoubleBlockSolver;
        /// Dynamic dispatching Decimal solver
        pub type DecimalSolver = crate::solver::simd128::DecimalSolver;
        /// Go away solver
        pub type GoAwaySolver = crate::solver::simd128::GoAwaySolver;
        /// Binary solver
        pub type BinarySolver = crate::solver::safe::BinarySolver;
        /// Cerberus solver
        pub type CerberusSolver = crate::solver::simd128::CerberusSolver;
        /// Solver name
        pub const SOLVER_NAME: &str = "SIMD128";
        /// Blake3 solver name
        pub const BLAKE3_SOLVER_NAME: &str = "SIMD128";
    } else {
        /// Single block solver
        pub type SingleBlockSolver = crate::solver::safe::SingleBlockSolver;
        /// Double block solver
        pub type DoubleBlockSolver = crate::solver::safe::DoubleBlockSolver;
        /// Dynamic dispatching Decimal solver
        pub type DecimalSolver = crate::solver::safe::DecimalSolver;
        /// Go away solver
        pub type GoAwaySolver = crate::solver::safe::GoAwaySolver;
        /// Binary solver
        pub type BinarySolver = crate::solver::safe::BinarySolver;
        /// Cerberus solver
        pub type CerberusSolver = crate::solver::safe::CerberusSolver;
        /// Solver name
        pub const SOLVER_NAME: &str = "Fallback";
        /// Blake3 solver name
        pub const BLAKE3_SOLVER_NAME: &str = "Fallback";
    }
}

/// Build a prefix for mCaptcha `PoW`
pub fn build_mcaptcha_prefix<E: Extend<u8>>(out: &mut E, string: &str, salt: &str) -> Option<()> {
    let string_len = u64::try_from(string.len()).ok()?;
    out.extend(salt.as_bytes().iter().copied());
    out.extend(string_len.to_le_bytes());
    out.extend(string.as_bytes().iter().copied());
    Some(())
}

/// Compute a plausible time for a SHA-256 `PoW` to prevent tainting metrics
///
/// # Arguments
///
/// * `hashes`: The number of hashes that have been computed
///
/// # Returns
///
/// The plausible time in milliseconds
#[expect(
    dead_code,
    reason = "kept as the SHA-256 counterpart to the active plausibility estimators"
)]
pub(crate) const fn compute_plausible_time_sha256(hashes: u64) -> u64 {
    hashes / 512
}

/// Compute the target for an mCaptcha `PoW`
#[must_use]
pub const fn compute_target_mcaptcha(difficulty_factor: NonZeroU64) -> u64 {
    let difficulty_factor = difficulty_factor.get();
    u64::MAX - u64::MAX / difficulty_factor
}

/// Compute a leading-zero-nibble SHA-256 mask (mask & (V\[0] << 32 | V\[1]) == 0)
#[must_use]
pub const fn compute_mask_zero_nibbles(difficulty_factor: NonZeroU8) -> u64 {
    let bits = difficulty_factor.get().saturating_mul(4);
    if bits >= 64 {
        u64::MAX
    } else {
        !(!0u64 >> bits)
    }
}

/// Test the full SHA-256 digest against go-away's leading-zero-bit rule.
///
/// Difficulties above 256 are unsatisfiable because SHA-256 has only 256 bits.
#[must_use]
pub fn goaway_hash_matches_difficulty(hash: &[u32; 8], difficulty: u16) -> bool {
    hash_has_leading_zero_bits(hash, u64::from(difficulty))
}

/// Test whether a SHA-256 digest has at least `bits` leading zero bits.
///
/// Values above the 256-bit digest width are unsatisfiable. This is shared by
/// protocols such as go-away and haphash whose verifier difficulty spans the
/// complete digest rather than only the first comparison word.
#[must_use]
pub fn hash_has_leading_zero_bits(hash: &[u32; 8], bits: u64) -> bool {
    if bits > 256 {
        return false;
    }

    let full_words = usize::try_from(bits / 32).unwrap_or(8);
    if hash[..full_words].iter().any(|word| *word != 0) {
        return false;
    }

    let remaining = u32::try_from(bits % 32).unwrap_or(31);
    remaining == 0 || hash[full_words].leading_zeros() >= remaining
}

/// Compute the mask for a `GoAway` `PoW` against the first 64 SHA-256 bits.
#[must_use]
pub const fn compute_mask_goaway(difficulty_factor: NonZeroU8) -> u64 {
    let bits = difficulty_factor.get();
    if bits >= 64 {
        u64::MAX
    } else {
        !(!0u64 >> bits)
    }
}

/// Compute a mask for a Cerberus `PoW` (mask & (V\[0] << 32) == 0)
#[must_use]
pub const fn compute_mask_cerberus(difficulty_factor: NonZeroU8) -> u64 {
    let doubled = difficulty_factor.get().saturating_mul(2);
    let bits = if doubled > 32 { 32 } else { doubled };
    let mask = if bits == 32 {
        u32::MAX
    } else {
        !(!0u32 >> bits)
    };
    let [b0, b1, b2, b3] = mask.swap_bytes().to_be_bytes();
    u64::from_be_bytes([b0, b1, b2, b3, 0, 0, 0, 0])
}

/// Extract top 128 bits from a 64-bit word array
#[must_use]
pub const fn extract128_be(inp: [u32; 8]) -> u128 {
    (inp[0] as u128) << 96 | (inp[1] as u128) << 64 | (inp[2] as u128) << 32 | (inp[3] as u128)
}

/// Encode a sha-256 hash into hex
pub fn encode_hex(out: &mut [u8; 64], inp: [u32; 8]) {
    for w in 0..8 {
        let be_bytes = inp[w].to_be_bytes();
        be_bytes.iter().enumerate().for_each(|(i, b)| {
            let high_nibble = b >> 4;
            let low_nibble = b & 0xf;
            out[w * 8 + i * 2] = if high_nibble < 10 {
                high_nibble + b'0'
            } else {
                high_nibble + b'a' - 10
            };
            out[w * 8 + i * 2 + 1] = if low_nibble < 10 {
                low_nibble + b'0'
            } else {
                low_nibble + b'a' - 10
            };
        });
    }
}

/// Encode a blake3 hash into hex
pub fn encode_hex_le(out: &mut [u8; 64], inp: [u32; 8]) {
    for w in 0..8 {
        let be_bytes = inp[w].to_le_bytes();
        be_bytes.iter().enumerate().for_each(|(i, b)| {
            let high_nibble = b >> 4;
            let low_nibble = b & 0xf;
            out[w * 8 + i * 2] = if high_nibble < 10 {
                high_nibble + b'0'
            } else {
                high_nibble + b'a' - 10
            };
            out[w * 8 + i * 2 + 1] = if low_nibble < 10 {
                low_nibble + b'0'
            } else {
                low_nibble + b'a' - 10
            };
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nonzero(value: u8) -> anyhow::Result<NonZeroU8> {
        NonZeroU8::new(value).ok_or_else(|| anyhow::anyhow!("expected nonzero test value"))
    }

    pub fn build_prefix_official<W: std::io::Write>(
        out: &mut W,
        string: &str,
        salt: &str,
    ) -> std::io::Result<()> {
        out.write_all(salt.as_bytes())?;
        if let Err(error) = bincode::serialize_into(out, string) {
            return match *error {
                bincode::ErrorKind::Io(error) => Err(error),
                other => Err(std::io::Error::other(other.to_string())),
            };
        }
        Ok(())
    }

    #[test]
    fn test_encode_hex() {
        let mut out = [0u8; 64];
        encode_hex(
            &mut out,
            [
                0x1234_5678,
                0x9abc_def0,
                0x1234_5678,
                0x9abc_def0,
                0x1234_5678,
                0x9abc_def0,
                0x1234_5678,
                0x9abc_def0,
            ],
        );
        assert_eq!(
            &out,
            b"123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0"
        );
    }

    #[test]
    fn test_compute_mask_zero_nibbles() -> anyhow::Result<()> {
        assert_eq!(
            compute_mask_zero_nibbles(nonzero(1)?),
            0xfu64.reverse_bits(),
        );
        assert_eq!(
            compute_mask_zero_nibbles(nonzero(2)?),
            0xffu64.reverse_bits(),
        );
        assert_eq!(
            compute_mask_zero_nibbles(nonzero(3)?),
            0xfffu64.reverse_bits(),
        );
        assert_eq!(compute_mask_zero_nibbles(nonzero(16)?), u64::MAX,);
        Ok(())
    }

    #[test]
    fn test_bincode_string_serialize() -> anyhow::Result<()> {
        let string = "hello";
        let mut homegrown = Vec::new();
        build_mcaptcha_prefix(&mut homegrown, string, "z")
            .ok_or_else(|| anyhow::anyhow!("small test string length is not representable"))?;
        let mut official = Vec::new();
        build_prefix_official(&mut official, string, "z")?;
        assert_eq!(homegrown, official);
        Ok(())
    }

    #[test]
    fn test_cerberus_mask() -> anyhow::Result<()> {
        fn check_small(hash: &[u8; 32], n: u32) -> bool {
            // <https://github.com/sjtug/cerberus/blob/ee8f903f1311da7022aec68c8686739b40f4a168/pow/src/check_dubit.rs>
            let first_word = u32::from(hash[0]) << 24
                | u32::from(hash[1]) << 16
                | u32::from(hash[2]) << 8
                | u32::from(hash[3]);
            first_word.leading_zeros() >= n * 2
        }

        for i in 1_u8..8 {
            let mask = compute_mask_cerberus(nonzero(i)?);
            eprintln!("mask: {mask:016x}");
        }

        // hash[0] is the LSB of the H0 register

        let mask = compute_mask_cerberus(nonzero(7)?);
        let mask_hi = u32::try_from(mask >> 32)?;
        let mask_lo = u32::try_from(mask & u64::from(u32::MAX))?;
        assert_eq!(mask_lo, 0);
        let hash_partial = (!mask_hi).to_le_bytes();
        eprintln!("hash_partial: {:02x?}", hash_partial);
        let mut test = [0; 32];
        test[..4].copy_from_slice(&hash_partial);
        assert!(check_small(&test, 7));
        Ok(())
    }
}

#[cfg(test)]
mod challenge_domain_tests {
    use super::*;

    fn nonzero(value: u8) -> anyhow::Result<NonZeroU8> {
        NonZeroU8::new(value).ok_or_else(|| anyhow::anyhow!("expected nonzero test value"))
    }

    #[test]
    fn goaway_mask_covers_full_u64_domain() -> anyhow::Result<()> {
        assert_eq!(compute_mask_goaway(nonzero(1)?), 1_u64 << 63);
        assert_eq!(compute_mask_goaway(nonzero(33)?), u64::MAX << 31);
        assert_eq!(compute_mask_goaway(nonzero(64)?), u64::MAX);
        Ok(())
    }

    #[test]
    fn goaway_full_digest_predicate_covers_zero_through_256_bits() {
        let all_zero = [0_u32; 8];
        assert!(goaway_hash_matches_difficulty(&all_zero, 0));
        assert!(goaway_hash_matches_difficulty(&all_zero, 256));
        assert!(!goaway_hash_matches_difficulty(&all_zero, 257));

        let hash = [
            0,
            0,
            0x07ff_ffff,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
        ];
        assert!(goaway_hash_matches_difficulty(&hash, 69));
        assert!(!goaway_hash_matches_difficulty(&hash, 70));
    }

    #[test]
    fn cerberus_mask_clamps_to_the_upstream_32_bit_check() -> anyhow::Result<()> {
        let full = compute_mask_cerberus(nonzero(16)?);
        assert_eq!(full, u64::from(u32::MAX) << 32);
        assert_eq!(compute_mask_cerberus(nonzero(17)?), full);
        assert_eq!(compute_mask_cerberus(nonzero(u8::MAX)?), full);
        Ok(())
    }
}
