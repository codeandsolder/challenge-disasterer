// Keep this AVX2 core architecture-specific. The portable-SIMD refactor in
// 3c8d0e3 regressed high-constant Cerberus decimal throughput by about 7% on
// Zen 2; holding the current solver constant, the affected optimized
// specialization is about 24% larger with the portable wrapper. Recheck the
// retained preexisting-baseline A/B before
// routing this implementation through a generic core again.
use super::*;
use core::arch::x86_64::*;

macro_rules! mm256_rorx_epi32 {
    ($x:expr, $shift:expr) => {{
        #[allow(unused_unsafe)]
        // SAFETY: every expansion is reached only from the AVX2-dispatched
        // compression path, so the required target feature is active.
        unsafe {
            _mm256_xor_si256(
                _mm256_srli_epi32($x, $shift),
                _mm256_slli_epi32($x, 32 - $shift),
            )
        }
    }};
}

#[inline(always)]
fn g4(
    va: &mut __m256i,
    vb: &mut __m256i,
    vc: &mut __m256i,
    vd: &mut __m256i,
    x: __m256i,
    y: __m256i,
) {
    // SAFETY: all call sites are inside the AVX2-dispatched compression path;
    // the operands are initialized SIMD values and these intrinsics do not
    // dereference memory.
    unsafe {
        *va = _mm256_add_epi32(*va, _mm256_add_epi32(*vb, x));
        *vd = _mm256_xor_si256(*vd, *va);
        *vd = mm256_rorx_epi32!(*vd, 16);
        *vc = _mm256_add_epi32(*vc, *vd);
        *vb = _mm256_xor_si256(*vb, *vc);
        *vb = mm256_rorx_epi32!(*vb, 12);
        *va = _mm256_add_epi32(*va, _mm256_add_epi32(*vb, y));
        *vd = _mm256_xor_si256(*vd, *va);
        *vd = mm256_rorx_epi32!(*vd, 8);
        *vc = _mm256_add_epi32(*vc, *vd);
        *vb = _mm256_xor_si256(*vb, *vc);
        *vb = mm256_rorx_epi32!(*vb, 7);
    }
}

#[inline(always)]
pub(crate) fn compress_mb8<const CONSTANT_WORD_COUNT: usize, const PATCH_1: usize>(
    v: &mut [__m256i; 16],
    block_template: &[u32; 16],
    patch_1: __m256i,
) {
    // SAFETY: this helper is only called from AVX2-dispatched solver paths.
    // Every literal [a, b, c, d] tuple below contains four distinct indices,
    // so get_disjoint_unchecked_mut never creates aliased mutable references.
    unsafe {
        repeat7!(i, {
            macro_rules! g4 {
                ($f:ident; $a:literal, $b:literal, $c:literal, $d:literal, $x:literal, $y:literal) => {{
                    let [va, vb, vc, vd] = v.get_disjoint_unchecked_mut([$a, $b, $c, $d]);
                    let ix = MESSAGE_SCHEDULE[i][$x];
                    let iy = MESSAGE_SCHEDULE[i][$y];
                    $f(
                        va,
                        vb,
                        vc,
                        vd,
                        if ix == PATCH_1 {
                            patch_1
                        } else {
                            _mm256_set1_epi32(block_template[ix] as _)
                        },
                        if iy == PATCH_1 {
                            patch_1
                        } else {
                            _mm256_set1_epi32(block_template[iy] as _)
                        },
                    );
                }};
                ($a:literal, $b:literal, $c:literal, $d:literal, $x:literal, $y:literal) => {{
                    g4!(g4; $a, $b, $c, $d, $x, $y);
                }};
            }
            if i > 0 || CONSTANT_WORD_COUNT < 2 {
                g4!(0, 4, 8, 12, 0, 1);
            }
            if i > 0 || CONSTANT_WORD_COUNT < 4 {
                g4!(1, 5, 9, 13, 2, 3);
            }
            if i > 0 || CONSTANT_WORD_COUNT < 6 {
                g4!(2, 6, 10, 14, 4, 5);
            }
            if i > 0 || CONSTANT_WORD_COUNT < 8 {
                g4!(3, 7, 11, 15, 6, 7);
            }
            if i > 0 || CONSTANT_WORD_COUNT < 10 {
                g4!(0, 5, 10, 15, 8, 9);
            }
            if i > 0 || CONSTANT_WORD_COUNT < 12 {
                g4!(1, 6, 11, 12, 10, 11);
            }
            if i > 0 || CONSTANT_WORD_COUNT < 14 {
                g4!(2, 7, 8, 13, 12, 13);
            }
            if i > 0 || CONSTANT_WORD_COUNT < 16 {
                g4!(3, 4, 9, 14, 14, 15);
            }
        });

        repeat8!(i, {
            v[i] = _mm256_xor_si256(v[i], v[i + 8]);
        });
    }
}

#[inline(always)]
pub(crate) fn compress_simd8<const CONSTANT_WORD_COUNT: usize, const PATCH_1: usize>(
    state: &mut [core::simd::u32x8; 16],
    block_template: &[u32; 16],
    patch: core::simd::u32x8,
) {
    let mut raw: [__m256i; 16] = core::array::from_fn(|i| state[i].into());
    compress_mb8::<CONSTANT_WORD_COUNT, PATCH_1>(&mut raw, block_template, patch.into());
    for i in 0..16 {
        state[i] = raw[i].into();
    }
}

#[cfg(target_feature = "avx2")]
#[cfg(test)]
mod tests {
    use super::*;
    use blake3::Hasher;
    use core::simd::u32x8;

    fn assert_matches_portable<const CONSTANT_WORD_COUNT: usize, const PATCH: usize>() {
        let block = core::array::from_fn(|word| {
            0x1020_3040u32
                .wrapping_add((word as u32).wrapping_mul(0x0102_0305))
                .rotate_left(word as u32)
        });
        let initial: [u32x8; 16] = core::array::from_fn(|word| {
            u32x8::from_array(core::array::from_fn(|lane| {
                0x6a09_e667u32
                    .wrapping_add((word as u32).wrapping_mul(0x9e37_79b9))
                    .wrapping_add((lane as u32).wrapping_mul(0x85eb_ca6b))
            }))
        });
        let patch = u32x8::from_array(core::array::from_fn(|lane| {
            0xa5a5_0000u32.wrapping_add((lane as u32).wrapping_mul(0x1f1f_0101))
        }));

        let mut expected = initial;
        crate::blake3::portable::compress::<8, CONSTANT_WORD_COUNT, PATCH>(
            &mut expected,
            &block,
            patch,
        );

        let mut actual: [__m256i; 16] = core::array::from_fn(|word| initial[word].into());
        compress_mb8::<CONSTANT_WORD_COUNT, PATCH>(&mut actual, &block, patch.into());

        for word in 0..16 {
            let actual_word: u32x8 = actual[word].into();
            assert_eq!(
                actual_word.to_array(),
                expected[word].to_array(),
                "word {word}, constant words {CONSTANT_WORD_COUNT}, patch {PATCH}"
            );
        }
    }

    #[test]
    fn test_g_function() {
        let mut state = core::array::from_fn(|i| crate::sha256::IV[i % 8].wrapping_add(i as u32));
        let mut state_v: [__m256i; 16] = core::array::from_fn(|i| u32x8::splat(state[i]).into());
        g(
            &mut state,
            0,
            4,
            8,
            12,
            crate::sha256::IV[0],
            crate::sha256::IV[1],
        );
        let [va, vb, vc, vd] = state_v.get_disjoint_mut([0, 4, 8, 12]).unwrap();
        g4(
            va,
            vb,
            vc,
            vd,
            u32x8::splat(crate::sha256::IV[0]).into(),
            u32x8::splat(crate::sha256::IV[1]).into(),
        );

        for i in 0..16 {
            let actual: u32x8 = state_v[i].into();
            assert_eq!(
                actual[0], state[i],
                "word {}: expected: {:08x}, results: {:08x}",
                i, state[i], actual[0]
            );
        }
    }

    #[test]
    fn test_compress_mb8() {
        let mut v = [0u32; 16];
        v[..8].copy_from_slice(&crate::blake3::IV);
        v[8..12].copy_from_slice(&crate::blake3::IV[..4]);
        v[12] = 0;
        v[13] = 0;
        v[14] = 4;
        v[15] = 0x0b;
        let mut v: [__m256i; 16] = core::array::from_fn(|i| u32x8::splat(v[i]).into());
        let mut block = [0u32; 16];
        block[0] = u32::from_le_bytes(*b"IETF");
        compress_mb8::<0, 8>(&mut v, &block, u32x8::splat(0).into());
        let expected = [
            0x1edea283, 0xabe6f4e6, 0x24896868, 0xcfc04e8f, 0x9470c54c, 0xff82a646, 0xd6b4cbd1,
            0xe2815116,
        ];
        let mut results = [0u32; 8];
        for i in 0..8 {
            let word: u32x8 = v[i].into();
            results[i] = word[0];
        }
        assert_eq!(
            results, expected,
            "expected: {:08x?}, results: {:08x?}",
            expected, results
        );
        let mut hasher = Hasher::new();
        hasher.update(b"IETF");
        let hash = hasher.finalize();
        let hash = hash.as_bytes();
        let mut expected = [0u32; 8];
        for i in 0..8 {
            expected[i] = u32::from_le_bytes(hash[i * 4..i * 4 + 4].try_into().unwrap());
        }
        assert_eq!(results, expected);
    }

    #[test]
    fn test_high_constant_word_specializations_match_portable() {
        // These are representative Cerberus decimal specializations near the
        // high-constant end where the architecture-specific implementation
        // materially outperforms the portable-SIMD wrapper.
        assert_matches_portable::<13, 14>();
        assert_matches_portable::<14, 15>();
        assert_matches_portable::<15, 16>();
    }
}
