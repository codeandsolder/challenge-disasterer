use crate::{Align16, Align64};

pub(crate) static DIGIT_LUT_10000_LE_EVEN: Align64<[u32; 10000 / 2]> = const {
    let mut out = [0; 10000 / 2];
    let mut i = 0;
    while i < 10000 / 2 {
        let mut copy = i * 2;
        let mut ds = [b'0'; 4];
        let mut j = 0;
        while j < 4 {
            ds[j] = (copy % 10) as u8 + b'0';
            copy /= 10;
            j += 1;
        }
        out[i] = u32::from_be_bytes(ds);
        i += 1;
    }
    Align64(out)
};

#[expect(dead_code)]
mod static_asserts {
    use super::DIGIT_LUT_10000_LE_EVEN;

    const ASSERT_DIGIT_LUT_10000_LE_0: [(); 1] =
        [(); (DIGIT_LUT_10000_LE_EVEN.0[0] == u32::from_be_bytes(*b"0000")) as usize];

    const ASSERT_DIGIT_LUT_10000_LE_1: [(); 1] =
        [(); (DIGIT_LUT_10000_LE_EVEN.0[1] == u32::from_be_bytes(*b"2000")) as usize];

    const ASSERT_DIGIT_LUT_10000_LE_5: [(); 1] =
        [(); (DIGIT_LUT_10000_LE_EVEN.0[5] == u32::from_be_bytes(*b"0100")) as usize];

    const ASSERT_DIGIT_LUT_10000_LE_1234: [(); 1] =
        [(); (DIGIT_LUT_10000_LE_EVEN.0[1234] == u32::from_be_bytes(*b"8642")) as usize];
}

/// Convert up to 8 decimal digits to ASCII.
///
/// `REGISTER_BSWAP` reverses each 32-bit half of the eight-byte output, which is
/// the layout consumed by the SHA solver backends. Values wider than `N` digits
/// intentionally keep their least-significant `N` digits, matching the historical
/// SIMD formatter.
#[inline(always)]
pub fn format_decimal_8<const N: usize, const REGISTER_BSWAP: bool, const PLACEHOLDER: u8>(
    out: &mut Align16<[u8; 8]>,
    mut input: u32,
) {
    debug_assert!(N <= 8);
    out.fill(PLACEHOLDER);
    for i in (0..N).rev() {
        out[i] = (input % 10) as u8 + b'0';
        input /= 10;
    }
    if REGISTER_BSWAP {
        out[..4].reverse();
        out[4..].reverse();
    }
}

#[cfg(not(target_feature = "avx512f"))]
#[inline(always)]
/// Convert up to 7 digits to ASCII
///
/// Parameters:
/// - REGISTER_BSWAP: Swap 32-bit register bytes order
/// - PLACEHOLDER: The placeholder character to use for the rest of the bytes
/// - ADDEND: The byte offset to add to each digit (1 -> use digits 1-8, 0 -> use digits 0-7)
pub fn to_octal_7<const REGISTER_BSWAP: bool, const PLACEHOLDER: u8, const ADDEND: u8>(
    out: &mut Align16<[u8; 8]>,
    mut input: u32,
) {
    out.fill(PLACEHOLDER);
    for i in (0..7).rev() {
        out[i] = ((input & 0b111) as u8) + b'0' + ADDEND;
        input >>= 3;
    }
    if REGISTER_BSWAP {
        let mut out_ptr = out.as_mut_ptr().cast::<u32>();
        unsafe {
            *out_ptr = (*out_ptr).swap_bytes();
            out_ptr = out_ptr.add(1);
            *out_ptr = (*out_ptr).swap_bytes();
        }
    }
}

#[cfg(target_feature = "avx512f")]
#[inline(always)]
/// Convert up to 7 digits to ASCII
///
/// Parameters:
/// - REGISTER_BSWAP: Swap 32-bit register bytes order
/// - PLACEHOLDER: The placeholder character to use for the rest of the bytes
/// - ADDEND: The byte offset to add to each digit (1 -> use digits 1-8, 0 -> use digits 0-7)
pub fn to_octal_7<const REGISTER_BSWAP: bool, const PLACEHOLDER: u8, const ADDEND: u8>(
    out: &mut Align16<[u8; 8]>,
    input: u32,
) {
    use core::arch::x86_64::*;

    unsafe {
        let mut x = _mm256_set1_epi32(input as _);
        if REGISTER_BSWAP {
            x = _mm256_srlv_epi32(x, _mm256_setr_epi32(9, 12, 15, 18, 32, 0, 3, 6));
        } else {
            x = _mm256_srlv_epi32(x, _mm256_setr_epi32(18, 15, 12, 9, 6, 3, 0, 32));
        }
        let d = _mm256_cvtepi32_epi8(x);
        let mut val = _mm_cvtsi128_si64(d) as u64;
        val &= u64::from_le_bytes([0b111; 8]);
        val += const {
            if REGISTER_BSWAP {
                u64::from_le_bytes([0, 0, 0, 0, PLACEHOLDER - (b'0' + ADDEND), 0, 0, 0])
                    + u64::from_le_bytes([b'0' + ADDEND; 8])
            } else {
                u64::from_le_bytes([0, 0, 0, 0, 0, 0, 0, PLACEHOLDER - (b'0' + ADDEND)])
                    + u64::from_le_bytes([b'0' + ADDEND; 8])
            }
        };
        out.as_mut_ptr().cast::<u64>().write(val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_octal_7() {
        let i = 0o1234567;
        let mut buf = Align16([0u8; 8]);
        to_octal_7::<false, 0x80, 0>(&mut buf, i);
        assert_eq!(buf, Align16(*b"1234567\x80"));
        to_octal_7::<true, 0x80, 0>(&mut buf, i);
        assert_eq!(buf, Align16(*b"4321\x80765"));
        to_octal_7::<false, 0x80, 1>(&mut buf, i);
        assert_eq!(buf, Align16(*b"2345678\x80"));
        to_octal_7::<true, 0x80, 1>(&mut buf, i);
        assert_eq!(buf, Align16(*b"5432\x80876"));
    }

    #[test]
    fn test_format_decimal_8() {
        let mut buf = Align16([0u8; 8]);
        format_decimal_8::<8, false, 0x80>(&mut buf, 12345678);
        assert_eq!(buf, Align16(*b"12345678"));

        format_decimal_8::<8, false, 0x80>(&mut buf, 1_0000_0000);
        assert_eq!(buf, Align16(*b"00000000"));

        format_decimal_8::<7, false, 0x80>(&mut buf, 1234567);
        assert_eq!(buf, Align16(*b"1234567\x80"));

        format_decimal_8::<7, false, 0x80>(&mut buf, 1000_0000);
        assert_eq!(buf, Align16(*b"0000000\x80"));

        format_decimal_8::<8, true, 0x80>(&mut buf, 12345678);
        assert_eq!(buf, Align16(*b"43218765"));

        format_decimal_8::<7, true, 0x80>(&mut buf, 1234567);
        assert_eq!(buf, Align16(*b"4321\x80765"));
    }
}
