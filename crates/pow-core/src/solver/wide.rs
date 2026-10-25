use core::simd::{
    Simd,
    cmp::{SimdPartialEq as _, SimdPartialOrd as _},
};

use crate::Align64;

#[inline(always)]
fn target_match<const LANES: usize, const TYPE: u8>(
    a: Simd<u32, LANES>,
    b: Simd<u32, LANES>,
    target: u64,
    mask: u64,
) -> core::simd::Mask<i32, LANES> {
    #[cfg(not(feature = "compare-64bit"))]
    let _ = b;

    let target = target & mask;
    let target_hi = (target >> 32) as u32;
    let mask_hi = (mask >> 32) as u32;

    #[cfg(not(feature = "compare-64bit"))]
    {
        if TYPE == crate::solver::SOLVE_TYPE_GT {
            a.simd_gt(Simd::splat(target_hi))
        } else if TYPE == crate::solver::SOLVE_TYPE_LT {
            a.simd_lt(Simd::splat(target_hi))
        } else {
            (a & Simd::splat(mask_hi)).simd_eq(Simd::splat(target_hi & mask_hi))
        }
    }

    #[cfg(feature = "compare-64bit")]
    {
        let target_lo = target as u32;
        let mask_lo = mask as u32;
        let hi_eq = a.simd_eq(Simd::splat(target_hi));
        if TYPE == crate::solver::SOLVE_TYPE_GT {
            a.simd_gt(Simd::splat(target_hi)) | (hi_eq & b.simd_gt(Simd::splat(target_lo)))
        } else if TYPE == crate::solver::SOLVE_TYPE_LT {
            a.simd_lt(Simd::splat(target_hi)) | (hi_eq & b.simd_lt(Simd::splat(target_lo)))
        } else {
            (a & Simd::splat(mask_hi)).simd_eq(Simd::splat(target_hi & mask_hi))
                & (b & Simd::splat(mask_lo)).simd_eq(Simd::splat(target_lo & mask_lo))
        }
    }
}

pub(crate) struct BinaryWork<'a> {
    pub(crate) attempted_nonces: &'a mut u64,
    pub(crate) limit: u64,
    pub(crate) prefix_state: [u32; 8],
    pub(crate) first_block: &'a Align64<[u32; 16]>,
    pub(crate) second_block_schedule: &'a [u32; 64],
    pub(crate) nonce_byte_offset: usize,
    pub(crate) nonce_byte_count: core::num::NonZeroU8,
}

#[inline(always)]
pub(crate) fn solve_binary_impl<
    const LANES: usize,
    const TYPE: u8,
    const FIRST_NONCE_WORD_IDX: usize,
    const NEED_SECOND_BLOCK: bool,
>(
    work: BinaryWork<'_>,
    target: u64,
    mask: u64,
) -> Option<u64> {
    let BinaryWork {
        attempted_nonces,
        limit,
        prefix_state,
        first_block,
        second_block_schedule,
        nonce_byte_offset,
        nonce_byte_count,
    } = work;
    let target = target & mask;
    let lane_shift = (3 - nonce_byte_offset % 4) * 8;
    let lane_ids =
        Simd::<u32, LANES>::from_array(core::array::from_fn(|lane| (lane as u32) << lane_shift));

    let mut memo_state = prefix_state;
    crate::sha256::ingest_message_prefix::<FIRST_NONCE_WORD_IDX>(
        &mut memo_state,
        core::array::from_fn(|i| first_block[i]),
    );

    let candidate_space = 256u64.saturating_pow(nonce_byte_count.get() as u32);
    let search_end = limit.min(candidate_space);
    let mut cursor = (*attempted_nonces).min(search_end);
    while cursor < search_end {
        let x = cursor >> 8;
        let mut block_template = **first_block;
        let x_bytes = x.to_le_bytes();
        for (i, byte) in x_bytes
            .iter()
            .take(nonce_byte_count.get() as usize - 1)
            .enumerate()
        {
            let byte_index = nonce_byte_offset + 1 + i;
            let shift = (3 - byte_index % 4) * 8;
            block_template[byte_index / 4] |= u32::from(*byte) << shift;
        }

        let mut lane_base = (cursor & 0xff) as usize;
        while lane_base < 256 && cursor < search_end {
            let valid_lanes = (search_end - cursor)
                .min(LANES as u64)
                .min((256 - lane_base) as u64) as usize;
            let lane_values = lane_ids + Simd::splat((lane_base as u32) << lane_shift);

            let mut state = core::array::from_fn(|i| Simd::splat(memo_state[i]));
            let mut message: [Simd<u32, LANES>; 16] =
                core::array::from_fn(|i| Simd::splat(block_template[i]));
            message[FIRST_NONCE_WORD_IDX] |= lane_values;

            crate::sha256::portable::multiway_arx::<LANES, FIRST_NONCE_WORD_IDX>(
                &mut state,
                &mut message,
            );

            if NEED_SECOND_BLOCK {
                for i in 0..8 {
                    state[i] += Simd::splat(prefix_state[i]);
                }
                let save_a = state[0];
                #[cfg(feature = "compare-64bit")]
                let save_b = state[1];

                crate::sha256::portable::bcst_multiway_arx::<LANES, 0>(
                    &mut state,
                    second_block_schedule,
                );
                state[0] += save_a;
                #[cfg(feature = "compare-64bit")]
                {
                    state[1] += save_b;
                }
            } else {
                state[0] += Simd::splat(prefix_state[0]);
                #[cfg(feature = "compare-64bit")]
                {
                    state[1] += Simd::splat(prefix_state[1]);
                }
            }

            let met_target = target_match::<LANES, TYPE>(state[0], state[1], target, mask);

            cursor += valid_lanes as u64;
            *attempted_nonces = cursor;

            if met_target.any() {
                crate::unlikely();
                let lanes = met_target.to_array();
                if let Some(success_lane) = (0..valid_lanes).find(|&lane| lanes[lane]) {
                    return Some((x << 8) | (lane_base + success_lane) as u64);
                }
            }

            lane_base += valid_lanes;
        }
    }

    None
}

#[inline(always)]
pub(crate) fn solve_goaway_impl<const LANES: usize, const TYPE: u8>(
    attempted_nonces: &mut u64,
    limit: u64,
    challenge: [u32; 8],
    high_word: u32,
    target: u64,
    mask: u64,
) -> Option<u64> {
    const MSG_LEN: u32 = 10 * 4 * 8;

    let lane_ids = Simd::<u32, LANES>::from_array(core::array::from_fn(|lane| lane as u32));
    let mut prefix_state = crate::sha256::IV;
    crate::sha256::ingest_message_prefix(&mut prefix_state, challenge);
    let mut partial_state = prefix_state;
    crate::sha256::sha2_arx::<8>(&mut partial_state, &[high_word]);

    let search_end = limit.min(u64::from(u32::MAX) + 1);
    let mut low_word = (*attempted_nonces).min(search_end);
    while low_word < search_end {
        let valid_lanes = (search_end - low_word).min(LANES as u64) as usize;
        let mut state = core::array::from_fn(|i| Simd::splat(partial_state[i]));
        let mut message = [
            Simd::splat(challenge[0]),
            Simd::splat(challenge[1]),
            Simd::splat(challenge[2]),
            Simd::splat(challenge[3]),
            Simd::splat(challenge[4]),
            Simd::splat(challenge[5]),
            Simd::splat(challenge[6]),
            Simd::splat(challenge[7]),
            Simd::splat(high_word),
            Simd::splat(low_word as u32) + lane_ids,
            Simd::splat(u32::from_be_bytes([0x80, 0, 0, 0])),
            Simd::splat(0),
            Simd::splat(0),
            Simd::splat(0),
            Simd::splat(0),
            Simd::splat(MSG_LEN),
        ];
        crate::sha256::portable::multiway_arx::<LANES, 9>(&mut state, &mut message);
        state[0] += Simd::splat(crate::sha256::IV[0]);
        #[cfg(feature = "compare-64bit")]
        {
            state[1] += Simd::splat(crate::sha256::IV[1]);
        }

        let met_target = target_match::<LANES, TYPE>(state[0], state[1], target, mask);
        low_word += valid_lanes as u64;
        *attempted_nonces = low_word;

        if met_target.any() {
            crate::unlikely();
            let lanes = met_target.to_array();
            if let Some(success_lane) = (0..valid_lanes).find(|&lane| lanes[lane]) {
                let final_low_word = (low_word - valid_lanes as u64) as u32 + success_lane as u32;
                return Some((u64::from(high_word) << 32) | u64::from(final_low_word));
            }
        }
    }
    None
}
