use crate::{
    Align64,
    message::{
        AltchaMessage, BinaryMessage, CerberusMessage, DecimalMessage, DoubleBlockMessage,
        GoAwayMessage, SingleBlockMessage,
    },
};

cpufeatures::new!(avx2, "avx2");

#[derive(Debug, Copy, Clone)]
/// Required features for AVX-2 solver.
pub struct RequiredFeatures;

impl Default for RequiredFeatures {
    fn default() -> Self {
        Self
    }
}

impl crate::solver::CpuIDToken for RequiredFeatures {
    fn get() -> bool {
        avx2::get()
    }
}

/// AVX-2 decimal nonce single block solver.
///
///
/// Current implementation: 8 way SIMD with 1-round hotstart granularity.
pub struct SingleBlockSolver {
    message: SingleBlockMessage,

    attempted_nonces: u64,

    limit: u64,
}

impl From<super::safe::SingleBlockSolver> for SingleBlockSolver {
    fn from(solver: super::safe::SingleBlockSolver) -> Self {
        Self {
            message: solver.message,
            attempted_nonces: solver.attempted_nonces,
            limit: solver.limit,
        }
    }
}

impl From<SingleBlockMessage> for SingleBlockSolver {
    fn from(message: SingleBlockMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

const MUTATION_TYPE_UNALIGNED: u8 = 0;
const MUTATION_TYPE_ALIGNED: u8 = 1;
const MUTATION_TYPE_OCTAL: u8 = 2;
const MUTATION_TYPE_ALIGNED_OCTAL: u8 = MUTATION_TYPE_ALIGNED | MUTATION_TYPE_OCTAL;
const MUTATION_TYPE_UNALIGNED_OCTAL: u8 = MUTATION_TYPE_UNALIGNED | MUTATION_TYPE_OCTAL;

impl SingleBlockSolver {
    #[inline(never)]
    #[target_feature(enable = "avx2")]
    fn solve_impl<
        const DIGIT_WORD_IDX0: usize,
        const DIGIT_WORD_IDX1_INCREMENT: bool,
        const TYPE: u8,
        const MUTATION_TYPE: u8,
    >(
        &mut self,
        target: u64,
        mask: u64,
    ) -> Option<u64> {
        crate::solver::wide::solve_single_block_impl::<
            8,
            DIGIT_WORD_IDX0,
            DIGIT_WORD_IDX1_INCREMENT,
            TYPE,
            MUTATION_TYPE,
        >(
            &mut self.message,
            &mut self.attempted_nonces,
            self.limit,
            target,
            mask,
        )
    }
}

impl crate::solver::Solver for SingleBlockSolver {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn get_attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    fn solve_nonce_only<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<u64> {
        if !<RequiredFeatures as crate::solver::CpuIDToken>::get() {
            return None;
        }
        if self.attempted_nonces >= self.limit {
            return None;
        }
        let target = target & mask;

        // the official default difficulty is 5e6, so we design for 1e8
        // and there should almost always be a valid solution within our supported solution space
        // pgeom(5 * 16e7, 1/5e7, lower=F) = 0.03%
        // pgeom(16e7, 1/5e7, lower=F) = 20%, which is too much so we need the prefix to change as well

        // pre-compute an OR to apply to the message to add the lane ID
        let lane_id_0_word_idx = self.message.digit_index / 4;
        let lane_id_1_word_idx = (self.message.digit_index + 1) / 4;

        macro_rules! dispatch {
            ($idx0:literal, $idx1_inc:literal) => {
                unsafe {
                    if self.message.digit_index % 4 == 2 {
                        // if we have to much search space it doesn't matter
                        // use the octal kernel
                        if self.message.no_trailing_zeros
                            || self.message.approx_working_set_count.get() >= 100
                        {
                            self.solve_impl::<$idx0, $idx1_inc, TYPE, MUTATION_TYPE_ALIGNED_OCTAL>(
                                target, mask,
                            )
                        } else {
                            self.solve_impl::<$idx0, $idx1_inc, TYPE, MUTATION_TYPE_ALIGNED>(
                                target, mask,
                            )
                        }
                    } else if self.message.no_trailing_zeros {
                        self.solve_impl::<$idx0, $idx1_inc, TYPE, MUTATION_TYPE_UNALIGNED_OCTAL>(
                            target, mask,
                        )
                    } else {
                        self.solve_impl::<$idx0, $idx1_inc, TYPE, MUTATION_TYPE_UNALIGNED>(
                            target, mask,
                        )
                    }
                }
            };
            ($idx0:literal) => {
                if lane_id_0_word_idx == lane_id_1_word_idx {
                    dispatch!($idx0, false)
                } else {
                    dispatch!($idx0, true)
                }
            };
        }

        let nonce = match lane_id_0_word_idx {
            0 => dispatch!(0),
            1 => dispatch!(1),
            2 => dispatch!(2),
            3 => dispatch!(3),
            4 => dispatch!(4),
            5 => dispatch!(5),
            6 => dispatch!(6),
            7 => dispatch!(7),
            8 => dispatch!(8),
            9 => dispatch!(9),
            10 => dispatch!(10),
            11 => dispatch!(11),
            12 => dispatch!(12),
            13 => dispatch!(13),
            _ => return None,
        }?;

        Some(nonce + self.message.nonce_addend)
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        let nonce = self.solve_nonce_only::<TYPE>(target, mask)?;

        // recompute the hash from the beginning
        // this prevents the compiler from having to compute the final B-H registers alive in tight loops
        let mut final_sha_state = self.message.prefix_state;
        crate::sha256::digest_block(&mut final_sha_state, &self.message.message);

        Some((nonce, final_sha_state))
    }
}

/// AVX2 decimal nonce double block solver.
///
///
/// Current implementation: 8 way SIMD with 1-round hotstart granularity.
pub struct DoubleBlockSolver {
    message: DoubleBlockMessage,
    attempted_nonces: u64,

    limit: u64,
}

impl From<super::safe::DoubleBlockSolver> for DoubleBlockSolver {
    fn from(solver: super::safe::DoubleBlockSolver) -> Self {
        Self {
            message: solver.message,
            attempted_nonces: solver.attempted_nonces,
            limit: solver.limit,
        }
    }
}

impl From<DoubleBlockMessage> for DoubleBlockSolver {
    fn from(message: DoubleBlockMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl crate::solver::Solver for DoubleBlockSolver {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn get_attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    #[inline]
    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        if !<RequiredFeatures as crate::solver::CpuIDToken>::get() {
            return None;
        }
        // SAFETY: the feature check above proves AVX2 is available.
        unsafe { self.solve_impl::<TYPE>(target, mask) }
    }
}

impl DoubleBlockSolver {
    #[inline(never)]
    #[target_feature(enable = "avx2")]
    fn solve_impl<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        crate::solver::wide::solve_double_block_impl::<8, TYPE>(
            &mut self.message,
            &mut self.attempted_nonces,
            self.limit,
            target,
            mask,
        )
    }
}

impl_decimal_solver!(
    [SingleBlockSolver, DoubleBlockSolver] => DecimalSolver
);

/// AVX2 binary nonce solver.
///
/// Output: nonce in little endian order
///
/// Current implementation: 8 way SIMD with 1-round hotstart granularity.
pub struct BinarySolver {
    message: BinaryMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<BinaryMessage> for BinarySolver {
    fn from(message: BinaryMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl From<crate::solver::safe::BinarySolver> for BinarySolver {
    fn from(solver: crate::solver::safe::BinarySolver) -> Self {
        Self {
            message: solver.message,
            attempted_nonces: solver.attempted_nonces,
            limit: solver.limit,
        }
    }
}

impl BinarySolver {
    #[inline(never)]
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    fn solve_impl<
        const TYPE: u8,
        const FIRST_NONCE_WORD_IDX: usize,
        const NEED_SECOND_BLOCK: bool,
    >(
        &mut self,
        prefix_state: [u32; 8],
        first_block: &Align64<[u32; 16]>,
        second_block_schedule: &[u32; 64],
        nonce_byte_offset: usize,
        nonce_byte_count: core::num::NonZeroU8,
        target: u64,
        mask: u64,
    ) -> Option<u64> {
        crate::solver::wide::solve_binary_impl::<8, TYPE, FIRST_NONCE_WORD_IDX, NEED_SECOND_BLOCK>(
            crate::solver::wide::BinaryWork {
                attempted_nonces: &mut self.attempted_nonces,
                limit: self.limit,
                prefix_state,
                first_block,
                second_block_schedule,
                nonce_byte_offset,
                nonce_byte_count,
            },
            target,
            mask,
        )
    }
}

impl crate::solver::Solver for BinarySolver {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn get_attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        if !<RequiredFeatures as crate::solver::CpuIDToken>::get() {
            return None;
        }
        // A one-byte search has only 256 candidates, and a nonce crossing a SHA block boundary
        // needs a distinct shuffle layout. Keep those rare cases on the scalar fallback instead of
        // burdening the hot SIMD path with extra setup and branches.
        if (self.message.nonce_byte_count.get() == 1)
            || (self.message.salt_residual_len + self.message.nonce_byte_count.get() as usize > 64)
        {
            crate::unlikely();
            let mut solver = crate::solver::safe::BinarySolver::from(self.message.clone());
            return solver.solve::<TYPE>(target, mask);
        }

        let salt = &self.message.salt_residual[..self.message.salt_residual_len];
        let mut blocks = [[0u8; 64]; 2];
        blocks[0][..salt.len()].copy_from_slice(salt);
        let mut ptr = salt.len();
        let mut cur_block = 0;

        for _ in 0..self.message.nonce_byte_count.get() {
            blocks[cur_block][ptr] = 0;
            ptr += 1;
            if ptr == 64 {
                cur_block = 1;
                ptr = 0;
            }
        }
        blocks[cur_block][ptr] = 0x80;
        ptr += 1;
        if ptr + 8 > 64 {
            cur_block = 1;
        }
        blocks[cur_block][(64 - 8)..]
            .copy_from_slice(&(self.message.message_length * 8).to_be_bytes());

        let used_blocks = &mut blocks[..=cur_block];

        let mut block_template_be = Align64([0; 16]);
        for i in 0..16 {
            block_template_be[i] =
                u32::from_be_bytes(used_blocks[0][i * 4..][..4].try_into().unwrap());
        }

        let mut second_block_schedule = [0; 64];
        if cur_block == 1 {
            for i in 0..16 {
                second_block_schedule[i] =
                    u32::from_be_bytes(used_blocks[1][i * 4..][..4].try_into().unwrap());
            }
            crate::sha256::do_message_schedule_k_w(&mut second_block_schedule);
        }

        macro_rules! dispatch {
            ($skipped_rounds:expr) => {
                unsafe {
                    if cur_block == 1 {
                        if let Some(nonce) = self.solve_impl::<TYPE, { $skipped_rounds }, true>(
                            *self.message.prefix_state,
                            &block_template_be,
                            &second_block_schedule,
                            self.message.salt_residual_len,
                            self.message.nonce_byte_count,
                            target,
                            mask,
                        ) {
                            let mut final_sha_state = self.message.prefix_state;
                            for i in 0..self.message.nonce_byte_count.get() as usize {
                                used_blocks[0][self.message.salt_residual_len + i] =
                                    nonce.to_le_bytes()[i];
                            }
                            sha2::block_api::compress256(&mut final_sha_state, &used_blocks);
                            return Some((nonce, final_sha_state.0));
                        }
                    } else {
                        if let Some(nonce) = self.solve_impl::<TYPE, { $skipped_rounds }, false>(
                            *self.message.prefix_state,
                            &block_template_be,
                            &[0; 64],
                            self.message.salt_residual_len,
                            self.message.nonce_byte_count,
                            target,
                            mask,
                        ) {
                            let mut final_sha_state = self.message.prefix_state;
                            for i in 0..self.message.nonce_byte_count.get() as usize {
                                used_blocks[0][self.message.salt_residual_len + i] =
                                    nonce.to_le_bytes()[i];
                            }
                            sha2::block_api::compress256(&mut final_sha_state, &used_blocks);
                            return Some((nonce, final_sha_state.0));
                        }
                    }
                }
            };
        }

        match self.message.salt_residual_len / 4 {
            0 => dispatch!(0),
            1 => dispatch!(1),
            2 => dispatch!(2),
            3 => dispatch!(3),
            4 => dispatch!(4),
            5 => dispatch!(5),
            6 => dispatch!(6),
            7 => dispatch!(7),
            8 => dispatch!(8),
            9 => dispatch!(9),
            10 => dispatch!(10),
            11 => dispatch!(11),
            12 => dispatch!(12),
            13 => dispatch!(13),
            14 => dispatch!(14),
            15 => dispatch!(15),
            _ => unreachable!(),
        }

        crate::unlikely();

        None
    }
}

/// AVX2 GoAway solver.
///
///
/// Current implementation: 8 way SIMD with 1-round hotstart granularity.
pub struct GoAwaySolver {
    message: GoAwayMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<super::safe::GoAwaySolver> for GoAwaySolver {
    fn from(solver: super::safe::GoAwaySolver) -> Self {
        Self {
            message: solver.message,
            attempted_nonces: solver.attempted_nonces,
            limit: solver.limit,
        }
    }
}

impl From<GoAwayMessage> for GoAwaySolver {
    fn from(message: GoAwayMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl GoAwaySolver {
    const MSG_LEN: u32 = 10 * 4 * 8;

    #[inline(never)]
    #[target_feature(enable = "avx2")]
    fn solve_nonce_only_impl<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<u64> {
        crate::solver::wide::solve_goaway_impl::<8, TYPE>(
            &mut self.attempted_nonces,
            self.limit,
            self.message.challenge,
            self.message.high_word,
            target,
            mask,
        )
    }
}

impl crate::solver::Solver for GoAwaySolver {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn get_attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    #[inline(always)]
    fn solve_nonce_only<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<u64> {
        if !<RequiredFeatures as crate::solver::CpuIDToken>::get() {
            return None;
        }
        // SAFETY: the feature check above proves AVX2 is available.
        unsafe { self.solve_nonce_only_impl::<TYPE>(target, mask) }
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        let mut output_msg = [0; 16];
        let nonce = self.solve_nonce_only::<TYPE>(target, mask)?;
        output_msg[..8].copy_from_slice(&self.message.challenge);
        output_msg[8] = (nonce >> 32) as u32;
        output_msg[9] = nonce as u32;
        output_msg[10] = u32::from_be_bytes([0x80, 0, 0, 0]);
        output_msg[15] = Self::MSG_LEN as _;

        let mut final_sha_state = crate::sha256::IV;
        crate::sha256::digest_block(&mut final_sha_state, &output_msg);

        Some((nonce, final_sha_state))
    }
}

/// AVX-2 Cerberus solver.
///
/// Current implementation: 9-digit out-of-order kernel with dual-wavefront 8 way SIMD and quarter-round hotstart granularity.
pub struct CerberusSolver {
    message: CerberusMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<CerberusMessage> for CerberusSolver {
    fn from(message: CerberusMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: !0,
        }
    }
}

impl CerberusSolver {
    /// Set the limit.
    pub fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    /// Get the attempted nonces.
    pub fn get_attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }
}

impl CerberusSolver {
    #[inline(never)]
    #[target_feature(enable = "avx2")]
    fn solve_decimal_impl<
        const CENTER_WORD_IDX: usize,
        const LANE_ID_WORD_IDX: usize,
        const CONSTANT_WORD_COUNT: usize,
    >(
        &mut self,
        msg_tpl: Align64<[u32; 16]>,
        target: u64,
        mask: u64,
    ) -> Option<(u64, u64)> {
        crate::solver::wide::solve_cerberus_decimal_avx2::<
            CENTER_WORD_IDX,
            LANE_ID_WORD_IDX,
            CONSTANT_WORD_COUNT,
        >(
            &self.message,
            msg_tpl,
            &mut self.attempted_nonces,
            self.limit,
            target,
            mask,
        )
    }

    #[inline(never)]
    #[target_feature(enable = "avx2")]
    fn solve_binary_impl(&mut self, target: u64, mask: u64) -> Option<u64> {
        crate::solver::wide::solve_cerberus_binary_avx2(
            &self.message,
            &mut self.attempted_nonces,
            self.limit,
            target,
            mask,
        )
    }
}

impl crate::solver::Solver for CerberusSolver {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn get_attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    fn solve_nonce_only<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<u64> {
        if !<RequiredFeatures as crate::solver::CpuIDToken>::get() {
            return None;
        }
        match &self.message {
            CerberusMessage::Binary(_) => unsafe { self.solve_binary_impl(target, mask) },
            CerberusMessage::Decimal(message) => {
                // two digits as lane ID, N=\x00, ? is prefix
                // position % 4 =0: |1234|5678|NNN9
                // position % 4 =1: |123?|4567|NN89
                // position % 4 =2: |12??|3456|N789
                // position % 4 =3: |1???|2345|6789

                let center_word_idx = message.salt_residual_len / 4 + 1;
                let nonce_addend = message.nonce_addend;
                let salt_residual = message.salt_residual;
                let salt_residual_len = message.salt_residual_len;
                let position_mod = message.salt_residual_len % 4;

                for resid0 in 0..10u64 {
                    for resid1 in 0..10u64 {
                        if self.attempted_nonces >= self.limit {
                            return None;
                        }
                        let mut msg = salt_residual;

                        match position_mod {
                            0 => {
                                msg[salt_residual_len] = resid0 as u8 + b'0';
                                msg[salt_residual_len + 8] = resid1 as u8 + b'0';
                            }
                            1 => {
                                msg[salt_residual_len + 7] = resid0 as u8 + b'0';
                                msg[salt_residual_len + 8] = resid1 as u8 + b'0';
                            }
                            2 => {
                                msg[salt_residual_len] = resid0 as u8 + b'0';
                                msg[salt_residual_len + 1] = resid1 as u8 + b'0';
                            }
                            3 => {
                                msg[salt_residual_len] = resid0 as u8 + b'0';
                                msg[salt_residual_len + 8] = resid1 as u8 + b'0';
                            }
                            _ => unreachable!(),
                        }

                        let msg = Align64(core::array::from_fn(|i| {
                            u32::from_le_bytes([
                                msg[i * 4],
                                msg[i * 4 + 1],
                                msg[i * 4 + 2],
                                msg[i * 4 + 3],
                            ])
                        }));

                        macro_rules! dispatch {
                            ($center_word_idx:literal) => {
                                if position_mod < 2 {
                                    unsafe {
                                        self.solve_decimal_impl::<$center_word_idx, { $center_word_idx - 1 }, {$center_word_idx - 1}>(
                                        msg, target, mask,
                                        )
                                    }
                                } else {
                                    unsafe {
                                        self.solve_decimal_impl::<$center_word_idx, { $center_word_idx + 1 }, $center_word_idx>(
                                        msg, target, mask,
                                        )
                                    }
                                }
                            };
                        }

                        if let Some((middle_word, success_lane_idx)) = match center_word_idx {
                            1 => dispatch!(1),
                            2 => dispatch!(2),
                            3 => dispatch!(3),
                            4 => dispatch!(4),
                            5 => dispatch!(5),
                            6 => dispatch!(6),
                            7 => dispatch!(7),
                            8 => dispatch!(8),
                            9 => dispatch!(9),
                            10 => dispatch!(10),
                            11 => dispatch!(11),
                            12 => dispatch!(12),
                            13 => dispatch!(13),
                            14 => dispatch!(14),
                            15 => dispatch!(15),
                            _ => unreachable!(),
                        } {
                            let output_nonce = nonce_addend
                                + match position_mod {
                                    0 => {
                                        10 * middle_word
                                            + 100_000 * success_lane_idx
                                            + 100_000_000 * resid0
                                            + resid1
                                    }
                                    1 => {
                                        100 * middle_word
                                            + 1_000_000 * success_lane_idx
                                            + 10 * resid0
                                            + resid1
                                    }
                                    2 => {
                                        1000 * middle_word
                                            + success_lane_idx
                                            + 100_000_000 * resid0
                                            + 10_000_000 * resid1
                                    }
                                    3 => {
                                        10000 * middle_word
                                            + 10 * success_lane_idx
                                            + 100_000_000 * resid0
                                            + resid1
                                    }
                                    _ => unreachable!(),
                                };

                            return Some(output_nonce);
                        }
                    }
                }

                None
            }
        }
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        if let Some(nonce) = self.solve_nonce_only::<TYPE>(target, mask) {
            match &self.message {
                CerberusMessage::Decimal(message) => {
                    let mut msg = message.salt_residual;

                    let mut nonce_copy = nonce;
                    for i in (0..9).rev() {
                        msg[message.salt_residual_len + i] = (nonce_copy % 10) as u8 + b'0';
                        nonce_copy /= 10;
                    }

                    let msg = core::array::from_fn(|i| {
                        u32::from_le_bytes([
                            msg[i * 4],
                            msg[i * 4 + 1],
                            msg[i * 4 + 2],
                            msg[i * 4 + 3],
                        ])
                    });

                    let hash = crate::blake3::compress8(
                        &message.prefix_state,
                        &msg,
                        0,
                        message.salt_residual_len as u32 + 9,
                        message.flags,
                    );

                    Some((nonce, hash))
                }
                CerberusMessage::Binary(message) => {
                    let mut msg = [0; 16];
                    msg[0] = message.first_word;
                    msg[1] = nonce as u32;
                    let hash = crate::blake3::compress8(
                        &message.midstate,
                        &msg,
                        0,
                        8,
                        crate::blake3::FLAG_CHUNK_END | crate::blake3::FLAG_ROOT,
                    );
                    Some((msg[1] as u64 | (msg[0] as u64) << 32, hash))
                }
            }
        } else {
            None
        }
    }
}

/// AVX2 Altcha SHA-256 solver
pub struct AltchaSha256Solver {
    pub(super) message: AltchaMessage,
    pub(super) attempted_nonces: u64,
    pub(super) limit: u64,
}

impl From<AltchaMessage> for AltchaSha256Solver {
    fn from(message: AltchaMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl AltchaSha256Solver {
    #[inline(never)]
    #[target_feature(enable = "avx2")]
    fn solve_impl<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        crate::solver::wide::solve_altcha_impl::<8, TYPE>(
            &self.message,
            &mut self.attempted_nonces,
            self.limit,
            target,
            mask,
        )
    }

    fn solve_fallback<const TYPE: u8>(
        &mut self,
        target: u64,
        mask: u64,
    ) -> Option<(u64, [u32; 8])> {
        crate::solver::wide::solve_altcha_fallback::<TYPE>(
            &self.message,
            &mut self.attempted_nonces,
            self.limit,
            target,
            mask,
        )
    }
}

impl crate::solver::Solver for AltchaSha256Solver {
    type Output = [u32; 8];

    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn get_attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        if !<RequiredFeatures as crate::solver::CpuIDToken>::get() {
            return self.solve_fallback::<TYPE>(target, mask);
        }

        // SAFETY: RequiredFeatures::get() above proves AVX2 is available.
        unsafe { self.solve_impl::<TYPE>(target, mask) }
    }
}

#[cfg(target_feature = "avx2")]
#[cfg(test)]
mod tests {
    use crate::message::{CerberusBinaryMessage, CerberusDecimalMessage};
    use crate::solver::Solver as _;

    use super::*;
    #[test]
    fn test_single_block_attempt_accounting() {
        for limit in [1, 8, 9, 17] {
            let message = SingleBlockMessage::new(&[b'a'; 64], 0).expect("single block");
            let mut solver = SingleBlockSolver::from(message);
            solver.set_limit(limit);
            assert!(
                solver
                    .solve::<{ crate::solver::SOLVE_TYPE_LT }>(0, u64::MAX)
                    .is_none()
            );
            assert_eq!(solver.get_attempted_nonces(), limit);
        }
    }

    #[test]
    fn test_vector_tail_attempt_limits() {
        fn check<S: crate::solver::Solver<Output = [u32; 8]>>(mut solver: S) {
            solver.set_limit(1);
            assert!(
                solver
                    .solve::<{ crate::solver::SOLVE_TYPE_LT }>(0, u64::MAX)
                    .is_none()
            );
            assert_eq!(solver.get_attempted_nonces(), 1);

            solver.set_limit(9);
            assert!(
                solver
                    .solve::<{ crate::solver::SOLVE_TYPE_LT }>(0, u64::MAX)
                    .is_none()
            );
            assert_eq!(solver.get_attempted_nonces(), 9);
        }

        let salt = [b'a'; 32];
        check(BinarySolver::from(BinaryMessage::new(
            &salt,
            4.try_into().unwrap(),
        )));
        check(GoAwaySolver::from(GoAwayMessage::new_bytes(&salt, 0)));
    }

    #[cfg(feature = "compare-64bit")]
    #[test]
    fn test_full64_lt_comparisons() {
        const TARGET: u64 = 1u64 << 63;
        fn check<S: crate::solver::Solver<Output = [u32; 8]>>(mut solver: S) {
            solver.set_limit(1_000_000);
            let (_, hash) = solver
                .solve::<{ crate::solver::SOLVE_TYPE_LT }>(TARGET, u64::MAX)
                .expect("full-width LT solver should find an easy target");
            let value = (u64::from(hash[0]) << 32) | u64::from(hash[1]);
            assert!(value < TARGET, "{value:#018x} is not below {TARGET:#018x}");
        }

        let salt = [b'a'; 64];
        check(SingleBlockSolver::from(
            SingleBlockMessage::new(&salt, 0).expect("single block"),
        ));
        let double = (0..64)
            .find_map(|len| DoubleBlockMessage::new(&salt[..len], 0))
            .expect("double block shape");
        check(DoubleBlockSolver::from(double));
        check(BinarySolver::from(BinaryMessage::new(
            &salt,
            4.try_into().unwrap(),
        )));
        check(GoAwaySolver::from(GoAwayMessage::new_bytes(
            (&salt[..32]).try_into().unwrap(),
            0,
        )));
    }

    #[test]
    fn test_solve_decimal() {
        crate::solver::tests::test_decimal_validator::<DecimalSolver, _>(|prefix, search_space| {
            if let Some(solver) = SingleBlockMessage::new(prefix, search_space).map(Into::into) {
                Some(DecimalSolver::SingleBlock(solver))
            } else {
                DoubleBlockMessage::new(prefix, search_space).map(Into::into)
            }
        });
    }

    #[test]
    fn test_solve_decimal_f64() {
        crate::solver::tests::test_decimal_validator_f64_safe::<DecimalSolver, _>(
            |prefix, search_space| {
                if let Some((solver, p)) =
                    SingleBlockMessage::new_f64(prefix, search_space).map(|(x, p)| (x.into(), p))
                {
                    Some((DecimalSolver::SingleBlock(solver), p))
                } else {
                    DoubleBlockMessage::new(prefix, search_space)
                        .map(|x| (DecimalSolver::DoubleBlock(x.into()), None))
                }
            },
        );
    }

    #[test]
    fn test_solve_cerberus_decimal() {
        for i in 0..=1 {
            crate::solver::tests::test_cerberus_decimal_validator::<CerberusSolver, _>(|prefix| {
                Some(CerberusMessage::Decimal(CerberusDecimalMessage::new(prefix, i)?).into())
            });
        }
    }

    #[test]
    fn test_cerberus_decimal_exact_limit() {
        let message = CerberusDecimalMessage::new(b"cerberus-decimal-benchmark|", 0)
            .expect("valid Cerberus decimal message");
        let mut solver = CerberusSolver::from(CerberusMessage::Decimal(message));
        solver.set_limit(17);
        assert!(
            solver
                .solve::<{ crate::solver::SOLVE_TYPE_MASK }>(0, u64::MAX)
                .is_none()
        );
        assert_eq!(solver.get_attempted_nonces(), 17);
    }

    #[test]
    fn test_solve_cerberus_binary() {
        for i in 0..=1 {
            crate::solver::tests::test_cerberus_binary_validator::<CerberusSolver, _>(|prefix| {
                Some(CerberusMessage::Binary(CerberusBinaryMessage::new(prefix, i)).into())
            });
        }
    }
    #[test]
    fn test_solve_binary() {
        crate::solver::tests::test_binary_validator::<BinarySolver, _>(
            |prefix, nonce_byte_count| {
                BinarySolver::from(BinaryMessage::new(prefix, nonce_byte_count))
            },
        )
    }

    #[test]
    fn test_solve_goaway() {
        crate::solver::tests::test_goaway_validator::<GoAwaySolver, _>(|prefix| {
            GoAwaySolver::from(GoAwayMessage::new(
                core::array::from_fn(|i| {
                    u32::from_be_bytes([
                        prefix[i * 4],
                        prefix[i * 4 + 1],
                        prefix[i * 4 + 2],
                        prefix[i * 4 + 3],
                    ])
                }),
                0,
            ))
        });
    }

    #[test]
    fn test_solve_altcha() {
        crate::solver::tests::test_altcha_validator::<AltchaSha256Solver, _>(|message| {
            Some(AltchaSha256Solver::from(message))
        });
    }
}
