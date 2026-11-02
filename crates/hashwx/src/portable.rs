use core::simd::{
    Mask, Select as _, Simd,
    cmp::{SimdPartialEq as _, SimdPartialOrd as _},
    num::{SimdInt as _, SimdUint as _},
};

use super::SEED_SIZE;
use super::tables::{OPCODE_TEMPLATES, SRC_DEEP, SRC_SHALLOW};

const LANES: usize = 4;
const PROGRAMS: usize = 32;
const REPEATS: usize = 4;
const BRANCH_COUNT: u64 = 32;
const WRITABLE_BYTES: u64 = 16_384;
const WORDS_PER_LANE: usize = 2_049;
const R8_WORD: usize = 2_048;

type V = Simd<u64, LANES>;
type I = Simd<usize, LANES>;
type M = Mask<i64, LANES>;

const LANE_BASES: I = I::from_array([0, WORDS_PER_LANE, 2 * WORDS_PER_LANE, 3 * WORDS_PER_LANE]);

#[derive(Clone, Copy)]
struct Regs {
    r0: V,
    r1: V,
    r2: V,
    r3: V,
    r4: V,
    r5: V,
    r6: V,
    r7: V,
    r8: V,
}

impl Regs {
    #[inline(always)]
    fn get(&self, index: u8) -> V {
        match index {
            0 => self.r0,
            1 => self.r1,
            2 => self.r2,
            3 => self.r3,
            4 => self.r4,
            5 => self.r5,
            6 => self.r6,
            7 => self.r7,
            8 => self.r8,
            _ => unreachable!("HashWX register index out of range"),
        }
    }

    #[inline(always)]
    fn set_masked(&mut self, index: u8, value: V, active: M) {
        match index {
            0 => self.r0 = active.select(value, self.r0),
            1 => self.r1 = active.select(value, self.r1),
            2 => self.r2 = active.select(value, self.r2),
            3 => self.r3 = active.select(value, self.r3),
            4 => self.r4 = active.select(value, self.r4),
            5 => self.r5 = active.select(value, self.r5),
            6 => self.r6 = active.select(value, self.r6),
            7 => self.r7 = active.select(value, self.r7),
            _ => unreachable!("HashWX writable register index out of range"),
        }
    }

    #[inline(always)]
    fn set(&mut self, index: u8, value: V) {
        match index {
            0 => self.r0 = value,
            1 => self.r1 = value,
            2 => self.r2 = value,
            3 => self.r3 = value,
            4 => self.r4 = value,
            5 => self.r5 = value,
            6 => self.r6 = value,
            7 => self.r7 = value,
            _ => unreachable!("HashWX writable register index out of range"),
        }
    }
}

#[derive(Clone, Copy)]
struct Locals {
    d0: V,
    d1: V,
    d2: V,
    d3: V,
    d4: V,
    d5: V,
    d6: V,
    d7: V,
}

impl Locals {
    #[inline(always)]
    fn load(global: &Regs, dst: &[u8; 8]) -> Self {
        Self {
            d0: global.get(dst[0]),
            d1: global.get(dst[1]),
            d2: global.get(dst[2]),
            d3: global.get(dst[3]),
            d4: global.get(dst[4]),
            d5: global.get(dst[5]),
            d6: global.get(dst[6]),
            d7: global.get(dst[7]),
        }
    }

    #[inline(always)]
    fn get(&self, slot: u8) -> V {
        match slot {
            0 => self.d0,
            1 => self.d1,
            2 => self.d2,
            3 => self.d3,
            4 => self.d4,
            5 => self.d5,
            6 => self.d6,
            7 => self.d7,
            _ => unreachable!("HashWX local register slot out of range"),
        }
    }

    #[inline(always)]
    fn commit(self, global: &mut Regs, dst: &[u8; 8]) {
        global.set(dst[0], self.d0);
        global.set(dst[1], self.d1);
        global.set(dst[2], self.d2);
        global.set(dst[3], self.d3);
        global.set(dst[4], self.d4);
        global.set(dst[5], self.d5);
        global.set(dst[6], self.d6);
        global.set(dst[7], self.d7);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Op {
    MulOr = 0,
    MulXor = 1,
    MulAdd = 2,
    Rmcg = 3,
    XorRor = 4,
    AddRor = 5,
    SubRor = 6,
    XorAsr = 7,
    AddAsr = 8,
    SubAsr = 9,
    XorLsr = 10,
    AddLsr = 11,
    SubLsr = 12,
}

impl Op {
    fn mul(selector: u32) -> Self {
        match selector % 3 {
            0 => Self::MulOr,
            1 => Self::MulXor,
            _ => Self::MulAdd,
        }
    }

    fn xas(selector: u32) -> Self {
        match selector % 9 {
            0 => Self::XorRor,
            1 => Self::AddRor,
            2 => Self::SubRor,
            3 => Self::XorAsr,
            4 => Self::AddAsr,
            5 => Self::SubAsr,
            6 => Self::XorLsr,
            7 => Self::AddLsr,
            _ => Self::SubLsr,
        }
    }

    fn immediate(self, selector: u64) -> u8 {
        match self {
            Self::MulOr | Self::MulXor | Self::MulAdd => [1, 9, 33][selector as usize % 3],
            Self::Rmcg | Self::XorRor | Self::AddRor | Self::SubRor => {
                1 + u8::try_from(selector % 63).expect("modulo 63 fits u8")
            }
            Self::XorAsr
            | Self::AddAsr
            | Self::SubAsr
            | Self::XorLsr
            | Self::AddLsr
            | Self::SubLsr => 1 + u8::try_from(selector % 3).expect("modulo 3 fits u8"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Instruction {
    op: Op,
    dst: u8,
    src: u8,
    imm: u8,
}

const EMPTY_INSTRUCTION: Instruction = Instruction {
    op: Op::Rmcg,
    dst: 0,
    src: 8,
    imm: 1,
};

#[derive(Clone, Copy, Debug)]
struct Program {
    body: [Instruction; 8],
    dst: [u8; 8],
    src_slot: [u8; 8],
    store_offset: [u8; 8],
    unconditional: bool,
}

const EMPTY_PROGRAM: Program = Program {
    body: [EMPTY_INSTRUCTION; 8],
    dst: [0; 8],
    src_slot: [0; 8],
    store_offset: [0; 8],
    unconditional: false,
};

#[derive(Clone, Copy, Debug)]
struct Key {
    k0: u64,
    k1: u64,
}

#[derive(Clone, Copy)]
struct SipRng {
    key: Key,
    state: [u64; 4],
    count: usize,
}

impl SipRng {
    fn new(key: Key, salt: u64) -> Self {
        let mut state = [
            0x736f_6d65_7073_6575 ^ key.k0,
            0x646f_7261_6e64_6f6d ^ key.k1,
            0x6c79_6765_6e65_7261 ^ key.k0,
            0x7465_6462_7974_6573 ^ key.k1,
        ];
        state[3] ^= salt;
        sipround_scalar(&mut state);
        state[0] ^= salt;
        state[2] ^= 0xbb;
        for _ in 0..3 {
            sipround_scalar(&mut state);
        }
        Self {
            key,
            state,
            count: 4,
        }
    }

    fn mix(&mut self) {
        self.state[0] ^= self.key.k0;
        self.state[1] ^= self.key.k1;
        self.state[2] ^= self.key.k0;
        self.state[3] ^= self.key.k1;
        for _ in 0..4 {
            sipround_scalar(&mut self.state);
        }
    }

    fn next(&mut self) -> u64 {
        if self.count == 0 {
            self.mix();
            self.count = 4;
        }
        self.count -= 1;
        self.state[self.count]
    }
}

#[inline(always)]
fn sipround_scalar(v: &mut [u64; 4]) {
    v[0] = v[0].wrapping_add(v[1]);
    v[2] = v[2].wrapping_add(v[3]);
    v[1] = v[1].rotate_left(13);
    v[3] = v[3].rotate_left(16);
    v[1] ^= v[0];
    v[3] ^= v[2];
    v[0] = v[0].rotate_left(32);
    v[2] = v[2].wrapping_add(v[1]);
    v[0] = v[0].wrapping_add(v[3]);
    v[1] = v[1].rotate_left(17);
    v[3] = v[3].rotate_left(21);
    v[1] ^= v[2];
    v[3] ^= v[0];
    v[2] = v[2].rotate_left(32);
}

#[inline(always)]
fn ror(value: V, shift: u64) -> V {
    (value >> shift) | (value << (64 - shift))
}

#[inline(always)]
fn sipround_simd(mut v0: V, mut v1: V, mut v2: V, mut v3: V) -> (V, V, V, V) {
    v0 += v1;
    v2 += v3;
    v1 = ror(v1, 51);
    v3 = ror(v3, 48);
    v1 ^= v0;
    v3 ^= v2;
    v0 = ror(v0, 32);
    v2 += v1;
    v0 += v3;
    v1 = ror(v1, 47);
    v3 = ror(v3, 43);
    v1 ^= v2;
    v3 ^= v0;
    v2 = ror(v2, 32);
    (v0, v1, v2, v3)
}

fn read_key(seed: &[u8; SEED_SIZE], offset: usize) -> Key {
    let k0 = u64::from_le_bytes(
        seed[offset..offset + 8]
            .try_into()
            .expect("8-byte seed chunk"),
    );
    let k1 = u64::from_le_bytes(
        seed[offset + 8..offset + 16]
            .try_into()
            .expect("8-byte seed chunk"),
    );
    Key { k0, k1 }
}

fn generate_program(rng: &mut SipRng, is_last: bool, is_deep: bool) -> Program {
    let mut generated = [0_u64; 8];
    for value in &mut generated {
        *value = rng.next();
    }

    let template = OPCODE_TEMPLATES[generated[7] as usize % OPCODE_TEMPLATES.len()];
    let mut op = [Op::Rmcg; 8];
    let mut dst_select = [0_u32; 7];
    for index in 0..7 {
        let low = generated[index] as u32;
        dst_select[index] = (generated[index] >> 32) as u32;
        op[index + 1] = if (template >> index) & 1 != 0 {
            Op::mul(low)
        } else {
            Op::xas(low)
        };
    }

    let mut dst = [0_u8; 8];
    for index in 0..8 {
        dst[index] = u8::try_from(index).expect("register index fits u8");
        if index != 0 {
            let swap = dst_select[index - 1] as usize % (index + 1);
            dst[index] = dst[swap];
            dst[swap] = u8::try_from(index).expect("register index fits u8");
        }
    }

    let permutation = if is_deep {
        &SRC_DEEP[generated[7] as usize % SRC_DEEP.len()]
    } else {
        &SRC_SHALLOW[generated[7] as usize % SRC_SHALLOW.len()]
    };
    let mut src = [0_u8; 8];
    src[0] = 8;
    for index in 1..8 {
        src[index] = dst[usize::from(permutation[index])];
    }

    let mut body = [EMPTY_INSTRUCTION; 8];
    for index in 0..8 {
        body[index] = Instruction {
            op: op[index],
            dst: dst[index],
            src: src[index],
            imm: op[index].immediate(rng.next()),
        };
    }
    let mut src_slot = [0_u8; 8];
    src_slot[0] = 8;
    for index in 1..8 {
        src_slot[index] = permutation[index];
    }
    let store_offset = core::array::from_fn(|index| 7 - dst[index]);
    Program {
        body,
        dst,
        src_slot,
        store_offset,
        unconditional: is_last,
    }
}

fn generate_programs(seed: &[u8; SEED_SIZE]) -> ([Program; PROGRAMS], Key) {
    let program_key = read_key(seed, 0);
    let hash_key = read_key(seed, 16);
    let mut rng = SipRng::new(program_key, u64::MAX);
    let mut programs = [EMPTY_PROGRAM; PROGRAMS];
    for (index, program) in programs.iter_mut().enumerate() {
        *program = generate_program(
            &mut rng,
            index + 1 == PROGRAMS,
            index < PROGRAMS - 1 && index % 6 == 0,
        );
    }
    (programs, hash_key)
}

/// Four-lane portable-SIMD HashWX executor.
///
/// It keeps one independent 16 KiB scratchpad per SIMD lane and executes the
/// same seed-generated program across four nonces. End-of-program branch
/// divergence is represented as a lane mask; no lane has an independent
/// program counter. Unaligned memory reads are reconstructed from two aligned
/// gathered words, preserving HashWX's byte-addressed scratch semantics.
pub struct PortableHashWx {
    programs: [Program; PROGRAMS],
    key: Key,
    scratch: Vec<u64>,
    has_program: bool,
}

impl PortableHashWx {
    #[must_use]
    pub fn new() -> Self {
        Self {
            programs: [EMPTY_PROGRAM; PROGRAMS],
            key: Key { k0: 0, k1: 0 },
            scratch: vec![0; WORDS_PER_LANE * LANES],
            has_program: false,
        }
    }

    pub fn make(&mut self, seed: &[u8; SEED_SIZE]) {
        (self.programs, self.key) = generate_programs(seed);
        self.has_program = true;
    }

    #[must_use]
    pub const fn lanes() -> usize {
        LANES
    }

    #[must_use]
    pub fn hash4(&mut self, input: [u64; LANES]) -> [u64; LANES] {
        assert!(
            self.has_program,
            "HashWX program must be generated before hashing"
        );
        let mut r = self.initialize(V::from_array(input));
        self.write_phase(&mut r);
        self.read_phase(&mut r);
        self.finalize(r).to_array()
    }

    fn initialize(&self, salt: V) -> Regs {
        let k0 = V::splat(self.key.k0);
        let k1 = V::splat(self.key.k1);
        let mut v0 = V::splat(0x736f_6d65_7073_6575) ^ k0;
        let mut v1 = V::splat(0x646f_7261_6e64_6f6d) ^ k1;
        let mut v2 = V::splat(0x6c79_6765_6e65_7261) ^ k0;
        let mut v3 = V::splat(0x7465_6462_7974_6573) ^ k1;
        v3 ^= salt;
        (v0, v1, v2, v3) = sipround_simd(v0, v1, v2, v3);
        v0 ^= salt;
        v2 ^= V::splat(0xbb);
        for _ in 0..3 {
            (v0, v1, v2, v3) = sipround_simd(v0, v1, v2, v3);
        }
        let r0 = v3;
        let r1 = v2;
        let r2 = v1;
        let r3 = v0;
        v0 ^= k0;
        v1 ^= k1;
        v2 ^= k0;
        v3 ^= k1;
        for _ in 0..4 {
            (v0, v1, v2, v3) = sipround_simd(v0, v1, v2, v3);
        }
        let r4 = v3;
        let r5 = v2;
        let r6 = v1;
        let r7 = v0;
        let r8 = ((r0 ^ r4) & V::splat(!7_u64)) | V::splat(3);
        Regs {
            r0,
            r1,
            r2,
            r3,
            r4,
            r5,
            r6,
            r7,
            r8,
        }
    }

    fn write_phase(&mut self, r: &mut Regs) {
        let tail_indices = LANE_BASES + I::splat(R8_WORD);
        r.r8.scatter(&mut self.scratch, tail_indices);
        let mut sp = V::splat(WRITABLE_BYTES / 8);
        for _ in 0..REPEATS {
            let mut branch_counter = V::splat(BRANCH_COUNT);
            for program in self.programs {
                self.execute_program_write(program, r, &mut branch_counter, &mut sp);
            }
        }
        debug_assert!(sp.simd_eq(V::splat(0)).all());
    }

    fn read_phase(&self, r: &mut Regs) {
        for _ in 0..REPEATS {
            let mut branch_counter = V::splat(BRANCH_COUNT);
            for program in self.programs {
                self.execute_program_read(program, r, &mut branch_counter);
            }
        }
    }

    fn execute_program_write(
        &mut self,
        program: Program,
        r: &mut Regs,
        branch_counter: &mut V,
        sp: &mut V,
    ) {
        let mut local = Locals::load(r, &program.dst);
        let mut active = M::splat(true);
        loop {
            self.store_local(&local, &program.store_offset, sp, active);
            let branch_flag = self.execute_local_body(program, &mut local, r.r8, active, false);
            let can_branch = branch_counter.simd_gt(V::splat(0));
            let take = if program.unconditional {
                active & can_branch
            } else {
                active & can_branch & (branch_flag & V::splat(32)).simd_eq(V::splat(0))
            };
            *branch_counter = take.select(*branch_counter - V::splat(1), *branch_counter);
            if !take.any() {
                break;
            }
            active = take;
        }
        local.commit(r, &program.dst);
    }

    fn execute_program_read(&self, program: Program, r: &mut Regs, branch_counter: &mut V) {
        let mut local = Locals::load(r, &program.dst);
        let mut active = M::splat(true);
        loop {
            let branch_flag = self.execute_local_body(program, &mut local, r.r8, active, true);
            let can_branch = branch_counter.simd_gt(V::splat(0));
            let take = if program.unconditional {
                active & can_branch
            } else {
                active & can_branch & (branch_flag & V::splat(32)).simd_eq(V::splat(0))
            };
            *branch_counter = take.select(*branch_counter - V::splat(1), *branch_counter);
            if !take.any() {
                break;
            }
            active = take;
        }
        local.commit(r, &program.dst);
    }

    #[inline(always)]
    fn apply(op: Op, old: V, source: V, imm: u8) -> V {
        let shift = u64::from(imm);
        match op {
            Op::MulOr => (old | V::splat(shift)) * source,
            Op::MulXor => (old ^ V::splat(shift)) * source,
            Op::MulAdd => (old + V::splat(shift)) * source,
            Op::XorRor => ror(old, shift) ^ source,
            Op::AddRor => ror(old, shift) + source,
            Op::SubRor => ror(old, shift) - source,
            Op::XorAsr => (old.cast::<i64>() >> i64::from(imm)).cast::<u64>() ^ source,
            Op::AddAsr => (old.cast::<i64>() >> i64::from(imm)).cast::<u64>() + source,
            Op::SubAsr => (old.cast::<i64>() >> i64::from(imm)).cast::<u64>() - source,
            Op::XorLsr => (old >> shift) ^ source,
            Op::AddLsr => (old >> shift) + source,
            Op::SubLsr => (old >> shift) - source,
            Op::Rmcg => unreachable!("RMCG only appears at body index zero"),
        }
    }

    #[inline(always)]
    fn execute_local_body(
        &self,
        program: Program,
        d: &mut Locals,
        r8: V,
        active: M,
        memory: bool,
    ) -> V {
        let rmcg = program.body[0];
        let branch_flag = ror(d.d0 * r8, u64::from(rmcg.imm));
        d.d0 = active.select(branch_flag, d.d0);
        macro_rules! step {
            ($field:ident, $index:expr) => {{
                let instruction = program.body[$index];
                let source_register = d.get(program.src_slot[$index]);
                let source = if memory {
                    self.load(source_register)
                } else {
                    source_register
                };
                let next = Self::apply(instruction.op, d.$field, source, instruction.imm);
                d.$field = active.select(next, d.$field);
            }};
        }
        step!(d1, 1);
        step!(d2, 2);
        step!(d3, 3);
        step!(d4, 4);
        step!(d5, 5);
        step!(d6, 6);
        step!(d7, 7);
        branch_flag
    }

    fn store_local(&mut self, d: &Locals, offsets: &[u8; 8], sp: &mut V, active: M) {
        let next_sp = *sp - V::splat(8);
        *sp = active.select(next_sp, *sp);
        macro_rules! scatter_local {
            ($value:expr, $slot:expr) => {{
                let indices =
                    LANE_BASES + (*sp + V::splat(u64::from(offsets[$slot]))).cast::<usize>();
                let old = V::gather_or_default(&self.scratch, indices);
                active
                    .select($value, old)
                    .scatter(&mut self.scratch, indices);
            }};
        }
        scatter_local!(d.d0, 0);
        scatter_local!(d.d1, 1);
        scatter_local!(d.d2, 2);
        scatter_local!(d.d3, 3);
        scatter_local!(d.d4, 4);
        scatter_local!(d.d5, 5);
        scatter_local!(d.d6, 6);
        scatter_local!(d.d7, 7);
    }

    #[inline(always)]
    fn load(&self, address: V) -> V {
        let byte_address = address & V::splat(WRITABLE_BYTES - 1);
        let word = byte_address >> 3;
        let byte_shift = (byte_address & V::splat(7)) << 3;
        let indices = LANE_BASES + word.cast::<usize>();
        let low = V::gather_or_default(&self.scratch, indices);
        let high = V::gather_or_default(&self.scratch, indices + I::splat(1));
        let high_shift = (V::splat(64) - byte_shift) & V::splat(63);
        let high_part = byte_shift
            .simd_ne(V::splat(0))
            .select(high << high_shift, V::splat(0));
        (low >> byte_shift) | high_part
    }

    fn finalize(&self, r: Regs) -> V {
        let (_, _, _, r3) = sipround_simd(r.r0, r.r1, r.r2, r.r3);
        let (_, _, _, r7) = sipround_simd(r.r4, r.r5, r.r6, r.r7);
        r3 ^ r7 ^ r.r8
    }
}

impl Default for PortableHashWx {
    fn default() -> Self {
        Self::new()
    }
}
