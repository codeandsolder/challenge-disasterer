//! Experimental HashWX SM75 direct-SASS compiler.
//!
//! This deliberately reuses the proven HashX fixed cubin shell.  For the
//! correctness prototype the existing 20-byte parameter ABI is reinterpreted as
//! `(workspace: *mut u8, nonce_lo: u32, nonce_hi: u32, unused: u32)` and the
//! kernel writes the 64-bit hash to `workspace[0..8]`.  Scratch starts at
//! `workspace + 16` and is 16_392 bytes for the single worker.

use hashwx_core::{Instruction, Op, Program, ProgramSet};

use super::*;

const HASHWX_TEXT_OFFSET: usize = 0x700;
const HASHWX_TEXT_SIZE: usize = 0x37000;
const HASHWX_TEXT_WORDS: usize = HASHWX_TEXT_SIZE / 16;
const HASHWX_TEMPLATE: &[u8] = include_bytes!("../hashwx_sm75_solve_template.cubin");

const WORKSPACE_SCRATCH_OFFSET: u64 = 16;
const SCRATCH_BYTES: u32 = 16_392;
const SCRATCH_TRACE_BYTES: u32 = 16_384;

// Register allocation. The current HashWX shell advertises 38 GPRs; generated
// code intentionally stays within R0..R34, leaving three registers of metadata headroom.
const WORKSPACE_LO: u8 = 0;
const WORKSPACE_HI: u8 = 1;
const NONCE_LO: u8 = 2;
const NONCE_HI: u8 = 3;
const REPEAT: u8 = 3; // nonce is dead after initialization
const SP: u8 = 4;
const BC: u8 = 5;
const MEM_LO: u8 = 6;
const MEM_HI: u8 = 7;
const ADDR_LO: u8 = 28;
const ADDR_HI: u8 = 29;
const SCRATCH_LO: u8 = 30;
const SCRATCH_HI: u8 = 31;
const R8_LO: u8 = 32;
const R8_HI: u8 = 33;
const SHIFT: u8 = 34;
const WORKSPACE_UR: u8 = 4;

const P_BRANCH: Pred = Pred::P2;

#[derive(Clone, Copy, PartialEq, Eq)]
enum BuildMode {
    InitOnly,
    WriteOnly,
    Full,
}

#[derive(Clone, Copy)]
enum ShiftMode {
    RightU64,
    RightU32Hi,
    RightS32Hi,
    LeftU64Hi,
    LeftU32,
}

impl ShiftMode {
    const fn bits(self) -> u128 {
        match self {
            Self::RightU64 => 0x1200,
            Self::RightU32Hi => 0x1_1600,
            Self::RightS32Hi => 0x1_1400,
            Self::LeftU64Hi => 0x1_0200,
            Self::LeftU32 => 0x0600,
        }
    }
}

fn shf_reg(
    mode: ShiftMode,
    rd: Reg,
    rs0: Reg,
    amount: Reg,
    rs2: Reg,
    guard: Pred,
    ctrl: Ctrl,
) -> u128 {
    let mut code = base(0x219, guard, ctrl);
    code |= u128::from(rd.bits()) << 80;
    code |= u128::from(rs0.bits()) << 88;
    code |= u128::from(rs2.bits());
    code |= mode.bits();
    put_src1(&mut code, Src1::Reg(amount));
    code
}

fn shf_imm(
    mode: ShiftMode,
    rd: Reg,
    rs0: Reg,
    amount: u8,
    rs2: Reg,
    guard: Pred,
    stall: u8,
) -> u128 {
    let mut code = base(0x219, guard, Ctrl::standard(stall));
    code |= u128::from(rd.bits()) << 80;
    code |= u128::from(rs0.bits()) << 88;
    code |= u128::from(rs2.bits());
    code |= mode.bits();
    put_src1(&mut code, Src1::Imm(u32::from(amount)));
    code
}

fn uldc64_c0(rd: u8, offset: u32) -> u128 {
    debug_assert_eq!(offset & 3, 0);
    let mut code = base(
        0xab9,
        Pred::PT,
        Ctrl {
            wait: 0,
            read: 7,
            write: 7,
            yield_hint: true,
            stall: 4,
        },
    );
    code |= u128::from(rd) << 80;
    code |= u128::from(offset / 4) << 104;
    code | 0x0a00
}

fn ldg64_u32_uniform(
    rd: u8,
    index: u8,
    base_ur: u8,
    offset: u32,
    guard: Pred,
    write_barrier: u8,
    stall: u8,
) -> u128 {
    let mut code = base(
        0x981,
        guard,
        Ctrl {
            wait: 0,
            read: 7,
            write: write_barrier,
            yield_hint: false,
            stall,
        },
    );
    code |= u128::from(rd) << 80;
    code |= u128::from(index) << 88;
    code |= u128::from(base_ur) << 96;
    code |= u128::from(offset) << 104;
    code | 0x081e_eb00
}

fn stg64(addr: u8, src: u8, offset: u32) -> u128 {
    let mut code = base(0x386, Pred::PT, Ctrl::standard(4));
    code |= u128::from(addr) << 88;
    code |= u128::from(src) << 96;
    code |= u128::from(offset) << 104;
    code | 0x0010_eb00
}

fn branch_word(guard: Pred, rel_bytes: i32) -> u128 {
    let ctrl = Ctrl {
        wait: 0,
        read: 7,
        write: 7,
        yield_hint: true,
        stall: 0,
    };
    let mut code = base(0x947, guard, ctrl);
    set_icr(&mut code, 0x8);
    code |= u128::from(rel_bytes as u32) << 96;
    if rel_bytes < 0 {
        code |= 0x3ffff;
    }
    put_pred(&mut code, Pred::PT, 23);
    code
}

fn membar_cta() -> u128 {
    // ptxas sm_75: MEMBAR.SC.CTA
    base(0x992, Pred::PT, Ctrl::standard(5))
}

fn patch_branch(words: &mut [u128], at: usize, target: usize, guard: Pred) {
    let source_next = (at + 1) * 16;
    let target = target * 16;
    let rel = i32::try_from(target as isize - source_next as isize).expect("SM75 branch fits i32");
    words[at] = branch_word(guard, rel);
}

fn pair(i: u8) -> (u8, u8) {
    super::pair(i)
}

impl Emitter {
    fn hashwx_predicate_delay(&mut self) {
        // SM75 predicate producer -> guarded consumer needs explicit distance.
        // This mirrors the six-NOP spacing validated by the existing HashX
        // direct-SASS solve tail.
        for _ in 0..6 {
            self.emit(nop(1));
        }
    }

    fn hashwx_addr_from_u32(&mut self, offset: u8) {
        self.emit(iadd3(
            r(ADDR_LO),
            Some(Pred::P0),
            None,
            r(SCRATCH_LO),
            false,
            Src1::Reg(r(offset)),
            Reg::Z,
            None,
            None,
            false,
            Pred::PT,
            4,
        ));
        self.emit(iadd3(
            r(ADDR_HI),
            None,
            None,
            r(SCRATCH_HI),
            false,
            Src1::Imm(0),
            Reg::Z,
            Some(Pred::P0),
            Some(Pred::NPT),
            true,
            Pred::PT,
            5,
        ));
    }

    fn hashwx_sub64(&mut self, dlo: u8, dhi: u8, slo: u8, shi: u8) {
        self.emit(lop3(
            r(ADDR_LO),
            Reg::Z,
            Src1::Reg(r(shi)),
            Reg::Z,
            0x33,
            Pred::NPT,
            Pred::PT,
            5,
        ));
        self.emit(iadd3(
            r(dlo),
            Some(Pred::P0),
            None,
            r(slo),
            true,
            Src1::Reg(r(dlo)),
            Reg::Z,
            None,
            None,
            false,
            Pred::PT,
            4,
        ));
        self.emit(iadd3(
            r(dhi),
            None,
            None,
            r(dhi),
            false,
            Src1::Reg(r(ADDR_LO)),
            Reg::Z,
            Some(Pred::P0),
            Some(Pred::NPT),
            true,
            Pred::PT,
            5,
        ));
    }

    fn hashwx_lsr64(&mut self, lo: u8, hi: u8, amount: u8) {
        self.emit(shf_imm(
            ShiftMode::RightU64,
            r(24),
            r(lo),
            amount,
            r(hi),
            Pred::PT,
            5,
        ));
        self.emit(shf_imm(
            ShiftMode::RightU32Hi,
            r(25),
            Reg::Z,
            amount,
            r(hi),
            Pred::PT,
            5,
        ));
        self.mov(lo, Src1::Reg(r(24)), Pred::PT);
        self.mov(hi, Src1::Reg(r(25)), Pred::PT);
    }

    fn hashwx_asr64(&mut self, lo: u8, hi: u8, amount: u8) {
        self.emit(shf(
            ShfKind::RightU64,
            r(24),
            r(lo),
            amount,
            r(hi),
            Pred::PT,
            5,
        ));
        self.emit(shf_imm(
            ShiftMode::RightS32Hi,
            r(25),
            Reg::Z,
            amount,
            r(hi),
            Pred::PT,
            5,
        ));
        self.mov(lo, Src1::Reg(r(24)), Pred::PT);
        self.mov(hi, Src1::Reg(r(25)), Pred::PT);
    }

    fn hashwx_load_source(&mut self, src_lo: u8) {
        // Byte offset within the 16 KiB trace and its aligned-down address.
        self.lop(SHIFT, src_lo, Src1::Imm(16_383), 0xc0, Pred::PT);
        self.lop(ADDR_LO, SHIFT, Src1::Imm(0xffff_fff8), 0xc0, Pred::PT);

        // Match ptxas's robust SM75 addressing shape for a dynamic unaligned
        // u64 load: 32-bit byte offset + uniform 64-bit base.  The previous
        // GPR-pair IADD3/IADD3.X -> LDG path needed six artificial NOPs and
        // still showed at-speed dependency failures hidden by sanitizer.
        self.emit(ldg64_u32_uniform(
            MEM_LO,
            ADDR_LO,
            WORKSPACE_UR,
            WORKSPACE_SCRATCH_OFFSET as u32,
            Pred::PT,
            2,
            4,
        ));
        self.emit(ldg64_u32_uniform(
            24,
            ADDR_LO,
            WORKSPACE_UR,
            WORKSPACE_SCRATCH_OFFSET as u32 + 8,
            Pred::PT,
            3,
            1,
        ));

        // Shift count in bits, 0..56.  This independent arithmetic fills load
        // latency; the first consumers below wait on the exact LDG barriers.
        self.lop(SHIFT, SHIFT, Src1::Imm(7), 0xc0, Pred::PT);
        self.emit(imad(
            false,
            false,
            r(SHIFT),
            None,
            r(SHIFT),
            Src1::Imm(8),
            Reg::Z,
            None,
            Pred::PT,
            Ctrl {
                wait: 0,
                read: 7,
                write: 7,
                yield_hint: false,
                stall: 4,
            },
        ));

        // Keep the alignment predicate out of the funnel itself.  Compute both
        // halves unconditionally, then select the original aligned load for s=0.
        self.mov(ADDR_HI, Src1::Reg(r(SHIFT)), Pred::PT);
        self.emit(isetp(
            Cmp::Ne,
            true,
            P_BRANCH,
            Pred::PT,
            r(ADDR_HI),
            Src1::Imm(0),
            Pred::PT,
            Pred::PT,
            4,
        ));
        // SHIFT = 64 - s; ADDR_HI retains s until the right-shift pair is done.
        self.emit(iadd3(
            r(SHIFT),
            None,
            None,
            r(ADDR_HI),
            true,
            Src1::Imm(64),
            Reg::Z,
            None,
            None,
            false,
            Pred::PT,
            5,
        ));
        self.emit(shf_reg(
            ShiftMode::RightU64,
            r(26),
            r(MEM_LO),
            r(ADDR_HI),
            r(MEM_HI),
            Pred::PT,
            Ctrl {
                wait: 0b000100,
                read: 7,
                write: 7,
                yield_hint: false,
                stall: 2,
            },
        ));
        self.emit(shf_reg(
            ShiftMode::RightU32Hi,
            r(27),
            Reg::Z,
            r(ADDR_HI),
            r(MEM_HI),
            Pred::PT,
            Ctrl::standard(2),
        ));
        self.emit(shf_reg(
            ShiftMode::LeftU64Hi,
            r(ADDR_LO),
            r(24),
            r(SHIFT),
            r(25),
            Pred::PT,
            Ctrl {
                wait: 0b001000,
                read: 7,
                write: 7,
                yield_hint: false,
                stall: 2,
            },
        ));
        // ADDR_HI is dead as the saved right-shift amount after the two SHF.Rs.
        self.emit(shf_reg(
            ShiftMode::LeftU32,
            r(ADDR_HI),
            r(24),
            r(SHIFT),
            Reg::Z,
            Pred::PT,
            Ctrl::standard(2),
        ));
        self.lop(26, 26, Src1::Reg(r(ADDR_HI)), 0xfc, Pred::PT);
        self.lop(27, 27, Src1::Reg(r(ADDR_LO)), 0xfc, Pred::PT);
        self.emit(sel(
            r(MEM_LO),
            r(26),
            Src1::Reg(r(MEM_LO)),
            P_BRANCH,
            Pred::PT,
            5,
        ));
        self.emit(sel(
            r(MEM_HI),
            r(27),
            Src1::Reg(r(MEM_HI)),
            P_BRANCH,
            Pred::PT,
            5,
        ));
    }

    fn hashwx_op(&mut self, x: Instruction, memory: bool) {
        let (dlo, dhi) = pair(x.dst);
        let (slo, shi) = pair(x.src);
        let (src_lo, src_hi) = if memory && !matches!(x.op, Op::Rmcg) {
            self.hashwx_load_source(slo);
            (MEM_LO, MEM_HI)
        } else {
            (slo, shi)
        };

        match x.op {
            Op::MulOr | Op::MulXor | Op::MulAdd => {
                match x.op {
                    Op::MulOr => self.lop(dlo, dlo, Src1::Imm(u32::from(x.imm)), 0xfc, Pred::PT),
                    Op::MulXor => self.lop(dlo, dlo, Src1::Imm(u32::from(x.imm)), 0x3c, Pred::PT),
                    Op::MulAdd => self.add64_const(dlo, dhi, u64::from(x.imm), Pred::PT),
                    _ => unreachable!(),
                }
                self.mullo(dlo, dhi, dlo, dhi, src_lo, src_hi, Pred::PT);
            }
            Op::Rmcg => {
                self.mullo(dlo, dhi, dlo, dhi, R8_LO, R8_HI, Pred::PT);
                self.ror64(dlo, dhi, x.imm, Pred::PT);
            }
            _ => {
                match x.op {
                    Op::XorRor | Op::AddRor | Op::SubRor => self.ror64(dlo, dhi, x.imm, Pred::PT),
                    Op::XorAsr | Op::AddAsr | Op::SubAsr => self.hashwx_asr64(dlo, dhi, x.imm),
                    Op::XorLsr | Op::AddLsr | Op::SubLsr => self.hashwx_lsr64(dlo, dhi, x.imm),
                    _ => unreachable!(),
                }
                match x.op {
                    Op::XorRor | Op::XorAsr | Op::XorLsr => {
                        self.xor64_reg(dlo, dhi, src_lo, src_hi, Pred::PT)
                    }
                    Op::AddRor | Op::AddAsr | Op::AddLsr => {
                        self.add64_reg(dlo, dhi, src_lo, src_hi, Pred::PT)
                    }
                    Op::SubRor | Op::SubAsr | Op::SubLsr => {
                        self.hashwx_sub64(dlo, dhi, src_lo, src_hi)
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    fn hashwx_snapshot(&mut self, wrap_address: bool) {
        self.emit(iadd3(
            r(SP),
            None,
            None,
            r(SP),
            false,
            Src1::Imm(u32::MAX - 63),
            Reg::Z,
            None,
            None,
            false,
            Pred::PT,
            4,
        ));
        if wrap_address {
            self.lop(ADDR_LO, SP, Src1::Imm(16_383), 0xc0, Pred::PT);
            self.hashwx_addr_from_u32(ADDR_LO);
        } else {
            self.hashwx_addr_from_u32(SP);
        }
        for logical in 0..8u8 {
            let src = pair(7 - logical).0;
            self.emit(stg64(ADDR_LO, src, u32::from(logical) * 8));
        }
    }

    fn hashwx_program(&mut self, p: &Program, memory: bool, wrap_snapshot: bool) {
        let start = self.words.len();
        if !memory {
            self.hashwx_snapshot(wrap_snapshot);
        }
        for &x in &p.body {
            self.hashwx_op(x, memory);
        }

        self.emit(isetp(
            Cmp::Eq,
            true,
            P_BRANCH,
            Pred::PT,
            r(BC),
            Src1::Imm(0),
            Pred::PT,
            Pred::PT,
            4,
        ));
        self.hashwx_predicate_delay();
        let bc_done = self.words.len();
        self.emit(0);

        let mut bf_done = None;
        if !p.unconditional {
            let (bf, _) = pair(p.body[0].dst);
            self.lop(SHIFT, bf, Src1::Imm(32), 0xc0, Pred::PT);
            self.emit(isetp(
                Cmp::Ne,
                true,
                P_BRANCH,
                Pred::PT,
                r(SHIFT),
                Src1::Imm(0),
                Pred::PT,
                Pred::PT,
                4,
            ));
            self.hashwx_predicate_delay();
            bf_done = Some(self.words.len());
            self.emit(0);
        }

        self.emit(iadd3(
            r(BC),
            None,
            None,
            r(BC),
            false,
            Src1::Imm(u32::MAX),
            Reg::Z,
            None,
            None,
            false,
            Pred::PT,
            4,
        ));
        let back = self.words.len();
        self.emit(branch_word(Pred::PT, 0));
        let done = self.words.len();
        patch_branch(&mut self.words, bc_done, done, P_BRANCH);
        if let Some(at) = bf_done {
            patch_branch(&mut self.words, at, done, P_BRANCH);
        }
        patch_branch(&mut self.words, back, start, Pred::PT);
    }

    fn hashwx_phase(&mut self, programs: &[Program; 32], memory: bool, wrap_snapshot: bool) {
        self.mov(REPEAT, Src1::Imm(4), Pred::PT);
        let repeat = self.words.len();
        self.mov(BC, Src1::Imm(32), Pred::PT);
        for p in programs {
            self.hashwx_program(p, memory, wrap_snapshot);
        }
        self.emit(iadd3(
            r(REPEAT),
            None,
            None,
            r(REPEAT),
            false,
            Src1::Imm(u32::MAX),
            Reg::Z,
            None,
            None,
            false,
            Pred::PT,
            4,
        ));
        self.emit(isetp(
            Cmp::Ne,
            true,
            P_BRANCH,
            Pred::PT,
            r(REPEAT),
            Src1::Imm(0),
            Pred::PT,
            Pred::PT,
            4,
        ));
        self.hashwx_predicate_delay();
        let back = self.words.len();
        self.emit(branch_word(P_BRANCH, 0));
        patch_branch(&mut self.words, back, repeat, P_BRANCH);
    }
}

fn finish_debug(mut e: Emitter) -> anyhow::Result<Vec<u128>> {
    // R8 is a convenient deterministic initialized value; debug stages only need
    // to prove the control/memory prefix reaches EXIT without a GPU fault.
    e.emit(stg64(WORKSPACE_LO, R8_LO, 0));
    e.emit(exit(2));
    let natural_len = e.words.len();
    anyhow::ensure!(
        natural_len < HASHWX_TEXT_WORDS - 5,
        "sm_75 HashWX debug kernel too large: {natural_len} words"
    );
    while e.words.len() < HASHWX_TEXT_WORDS - 5 {
        e.emit(nop(1));
    }
    e.emit(tail_bra());
    while e.words.len() < HASHWX_TEXT_WORDS {
        e.emit(tail_nop());
    }
    Ok(e.words)
}

fn generate(set: &ProgramSet, mode: BuildMode) -> anyhow::Result<Vec<u128>> {
    let [k0, k1] = set.hash_key;
    let mut e = Emitter::new();

    // Existing HashX shell parameter layout: ptr @ 0x160, then three u32s.
    e.mov(
        WORKSPACE_LO,
        Src1::Const {
            bank: 0,
            offset: 0x160,
        },
        Pred::PT,
    );
    e.mov(
        WORKSPACE_HI,
        Src1::Const {
            bank: 0,
            offset: 0x164,
        },
        Pred::PT,
    );
    // Uniform copy of the same kernel parameter for scratch reads. ptxas uses
    // this U32-offset + UR-base form for dynamic unaligned global loads on SM75.
    e.emit(uldc64_c0(WORKSPACE_UR, 0x160));
    e.mov(
        NONCE_LO,
        Src1::Const {
            bank: 0,
            offset: 0x168,
        },
        Pred::PT,
    );
    e.mov(
        NONCE_HI,
        Src1::Const {
            bank: 0,
            offset: 0x16c,
        },
        Pred::PT,
    );

    e.copy64(SCRATCH_LO, SCRATCH_HI, WORKSPACE_LO, WORKSPACE_HI, Pred::PT);
    e.add64_const(SCRATCH_LO, SCRATCH_HI, WORKSPACE_SCRATCH_OFFSET, Pred::PT);

    // SipHash-like nonce initialization in logical pairs 4..7.
    for (logical, value) in [
        0x736f_6d65_7073_6575 ^ k0,
        0x646f_7261_6e64_6f6d ^ k1,
        0x6c79_6765_6e65_7261 ^ k0,
        0x7465_6462_7974_6573 ^ k1,
    ]
    .into_iter()
    .enumerate()
    {
        let (lo, hi) = pair(4 + logical as u8);
        e.mov(lo, Src1::Imm(value as u32), Pred::PT);
        e.mov(hi, Src1::Imm((value >> 32) as u32), Pred::PT);
    }
    e.xor64_reg(pair(7).0, pair(7).1, NONCE_LO, NONCE_HI, Pred::PT);
    e.sip([4, 5, 6, 7], Pred::PT);
    e.xor64_reg(pair(4).0, pair(4).1, NONCE_LO, NONCE_HI, Pred::PT);
    e.xor64_const(pair(6).0, pair(6).1, 0xbb, Pred::PT);
    for _ in 0..3 {
        e.sip([4, 5, 6, 7], Pred::PT);
    }
    for i in 0..4u8 {
        let (dlo, dhi) = pair(i);
        let (slo, shi) = pair(7 - i);
        e.copy64(dlo, dhi, slo, shi, Pred::PT);
    }
    e.xor64_const(pair(4).0, pair(4).1, k0, Pred::PT);
    e.xor64_const(pair(5).0, pair(5).1, k1, Pred::PT);
    e.xor64_const(pair(6).0, pair(6).1, k0, Pred::PT);
    e.xor64_const(pair(7).0, pair(7).1, k1, Pred::PT);
    for _ in 0..4 {
        e.sip([4, 5, 6, 7], Pred::PT);
    }
    // Reverse the second group into logical R4..R7.
    e.copy64(R8_LO, R8_HI, pair(4).0, pair(4).1, Pred::PT);
    e.copy64(pair(4).0, pair(4).1, pair(7).0, pair(7).1, Pred::PT);
    e.copy64(pair(7).0, pair(7).1, R8_LO, R8_HI, Pred::PT);
    e.copy64(R8_LO, R8_HI, pair(5).0, pair(5).1, Pred::PT);
    e.copy64(pair(5).0, pair(5).1, pair(6).0, pair(6).1, Pred::PT);
    e.copy64(pair(6).0, pair(6).1, R8_LO, R8_HI, Pred::PT);

    // R8 = ((R0 ^ R4) & !7) | 3.
    e.copy64(R8_LO, R8_HI, pair(0).0, pair(0).1, Pred::PT);
    e.xor64_reg(R8_LO, R8_HI, pair(4).0, pair(4).1, Pred::PT);
    e.lop(R8_LO, R8_LO, Src1::Imm(0xffff_fff8), 0xc0, Pred::PT);
    e.lop(R8_LO, R8_LO, Src1::Imm(3), 0xfc, Pred::PT);
    e.emit(stg64(SCRATCH_LO, R8_LO, SCRATCH_TRACE_BYTES));
    if mode == BuildMode::InitOnly {
        return finish_debug(e);
    }

    e.mov(SP, Src1::Imm(SCRATCH_TRACE_BYTES), Pred::PT);
    e.hashwx_phase(&set.programs, false, mode == BuildMode::WriteOnly);
    if mode != BuildMode::WriteOnly {
        e.emit(membar_cta());
    }
    if mode == BuildMode::WriteOnly {
        e.emit(stg64(WORKSPACE_LO, SP, 0));
        e.emit(exit(2));
        let natural_len = e.words.len();
        anyhow::ensure!(
            natural_len < HASHWX_TEXT_WORDS - 5,
            "debug kernel too large"
        );
        while e.words.len() < HASHWX_TEXT_WORDS - 5 {
            e.emit(nop(1));
        }
        e.emit(tail_bra());
        while e.words.len() < HASHWX_TEXT_WORDS {
            e.emit(tail_nop());
        }
        return Ok(e.words);
    }
    e.hashwx_phase(&set.programs, true, false);

    e.sip([0, 1, 2, 3], Pred::PT);
    e.sip([4, 5, 6, 7], Pred::PT);
    e.copy64(MEM_LO, MEM_HI, pair(3).0, pair(3).1, Pred::PT);
    e.xor64_reg(MEM_LO, MEM_HI, pair(7).0, pair(7).1, Pred::PT);
    e.xor64_reg(MEM_LO, MEM_HI, R8_LO, R8_HI, Pred::PT);
    e.emit(stg64(WORKSPACE_LO, MEM_LO, 0));
    e.emit(exit(2));

    let natural_len = e.words.len();
    anyhow::ensure!(
        natural_len < HASHWX_TEXT_WORDS - 5,
        "sm_75 HashWX kernel too large: {natural_len} words"
    );
    while e.words.len() < HASHWX_TEXT_WORDS - 5 {
        e.emit(nop(1));
    }
    e.emit(tail_bra());
    while e.words.len() < HASHWX_TEXT_WORDS {
        e.emit(tail_nop());
    }
    Ok(e.words)
}

pub(crate) fn build_solve_cubin(set: &ProgramSet) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        HASHWX_TEMPLATE.len() >= HASHWX_TEXT_OFFSET + HASHWX_TEXT_SIZE,
        "embedded HashWX sm_75 cubin template is truncated"
    );
    let words = generate(set, BuildMode::Full)?;
    let mut cubin = HASHWX_TEMPLATE.to_vec();
    let text = &mut cubin[HASHWX_TEXT_OFFSET..HASHWX_TEXT_OFFSET + HASHWX_TEXT_SIZE];
    for (i, &word) in words.iter().enumerate() {
        let slot = &mut text[i * 16..(i + 1) * 16];
        slot[..8].copy_from_slice(&((word >> 64) as u64).to_le_bytes());
        slot[8..].copy_from_slice(&(word as u64).to_le_bytes());
    }
    Ok(cubin)
}

pub(crate) fn build_debug_cubin(set: &ProgramSet, stage: u8) -> anyhow::Result<Vec<u8>> {
    let mode = match stage {
        0 => BuildMode::InitOnly,
        1 => BuildMode::WriteOnly,
        _ => BuildMode::Full,
    };
    let words = generate(set, mode)?;
    let mut cubin = HASHWX_TEMPLATE.to_vec();
    let text = &mut cubin[HASHWX_TEXT_OFFSET..HASHWX_TEXT_OFFSET + HASHWX_TEXT_SIZE];
    for (i, &word) in words.iter().enumerate() {
        let slot = &mut text[i * 16..(i + 1) * 16];
        slot[..8].copy_from_slice(&((word >> 64) as u64).to_le_bytes());
        slot[8..].copy_from_slice(&(word as u64).to_le_bytes());
    }
    Ok(cubin)
}

pub(crate) const fn workspace_bytes() -> usize {
    WORKSPACE_SCRATCH_OFFSET as usize + SCRATCH_BYTES as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branch_encoding_matches_ptxas_sm75() {
        let expected = (0x0000_0030_0000_0947_u128 << 64) | 0x000f_c000_0380_0000_u128;
        assert_eq!(branch_word(Pred::P0, 0x30), expected);
    }

    #[test]
    fn membar_cta_encoding_matches_ptxas_sm75() {
        let expected = (0x0000_0000_0000_7992_u128 << 64) | 0x000f_ea00_0000_0000_u128;
        assert_eq!(membar_cta(), expected);
    }

    #[test]
    fn uniform_scratch_load_encodings_match_ptxas_sm75() {
        let uldc = (0x0000_5800_0004_7ab9_u128 << 64) | 0x000f_c800_0000_0a00_u128;
        assert_eq!(uldc64_c0(4, 0x160), uldc);

        let first = (0x0000_1004_0008_7981_u128 << 64) | 0x000e_a800_081e_eb00_u128;
        assert_eq!(ldg64_u32_uniform(8, 0, 4, 0x10, Pred::PT, 2, 4), first);

        let second = (0x0000_1804_0002_0981_u128 << 64) | 0x000e_e200_081e_eb00_u128;
        assert_eq!(ldg64_u32_uniform(2, 0, 4, 0x18, Pred::P0, 3, 1), second);

        let right = (0x0000_0004_0807_0219_u128 << 64) | 0x004f_e400_0000_1209_u128;
        assert_eq!(
            shf_reg(
                ShiftMode::RightU64,
                r(7),
                r(8),
                r(4),
                r(9),
                Pred::P0,
                Ctrl {
                    wait: 0b000100,
                    read: 7,
                    write: 7,
                    yield_hint: false,
                    stall: 2,
                },
            ),
            right
        );

        let left = (0x0000_0005_0203_0219_u128 << 64) | 0x008f_e400_0001_0203_u128;
        assert_eq!(
            shf_reg(
                ShiftMode::LeftU64Hi,
                r(3),
                r(2),
                r(5),
                r(3),
                Pred::P0,
                Ctrl {
                    wait: 0b001000,
                    read: 7,
                    write: 7,
                    yield_hint: false,
                    stall: 2,
                },
            ),
            left
        );
    }

    #[test]
    fn generated_program_corpus_fits_shell() -> anyhow::Result<()> {
        let mut max_used = 0usize;
        let mut max_seed = 0u64;
        for seed_number in 0u64..4096 {
            let mut seed = [0u8; 32];
            seed[..8].copy_from_slice(&seed_number.to_le_bytes());
            let words = generate(&hashwx_core::generate(&seed), BuildMode::Full)?;
            let used = words.iter().position(|&word| word == exit(2)).unwrap() + 1;
            if used > max_used {
                max_used = used;
                max_seed = seed_number;
            }
        }
        eprintln!("HashWX SM75 corpus max words={max_used} seed={max_seed}");
        assert!(max_used < HASHWX_TEXT_WORDS - 5);
        Ok(())
    }

    #[test]
    fn generated_reference_program_fits_shell() -> anyhow::Result<()> {
        let mut seed = [0u8; 32];
        for (i, b) in seed.iter_mut().enumerate() {
            *b = i as u8;
        }
        let set = hashwx_core::generate(&seed);
        let words = generate(&set, BuildMode::Full)?;
        let used = words.iter().position(|&word| word == exit(2)).unwrap() + 1;
        eprintln!("HashWX SM75 used words={used}");
        assert!(used < HASHWX_TEXT_WORDS - 5);
        Ok(())
    }
}
