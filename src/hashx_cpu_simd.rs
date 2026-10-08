use std::time::Instant;

use anyhow::Context;

use crate::{
    anubis::{Answer, BackendResult},
    hashx::{Challenge, Instr, describe, valid_prefix},
};

const BATCH: usize = 256;
const SEARCH_LIMIT: u64 = (u32::MAX as u64) + 1;

#[repr(align(32))]
struct Batch<const N: usize> {
    r: [[u64; N]; 8],
    mh: [u64; N],
    allow: [u64; N],
    take: [u64; N],
}

#[inline(always)]
fn sip_round(mut s: [u64; 4]) -> [u64; 4] {
    s[0] = s[0].wrapping_add(s[1]);
    s[2] = s[2].wrapping_add(s[3]);
    s[1] = s[1].rotate_left(13);
    s[3] = s[3].rotate_left(16);
    s[1] ^= s[0];
    s[3] ^= s[2];
    s[0] = s[0].rotate_left(32);
    s[2] = s[2].wrapping_add(s[1]);
    s[0] = s[0].wrapping_add(s[3]);
    s[1] = s[1].rotate_left(17);
    s[3] = s[3].rotate_left(21);
    s[1] ^= s[2];
    s[3] ^= s[0];
    s[2] = s[2].rotate_left(32);
    s
}

#[inline(always)]
fn init_lane(key: [u64; 4], input: u64) -> [u64; 8] {
    let mut s = [key[0], key[1] ^ 0xee, key[2], key[3] ^ input];
    s = sip_round(s);
    s = sip_round(s);
    s[0] ^= input;
    s[2] ^= 0xee;
    s = sip_round(s);
    s = sip_round(s);
    s = sip_round(s);
    s = sip_round(s);
    let mut t = s;
    t[1] ^= 0xdd;
    t = sip_round(t);
    t = sip_round(t);
    t = sip_round(t);
    t = sip_round(t);
    [s[0], s[1], s[2], s[3], t[0], t[1], t[2], t[3]]
}

#[inline(always)]
fn umulhi(a: u64, b: u64) -> u64 {
    let m = 0xffff_ffff;
    let a0 = a & m;
    let a1 = a >> 32;
    let b0 = b & m;
    let b1 = b >> 32;
    let p0 = a0.wrapping_mul(b0);
    let p1 = a0.wrapping_mul(b1);
    let p2 = a1.wrapping_mul(b0);
    let p3 = a1.wrapping_mul(b1);
    let carry = ((p0 >> 32).wrapping_add(p1 & m).wrapping_add(p2 & m)) >> 32;
    p3.wrapping_add(p1 >> 32)
        .wrapping_add(p2 >> 32)
        .wrapping_add(carry)
}

#[inline(always)]
fn smulhi(a: u64, b: u64) -> u64 {
    let h = umulhi(a, b);
    let a_mask = 0_u64.wrapping_sub(a >> 63);
    let b_mask = 0_u64.wrapping_sub(b >> 63);
    h.wrapping_sub(b & a_mask).wrapping_sub(a & b_mask)
}

#[inline(always)]
fn reg_pair<const N: usize>(
    r: &mut [[u64; N]; 8],
    dst: usize,
    src: usize,
) -> (&mut [u64; N], &[u64; N]) {
    debug_assert_ne!(dst, src);
    if dst < src {
        let (left, right) = r.split_at_mut(src);
        (&mut left[dst], &right[0])
    } else {
        let (left, right) = r.split_at_mut(dst);
        (&mut right[0], &left[src])
    }
}

#[inline(always)]
fn full<const N: usize>(batch: &mut Batch<N>, i: Instr) {
    let d = i.dst as usize;
    let s = i.src as usize;
    match i.op {
        0 => {
            let (dst, src) = reg_pair(&mut batch.r, d, s);
            for lane in 0..N {
                dst[lane] = dst[lane].wrapping_mul(src[lane]);
            }
        }
        1 | 2 => {
            if d == s {
                let dst = &mut batch.r[d];
                for lane in 0..N {
                    let old = dst[lane];
                    let x = if i.op == 1 {
                        umulhi(old, old)
                    } else {
                        smulhi(old, old)
                    };
                    dst[lane] = x;
                    batch.mh[lane] = x;
                }
            } else {
                let (dst, src) = reg_pair(&mut batch.r, d, s);
                for lane in 0..N {
                    let x = if i.op == 1 {
                        umulhi(dst[lane], src[lane])
                    } else {
                        smulhi(dst[lane], src[lane])
                    };
                    dst[lane] = x;
                    batch.mh[lane] = x;
                }
            }
        }
        3 => {
            let (dst, src) = reg_pair(&mut batch.r, d, s);
            for lane in 0..N {
                dst[lane] = dst[lane].wrapping_add(src[lane] << i.imm);
            }
        }
        4 => {
            let imm = (i.imm as i32 as i64) as u64;
            for value in &mut batch.r[d] {
                *value = value.wrapping_add(imm);
            }
        }
        5 => {
            let (dst, src) = reg_pair(&mut batch.r, d, s);
            for lane in 0..N {
                dst[lane] = dst[lane].wrapping_sub(src[lane]);
            }
        }
        6 => {
            let (dst, src) = reg_pair(&mut batch.r, d, s);
            for lane in 0..N {
                dst[lane] ^= src[lane];
            }
        }
        7 => {
            let imm = (i.imm as i32 as i64) as u64;
            for value in &mut batch.r[d] {
                *value ^= imm;
            }
        }
        8 => {
            for value in &mut batch.r[d] {
                *value = value.rotate_right(i.imm);
            }
        }
        _ => unreachable!(),
    }
}

#[inline(always)]
fn blend(mask: u64, new: u64, old: u64) -> u64 {
    (new & mask) | (old & !mask)
}

#[inline(always)]
fn masked<const N: usize>(batch: &mut Batch<N>, i: Instr) {
    let d = i.dst as usize;
    let s = i.src as usize;
    match i.op {
        0 => {
            let (dst, src) = reg_pair(&mut batch.r, d, s);
            for lane in 0..N {
                let old = dst[lane];
                dst[lane] = blend(batch.take[lane], old.wrapping_mul(src[lane]), old);
            }
        }
        1 | 2 => {
            if d == s {
                let dst = &mut batch.r[d];
                for lane in 0..N {
                    let old = dst[lane];
                    let x = if i.op == 1 {
                        umulhi(old, old)
                    } else {
                        smulhi(old, old)
                    };
                    let mask = batch.take[lane];
                    dst[lane] = blend(mask, x, old);
                    batch.mh[lane] = blend(mask, x, batch.mh[lane]);
                }
            } else {
                let (dst, src) = reg_pair(&mut batch.r, d, s);
                for lane in 0..N {
                    let old = dst[lane];
                    let x = if i.op == 1 {
                        umulhi(old, src[lane])
                    } else {
                        smulhi(old, src[lane])
                    };
                    let mask = batch.take[lane];
                    dst[lane] = blend(mask, x, old);
                    batch.mh[lane] = blend(mask, x, batch.mh[lane]);
                }
            }
        }
        3 => {
            let (dst, src) = reg_pair(&mut batch.r, d, s);
            for lane in 0..N {
                let old = dst[lane];
                let new = old.wrapping_add(src[lane] << i.imm);
                dst[lane] = blend(batch.take[lane], new, old);
            }
        }
        4 => {
            let imm = (i.imm as i32 as i64) as u64;
            for lane in 0..N {
                let old = batch.r[d][lane];
                batch.r[d][lane] = blend(batch.take[lane], old.wrapping_add(imm), old);
            }
        }
        5 => {
            let (dst, src) = reg_pair(&mut batch.r, d, s);
            for lane in 0..N {
                let old = dst[lane];
                dst[lane] = blend(batch.take[lane], old.wrapping_sub(src[lane]), old);
            }
        }
        6 => {
            let (dst, src) = reg_pair(&mut batch.r, d, s);
            for lane in 0..N {
                let old = dst[lane];
                dst[lane] = blend(batch.take[lane], old ^ src[lane], old);
            }
        }
        7 => {
            let imm = (i.imm as i32 as i64) as u64;
            for lane in 0..N {
                let old = batch.r[d][lane];
                batch.r[d][lane] = blend(batch.take[lane], old ^ imm, old);
            }
        }
        8 => {
            for lane in 0..N {
                let old = batch.r[d][lane];
                batch.r[d][lane] = blend(batch.take[lane], old.rotate_right(i.imm), old);
            }
        }
        _ => unreachable!(),
    }
}

#[inline(always)]
fn hash_batch<const N: usize>(base: u64, key: [u64; 4], program: &[Instr]) -> [u64; N] {
    let mut batch = Batch {
        r: [[0; N]; 8],
        mh: [0; N],
        allow: [u64::MAX; N],
        take: [0; N],
    };
    #[allow(
        clippy::needless_range_loop,
        reason = "indexed SoA lane loop is intentional for LLVM loop vectorization"
    )]
    for lane in 0..N {
        let regs = init_lane(key, base + lane as u64);
        for (reg, value) in regs.into_iter().enumerate() {
            batch.r[reg][lane] = value;
        }
    }

    let mut target = 0usize;
    for (pc, &instruction) in program.iter().enumerate() {
        match instruction.op {
            9 => target = pc,
            10 => {
                let mask = instruction.imm as u64;
                for lane in 0..N {
                    let condition = u64::from((batch.mh[lane] & mask) == 0);
                    let take = batch.allow[lane] & 0_u64.wrapping_sub(condition);
                    batch.take[lane] = take;
                    batch.allow[lane] &= !take;
                }
                // A wide nonce batch almost always has at least one lane taking each
                // branch site. Replaying branchlessly avoids a scalar "any" reduction
                // and keeps every replay loop vectorizable.
                for &replay in &program[target + 1..pc] {
                    masked(&mut batch, replay);
                }
            }
            _ => full(&mut batch, instruction),
        }
    }

    let mut out = [0; N];
    for lane in 0..N {
        let r: [u64; 8] = std::array::from_fn(|reg| batch.r[reg][lane]);
        let x = sip_round([
            r[0].wrapping_add(key[0]),
            r[1].wrapping_add(key[1]),
            r[2],
            r[3],
        ]);
        let y = sip_round([
            r[4],
            r[5],
            r[6].wrapping_add(key[2]),
            r[7].wrapping_add(key[3]),
        ]);
        out[lane] = x[0] ^ y[0];
    }
    out
}

#[inline(always)]
fn word_matches(word: u64, difficulty: u32) -> bool {
    word.swap_bytes().leading_zeros() >= difficulty
}

pub fn solve(challenge: &Challenge, workers: usize) -> anyhow::Result<BackendResult> {
    challenge.validate()?;
    anyhow::ensure!(workers > 0, "auto-vector HashX needs at least one worker");
    let seed = hex::decode(&challenge.data).context("decode HashX challenge")?;
    let started = Instant::now();
    let (hashx, key, program) = describe(&seed)?;
    let found = std::sync::atomic::AtomicU64::new(u64::MAX);
    let stride = (workers * BATCH) as u64;

    std::thread::scope(|scope| {
        for worker in 0..workers {
            let found = &found;
            let program = &program;
            scope.spawn(move || {
                let mut base = (worker * BATCH) as u64;
                while base < SEARCH_LIMIT
                    && found.load(std::sync::atomic::Ordering::Relaxed) == u64::MAX
                {
                    let words = hash_batch::<BATCH>(base, key, program);
                    for (lane, word) in words.into_iter().enumerate() {
                        let nonce = base + lane as u64;
                        if nonce >= SEARCH_LIMIT {
                            break;
                        }
                        if word_matches(word, challenge.difficulty) {
                            let _ = found.compare_exchange(
                                u64::MAX,
                                nonce,
                                std::sync::atomic::Ordering::Relaxed,
                                std::sync::atomic::Ordering::Relaxed,
                            );
                            return;
                        }
                    }
                    base = base.saturating_add(stride);
                }
            });
        }
    });

    let nonce = found.load(std::sync::atomic::Ordering::Relaxed);
    anyhow::ensure!(
        nonce != u64::MAX,
        "auto-vector HashX exhausted u32 nonce space"
    );
    let hash = hashx.hash_to_bytes(nonce);
    anyhow::ensure!(
        valid_prefix(&hash, challenge.difficulty),
        "auto-vector HashX candidate failed scalar validation"
    );
    Ok(BackendResult {
        answer: Answer {
            nonce: nonce.to_string(),
            hash: hex::encode(hash),
        },
        solve_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashx08::HashX;
    use std::hint::black_box;

    fn sample() -> (HashX, [u64; 4], Vec<Instr>) {
        describe(b"anubis-hashx-vulkan-acer-test-seed").unwrap()
    }

    #[test]
    fn auto_matches_native() {
        let (hashx, key, program) = sample();
        for base in (0..4096u64).step_by(BATCH) {
            let got = hash_batch::<BATCH>(base, key, &program);
            for (lane, word) in got.into_iter().enumerate() {
                assert_eq!(word, hashx.hash_to_u64(base + lane as u64));
            }
        }
    }

    #[test]
    #[ignore = "microbenchmark"]
    fn benchmark_auto() {
        let (hashx, key, program) = sample();
        const TOTAL: usize = 1 << 20;
        let started = Instant::now();
        let mut sink = 0u64;
        for base in (0..TOTAL).step_by(BATCH) {
            for word in black_box(hash_batch::<BATCH>(black_box(base as u64), key, &program)) {
                sink ^= word;
            }
        }
        let elapsed = started.elapsed().as_secs_f64();
        eprintln!(
            "auto_vector batch={BATCH} mh_s={:.3} sink={sink:016x}",
            TOTAL as f64 / elapsed / 1e6
        );

        let started = Instant::now();
        let mut sink = 0u64;
        for nonce in 0..TOTAL as u64 {
            sink ^= black_box(hashx.hash_to_u64(black_box(nonce)));
        }
        let elapsed = started.elapsed().as_secs_f64();
        eprintln!(
            "hashx_jit mh_s={:.3} sink={sink:016x}",
            TOTAL as f64 / elapsed / 1e6
        );
    }

    #[test]
    #[ignore = "cachegrind"]
    fn cachegrind_auto() {
        let (_hashx, key, program) = sample();
        const TOTAL: usize = 1 << 15;
        let mut sink = 0u64;
        for base in (0..TOTAL).step_by(BATCH) {
            for word in black_box(hash_batch::<BATCH>(black_box(base as u64), key, &program)) {
                sink ^= word;
            }
        }
        black_box(sink);
    }
}
