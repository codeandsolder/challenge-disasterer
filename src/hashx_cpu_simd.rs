use std::{
    simd::{Mask, Select, Simd, cmp::SimdPartialEq},
    time::Instant,
};

use anyhow::Context;

use crate::{
    anubis::{Answer, BackendResult},
    hashx_gpu::{Challenge, Instr, describe, valid_prefix},
};

type V<const N: usize> = Simd<u64, N>;
type M<const N: usize> = Mask<i64, N>;

const LANES: usize = 32;
const SEARCH_LIMIT: u64 = (u32::MAX as u64) + 1;

#[derive(Clone, Copy)]
struct State<const N: usize> {
    v0: V<N>,
    v1: V<N>,
    v2: V<N>,
    v3: V<N>,
}

#[inline(always)]
fn vv<const N: usize>(x: u64) -> V<N> {
    V::splat(x)
}

#[inline(always)]
fn rol<const N: usize>(x: V<N>, n: u32) -> V<N> {
    if n == 0 {
        x
    } else {
        (x << vv(n as u64)) | (x >> vv((64 - n) as u64))
    }
}

#[inline(always)]
fn ror<const N: usize>(x: V<N>, n: u32) -> V<N> {
    if n == 0 {
        x
    } else {
        (x >> vv(n as u64)) | (x << vv((64 - n) as u64))
    }
}

#[inline(always)]
fn sip<const N: usize>(s: State<N>) -> State<N> {
    let mut v0 = s.v0 + s.v1;
    let mut v2 = s.v2 + s.v3;
    let mut v1 = rol(s.v1, 13);
    let mut v3 = rol(s.v3, 16);
    v1 ^= v0;
    v3 ^= v2;
    v0 = rol(v0, 32);
    v2 += v1;
    v0 += v3;
    v1 = rol(v1, 17);
    v3 = rol(v3, 21);
    v1 ^= v2;
    v3 ^= v0;
    v2 = rol(v2, 32);
    State { v0, v1, v2, v3 }
}

#[inline(always)]
fn init<const N: usize>(key: [u64; 4], input: V<N>) -> [V<N>; 8] {
    let mut s = State {
        v0: vv(key[0]),
        v1: vv(key[1] ^ 0xee),
        v2: vv(key[2]),
        v3: vv(key[3]) ^ input,
    };
    s = sip(s);
    s = sip(s);
    s.v0 ^= input;
    s.v2 ^= vv(0xee);
    s = sip(s);
    s = sip(s);
    s = sip(s);
    s = sip(s);
    let mut t = s;
    t.v1 ^= vv(0xdd);
    t = sip(t);
    t = sip(t);
    t = sip(t);
    t = sip(t);
    [s.v0, s.v1, s.v2, s.v3, t.v0, t.v1, t.v2, t.v3]
}

#[inline(always)]
fn umulhi<const N: usize>(a: V<N>, b: V<N>) -> V<N> {
    let m = vv(0xffff_ffff);
    let a0 = a & m;
    let a1 = a >> vv(32);
    let b0 = b & m;
    let b1 = b >> vv(32);
    let p0 = a0 * b0;
    let p1 = a0 * b1;
    let p2 = a1 * b0;
    let p3 = a1 * b1;
    let carry = ((p0 >> vv(32)) + (p1 & m) + (p2 & m)) >> vv(32);
    p3 + (p1 >> vv(32)) + (p2 >> vv(32)) + carry
}

#[inline(always)]
fn smulhi<const N: usize>(a: V<N>, b: V<N>) -> V<N> {
    let mut h = umulhi(a, b);
    let a_neg = (a >> vv(63)).simd_eq(vv(1));
    let b_neg = (b >> vv(63)).simd_eq(vv(1));
    h = a_neg.select(h - b, h);
    b_neg.select(h - a, h)
}

#[inline(always)]
fn full<const N: usize>(r: &mut [V<N>; 8], mh: &mut V<N>, i: Instr) {
    let d = i.dst as usize;
    let s = i.src as usize;
    match i.op {
        0 => r[d] *= r[s],
        1 => {
            let x = umulhi(r[d], r[s]);
            r[d] = x;
            *mh = x;
        }
        2 => {
            let x = smulhi(r[d], r[s]);
            r[d] = x;
            *mh = x;
        }
        3 => r[d] += r[s] << vv(i.imm as u64),
        4 => r[d] += vv((i.imm as i32 as i64) as u64),
        5 => r[d] -= r[s],
        6 => r[d] ^= r[s],
        7 => r[d] ^= vv((i.imm as i32 as i64) as u64),
        8 => r[d] = ror(r[d], i.imm),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn masked<const N: usize>(r: &mut [V<N>; 8], mh: &mut V<N>, i: Instr, mask: M<N>) {
    let d = i.dst as usize;
    let s = i.src as usize;
    let old = r[d];
    match i.op {
        0 => r[d] = mask.select(old * r[s], old),
        1 => {
            let x = umulhi(old, r[s]);
            r[d] = mask.select(x, old);
            *mh = mask.select(x, *mh);
        }
        2 => {
            let x = smulhi(old, r[s]);
            r[d] = mask.select(x, old);
            *mh = mask.select(x, *mh);
        }
        3 => r[d] = mask.select(old + (r[s] << vv(i.imm as u64)), old),
        4 => r[d] = mask.select(old + vv((i.imm as i32 as i64) as u64), old),
        5 => r[d] = mask.select(old - r[s], old),
        6 => r[d] = mask.select(old ^ r[s], old),
        7 => r[d] = mask.select(old ^ vv((i.imm as i32 as i64) as u64), old),
        8 => r[d] = mask.select(ror(old, i.imm), old),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn run_program<const N: usize>(mut r: [V<N>; 8], program: &[Instr]) -> [V<N>; 8] {
    let mut mh = vv(0);
    let mut allow = M::splat(true);
    let mut target = 0usize;
    for (pc, &instruction) in program.iter().enumerate() {
        match instruction.op {
            9 => target = pc,
            10 => {
                let take = allow & (mh & vv(instruction.imm as u64)).simd_eq(vv(0));
                if take.any() {
                    allow &= !take;
                    for &replay in &program[target + 1..pc] {
                        masked(&mut r, &mut mh, replay, take);
                    }
                }
            }
            _ => full(&mut r, &mut mh, instruction),
        }
    }
    r
}

#[inline(always)]
fn digest0<const N: usize>(r: [V<N>; 8], key: [u64; 4]) -> V<N> {
    let x = sip(State {
        v0: r[0] + vv(key[0]),
        v1: r[1] + vv(key[1]),
        v2: r[2],
        v3: r[3],
    });
    let y = sip(State {
        v0: r[4],
        v1: r[5],
        v2: r[6] + vv(key[2]),
        v3: r[7] + vv(key[3]),
    });
    x.v0 ^ y.v0
}

#[inline(always)]
fn hash_batch<const N: usize>(base: u64, key: [u64; 4], program: &[Instr]) -> [u64; N] {
    let input = V::from_array(std::array::from_fn(|lane| base + lane as u64));
    digest0(run_program(init(key, input), program), key).to_array()
}

#[inline(always)]
fn word_matches(word: u64, difficulty: u32) -> bool {
    word.swap_bytes().leading_zeros() >= difficulty
}

pub fn solve(challenge: &Challenge, workers: usize) -> anyhow::Result<BackendResult> {
    challenge.validate()?;
    anyhow::ensure!(workers > 0, "SIMD HashX needs at least one worker");
    let seed = hex::decode(&challenge.data).context("decode HashX challenge")?;
    let started = Instant::now();
    let (hashx, key, program) = describe(&seed)?;
    let found = std::sync::atomic::AtomicU64::new(u64::MAX);
    let stride = (workers * LANES) as u64;

    std::thread::scope(|scope| {
        for worker in 0..workers {
            let found = &found;
            let program = &program;
            scope.spawn(move || {
                let mut base = (worker * LANES) as u64;
                while base < SEARCH_LIMIT
                    && found.load(std::sync::atomic::Ordering::Relaxed) == u64::MAX
                {
                    let words = hash_batch::<LANES>(base, key, program);
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
        "SIMD HashX search exhausted u32 nonce space"
    );
    let hash = hashx.hash_to_bytes(nonce);
    anyhow::ensure!(
        valid_prefix(&hash, challenge.difficulty),
        "SIMD HashX candidate failed scalar validation"
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
    fn simd_matches_native() {
        let (hashx, key, program) = sample();
        for base in (0..4096u64).step_by(LANES) {
            let got = hash_batch::<LANES>(base, key, &program);
            for (lane, word) in got.into_iter().enumerate() {
                assert_eq!(word, hashx.hash_to_u64(base + lane as u64));
            }
        }
    }

    #[test]
    #[ignore = "microbenchmark"]
    fn benchmark_lane_widths() {
        let (hashx, key, program) = sample();
        const TOTAL: usize = 1 << 20;
        fn bench<const N: usize>(key: [u64; 4], program: &[Instr]) {
            let started = Instant::now();
            let mut sink = 0u64;
            for base in (0..TOTAL).step_by(N) {
                for word in black_box(hash_batch::<N>(black_box(base as u64), key, program)) {
                    sink ^= word;
                }
            }
            let elapsed = started.elapsed().as_secs_f64();
            eprintln!(
                "portable_simd lanes={N} mh_s={:.3} sink={sink:016x}",
                TOTAL as f64 / elapsed / 1e6
            );
        }
        bench::<2>(key, &program);
        bench::<4>(key, &program);
        bench::<8>(key, &program);
        bench::<16>(key, &program);
        bench::<32>(key, &program);
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
}
