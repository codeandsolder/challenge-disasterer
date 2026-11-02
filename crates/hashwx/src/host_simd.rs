use core::simd::Simd;

const LANES: usize = 4;
type V = Simd<u64, LANES>;

#[inline(always)]
fn ror(value: V, shift: u64) -> V {
    (value >> shift) | (value << (64 - shift))
}

#[inline(always)]
fn sipround(mut v0: V, mut v1: V, mut v2: V, mut v3: V) -> (V, V, V, V) {
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

pub(super) fn initialize(key: [u64; 2], salts: [u64; LANES]) -> [[u64; 9]; LANES] {
    let k0 = V::splat(key[0]);
    let k1 = V::splat(key[1]);
    let salt = V::from_array(salts);
    let mut v0 = V::splat(0x736f_6d65_7073_6575) ^ k0;
    let mut v1 = V::splat(0x646f_7261_6e64_6f6d) ^ k1;
    let mut v2 = V::splat(0x6c79_6765_6e65_7261) ^ k0;
    let mut v3 = V::splat(0x7465_6462_7974_6573) ^ k1;
    v3 ^= salt;
    (v0, v1, v2, v3) = sipround(v0, v1, v2, v3);
    v0 ^= salt;
    v2 ^= V::splat(0xbb);
    for _ in 0..3 {
        (v0, v1, v2, v3) = sipround(v0, v1, v2, v3);
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
        (v0, v1, v2, v3) = sipround(v0, v1, v2, v3);
    }
    let r4 = v3;
    let r5 = v2;
    let r6 = v1;
    let r7 = v0;
    let r8 = ((r0 ^ r4) & V::splat(!7_u64)) | V::splat(3);
    let vectors = [r0, r1, r2, r3, r4, r5, r6, r7, r8];
    let arrays = vectors.map(Simd::to_array);
    core::array::from_fn(|lane| core::array::from_fn(|register| arrays[register][lane]))
}

pub(super) fn finalize(registers: &[[u64; 9]; LANES]) -> [u64; LANES] {
    let r: [V; 9] = core::array::from_fn(|register| {
        V::from_array(core::array::from_fn(|lane| registers[lane][register]))
    });
    let (_, _, _, r3) = sipround(r[0], r[1], r[2], r[3]);
    let (_, _, _, r7) = sipround(r[4], r[5], r[6], r[7]);
    (r3 ^ r7 ^ r[8]).to_array()
}
