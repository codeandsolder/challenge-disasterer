use anyhow::Context;
#[cfg(feature = "gpu")]
use anyhow::anyhow;
use hashx::{HashX, HashXBuilder, RuntimeOption};
use serde::Deserialize;

use crate::anubis::Answer;

#[derive(Debug, Clone, Deserialize)]
pub struct Challenge {
    pub data: String,
    pub difficulty: u32,
}

impl Challenge {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            (1..=64).contains(&self.difficulty),
            "difficulty must be in 1..=64"
        );
        let bytes = hex::decode(&self.data).context("HashX challenge data must be hex")?;
        anyhow::ensure!(
            !bytes.is_empty() && bytes.len() <= 4096,
            "HashX challenge data length out of range"
        );
        Ok(())
    }
}

#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum GpuOp {
    Mul = 0,
    UMulH = 1,
    SMulH = 2,
    AddShift = 3,
    AddConst = 4,
    Sub = 5,
    Xor = 6,
    XorConst = 7,
    Rotate = 8,
    Target = 9,
    Branch = 10,
}

#[cfg(feature = "gpu")]
impl TryFrom<u32> for GpuOp {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Mul),
            1 => Ok(Self::UMulH),
            2 => Ok(Self::SMulH),
            3 => Ok(Self::AddShift),
            4 => Ok(Self::AddConst),
            5 => Ok(Self::Sub),
            6 => Ok(Self::Xor),
            7 => Ok(Self::XorConst),
            8 => Ok(Self::Rotate),
            9 => Ok(Self::Target),
            10 => Ok(Self::Branch),
            other => Err(anyhow!("unknown HashX GPU opcode {other}")),
        }
    }
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "gpu", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub(crate) struct Instr {
    pub(crate) op: u32,
    pub(crate) dst: u32,
    pub(crate) src: u32,
    pub(crate) imm: u32,
}

#[cfg(feature = "gpu")]
impl Instr {
    pub(crate) fn kind(self) -> anyhow::Result<GpuOp> {
        self.op.try_into()
    }
}

#[cfg(feature = "gpu")]
#[derive(Clone, Debug)]
pub(crate) struct GpuProgram {
    key: [u64; 4],
    instructions: Vec<Instr>,
}

#[cfg(feature = "gpu")]
impl GpuProgram {
    fn new(key: [u64; 4], instructions: Vec<Instr>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            instructions.len() == 512,
            "expected 512 HashX instructions, got {}",
            instructions.len()
        );
        let mut target = None;
        for (index, instruction) in instructions.iter().copied().enumerate() {
            let kind = instruction.kind()?;
            match kind {
                GpuOp::Mul | GpuOp::UMulH | GpuOp::SMulH | GpuOp::Sub | GpuOp::Xor => {
                    anyhow::ensure!(
                        instruction.dst < 8 && instruction.src < 8,
                        "HashX GPU register out of range at instruction {index}"
                    );
                }
                GpuOp::AddShift => {
                    anyhow::ensure!(
                        instruction.dst < 8 && instruction.src < 8,
                        "HashX GPU register out of range at instruction {index}"
                    );
                    anyhow::ensure!(
                        instruction.imm < 64,
                        "HashX GPU AddShift amount out of range at instruction {index}"
                    );
                }
                GpuOp::AddConst | GpuOp::XorConst | GpuOp::Rotate => {
                    anyhow::ensure!(
                        instruction.dst < 8,
                        "HashX GPU destination register out of range at instruction {index}"
                    );
                    if kind == GpuOp::Rotate {
                        anyhow::ensure!(
                            instruction.imm < 64,
                            "HashX GPU rotate amount out of range at instruction {index}"
                        );
                    }
                }
                GpuOp::Target => target = Some(index),
                GpuOp::Branch => {
                    let target_index =
                        target.context("HashX GPU branch without preceding target")?;
                    anyhow::ensure!(
                        target_index < index,
                        "HashX GPU branch target is not before branch at instruction {index}"
                    );
                    target = None;
                }
            }
        }
        anyhow::ensure!(target.is_none(), "HashX GPU target without matching branch");
        Ok(Self { key, instructions })
    }

    pub(crate) fn key(&self) -> &[u64; 4] {
        &self.key
    }

    pub(crate) fn instructions(&self) -> &[Instr] {
        &self.instructions
    }
}

#[cfg(feature = "gpu")]
fn prefix_mask(difficulty: u32) -> u32 {
    if difficulty >= 32 {
        return u32::MAX;
    }
    let mut mask = 0_u32;
    let full = difficulty / 8;
    let rem = difficulty % 8;
    let mut i = 0;
    while i < full {
        mask |= 0xff << (8 * i);
        i += 1;
    }
    if rem != 0 {
        mask |= ((0xff << (8 - rem)) & 0xff) << (8 * full);
    }
    mask
}

#[cfg(feature = "gpu")]
pub(crate) fn difficulty_masks(difficulty: u32) -> (u32, u32) {
    let low = prefix_mask(difficulty.min(32));
    let high = if difficulty > 32 {
        prefix_mask(difficulty - 32)
    } else {
        0
    };
    (low, high)
}

pub(crate) fn build_compiled_seed(seed: &[u8]) -> anyhow::Result<(Vec<u8>, HashX)> {
    let mut candidate = seed.to_vec();
    let mut counter = 0_u32;
    loop {
        let mut builder = HashXBuilder::new();
        builder.runtime(RuntimeOption::CompileOnly);
        if let Ok(hashx) = builder.build(&candidate) {
            return Ok((candidate, hashx));
        }
        candidate.clear();
        candidate.extend_from_slice(seed);
        candidate.extend_from_slice(&counter.to_le_bytes());
        counter = counter.wrapping_add(1);
        anyhow::ensure!(counter != 0, "HashX seed search exhausted");
    }
}

#[cfg(feature = "gpu")]
pub(crate) fn describe(seed: &[u8]) -> anyhow::Result<(HashX, GpuProgram)> {
    let (usable_seed, compiled) = build_compiled_seed(seed)?;
    let mut builder = HashXBuilder::new();
    builder.runtime(RuntimeOption::InterpretOnly);
    let interpreted = builder
        .build(&usable_seed)
        .context("build interpretable HashX")?;
    let text = format!("{interpreted:?}");
    let key_text = after(&text, "SipState[")?;
    let key_end = key_text.find(']').context("HashX debug key terminator")?;
    let mut key = [0_u64; 4];
    for (index, value) in key_text[..key_end].split(',').enumerate() {
        if index >= key.len() {
            break;
        }
        key[index] = u64::from_str_radix(value.trim().trim_start_matches("0x"), 16)
            .context("parse HashX key")?;
    }
    let mut program = Vec::with_capacity(512);
    for line in text.lines().skip(1) {
        let Some((_, rhs)) = line.split_once("]: ") else {
            continue;
        };
        let s = rhs.trim();
        let instr = if s.starts_with("Mul {") {
            binary(s, 0)?
        } else if s.starts_with("UMulH {") {
            binary(s, 1)?
        } else if s.starts_with("SMulH {") {
            binary(s, 2)?
        } else if s.starts_with("AddShift {") {
            Instr {
                op: 3,
                dst: reg(s, "dst: R")?,
                src: reg(s, "src: R")?,
                imm: number::<u32>(s, "left_shift: ")?,
            }
        } else if s.starts_with("AddConst {") {
            Instr {
                op: 4,
                dst: reg(s, "dst: R")?,
                src: 0,
                imm: number::<i32>(s, "src: ")? as u32,
            }
        } else if s.starts_with("Sub {") {
            binary(s, 5)?
        } else if s.starts_with("Xor {") {
            binary(s, 6)?
        } else if s.starts_with("XorConst {") {
            Instr {
                op: 7,
                dst: reg(s, "dst: R")?,
                src: 0,
                imm: number::<i32>(s, "src: ")? as u32,
            }
        } else if s.starts_with("Rotate {") {
            Instr {
                op: 8,
                dst: reg(s, "dst: R")?,
                src: 0,
                imm: number::<u32>(s, "right_rotate: ")?,
            }
        } else if s == "Target" {
            Instr {
                op: 9,
                dst: 0,
                src: 0,
                imm: 0,
            }
        } else if s.starts_with("Branch {") {
            Instr {
                op: 10,
                dst: 0,
                src: 0,
                imm: number::<u32>(s, "mask: ")?,
            }
        } else {
            return Err(anyhow!("unparsed HashX instruction: {s}"));
        };
        program.push(instr);
    }
    Ok((compiled, GpuProgram::new(key, program)?))
}

#[cfg(feature = "gpu")]
fn after<'a>(text: &'a str, key: &str) -> anyhow::Result<&'a str> {
    text.find(key)
        .map(|offset| &text[offset + key.len()..])
        .with_context(|| format!("missing {key} in HashX debug output"))
}

#[cfg(feature = "gpu")]
fn reg(text: &str, key: &str) -> anyhow::Result<u32> {
    after(text, key)?
        .chars()
        .next()
        .and_then(|c| c.to_digit(10))
        .context("parse HashX register")
}

#[cfg(feature = "gpu")]
fn number<T: std::str::FromStr>(text: &str, key: &str) -> anyhow::Result<T>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let tail = after(text, key)?;
    let end = tail.find([',', ' ', '}']).unwrap_or(tail.len());
    tail[..end].parse().context("parse HashX immediate")
}

#[cfg(feature = "gpu")]
fn binary(text: &str, op: u32) -> anyhow::Result<Instr> {
    Ok(Instr {
        op,
        dst: reg(text, "dst: R")?,
        src: reg(text, "src: R")?,
        imm: 0,
    })
}

pub(crate) fn verify(challenge: &Challenge, answer: &Answer) -> anyhow::Result<bool> {
    challenge.validate()?;
    let seed = hex::decode(&challenge.data).context("decode HashX challenge")?;
    let (_usable_seed, verifier) = build_compiled_seed(&seed)?;
    let nonce = answer
        .nonce
        .parse::<u64>()
        .context("parse HashX answer nonce")?;
    let hash = verifier.hash_to_bytes(nonce);
    Ok(hex::encode(hash) == answer.hash && valid_prefix(&hash, challenge.difficulty))
}

pub(crate) fn valid_prefix(hash: &[u8; 32], difficulty: u32) -> bool {
    let whole = (difficulty / 8) as usize;
    let rem = difficulty % 8;
    hash[..whole].iter().all(|&byte| byte == 0)
        && (rem == 0 || hash[whole] & (0xff_u8 << (8 - rem)) == 0)
}
