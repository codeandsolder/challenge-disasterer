use anyhow::{Context, anyhow};
use hashx08::{HashX, HashXBuilder, RuntimeOption};
use serde::Deserialize;

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

#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "gpu", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub(crate) struct Instr {
    pub(crate) op: u32,
    pub(crate) dst: u32,
    pub(crate) src: u32,
    pub(crate) imm: u32,
}

fn build_seed(seed: &[u8]) -> anyhow::Result<(Vec<u8>, HashX)> {
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

pub(crate) fn describe(seed: &[u8]) -> anyhow::Result<(HashX, [u64; 4], Vec<Instr>)> {
    let (usable_seed, compiled) = build_seed(seed)?;
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
    anyhow::ensure!(
        program.len() == 512,
        "expected 512 HashX instructions, got {}",
        program.len()
    );
    Ok((compiled, key, program))
}

fn after<'a>(text: &'a str, key: &str) -> anyhow::Result<&'a str> {
    text.find(key)
        .map(|offset| &text[offset + key.len()..])
        .with_context(|| format!("missing {key} in HashX debug output"))
}

fn reg(text: &str, key: &str) -> anyhow::Result<u32> {
    after(text, key)?
        .chars()
        .next()
        .and_then(|c| c.to_digit(10))
        .context("parse HashX register")
}

fn number<T: std::str::FromStr>(text: &str, key: &str) -> anyhow::Result<T>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let tail = after(text, key)?;
    let end = tail.find([',', ' ', '}']).unwrap_or(tail.len());
    tail[..end].parse().context("parse HashX immediate")
}

fn binary(text: &str, op: u32) -> anyhow::Result<Instr> {
    Ok(Instr {
        op,
        dst: reg(text, "dst: R")?,
        src: reg(text, "src: R")?,
        imm: 0,
    })
}

pub(crate) fn valid_prefix(hash: &[u8; 32], difficulty: u32) -> bool {
    let whole = (difficulty / 8) as usize;
    let rem = difficulty % 8;
    hash[..whole].iter().all(|&byte| byte == 0)
        && (rem == 0 || hash[whole] & (0xff_u8 << (8 - rem)) == 0)
}
