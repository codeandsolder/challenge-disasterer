use std::{borrow::Cow, sync::mpsc, time::Instant};

use anyhow::{Context, anyhow};
use bytemuck::{Pod, Zeroable};
use hashx08::{HashX, HashXBuilder, RuntimeOption};
use serde::Deserialize;

use crate::anubis::{Answer, BackendResult};

const WG_SIZE: u32 = 64;
const SPECIALIZED_BATCH: u32 = 262_144;
const PERSISTENT_MAX_BATCH: u32 = 262_144;
const PERSISTENT_CROSSOVER_DIFFICULTY: u32 = 20;
const PROGRAM_BYTES: u64 = 512 * 16;

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
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct Instr {
    pub(crate) op: u32,
    pub(crate) dst: u32,
    pub(crate) src: u32,
    pub(crate) imm: u32,
}

pub struct GpuSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: String,
    params: wgpu::Buffer,
    program: wgpu::Buffer,
    solution: wgpu::Buffer,
    staging: wgpu::Buffer,
    persistent_pipeline: Option<wgpu::ComputePipeline>,
    persistent_bind_group: Option<wgpu::BindGroup>,
}

impl GpuSolver {
    pub async fn create() -> anyhow::Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .context("no WebGPU/Vulkan adapter for HashX")?;
        let adapter_features = adapter.features();
        anyhow::ensure!(
            adapter_features.contains(wgpu::Features::SHADER_INT64),
            "HashX GPU backend requires shaderInt64"
        );
        let persistent_supported = adapter_features.contains(wgpu::Features::PASSTHROUGH_SHADERS);
        let required_features = if persistent_supported {
            wgpu::Features::SHADER_INT64 | wgpu::Features::PASSTHROUGH_SHADERS
        } else {
            wgpu::Features::SHADER_INT64
        };
        let info = adapter.get_info();
        let adapter_name = format!("{} ({:?}, {})", info.name, info.backend, info.driver);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features,
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            })
            .await
            .context("request HashX GPU device")?;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hashx-params"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let program = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hashx-program"),
            size: PROGRAM_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let solution = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hashx-solution"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hashx-staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (persistent_pipeline, persistent_bind_group) = if persistent_supported {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("hashx-persistent-layout"),
                entries: &[
                    storage_layout_entry(0, true),
                    storage_layout_entry(1, true),
                    storage_layout_entry(2, false),
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hashx-persistent-pipeline-layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let module = create_persistent_shader_module(&device);
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("hashx-persistent-extmul-solve"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("solve"),
                compilation_options: Default::default(),
                cache: None,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hashx-persistent-bind"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: program.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: solution.as_entire_binding(),
                    },
                ],
            });
            (Some(pipeline), Some(bind_group))
        } else {
            (None, None)
        };

        Ok(Self {
            device,
            queue,
            adapter: adapter_name,
            params,
            program,
            solution,
            staging,
            persistent_pipeline,
            persistent_bind_group,
        })
    }

    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    pub fn solve(&mut self, challenge: &Challenge) -> anyhow::Result<BackendResult> {
        challenge.validate()?;
        let started = Instant::now();
        let seed = hex::decode(&challenge.data).context("decode HashX challenge")?;
        let (hashx, key, program) = describe(&seed)?;

        if challenge.difficulty <= PERSISTENT_CROSSOVER_DIFFICULTY
            && self.persistent_pipeline.is_some()
        {
            self.solve_persistent(challenge, &hashx, &key, &program, started)
        } else {
            self.solve_specialized(challenge, &hashx, &key, &program, started)
        }
    }

    fn solve_persistent(
        &mut self,
        challenge: &Challenge,
        hashx: &HashX,
        key: &[u64; 4],
        program: &[Instr],
        started: Instant,
    ) -> anyhow::Result<BackendResult> {
        let pipeline = self
            .persistent_pipeline
            .as_ref()
            .context("persistent HashX pipeline unavailable")?;
        let bind_group = self
            .persistent_bind_group
            .as_ref()
            .context("persistent HashX bind group unavailable")?;
        self.queue
            .write_buffer(&self.program, 0, bytemuck::cast_slice(program));
        self.queue
            .write_buffer(&self.solution, 0, &u32::MAX.to_le_bytes());

        let batch_size = persistent_batch(challenge.difficulty);
        let mut base = 0_u64;
        while base <= u64::from(u32::MAX) {
            let remaining = u64::from(u32::MAX) + 1 - base;
            let batch = u64::from(batch_size).min(remaining) as u32;
            self.write_params(key, base, challenge.difficulty);
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(batch.div_ceil(WG_SIZE), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&self.solution, 0, &self.staging, 0, 4);
            let submission = self.queue.submit([encoder.finish()]);
            self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })?;
            let nonce = self.read_solution()?;
            if nonce != u32::MAX {
                return validated_result(hashx, nonce, challenge.difficulty, started);
            }
            base += u64::from(batch);
        }
        Err(anyhow!("HashX GPU search exhausted u32 nonce space"))
    }

    fn solve_specialized(
        &mut self,
        challenge: &Challenge,
        hashx: &HashX,
        key: &[u64; 4],
        program: &[Instr],
        started: Instant,
    ) -> anyhow::Result<BackendResult> {
        let shader = specialized_shader(program);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("hashx-specialized"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader)),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("hashx-specialized-solve"),
                layout: None,
                module: &module,
                entry_point: Some("solve"),
                compilation_options: Default::default(),
                cache: None,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hashx-specialized-bind"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.solution.as_entire_binding(),
                },
            ],
        });
        self.queue
            .write_buffer(&self.solution, 0, &u32::MAX.to_le_bytes());
        let mut base = 0_u64;
        while base <= u64::from(u32::MAX) {
            let remaining = u64::from(u32::MAX) + 1 - base;
            let batch = u64::from(SPECIALIZED_BATCH).min(remaining) as u32;
            self.write_params(key, base, challenge.difficulty);
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(batch.div_ceil(WG_SIZE), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&self.solution, 0, &self.staging, 0, 4);
            let submission = self.queue.submit([encoder.finish()]);
            self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })?;
            let nonce = self.read_solution()?;
            if nonce != u32::MAX {
                return validated_result(hashx, nonce, challenge.difficulty, started);
            }
            base += u64::from(batch);
        }
        Err(anyhow!("HashX GPU search exhausted u32 nonce space"))
    }

    fn write_params(&self, key: &[u64; 4], base: u64, difficulty: u32) {
        let mut words = [0_u32; 16];
        for (index, value) in key.iter().copied().enumerate() {
            words[index * 2] = value as u32;
            words[index * 2 + 1] = (value >> 32) as u32;
        }
        words[8] = base as u32;
        words[9] = difficulty;
        self.queue
            .write_buffer(&self.params, 0, bytemuck::cast_slice(&words));
    }

    fn read_solution(&self) -> anyhow::Result<u32> {
        let slice = self.staging.slice(..4);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv().context("HashX map callback dropped")??;
        let mapped = slice.get_mapped_range().context("map HashX result")?;
        let result = u32::from_le_bytes(mapped[..4].try_into().expect("four-byte result"));
        drop(mapped);
        self.staging.unmap();
        Ok(result)
    }
}

fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[allow(unsafe_code)]
fn create_persistent_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
    let bytes = include_bytes!("hashx_blocksolve_extmul.spv");
    let words = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    // SAFETY: the embedded module is generated from blocksolve08.wgsl, patched only to replace
    // the two validated 64x64 high-half multiply helpers with SPIR-V extended-multiply ops, and
    // checked by spirv-val. The passthrough path is required because Naga rejects those ops.
    unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: Some("hashx-persistent-extmul"),
            entry_points: Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint {
                name: Cow::Borrowed("solve"),
                workgroup_size: (WG_SIZE, 1, 1),
            }]),
            spirv: Some(Cow::Owned(words)),
            ..Default::default()
        })
    }
}

fn persistent_batch(difficulty: u32) -> u32 {
    let expected = 1_u64 << difficulty.min(PERSISTENT_CROSSOVER_DIFFICULTY);
    expected
        .div_ceil(4)
        .clamp(u64::from(WG_SIZE), u64::from(PERSISTENT_MAX_BATCH)) as u32
}

fn validated_result(
    hashx: &HashX,
    nonce: u32,
    difficulty: u32,
    started: Instant,
) -> anyhow::Result<BackendResult> {
    let hash = hashx.hash_to_bytes(u64::from(nonce));
    anyhow::ensure!(
        valid_prefix(&hash, difficulty),
        "HashX GPU solver returned nonce with invalid difficulty"
    );
    Ok(BackendResult {
        answer: Answer {
            nonce: nonce.to_string(),
            hash: hex::encode(hash),
        },
        solve_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
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

fn op_source(i: Instr) -> String {
    match i.op {
        0 => format!("r[{d}u]=r[{d}u]*r[{s}u];\n", d = i.dst, s = i.src),
        1 => format!(
            "{{let x=umulhi(r[{d}u],r[{s}u]);r[{d}u]=x;mh=u32(x);}}\n",
            d = i.dst,
            s = i.src
        ),
        2 => format!(
            "{{let x=smulhi(r[{d}u],r[{s}u]);r[{d}u]=x;mh=u32(x);}}\n",
            d = i.dst,
            s = i.src
        ),
        3 => format!(
            "r[{d}u]=r[{d}u]+(r[{s}u]<<{n}u);\n",
            d = i.dst,
            s = i.src,
            n = i.imm
        ),
        4 => format!(
            "r[{d}u]=r[{d}u]+0x{v:016x}lu;\n",
            d = i.dst,
            v = (i.imm as i32 as i64) as u64
        ),
        5 => format!("r[{d}u]=r[{d}u]-r[{s}u];\n", d = i.dst, s = i.src),
        6 => format!("r[{d}u]=r[{d}u]^r[{s}u];\n", d = i.dst, s = i.src),
        7 => format!(
            "r[{d}u]=r[{d}u]^0x{v:016x}lu;\n",
            d = i.dst,
            v = (i.imm as i32 as i64) as u64
        ),
        8 => format!("r[{d}u]=ror(r[{d}u],{n}u);\n", d = i.dst, n = i.imm),
        9 | 10 => String::new(),
        _ => unreachable!(),
    }
}

fn specialized_shader(program: &[Instr]) -> String {
    let mut body = String::new();
    let mut target = None;
    for (index, &instruction) in program.iter().enumerate() {
        match instruction.op {
            9 => target = Some(index),
            10 => {
                let start = target.expect("HashX branch without target") + 1;
                body.push_str(&format!(
                    "if(allow&&((mh&0x{:08x}u)==0u)){{allow=false;\n",
                    instruction.imm
                ));
                for &replay in &program[start..index] {
                    body.push_str(&op_source(replay));
                }
                body.push_str("}\n");
            }
            _ => body.push_str(&op_source(instruction)),
        }
    }
    format!(
        r#"
struct Params{{w:array<u32,16>}};struct Out{{solution:atomic<u32>}};
@group(0) @binding(0) var<storage,read> params:Params;@group(0) @binding(1) var<storage,read_write> out:Out;
struct S{{v0:u64,v1:u64,v2:u64,v3:u64}};
fn rol(x:u64,n:u32)->u64{{return(x<<n)|(x>>(64u-n));}}fn ror(x:u64,n:u32)->u64{{return(x>>n)|(x<<(64u-n));}}
fn sip(q:S)->S{{var s=q;s.v0=s.v0+s.v1;s.v2=s.v2+s.v3;s.v1=rol(s.v1,13u);s.v3=rol(s.v3,16u);s.v1=s.v1^s.v0;s.v3=s.v3^s.v2;s.v0=rol(s.v0,32u);s.v2=s.v2+s.v1;s.v0=s.v0+s.v3;s.v1=rol(s.v1,17u);s.v3=rol(s.v3,21u);s.v1=s.v1^s.v2;s.v3=s.v3^s.v0;s.v2=rol(s.v2,32u);return s;}}
fn key()->S{{return S(u64(params.w[0])|(u64(params.w[1])<<32u),u64(params.w[2])|(u64(params.w[3])<<32u),u64(params.w[4])|(u64(params.w[5])<<32u),u64(params.w[6])|(u64(params.w[7])<<32u));}}
fn init(k:S,n:u64)->array<u64,8>{{var s=k;s.v1=s.v1^0xeelu;s.v3=s.v3^n;s=sip(s);s=sip(s);s.v0=s.v0^n;s.v2=s.v2^0xeelu;s=sip(s);s=sip(s);s=sip(s);s=sip(s);var t=s;t.v1=t.v1^0xddlu;t=sip(t);t=sip(t);t=sip(t);t=sip(t);return array<u64,8>(s.v0,s.v1,s.v2,s.v3,t.v0,t.v1,t.v2,t.v3);}}
fn umulhi(a:u64,b:u64)->u64{{let m=0xffffffffu;let a0=u64(u32(a&u64(m)));let a1=a>>32u;let b0=u64(u32(b&u64(m)));let b1=b>>32u;let p0=a0*b0;let p1=a0*b1;let p2=a1*b0;let p3=a1*b1;let carry=((p0>>32u)+(p1&u64(m))+(p2&u64(m)))>>32u;return p3+(p1>>32u)+(p2>>32u)+carry;}}
fn smulhi(a:u64,b:u64)->u64{{var h=umulhi(a,b);if((a>>63u)!=0lu){{h=h-b;}}if((b>>63u)!=0lu){{h=h-a;}}return h;}}
fn run(r0:array<u64,8>)->array<u64,8>{{var r=r0;var allow=true;var mh=0u;{body}return r;}}
fn hx(n:u64)->array<u64,4>{{let k=key();let r=run(init(k,n));var x=S(r[0]+k.v0,r[1]+k.v1,r[2],r[3]);var y=S(r[4],r[5],r[6]+k.v2,r[7]+k.v3);x=sip(x);y=sip(y);return array<u64,4>(x.v0^y.v0,x.v1^y.v1,x.v2^y.v2,x.v3^y.v3);}}
fn bswap(x:u32)->u32{{return((x&0xffu)<<24u)|((x&0xff00u)<<8u)|((x&0xff0000u)>>8u)|((x&0xff000000u)>>24u);}}
@compute @workgroup_size(64) fn solve(@builtin(global_invocation_id) gid:vec3<u32>){{if(atomicLoad(&out.solution)!=0xffffffffu){{return;}}let nonce=params.w[8]+gid.x;let h=hx(u64(nonce));let d=params.w[9];let p0=bswap(u32(h[0]));let p1=bswap(u32(h[0]>>32u));var ok=false;if(d<=32u){{ok=(d==0u)||((p0>>(32u-d))==0u);}}else{{ok=(p0==0u)&&((p1>>(64u-d))==0u);}}if(ok){{_=atomicMin(&out.solution,nonce);}}}}
"#
    )
}
