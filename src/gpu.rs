use std::{borrow::Cow, sync::mpsc, time::Instant};

use anyhow::{Context, anyhow};
use bytemuck::{Pod, Zeroable};

use crate::anubis::{Answer, BackendResult, Challenge, verify};

const WG_SIZE: u32 = 64;
const STEPS: u32 = 128;
const HASHES_PER_WG: u64 = WG_SIZE as u64 * STEPS as u64;
const CANCEL_INTERVAL: u32 = 16;
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];
const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    words: [u32; 16],
}

pub struct GpuSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    in_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    max_wgs: u32,
    adapter: String,
}

impl GpuSolver {
    pub async fn create() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
            .await
            .context("no WebGPU/Vulkan adapter")?;
        let info = adapter.get_info();
        let adapter_name = format!("{} ({:?}, {})", info.name, info.backend, info.driver);
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .context("request GPU device")?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anubis"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_source())),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("anubis"),
            layout: None,
            module: &shader,
            entry_point: Some("solve"),
            compilation_options: Default::default(),
            cache: None,
        });
        let in_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("in"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: 48,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: 48,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anubis"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        Ok(Self {
            device,
            queue,
            pipeline,
            in_buf,
            out_buf,
            staging,
            bind_group,
            max_wgs: limits.max_compute_workgroups_per_dimension,
            adapter: adapter_name,
        })
    }

    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    pub fn calibrate(&mut self, hashes: u64) -> anyhow::Result<f64> {
        let challenge = Challenge {
            data: "a".repeat(128),
            difficulty: 16,
        };
        let started = Instant::now();
        let _ = self.solve_limited(&challenge, hashes)?;
        let seconds = started.elapsed().as_secs_f64();
        anyhow::ensure!(seconds > 0.0, "invalid GPU calibration time");
        Ok(hashes as f64 / seconds)
    }

    pub fn solve(&mut self, challenge: &Challenge) -> anyhow::Result<BackendResult> {
        self.solve_limited(challenge, u64::MAX)?
            .context("GPU search exhausted")
    }

    fn solve_limited(
        &mut self,
        challenge: &Challenge,
        max_hashes: u64,
    ) -> anyhow::Result<Option<BackendResult>> {
        challenge.validate()?;
        if !challenge.gpu_compatible() {
            return Err(anyhow!("GPU backend requires data length multiple of 64"));
        }
        let mid = midstate(challenge.data.as_bytes());
        let (mask0, mask1) = masks(challenge.difficulty);
        let mut params = Params { words: [0; 16] };
        params.words[..8].copy_from_slice(&mid);
        params.words[10] = ((challenge.data.len() + 19) * 8) as u32;
        params.words[11] = mask0;
        params.words[12] = mask1;
        self.queue.write_buffer(&self.out_buf, 0, &[0; 48]);
        let started = Instant::now();
        let mut base = 0_u64;
        let ideal = challenge.expected_work();
        let desired = ideal.saturating_mul(8).max(HASHES_PER_WG);
        while base < max_hashes {
            let remaining = max_hashes.saturating_sub(base);
            let batch = desired
                .min(remaining)
                .min(u64::from(self.max_wgs) * HASHES_PER_WG);
            let wgs = batch.div_ceil(HASHES_PER_WG).max(1) as u32;
            params.words[8] = base as u32;
            params.words[9] = (base >> 32) as u32;
            self.queue
                .write_buffer(&self.in_buf, 0, bytemuck::bytes_of(&params));
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("solve"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("solve"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(wgs, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging, 0, 48);
            let submission = self.queue.submit([encoder.finish()]);
            self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })?;
            let values = self.read_result()?;
            if values[0] != 0 {
                let lo = values[1];
                let hi = values[2];
                let nonce = nonce_string(hi, lo);
                let hash = values[3..11]
                    .iter()
                    .map(|word| format!("{word:08x}"))
                    .collect();
                let result = BackendResult {
                    answer: Answer { nonce, hash },
                    solve_ms: started.elapsed().as_secs_f64() * 1000.0,
                };
                anyhow::ensure!(
                    verify(challenge, &result.answer),
                    "GPU solver returned invalid answer"
                );
                return Ok(Some(result));
            }
            base = base.saturating_add(u64::from(wgs) * HASHES_PER_WG);
        }
        Ok(None)
    }

    fn read_result(&self) -> anyhow::Result<Vec<u32>> {
        let slice = self.staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv().context("GPU map callback dropped")??;
        let mapped = slice.get_mapped_range().context("map result buffer")?;
        let values = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
        drop(mapped);
        self.staging.unmap();
        Ok(values)
    }
}

fn masks(difficulty: u8) -> (u32, u32) {
    fn one(bits: u32) -> u32 {
        if bits == 0 {
            0
        } else {
            u32::MAX << (32 - bits)
        }
    }
    (
        one(u32::from(difficulty.min(8)) * 4),
        one(u32::from(difficulty.saturating_sub(8).min(8)) * 4),
    )
}

fn nonce_string(hi: u32, lo: u32) -> String {
    let n = (u64::from(hi) << 32) | u64::from(lo);
    format!("1{n:018o}")
}

fn midstate(bytes: &[u8]) -> [u32; 8] {
    let mut state = H0;
    for block in bytes.as_chunks::<64>().0 {
        compress(&mut state, block);
    }
    state
}

fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0_u32; 64];
    for (i, chunk) in block.as_chunks::<4>().0.iter().enumerate() {
        w[i] = u32::from_be_bytes(*chunk);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = g ^ (e & (f ^ g));
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    for (dst, add) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *dst = dst.wrapping_add(add);
    }
}

fn shader_source() -> String {
    let mut rounds = String::new();
    for (t, k) in K.iter().enumerate() {
        let i = t & 15;
        if t >= 16 {
            rounds.push_str(&format!("w{i}=(rotr(w{},17u)^rotr(w{},19u)^(w{}>>10u))+w{}+(rotr(w{},7u)^rotr(w{},18u)^(w{}>>3u))+w{i};\n",(t-2)&15,(t-2)&15,(t-2)&15,(t-7)&15,(t-15)&15,(t-15)&15,(t-15)&15));
        }
        rounds.push_str(&format!("{{let t1=h+(rotr(e,6u)^rotr(e,11u)^rotr(e,25u))+(g^(e&(f^g)))+0x{k:08x}u+w{i};let t2=(rotr(a,2u)^rotr(a,13u)^rotr(a,22u))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}}\n"));
    }
    format!(
        r"
struct InParams {{ s0:u32,s1:u32,s2:u32,s3:u32,s4:u32,s5:u32,s6:u32,s7:u32,base_lo:u32,base_hi:u32,bitlen:u32,mask0:u32,mask1:u32 }};
struct OutBuf {{ flag:atomic<u32>,nonce_lo:u32,nonce_hi:u32,hash:array<u32,8> }};
@group(0) @binding(0) var<uniform> P:InParams;
@group(0) @binding(1) var<storage,read_write> R:OutBuf;
fn rotr(x:u32,n:u32)->u32{{return (x>>n)|(x<<(32u-n));}}
@compute @workgroup_size({WG_SIZE})
fn solve(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(num_workgroups) nwg:vec3<u32>){{
 let stride=nwg.x*{WG_SIZE}u; var lo=P.base_lo+gid.x; var hi=P.base_hi; if(lo<P.base_lo){{hi=hi+1u;}}
 for(var step=0u;step<{STEPS}u;step=step+1u){{
  if((step&{}u)==0u && atomicLoad(&R.flag)!=0u){{return;}}
  let d0=lo&7u;let d1=(lo>>3u)&7u;let d2=(lo>>6u)&7u;let d3=(lo>>9u)&7u;let d4=(lo>>12u)&7u;let d5=(lo>>15u)&7u;let d6=(lo>>18u)&7u;let d7=(lo>>21u)&7u;let d8=(lo>>24u)&7u;let d9=(lo>>27u)&7u;let d10=((lo>>30u)|(hi<<2u))&7u;let d11=(hi>>1u)&7u;let d12=(hi>>4u)&7u;let d13=(hi>>7u)&7u;let d14=(hi>>10u)&7u;let d15=(hi>>13u)&7u;let d16=(hi>>16u)&7u;let d17=(hi>>19u)&7u;
  var w0=0x31303030u|(d17<<16u)|(d16<<8u)|d15;var w1=0x30303030u|(d14<<24u)|(d13<<16u)|(d12<<8u)|d11;var w2=0x30303030u|(d10<<24u)|(d9<<16u)|(d8<<8u)|d7;var w3=0x30303030u|(d6<<24u)|(d5<<16u)|(d4<<8u)|d3;var w4=0x30303080u|(d2<<24u)|(d1<<16u)|(d0<<8u);
  var w5=0u;var w6=0u;var w7=0u;var w8=0u;var w9=0u;var w10=0u;var w11=0u;var w12=0u;var w13=0u;var w14=0u;var w15=P.bitlen;
  var a=P.s0;var b=P.s1;var c=P.s2;var d=P.s3;var e=P.s4;var f=P.s5;var g=P.s6;var h=P.s7;
  {rounds}
  let r0=P.s0+a;let r1=P.s1+b;
  if((r0&P.mask0)==0u&&(r1&P.mask1)==0u){{if(atomicExchange(&R.flag,1u)==0u){{R.nonce_lo=lo;R.nonce_hi=hi;R.hash[0]=r0;R.hash[1]=r1;R.hash[2]=P.s2+c;R.hash[3]=P.s3+d;R.hash[4]=P.s4+e;R.hash[5]=P.s5+f;R.hash[6]=P.s6+g;R.hash[7]=P.s7+h;}}}}
  let nl=lo+stride;if(nl<lo){{hi=hi+1u;}}lo=nl;
 }}
}}
",
        CANCEL_INTERVAL - 1
    )
}
