use std::{borrow::Cow, fmt::Write as _, sync::mpsc, time::Instant};

use anyhow::{Context, anyhow};
use bytemuck::{Pod, Zeroable};

use crate::anubis::{Answer, BackendResult, Challenge, verify};

const WG_SIZE: u32 = 64;
const STEPS: u32 = 128;
const HASHES_PER_WG: u64 = 64 * 128;
const CANCEL_INTERVAL: u32 = 16;
use challenge_solvers::SHA256_ROUND_CONSTANTS;

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
            compilation_options: wgpu::PipelineCompilationOptions::default(),
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

    pub const fn work_quantum() -> u64 {
        HASHES_PER_WG
    }

    pub fn calibrate(&self, hashes: u64) -> anyhow::Result<f64> {
        anyhow::ensure!(
            hashes > 0,
            "GPU calibration hash budget must be greater than zero"
        );
        let remainder = hashes % HASHES_PER_WG;
        let hashes = if remainder == 0 {
            hashes
        } else {
            hashes
                .checked_add(HASHES_PER_WG - remainder)
                .context("GPU calibration hash budget overflow")?
        };
        let challenge = Challenge {
            data: "a".repeat(128),
            difficulty: 16,
        };
        let started = Instant::now();
        let _ = self.solve_limited(&challenge, hashes)?;
        let seconds = started.elapsed().as_secs_f64();
        anyhow::ensure!(seconds > 0.0, "invalid GPU calibration time");
        Ok(crate::metrics::approximate_work_f64(hashes) / seconds)
    }

    pub fn solve(&self, challenge: &Challenge) -> anyhow::Result<BackendResult> {
        self.solve_limited(challenge, u64::MAX)?
            .context("GPU search exhausted")
    }

    fn solve_limited(
        &self,
        challenge: &Challenge,
        max_hashes: u64,
    ) -> anyhow::Result<Option<BackendResult>> {
        challenge.validate()?;
        if !challenge.gpu_compatible() {
            return Err(anyhow!("GPU backend requires data length multiple of 64"));
        }
        let mid = challenge_solvers::sha256_full_block_midstate(challenge.data.as_bytes());
        let (mask0, mask1) = masks(challenge.difficulty);
        let mut params = Params { words: [0; 16] };
        params.words[..8].copy_from_slice(&mid);
        params.words[10] = u32::try_from((challenge.data.len() + 19) * 8)
            .context("GPU challenge bit length exceeds u32")?;
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
            let wgs = u32::try_from(batch.div_ceil(HASHES_PER_WG).max(1))
                .context("GPU workgroup count exceeds u32")?;
            let (base_lo, base_hi) = split_u64(base);
            params.words[8] = base_lo;
            params.words[9] = base_hi;
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
                let hash = hex::encode(
                    values[3..11]
                        .iter()
                        .flat_map(|word| word.to_be_bytes())
                        .collect::<Vec<_>>(),
                );
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

const fn split_u64(value: u64) -> (u32, u32) {
    let bytes = value.to_le_bytes();
    (
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
    )
}

fn masks(difficulty: u8) -> (u32, u32) {
    const fn one(bits: u32) -> u32 {
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

fn shader_source() -> String {
    let mut rounds = String::new();
    for (t, k) in SHA256_ROUND_CONSTANTS.iter().enumerate() {
        let i = t & 15;
        if t >= 16 {
            let _ = writeln!(
                rounds,
                "w{i}=(rotr(w{},17u)^rotr(w{},19u)^(w{}>>10u))+w{}+(rotr(w{},7u)^rotr(w{},18u)^(w{}>>3u))+w{i};",
                (t - 2) & 15,
                (t - 2) & 15,
                (t - 2) & 15,
                (t - 7) & 15,
                (t - 15) & 15,
                (t - 15) & 15,
                (t - 15) & 15
            );
        }
        let _ = writeln!(
            rounds,
            "{{let t1=h+(rotr(e,6u)^rotr(e,11u)^rotr(e,25u))+(g^(e&(f^g)))+0x{k:08x}u+w{i};let t2=(rotr(a,2u)^rotr(a,13u)^rotr(a,22u))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}}"
        );
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
 let gid_lo=gid.x<<7u;let gid_hi=gid.x>>25u;let base_lo=P.base_lo+gid_lo;var base_hi=P.base_hi+gid_hi;if(base_lo<P.base_lo){{base_hi=base_hi+1u;}}
 let d2=(base_lo>>6u)&7u;let d3=(base_lo>>9u)&7u;let d4=(base_lo>>12u)&7u;let d5=(base_lo>>15u)&7u;let d6=(base_lo>>18u)&7u;let d7=(base_lo>>21u)&7u;let d8=(base_lo>>24u)&7u;let d9=(base_lo>>27u)&7u;let d10=((base_lo>>30u)|(base_hi<<2u))&7u;let d11=(base_hi>>1u)&7u;let d12=(base_hi>>4u)&7u;let d13=(base_hi>>7u)&7u;let d14=(base_hi>>10u)&7u;let d15=(base_hi>>13u)&7u;let d16=(base_hi>>16u)&7u;let d17=(base_hi>>19u)&7u;
 let nw0=0x31303030u|(d17<<16u)|(d16<<8u)|d15;let nw1=0x30303030u|(d14<<24u)|(d13<<16u)|(d12<<8u)|d11;let nw2=0x30303030u|(d10<<24u)|(d9<<16u)|(d8<<8u)|d7;let nw3=0x30303030u|(d6<<24u)|(d5<<16u)|(d4<<8u)|d3;var nw4=0x30303080u|(d2<<24u);
 const GM:array<u32,7>=array<u32,7>(0x00000100u,0x00000200u,0x00000400u,0x00010000u,0x00020000u,0x00040000u,0x01000000u);
 for(var step=0u;step<{STEPS}u;step=step+1u){{
  if((step&{}u)==0u && atomicLoad(&R.flag)!=0u){{return;}}
  var w0=nw0;var w1=nw1;var w2=nw2;var w3=nw3;var w4=nw4;
  var w5=0u;var w6=0u;var w7=0u;var w8=0u;var w9=0u;var w10=0u;var w11=0u;var w12=0u;var w13=0u;var w14=0u;var w15=P.bitlen;
  var a=P.s0;var b=P.s1;var c=P.s2;var d=P.s3;var e=P.s4;var f=P.s5;var g=P.s6;var h=P.s7;
  {rounds}
  let r0=P.s0+a;let r1=P.s1+b;
  if((r0&P.mask0)==0u&&(r1&P.mask1)==0u){{if(atomicExchange(&R.flag,1u)==0u){{let gray=step^(step>>1u);R.nonce_lo=base_lo+gray;R.nonce_hi=base_hi;R.hash[0]=r0;R.hash[1]=r1;R.hash[2]=P.s2+c;R.hash[3]=P.s3+d;R.hash[4]=P.s4+e;R.hash[5]=P.s5+f;R.hash[6]=P.s6+g;R.hash[7]=P.s7+h;}}}}
  let ns=step+1u;if(ns<{STEPS}u){{nw4=nw4^GM[countTrailingZeros(ns)];}}
 }}
}}
",
        CANCEL_INTERVAL - 1
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_block_masks_cover_exactly_128_candidates() {
        const GRAY_MASKS: [u32; 7] = [
            0x0000_0100,
            0x0000_0200,
            0x0000_0400,
            0x0001_0000,
            0x0002_0000,
            0x0004_0000,
            0x0100_0000,
        ];

        assert_eq!(STEPS, 128);
        assert_eq!(HASHES_PER_WG, u64::from(WG_SIZE) * u64::from(STEPS));

        let mut word = 0x3030_3080_u32;
        let mut seen = [false; STEPS as usize];
        let mut previous_gray: Option<u32> = None;
        for step in 0..STEPS {
            let gray = step ^ (step >> 1);
            assert!(gray < STEPS);
            assert!(!seen[gray as usize]);
            seen[gray as usize] = true;

            if let Some(previous) = previous_gray {
                assert_eq!((previous ^ gray).count_ones(), 1);
            }
            previous_gray = Some(gray);

            let expected = 0x3030_3080_u32
                | ((gray & 0o7) << 8)
                | (((gray >> 3) & 0o7) << 16)
                | (((gray >> 6) & 0o1) << 24);
            assert_eq!(word, expected, "step={step} gray={gray}");

            let next = step + 1;
            if next < STEPS {
                word ^= GRAY_MASKS[next.trailing_zeros() as usize];
            }
        }
        assert!(seen.into_iter().all(|entry| entry));
    }
}
