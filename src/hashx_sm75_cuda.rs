use std::sync::Arc;

use anyhow::{Context, anyhow};
use cudarc::{
    driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

use crate::{
    hashx::Instr,
    hashx_sm75::{build_solve_cubin, difficulty_masks, template_solve_cubin},
};

const WARP_SIZE: u32 = 32;
const MAX_BATCH: u32 = 262_144;

pub(crate) struct Sm75Cuda {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    solution: CudaSlice<u32>,
    adapter: String,
}

impl Sm75Cuda {
    pub(crate) fn create_for_wgpu_adapter(adapter_name: &str) -> anyhow::Result<Option<Self>> {
        let count = match CudaContext::device_count() {
            Ok(count) => count,
            Err(error) => {
                tracing::debug!(%error, "CUDA driver unavailable for direct sm_75 HashX backend");
                return Ok(None);
            }
        };

        let mut matches = Vec::new();
        for ordinal in 0..count {
            let context = match CudaContext::new(ordinal as usize) {
                Ok(context) => context,
                Err(error) => {
                    tracing::debug!(ordinal, %error, "failed to inspect CUDA device");
                    continue;
                }
            };
            let name = context.name().context("read CUDA device name")?;
            let capability = context
                .compute_capability()
                .context("read CUDA compute capability")?;
            if name == adapter_name && capability == (7, 5) {
                matches.push((context, name));
            }
        }

        let Some((context, name)) = matches.pop() else {
            return Ok(None);
        };
        if !matches.is_empty() {
            tracing::warn!(
                adapter = adapter_name,
                "multiple sm_75 CUDA devices have the selected wgpu adapter name; disabling direct SASS backend to avoid ambiguous device selection"
            );
            return Ok(None);
        }

        let stream = context.default_stream();
        let solution = stream
            .alloc_zeros::<u32>(1)
            .context("allocate CUDA HashX solution buffer")?;

        // Pay the CUDA driver's one-time cubin/module setup cost during backend creation,
        // not on the first real challenge. This also verifies that the embedded SM75 shell
        // is loadable by the selected driver before we advertise the backend as ready.
        let warm_module = context
            .load_module(Ptx::from_binary(template_solve_cubin()))
            .context("warm sm_75 CUDA module loader")?;
        warm_module
            .load_function("kern")
            .context("verify embedded sm_75 HashX kernel")?;
        drop(warm_module);

        Ok(Some(Self {
            context,
            stream,
            solution,
            adapter: name,
        }))
    }

    pub(crate) fn adapter(&self) -> &str {
        &self.adapter
    }

    pub(crate) fn solve(
        &mut self,
        key: &[u64; 4],
        program: &[Instr],
        difficulty: u32,
    ) -> anyhow::Result<u32> {
        let cubin = build_solve_cubin(key, program)?;
        let module = self
            .context
            .load_module(Ptx::from_binary(cubin))
            .context("load generated sm_75 HashX cubin")?;
        let function = module
            .load_function("kern")
            .context("load sm_75 HashX kernel")?;
        self.stream
            .memcpy_htod(&[u32::MAX], &mut self.solution)
            .context("reset CUDA HashX solution")?;

        let (mask0, mask1) = difficulty_masks(difficulty);
        let batch_size = batch_size(difficulty);
        let mut base = 0_u64;
        while base <= u64::from(u32::MAX) {
            let remaining = u64::from(u32::MAX) + 1 - base;
            let batch = u64::from(batch_size).min(remaining) as u32;
            debug_assert_eq!(batch % WARP_SIZE, 0);
            let base32 = base as u32;
            self.launch(&function, base32, batch, mask0, mask1)?;
            let mut answer = [u32::MAX];
            self.stream
                .memcpy_dtoh(&self.solution, &mut answer)
                .context("queue sm_75 HashX solution readback")?;
            self.stream
                .synchronize()
                .context("synchronize sm_75 HashX solve and readback")?;
            if answer[0] != u32::MAX {
                return Ok(answer[0]);
            }
            base += u64::from(batch);
        }
        Err(anyhow!("sm_75 HashX search exhausted u32 nonce space"))
    }

    #[allow(unsafe_code)]
    fn launch(
        &mut self,
        function: &cudarc::driver::CudaFunction,
        base: u32,
        batch: u32,
        mask0: u32,
        mask1: u32,
    ) -> anyhow::Result<()> {
        let config = LaunchConfig {
            grid_dim: (batch / WARP_SIZE, 1, 1),
            block_dim: (WARP_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `kern` is emitted by `hashx_sm75` from a fixed cubin template whose
        // parameter ABI is `(u32 *solution, u32 base, u32 mask0, u32 mask1)`. The
        // generated kernel only touches `solution[0]`; the remaining arguments are
        // copied by value. `self.solution` stays alive until the synchronized launch
        // completes, and grid/block dimensions are warp-aligned by construction.
        unsafe {
            self.stream
                .launch_builder(function)
                .arg(&mut self.solution)
                .arg(&base)
                .arg(&mask0)
                .arg(&mask1)
                .launch(config)
        }
        .context("launch sm_75 HashX kernel")?;
        Ok(())
    }
}

fn batch_size(difficulty: u32) -> u32 {
    let expected = 1_u64 << difficulty.min(31);
    let wanted = expected
        .div_ceil(4)
        .clamp(u64::from(WARP_SIZE), u64::from(MAX_BATCH));
    let aligned = wanted.div_ceil(u64::from(WARP_SIZE)) * u64::from(WARP_SIZE);
    aligned as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_are_warp_aligned_and_bounded() {
        for difficulty in 1..=64 {
            let batch = batch_size(difficulty);
            assert_eq!(batch % WARP_SIZE, 0);
            assert!((WARP_SIZE..=MAX_BATCH).contains(&batch));
        }
    }
}
