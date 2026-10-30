use std::sync::Arc;

use anyhow::Context;
use cudarc::{
    driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use hashwx_core::ProgramSet;

use crate::hashx_sm75::hashwx::{build_solve_cubin, workspace_bytes};

pub(crate) struct HashWxSm75Cuda {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    workspace: CudaSlice<u8>,
}

impl HashWxSm75Cuda {
    pub(crate) fn create(context: Arc<CudaContext>) -> anyhow::Result<Self> {
        let stream = context.default_stream();
        let workspace = stream
            .alloc_zeros::<u8>(workspace_bytes())
            .context("allocate SM75 HashWX workspace")?;
        Ok(Self {
            context,
            stream,
            workspace,
        })
    }

    pub(crate) fn hash(&mut self, set: &ProgramSet, nonce: u64) -> anyhow::Result<u64> {
        let cubin = build_solve_cubin(set)?;
        let module = self
            .context
            .load_module(Ptx::from_binary(cubin))
            .context("load generated SM75 HashWX cubin")?;
        let function = module
            .load_function("kern")
            .context("load SM75 HashWX kernel")?;

        let nonce_lo = nonce as u32;
        let nonce_hi = (nonce >> 32) as u32;
        let unused = 0_u32;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        self.launch(&function, nonce_lo, nonce_hi, unused, config)?;

        let output = self.workspace.slice(..8);
        let mut bytes = [0_u8; 8];
        self.stream
            .memcpy_dtoh(&output, &mut bytes)
            .context("queue SM75 HashWX hash readback")?;
        self.stream
            .synchronize()
            .context("synchronize SM75 HashWX hash and readback")?;
        Ok(u64::from_le_bytes(bytes))
    }

    #[allow(unsafe_code)]
    fn launch(
        &mut self,
        function: &cudarc::driver::CudaFunction,
        nonce_lo: u32,
        nonce_hi: u32,
        unused: u32,
        config: LaunchConfig,
    ) -> anyhow::Result<()> {
        // SAFETY: `kern` comes from the fixed HashX SM75 cubin shell.  The
        // experimental HashWX compiler preserves its 20-byte parameter ABI and
        // interprets it as `(u8 *workspace, u32 nonce_lo, u32 nonce_hi, u32 unused)`.
        // `workspace` remains allocated through the synchronized launch.
        unsafe {
            self.stream
                .launch_builder(function)
                .arg(&mut self.workspace)
                .arg(&nonce_lo)
                .arg(&nonce_hi)
                .arg(&unused)
                .launch(config)
        }
        .context("launch SM75 HashWX kernel")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires an SM75 NVIDIA GPU"]
    fn hashes_reference_nonces() -> anyhow::Result<()> {
        let context = CudaContext::new(0).context("open CUDA device 0")?;
        anyhow::ensure!(
            context.compute_capability()? == (7, 5),
            "test requires SM75, got {:?}",
            context.compute_capability()?
        );
        let mut gpu = HashWxSm75Cuda::create(context)?;

        let cases: &[(&[u8], u64, u64)] = &[
            (b"This is a test seed for hashwx", 0, 0x9736_8417_6f8e_e362),
            (
                b"This is a test seed for hashwx",
                123_456,
                0x4019_83bb_07d6_9b07,
            ),
            (
                b"Lorem ipsum dolor sit amet",
                123_456,
                0x4af3_8d83_4a9a_8d3d,
            ),
            (
                b"Lorem ipsum dolor sit amet",
                987_654_321_123_456_789,
                0x6a8a_5514_432e_17a3,
            ),
        ];
        let mut active_seed = None;
        let mut set = None;
        for &(text, nonce, expected) in cases {
            let mut seed = [0_u8; 32];
            seed[..text.len()].copy_from_slice(text);
            if active_seed != Some(seed) {
                set = Some(hashwx_core::generate(&seed));
                active_seed = Some(seed);
            }
            let actual = gpu.hash(set.as_ref().unwrap(), nonce)?;
            assert_eq!(actual, expected, "nonce={nonce}");
        }
        Ok(())
    }
}
