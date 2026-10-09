use cudarc::driver::CudaContext;

use crate::{hashx::GpuProgram, hashx_sm75_cuda::Sm75Cuda};

pub(crate) const SM75_SASS_IMPLEMENTATION: &str = "cuda-sm75-sass";

pub(crate) enum NativeGpuBackend {
    Sm75(Sm75Cuda),
}

impl NativeGpuBackend {
    pub(crate) fn create_for_adapter(info: &wgpu::AdapterInfo) -> anyhow::Result<Option<Self>> {
        if info.vendor != 0x10de {
            return Ok(None);
        }

        let count = match CudaContext::device_count() {
            Ok(count) => count,
            Err(error) => {
                tracing::debug!(%error, "CUDA driver unavailable for native HashX backend");
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
            let name = context.name()?;
            if name != info.name {
                continue;
            }
            let capability = context.compute_capability()?;
            matches.push((context, name, capability));
        }

        let Some((context, name, capability)) = matches.pop() else {
            return Ok(None);
        };
        if !matches.is_empty() {
            tracing::warn!(
                adapter = %info.name,
                "multiple CUDA devices match the selected wgpu adapter name; disabling native HashX backend"
            );
            return Ok(None);
        }

        match capability {
            (7, 5) => Ok(Some(Self::Sm75(Sm75Cuda::create(context, name)?))),
            (major, minor) => {
                tracing::debug!(major, minor, adapter = %info.name, "no native HashX encoder for CUDA architecture");
                Ok(None)
            }
        }
    }

    pub(crate) fn implementation(&self) -> &'static str {
        match self {
            Self::Sm75(_) => SM75_SASS_IMPLEMENTATION,
        }
    }

    pub(crate) fn adapter(&self) -> &str {
        match self {
            Self::Sm75(backend) => backend.adapter(),
        }
    }

    pub(crate) fn solve(&mut self, program: &GpuProgram, difficulty: u32) -> anyhow::Result<u32> {
        match self {
            Self::Sm75(backend) => backend.solve(program, difficulty),
        }
    }
}
