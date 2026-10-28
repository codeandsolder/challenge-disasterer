#![deny(unsafe_code)]

//! Safe host for the official `HashWX` WebAssembly implementation.
//!
//! The bundled module is the exact `HashWX` 1.0 artifact used by Cap.js at
//! `tiagozip/cap@278796641bc82972b272c5d16dc6f479aeadc753`, built from
//! `tevador/hashwx@74b567a31276a4c5d7cb232a5b7639467f49f961`.
//! Compiled `HashWX` contexts generate a small per-seed WebAssembly module. We
//! hand that module back to Wasmtime/Cranelift, which supplies host-native code
//! generation without architecture-specific or executable-memory code here.

use std::{fmt, sync::OnceLock};

use wasmtime::{Engine, Extern, Instance, Memory, Module, Store, TypedFunc};

pub const SEED_SIZE: usize = 32;
const HASHWX_WASM: &[u8] = include_bytes!("../vendor/hashwx.wasm");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashWxError(String);

impl fmt::Display for HashWxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for HashWxError {}

impl From<wasmtime::Error> for HashWxError {
    fn from(error: wasmtime::Error) -> Self {
        Self(error.to_string())
    }
}

pub type Result<T> = std::result::Result<T, HashWxError>;

fn error(message: impl fmt::Display) -> HashWxError {
    HashWxError(message.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Runtime {
    Interpreted,
    Compiled,
}

struct SharedRuntime {
    engine: Engine,
    module: Module,
}

static SHARED_RUNTIME: OnceLock<std::result::Result<SharedRuntime, String>> = OnceLock::new();

fn shared_runtime() -> Result<&'static SharedRuntime> {
    match SHARED_RUNTIME.get_or_init(|| {
        let engine = Engine::default();
        let module =
            Module::from_binary(&engine, HASHWX_WASM).map_err(|source| source.to_string())?;
        Ok(SharedRuntime { engine, module })
    }) {
        Ok(runtime) => Ok(runtime),
        Err(source) => Err(error(format!("compile bundled HashWX module: {source}"))),
    }
}

/// One reusable `HashWX` program context.
///
/// `Compiled` uses `HashWX`'s official per-seed WebAssembly code generator and
/// lets Wasmtime compile the resulting module for the host CPU. `Interpreted`
/// executes the official `HashWX` interpreter inside the already-compiled main
/// module and exists primarily as a correctness/performance reference.
pub struct HashWx {
    store: Store<()>,
    memory: Memory,
    ctx: i32,
    runtime: Runtime,
    seed_ptr: usize,
    registers: i32,
    scratch: i32,
    make: TypedFunc<(i32, i32), ()>,
    direct_exec: TypedFunc<(i32, i64), i64>,
    module_ptr: TypedFunc<i32, i32>,
    module_size: TypedFunc<i32, i32>,
    exec_begin: TypedFunc<(i32, i64), ()>,
    exec_final: TypedFunc<i32, i64>,
    free: TypedFunc<i32, ()>,
    compiled_exec: Option<TypedFunc<(i32, i32), ()>>,
}

impl HashWx {
    /// Create the production compiled runtime.
    ///
    /// # Errors
    /// Returns an error if the bundled module cannot be compiled or instantiated,
    /// or if the `HashWX` runtime cannot allocate a compiled context.
    pub fn new() -> Result<Self> {
        Self::with_runtime(Runtime::Compiled)
    }

    /// Create either the official interpreted or compiled runtime.
    ///
    /// # Errors
    /// Returns an error if the bundled module cannot be compiled or instantiated,
    /// required exports are missing, or the requested runtime cannot allocate a context.
    pub fn with_runtime(runtime: Runtime) -> Result<Self> {
        let shared = shared_runtime()?;
        let mut store = Store::new(&shared.engine, ());
        let instance = Instance::new(&mut store, &shared.module, &[])?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| error("bundled HashWX module has no memory export"))?;

        if let Some(initialize) = instance.get_func(&mut store, "_initialize") {
            initialize.typed::<(), ()>(&store)?.call(&mut store, ())?;
        }

        let alloc = instance.get_typed_func::<i32, i32>(&mut store, "hashwx_alloc")?;
        let seed = instance.get_typed_func::<i32, i32>(&mut store, "hashwx_seed")?;
        let registers_fn = instance.get_typed_func::<i32, i32>(&mut store, "hashwx_registers")?;
        let scratch_fn = instance.get_typed_func::<i32, i32>(&mut store, "hashwx_memory")?;
        let make = instance.get_typed_func::<(i32, i32), ()>(&mut store, "hashwx_make")?;
        let direct_exec = instance.get_typed_func::<(i32, i64), i64>(&mut store, "hashwx_exec")?;
        let module_ptr = instance.get_typed_func::<i32, i32>(&mut store, "hashwx_module")?;
        let module_size = instance.get_typed_func::<i32, i32>(&mut store, "hashwx_module_size")?;
        let exec_begin =
            instance.get_typed_func::<(i32, i64), ()>(&mut store, "hashwx_exec_begin")?;
        let exec_final = instance.get_typed_func::<i32, i64>(&mut store, "hashwx_exec_final")?;
        let free = instance.get_typed_func::<i32, ()>(&mut store, "hashwx_free")?;

        let kind = match runtime {
            Runtime::Interpreted => 0,
            Runtime::Compiled => 1,
        };
        let ctx = alloc.call(&mut store, kind)?;
        if ctx <= 0 {
            return Err(error(format!("HashWX runtime allocation failed: {ctx}")));
        }
        let seed_ptr = usize::try_from(seed.call(&mut store, ctx)?)
            .map_err(|source| error(format!("invalid HashWX seed pointer: {source}")))?;
        let registers = registers_fn.call(&mut store, ctx)?;
        let scratch = scratch_fn.call(&mut store, ctx)?;

        Ok(Self {
            store,
            memory,
            ctx,
            runtime,
            seed_ptr,
            registers,
            scratch,
            make,
            direct_exec,
            module_ptr,
            module_size,
            exec_begin,
            exec_final,
            free,
            compiled_exec: None,
        })
    }

    #[must_use]
    pub const fn runtime(&self) -> Runtime {
        self.runtime
    }

    /// Generate a new `HashWX` function from a 256-bit seed.
    ///
    /// # Errors
    /// Returns an error if seed memory cannot be written or the generated WebAssembly
    /// function cannot be compiled and instantiated.
    pub fn make(&mut self, seed: &[u8; SEED_SIZE]) -> Result<()> {
        self.memory
            .write(&mut self.store, self.seed_ptr, seed)
            .map_err(|source| error(format!("write HashWX seed: {source}")))?;
        let seed_ptr = i32::try_from(self.seed_ptr)
            .map_err(|source| error(format!("HashWX seed pointer does not fit i32: {source}")))?;
        self.make.call(&mut self.store, (self.ctx, seed_ptr))?;

        if self.runtime == Runtime::Compiled {
            let module_ptr = usize::try_from(self.module_ptr.call(&mut self.store, self.ctx)?)
                .map_err(|source| error(format!("invalid generated module pointer: {source}")))?;
            let module_size = usize::try_from(self.module_size.call(&mut self.store, self.ctx)?)
                .map_err(|source| error(format!("invalid generated module size: {source}")))?;
            let mut bytes = vec![0_u8; module_size];
            self.memory
                .read(&self.store, module_ptr, &mut bytes)
                .map_err(|source| error(format!("read generated HashWX module: {source}")))?;
            let module = Module::from_binary(self.store.engine(), &bytes)?;
            let instance = Instance::new(&mut self.store, &module, &[Extern::Memory(self.memory)])?;
            self.compiled_exec =
                Some(instance.get_typed_func::<(i32, i32), ()>(&mut self.store, "exec")?);
        }
        Ok(())
    }

    /// Hash one 64-bit nonce with the current generated function.
    ///
    /// # Errors
    /// Returns an error if no compiled function has been generated yet or execution of
    /// the selected runtime fails.
    pub fn hash(&mut self, input: u64) -> Result<u64> {
        match self.runtime {
            Runtime::Interpreted => Ok(self
                .direct_exec
                .call(&mut self.store, (self.ctx, input.cast_signed()))?
                .cast_unsigned()),
            Runtime::Compiled => {
                let exec = self
                    .compiled_exec
                    .as_ref()
                    .ok_or_else(|| error("HashWX function has not been generated"))?;
                self.exec_begin
                    .call(&mut self.store, (self.ctx, input.cast_signed()))?;
                exec.call(&mut self.store, (self.registers, self.scratch))?;
                Ok(self
                    .exec_final
                    .call(&mut self.store, self.ctx)?
                    .cast_unsigned())
            }
        }
    }
}

impl Drop for HashWx {
    fn drop(&mut self) {
        let _ = self.free.call(&mut self.store, self.ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiled_and_interpreted_match_upstream_vectors() -> Result<()> {
        let mut seed1 = [0_u8; SEED_SIZE];
        let text1 = b"This is a test seed for hashwx";
        seed1[..text1.len()].copy_from_slice(text1);
        let mut seed2 = [0_u8; SEED_SIZE];
        let text2 = b"Lorem ipsum dolor sit amet";
        seed2[..text2.len()].copy_from_slice(text2);
        let cases = [
            (seed1, 0_u64, 0x9736_8417_6f8e_e362_u64),
            (seed1, 123_456, 0x4019_83bb_07d6_9b07_u64),
            (seed2, 123_456, 0x4af3_8d83_4a9a_8d3d_u64),
            (seed2, 987_654_321_123_456_789, 0x6a8a_5514_432e_17a3_u64),
        ];
        for runtime in [Runtime::Interpreted, Runtime::Compiled] {
            let mut hashwx = HashWx::with_runtime(runtime)?;
            let mut last_seed = None;
            for (seed, input, expected) in cases {
                if last_seed != Some(seed) {
                    hashwx.make(&seed)?;
                    last_seed = Some(seed);
                }
                assert_eq!(hashwx.hash(input)?, expected);
            }
        }
        Ok(())
    }
}
