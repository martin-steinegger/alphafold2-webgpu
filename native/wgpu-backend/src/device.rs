//! The device, the handle registry and the recorded command form.
//!
//! Commands are recorded rather than encoded as they arrive. A wgpu compute
//! pass borrows its encoder, and a handle that lives between two calls from
//! Node cannot hold that borrow, so an encoder collects owned commands and
//! replays them into a real encoder when it is finished. This is not the
//! command-list batching the specification rejects: every call from Node is
//! still one call across the boundary.

use napi::bindgen_prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

pub enum PassCommand {
    SetPipeline(u32),
    SetBindGroup(u32, u32),
    Dispatch(u32, u32, u32),
    PushDebugGroup(String),
    PopDebugGroup,
}

pub struct Pass {
    pub commands: Vec<PassCommand>,
    /// Query set and the two indices, as WebGPU's timestampWrites carries them.
    pub timestamps: Option<(u32, Option<u32>, Option<u32>)>,
}

pub enum Command {
    Copy {
        source: u32,
        source_offset: u64,
        destination: u32,
        destination_offset: u64,
        bytes: u64,
    },
    Clear {
        buffer: u32,
        offset: u64,
        bytes: u64,
    },
    Pass(Pass),
    ResolveQuerySet {
        set: u32,
        first: u32,
        count: u32,
        destination: u32,
        offset: u64,
    },
}

#[derive(Default)]
pub struct Encoder {
    pub commands: Vec<Command>,
    /// The pass being recorded, moved into commands when it ends.
    pub open_pass: Option<Pass>,
}

#[derive(Default)]
pub struct Registry {
    pub buffers: HashMap<u32, wgpu::Buffer>,
    pub modules: HashMap<u32, wgpu::ShaderModule>,
    pub pipelines: HashMap<u32, wgpu::ComputePipeline>,
    pub groups: HashMap<u32, wgpu::BindGroup>,
    pub encoders: HashMap<u32, Encoder>,
    pub finished: HashMap<u32, wgpu::CommandBuffer>,
    pub query_sets: HashMap<u32, wgpu::QuerySet>,
    next: AtomicU32,
}

impl Registry {
    pub fn claim(&self) -> u32 {
        self.next.fetch_add(1, Ordering::Relaxed) + 1
    }

    pub fn buffer(&self, handle: u32) -> Result<&wgpu::Buffer> {
        self.buffers
            .get(&handle)
            .ok_or_else(|| Error::from_reason(format!("no such buffer: {handle}")))
    }

    pub fn encoder(&mut self, handle: u32) -> Result<&mut Encoder> {
        self.encoders
            .get_mut(&handle)
            .ok_or_else(|| Error::from_reason(format!("no such command encoder: {handle}")))
    }

    /// The pass an encoder has open, which every pass call needs.
    pub fn pass(&mut self, encoder: u32) -> Result<&mut Pass> {
        self.encoder(encoder)?
            .open_pass
            .as_mut()
            .ok_or_else(|| Error::from_reason("the command encoder has no open compute pass"))
    }
}

pub struct Context {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub unchecked: bool,
    pub registry: Mutex<Registry>,
    /// Errors raised outside any error scope. wgpu's default handler panics,
    /// and a panic across the N-API boundary aborts the process rather than
    /// reaching the caller, so they are collected here and thrown from the
    /// next submit instead.
    pub uncaptured: Mutex<Vec<String>>,
}

impl Context {
    pub fn wait(&self) {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
    }
}

/// What the adapter reports, kept from before the device exists so the shim
/// can present a WebGPU adapter and let its caller choose limits.
pub struct Chosen {
    pub adapter: wgpu::Adapter,
    pub info: wgpu::AdapterInfo,
    pub features: wgpu::Features,
    pub limits: wgpu::Limits,
    pub matrix_configs: Vec<wgpu::CooperativeMatrixProperties>,
}

static ADAPTER: Mutex<Option<&'static Chosen>> = Mutex::new(None);
static CONTEXT: Mutex<Option<&'static Context>> = Mutex::new(None);

pub fn context() -> &'static Context {
    CONTEXT
        .lock()
        .unwrap()
        .expect("the wgpu device has not been requested")
}

pub fn chosen() -> Result<&'static Chosen> {
    ADAPTER
        .lock()
        .unwrap()
        .ok_or_else(|| Error::from_reason("the wgpu adapter has not been opened"))
}

pub fn registry() -> std::sync::MutexGuard<'static, Registry> {
    context().registry.lock().unwrap()
}

/// Everything the adapter offers that a kernel here can use. Asked for as the
/// intersection with what the adapter has, so a card missing one still opens.
fn wanted_features() -> wgpu::Features {
    wgpu::Features::SHADER_F16
        | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX
        | wgpu::Features::SUBGROUP
        | wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
}

/// Picks the adapter. Its own maxima are reported, not a tier: the caller
/// chooses what to ask for, exactly as it does against Dawn.
pub fn open_adapter() -> Result<&'static Chosen> {
    let mut held = ADAPTER.lock().unwrap();
    if let Some(existing) = *held {
        return Ok(existing);
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .map_err(|error| Error::from_reason(format!("no wgpu adapter: {error}")))?;
    let features = adapter.features() & wanted_features();
    let chosen: &'static Chosen = Box::leak(Box::new(Chosen {
        info: adapter.get_info(),
        limits: adapter.limits(),
        matrix_configs: if features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            adapter.cooperative_matrix_properties()
        } else {
            Vec::new()
        },
        features,
        adapter,
    }));
    *held = Some(chosen);
    Ok(chosen)
}

/// Opens the device with the limits the caller asked for, defaulting the rest.
/// `unchecked` drops the injected bounds and loop checks, to match a Dawn
/// instance built with the robustness transform off.
pub fn request_device(
    unchecked: bool,
    features: wgpu::Features,
    limits: wgpu::Limits,
) -> Result<(&'static Context, wgpu::Limits)> {
    let mut held = CONTEXT.lock().unwrap();
    if let Some(existing) = *held {
        return Ok((existing, existing.device.limits()));
    }
    let chosen = chosen()?;
    let (device, queue) = pollster::block_on(chosen.adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("afwebgpu"),
            required_features: features,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            // Cooperative matrix is experimental in wgpu and gated behind this
            // token. The kernels using it are the same ones Dawn runs behind
            // its own experimental flag.
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        },
    ))
    .map_err(|error| Error::from_reason(format!("no wgpu device: {error}")))?;
    let granted = device.limits();
    let context: &'static Context = Box::leak(Box::new(Context {
        device,
        queue,
        unchecked,
        registry: Mutex::new(Registry::default()),
        uncaptured: Mutex::new(Vec::new()),
    }));
    context
        .device
        .on_uncaptured_error(std::sync::Arc::new(|error: wgpu::Error| {
            if let Some(held) = *CONTEXT.lock().unwrap() {
                held.uncaptured.lock().unwrap().push(error.to_string());
            }
        }));
    *held = Some(context);
    // One thread pumps the device, so a map or a queue wait costs no thread of
    // the caller's. Node runs an AsyncTask on the libuv pool, which is four
    // threads by default, and the model waits on several readbacks at once:
    // a blocking poll for each starves the pool and stalls the fold.
    std::thread::spawn(move || loop {
        context.device.poll(wgpu::PollType::Poll).ok();
        std::thread::sleep(std::time::Duration::from_micros(100));
    });
    Ok((context, granted))
}
