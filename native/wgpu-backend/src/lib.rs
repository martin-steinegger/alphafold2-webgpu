//! A wgpu device driven from Node, over integer handles.
//!
//! Enough of WebGPU to run the model: buffers, modules, pipelines, bind
//! groups, command encoders with one compute pass each, query sets, mapping
//! and the queue. Everything crossing the boundary is a handle, an integer, a
//! string or a byte buffer, so an upstream rename reaches device.rs and
//! nothing else.
//!
//! The runtime checks are switchable because a comparison against Dawn depends
//! on them. Dawn is measured with dawnInstanceFlags({ unclamped: true }),
//! which turns off Tint's robustness transform; wgpu's default injects index
//! clamps, loop counters and integer division wrappers. Comparing the two as
//! they come out of the box compares the checks, not the backends.

mod device;
mod report;

use device::{chosen, context, registry, Command, Encoder, Pass, PassCommand};
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Picks the adapter and returns what it offers, as JSON.
#[napi]
pub fn open_adapter() -> Result<String> {
    let chosen = device::open_adapter()?;
    Ok(report::report(chosen, &chosen.limits, chosen.features, None))
}

/// Opens the device with the features and limits asked for, and returns what
/// was granted. Only the limits the model raises are settable; a name outside
/// that set is refused rather than quietly ignored.
///
/// `unchecked` drops wgpu's injected bounds and loop checks.
#[napi]
pub fn request_device(
    unchecked: bool,
    features: Vec<String>,
    limit_names: Vec<String>,
    limit_values: Vec<f64>,
) -> Result<String> {
    if limit_names.len() != limit_values.len() {
        return Err(Error::from_reason("limit names and values differ in count"));
    }
    let chosen = chosen()?;
    let mut wanted = wgpu::Features::empty();
    for name in &features {
        match name.as_str() {
            "shader-f16" => wanted |= wgpu::Features::SHADER_F16,
            "subgroups" => wanted |= wgpu::Features::SUBGROUP,
            "timestamp-query" => {
                wanted |= wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            }
            "wgpu-cooperative-matrix" => {
                wanted |= wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX
            }
            other => return Err(Error::from_reason(format!("no such wgpu feature: {other}"))),
        }
    }
    if !chosen.features.contains(wanted) {
        return Err(Error::from_reason("the adapter does not have every feature asked for"));
    }
    let mut limits = wgpu::Limits::default().using_resolution(chosen.limits.clone());
    for (name, value) in limit_names.iter().zip(limit_values.iter().copied()) {
        match name.as_str() {
            "maxBufferSize" => limits.max_buffer_size = value as u64,
            "maxStorageBufferBindingSize" => limits.max_storage_buffer_binding_size = value as u64,
            "maxStorageBuffersPerShaderStage" => {
                limits.max_storage_buffers_per_shader_stage = value as u32
            }
            "maxComputeWorkgroupStorageSize" => {
                limits.max_compute_workgroup_storage_size = value as u32
            }
            "maxComputeInvocationsPerWorkgroup" => {
                limits.max_compute_invocations_per_workgroup = value as u32
            }
            "maxComputeWorkgroupSizeX" => limits.max_compute_workgroup_size_x = value as u32,
            other => return Err(Error::from_reason(format!("no settable limit: {other}"))),
        }
    }
    let (_, granted) = device::request_device(unchecked, wanted, limits)?;
    Ok(report::report(chosen, &granted, wanted, Some(!unchecked)))
}

/// `usage` carries the WebGPU GPUBufferUsage bits. Each is translated by name
/// rather than by a truncating bit cast, so a bit wgpu has no counterpart for
/// is refused instead of dropped.
#[napi]
pub fn create_buffer(bytes: f64, usage: u32, mapped: bool) -> Result<u32> {
    const WEBGPU_USAGES: [(u32, wgpu::BufferUsages); 7] = [
        (1, wgpu::BufferUsages::MAP_READ),
        (2, wgpu::BufferUsages::MAP_WRITE),
        (4, wgpu::BufferUsages::COPY_SRC),
        (8, wgpu::BufferUsages::COPY_DST),
        (64, wgpu::BufferUsages::UNIFORM),
        (128, wgpu::BufferUsages::STORAGE),
        (512, wgpu::BufferUsages::QUERY_RESOLVE),
    ];
    let mut translated = wgpu::BufferUsages::empty();
    let mut left = usage;
    for (bit, flag) in WEBGPU_USAGES {
        if usage & bit != 0 {
            translated |= flag;
            left &= !bit;
        }
    }
    if left != 0 {
        return Err(Error::from_reason(format!("unsupported buffer usage bits: {left:#x}")));
    }
    let buffer = context().device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes as u64,
        usage: translated,
        mapped_at_creation: mapped,
    });
    let mut registry = registry();
    let handle = registry.claim();
    registry.buffers.insert(handle, buffer);
    Ok(handle)
}

#[napi]
pub fn write_buffer(handle: u32, offset: f64, data: Buffer) -> Result<()> {
    let context = context();
    let registry = registry();
    context
        .queue
        .write_buffer(registry.buffer(handle)?, offset as u64, &data);
    Ok(())
}

/// Compiles WGSL. There is no second entry point: SPIR-V passthrough would
/// commit us to writing MSL and HLSL kernels as well.
#[napi]
pub fn create_shader_module(code: String) -> u32 {
    let context = context();
    let descriptor = wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(code.into()),
    };
    let module = if context.unchecked {
        // Safety: the kernels index workgroup and storage arrays through
        // bounds this project computes, and the same sources run under Dawn
        // with its robustness transform off.
        unsafe {
            context
                .device
                .create_shader_module_trusted(descriptor, wgpu::ShaderRuntimeChecks::unchecked())
        }
    } else {
        context.device.create_shader_module(descriptor)
    };
    let mut registry = registry();
    let handle = registry.claim();
    registry.modules.insert(handle, module);
    handle
}

/// `constants` is a flat name, value, name, value list: napi has no ordered
/// map, and the pairs are few and built beside this call.
#[napi]
pub fn create_compute_pipeline(
    module: u32,
    entry: String,
    constant_names: Vec<String>,
    constant_values: Vec<f64>,
) -> Result<u32> {
    if constant_names.len() != constant_values.len() {
        return Err(Error::from_reason("override names and values differ in count"));
    }
    let context = context();
    let constants: Vec<(&str, f64)> = constant_names
        .iter()
        .map(String::as_str)
        .zip(constant_values.iter().copied())
        .collect();
    let pipeline = {
        let registry = registry();
        let module = registry
            .modules
            .get(&module)
            .ok_or_else(|| Error::from_reason("no such shader module"))?;
        context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module,
                entry_point: Some(&entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    // On whatever the checks are doing, because Dawn does it:
                    // disabling Tint's workgroup initialisation is a separate
                    // toggle from the robustness transform, and dawnInstanceFlags
                    // turns off only the second. The matrix attention kernel
                    // depends on the zeroing, and folds to a pLDDT of 48 without
                    // it, so tying this to unchecked made a whole prediction
                    // wrong for a saving that was never measured.
                    zero_initialize_workgroup_memory: true,
                },
                cache: None,
            })
    };
    let mut registry = registry();
    let handle = registry.claim();
    registry.pipelines.insert(handle, pipeline);
    Ok(handle)
}

/// Binds buffer ranges in binding order. A zero size means the rest of the
/// buffer, as WebGPU's omitted size does. `group_index` selects the pipeline's
/// automatic layout.
#[napi]
pub fn create_bind_group(
    pipeline: u32,
    group_index: u32,
    buffers: Vec<u32>,
    offsets: Vec<f64>,
    sizes: Vec<f64>,
) -> Result<u32> {
    if offsets.len() != buffers.len() || sizes.len() != buffers.len() {
        return Err(Error::from_reason("bindings, offsets and sizes differ in count"));
    }
    let context = context();
    let mut registry = registry();
    let handle = registry.claim();
    let group = {
        let layout = registry
            .pipelines
            .get(&pipeline)
            .ok_or_else(|| Error::from_reason("no such pipeline"))?
            .get_bind_group_layout(group_index);
        let mut bound = Vec::with_capacity(buffers.len());
        for (binding, buffer) in buffers.iter().enumerate() {
            bound.push(wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: registry.buffer(*buffer)?,
                    offset: offsets[binding] as u64,
                    size: wgpu::BufferSize::new(sizes[binding] as u64),
                }),
            });
        }
        context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &bound,
        })
    };
    registry.groups.insert(handle, group);
    Ok(handle)
}

#[napi]
pub fn create_query_set(count: u32) -> u32 {
    let context = context();
    let set = context.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count,
    });
    let mut registry = registry();
    let handle = registry.claim();
    registry.query_sets.insert(handle, set);
    handle
}

#[napi]
pub fn create_command_encoder() -> u32 {
    let mut registry = registry();
    let handle = registry.claim();
    registry.encoders.insert(handle, Encoder::default());
    handle
}

/// Opens the encoder's compute pass. A negative query set means no timestamps;
/// a negative write index means that end of the pass is not written.
#[napi]
pub fn begin_compute_pass(
    encoder: u32,
    query_set: i32,
    beginning: i32,
    end: i32,
) -> Result<()> {
    let mut registry = registry();
    let encoder = registry.encoder(encoder)?;
    if encoder.open_pass.is_some() {
        return Err(Error::from_reason("a compute pass is already open"));
    }
    let index = |value: i32| if value < 0 { None } else { Some(value as u32) };
    encoder.open_pass = Some(Pass {
        commands: Vec::new(),
        timestamps: if query_set < 0 {
            None
        } else {
            Some((query_set as u32, index(beginning), index(end)))
        },
    });
    Ok(())
}

#[napi]
pub fn pass_set_pipeline(encoder: u32, pipeline: u32) -> Result<()> {
    registry()
        .pass(encoder)?
        .commands
        .push(PassCommand::SetPipeline(pipeline));
    Ok(())
}

#[napi]
pub fn pass_set_bind_group(encoder: u32, index: u32, group: u32) -> Result<()> {
    registry()
        .pass(encoder)?
        .commands
        .push(PassCommand::SetBindGroup(index, group));
    Ok(())
}

#[napi]
pub fn pass_dispatch(encoder: u32, x: u32, y: u32, z: u32) -> Result<()> {
    registry()
        .pass(encoder)?
        .commands
        .push(PassCommand::Dispatch(x, y, z));
    Ok(())
}

#[napi]
pub fn pass_push_debug_group(encoder: u32, label: String) -> Result<()> {
    registry()
        .pass(encoder)?
        .commands
        .push(PassCommand::PushDebugGroup(label));
    Ok(())
}

#[napi]
pub fn pass_pop_debug_group(encoder: u32) -> Result<()> {
    registry()
        .pass(encoder)?
        .commands
        .push(PassCommand::PopDebugGroup);
    Ok(())
}

#[napi]
pub fn pass_end(encoder: u32) -> Result<()> {
    let mut registry = registry();
    let encoder = registry.encoder(encoder)?;
    let pass = encoder
        .open_pass
        .take()
        .ok_or_else(|| Error::from_reason("no compute pass is open"))?;
    encoder.commands.push(Command::Pass(pass));
    Ok(())
}

#[napi]
pub fn copy_buffer_to_buffer(
    encoder: u32,
    source: u32,
    source_offset: f64,
    destination: u32,
    destination_offset: f64,
    bytes: f64,
) -> Result<()> {
    registry().encoder(encoder)?.commands.push(Command::Copy {
        source,
        source_offset: source_offset as u64,
        destination,
        destination_offset: destination_offset as u64,
        bytes: bytes as u64,
    });
    Ok(())
}

/// A zero size clears to the end of the buffer, as WebGPU's omitted size does.
#[napi]
pub fn clear_buffer(encoder: u32, buffer: u32, offset: f64, bytes: f64) -> Result<()> {
    registry().encoder(encoder)?.commands.push(Command::Clear {
        buffer,
        offset: offset as u64,
        bytes: bytes as u64,
    });
    Ok(())
}

#[napi]
pub fn resolve_query_set(
    encoder: u32,
    set: u32,
    first: u32,
    count: u32,
    destination: u32,
    offset: f64,
) -> Result<()> {
    registry()
        .encoder(encoder)?
        .commands
        .push(Command::ResolveQuerySet {
            set,
            first,
            count,
            destination,
            offset: offset as u64,
        });
    Ok(())
}

/// Replays the recorded commands into a real encoder and keeps the result
/// until it is submitted.
#[napi]
pub fn finish(encoder: u32) -> Result<u32> {
    let context = context();
    let mut registry = registry();
    let recorded = registry
        .encoders
        .remove(&encoder)
        .ok_or_else(|| Error::from_reason("no such command encoder"))?;
    if recorded.open_pass.is_some() {
        return Err(Error::from_reason("a compute pass is still open"));
    }
    let mut raw = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    for command in &recorded.commands {
        match command {
            Command::Copy {
                source,
                source_offset,
                destination,
                destination_offset,
                bytes,
            } => raw.copy_buffer_to_buffer(
                registry.buffer(*source)?,
                *source_offset,
                registry.buffer(*destination)?,
                *destination_offset,
                *bytes,
            ),
            Command::Clear {
                buffer,
                offset,
                bytes,
            } => raw.clear_buffer(
                registry.buffer(*buffer)?,
                *offset,
                if *bytes == 0 { None } else { Some(*bytes) },
            ),
            Command::ResolveQuerySet {
                set,
                first,
                count,
                destination,
                offset,
            } => {
                let set = registry
                    .query_sets
                    .get(set)
                    .ok_or_else(|| Error::from_reason("no such query set"))?;
                raw.resolve_query_set(
                    set,
                    *first..*first + *count,
                    registry.buffer(*destination)?,
                    *offset,
                );
            }
            Command::Pass(pass) => replay(&registry, &mut raw, pass)?,
        }
    }
    let handle = registry.claim();
    registry.finished.insert(handle, raw.finish());
    Ok(handle)
}

fn replay(
    registry: &device::Registry,
    encoder: &mut wgpu::CommandEncoder,
    pass: &Pass,
) -> Result<()> {
    let writes = match pass.timestamps {
        None => None,
        Some((set, beginning, end)) => Some(wgpu::ComputePassTimestampWrites {
            query_set: registry
                .query_sets
                .get(&set)
                .ok_or_else(|| Error::from_reason("no such query set"))?,
            beginning_of_pass_write_index: beginning,
            end_of_pass_write_index: end,
        }),
    };
    let mut raw = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: writes,
    });
    for command in &pass.commands {
        match command {
            PassCommand::SetPipeline(pipeline) => raw.set_pipeline(
                registry
                    .pipelines
                    .get(pipeline)
                    .ok_or_else(|| Error::from_reason("no such pipeline"))?,
            ),
            PassCommand::SetBindGroup(index, group) => raw.set_bind_group(
                *index,
                registry
                    .groups
                    .get(group)
                    .ok_or_else(|| Error::from_reason("no such bind group"))?,
                &[],
            ),
            PassCommand::Dispatch(x, y, z) => raw.dispatch_workgroups(*x, *y, *z),
            PassCommand::PushDebugGroup(label) => raw.push_debug_group(label),
            PassCommand::PopDebugGroup => raw.pop_debug_group(),
        }
    }
    Ok(())
}

/// The errors raised outside any error scope since the last call, and clears
/// them. The shim throws them from the call that follows the one that failed.
#[napi]
pub fn take_uncaptured_errors() -> Vec<String> {
    std::mem::take(&mut *context().uncaptured.lock().unwrap())
}

#[napi]
pub fn submit(buffers: Vec<u32>) -> Result<()> {
    let context = context();
    let mut registry = registry();
    let mut finished = Vec::with_capacity(buffers.len());
    for handle in buffers {
        finished.push(
            registry
                .finished
                .remove(&handle)
                .ok_or_else(|| Error::from_reason("no such command buffer"))?,
        );
    }
    context.queue.submit(finished);
    Ok(())
}

/// Runs one pipeline `repeats` times in one submission and returns the wall
/// time in milliseconds, for a microbenchmark that wants the dispatches
/// timed rather than the host round trip around them.
#[napi]
pub fn time_dispatches(
    pipeline: u32,
    group: u32,
    x: u32,
    y: u32,
    z: u32,
    repeats: u32,
) -> Result<f64> {
    let context = context();
    let registry = registry();
    let pipeline = registry
        .pipelines
        .get(&pipeline)
        .ok_or_else(|| Error::from_reason("no such pipeline"))?;
    let group = registry
        .groups
        .get(&group)
        .ok_or_else(|| Error::from_reason("no such bind group"))?;
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        for _ in 0..repeats {
            pass.dispatch_workgroups(x, y, z);
        }
    }
    let started = std::time::Instant::now();
    context.queue.submit(Some(encoder.finish()));
    context.wait();
    Ok(started.elapsed().as_secs_f64() * 1000.0)
}

/// Maps a buffer, resolving when wgpu says the mapping is ready.
///
/// The promise is settled from the device's own polling thread. An AsyncTask
/// would settle it from the libuv pool, whose four threads the model can
/// exhaust waiting on several readbacks at once.
#[napi(ts_return_type = "Promise<void>")]
pub fn map_buffer(env: Env, buffer: u32, write: bool) -> Result<Object> {
    let (deferred, promise) = env.create_deferred()?;
    let mode = if write {
        wgpu::MapMode::Write
    } else {
        wgpu::MapMode::Read
    };
    let registry = registry();
    registry
        .buffer(buffer)?
        .slice(..)
        .map_async(mode, move |result| match result {
            Ok(()) => deferred.resolve(|_| Ok(())),
            Err(error) => deferred.reject(Error::from_reason(format!("the map failed: {error}"))),
        });
    Ok(promise)
}

/// A copy of the mapped range. WebGPU hands out a view that stays valid until
/// unmap; a copy is simpler across the boundary and the ranges here are read
/// once into a typed array anyway.
#[napi]
pub fn mapped_range(buffer: u32, offset: f64, bytes: f64) -> Result<Buffer> {
    let registry = registry();
    let buffer = registry.buffer(buffer)?;
    let offset = offset as u64;
    let range = buffer.slice(offset..offset + bytes as u64);
    let view = range
        .get_mapped_range()
        .map_err(|error| Error::from_reason(format!("the buffer is not mapped: {error}")))?;
    Ok(Buffer::from(view.to_vec())) // the view borrows the buffer, so it cannot escape
}

#[napi]
pub fn write_mapped_range(buffer: u32, offset: f64, data: Buffer) -> Result<()> {
    let registry = registry();
    let target = registry.buffer(buffer)?;
    let offset = offset as u64;
    let mut view = target
        .slice(offset..offset + data.len() as u64)
        .get_mapped_range_mut()
        .map_err(|error| Error::from_reason(format!("the buffer is not mapped: {error}")))?;
    view.copy_from_slice(&data);
    Ok(())
}

#[napi]
pub fn unmap(buffer: u32) -> Result<()> {
    registry().buffer(buffer)?.unmap();
    Ok(())
}

/// Resolves when everything submitted so far has run, settled from the
/// device's polling thread for the same reason the map is.
#[napi(ts_return_type = "Promise<void>")]
pub fn on_submitted_work_done(env: Env) -> Result<Object> {
    let (deferred, promise) = env.create_deferred()?;
    context()
        .queue
        .on_submitted_work_done(move || deferred.resolve(|_| Ok(())));
    Ok(promise)
}

// The scope guards are neither Send nor Sync, because wgpu keeps error scopes
// per thread. Both calls come from the JavaScript thread, so a thread-local
// stack is the shape that fits.
thread_local! {
    static SCOPES: std::cell::RefCell<Vec<wgpu::ErrorScopeGuard>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[napi]
pub fn push_error_scope() {
    let guard = context().device.push_error_scope(wgpu::ErrorFilter::Validation);
    SCOPES.with(|scopes| scopes.borrow_mut().push(guard));
}

/// The message of the first error in the scope, or an empty string.
///
/// The device is not polled first. wgpu validates a call as it is made, so a
/// validation error is already recorded in the scope by the time this runs,
/// and waiting for the queue here would serialise the caller against the GPU
/// at every one of the model's error scopes.
#[napi]
pub fn pop_error_scope() -> Result<String> {
    let guard = SCOPES
        .with(|scopes| scopes.borrow_mut().pop())
        .ok_or_else(|| Error::from_reason("no error scope is open"))?;
    Ok(match pollster::block_on(guard.pop()) {
        Some(error) => error.to_string(),
        None => String::new(),
    })
}

/// The size of each map, so a test can fold twice and compare.
#[napi]
pub fn handle_counts() -> Vec<u32> {
    let registry = registry();
    vec![
        registry.buffers.len() as u32,
        registry.modules.len() as u32,
        registry.pipelines.len() as u32,
        registry.groups.len() as u32,
        registry.encoders.len() as u32,
        registry.finished.len() as u32,
        registry.query_sets.len() as u32,
    ]
}

#[napi]
pub fn destroy_bind_group(handle: u32) {
    registry().groups.remove(&handle);
}

#[napi]
pub fn destroy_pipeline(handle: u32) {
    registry().pipelines.remove(&handle);
}

#[napi]
pub fn destroy_shader_module(handle: u32) {
    registry().modules.remove(&handle);
}

#[napi]
pub fn destroy_query_set(handle: u32) {
    registry().query_sets.remove(&handle);
}

#[napi]
pub fn destroy_buffer(handle: u32) {
    if let Some(buffer) = registry().buffers.remove(&handle) {
        buffer.destroy();
    }
}
