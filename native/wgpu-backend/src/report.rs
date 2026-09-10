//! What the adapter offers, in the shape the WebGPU shim presents it.
//!
//! Written as JSON by hand rather than through serde: it is one object built
//! once at open time, and a serialisation dependency would be carried into
//! every fork build for it.

use crate::device::Chosen;

/// Which wgpu tree this binary was built against, so a measurement can say
/// whether it came from the pinned release or from a fork.
const WGPU_SOURCE: &str = match option_env!("AFWEBGPU_WGPU_PATH") {
    Some(path) => path,
    None => "crates.io",
};

/// The WebGPU names for the features the model asks about. wgpu has no
/// subgroup-size-control, so the shim reports none and the dialect leaves the
/// entry point's width attribute off, which is what naga accepts anyway.
fn feature_names(features: wgpu::Features) -> Vec<&'static str> {
    let mut names = Vec::new();
    for (flag, name) in [
        (wgpu::Features::SHADER_F16, "shader-f16"),
        (wgpu::Features::SUBGROUP, "subgroups"),
        (wgpu::Features::SUBGROUP_BARRIER, "subgroup-barrier"),
        (wgpu::Features::TIMESTAMP_QUERY, "timestamp-query"),
        (
            wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
            "wgpu-cooperative-matrix",
        ),
    ] {
        if features.contains(flag) {
            names.push(name);
        }
    }
    names
}

fn scalar(kind: wgpu::CooperativeScalarType) -> &'static str {
    match kind {
        wgpu::CooperativeScalarType::F32 => "f32",
        wgpu::CooperativeScalarType::F16 => "f16",
        wgpu::CooperativeScalarType::I32 => "i32",
        wgpu::CooperativeScalarType::U32 => "u32",
    }
}

fn quoted(values: &[&str]) -> String {
    let joined: Vec<String> = values.iter().map(|value| format!("\"{value}\"")).collect();
    format!("[{}]", joined.join(","))
}

/// The adapter as the shim presents it. `limits` and `features` are whichever
/// set the caller should see: the adapter's own before a device exists, the
/// granted ones after.
///
/// A device must report what it was granted, not what the adapter has. The
/// calibration key is built from `device.features`, so reporting the adapter's
/// set makes the key identical whether or not the feature was asked for, and a
/// cached answer measured without the matrix units is then reused by a build
/// that has them.
pub fn report(
    chosen: &Chosen,
    limits: &wgpu::Limits,
    features: wgpu::Features,
    checks: Option<bool>,
) -> String {
    let info = &chosen.info;
    let configs: Vec<String> = chosen
        .matrix_configs
        .iter()
        .map(|config| {
            format!(
                "{{\"M\":{},\"N\":{},\"K\":{},\"componentType\":\"{}\",\
                 \"resultComponentType\":\"{}\"}}",
                config.m_size,
                config.n_size,
                config.k_size,
                scalar(config.ab_type),
                scalar(config.cr_type)
            )
        })
        .collect();
    format!(
        "{{\"name\":\"{}\",\"driver\":\"{}\",\"backend\":\"{:?}\",\
         \"wgpuSource\":\"{}\",\"checks\":{},\
         \"features\":{},\"subgroupMinSize\":{},\"subgroupMaxSize\":{},\
         \"matrixConfigs\":[{}],\"limits\":{{\
         \"maxBufferSize\":{},\"maxStorageBufferBindingSize\":{},\
         \"maxStorageBuffersPerShaderStage\":{},\"maxComputeWorkgroupStorageSize\":{},\
         \"maxComputeInvocationsPerWorkgroup\":{},\"maxComputeWorkgroupSizeX\":{},\
         \"maxComputeWorkgroupsPerDimension\":{},\"minStorageBufferOffsetAlignment\":{}}}}}",
        info.name.replace('"', "'"),
        info.driver.replace('"', "'"),
        info.backend,
        WGPU_SOURCE.replace('\\', "/"),
        match checks { Some(on) => on.to_string(), None => "null".to_string() },
        quoted(&feature_names(features)),
        info.subgroup_min_size,
        info.subgroup_max_size,
        configs.join(","),
        limits.max_buffer_size,
        limits.max_storage_buffer_binding_size,
        limits.max_storage_buffers_per_shader_stage,
        limits.max_compute_workgroup_storage_size,
        limits.max_compute_invocations_per_workgroup,
        limits.max_compute_workgroup_size_x,
        limits.max_compute_workgroups_per_dimension,
        limits.min_storage_buffer_offset_alignment,
    )
}
