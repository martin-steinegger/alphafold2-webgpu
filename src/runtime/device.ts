/** Requests optional WebGPU features used by AFWebGPU's exact fast paths. */
export async function requestAlphaFoldDevice(adapter: GPUAdapter): Promise<GPUDevice> {
  // subgroup-size-control is shipping ahead of the current @webgpu/types union.
  const optional = ["subgroups", "subgroup-size-control", "timestamp-query"] as const;
  const requiredFeatures = optional.filter(
    (feature) => adapter.features.has(feature as GPUFeatureName),
  ) as GPUFeatureName[];
  return adapter.requestDevice({ requiredFeatures });
}
