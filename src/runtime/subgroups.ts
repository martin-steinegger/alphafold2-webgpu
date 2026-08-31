interface SubgroupAdapterInfo extends GPUAdapterInfo {
  readonly subgroupMinSize?: number;
  readonly subgroupMaxSize?: number;
}

interface SubgroupDeviceLimits extends GPUSupportedLimits {
  readonly minSubgroupSize?: number;
  readonly maxSubgroupSize?: number;
}

const subgroupRanges = new WeakMap<GPUDevice, readonly [number, number]>();

/** Records adapter-only subgroup properties on the device selected for inference. */
export function recordSubgroupRange(device: GPUDevice, adapter: GPUAdapter): void {
  const info = adapter.info as SubgroupAdapterInfo | undefined;
  if (info === undefined) return;
  const min = info.subgroupMinSize;
  const max = info.subgroupMaxSize;
  if (min !== undefined && max !== undefined
    && Number.isSafeInteger(min) && Number.isSafeInteger(max) && min > 0 && min <= max) {
    subgroupRanges.set(device, [min, max]);
  }
}

/** Returns the advertised subgroup range across current and older WebGPU bindings. */
export function subgroupRange(device: GPUDevice): readonly [number, number] | undefined {
  const recorded = subgroupRanges.get(device);
  if (recorded !== undefined) return recorded;
  const limits = device.limits as SubgroupDeviceLimits;
  if (limits.minSubgroupSize === undefined || limits.maxSubgroupSize === undefined) return undefined;
  return [limits.minSubgroupSize, limits.maxSubgroupSize];
}
