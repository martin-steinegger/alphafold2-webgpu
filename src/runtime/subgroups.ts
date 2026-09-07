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

/** One entry of the adapter's `subgroupMatrixConfigs`, which the device omits. */
export interface RecordedMatrixConfig {
  readonly componentType: string;
  readonly resultComponentType: string;
  readonly M: number;
  readonly N: number;
  readonly K: number;
}

const matrixConfigs = new WeakMap<GPUDevice, readonly RecordedMatrixConfig[]>();

/**
 * Records which matrix shapes the units implement.
 *
 * The list is on the adapter and not on the device, and a kernel that wants it
 * has only the device, so it is stashed here the way the subgroup range above
 * already is.
 */
export function recordSubgroupMatrixConfigs(device: GPUDevice, adapter: GPUAdapter): void {
  const info = adapter.info as unknown as
    { subgroupMatrixConfigs?: readonly RecordedMatrixConfig[] } | undefined;
  const configs = info?.subgroupMatrixConfigs;
  if (configs !== undefined && configs.length > 0) matrixConfigs.set(device, configs);
}

/** The recorded shapes, empty on a device that reported none. */
export function subgroupMatrixConfigs(device: GPUDevice): readonly RecordedMatrixConfig[] {
  return matrixConfigs.get(device) ?? [];
}
