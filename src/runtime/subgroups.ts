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
  // A stub standing in for a device in a test carries no limits at all.
  const limits = device.limits as SubgroupDeviceLimits | undefined;
  if (limits?.minSubgroupSize === undefined || limits.maxSubgroupSize === undefined) {
    return undefined;
  }
  return [limits.minSubgroupSize, limits.maxSubgroupSize];
}

/**
 * Whether this device can be pinned to subgroups of size.
 *
 * The matrix kernels are written for thirty-two lanes a subgroup: they index
 * lane % 32, pair two lanes to a row through subgroupShuffleXor, and lay
 * one matrix tile across exactly one subgroup. They declare @subgroup_size
 * to hold the driver to it, which needs the feature; and a device may expose
 * the feature while advertising a range that does not contain thirty-two —
 * SwiftShader fixes it at [4, 4] — in which case the kernels are simply not
 * for that device.
 *
 * A device whose advertised range is the single value asked for needs no
 * attribute and so no feature: there is no other width for the driver to
 * choose. That is the wgpu case, which implements the subgroup builtins and
 * has no counterpart to subgroup-size-control, and it is why the width is read
 * before the feature rather than after it.
 */
export function supportsSubgroupSize(device: GPUDevice, size: number): boolean {
  if (!device.features.has("subgroups" as GPUFeatureName)) return false;
  const range = subgroupRange(device);
  if (range === undefined) return false;
  if (range[0] === size && range[1] === size) return true;
  return device.features.has("subgroup-size-control" as GPUFeatureName)
    && range[0] <= size && size <= range[1];
}

/** One entry of the adapter's subgroupMatrixConfigs, which the device omits. */
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
