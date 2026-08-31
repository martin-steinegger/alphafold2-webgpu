import {
  createAttentionParameters, selectAttentionFlashKernel, supportsAttentionSubgroups,
  type AttentionFlashKernel, type AttentionFlashVariant, type AttentionInput,
} from "./attention.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

/**
 * Runtime selection of the flash-attention kernel.
 *
 * The static heuristic in `selectAttentionFlashKernel` prefers a subgroup
 * kernel wherever the device advertises subgroups with a 32-lane range and
 * 16 KiB of workgroup storage. On discrete NVIDIA hardware that heuristic is
 * inverted: the register kernel, which keeps one whole head per invocation and
 * needs no cross-lane traffic, measured 3.6x faster than every subgroup variant
 * at each MSA shape the Evoformer runs. The ranking is a property of the driver
 * and cannot be derived from any limit WebGPU exposes, so it is measured once
 * per device instead of guessed.
 *
 * The probe runs only when more than one kernel is actually available, uses
 * transient buffers it frees immediately, and falls back to the static
 * heuristic if anything about it fails.
 */

/** Rows and columns are large enough to be throughput-bound, small enough to stay cheap. */
const PROBE_BATCH = 16;
const PROBE_QUERIES = 256;
const PROBE_REPEATS = 3;

const calibrations = new WeakMap<GPUDevice, Map<number, Promise<AttentionFlashKernel>>>();

/** Kernels worth measuring against each other on this device. */
export function attentionFlashCandidates(device: GPUDevice, headDim: number): readonly AttentionFlashVariant[] {
  const candidates: AttentionFlashVariant[] = [];
  if (headDim % 4 === 0 && headDim <= 32) candidates.push("register");
  if (supportsAttentionSubgroups(device, headDim)) {
    candidates.push(selectAttentionFlashKernel(device, headDim, "auto").variant);
  }
  return [...new Set(candidates)];
}

async function timeFlashKernel(
  device: GPUDevice, kernel: AttentionFlashKernel, buffers: readonly GPUBuffer[], heads: number,
): Promise<number> {
  const pipeline = await pipelineCacheForDevice(device).get(kernel.cacheKey, kernel.shader);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const groupsX = Math.ceil(PROBE_QUERIES / kernel.queryTile);
  const dispatch = async (): Promise<number> => {
    const encoder = device.createCommandEncoder({ label: `calibrate.${kernel.variant}` });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(groupsX, PROBE_BATCH, heads);
    pass.end();
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return performance.now() - start;
  };
  await dispatch();
  let best = Number.POSITIVE_INFINITY;
  for (let repeat = 0; repeat < PROBE_REPEATS; repeat += 1) best = Math.min(best, await dispatch());
  return best;
}

async function measureFlashKernel(device: GPUDevice, headDim: number): Promise<AttentionFlashKernel> {
  const fallback = selectAttentionFlashKernel(device, headDim, "auto");
  const candidates = attentionFlashCandidates(device, headDim);
  if (candidates.length < 2) return fallback;
  const heads = 8;
  const channels = heads * headDim;
  const rows = PROBE_BATCH * PROBE_QUERIES;
  const buffers: GPUBuffer[] = [];
  const create = (elements: number, usage: GPUBufferUsageFlags): GPUBuffer => {
    const buffer = device.createBuffer({ label: "calibrate.flash", size: elements * 4, usage });
    buffers.push(buffer);
    return buffer;
  };
  try {
    const storage = GPUBufferUsage.STORAGE;
    const query = create(rows * channels, storage);
    const key = create(rows * channels, storage);
    const value = create(rows * channels, storage);
    const gate = create(rows * channels, storage);
    const mask = create(rows, storage | GPUBufferUsage.COPY_DST);
    const pairBias = create(1, storage);
    const output = create(rows * channels, storage);
    const descriptor = {
      activations: new Float32Array(0), mask: new Float32Array(0),
      batch: PROBE_BATCH, queryLength: PROBE_QUERIES, channels, heads,
    } as unknown as AttentionInput;
    const parameterValues = createAttentionParameters(descriptor, new Array(16).fill(0));
    const parameters = device.createBuffer({
      label: "calibrate.flash-parameters", size: parameterValues.byteLength,
      usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true,
    });
    new Uint32Array(parameters.getMappedRange()).set(parameterValues);
    parameters.unmap();
    buffers.push(parameters);
    // An all-ones mask keeps every key live so the measurement covers the full loop.
    device.queue.writeBuffer(mask, 0, new Float32Array(rows).fill(1));

    const bound = [query, key, value, gate, mask, pairBias, parameters, output];
    let best: { kernel: AttentionFlashKernel; milliseconds: number } | undefined;
    for (const variant of candidates) {
      const kernel = selectAttentionFlashKernel(device, headDim, variant);
      const milliseconds = await timeFlashKernel(device, kernel, bound, heads);
      if (best === undefined || milliseconds < best.milliseconds) best = { kernel, milliseconds };
    }
    return best?.kernel ?? fallback;
  } catch {
    return fallback;
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
}

/**
 * Query count from which blocking two queries into one invocation pays off.
 *
 * The register kernel reads every key and value once per query it serves, so
 * holding two queries halves that traffic. Below this many queries the second
 * slot is mostly out of range and the padding costs more than the sharing
 * saves: measured on GB10 at 1.17x-1.42x for 128 to 1024 queries, and 0.89x at
 * 59, which is what row attention runs.
 */
export const REGISTER_QUERY_BLOCK_THRESHOLD = 128;

/**
 * Flash kernel for one attention shape.
 *
 * The device measurement chooses the kernel family; the query count then picks
 * how many queries one invocation should carry.
 */
export async function attentionFlashKernelForShape(
  device: GPUDevice, headDim: number, queries: number,
): Promise<AttentionFlashKernel> {
  const calibrated = await calibrateAttentionFlashKernel(device, headDim);
  if (!calibrated.variant.startsWith("register")) return calibrated;
  return selectAttentionFlashKernel(device, headDim,
    queries >= REGISTER_QUERY_BLOCK_THRESHOLD ? "register-2q" : "register");
}

/** Fastest measured flash kernel for this device and head dimension, measured once. */
export function calibrateAttentionFlashKernel(device: GPUDevice, headDim: number): Promise<AttentionFlashKernel> {
  let byHeadDim = calibrations.get(device);
  if (byHeadDim === undefined) {
    byHeadDim = new Map();
    calibrations.set(device, byHeadDim);
  }
  let calibration = byHeadDim.get(headDim);
  if (calibration === undefined) {
    calibration = measureFlashKernel(device, headDim)
      .catch(() => selectAttentionFlashKernel(device, headDim, "auto"));
    byHeadDim.set(headDim, calibration);
  }
  return calibration;
}

/** Overrides the measurement, for differential tests and benchmarks. */
export function presetAttentionFlashKernel(
  device: GPUDevice, headDim: number, variant: AttentionFlashVariant,
): void {
  let byHeadDim = calibrations.get(device);
  if (byHeadDim === undefined) {
    byHeadDim = new Map();
    calibrations.set(device, byHeadDim);
  }
  byHeadDim.set(headDim, Promise.resolve(selectAttentionFlashKernel(device, headDim, variant)));
}
