import {
  createAttentionParameters, selectAttentionFlashKernel, supportsAttentionMatrix,
  supportsAttentionSubgroups,
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
/**
 * A batch long enough that the queue round trip is not what is measured.
 *
 * One dispatch of this probe runs in about a tenth of a millisecond, and a
 * wall clock around a submit and its completion cannot see that: on a card
 * another process is saturating, a single dispatch of the same kernel reads
 * anywhere between 0.15 ms and 2.4 ms, which ranks the quiet slots rather than
 * the kernels. Batching many dispatches into one command buffer and dividing
 * puts the kernel back in charge of the number. `measureGemmVariants` is sized
 * the same way and for the same reason.
 */
const PROBE_BATCH_MILLISECONDS = 12;
const PROBE_ROUGH_DISPATCHES = 4;
const PROBE_MAX_DISPATCHES = 2000;

const calibrations = new WeakMap<GPUDevice, Map<number, Promise<AttentionFlashKernel>>>();

/** Kernels worth measuring against each other on this device. */
export function attentionFlashCandidates(device: GPUDevice, headDim: number): readonly AttentionFlashVariant[] {
  const candidates: AttentionFlashVariant[] = [];
  if (headDim % 4 === 0 && headDim <= 32) candidates.push("register");
  // The matrix units reduce each query-key dot product in hardware, so unlike
  // every subgroup variant this one pays no cross-lane traffic for it. It is
  // measured rather than assumed: it stages its tiles, which the register
  // kernel does not, and only wins where that trade pays.
  if (supportsAttentionMatrix(device, headDim)) candidates.push("matrix");
  if (supportsAttentionSubgroups(device, headDim)) {
    candidates.push(selectAttentionFlashKernel(device, headDim, "auto").variant);
  }
  return [...new Set(candidates)];
}

/** One candidate, ready to dispatch, so the sweep below can interleave them. */
interface FlashTiming {
  readonly kernel: AttentionFlashKernel;
  readonly dispatch: (dispatches: number) => Promise<number>;
  /** Sized so one batch reaches `PROBE_BATCH_MILLISECONDS`, once, up front. */
  dispatches: number;
  best: number;
}

async function prepareFlashKernel(
  device: GPUDevice, kernel: AttentionFlashKernel, buffers: readonly GPUBuffer[], heads: number,
): Promise<FlashTiming> {
  const pipeline = await pipelineCacheForDevice(device).get(kernel.cacheKey, kernel.shader);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  // Dispatched the way the kernel asks. Sent the query blocks where a
  // batch-first kernel looks for its batch, the probe gives it a quarter of
  // the work — the blocks past the query count exit at once — and times it as
  // though it had done all of it.
  const blocks = Math.ceil(PROBE_QUERIES / kernel.queryTile);
  const groupsX = kernel.batchFirst === true ? PROBE_BATCH : blocks;
  const groupsY = kernel.batchFirst === true ? blocks : PROBE_BATCH;
  const dispatch = async (dispatches: number): Promise<number> => {
    const encoder = device.createCommandEncoder({ label: `calibrate.${kernel.variant}` });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    for (let index = 0; index < dispatches; index += 1) {
      pass.dispatchWorkgroups(groupsX, groupsY, heads);
    }
    pass.end();
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - start) / dispatches;
  };
  return {
    kernel, dispatch, dispatches: PROBE_ROUGH_DISPATCHES, best: Number.POSITIVE_INFINITY,
  };
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
    // The probe has no bias, but the matrix kernel binds one as vectors and a
    // binding of a single element is under the minimum size for that.
    const pairBias = create(4, storage);
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
    const timings: FlashTiming[] = [];
    for (const variant of candidates) {
      timings.push(await prepareFlashKernel(
        device, selectAttentionFlashKernel(device, headDim, variant), bound, heads,
      ));
    }
    // Warm every candidate before any of them is timed, then sweep them
    // round-robin rather than finishing one before starting the next. Swept
    // the other way each candidate owns a different window of wall time, so a
    // clock ramping across the sweep ranks the windows and not the kernels: on
    // a Strix Halo the same two kernels at headDim 8 read 0.298 against 0.212
    // ms in one order and 0.220 against 0.257 in the other.
    // `measureGemmVariants` is swept the same way.
    // The rough pass warms the pipeline as well as sizing the batch, so its
    // own time is discarded.
    for (const timing of timings) {
      const rough = await timing.dispatch(PROBE_ROUGH_DISPATCHES);
      timing.dispatches = Math.max(PROBE_ROUGH_DISPATCHES, Math.min(PROBE_MAX_DISPATCHES,
        Math.ceil(PROBE_BATCH_MILLISECONDS / Math.max(rough, 0.01))));
    }
    for (let repeat = 0; repeat < PROBE_REPEATS; repeat += 1) {
      for (const timing of timings) {
        timing.best = Math.min(timing.best, await timing.dispatch(timing.dispatches));
      }
    }
    const best = [...timings].sort((left, right) => left.best - right.best)[0];
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
