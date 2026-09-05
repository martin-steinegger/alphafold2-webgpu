import { packHalfWords } from "./storage.js";
import {
  createAttentionRegisterFlashShader, type AttentionKeyValueStorage,
} from "../evoformer/attention.js";

/**
 * How many queries one attention invocation should carry, measured.
 *
 * `attentionFlashKernelForShape` gives an invocation two queries once a shape
 * has 128 or more of them, so that one key and value load serves both. Its own
 * comment records where that came from: "measured on GB10 at 1.17x-1.42x for
 * 128 to 1024 queries". On an M4 Pro it is the wrong way round — two queries
 * measured 2.2x slower than one at 708 queries and four measured 10.9x slower,
 * and a 1,416-residue complex went from 698 seconds to 400 by carrying one.
 *
 * Which is not a reason to write 1 down instead of 2. It is a reason to stop
 * writing it down: the ratio is a property of a driver's register budget, the
 * same kind of fact as which flash kernel to use or which arithmetic the
 * projections want, and this codebase already measures both of those. So this
 * measures it too, once per device, and caches the winner.
 *
 * It matters more than anything else here. Triangle attention is 57% of an
 * Evoformer block at 708 residues and grows with length, which makes this the
 * largest single cost in a long prediction.
 */

/** Rows and keys enough to be throughput-bound, small enough to stay cheap. */
const PROBE_BATCH = 4;
const PROBE_QUERIES = 512;
// The model's attentions carry four to eight heads, and the dispatch grid is
// queries by batch by head: at one head the probe launched 32 workgroups,
// far too few to occupy any GPU, so it ranked kernels on launch overhead.
const PROBE_HEADS = 8;
const PROBE_REPEATS = 2;
const PROBE_BATCH_MILLISECONDS = 8;

/**
 * How far a candidate may depart from the single-precision kernel.
 *
 * Speed cannot be the only thing measured. The block's differential test
 * against official AlphaFold intermediates holds its MSA output to a mean
 * absolute error of 5e-5, and this kernel feeds that output, so an
 * arrangement contributing more than that cannot be admitted however fast it
 * is: reading the keys and values as half words contributes about 4e-4, which
 * is how a green suite turned red on a machine the packing was never measured
 * on. The same rule as the projection selection, which filters on accuracy
 * before it ranks on time.
 */
const PROBE_ERROR_TOLERANCE = 5e-5;

/**
 * How many queries an invocation carries. Measured, not written down: the
 * threshold this replaces was recorded on one GPU and cost Apple 40% of a long
 * prediction.
 *
 * Only the keys are offered as half words, not the values. Packing both
 * measured 1.29x on Apple but moved the Evoformer block's MSA output 4.06e-4
 * from the official AlphaFold intermediates, eight times what that block's
 * differential test allows. The values are the half that costs it: they are
 * averaged under weights summing to one and land in the output directly, so
 * rounding them shows up undamped, and packing them alone still reads 4.05e-4.
 * A key's error is normalised away by the softmax, and packing keys alone
 * keeps every reference test green while still halving what the hot loop reads
 * of them.
 *
 * The probe cannot be what decides this. On its synthetic inputs the packing
 * that breaks the model reads 6e-7, six hundred times smaller, so its accuracy
 * filter catches gross divergence only; the differential tests are the gate.
 */
const CANDIDATES: readonly { readonly slots: number; readonly keyValue: AttentionKeyValueStorage }[] = [
  { slots: 1, keyValue: "f32" }, { slots: 1, keyValue: "f16-key" },
  { slots: 2, keyValue: "f32" }, { slots: 2, keyValue: "f16-key" },
];

export interface AttentionShapeChoice {
  /** Queries one invocation carries, for shapes the rule would give two. */
  readonly slots: number;
  readonly keyValue: AttentionKeyValueStorage;
}

const calibrations = new WeakMap<GPUDevice, Map<number, Promise<AttentionShapeChoice | undefined>>>();

/**
 * Whether to report each candidate's time and error.
 *
 * Guarded, because this module runs in a browser as well as in a test process
 * and `process` does not exist there. Reading it unguarded threw a
 * ReferenceError that the catch below turned into "no measurement", which
 * silently reverted every device to the arrangement this file exists to
 * replace — and reverted it only in the browser, which is the only place the
 * model actually runs.
 */
function probeDebugRequested(): boolean {
  return typeof process !== "undefined" && process.env?.AFWEBGPU_PROBE_DEBUG === "1";
}

async function runCandidate(
  device: GPUDevice, headDim: number, candidate: AttentionShapeChoice,
  buffers: readonly GPUBuffer[], readback: GPUBuffer, outputBytes: number,
): Promise<{ readonly milliseconds: number; readonly output: Float32Array }> {
  const { slots } = candidate;
  const code = createAttentionRegisterFlashShader(headDim, slots, candidate.keyValue);
  const pipeline = await device.createComputePipelineAsync({
    label: `attention-queries.${slots}`, layout: "auto",
    compute: {
      module: device.createShaderModule({ label: `attention-queries.${slots}.wgsl`, code }),
      entryPoint: "main",
    },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const groups = Math.ceil(PROBE_QUERIES / (64 * slots));
  const batch = async (iterations: number): Promise<number> => {
    const encoder = device.createCommandEncoder({ label: `attention-queries.${slots}` });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      pass.dispatchWorkgroups(groups, PROBE_BATCH, PROBE_HEADS);
    }
    pass.end();
    const started = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - started) / iterations;
  };
  // A rough pass warms the pipeline and sizes the timed one against a clock
  // the browser clamps to about 0.1 ms.
  const rough = await batch(1);
  const iterations = Math.max(1, Math.min(64,
    Math.ceil(PROBE_BATCH_MILLISECONDS / Math.max(rough, 0.01))));
  let best = Number.POSITIVE_INFINITY;
  for (let repeat = 0; repeat < PROBE_REPEATS; repeat += 1) {
    best = Math.min(best, await batch(iterations));
  }
  // What it computed, so speed is not the only thing measured.
  const encoder = device.createCommandEncoder({ label: "attention-queries.readback" });
  encoder.copyBufferToBuffer(buffers[buffers.length - 1]!, 0, readback, 0, outputBytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const output = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  return { milliseconds: best, output };
}

/**
 * Measures the four arrangements and caches the fastest.
 *
 * Anything that throws returns nothing and leaves the shape rule and the
 * single-precision operands alone, since a device that cannot run the probe
 * can still run the model.
 */
export function calibrateAttentionShape(
  device: GPUDevice, headDim: number,
): Promise<AttentionShapeChoice | undefined> {
  let byHeadDim = calibrations.get(device);
  if (byHeadDim === undefined) {
    byHeadDim = new Map();
    calibrations.set(device, byHeadDim);
  }
  const cached = byHeadDim.get(headDim);
  if (cached !== undefined) return cached;
  const measurement = (async (): Promise<AttentionShapeChoice | undefined> => {
    const buffers: GPUBuffer[] = [];
    try {
      const elements = PROBE_BATCH * PROBE_QUERIES * PROBE_HEADS * headDim;
      const create = (size: number, usage: GPUBufferUsageFlags): GPUBuffer => {
        const buffer = device.createBuffer({ label: "attention-queries.probe", size, usage });
        buffers.push(buffer);
        return buffer;
      };
      const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
      // Keys and values in both widths, so the two can be compared; the query
      // and the gate stay single precision, being read once per invocation.
      const half = create(elements * 2, storage);
      const full = create(elements * 4, storage);
      const query = create(elements * 4, storage);
      const gate = create(elements * 4, storage);
      const mask = create(PROBE_BATCH * PROBE_QUERIES * 4, storage);
      const bias = create(PROBE_HEADS * PROBE_QUERIES * PROBE_QUERIES * 4, storage);
      const output = create(elements * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const readback = create(elements * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const parameters = create(80, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      // Real values, not zeros. A kernel reading an all-zero key buffer is not
      // the kernel the model runs: every logit is equal, and the memory it is
      // supposed to be waiting for compresses away. Zeros are how a probe
      // measures something other than the thing it is choosing between.
      let state = 0x9e3779b9;
      const noise = (count: number): Float32Array => Float32Array.from({ length: count }, () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 0x100000000 - 0.5;
      });
      const keyValues = noise(elements);
      device.queue.writeBuffer(full, 0, keyValues);
      device.queue.writeBuffer(half, 0, packHalfWords(keyValues));
      device.queue.writeBuffer(query, 0, noise(elements));
      device.queue.writeBuffer(gate, 0, noise(elements));
      device.queue.writeBuffer(bias, 0, noise(PROBE_HEADS * PROBE_QUERIES * PROBE_QUERIES));
      device.queue.writeBuffer(mask, 0, new Float32Array(PROBE_BATCH * PROBE_QUERIES).fill(1));
      // batch, queries, channels, heads, head_dim, transpose, has_pair_bias,
      // then the weight offsets, then batch_offset and batch_total.
      const fields = new Uint32Array(20);
      fields[0] = PROBE_BATCH; fields[1] = PROBE_QUERIES; fields[2] = PROBE_HEADS * headDim;
      fields[3] = PROBE_HEADS; fields[4] = headDim; fields[6] = 1; fields[17] = PROBE_BATCH;
      device.queue.writeBuffer(parameters, 0, fields);
      const measured: { candidate: AttentionShapeChoice; milliseconds: number; error: number }[] = [];
      let reference: Float32Array | undefined;
      for (const candidate of CANDIDATES) {
        // The keys and the values are packed independently, so each binding
        // takes the width its own kernel declares.
        const keyBuffer = candidate.keyValue === "f16" || candidate.keyValue === "f16-key" ? half : full;
        const valueBuffer = candidate.keyValue === "f16" || candidate.keyValue === "f16-value" ? half : full;
        const result = await runCandidate(device, headDim, candidate,
          [query, keyBuffer, valueBuffer, gate, mask, bias, parameters, output], readback, elements * 4);
        // The first candidate is single precision with one query, which is what
        // the model ran before any of this: every other arrangement has to
        // reproduce it, not merely beat it.
        reference ??= result.output;
        let total = 0;
        for (let index = 0; index < result.output.length; index += 1) {
          total += Math.abs(result.output[index]! - reference[index]!);
        }
        measured.push({
          candidate, milliseconds: result.milliseconds, error: total / Math.max(1, result.output.length),
        });
      }
      if (probeDebugRequested()) {
        for (const entry of measured) {
          console.error(`probe ${entry.candidate.slots}q ${entry.candidate.keyValue}`
            + ` ${entry.milliseconds.toFixed(4)} ms error ${entry.error.toExponential(2)}`);
        }
      }
      const usable = measured.filter((entry) => entry.error <= PROBE_ERROR_TOLERANCE);
      if (usable.length === 0) return CANDIDATES[0];
      return usable.sort((left, right) => left.milliseconds - right.milliseconds)[0]!.candidate;
    } finally {
      for (const buffer of buffers) buffer.destroy();
    }
  })().catch((error: unknown) => {
    // The fallback is safe — the shape rule stands and the model runs — but it
    // is not free, and a probe that fails silently is indistinguishable from
    // one that ran. Say so once per device.
    console.warn("attention shape probe failed; keeping the shape rule", error);
    return undefined;
  });
  byHeadDim.set(headDim, measurement);
  return measurement;
}
