import { createAttentionRegisterFlashShader } from "../evoformer/attention.js";

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
const PROBE_HEADS = 1;
const PROBE_REPEATS = 2;
const PROBE_BATCH_MILLISECONDS = 8;

/**
 * Every combination worth trying: how many queries an invocation carries, and
 * whether it reads the keys and values as half words.
 *
 * Both are measured rather than assumed, and for the same reason. Packing the
 * operands halves what the key loop reads and costs an unpack for each; on a
 * device that is waiting for memory that is 1.29x and on one that is waiting
 * for issue slots it may be nothing. Writing either down is how the threshold
 * this replaces came to cost Apple 40% of a long prediction.
 */
const CANDIDATES: readonly { readonly slots: number; readonly keyValue: "f32" | "f16" }[] = [
  { slots: 1, keyValue: "f32" }, { slots: 1, keyValue: "f16" },
  { slots: 2, keyValue: "f32" }, { slots: 2, keyValue: "f16" },
];

export interface AttentionShapeChoice {
  /** Queries one invocation carries, for shapes the rule would give two. */
  readonly slots: number;
  readonly keyValue: "f32" | "f16";
}

const calibrations = new WeakMap<GPUDevice, Map<number, Promise<AttentionShapeChoice | undefined>>>();

async function timeCandidate(
  device: GPUDevice, headDim: number, candidate: AttentionShapeChoice,
  buffers: readonly GPUBuffer[],
): Promise<number> {
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
  return best;
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
      const output = create(elements * 4, GPUBufferUsage.STORAGE);
      const parameters = create(80, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(mask, 0, new Float32Array(PROBE_BATCH * PROBE_QUERIES).fill(1));
      // batch, queries, channels, heads, head_dim, transpose, has_pair_bias,
      // then the weight offsets, then batch_offset and batch_total.
      const fields = new Uint32Array(20);
      fields[0] = PROBE_BATCH; fields[1] = PROBE_QUERIES; fields[2] = PROBE_HEADS * headDim;
      fields[3] = PROBE_HEADS; fields[4] = headDim; fields[6] = 1; fields[17] = PROBE_BATCH;
      device.queue.writeBuffer(parameters, 0, fields);
      let winner = CANDIDATES[0]!;
      let best = Number.POSITIVE_INFINITY;
      for (const candidate of CANDIDATES) {
        const kv = candidate.keyValue === "f16" ? half : full;
        const milliseconds = await timeCandidate(device, headDim, candidate,
          [query, kv, kv, gate, mask, bias, parameters, output]);
        if (milliseconds < best) { best = milliseconds; winner = candidate; }
      }
      return winner;
    } finally {
      for (const buffer of buffers) buffer.destroy();
    }
  })().catch(() => undefined);
  byHeadDim.set(headDim, measurement);
  return measurement;
}
