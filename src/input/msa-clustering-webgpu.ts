/**
 * Assigning every extra alignment row to its nearest cluster centre, on the GPU.
 *
 * It is almost all of featurisation. Each extra row is compared with every
 * centre at every residue, which at 825 residues, 508 centres and 1024 extras
 * is 429 million comparisons a recycle -- measured at 3.2 s against 0.12 s for
 * everything else featurisation does. The work is a reduction over residues and
 * an argmax over centres, so it belongs on the device.
 *
 * The port that runs this in JAX keeps it on the CPU to avoid paying a compile,
 * which is a trade we do not have to make: a kernel costs us one compile, once.
 */
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

/** Codes above this are gaps or masks, which never agree. See the CPU loop. */
const HIGHEST_RESIDUE_CODE = 20;
const LANES = 256;

const SHADER = `
@group(0) @binding(0) var<storage, read> centres: array<u32>;
@group(0) @binding(1) var<storage, read> extras: array<u32>;
@group(0) @binding(2) var<storage, read_write> assignment: array<u32>;
struct P { centres: u32, extras: u32, length: u32, pad: u32 };
@group(0) @binding(3) var<uniform> p: P;

// One score and one centre a lane, joined at the end. A tie must go to the
// LOWEST centre, which is what the host loop's strictly-greater test does, so
// the join compares scores first and indices second.
var<workgroup> best_score: array<u32, ${LANES}>;
var<workgroup> best_centre: array<u32, ${LANES}>;

@compute @workgroup_size(${LANES}, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let extra = group.x;
  let lane = local.x;
  var score = 0u;
  var centre = 0xffffffffu;
  if (extra < p.extras) {
    let extra_base = extra * p.length;
    for (var candidate = lane; candidate < p.centres; candidate += ${LANES}u) {
      let centre_base = candidate * p.length;
      var agree = 0u;
      for (var residue = 0u; residue < p.length; residue += 1u) {
        let code = centres[centre_base + residue];
        if (code <= ${HIGHEST_RESIDUE_CODE}u && code == extras[extra_base + residue]) {
          agree += 1u;
        }
      }
      // Strictly greater, so the first candidate a lane sees at a given score
      // keeps it, and the lane walks candidates in increasing order.
      if (centre == 0xffffffffu || agree > score) { score = agree; centre = candidate; }
    }
  }
  best_score[lane] = score;
  best_centre[lane] = centre;
  workgroupBarrier();
  for (var stride = ${LANES / 2}u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      let other = lane + stride;
      let take = best_centre[lane] == 0xffffffffu
        || (best_centre[other] != 0xffffffffu
          && (best_score[other] > best_score[lane]
            || (best_score[other] == best_score[lane]
              && best_centre[other] < best_centre[lane])));
      if (take) {
        best_score[lane] = best_score[other];
        best_centre[lane] = best_centre[other];
      }
    }
    workgroupBarrier();
  }
  if (lane == 0u && extra < p.extras) { assignment[extra] = best_centre[0]; }
}`;

/**
 * The nearest centre for every extra row, matching the host loop exactly.
 *
 * `centreCodes` and `extraCodes` are one code a residue, row-major. The result
 * is one centre index an extra row.
 */
export async function assignNearestCentres(
  device: GPUDevice,
  centreCodes: Uint8Array, centres: number,
  extraCodes: Uint8Array, extras: number,
  length: number,
): Promise<Uint32Array> {
  if (centreCodes.length !== centres * length || extraCodes.length !== extras * length) {
    throw new RangeError("centre and extra codes must be one code a residue, row-major");
  }
  const pipeline = await pipelineCacheForDevice(device).get("msa:nearest-centre", SHADER);
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  // A code a word: the codes are bytes, but a storage buffer indexes words and
  // unpacking in the shader would cost more than the four times the bytes,
  // which is 3.4 MB at the largest shape this runs.
  const widen = (codes: Uint8Array): Uint32Array => Uint32Array.from(codes);
  const centreBuffer = device.createBuffer({ size: centres * length * 4, usage });
  const extraBuffer = device.createBuffer({ size: extras * length * 4, usage });
  const out = device.createBuffer({
    size: Math.max(4, extras * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({
    size: Math.max(4, extras * 4),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const parameters = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  try {
    device.queue.writeBuffer(centreBuffer, 0, widen(centreCodes));
    device.queue.writeBuffer(extraBuffer, 0, widen(extraCodes));
    device.queue.writeBuffer(parameters, 0, new Uint32Array([centres, extras, length, 0]));
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [centreBuffer, extraBuffer, out, parameters].map((buffer, binding) => ({
        binding, resource: { buffer },
      })),
    });
    const encoder = device.createCommandEncoder({ label: "msa.nearest-centre" });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.max(1, extras), 1, 1);
    pass.end();
    encoder.copyBufferToBuffer(out, 0, readback, 0, Math.max(4, extras * 4));
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const assignments = new Uint32Array(readback.getMappedRange().slice(0, extras * 4));
    readback.unmap();
    return assignments;
  } finally {
    for (const buffer of [centreBuffer, extraBuffer, out, readback, parameters]) buffer.destroy();
  }
}
