/**
 * Assigning every extra alignment row to its nearest cluster centre, on the GPU.
 *
 * It is almost all of featurisation. Each extra row is compared with every
 * centre at every residue, which at 825 residues, 508 centres and 1024 extras
 * is 429 million comparisons a recycle -- measured at 3.2 s against 0.12 s for
 * everything else featurisation does. The work is a reduction over residues and
 * an argmax over centres, so it belongs on the device.
 *
 * The two callers want different answers from the same search. The monomer
 * keeps one centre, the lowest index at the best agreement. The multimer keeps
 * every centre tied at it and splits the row's weight between them. So the
 * kernel returns the whole tie set as a bitmask and each caller reads what it
 * needs: the monomer takes the lowest set bit.
 *
 * The port that runs this in JAX keeps it on the CPU to avoid paying a compile,
 * which is a trade we do not have to make: a kernel costs us one compile, once.
 */
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

/** Codes above this are gaps or masks, which never agree. See the CPU loops. */
const HIGHEST_RESIDUE_CODE = 20;
const LANES = 256;

const SHADER = `
@group(0) @binding(0) var<storage, read> centres: array<u32>;
@group(0) @binding(1) var<storage, read> extras: array<u32>;
@group(0) @binding(2) var<storage, read> valid: array<u32>;
@group(0) @binding(3) var<storage, read_write> sets: array<atomic<u32>>;
// The shape is supplied at pipeline creation, not read from a uniform: these
// are the bounds of the two loops that do all the work, and a uniform load in
// an inner loop measured 4.7x an override on the triangle kernels. The source
// does not change with them, so a new alignment costs a pipeline and no
// compile. See clusteringOverrides, and triangleOverrides for the same idea.
override CENTRES: u32 = 1u;
override LENGTH: u32 = 1u;
override WORDS: u32 = 1u;
// Only a bound on the dispatch, and it moves between recycles when rows are
// masked, so it stays a uniform rather than costing a pipeline each time.
struct P { extras: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(4) var<uniform> p: P;

// One score a lane, joined by max. A score is stored one above the agreement so
// that zero means "this lane saw no valid centre" and the join needs no
// sentinel of its own.
var<workgroup> best: array<u32, ${LANES}>;

fn agreement(centre: u32, extra: u32) -> u32 {
  let centre_base = centre * LENGTH;
  let extra_base = extra * LENGTH;
  var agree = 0u;
  for (var residue = 0u; residue < LENGTH; residue += 1u) {
    let code = centres[centre_base + residue];
    if (code <= ${HIGHEST_RESIDUE_CODE}u && code == extras[extra_base + residue]) {
      agree += 1u;
    }
  }
  return agree;
}

@compute @workgroup_size(${LANES}, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let extra = group.x;
  let lane = local.x;
  var mine = 0u;
  if (extra < p.extras) {
    for (var candidate = lane; candidate < CENTRES; candidate += ${LANES}u) {
      if (valid[candidate] == 0u) { continue; }
      mine = max(mine, agreement(candidate, extra) + 1u);
    }
  }
  best[lane] = mine;
  workgroupBarrier();
  for (var stride = ${LANES / 2}u; stride > 0u; stride /= 2u) {
    if (lane < stride) { best[lane] = max(best[lane], best[lane + stride]); }
    workgroupBarrier();
  }
  // Only a lane holding a tied centre can match the best, so in the usual case
  // of no tie exactly one lane walks its candidates a second time.
  let winner = best[0];
  if (extra >= p.extras || winner == 0u || mine != winner) { return; }
  for (var candidate = lane; candidate < CENTRES; candidate += ${LANES}u) {
    if (valid[candidate] == 0u) { continue; }
    if (agreement(candidate, extra) + 1u == winner) {
      atomicOr(&sets[extra * WORDS + candidate / 32u], 1u << (candidate % 32u));
    }
  }
}`;

/** Words a row in the returned bitmask. */
export const tieSetWords = (centres: number): number => Math.max(1, Math.ceil(centres / 32));

/** The shape constants baked into the kernel, and the key that must carry them. */
export function clusteringOverrides(centres: number, length: number): Record<string, number> {
  return { CENTRES: centres, LENGTH: length, WORDS: tieSetWords(centres) };
}

/**
 * For every extra row, the set of centres tied at the best agreement.
 *
 * centreCodes and extraCodes are one code a residue, row-major. The result
 * is tieSetWords(centres) words an extra row, bit c set when centre c
 * ties. A row with no valid centre comes back empty. centreValid excludes
 * centres from the search without moving the indices the caller writes to;
 * omitting it lets every centre take part.
 */
export async function nearestCentreSets(
  device: GPUDevice,
  centreCodes: Uint8Array, centres: number,
  extraCodes: Uint8Array, extras: number,
  length: number,
  centreValid?: Uint8Array,
): Promise<Uint32Array> {
  if (centreCodes.length !== centres * length || extraCodes.length !== extras * length) {
    throw new RangeError("centre and extra codes must be one code a residue, row-major");
  }
  if (centreValid !== undefined && centreValid.length !== centres) {
    throw new RangeError("centre validity must be one flag a centre");
  }
  const words = tieSetWords(centres);
  const overrides = clusteringOverrides(centres, length);
  const pipeline = await pipelineCacheForDevice(device).get(
    `msa:nearest-centre:${centres}x${length}`, SHADER, "main", overrides);
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  // A code a word: the codes are bytes, but a storage buffer indexes words and
  // unpacking in the shader would cost more than the four times the bytes,
  // which is 3.4 MB at the largest shape this runs.
  const widen = (codes: Uint8Array): Uint32Array => Uint32Array.from(codes);
  const bytes = Math.max(4, extras * words * 4);
  const centreBuffer = device.createBuffer({ size: Math.max(4, centres * length * 4), usage });
  const extraBuffer = device.createBuffer({ size: Math.max(4, extras * length * 4), usage });
  const validBuffer = device.createBuffer({ size: Math.max(4, centres * 4), usage });
  const out = device.createBuffer({
    size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({
    size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const parameters = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  try {
    device.queue.writeBuffer(centreBuffer, 0, widen(centreCodes));
    device.queue.writeBuffer(extraBuffer, 0, widen(extraCodes));
    device.queue.writeBuffer(validBuffer, 0, centreValid === undefined
      ? new Uint32Array(Math.max(1, centres)).fill(1)
      : Uint32Array.from(centreValid, (flag) => (flag === 0 ? 0 : 1)));
    device.queue.writeBuffer(parameters, 0, new Uint32Array([extras, 0, 0, 0]));
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [centreBuffer, extraBuffer, validBuffer, out, parameters]
        .map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const encoder = device.createCommandEncoder({ label: "msa.nearest-centre" });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.max(1, extras), 1, 1);
    pass.end();
    encoder.copyBufferToBuffer(out, 0, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const sets = new Uint32Array(readback.getMappedRange().slice(0, extras * words * 4));
    readback.unmap();
    return sets;
  } finally {
    for (const buffer of [centreBuffer, extraBuffer, validBuffer, out, readback, parameters]) {
      buffer.destroy();
    }
  }
}

/**
 * The nearest centre for every extra row, breaking a tie towards the lowest
 * index, which is what a host loop keeping the first centre at a given score
 * does. A row with no centre comes back as 0xffffffff.
 */
export async function assignNearestCentres(
  device: GPUDevice,
  centreCodes: Uint8Array, centres: number,
  extraCodes: Uint8Array, extras: number,
  length: number,
): Promise<Uint32Array> {
  const words = tieSetWords(centres);
  const sets = await nearestCentreSets(
    device, centreCodes, centres, extraCodes, extras, length);
  const assignments = new Uint32Array(extras).fill(0xffffffff);
  for (let extra = 0; extra < extras; extra += 1) {
    for (let word = 0; word < words; word += 1) {
      const bits = sets[extra * words + word]!;
      if (bits === 0) continue;
      assignments[extra] = word * 32 + 31 - Math.clz32(bits & -bits);
      break;
    }
  }
  return assignments;
}
