/**
 * AlphaFold-Multimer's masked-MSA sampling on the GPU.
 *
 * Every centre position draws once to decide whether it is masked, and a
 * masked one then takes an argmax over 23 categorical scores, each carrying
 * its own Gumbel noise. At 252 centres and 825 residues that is 208,000
 * decisions and 724,000 Gumbel draws a recycle, measured at 204 ms and the
 * largest host loop left in featurisation once the clustering moved.
 *
 * It is embarrassingly parallel -- a position depends on nothing but its own
 * indices -- and the only reason it stayed on the host is the random number
 * generator. JAX's is Threefry2x32, which is integer arithmetic and ports
 * directly. AlphaFold's padding_consistent_rng folds each index into the key in
 * turn (see grid_keys in alphafold/model/utils.py), so a position's key depends
 * on its coordinates and not on how large the array around it is, which is what
 * lets a thread derive its own without any sequence.
 *
 * The scores are f32 here and f64 on the host. JAX computes them in f32, so
 * this is the closer of the two to the reference, but an argmax between two
 * near-equal codes can still land differently: the official comparison in
 * test/multimer-process-features.test.ts is what settles that, not this note.
 */
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import type { JaxKey } from "./jax-prng.js";

const LANES = 64;
const CODES = 23;
/** Positions are masked at this rate, and the mask code is 22. */
const MASK_RATE = 0.15;

const SHADER = `
@group(0) @binding(0) var<storage, read> codes: array<u32>;
@group(0) @binding(1) var<storage, read> profile: array<f32>;
@group(0) @binding(2) var<storage, read> row_valid: array<u32>;
@group(0) @binding(3) var<storage, read_write> masked: array<u32>;
struct P {
  position_key0: u32, position_key1: u32, gumbel_key0: u32, gumbel_key1: u32,
};
@group(0) @binding(4) var<uniform> p: P;

override CENTRES: u32 = 1u;
override LENGTH: u32 = 1u;

const PARITY: u32 = 0x1bd11bdau;
const ROTATIONS = array<u32, 8>(13u, 15u, 26u, 6u, 17u, 29u, 16u, 24u);

// Threefry2x32, the generator JAX uses, in the five-injection form its
// implementation fixes. Everything here is wrapping u32 arithmetic, which WGSL
// gives directly, so this is a transcription rather than a reimplementation.
fn threefry(key0: u32, key1: u32, count0: u32, count1: u32) -> vec2<u32> {
  let keys = array<u32, 3>(key0, key1, key0 ^ key1 ^ PARITY);
  var first = count0 + keys[0];
  var second = count1 + keys[1];
  for (var injection = 1u; injection <= 5u; injection += 1u) {
    let base = ((injection - 1u) % 2u) * 4u;
    for (var step = 0u; step < 4u; step += 1u) {
      let rotation = ROTATIONS[base + step];
      first = first + second;
      second = ((second << rotation) | (second >> (32u - rotation))) ^ first;
    }
    first = first + keys[injection % 3u];
    second = second + keys[(injection + 1u) % 3u] + injection;
  }
  return vec2<u32>(first, second);
}

/** fold_in, which is a hash of the key with one scalar index. */
fn fold_in(key: vec2<u32>, value: u32) -> vec2<u32> {
  return threefry(key.x, key.y, 0u, value);
}

/** The scalar float32 uniform AlphaFold's wrapper draws. */
fn uniform_of(key: vec2<u32>) -> f32 {
  let bits = threefry(key.x, key.y, 0u, 0u);
  return f32((bits.x ^ bits.y) >> 9u) / 8388608.0;
}

fn gumbel_of(key: vec2<u32>) -> f32 {
  let epsilon = 1e-6;
  return -log(-log(uniform_of(key) + epsilon) + epsilon);
}

@compute @workgroup_size(${LANES}, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let slot = id.x;
  if (slot >= CENTRES * LENGTH) { return; }
  let centre = slot / LENGTH;
  let residue = slot % LENGTH;
  let original = codes[slot];
  masked[slot] = original;
  if (row_valid[centre] == 0u) { return; }

  // The position's own key, folded from its coordinates, so no thread depends
  // on the order any other ran in.
  let position = fold_in(fold_in(vec2<u32>(p.position_key0, p.position_key1), centre), residue);
  if (uniform_of(position) >= ${MASK_RATE}) { return; }

  let gumbel_row = fold_in(fold_in(vec2<u32>(p.gumbel_key0, p.gumbel_key1), centre), residue);
  var best_code = 0u;
  var best_score = -1.0e38;
  for (var code = 0u; code < ${CODES}u; code += 1u) {
    var probability = 1e-6;
    if (code < 20u) { probability += 0.005; }
    if (code < 22u) { probability += 0.1 * profile[residue * 22u + code]; }
    if (code == original) { probability += 0.1; }
    if (code == 22u) { probability += 0.7; }
    let score = log(probability) + gumbel_of(fold_in(gumbel_row, code));
    if (score > best_score) { best_score = score; best_code = code; }
  }
  masked[slot] = best_code;
}`;

export interface MaskedMsaInput {
  /** Centre codes, one a residue, row-major. */
  readonly codes: Uint8Array;
  readonly centres: number;
  readonly length: number;
  /** The categorical profile over the whole raw MSA, 22 channels a residue. */
  readonly profile: Float32Array;
  /** Whether each centre row is real rather than block padding. */
  readonly rowValid: Uint8Array;
  readonly positionKey: JaxKey;
  readonly gumbelKey: JaxKey;
}

/** The shape constants baked into the kernel; see clusteringOverrides. */
export function maskingOverrides(centres: number, length: number): Record<string, number> {
  return { CENTRES: centres, LENGTH: length };
}

/** The centre codes after masking, one a residue, row-major. */
export async function maskCentreCodes(
  device: GPUDevice, input: MaskedMsaInput,
): Promise<Uint8Array> {
  const { centres, length } = input;
  if (input.codes.length !== centres * length) {
    throw new RangeError("centre codes must be one code a residue, row-major");
  }
  if (input.profile.length !== length * 22) {
    throw new RangeError("the profile must be 22 channels a residue");
  }
  if (input.rowValid.length !== centres) throw new RangeError("one validity flag a centre");

  const pipeline = await pipelineCacheForDevice(device).get(
    `msa:mask:${centres}x${length}`, SHADER, "main", maskingOverrides(centres, length));
  const read = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const slots = Math.max(1, centres * length);
  const buffers = {
    codes: device.createBuffer({ size: slots * 4, usage: read }),
    profile: device.createBuffer({ size: Math.max(4, length * 22 * 4), usage: read }),
    rowValid: device.createBuffer({ size: Math.max(4, centres * 4), usage: read }),
    masked: device.createBuffer({
      size: slots * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
    parameters: device.createBuffer({
      size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
  };
  const readback = device.createBuffer({
    size: slots * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    device.queue.writeBuffer(buffers.codes, 0, Uint32Array.from(input.codes));
    device.queue.writeBuffer(buffers.profile, 0, input.profile);
    device.queue.writeBuffer(buffers.rowValid, 0,
      Uint32Array.from(input.rowValid, (flag) => (flag === 0 ? 0 : 1)));
    device.queue.writeBuffer(buffers.parameters, 0, new Uint32Array([
      input.positionKey[0] >>> 0, input.positionKey[1] >>> 0,
      input.gumbelKey[0] >>> 0, input.gumbelKey[1] >>> 0,
    ]));
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [buffers.codes, buffers.profile, buffers.rowValid, buffers.masked,
        buffers.parameters].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const encoder = device.createCommandEncoder({ label: "msa.mask" });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(slots / LANES), 1, 1);
    pass.end();
    encoder.copyBufferToBuffer(buffers.masked, 0, readback, 0, slots * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(readback.getMappedRange().slice(0, centres * length * 4));
    readback.unmap();
    return Uint8Array.from(words);
  } finally {
    for (const buffer of Object.values(buffers)) buffer.destroy();
    readback.destroy();
  }
}
