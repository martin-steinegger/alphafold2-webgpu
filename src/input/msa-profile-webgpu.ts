/**
 * The cluster profile on the GPU: the other half of featurisation's clustering.
 *
 * Assigning the extra rows was moved to the device first, and doing only that
 * left this port 1.1x ahead of the CPU it was compared against instead of the
 * 6.6x the search alone wins. The rest was here. Accumulating each extra row
 * into its centre's profile is 845,000 iterations, which is nothing, but every
 * one of them writes into a 45 MB array at an address the assignment chose, so
 * it misses cache on nearly every access and measured 220 ms a recycle.
 *
 * AlphaFold does the same work as an einsum over a 23-way one-hot: 23x the
 * arithmetic, and far faster, because it is a dense matmul. This takes the
 * other route to the same place -- invert the assignment on the host, which is
 * a counting sort over the extras, and give each (centre, residue) thread the
 * short list of rows that landed on it. The arithmetic stays at 1x and the
 * writes become one coalesced store a thread.
 */
import { CLUSTERED_MSA_CHANNELS } from "./msa-features.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";

const LANES = 64;
/** The 20 amino acids, X, gap and the BERT mask. */
const CODES = 23;

const SHADER = `
@group(0) @binding(0) var<storage, read> centre_codes: array<u32>;
@group(0) @binding(1) var<storage, read> extra_codes: array<u32>;
@group(0) @binding(2) var<storage, read> centre_deletion: array<f32>;
@group(0) @binding(3) var<storage, read> extra_deletion: array<f32>;
// The assignment inverted: rows grouped by centre, with a start a centre.
@group(0) @binding(4) var<storage, read> bucket_start: array<u32>;
@group(0) @binding(5) var<storage, read> bucket_rows: array<u32>;
@group(0) @binding(6) var<storage, read_write> features: array<f32>;
struct P { centres: u32, length: u32, channels: u32, pad: u32 };
@group(0) @binding(7) var<uniform> p: P;

fn deletion_value(value: f32) -> f32 { return atan(value / 3.0) * 2.0 / ${Math.PI}; }

@compute @workgroup_size(${LANES}, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let slot = id.x;
  if (slot >= p.centres * p.length) { return; }
  let centre = slot / p.length;
  let residue = slot % p.length;

  var profile: array<f32, ${CODES}>;
  for (var code = 0u; code < ${CODES}u; code += 1u) { profile[code] = 0.0; }
  // The centre counts as one member of its own cluster, and the 1e-6 is the
  // host's, kept so the division matches it bit for bit.
  let centre_code = centre_codes[slot];
  profile[centre_code] = 1.0;
  let own_deletion = centre_deletion[slot];
  var deletion_sum = own_deletion;

  let start = bucket_start[centre];
  let end = bucket_start[centre + 1u];
  for (var index = start; index < end; index += 1u) {
    let row = bucket_rows[index];
    profile[extra_codes[row * p.length + residue]] += 1.0;
    deletion_sum += extra_deletion[row * p.length + residue];
  }
  let count = 1.0 + 1e-6 + f32(end - start);

  let output = slot * p.channels;
  features[output] = f32(centre_code);
  features[output + 1u] = min(own_deletion, 1.0);
  features[output + 2u] = deletion_value(own_deletion);
  for (var code = 0u; code < ${CODES}u; code += 1u) {
    features[output + 3u + code] = profile[code] / count;
  }
  features[output + 26u] = deletion_value(deletion_sum / count);
}`;

export interface ClusterProfileInput {
  /** Post-masking centre codes, one a residue, row-major. */
  readonly centreCodes: Uint8Array;
  readonly centres: number;
  /** Extra rows gathered densely, matching the indices `assignments` refers to. */
  readonly extraCodes: Uint8Array;
  readonly extras: number;
  readonly length: number;
  /** The centre each extra row was assigned to. */
  readonly assignments: Uint32Array;
  /** Deletion counts for the centre rows and the gathered extra rows. */
  readonly centreDeletion: Float32Array;
  readonly extraDeletion: Float32Array;
}

/**
 * The clustered MSA feature block, `centres * length * 27` floats.
 *
 * An extra row whose assignment is out of range is dropped, which is what the
 * host loop's "no centre" case did.
 */
interface ProfileBuffers {
  readonly key: string;
  readonly centreCodes: GPUBuffer; readonly extraCodes: GPUBuffer;
  readonly centreDeletion: GPUBuffer; readonly extraDeletion: GPUBuffer;
  readonly bucketStart: GPUBuffer; readonly bucketRows: GPUBuffer;
  readonly output: GPUBuffer; readonly parameters: GPUBuffer; readonly readback: GPUBuffer;
}

/**
 * The buffers are held between recycles rather than allocated a call. The
 * output alone is 45 MB, and churning a MAP_READ buffer that size three times
 * a fold made Dawn abort at teardown.
 */
const held = new WeakMap<GPUDevice, ProfileBuffers>();

function buffersFor(
  device: GPUDevice, centres: number, extras: number, length: number,
): ProfileBuffers {
  const key = `${centres}/${extras}/${length}`;
  const existing = held.get(device);
  if (existing?.key === key) return existing;
  if (existing !== undefined) {
    for (const buffer of Object.values(existing)) {
      if (typeof buffer !== "string") buffer.destroy();
    }
  }
  const read = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const slots = Math.max(1, centres * length);
  const rows = Math.max(1, extras * length);
  const outputBytes = slots * CLUSTERED_MSA_CHANNELS * 4;
  const made: ProfileBuffers = {
    key,
    centreCodes: device.createBuffer({ size: slots * 4, usage: read }),
    extraCodes: device.createBuffer({ size: rows * 4, usage: read }),
    centreDeletion: device.createBuffer({ size: slots * 4, usage: read }),
    extraDeletion: device.createBuffer({ size: rows * 4, usage: read }),
    bucketStart: device.createBuffer({ size: (centres + 1) * 4, usage: read }),
    bucketRows: device.createBuffer({ size: Math.max(4, extras * 4), usage: read }),
    output: device.createBuffer({
      size: outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
    parameters: device.createBuffer({
      size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    readback: device.createBuffer({
      size: outputBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
  };
  held.set(device, made);
  return made;
}

export async function clusterProfile(
  device: GPUDevice, input: ClusterProfileInput,
): Promise<Float32Array> {
  const { centreCodes, centres, extraCodes, extras, length, assignments } = input;
  if (centreCodes.length !== centres * length || extraCodes.length !== extras * length) {
    throw new RangeError("centre and extra codes must be one code a residue, row-major");
  }
  if (assignments.length !== extras) throw new RangeError("one assignment an extra row");
  if (input.centreDeletion.length !== centres * length
    || input.extraDeletion.length !== extras * length) {
    throw new RangeError("deletion counts must match the code arrays");
  }

  // Counting sort: the extras grouped by the centre they landed on, so a
  // thread reads a contiguous run rather than searching for its members.
  const bucketStart = new Uint32Array(centres + 1);
  for (let extra = 0; extra < extras; extra += 1) {
    const centre = assignments[extra]!;
    if (centre < centres) bucketStart[centre + 1] = bucketStart[centre + 1]! + 1;
  }
  for (let centre = 0; centre < centres; centre += 1) {
    bucketStart[centre + 1] = bucketStart[centre + 1]! + bucketStart[centre]!;
  }
  const cursor = bucketStart.slice(0, centres);
  const bucketRows = new Uint32Array(Math.max(1, extras));
  for (let extra = 0; extra < extras; extra += 1) {
    const centre = assignments[extra]!;
    if (centre < centres) { bucketRows[cursor[centre]!] = extra; cursor[centre] = cursor[centre]! + 1; }
  }

  const pipeline = await pipelineCacheForDevice(device).get("msa:cluster-profile", SHADER);
  const slots = centres * length;
  const outputBytes = slots * CLUSTERED_MSA_CHANNELS * 4;
  const buffers = buffersFor(device, centres, extras, length);
  const readback = buffers.readback;
  device.queue.writeBuffer(buffers.centreCodes, 0, Uint32Array.from(centreCodes));
  device.queue.writeBuffer(buffers.extraCodes, 0, Uint32Array.from(extraCodes));
  device.queue.writeBuffer(buffers.centreDeletion, 0, input.centreDeletion);
  device.queue.writeBuffer(buffers.extraDeletion, 0, input.extraDeletion);
  device.queue.writeBuffer(buffers.bucketStart, 0, bucketStart);
  device.queue.writeBuffer(buffers.bucketRows, 0, bucketRows);
  device.queue.writeBuffer(buffers.parameters, 0,
    new Uint32Array([centres, length, CLUSTERED_MSA_CHANNELS, 0]));
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [buffers.centreCodes, buffers.extraCodes, buffers.centreDeletion,
      buffers.extraDeletion, buffers.bucketStart, buffers.bucketRows, buffers.output,
      buffers.parameters].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const encoder = device.createCommandEncoder({ label: "msa.cluster-profile" });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(slots / LANES), 1, 1);
  pass.end();
  encoder.copyBufferToBuffer(buffers.output, 0, readback, 0, outputBytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const features = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  return features;
}
