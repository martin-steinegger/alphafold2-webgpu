import { GpuBufferAllocator, type AllocatedGpuBuffer, type AllocationSnapshot } from "../runtime/allocator.js";
import { pipelineCacheForDevice, type ComputePipelineCache } from "../runtime/pipeline-cache.js";
import {
  MULTIMER_MAX_RELATIVE_CHAIN, MULTIMER_MAX_RELATIVE_INDEX, MULTIMER_RELATIVE_CHANNELS,
} from "../input/multimer-features.js";

export interface MultimerRelativePositionInput {
  readonly residueIndex: Float32Array;
  readonly asymId: Float32Array;
  readonly entityId: Float32Array;
  readonly symId: Float32Array;
  readonly length: number;
  readonly pairChannels: number;
  readonly weight: Float32Array;
  readonly bias: Float32Array;
}

export interface MultimerRelativePositionResult {
  readonly output: Float32Array;
  readonly elapsedMilliseconds: number;
  readonly memory: AllocationSnapshot;
}

function validate(input: MultimerRelativePositionInput): void {
  if (![input.length, input.pairChannels].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("multimer relative dimensions must be positive safe integers");
  }
  for (const [name, value] of [
    ["residueIndex", input.residueIndex], ["asymId", input.asymId],
    ["entityId", input.entityId], ["symId", input.symId],
  ] as const) {
    if (value.length !== input.length || value.some((item) => !Number.isSafeInteger(item) || item < 0)) {
      throw new RangeError(`${name} must contain one non-negative integer per residue`);
    }
  }
  if (input.weight.length !== MULTIMER_RELATIVE_CHANNELS * input.pairChannels
    || input.bias.length !== input.pairChannels) {
    throw new RangeError("multimer relative projection weights have an invalid shape");
  }
}

const SHADER = `
struct Parameters { length: u32, channels: u32 };
const GRID_WIDTH: u32 = 32768u;
const MAX_RELATIVE: i32 = ${MULTIMER_MAX_RELATIVE_INDEX};
const MAX_CHAIN: i32 = ${MULTIMER_MAX_RELATIVE_CHAIN};
@group(0) @binding(0) var<storage, read> residue_index: array<f32>;
@group(0) @binding(1) var<storage, read> asym_id: array<f32>;
@group(0) @binding(2) var<storage, read> entity_id: array<f32>;
@group(0) @binding(3) var<storage, read> sym_id: array<f32>;
@group(0) @binding(4) var<storage, read> weights: array<f32>;
@group(0) @binding(5) var<uniform> p: Parameters;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.length * p.channels) { return; }
  let channel = index % p.channels;
  let pair = index / p.channels;
  let i = pair / p.length;
  let j = pair % p.length;
  let same_chain = u32(asym_id[i]) == u32(asym_id[j]);
  var relative = 2u * u32(MAX_RELATIVE) + 1u;
  if (same_chain) {
    relative = u32(clamp(i32(residue_index[i]) - i32(residue_index[j]) + MAX_RELATIVE,
      0, 2 * MAX_RELATIVE));
  }
  let same_entity = u32(entity_id[i]) == u32(entity_id[j]);
  var relative_chain = 2u * u32(MAX_CHAIN) + 1u;
  if (same_entity) {
    relative_chain = u32(clamp(i32(sym_id[i]) - i32(sym_id[j]) + MAX_CHAIN, 0, 2 * MAX_CHAIN));
  }
  var value = weights[${MULTIMER_RELATIVE_CHANNELS}u * p.channels + channel];
  value += weights[relative * p.channels + channel];
  if (same_entity) { value += weights[66u * p.channels + channel]; }
  value += weights[(67u + relative_chain) * p.channels + channel];
  output[index] = value;
}`;

/** GPU projection of the official multimer-v3 chain-relative encoding without materializing [L,L,73]. */
export class MultimerRelativePositionGpu {
  readonly device: GPUDevice;
  readonly allocator: GpuBufferAllocator;
  readonly pipelines: ComputePipelineCache;
  constructor(device: GPUDevice) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  async run(input: MultimerRelativePositionInput): Promise<MultimerRelativePositionResult> {
    validate(input);
    const allocations: AllocatedGpuBuffer[] = [];
    const keep = (value: AllocatedGpuBuffer): AllocatedGpuBuffer => { allocations.push(value); return value; };
    try {
      const pipeline = await this.pipelines.get("multimer:relative-position", SHADER);
      const upload = (label: string, value: ArrayBufferView, usage = GPUBufferUsage.STORAGE) =>
        keep(this.allocator.upload(label, value, usage));
      const residueIndex = upload("multimer-relative.residue-index", input.residueIndex);
      const asymId = upload("multimer-relative.asym-id", input.asymId);
      const entityId = upload("multimer-relative.entity-id", input.entityId);
      const symId = upload("multimer-relative.sym-id", input.symId);
      const packed = new Float32Array(input.weight.length + input.bias.length);
      packed.set(input.weight); packed.set(input.bias, input.weight.length);
      const weights = upload("multimer-relative.weights", packed);
      const params = upload("multimer-relative.params", new Uint32Array([
        input.length, input.pairChannels, 0, 0,
      ]), GPUBufferUsage.UNIFORM);
      const elements = input.length * input.length * input.pairChannels;
      const output = keep(this.allocator.allocate(
        "multimer-relative.output", elements * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      ));
      const readback = keep(this.allocator.allocate(
        "multimer-relative.readback", elements * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      ));
      const encoder = this.device.createCommandEncoder({ label: "multimer-relative-position" });
      this.device.pushErrorScope("validation");
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [residueIndex, asymId, entityId, symId, weights, params, output].map(
          (buffer, binding) => ({ binding, resource: { buffer: buffer.buffer } }),
        ),
      }));
      const groups = Math.ceil(elements / 64);
      pass.dispatchWorkgroups(Math.min(groups, 32_768), Math.ceil(groups / 32_768));
      pass.end();
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, elements * 4);
      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU multimer relative projection failed: ${error.message}`);
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      return { output: result, elapsedMilliseconds: performance.now() - start, memory: this.allocator.snapshot() };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index]!.release();
    }
  }
}
