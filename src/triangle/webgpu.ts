import { GpuBufferAllocator, type AllocatedGpuBuffer, type AllocationSnapshot } from "../runtime/allocator.js";
import { float32ToFloat16Array } from "../runtime/float16.js";
import { pipelineCacheForDevice, type ComputePipelineCache } from "../runtime/pipeline-cache.js";
import { createTriangleShaders, type TriangleDirection } from "./shaders.js";
import type { Precision, TriangleMultiplicationInput } from "./types.js";
import { validateTriangleInput } from "./types.js";
import { packWeights } from "./weights.js";
import { gemmGrid } from "../runtime/gemm.js";

export interface TriangleGpuOptions {
  readonly precision?: Precision;
}

export interface TriangleGpuResult {
  readonly output: Float32Array;
  readonly elapsedMilliseconds: number;
  readonly memory: AllocationSnapshot;
}

const ceilDivide = (value: number, divisor: number): number => Math.ceil(value / divisor);
const LINEAR_GRID_WIDTH = 32_768;

function makeBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  buffers: readonly GPUBuffer[],
  label: string,
): GPUBindGroup {
  return device.createBindGroup({
    label,
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
}

class TriangleMultiplicationGpu {
  readonly device: GPUDevice;
  readonly allocator: GpuBufferAllocator;
  readonly pipelines: ComputePipelineCache;
  readonly direction: TriangleDirection;

  constructor(device: GPUDevice, direction: TriangleDirection) {
    this.device = device;
    this.direction = direction;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  async run(input: TriangleMultiplicationInput, options: TriangleGpuOptions = {}): Promise<TriangleGpuResult> {
    validateTriangleInput(input);
    const precision = options.precision ?? "f32";
    if (precision === "f16" && !this.device.features.has("shader-f16")) {
      throw new Error("f16 execution requires the WebGPU shader-f16 feature");
    }

    const { length, cZ, cHidden } = input.shape;
    const pairCount = length * length;
    const packedWeights = packWeights(input.weights, precision);
    const shaders = createTriangleShaders(
      input.shape, precision, packedWeights.offsets, input.epsilon ?? 1e-5, this.direction,
    );
    const pipelineKey = `${this.direction}:${precision}:${length}:${cZ}:${cHidden}:${input.epsilon ?? 1e-5}`;
    const [normalizeInput, projectAB, contract, normalizeHidden, projectOutput] = await Promise.all([
      this.pipelines.get(`${pipelineKey}:normalize-input`, shaders.normalizeInput),
      this.pipelines.get(`${pipelineKey}:project-ab`, shaders.projectAB),
      this.pipelines.get(`${pipelineKey}:contract`, shaders.contract),
      this.pipelines.get(`${pipelineKey}:normalize-hidden`, shaders.normalizeHidden),
      this.pipelines.get(`${pipelineKey}:project-output`, shaders.projectOutput),
    ]);

    const zData = precision === "f16" ? float32ToFloat16Array(input.z) : input.z;
    const storage = GPUBufferUsage.STORAGE;
    const allocations: AllocatedGpuBuffer[] = [];
    const keep = (allocation: AllocatedGpuBuffer): AllocatedGpuBuffer => {
      allocations.push(allocation);
      return allocation;
    };

    try {
      const z = keep(this.allocator.upload("triangle.z", zData, storage));
      const mask = keep(this.allocator.upload("triangle.mask", input.mask, storage));
      const weights = keep(this.allocator.upload("triangle.weights", packedWeights.data, storage));
      const zNormalized = keep(this.allocator.allocate(
        "triangle.z-normalized", pairCount * cZ * 4, storage,
      ));
      const a = keep(this.allocator.allocate("triangle.a", pairCount * cHidden * 4, storage));
      const b = keep(this.allocator.allocate("triangle.b", pairCount * cHidden * 4, storage));
      const contracted = keep(this.allocator.allocate("triangle.contracted", pairCount * cHidden * 4, storage));
      const xNormalized = keep(this.allocator.allocate(
        "triangle.x-normalized", pairCount * cHidden * 4, storage,
      ));
      const output = keep(this.allocator.allocate(
        "triangle.output", pairCount * cZ * 4, storage | GPUBufferUsage.COPY_SRC,
      ));
      const readback = keep(this.allocator.allocate(
        "triangle.readback", pairCount * cZ * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      ));

      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({ label: `triangle-${this.direction}` });
      const runPass = (
        label: string,
        pipeline: GPUComputePipeline,
        buffers: readonly GPUBuffer[],
        x: number,
        y = 1,
        zGroups = 1,
      ): void => {
        const pass = encoder.beginComputePass({ label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, makeBindGroup(this.device, pipeline, buffers, `${label}.bindings`));
        pass.dispatchWorkgroups(x, y, zGroups);
        pass.end();
      };
      const linearDispatch = (elements: number): readonly [number, number] => {
        const groups = ceilDivide(elements, 64);
        return [Math.min(groups, LINEAR_GRID_WIDTH), ceilDivide(groups, LINEAR_GRID_WIDTH)];
      };

      runPass("normalize-input", normalizeInput, [z.buffer, weights.buffer, zNormalized.buffer], ceilDivide(pairCount, 64));
      runPass("project-ab", projectAB,
        [zNormalized.buffer, mask.buffer, weights.buffer, a.buffer, b.buffer],
        ceilDivide(cHidden, 16), ceilDivide(pairCount, 16));
      const contractGrid = gemmGrid(length, length);
      runPass("contract", contract, [a.buffer, b.buffer, contracted.buffer],
        contractGrid[0], contractGrid[1], cHidden);
      runPass("normalize-hidden", normalizeHidden,
        [contracted.buffer, weights.buffer, xNormalized.buffer], ceilDivide(pairCount, 64));
      runPass("project-output", projectOutput,
        [zNormalized.buffer, xNormalized.buffer, weights.buffer, output.buffer],
        ceilDivide(cZ, 16), ceilDivide(pairCount, 16));
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, pairCount * cZ * 4);

      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const validationError = await this.device.popErrorScope();
      if (validationError !== null) throw new Error(`WebGPU validation failed: ${validationError.message}`);
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      const elapsedMilliseconds = performance.now() - start;
      return { output: result, elapsedMilliseconds, memory: this.allocator.snapshot() };
    } finally {
      for (let i = allocations.length - 1; i >= 0; i -= 1) allocations[i]!.release();
    }
  }
}

export class TriangleMultiplicationOutgoingGpu extends TriangleMultiplicationGpu {
  constructor(device: GPUDevice) {
    super(device, "outgoing");
  }
}

export class TriangleMultiplicationIncomingGpu extends TriangleMultiplicationGpu {
  constructor(device: GPUDevice) {
    super(device, "incoming");
  }
}
