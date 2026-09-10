import { GpuBufferAllocator, type AllocatedGpuBuffer, type AllocationSnapshot } from "../runtime/allocator.js";
import { pipelineCacheForDevice, type ComputePipelineCache } from "../runtime/pipeline-cache.js";
import { triangleOverrides, wholeProjectionStride, createTriangleShaders, type TriangleDirection, type TriangleWholeStorage } from "./shaders.js";
import type { Precision, TriangleMultiplicationInput } from "./types.js";
import { validateTriangleInput } from "./types.js";
import { packWeights } from "./weights.js";
import { gemmGrid } from "../runtime/gemm.js";

export interface TriangleGpuOptions {
  readonly precision?: Precision;
  /** Residues of the output axis per block; defaults to the whole length. */
  readonly blockRows?: number;
  /** Storage of the whole projection; f16 is inexact. */
  readonly wholeStorage?: TriangleWholeStorage;
}

type Binding = GPUBuffer | GPUBufferBinding;

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
  buffers: readonly Binding[],
  label: string,
): GPUBindGroup {
  return device.createBindGroup({
    label,
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({
      binding, resource: "buffer" in buffer && !(buffer instanceof GPUBuffer) ? buffer : { buffer: buffer as GPUBuffer },
    })),
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
    const blockRows = Math.max(1, Math.min(length, options.blockRows ?? length));
    const wholeStorage = options.wholeStorage ?? "f32";
    const blockPairs = blockRows * length;
    // Padded so a channel starts where a binding may start. This path binds
    // the projection whole and needs no groups, but the stride is an override
    // the shader shares with the block encoder, which does.
    const wholeStride = wholeProjectionStride(length);
    const packedWeights = packWeights(input.weights, precision);
    const shaders = createTriangleShaders(
      input.shape, precision, packedWeights.offsets, input.epsilon ?? 1e-5, this.direction, blockRows, wholeStorage,
    );
    const pipelineKey = `${this.direction}:${precision}:${length}:${cZ}:${cHidden}:${input.epsilon ?? 1e-5}`
      + `:${blockRows}:${wholeStorage}`;
    // The length and the strides are override constants, so a pipeline built
    // without them silently keeps their defaults of one and the kernel reads a
    // single residue. The block encoder passes these; this entry point did not.
    const overrides = triangleOverrides(input.shape, blockRows);
    const [inputStatistics, projectGate, projectBlockOperand, projectWholeOperand, contract, hiddenStatistics,
      projectOutput] = await Promise.all([
      this.pipelines.get(`${pipelineKey}:input-statistics`, shaders.inputStatistics, "main", overrides),
      this.pipelines.get(`${pipelineKey}:project-gate`, shaders.projectGate, "main", overrides),
      this.pipelines.get(`${pipelineKey}:project-block-operand`, shaders.projectBlockOperand, "main", overrides),
      this.pipelines.get(`${pipelineKey}:project-whole-operand`, shaders.projectWholeOperand, "main", overrides),
      this.pipelines.get(`${pipelineKey}:contract`, shaders.contract, "main", overrides),
      this.pipelines.get(`${pipelineKey}:hidden-statistics`, shaders.hiddenStatistics, "main", overrides),
      this.pipelines.get(`${pipelineKey}:project-output`, shaders.projectOutput, "main", overrides),
    ]);

    // The pair stays f32 whatever precision the weights are in. Packing both
    // into halves of a word is what createTriangleShaders refuses outright, so
    // the kernel declares z as array<f32> here; handing it f16 words made it
    // read one number out of every two and produced a plausible-looking answer
    // that was wrong by 9e-2.
    const zData = input.z;
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
      const statistics = keep(this.allocator.allocate("triangle.statistics", pairCount * 2 * 4, storage));
      const gate = keep(this.allocator.allocate("triangle.gate", blockPairs * cZ * 4, storage));
      const blocked = keep(this.allocator.allocate("triangle.blocked", blockPairs * cHidden * 4, storage));
      const whole = keep(this.allocator.allocate("triangle.whole",
        wholeStorage === "f16" ? wholeStride * cHidden * 2 : wholeStride * cHidden * 4, storage));
      const contracted = keep(this.allocator.allocate("triangle.contracted", blockPairs * cHidden * 4, storage));
      // The projection is bound entire here, so the group starts at channel
      // zero and covers all of them.
      const channels = keep(this.allocator.upload("triangle.channels",
        new Uint32Array([0, cHidden, 0, 0]), GPUBufferUsage.UNIFORM));
      const hiddenStatisticsBuffer = keep(this.allocator.allocate("triangle.hidden-statistics", blockPairs * 2 * 4, storage));
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
        buffers: readonly Binding[],
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

      // Blocks of output rows, as the Evoformer block encoder issues them.
      const blocks: { readonly offset: number; readonly count: number; readonly params: GPUBuffer }[] = [];
      for (let offset = 0; offset < length; offset += blockRows) {
        const count = Math.min(blockRows, length - offset);
        blocks.push({ offset, count, params: keep(this.allocator.upload(`triangle.block-${offset}`,
          new Uint32Array([offset * length, count * length, offset === 0 ? 1 : 0, count]), GPUBufferUsage.UNIFORM)).buffer });
      }
      // One workgroup per pair row.
      const rowGrid: readonly [number, number] = [
        Math.min(pairCount, LINEAR_GRID_WIDTH), ceilDivide(pairCount, LINEAR_GRID_WIDTH),
      ];
      runPass("input-statistics", inputStatistics, [z.buffer, statistics.buffer], rowGrid[0], rowGrid[1]);
      for (const block of blocks) {
        const grid = gemmGrid(block.count * length, 2 * cHidden);
        runPass(`project-whole-operand-${block.offset}`, projectWholeOperand,
          [z.buffer, mask.buffer, weights.buffer, statistics.buffer, whole.buffer, block.params], grid[0], grid[1]);
      }
      for (const block of blocks) {
        const rows = block.count * length;
        const projectionGrid = gemmGrid(rows, 2 * cHidden);
        runPass(`project-block-operand-${block.offset}`, projectBlockOperand,
          [z.buffer, mask.buffer, weights.buffer, statistics.buffer, blocked.buffer, block.params],
          projectionGrid[0], projectionGrid[1]);
        const contractGrid = gemmGrid(block.count, length);
        runPass(`contract-${block.offset}`, contract,
          [blocked.buffer, whole.buffer, contracted.buffer, block.params, channels.buffer],
          contractGrid[0], contractGrid[1], cHidden);
        runPass(`hidden-statistics-${block.offset}`, hiddenStatistics,
          [contracted.buffer, hiddenStatisticsBuffer.buffer, block.params], ceilDivide(rows, 64));
        const outputGrid = gemmGrid(rows, cZ);
        runPass(`project-gate-${block.offset}`, projectGate,
          [z.buffer, weights.buffer, statistics.buffer, gate.buffer, block.params], outputGrid[0], outputGrid[1]);
        runPass(`project-output-${block.offset}`, projectOutput,
          [gate.buffer, contracted.buffer, weights.buffer, hiddenStatisticsBuffer.buffer, output.buffer, block.params],
          outputGrid[0], outputGrid[1]);
      }
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
