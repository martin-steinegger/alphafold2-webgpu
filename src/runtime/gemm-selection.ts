import {
  createTiledGemmShader, gemmGrid, GEMM_VARIANT_F32, setGemmVariant,
  type GemmVariant,
} from "./gemm.js";
import { pipelineCacheForDevice } from "./pipeline-cache.js";

/**
 * Which arithmetic and which k depth the dense projections use on this device.
 *
 * One hand-tiled GEMM serves every projection in the model, and two of its
 * choices cannot be derived from anything WebGPU reports. Apple's shader cores
 * issue half-precision multiply-accumulate at twice the f32 rate, which made
 * pure f16 1.36x to 1.55x faster than f32 at every shape the model runs on an
 * M4 Pro; the same kernel is no faster on adapters whose f16 is emulated, and
 * the depth of the staged k tile swings a few percent either way by driver.
 * So both are measured once per device and the winner is cached, the way
 * `src/evoformer/attention-calibration.ts` already picks the flash kernel.
 *
 * The selection is installed before `requestAlphaFoldDevice` hands the device
 * out, which is what keeps it invisible to every call site. No consumer can
 * have generated a projection shader yet, so one pipeline cache key can never
 * come to describe two different shaders, and neither choice enters the
 * dispatch grid: `gemmGrid` derives that from the output tile, which does not
 * move. Callers keep calling `createTiledGemmShader` with no idea any of this
 * happened.
 */

/** Large enough to be throughput-bound, small enough to stay under 6 MiB. */
const PROBE_ROWS = 1024;
const PROBE_INNER = 512;
const PROBE_COLUMNS = 512;
const PROBE_DISPATCHES = 4;
const PROBE_REPEATS = 3;

/**
 * Shape the correctness check runs at, and how wrong a candidate may be.
 *
 * This is not the accuracy gate. Whether half precision costs the model
 * anything was settled end to end, by predicting the same input with and
 * without it; 1% here would be far too loose for that. What this catches is a
 * device whose f16 does not compute a projection at all, which matters
 * precisely because the winner ships to adapters neither of us has measured.
 * A kernel that is merely inaccurate passes; one that is broken cannot win.
 */
const CHECK_ROWS = 64;
const CHECK_INNER = 256;
const CHECK_COLUMNS = 64;
const CHECK_TOLERANCE = 0.02;

/**
 * How much faster half precision has to be before it is worth its rounding.
 *
 * Half precision is not free: it rounds every staged operand and every partial
 * sum, which the end-to-end differential showed the model tolerates but which
 * buys nothing on an adapter that merely emulates f16. Requiring a clear
 * margin keeps such a device exact instead of trading accuracy for measurement
 * noise. Apple clears it by a wide margin, at 1.36x to 1.55x.
 */
const HALF_PRECISION_MARGIN = 1.1;

const selections = new WeakMap<GPUDevice, Promise<GemmVariant>>();

/**
 * Whether a device without `shader-f16` has ever been calibrated.
 *
 * The variant is process-wide, because `createTiledGemmShader` is called
 * without a device in scope. A process that drives two adapters at once must
 * therefore emit shaders both of them accept, and only f32 qualifies. Tests
 * are the realistic case; a browser session has one device.
 */
let sawDeviceWithoutHalfPrecision = false;

/**
 * A device is only half-precision capable if it says so. Anything that does
 * not report a feature set at all, such as a stub standing in for a device in
 * a test, is treated as lacking the feature rather than as an error: this runs
 * inside device creation and must never be what makes it fail.
 */
function hasHalfPrecision(device: GPUDevice): boolean {
  const features: GPUSupportedFeatures | undefined = device.features;
  return features?.has("shader-f16" as GPUFeatureName) === true;
}

/** Variants worth measuring against each other on this device. */
export function gemmVariantCandidates(device: GPUDevice): readonly GemmVariant[] {
  const depths = [8, 16] as const;
  const precisions: GemmVariant["precision"][] = ["f32"];
  if (hasHalfPrecision(device) && !sawDeviceWithoutHalfPrecision) precisions.push("f16");
  return precisions.flatMap((precision) => depths.map((inner) => ({ precision, inner })));
}

export function gemmVariantName(variant: GemmVariant): string {
  return `${variant.precision}-64x128k${variant.inner}`;
}

/** A bias-free projection with the shared tiling, for probing one variant. */
function probeShader(variant: GemmVariant): string {
  return createTiledGemmShader({
    preamble: `
struct ProbeParameters { rows: u32, inner: u32, columns: u32, padding: u32 };
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> parameters: ProbeParameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;`,
    rows: "parameters.rows",
    inner: "parameters.inner",
    columns: "parameters.columns",
    sourceElement: "source[row * parameters.inner + k]",
    weightElement: "weights[k * parameters.columns + column]",
    store: "output[row * parameters.columns + column] = element;",
  }, variant);
}

/** The same deterministic values on every device, so a run is reproducible. */
function probeValues(count: number): Float32Array {
  let state = 0x1234567;
  return Float32Array.from({ length: count }, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000 - 0.5;
  });
}

interface Probe {
  readonly source: GPUBuffer;
  readonly weights: GPUBuffer;
  readonly parameters: GPUBuffer;
  readonly output: GPUBuffer;
  readonly rows: number;
  readonly inner: number;
  readonly columns: number;
}

function createProbe(
  device: GPUDevice, rows: number, inner: number, columns: number, readable: boolean,
): Probe {
  const buffer = (elements: number, usage: GPUBufferUsageFlags): GPUBuffer =>
    device.createBuffer({ label: "gemm-selection.probe", size: elements * 4, usage });
  const source = buffer(rows * inner, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const weights = buffer(inner * columns, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(source, 0, probeValues(rows * inner));
  device.queue.writeBuffer(weights, 0, probeValues(inner * columns));
  const parameters = buffer(4, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(parameters, 0, new Uint32Array([rows, inner, columns, 0]));
  const output = buffer(rows * columns,
    GPUBufferUsage.STORAGE | (readable ? GPUBufferUsage.COPY_SRC : 0));
  return { source, weights, parameters, output, rows, inner, columns };
}

function destroyProbe(probe: Probe): void {
  for (const buffer of [probe.source, probe.weights, probe.parameters, probe.output]) {
    buffer.destroy();
  }
}

async function pipelineFor(device: GPUDevice, variant: GemmVariant): Promise<GPUComputePipeline> {
  return pipelineCacheForDevice(device)
    .get(`gemm-selection.${gemmVariantName(variant)}`, probeShader(variant));
}

function bindProbe(
  device: GPUDevice, pipeline: GPUComputePipeline, probe: Probe,
): GPUBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [probe.source, probe.weights, probe.parameters, probe.output]
      .map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
}

/**
 * Worst error of one variant against a reference computed here, relative to
 * the largest value in it.
 */
async function measureError(device: GPUDevice, variant: GemmVariant): Promise<number> {
  const probe = createProbe(device, CHECK_ROWS, CHECK_INNER, CHECK_COLUMNS, true);
  const readback = device.createBuffer({
    label: "gemm-selection.readback",
    size: CHECK_ROWS * CHECK_COLUMNS * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  try {
    const pipeline = await pipelineFor(device, variant);
    const encoder = device.createCommandEncoder({ label: `gemm-selection.check.${variant.precision}` });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindProbe(device, pipeline, probe));
    const [x, y] = gemmGrid(CHECK_ROWS, CHECK_COLUMNS);
    pass.dispatchWorkgroups(x, y, 1);
    pass.end();
    encoder.copyBufferToBuffer(probe.output, 0, readback, 0, CHECK_ROWS * CHECK_COLUMNS * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    const source = probeValues(CHECK_ROWS * CHECK_INNER);
    const weights = probeValues(CHECK_INNER * CHECK_COLUMNS);
    let worst = 0;
    let scale = 0;
    for (let row = 0; row < CHECK_ROWS; row += 1) {
      for (let column = 0; column < CHECK_COLUMNS; column += 1) {
        let total = 0;
        for (let k = 0; k < CHECK_INNER; k += 1) {
          total += source[row * CHECK_INNER + k]! * weights[k * CHECK_COLUMNS + column]!;
        }
        worst = Math.max(worst, Math.abs(values[row * CHECK_COLUMNS + column]! - total));
        scale = Math.max(scale, Math.abs(total));
      }
    }
    return scale === 0 ? Number.POSITIVE_INFINITY : worst / scale;
  } finally {
    destroyProbe(probe);
    readback.destroy();
  }
}

/** Milliseconds per dispatch of one variant, best of several batches. */
async function measureTime(device: GPUDevice, variant: GemmVariant, probe: Probe): Promise<number> {
  const pipeline = await pipelineFor(device, variant);
  const group = bindProbe(device, pipeline, probe);
  const [x, y] = gemmGrid(probe.rows, probe.columns);
  // A submission costs a millisecond or two to come back and the clock is
  // coarse, both of which swamp one dispatch of a few hundred microseconds, so
  // a batch is what gets timed.
  const batch = async (): Promise<number> => {
    const encoder = device.createCommandEncoder({ label: `gemm-selection.time.${variant.precision}` });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    for (let dispatch = 0; dispatch < PROBE_DISPATCHES; dispatch += 1) pass.dispatchWorkgroups(x, y, 1);
    pass.end();
    const started = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - started) / PROBE_DISPATCHES;
  };
  await batch();
  let best = Number.POSITIVE_INFINITY;
  for (let repeat = 0; repeat < PROBE_REPEATS; repeat += 1) best = Math.min(best, await batch());
  return best;
}

export interface GemmVariantMeasurement {
  readonly variant: GemmVariant;
  readonly milliseconds: number;
  readonly relativeError: number;
}

/** Measures every candidate on this device, fastest first among the correct ones. */
export async function measureGemmVariants(
  device: GPUDevice,
): Promise<readonly GemmVariantMeasurement[]> {
  const probe = createProbe(device, PROBE_ROWS, PROBE_INNER, PROBE_COLUMNS, false);
  try {
    const measurements: GemmVariantMeasurement[] = [];
    for (const variant of gemmVariantCandidates(device)) {
      const relativeError = await measureError(device, variant);
      measurements.push({
        variant, relativeError,
        milliseconds: await measureTime(device, variant, probe),
      });
    }
    return measurements;
  } finally {
    destroyProbe(probe);
  }
}

/**
 * Picks the projection variant for this device and installs it.
 *
 * Correctness first: a candidate that does not reproduce the reference cannot
 * win however fast it is. Among the rest the fastest wins, and f32 keeps the
 * tie so that an adapter where half precision buys nothing stays exact.
 * Anything that throws leaves the f32 kernel in place.
 */
export function calibrateGemmVariant(device: GPUDevice): Promise<GemmVariant> {
  const cached = selections.get(device);
  if (cached !== undefined) return cached;
  const selection = (async (): Promise<GemmVariant> => {
    try {
      if (!hasHalfPrecision(device)) sawDeviceWithoutHalfPrecision = true;
      const measurements = await measureGemmVariants(device);
      const usable = measurements.filter(
        (measurement) => measurement.relativeError <= CHECK_TOLERANCE,
      );
      // A shallower tile keeps a tie, having less workgroup storage to lose.
      const fastest = (precision: GemmVariant["precision"]): GemmVariantMeasurement | undefined =>
        usable.filter((measurement) => measurement.variant.precision === precision)
          .sort((left, right) => left.milliseconds - right.milliseconds
            || left.variant.inner - right.variant.inner)[0];
      const exact = fastest("f32");
      const half = fastest("f16");
      const winner = exact === undefined ? half?.variant ?? GEMM_VARIANT_F32
        : half !== undefined && half.milliseconds * HALF_PRECISION_MARGIN < exact.milliseconds
          ? half.variant : exact.variant;
      setGemmVariant(winner);
      return winner;
    } catch {
      setGemmVariant(GEMM_VARIANT_F32);
      return GEMM_VARIANT_F32;
    }
  })();
  selections.set(device, selection);
  return selection;
}

/** Forces a variant, for tests that need one deliberately. */
export function presetGemmVariant(device: GPUDevice, variant: GemmVariant): void {
  selections.set(device, Promise.resolve(variant));
  setGemmVariant(variant);
}
