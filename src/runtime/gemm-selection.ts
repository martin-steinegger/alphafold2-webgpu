import {
  createTiledGemmShader, gemmGrid, GEMM_VARIANT_F32, MATRIX_LANES, matrixGemmStorageBytes,
  MATRIX_REGION, MATRIX_SHAPE_F32_8, setGemmVariant, type GemmVariant, type MatrixUnitShape,
} from "./gemm.js";
import { supportsSubgroupSize } from "./subgroups.js";

/**
 * Which arithmetic and which k depth the dense projections use on this device.
 *
 * One hand-tiled GEMM serves every projection in the model, and two of its
 * choices cannot be derived from anything WebGPU reports. Apple's shader cores
 * issue half-precision multiply-accumulate at twice the f32 rate, worth 1.15x
 * to 1.26x on the projection shapes once the reduction is kept safe; the same
 * kernel is no faster on adapters whose f16 is emulated, and the depth of the
 * staged k tile swings a few percent either way by driver. So both are
 * measured once per device and the winner is cached, the way
 * `src/evoformer/attention-calibration.ts` already picks the flash kernel.
 *
 * Which arrangements may be measured at all is not a runtime question, and
 * `SHIPPABLE_GEMM_PRECISIONS` below says why.
 *
 * The selection is installed before `requestAlphaFoldDevice` hands the device
 * out, which is what keeps it invisible to every call site. No consumer can
 * have generated a projection shader yet, so one pipeline cache key can never
 * come to describe two different shaders, and neither choice enters the
 * dispatch grid: `gemmGrid` derives that from the output tile, which does not
 * move. Callers keep calling `createTiledGemmShader` with no idea any of this
 * happened.
 */

/**
 * Two shapes, because one kernel has to serve both and they do not rank the
 * same. The model runs wide projections and narrow ones, and the matrix kernel
 * in particular is fast on the wide shapes and slow on the narrow ones, where
 * a 128-column output leaves the dispatch grid one workgroup across.
 *
 * The row count of the first is not arbitrary and is the expensive part. The
 * matrix kernel puts one subgroup in a workgroup, so its occupancy comes
 * entirely from having many workgroups, and a probe with too few rows starves
 * it: measured against `f16-chunked`, a 1,024-row shape ranks it at 0.53 while
 * the projections the model actually runs rank it between 1.23 and 1.45. At
 * 16,384 rows the probe reproduces that ordering at 1.17, which is the
 * cheapest shape tried that still gets the answer right rather than backwards.
 * It costs about 34 MiB while it runs, and it is freed before the device is
 * handed out.
 */
const PROBE_SHAPES = [
  { rows: 16384, inner: 256, columns: 256 },
  { rows: 2048, inner: 512, columns: 128 },
] as const;
const PROBE_REPEATS = 2;

/**
 * How long one timed batch should take, and the resolution that forces.
 *
 * A browser clamps `performance.now` to about 0.1 ms, so a batch of four
 * dispatches of a 0.4 ms kernel can only be measured to about 6% — coarser
 * than the 10% to 15% that separates these variants, which makes the ranking
 * noise. Measured that way the probe picked the slowest half-precision
 * arrangement over the fastest. So a rough pass sizes the real batch to reach
 * this many milliseconds, the way `gemm-calibration.spec.ts` already does.
 */
const PROBE_BATCH_MILLISECONDS = 12;
const PROBE_ROUGH_DISPATCHES = 4;
const PROBE_MAX_DISPATCHES = 2000;

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
 * noise. Apple clears it comfortably.
 */
const HALF_PRECISION_MARGIN = 1.1;

/**
 * How much faster the matrix units have to be before they are used.
 *
 * Lower than the half-precision margin, and deliberately: the matrix kernel
 * accumulates in f32 and reproduces the reference to the digit, so there is no
 * accuracy being traded and nothing to be cautious about beyond measurement
 * noise. It still has to win, because it is an experimental extension and
 * because it is slow on narrow outputs.
 */
const MATRIX_MARGIN = 1.05;

const selections = new WeakMap<GPUDevice, Promise<GemmVariant>>();

/**
 * A variant pinned ahead of any device, for differential testing.
 *
 * Whether half precision costs the model anything can only be answered by
 * predicting one input both ways and comparing, which needs the choice held
 * still across two runs that would otherwise measure it. This is that hold,
 * the counterpart of `presetAttentionFlashKernel`. It is not a setting: no URL,
 * environment variable or stored preference reaches it, and production never
 * calls it, so a device still measures its own arithmetic.
 */
let pinnedVariant: GemmVariant | undefined;

export function forceGemmVariant(variant: GemmVariant | undefined): void {
  pinnedVariant = variant;
  setGemmVariant(variant ?? GEMM_VARIANT_F32);
}

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

/**
 * Arrangements a device is allowed to be measured into, and why not more.
 *
 * Pure `f16` is deliberately absent. It is the fastest of them, at 1.36x to
 * 1.55x, and it is not safe: accumulating a whole contraction in half
 * precision overflows on a deep MSA, which took the 508-row acceptance
 * prediction from 96.80 pLDDT to 69.94 and its pTM to NaN. No runtime probe
 * can rediscover that, because it only shows up at a depth and a magnitude a
 * cheap probe does not reach, so the exclusion is recorded here instead of
 * being left to a measurement. `test/browser/gemm-differential.spec.ts` holds
 * every variant in this list to the prediction gate.
 */
export const SHIPPABLE_GEMM_PRECISIONS: readonly GemmVariant["precision"][] = [
  "f32", "matrix", "f16-chunked", "f16-mixed",
];

/** The matrix units are an experimental Chromium extension, not core WGSL. */
const MATRIX_FEATURE = "chromium-experimental-subgroup-matrix";

function hasMatrixUnits(device: GPUDevice): boolean {
  const features: GPUSupportedFeatures | undefined = device.features;
  // The kernel lays one tile across one subgroup and indexes `lane % 32`, and
  // says so with `@subgroup_size`. A device that cannot be held to that width
  // is not offered the units. A stub in a test carries no features at all.
  return features?.has(MATRIX_FEATURE as GPUFeatureName) === true
    && (features.has("subgroups" as GPUFeatureName) !== true
      || supportsSubgroupSize(device, MATRIX_LANES));
}

/**
 * One entry of `GPUAdapterInfo.subgroupMatrixConfigs`.
 *
 * Declared structurally because the shipping `@webgpu/types` does not carry
 * the experimental extension, and because only these five fields are read.
 */
export interface SubgroupMatrixConfig {
  readonly componentType: string;
  readonly resultComponentType: string;
  readonly M: number;
  readonly N: number;
  readonly K: number;
}

/**
 * The configuration this device should run the matrix kernel with, if any.
 *
 * Exact before fast: an f32 configuration reproduces the reference to the
 * digit, so it is taken whenever one exists and no f16 shape is considered.
 * Only a device that offers none — every current Nvidia part — falls to f16
 * operands, and then the widest shape wins, being the most arithmetic per
 * instruction issued.
 *
 * The accumulator must be f32 either way. A configuration that can only
 * accumulate in f16 is not a faster matrix kernel, it is the `f16` precision
 * this file already refuses to ship, reached by another route.
 */
export function selectMatrixShape(
  device: GPUDevice, configs: readonly SubgroupMatrixConfig[],
): MatrixUnitShape | undefined {
  // f32 components only, and not for want of trying an f16 kernel: one was
  // written, reproduced the reference to 3e-4, and aborted the driver on
  // dispatch as soon as the contraction ran past five sixteen-deep steps.
  // Every contraction in this model is far longer than that — the projections
  // reduce over 256 and the transitions over 512 — and this list is walked by
  // the calibration probe, so offering the shape here would not merely lose
  // the measurement, it would take the process down before the model started.
  // The attention kernel uses the same units without this, which is why it is
  // reached through `attentionMatrixConfig` and not through here.
  const usable = configs.filter((config) =>
    config.resultComponentType === "f32"
    && (config.componentType === "f32"
      || (config.componentType === "f16" && hasHalfPrecision(device)))
    // The kernel walks a 32x32 region with whole units, so both must divide it.
    && MATRIX_REGION % config.M === 0 && MATRIX_REGION % config.N === 0);
  // Exact before fast: an f32 configuration reproduces the reference to the
  // digit, so it wins whenever one exists.
  const exact = usable.filter((config) => config.componentType === "f32");
  const preferred = (exact.length > 0 ? exact : usable)
    .sort((left, right) => right.M * right.N * right.K - left.M * left.N * left.K)[0];
  return preferred === undefined ? undefined : {
    componentType: preferred.componentType as "f32" | "f16",
    M: preferred.M, N: preferred.N, K: preferred.K,
  };
}

/** Variants worth measuring against each other on this device. */
export function gemmVariantCandidates(
  device: GPUDevice, configs: readonly SubgroupMatrixConfig[] = [],
): readonly GemmVariant[] {
  const depths = [8, 16] as const;
  const half = hasHalfPrecision(device) && !sawDeviceWithoutHalfPrecision;
  const candidates: GemmVariant[] = [];
  for (const precision of SHIPPABLE_GEMM_PRECISIONS) {
    if (precision === "matrix") {
      if (!hasMatrixUnits(device)) continue;
      // A device that reports no configurations at all is Apple, whose shape
      // predates the reporting; anything else is taken at its word.
      const shape = configs.length === 0
        ? MATRIX_SHAPE_F32_8 : selectMatrixShape(device, configs);
      // The f16 kernel stages both operands and the result, past the 16 KiB
      // baseline; a device that grants only that keeps the hand-tiled kernel.
      // A stub standing in for a device in a test reports no limits at all,
      // and is read as the baseline rather than as an error, the same way
      // `hasHalfPrecision` reads a missing feature set.
      const granted = device.limits?.maxComputeWorkgroupStorageSize ?? 16384;
      if (shape === undefined) continue;
      // One unit of depth a step spends two barriers on as many multiplies as
      // the tile is wide; two units spend them once for twice as many, and
      // stage the same operands to do it. Which wins is the device's answer,
      // so both are offered wherever the storage is granted.
      // Only the f16 kernel stages the operands at all, so only it has a
      // depth to stage. The f32 one hands the units a pointer into the
      // caller's own array and reads a unit's worth at a time.
      const depths = shape.componentType === "f16" ? [1, 2] : [1];
      for (const depth of depths) {
        if (matrixGemmStorageBytes(shape, undefined, depth * shape.K) > granted) continue;
        candidates.push({ precision, inner: 8, matrix: shape, matrixDepth: depth });
      }
      continue;
    }
    if (precision !== "f32" && !half) continue;
    for (const inner of depths) candidates.push({ precision, inner });
  }
  return candidates;
}

export function gemmVariantName(variant: GemmVariant): string {
  if (variant.precision !== "matrix") return `${variant.precision}-64x128k${variant.inner}`;
  const unit = variant.matrix;
  const apple = unit === undefined || (unit.componentType === "f32"
    && unit.M === MATRIX_SHAPE_F32_8.M && unit.N === MATRIX_SHAPE_F32_8.N
    && unit.K === MATRIX_SHAPE_F32_8.K);
  return apple ? "matrix-64x128"
    : `matrix-${unit!.componentType}${unit!.M}x${unit!.N}x${unit!.K}-64x128`
      + ((variant.matrixDepth ?? 1) === 1 ? "" : `d${variant.matrixDepth}`);
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
    // Plain row-major arrays, so this probe can measure the matrix units too.
    sourceArray: { array: "source", stride: "parameters.inner" },
    weightArray: { array: "weights", stride: "parameters.columns" },
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

/**
 * The probe builds its pipelines directly rather than through the shared
 * cache: they are used once, at device creation, and would otherwise sit in a
 * cache meant for the kernels the model actually runs for the device's whole
 * lifetime.
 */
async function pipelineFor(device: GPUDevice, variant: GemmVariant): Promise<GPUComputePipeline> {
  const label = `gemm-selection.${gemmVariantName(variant)}`;
  return device.createComputePipelineAsync({
    label, layout: "auto",
    compute: {
      module: device.createShaderModule({ label: `${label}.wgsl`, code: probeShader(variant) }),
      entryPoint: "main",
    },
  });
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
async function measureError(
  device: GPUDevice, variant: GemmVariant, pipeline: GPUComputePipeline,
): Promise<number> {
  const probe = createProbe(device, CHECK_ROWS, CHECK_INNER, CHECK_COLUMNS, true);
  const readback = device.createBuffer({
    label: "gemm-selection.readback",
    size: CHECK_ROWS * CHECK_COLUMNS * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  try {
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

/**
 * One timed batch of a candidate on one shape, in milliseconds per dispatch.
 *
 * A submission costs a millisecond or two to come back and the clock is
 * coarse, both of which swamp one dispatch of a few hundred microseconds, so a
 * batch of dispatches is what gets timed.
 */
async function timeBatch(
  device: GPUDevice, variant: GemmVariant, timing: VariantTiming, dispatches: number,
): Promise<number> {
  const encoder = device.createCommandEncoder({ label: `gemm-selection.time.${variant.precision}` });
  const pass = encoder.beginComputePass();
  pass.setPipeline(timing.pipeline);
  pass.setBindGroup(0, timing.group);
  const [x, y] = timing.grid;
  for (let dispatch = 0; dispatch < dispatches; dispatch += 1) pass.dispatchWorkgroups(x, y, 1);
  pass.end();
  const started = performance.now();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - started) / dispatches;
}

/** One candidate measured on one shape: what to dispatch and the best seen. */
interface VariantTiming {
  readonly pipeline: GPUComputePipeline;
  readonly group: GPUBindGroup;
  readonly grid: readonly [number, number];
  /** Sized so a batch reaches `PROBE_BATCH_MILLISECONDS`, once, up front. */
  dispatches: number;
  best: number;
}

export interface GemmVariantMeasurement {
  readonly variant: GemmVariant;
  /** Total across the probe shapes; what the hand-tiled kernel is ranked by. */
  readonly milliseconds: number;
  /** Per shape, in `PROBE_SHAPES` order, so unlike things are not compared. */
  readonly perShape: readonly number[];
  readonly relativeError: number;
}

/**
 * Measures every candidate on this device, in candidate order.
 *
 * A candidate that will not compile or run is dropped rather than allowed to
 * end the whole calibration: an experimental extension may be advertised and
 * still reject a kernel, and losing the other candidates over it would leave
 * the device on f32 for no reason.
 */
export async function measureGemmVariants(
  device: GPUDevice, configs: readonly SubgroupMatrixConfig[] = [],
): Promise<readonly GemmVariantMeasurement[]> {
  const probes = PROBE_SHAPES.map(
    (shape) => createProbe(device, shape.rows, shape.inner, shape.columns, false),
  );
  try {
    // Compile and check every candidate first, so the timing loop below holds
    // nothing but candidates that already reproduced the reference.
    const entries: { variant: GemmVariant; relativeError: number; timings: VariantTiming[] }[] = [];
    for (const variant of gemmVariantCandidates(device, configs)) {
      try {
        // One pipeline per candidate, not one per measurement: compiling each
        // shader twice was most of what this probe cost at device creation.
        const pipeline = await pipelineFor(device, variant);
        const relativeError = await measureError(device, variant, pipeline);
        const timings = probes.map((probe) => ({
          pipeline, group: bindProbe(device, pipeline, probe),
          grid: gemmGrid(probe.rows, probe.columns),
          dispatches: PROBE_ROUGH_DISPATCHES, best: Number.POSITIVE_INFINITY,
        }));
        entries.push({ variant, relativeError, timings });
      } catch {
        // Left out of the ranking entirely.
      }
    }
    // Size each batch to reach `PROBE_BATCH_MILLISECONDS`. The rough pass also
    // warms the pipeline, so its own time is discarded.
    for (const entry of entries) {
      for (const timing of entry.timings) {
        const rough = await timeBatch(device, entry.variant, timing, PROBE_ROUGH_DISPATCHES);
        timing.dispatches = Math.max(PROBE_ROUGH_DISPATCHES, Math.min(PROBE_MAX_DISPATCHES,
          Math.ceil(PROBE_BATCH_MILLISECONDS / Math.max(rough, 0.01))));
      }
    }
    // Sweep the candidates round-robin rather than finishing one before
    // starting the next. Measured the other way each candidate owns a
    // different window of wall time, so anything that drifts across the sweep
    // — a clock ramping, a neighbouring process on a shared GPU — lands on
    // whichever candidates happened to be running and ranks them, not the
    // kernels. Interleaving gives every candidate the same windows, and the
    // per-batch minimum then rejects the spikes instead of averaging them in.
    for (let repeat = 0; repeat < PROBE_REPEATS; repeat += 1) {
      for (const entry of entries) {
        for (const timing of entry.timings) {
          timing.best = Math.min(timing.best,
            await timeBatch(device, entry.variant, timing, timing.dispatches));
        }
      }
    }
    return entries.map(({ variant, relativeError, timings }) => {
      const perShape = timings.map((timing) => timing.best);
      return {
        variant, relativeError, perShape,
        milliseconds: perShape.reduce((total, value) => total + value, 0),
      };
    });
  } finally {
    for (const probe of probes) destroyProbe(probe);
  }
}

/**
 * Picks the projection variant for this device and installs it.
 *
 * Correctness first: a candidate that does not reproduce the reference cannot
 * win however fast it is. Among the rest the fastest f32 tile is the one to
 * beat, and half precision has to beat it by `HALF_PRECISION_MARGIN` to be
 * chosen, so an adapter where f16 buys nothing stays exact. Anything that
 * throws leaves the f32 kernel in place.
 */
export function calibrateGemmVariant(
  device: GPUDevice, configs: readonly SubgroupMatrixConfig[] = [],
): Promise<GemmVariant> {
  if (pinnedVariant !== undefined) {
    setGemmVariant(pinnedVariant);
    return Promise.resolve(pinnedVariant);
  }
  const cached = selections.get(device);
  if (cached !== undefined) return cached;
  const selection = (async (): Promise<GemmVariant> => {
    try {
      if (!hasHalfPrecision(device)) sawDeviceWithoutHalfPrecision = true;
      const measurements = await measureGemmVariants(device, configs);
      const usable = measurements.filter(
        (measurement) => measurement.relativeError <= CHECK_TOLERANCE,
      );
      // A shallower tile keeps a tie, having less workgroup storage to lose.
      const fastest = (
        precision: GemmVariant["precision"],
      ): GemmVariantMeasurement | undefined =>
        usable.filter((measurement) => measurement.variant.precision === precision)
          .sort((left, right) => left.milliseconds - right.milliseconds
            || left.variant.inner - right.variant.inner)[0];
      const exact = fastest("f32");
      const half = [fastest("f16-chunked"), fastest("f16-mixed")]
        .filter((measurement) => measurement !== undefined)
        .sort((left, right) => left.milliseconds - right.milliseconds)[0];
      // If the f32 kernel itself did not reproduce the reference, the probe is
      // what is broken, not the arithmetic: there is nothing to compare
      // against, so nothing is changed.
      const classic = exact === undefined ? GEMM_VARIANT_F32
        : half !== undefined && half.milliseconds * HALF_PRECISION_MARGIN < exact.milliseconds
          ? half.variant : exact.variant;
      // The matrix units are a separate question from the arithmetic, because
      // they serve a different set of callers: only the wide projections can
      // declare their operands as arrays. So they are judged on the wide probe
      // shape alone, against whatever the hand-tiled kernel does on that same
      // shape — comparing a one-shape score against a two-shape total would
      // simply make the matrix kernel look slower than everything.
      const wideTime = (measurement: GemmVariantMeasurement): number =>
        measurement.perShape[0] ?? Number.POSITIVE_INFINITY;
      const matrix = usable.find((measurement) => measurement.variant.precision === "matrix");
      const bestClassicWide = usable
        .filter((measurement) => measurement.variant.precision !== "matrix")
        .sort((left, right) => wideTime(left) - wideTime(right))[0];
      // An f16 configuration rounds both operands, so it is held to the
      // half-precision margin; only an f32 one is exact enough for the
      // narrower matrix margin this path was given.
      const matrixMargin = matrix?.variant.matrix?.componentType === "f16"
        ? HALF_PRECISION_MARGIN : MATRIX_MARGIN;
      // What a caller that cannot reach the units computes instead. The
      // half-precision margin keeps a device exact when half precision buys
      // too little to be worth its rounding — but a run whose units already
      // take f16 operands has made that trade, and holding the callers that
      // missed out to single precision only makes them slower for a
      // consistency the run does not have. So they take the fastest kernel
      // that reproduced the reference. Measured: the attention projections
      // fell from the chunked kernel to f32 under a matrix winner and cost
      // 0.11s of a recycle for it.
      const matrixIsHalf = matrix?.variant.matrix?.componentType === "f16";
      const fastestClassic = half !== undefined && half.milliseconds < (exact?.milliseconds ?? Infinity)
        ? half.variant : classic;
      const fallback = matrixIsHalf ? fastestClassic : classic;
      const winner: GemmVariant = matrix !== undefined && bestClassicWide !== undefined
        && wideTime(matrix) * matrixMargin < wideTime(bestClassicWide)
        ? { precision: "matrix", inner: fallback.inner,
          ...(matrix.variant.matrixDepth === undefined
            ? {} : { matrixDepth: matrix.variant.matrixDepth }),
          fallback: fallback.precision as
            "f32" | "f16-mixed" | "f16-chunked",
            ...(matrix.variant.matrix === undefined ? {} : { matrix: matrix.variant.matrix }) }
        : classic;
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
