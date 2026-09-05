import {
  createTiledGemmShader, gemmGrid, GEMM_VARIANT_F32, setGemmVariant,
  type GemmVariant,
} from "./gemm.js";

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
  return features?.has(MATRIX_FEATURE as GPUFeatureName) === true;
}

/** Variants worth measuring against each other on this device. */
export function gemmVariantCandidates(device: GPUDevice): readonly GemmVariant[] {
  const depths = [8, 16] as const;
  const half = hasHalfPrecision(device) && !sawDeviceWithoutHalfPrecision;
  const candidates: GemmVariant[] = [];
  for (const precision of SHIPPABLE_GEMM_PRECISIONS) {
    if (precision === "matrix") {
      // The units fix the contraction step, so there is one of these.
      if (hasMatrixUnits(device)) candidates.push({ precision, inner: 8 });
      continue;
    }
    if (precision !== "f32" && !half) continue;
    for (const inner of depths) candidates.push({ precision, inner });
  }
  return candidates;
}

export function gemmVariantName(variant: GemmVariant): string {
  return variant.precision === "matrix"
    ? "matrix-64x128" : `${variant.precision}-64x128k${variant.inner}`;
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
  // a batch of dispatches is what gets timed.
  const batch = async (dispatches: number): Promise<number> => {
    const encoder = device.createCommandEncoder({ label: `gemm-selection.time.${variant.precision}` });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    for (let dispatch = 0; dispatch < dispatches; dispatch += 1) pass.dispatchWorkgroups(x, y, 1);
    pass.end();
    const started = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - started) / dispatches;
  };
  // The rough pass also warms the pipeline, so its own time is discarded.
  const rough = await batch(PROBE_ROUGH_DISPATCHES);
  const dispatches = Math.max(PROBE_ROUGH_DISPATCHES, Math.min(PROBE_MAX_DISPATCHES,
    Math.ceil(PROBE_BATCH_MILLISECONDS / Math.max(rough, 0.01))));
  let best = Number.POSITIVE_INFINITY;
  for (let repeat = 0; repeat < PROBE_REPEATS; repeat += 1) {
    best = Math.min(best, await batch(dispatches));
  }
  return best;
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
  device: GPUDevice,
): Promise<readonly GemmVariantMeasurement[]> {
  const probes = PROBE_SHAPES.map(
    (shape) => createProbe(device, shape.rows, shape.inner, shape.columns, false),
  );
  try {
    const measurements: GemmVariantMeasurement[] = [];
    for (const variant of gemmVariantCandidates(device)) {
      try {
        const relativeError = await measureError(device, variant);
        const perShape: number[] = [];
        for (const probe of probes) perShape.push(await measureTime(device, variant, probe));
        measurements.push({
          variant, relativeError, perShape,
          milliseconds: perShape.reduce((total, value) => total + value, 0),
        });
      } catch {
        // Left out of the ranking entirely.
      }
    }
    return measurements;
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
export function calibrateGemmVariant(device: GPUDevice): Promise<GemmVariant> {
  if (pinnedVariant !== undefined) {
    setGemmVariant(pinnedVariant);
    return Promise.resolve(pinnedVariant);
  }
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
      const winner: GemmVariant = matrix !== undefined && bestClassicWide !== undefined
        && wideTime(matrix) * MATRIX_MARGIN < wideTime(bestClassicWide)
        ? { precision: "matrix", inner: classic.inner, fallback: classic.precision as
            "f32" | "f16-mixed" | "f16-chunked" }
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
