/**
 * Writes every shader the model generates, in both dialects, for naga to check.
 *
 * The wgpu half is the point: nothing else compiles it until a wgpu device
 * exists, so without this the wgpu emitter is unexercised.
 *
 * Usage: tsx tools/dump-shaders.ts [directory]
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { create, globals } from "webgpu";
import { DAWN_MATRIX, WGPU_MATRIX, dialect, presetDialect } from "../src/runtime/dialect.js";
import { dawnInstanceFlags } from "../src/runtime/dawn.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";
Object.assign(globalThis, globals);

const out = process.argv[2] ?? "/tmp/afwebgpu-shaders";
mkdirSync(out, { recursive: true });
const gpu = create(dawnInstanceFlags({ unclamped: true }));
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) throw new Error("no WebGPU adapter");
const device = await requestAlphaFoldDevice(adapter);

const written: string[] = [];
const emit = (name: string, code: string): void => {
  writeFileSync(`${out}/${name}.wgsl`, code);
  written.push(name);
};

const attention = await import("../src/evoformer/attention.js");
const attentionMatrix = await import("../src/evoformer/attention-matrix.js");
const reduction = await import("../src/runtime/reduction.js");
const gemm = await import("../src/runtime/gemm.js");
const transition = await import("../src/evoformer/transition.js");
const opm = await import("../src/evoformer/outer-product-mean.js");
const { subgroupMatrixConfigs } = await import("../src/runtime/subgroups.js");
const triangle = await import("../src/triangle/shaders.js");
const { ORDER } = await import("../src/triangle/weights.js");

/**
 * The triangle multiplication, which is the trunk's largest kernel and was
 * missing from this dump entirely.
 *
 * Its shape is the one a fold builds: 128 channels either side, blocked, with
 * the whole projection packed. The weight offsets only name constants in the
 * source, so zeroes serve.
 */
function emitTriangle(prefix: string, spelling?: import("../src/runtime/dialect.js").MatrixSpelling): void {
  const shape = { length: 256, cZ: 128, cHidden: 128 };
  const offsets = Object.fromEntries(ORDER.map((name) => [name, 0])) as never;
  for (const direction of ["outgoing", "incoming"] as const) {
    const shaders = triangle.createTriangleShaders(shape, "f16", offsets, 1e-5, direction,
      64, "f16", "f32", false, undefined, undefined, true, spelling);
    for (const [name, code] of Object.entries(shaders)) {
      emit(`${prefix}-triangle-${direction}-${name.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}`,
        code);
    }
  }
}

const GEMM_PROBE = {
  preamble: `@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
struct P { rows: u32, inner: u32, columns: u32, pad: u32 };
@group(0) @binding(3) var<uniform> p: P;`,
  rows: "p.rows", inner: "p.inner", columns: "p.columns",
  sourceElement: "a[row * p.inner + k]", weightElement: "b[k * p.columns + column]",
  store: "o[row * p.columns + column] = element;",
  sourceArray: { array: "a", stride: "p.inner" },
  weightArray: { array: "b", stride: "p.columns" },
};

/** The kernels that do not depend on the matrix spelling. */
function emitPortable(): void {
  for (const storage of ["f32", "f16"] as const) {
    emit(`attention-normalize-${storage}`, attention.createAttentionNormalizeShader(storage));
    emit(`attention-statistics-${storage}`, attention.createAttentionStatisticsShader(storage));
  }
  emit("gemm-f32", gemm.createTiledGemmShader(GEMM_PROBE, { precision: "f32", inner: 8 }));
  emit("gemm-f16", gemm.createTiledGemmShader(GEMM_PROBE, { precision: "f16", inner: 8 }));
  const tree = reduction.rowNormalizeLayout(undefined);
  emit("transition-tree", transition.createTransitionShaders({} as never, [], "f32", tree)[0]!);
  emit("opm-normalize-tree", opm.createOuterProductMeanNormalizeShader("f32", tree));
}

/** The kernels that carry an implementation's directives or matrix spelling. */
function emitMatrix(prefix: string, reductionDevice: GPUDevice): void {
  // The subgroup reductions carry the enable directive, which is the other half
  // of the dialect: Dawn needs it and naga rejects it.
  const layout = reduction.rowNormalizeLayout(reductionDevice);
  emit(`${prefix}-transition-subgroup`,
    transition.createTransitionShaders({} as never, [], "f32", layout)[0]!);
  emit(`${prefix}-opm-normalize-subgroup`,
    opm.createOuterProductMeanNormalizeShader("f32", layout));
  emit(`${prefix}-attention-project-f32`, attention.attentionProjectShader("f32"));
  emit(`${prefix}-attention-project-f16`, attention.attentionProjectShader("f16"));
  emit(`${prefix}-attention-project-wide`, attention.attentionProjectShader("f32", true));
  emit(`${prefix}-transition-linear`, transition.createLinearShader(false));
  emitTriangle(prefix, dialect(device).matrix);
  const matrix = { precision: "matrix", inner: 8 } as const;
  // The recorded list is untyped strings; the builders want the narrowed shape.
  for (const raw of subgroupMatrixConfigs(device)) {
    const componentType: "f16" | "f32" | undefined =
      raw.componentType === "f16" ? "f16" : raw.componentType === "f32" ? "f32" : undefined;
    if (componentType === undefined) continue;
    const config = {
      M: raw.M, N: raw.N, K: raw.K, componentType,
      resultComponentType: raw.resultComponentType === "f16" ? "f16" as const : "f32" as const,
    };
    const { M, N, K } = config;
    const name = `${componentType}${M}x${N}x${K}`;
    try {
      emit(`${prefix}-gemm-matrix-${name}`,
        gemm.createTiledGemmShader(GEMM_PROBE, { ...matrix, matrix: config }));
    } catch (error) { console.error(`  skipped ${prefix}-gemm-matrix-${name}: ${String(error)}`); }
    for (const headDim of [8, 32]) {
      try {
        emit(`${prefix}-flash-matrix-${name}-d${headDim}`,
          attentionMatrix.createAttentionMatrixFlashShader(headDim, config,
            dialect(device).matrix!));
      } catch { /* no matrix kernel at this width and shape */ }
    }
  }
}

emitPortable();
const settled = dialect(device);
presetDialect(device, { ...settled, matrix: DAWN_MATRIX });
emitMatrix("dawn", device);
// What wgpu looks like: subgroups without the directive and without
// subgroup-size-control, so the reduction takes its workgroup path.
const wgpuLike = { features: new Set(["subgroups", "shader-f16"]) } as unknown as GPUDevice;
presetDialect(wgpuLike, { subgroupEnable: "", subgroupSize: () => "", matrix: WGPU_MATRIX });
presetDialect(device, { subgroupEnable: "", subgroupSize: () => "", matrix: WGPU_MATRIX });
emitMatrix("wgpu", wgpuLike);
presetDialect(device, settled);

console.log(`wrote ${written.length} shaders to ${out}`);
device.destroy();
