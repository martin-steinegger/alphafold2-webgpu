/**
 * Compiles every shader the model can generate.
 *
 * A WGSL module that fails to parse is reported as "invalid due to a previous
 * error" by whichever pipeline happens to be created next, so a broken shader
 * can look like a fault in an unrelated kernel, and one that only appears in a
 * rare configuration can ship unnoticed. This walks the exports of every
 * module that holds a shader, compiles the constants and the generators over
 * their storage and sharding options, and fails on any compilation message.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import { planShards, type ShardLayout } from "../src/runtime/sharded.js";
import { createTriangleShaders } from "../src/triangle/shaders.js";
import { packWeights } from "../src/triangle/weights.js";
import type { TriangleMultiplicationWeights } from "../src/triangle/types.js";
import type { ActivationStorage } from "../src/runtime/storage.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";

const MODULES = [
  "../src/evoformer/attention.js", "../src/evoformer/block.js",
  "../src/evoformer/input-embedder.js", "../src/evoformer/multimer-relative.js",
  "../src/evoformer/multimer-template.js", "../src/evoformer/outer-product-mean.js",
  "../src/evoformer/template.js", "../src/evoformer/transition.js",
  "../src/heads/confidence.js", "../src/runtime/elementwise.js",
  "../src/runtime/execution.js", "../src/runtime/gemm.js",
  "../src/structure/geometry.js", "../src/structure/initialize.js",
  "../src/structure/ipa.js", "../src/structure/iteration.js",
  "../src/structure/sidechain.js", "../src/triangle/shaders.js",
] as const;

const STORAGES: readonly ActivationStorage[] = ["f32", "f16"];
/** One binding, and three that a whole tensor is spread over. */
const LAYOUTS: readonly ShardLayout[] = [
  { count: 1, shardElements: Number.MAX_SAFE_INTEGER, totalElements: 0 },
  planShards(3 * 64, 64, 64 * 4, 4),
];

/** The parameter list of a function, as written, with types already stripped. */
function parameterNames(generator: Function): string[] | undefined {
  const source = generator.toString();
  const open = source.indexOf("(");
  if (open < 0) return undefined;
  let depth = 0;
  let close = -1;
  for (let index = open; index < source.length; index += 1) {
    const character = source[index];
    if (character === "(" || character === "[" || character === "{") depth += 1;
    if (character === ")" || character === "]" || character === "}") { depth -= 1; if (depth === 0) { close = index; break; } }
  }
  if (close < 0) return undefined;
  const list = source.slice(open + 1, close).trim();
  if (list === "") return [];
  const parameters: string[] = [];
  let level = 0;
  let current = "";
  for (const character of list) {
    if ("([{".includes(character)) level += 1;
    if (")]}".includes(character)) level -= 1;
    if (character === "," && level === 0) { parameters.push(current); current = ""; continue; }
    current += character;
  }
  parameters.push(current);
  return parameters.map((parameter) => parameter.trim());
}

/**
 * The values a parameter may take, by name.
 *
 * A parameter this does not recognise must carry a default, or the generator
 * is skipped: guessing a value produces a shader the model never builds, and
 * its errors would be noise.
 */
function parameterValues(parameter: string): readonly unknown[] | undefined {
  const [declaration = "", ...rest] = parameter.split("=");
  const name = declaration.trim();
  const lower = name.toLowerCase();
  if (lower.includes("shard")) return LAYOUTS;
  if (lower.includes("storage")) return STORAGES;
  if (lower.includes("headdim")) return [4, 32];
  const literal = rest.join("=").trim();
  // A default this does not recognise stands, along with every later default.
  if (literal !== "") return literal === "false" || literal === "true" ? [false, true] : [];
  return undefined;
}

/** Calls a generator over every combination its parameters accept. */
function generatorSources(name: string, generator: Function): [string, string][] {
  const parameters = parameterNames(generator);
  if (parameters === undefined) return [];
  let combinations: unknown[][] = [[]];
  for (const parameter of parameters) {
    const values = parameterValues(parameter);
    if (values === undefined) return [];
    if (values.length === 0) break; // Keep this default and every later one.
    combinations = combinations.flatMap((prefix) => values.map((value) => [...prefix, value]));
  }
  const sources: [string, string][] = [];
  for (const args of combinations) {
    const produced = (generator as (...a: unknown[]) => unknown)(...args);
    if (typeof produced !== "string" || !produced.includes("@compute")) continue;
    const label = args.map((value) => typeof value === "object" && value !== null
      ? `shards=${(value as ShardLayout).count}` : String(value)).join(",");
    sources.push([`${name}(${label})`, produced]);
  }
  return sources;
}

/**
 * The triangle shaders, over the options a prediction can reach.
 *
 * They take a configuration rather than flags, so they are listed by hand;
 * they are also the ones the pair's sharding runs through, which is where a
 * generated accessor is most likely to go wrong.
 */
function triangleSources(): [string, string][] {
  const shape = { length: 8, cZ: 4, cHidden: 4 } as const;
  const lengths = {
    layerNormInWeight: shape.cZ, layerNormInBias: shape.cZ,
    linearAPWeight: shape.cZ * shape.cHidden, linearAPBias: shape.cHidden,
    linearAGWeight: shape.cZ * shape.cHidden, linearAGBias: shape.cHidden,
    linearBPWeight: shape.cZ * shape.cHidden, linearBPBias: shape.cHidden,
    linearBGWeight: shape.cZ * shape.cHidden, linearBGBias: shape.cHidden,
    layerNormOutWeight: shape.cHidden, layerNormOutBias: shape.cHidden,
    linearZWeight: shape.cHidden * shape.cZ, linearZBias: shape.cZ,
    linearGWeight: shape.cZ * shape.cZ, linearGBias: shape.cZ,
  };
  const weights = Object.fromEntries(Object.entries(lengths)
    .map(([name, length]) => [name, new Float32Array(length)])) as unknown as TriangleMultiplicationWeights;
  const { offsets } = packWeights(weights, "f32");
  const pairElements = shape.length * shape.length * shape.cZ;
  const sources: [string, string][] = [];
  for (const direction of ["outgoing", "incoming"] as const) {
    for (const wholeStorage of ["f32", "f16"] as const) {
      for (const pairStorage of STORAGES) {
        for (const residual of [false, true]) {
          for (const shards of [1, 3]) {
            const layout = planShards(pairElements, shape.cZ,
              shards === 1 ? Number.MAX_SAFE_INTEGER : Math.ceil(pairElements / shards) * 4,
              pairStorage === "f16" ? 2 : 4);
            const pairs = shape.length * shape.length;
            const wholeElements = (pairs + pairs % 2) * shape.cHidden;
            const wholeLayout = planShards(wholeElements, 2,
              shards === 1 ? Number.MAX_SAFE_INTEGER : Math.ceil(wholeElements / shards) * 4,
              wholeStorage === "f16" ? 2 : 4);
            const label = `triangle(${direction},whole=${wholeStorage},pair=${pairStorage},`
              + `residual=${residual},shards=${layout.count}/${wholeLayout.count})`;
            const shaders = createTriangleShaders(shape, "f32", offsets, 1e-5, direction, 4,
              wholeStorage, pairStorage, residual, layout, wholeLayout);
            for (const [name, code] of Object.entries(shaders)) {
              if (typeof code === "string" && code.includes("@compute")) sources.push([`${label}.${name}`, code]);
            }
          }
        }
      }
    }
  }
  return sources;
}

describe.skipIf(!enabled)("every generated shader compiles", () => {
  let gpu: ReturnType<typeof create>;
  let device: GPUDevice;
  let subgroups = false;

  beforeAll(async () => {
    Object.assign(globalThis, globals);
    gpu = create([]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter");
    // The subgroup kernels need the feature enabled or they will not parse.
    const wanted = ["subgroups", "subgroup-size-control"];
    const features = wanted.filter((feature) => adapter.features.has(feature)) as GPUFeatureName[];
    subgroups = features.length === wanted.length;
    device = await adapter.requestDevice({ requiredFeatures: features });
  });

  afterAll(() => { device?.destroy(); });

  it("reports no compilation message for any module or option", async () => {
    const failures: string[] = [];
    let compiled = 0;
    const compile = async (name: string, code: string): Promise<void> => {
      compiled += 1;
      const info = await device.createShaderModule({ label: name, code }).getCompilationInfo();
      for (const message of info.messages) {
        failures.push(`${name}: ${message.type} line ${message.lineNum}: ${message.message}`);
      }
    };
    for (const [name, code] of triangleSources()) await compile(name, code);
    for (const specifier of MODULES) {
      const module = (await import(specifier)) as Record<string, unknown>;
      const sources: [string, string][] = [];
      for (const [name, value] of Object.entries(module)) {
        if (typeof value === "string" && value.includes("@compute")) {
          if (!subgroups && value.includes("enable subgroups")) continue;
          sources.push([name, value]);
        }
        if (typeof value === "function" && /^create[A-Za-z]*Shader$/.test(name)) {
          sources.push(...generatorSources(name, value as Function));
        }
      }
      for (const [name, code] of sources) await compile(`${specifier} ${name}`, code);
    }
    expect(compiled).toBeGreaterThan(200);
    expect(failures).toEqual([]);
  }, 300_000);
});
