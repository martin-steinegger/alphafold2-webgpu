import { timedSync } from "./phase-ledger.js";

const OVERRIDE = /^\s*override\s+(\w+)\s*(?::\s*\w+)?\s*(?:=\s*([^;]*))?;/gm;

/**
 * The overrides a WGSL source uses, so a pipeline is given only those.
 *
 * WebGPU lets a pipeline supply a value for an override its entry point does
 * not use, and Dawn and naga ignore it. Safari 26 does not: WebKit's
 * createLibrary fails the pipeline, with only "Compute library failed
 * creation" to say so, for any constant the entry point does not use. The
 * triangle kernels share one preamble of five overrides and each uses some of
 * them, which failed every prediction in Safari from d66fc10 on.
 *
 * An override is used when its name is anywhere in the source outside the
 * override declarations, or in the initializer of one that is used. This keeps
 * too many rather than too few: an override that is used but dropped would
 * take its default, which is a wrong answer where a kept unused one is only a
 * failure in Safari.
 */
export function usedOverrides(source: string): ReadonlySet<string> {
  const code = source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
  const initializers = new Map([...code.matchAll(OVERRIDE)].map((match) => [match[1]!, match[2] ?? ""]));
  const words = (text: string): string[] => text.match(/\b\w+\b/g) ?? [];
  const used = new Set(words(code.replace(OVERRIDE, "")).filter((word) => initializers.has(word)));
  for (const name of used) {
    for (const word of words(initializers.get(name)!)) if (initializers.has(word)) used.add(word);
  }
  return used;
}

export class ComputePipelineCache {
  readonly device: GPUDevice;
  /** Modules by source, so an override-only difference costs no compile. */
  readonly #modules = new Map<string, { module: GPUShaderModule; overrides: ReadonlySet<string> }>();
  readonly #pipelines = new Map<string, {
    readonly code: string;
    readonly entryPoint: string;
    readonly pipeline: Promise<GPUComputePipeline>;
  }>();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * The pipeline for key, compiling it from code if it is not held.
   *
   * code may be a thunk, and where the caller can pass one it should. A
   * shader source is built by string concatenation from the shape and the
   * storage choices, and a block asks for the same pipelines on every block of
   * every recycle: one fold at 256 residues took 7,707 pipelines from this
   * cache and generated 125.7 MB of source for them that was thrown away
   * unread. A thunk is not called at all on a hit.
   *
   * The collision check still runs for a caller that passes a string, and
   * cannot for one that passes a thunk -- checking would mean generating the
   * source, which is the cost being avoided. A thunk therefore trades that
   * check for the saving, so give a key that names everything the source
   * depends on.
   */
  /**
   * constants are WGSL override values. A module is cached by its source,
   * so kernels differing only in an override share one and each length costs a
   * pipeline rather than a compile.
   */
  get(
    key: string, code: string | (() => string), entryPoint = "main",
    constants?: Record<string, number>,
  ): Promise<GPUComputePipeline> {
    const cached = this.#pipelines.get(key);
    if (cached !== undefined) {
      if (typeof code === "string"
        && (cached.code !== code || cached.entryPoint !== entryPoint)) {
        throw new Error(`WebGPU pipeline cache key collision for ${key}`);
      }
      return cached.pipeline;
    }
    const source = timedSync("shader source", () => typeof code === "string" ? code : code());
    let compiled = this.#modules.get(source);
    if (compiled === undefined) {
      compiled = timedSync("shader module", () => ({
        module: this.device.createShaderModule({ label: `${key}.wgsl`, code: source }),
        overrides: usedOverrides(source),
      }));
      this.#modules.set(source, compiled);
    }
    const { module, overrides } = compiled;
    const given = Object.entries(constants ?? {}).filter(([name]) => overrides.has(name));
    // A failure names the pipeline and is not kept. WebKit reports a shader it
    // cannot compile as "Compute library failed creation" and nothing else,
    // which says neither which of ninety kernels it was nor why; and a device
    // held across predictions would otherwise hand the same rejected promise
    // to every later one, so one bad compile fails the page until it reloads.
    const pipeline = this.device.createComputePipelineAsync({
      label: key,
      layout: "auto",
      compute: { module, entryPoint, ...(given.length === 0 ? {} : { constants: Object.fromEntries(given) }) },
    }).catch(async (error: unknown) => {
      this.#pipelines.delete(key);
      const messages = (await module.getCompilationInfo().catch(() => undefined))?.messages
        .map((message) => `${message.type} at ${message.lineNum}:${message.linePos}: ${message.message}`) ?? [];
      throw new Error(`Pipeline ${key} failed: ${error instanceof Error ? error.message : String(error)}`
        + (messages.length === 0 ? "" : ` (${messages.join("; ")})`), { cause: error });
    });
    this.#pipelines.set(key, { code: source, entryPoint, pipeline });
    return pipeline;
  }

  get size(): number {
    return this.#pipelines.size;
  }
}

const DEVICE_PIPELINE_CACHES = new WeakMap<GPUDevice, ComputePipelineCache>();

/**
 * Returns the pipeline cache owned by a device.
 *
 * AlphaFold executes the same kernels in every block and recycle. Keeping this
 * cache at device lifetime avoids asking the browser to recreate identical
 * compute pipelines whenever a short-lived operator/execution object is made.
 */
export function pipelineCacheForDevice(device: GPUDevice): ComputePipelineCache {
  let cache = DEVICE_PIPELINE_CACHES.get(device);
  if (cache === undefined) {
    cache = new ComputePipelineCache(device);
    DEVICE_PIPELINE_CACHES.set(device, cache);
  }
  return cache;
}
