export class ComputePipelineCache {
  readonly device: GPUDevice;
  readonly #pipelines = new Map<string, {
    readonly code: string;
    readonly entryPoint: string;
    readonly pipeline: Promise<GPUComputePipeline>;
  }>();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * The pipeline for `key`, compiling it from `code` if it is not held.
   *
   * `code` may be a thunk, and where the caller can pass one it should. A
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
  get(key: string, code: string | (() => string), entryPoint = "main"): Promise<GPUComputePipeline> {
    const cached = this.#pipelines.get(key);
    if (cached !== undefined) {
      if (typeof code === "string"
        && (cached.code !== code || cached.entryPoint !== entryPoint)) {
        throw new Error(`WebGPU pipeline cache key collision for ${key}`);
      }
      return cached.pipeline;
    }
    const source = typeof code === "string" ? code : code();
    const pipeline = this.device.createComputePipelineAsync({
        label: key,
        layout: "auto",
        compute: {
          module: this.device.createShaderModule({ label: `${key}.wgsl`, code: source }),
          entryPoint,
        },
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
