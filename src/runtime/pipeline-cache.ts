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

  get(key: string, code: string, entryPoint = "main"): Promise<GPUComputePipeline> {
    const cached = this.#pipelines.get(key);
    if (cached !== undefined) {
      if (cached.code !== code || cached.entryPoint !== entryPoint) {
        throw new Error(`WebGPU pipeline cache key collision for ${key}`);
      }
      return cached.pipeline;
    }
    const pipeline = this.device.createComputePipelineAsync({
        label: key,
        layout: "auto",
        compute: {
          module: this.device.createShaderModule({ label: `${key}.wgsl`, code }),
          entryPoint,
        },
      });
    this.#pipelines.set(key, { code, entryPoint, pipeline });
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
