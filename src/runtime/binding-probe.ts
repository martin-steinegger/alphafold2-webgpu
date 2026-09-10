/**
 * Every storage binding a device makes, recorded as it makes it.
 *
 * WebGpuExecution can report its own bindings, but it does not make all of
 * them: the structure module and the confidence heads build their bind groups
 * against the device directly, so a report taken inside the execution misses
 * exactly the two stages that hold the largest single tensors. This sits under
 * all of them. Anything that reaches createBindGroup is seen, whoever built it.
 *
 * What it costs is the dispatch label: a bind group knows its buffers, not the
 * kernel that is about to read them. That trade is the right way round for the
 * question this answers, which is which tensor stops a long prediction rather
 * than which kernel touched it last. Buffers are labelled by the allocator, so
 * the tensor name survives.
 */

export interface ProbedBinding {
  readonly label: string;
  /** Bytes of the largest single binding seen under this label. */
  readonly bytes: number;
  /** How many bindings this label was seen in, which separates hot from rare. */
  readonly count: number;
}

export interface BindingProbe {
  /** Largest binding first, which is the order they become ceilings in. */
  readonly bindings: readonly ProbedBinding[];
  /** Puts the device's own createBindGroup back. */
  stop(): void;
}

/**
 * Records the bindings of every bind group this device creates from now on.
 *
 * Only buffer bindings are counted, and only their bound size, which is what
 * maxStorageBufferBindingSize applies to. A buffer bound in part reports the
 * part; the whole buffer is the memory report's business, not this one.
 */
export function probeBindings(device: GPUDevice, minimumBytes = 0): BindingProbe {
  const seen = new Map<string, { bytes: number; count: number }>();
  const original = device.createBindGroup.bind(device);
  const wrapper = (descriptor: GPUBindGroupDescriptor): GPUBindGroup => {
    for (const entry of descriptor.entries) {
      const resource = entry.resource as GPUBufferBinding | undefined;
      const buffer = resource?.buffer;
      if (buffer === undefined) continue;
      const bytes = resource?.size ?? (buffer.size - (resource?.offset ?? 0));
      if (bytes < minimumBytes) continue;
      // Trailing offsets distinguish windows of one tensor, not tensors.
      const label = (buffer.label === "" ? "unlabelled" : buffer.label).replace(/-?\d+$/, "");
      const previous = seen.get(label);
      seen.set(label, { bytes: Math.max(previous?.bytes ?? 0, bytes), count: (previous?.count ?? 0) + 1 });
    }
    return original(descriptor);
  };
  (device as { createBindGroup: GPUDevice["createBindGroup"] }).createBindGroup = wrapper;
  return {
    get bindings(): readonly ProbedBinding[] {
      return [...seen].map(([label, entry]) => ({ label, ...entry }))
        .sort((left, right) => right.bytes - left.bytes);
    },
    stop(): void {
      (device as { createBindGroup: GPUDevice["createBindGroup"] }).createBindGroup = original;
    },
  };
}

export function formatProbedBindings(bindings: readonly ProbedBinding[]): string {
  const mib = (bytes: number): string => `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return bindings.map((binding) => `${mib(binding.bytes).padStart(11)}`
    + ` x${String(binding.count).padStart(6)} ${String(binding.bytes).padStart(11)}`
    + `  ${binding.label}`).join("\n");
}
