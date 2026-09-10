import { describe, expect, it } from "vitest";
import { formatProbedBindings, probeBindings } from "../src/runtime/binding-probe.js";

/** Enough of a device for the probe, which only wraps one method. */
const fakeDevice = () => {
  const made: GPUBindGroupDescriptor[] = [];
  const device = {
    createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup {
      made.push(descriptor); return { label: "" } as GPUBindGroup;
    },
  };
  return { device: device as unknown as GPUDevice, made };
};

const buffer = (label: string, size: number) => ({ label, size } as GPUBuffer);
const bind = (device: GPUDevice, entries: GPUBindGroupEntry[]) =>
  device.createBindGroup({ layout: {} as GPUBindGroupLayout, entries });

describe("what a device was seen to bind", () => {
  it("keeps the largest binding of a label and counts the rest", () => {
    const { device } = fakeDevice();
    const probe = probeBindings(device);
    bind(device, [{ binding: 0, resource: { buffer: buffer("pair", 4096), size: 1024 } }]);
    bind(device, [{ binding: 0, resource: { buffer: buffer("pair", 4096), size: 4096 } }]);
    bind(device, [{ binding: 0, resource: { buffer: buffer("pair", 4096), size: 2048 } }]);
    expect(probe.bindings).toEqual([{ label: "pair", bytes: 4096, count: 3 }]);
  });

  it("reads the bound part, not the buffer it is part of", () => {
    const { device } = fakeDevice();
    const probe = probeBindings(device);
    // A window of a tensor is what the limit applies to. Reporting the whole
    // buffer would say a windowed tensor is over the limit when it never is.
    bind(device, [{ binding: 0, resource: { buffer: buffer("residual", 1 << 20), offset: 0, size: 4096 } }]);
    expect(probe.bindings[0]?.bytes).toBe(4096);
  });

  it("falls back to what is left of the buffer where no size is given", () => {
    const { device } = fakeDevice();
    const probe = probeBindings(device);
    bind(device, [{ binding: 0, resource: { buffer: buffer("whole", 4096), offset: 1024 } }]);
    expect(probe.bindings[0]?.bytes).toBe(3072);
  });

  it("takes the window offset off the label, so one tensor is one row", () => {
    const { device } = fakeDevice();
    const probe = probeBindings(device);
    for (const offset of [0, 64, 128]) {
      bind(device, [{ binding: 0, resource: { buffer: buffer(`transition.first-${offset}`, 512) } }]);
    }
    expect(probe.bindings).toEqual([{ label: "transition.first", bytes: 512, count: 3 }]);
  });

  it("passes everything through and puts the method back", () => {
    const { device, made } = fakeDevice();
    const probe = probeBindings(device);
    bind(device, [{ binding: 0, resource: { buffer: buffer("a", 8) } }]);
    probe.stop();
    bind(device, [{ binding: 0, resource: { buffer: buffer("b", 8) } }]);
    expect(made).toHaveLength(2);
    expect(probe.bindings.map((entry) => entry.label)).toEqual(["a"]);
  });

  it("ignores bindings under the floor and samplers of any size", () => {
    const { device } = fakeDevice();
    const probe = probeBindings(device, 1024);
    bind(device, [
      { binding: 0, resource: { buffer: buffer("small", 16) } },
      { binding: 1, resource: { buffer: buffer("large", 4096) } },
      { binding: 2, resource: {} as GPUSampler },
    ]);
    expect(probe.bindings.map((entry) => entry.label)).toEqual(["large"]);
  });

  it("reports largest first, which is the order they become ceilings in", () => {
    const { device } = fakeDevice();
    const probe = probeBindings(device);
    bind(device, [
      { binding: 0, resource: { buffer: buffer("small", 16) } },
      { binding: 1, resource: { buffer: buffer("large", 1 << 21) } },
    ]);
    expect(probe.bindings.map((entry) => entry.label)).toEqual(["large", "small"]);
    expect(formatProbedBindings(probe.bindings).split("\n")[0]).toContain("2.0 MiB");
  });
});
