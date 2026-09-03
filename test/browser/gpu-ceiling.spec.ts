import { expect, test } from "@playwright/test";
test.skip(process.env.AFWEBGPU_GPU_CEILING !== "1", "set AFWEBGPU_GPU_CEILING=1");
test("finds the practical WebGPU allocation ceiling", async ({ page }) => {
  test.setTimeout(300_000);
  await page.goto("/");
  const report = await page.evaluate(async () => {
    const mib = 1024 ** 2;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) return { error: "no adapter" };
    const wanted = Math.min(adapter.limits.maxBufferSize, 2048 * mib);
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: wanted,
        maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, wanted),
      },
    });
    const lost = device.lost.then((info) => info.reason ?? "lost");
    const held: GPUBuffer[] = [];
    let allocatedMiB = 0;
    let failure = "";
    try {
      for (let step = 0; step < 32; step += 1) {
        const buffer = device.createBuffer({ size: 256 * mib, usage: GPUBufferUsage.STORAGE });
        device.queue.writeBuffer(buffer, 0, new Uint8Array(1024));
        await device.queue.onSubmittedWorkDone();
        held.push(buffer);
        allocatedMiB += 256;
      }
    } catch (error) {
      failure = error instanceof Error ? error.message : String(error);
    }
    const raced = await Promise.race([lost, new Promise((resolve) => setTimeout(() => resolve("alive"), 500))]);
    for (const buffer of held) buffer.destroy();
    return {
      adapterBufferMiB: adapter.limits.maxBufferSize / mib,
      adapterBindingMiB: adapter.limits.maxStorageBufferBindingSize / mib,
      grantedBufferMiB: device.limits.maxBufferSize / mib,
      grantedBindingMiB: device.limits.maxStorageBufferBindingSize / mib,
      storageBuffersPerStage: device.limits.maxStorageBuffersPerShaderStage,
      workgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
      allocatedMiB, failure, deviceState: raced,
      reportedMemoryGB: (navigator as { deviceMemory?: number }).deviceMemory,
    };
  });
  console.log("CEILING:", JSON.stringify(report));
  expect(report).toBeTruthy();
});
