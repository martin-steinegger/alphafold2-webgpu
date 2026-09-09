/** Copies a device buffer out as floats, for a caller that wants the values. */
export async function readFloat32(device: GPUDevice, buffer: GPUBuffer): Promise<Float32Array> {
  const readback = device.createBuffer({
    size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder({ label: "readback" });
    encoder.copyBufferToBuffer(buffer, 0, readback, 0, buffer.size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    return new Float32Array(readback.getMappedRange().slice(0));
  } finally {
    readback.destroy();
  }
}
