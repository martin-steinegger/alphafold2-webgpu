import type { GpuTensor } from "../runtime/execution.js";

/**
 * Returns an operation's scratch tensors to the allocator pool.
 *
 * Every Evoformer operation allocates full-size temporaries. Holding them for
 * a whole block means row attention's six MSA-sized tensors stay resident while
 * column attention allocates six more, so the block's peak is the sum of every
 * operation's scratch rather than the largest one.
 *
 * Recycling within one unsubmitted command buffer is safe for tensors written
 * only by dispatches: those dispatches are already encoded, and commands in a
 * compute pass observe each other's writes in order, so a later dispatch cannot
 * overwrite a buffer an earlier one still reads. Tensors filled by
 * queue.writeBuffer must NOT be recycled this way: a queue write is ordered at
 * the point it is issued, which is before this encoder is submitted, so it
 * would land ahead of the dispatches meant to read the old contents.
 */
export function releaseScratch(
  tensors: ReadonlyArray<GpuTensor | undefined>, keep: GpuTensor,
): void {
  for (const tensor of tensors) {
    if (tensor === undefined || tensor.allocation === keep.allocation) continue;
    tensor.allocation.release();
  }
}
