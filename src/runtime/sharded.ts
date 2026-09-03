/**
 * Binding a tensor larger than the device allows.
 *
 * WebGPU limits how much of a buffer one binding may cover, and a device only
 * gets the limit it asks for, which an adapter may refuse. The pair at 1500
 * residues is 549 MiB packed, well past the 128 MiB a device gets by default,
 * and the operations that read it by column cannot be given a window of rows.
 *
 * They are given several windows instead: disjoint ranges of the same buffer,
 * bound to consecutive slots, with a generated accessor that picks the range
 * an index falls in. One shard is the common case and costs nothing, since the
 * accessor collapses to the plain array access it replaces.
 */
import { type ActivationStorage, storageArray, storedElement } from "./storage.js";

/** How a tensor too large for one binding is spread over several. */
export interface ShardLayout {
  /** Bindings the tensor occupies; one when it fits. */
  readonly count: number;
  /** Storage elements in each shard but possibly the last. */
  readonly shardElements: number;
  readonly totalElements: number;
}

/** WebGPU's guaranteed storage-binding offset alignment. */
const BINDING_OFFSET_BYTES = 256;

function greatestCommonDivisor(a: number, b: number): number {
  return b === 0 ? a : greatestCommonDivisor(b, a % b);
}

/**
 * Splits `totalElements` so no shard exceeds the binding limit.
 *
 * Shards start on a multiple of `alignElements`, so a row never straddles two
 * of them and a consumer that reads whole rows stays inside one binding, and
 * on a 256-byte boundary, which is the offset alignment a binding must have.
 */
export function planShards(
  totalElements: number, alignElements: number, bindingBytes: number, bytesPerElement = 4,
): ShardLayout {
  if (![totalElements, alignElements, bindingBytes].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("shard planning needs positive sizes");
  }
  const offsetElements = BINDING_OFFSET_BYTES / bytesPerElement;
  const alignment = alignElements * offsetElements
    / greatestCommonDivisor(alignElements, offsetElements);
  const perBinding = Math.floor(bindingBytes / bytesPerElement / alignment) * alignment;
  if (perBinding <= 0) throw new RangeError("a single aligned row exceeds the binding limit");
  if (!Number.isSafeInteger(perBinding)) throw new RangeError("shard alignment is not a whole element count");
  if (totalElements <= perBinding) {
    return { count: 1, shardElements: totalElements, totalElements };
  }
  return { count: Math.ceil(totalElements / perBinding), shardElements: perBinding, totalElements };
}

/**
 * Windows of rows, each inside one binding and starting where a binding
 * offset may start.
 *
 * This is the other half of the same problem `planShards` solves. A kernel
 * that reads a tensor row by row can be given the rows it touches instead of
 * the whole tensor, which costs a uniform rather than a binding slot; one that
 * reads across the whole tensor cannot, and takes shards. Several bindings may
 * be cut from one window, so it holds a whole number of 256-byte boundaries of
 * every row size passed.
 */
export function rowWindows(
  rows: number, bindingBytes: number, rowByteSizes: readonly number[],
): readonly { readonly offset: number; readonly count: number }[] {
  let alignment = 1;
  for (const rowBytes of rowByteSizes) {
    const rowsPerBoundary = BINDING_OFFSET_BYTES / greatestCommonDivisor(rowBytes, BINDING_OFFSET_BYTES);
    alignment = alignment * rowsPerBoundary / greatestCommonDivisor(alignment, rowsPerBoundary);
  }
  const widest = Math.max(...rowByteSizes);
  const perBinding = Math.floor(bindingBytes / widest / alignment) * alignment;
  if (perBinding <= 0) throw new RangeError("an aligned window of rows does not fit one binding");
  if (rows <= perBinding) return [{ offset: 0, count: rows }];
  const windows: { offset: number; count: number }[] = [];
  for (let offset = 0; offset < rows; offset += perBinding) {
    windows.push({ offset, count: Math.min(perBinding, rows - offset) });
  }
  return windows;
}

/** WGSL binding declarations for one sharded tensor. */
export function shardBindings(
  layout: ShardLayout, name: string, storage: ActivationStorage, firstBinding: number, writable: boolean,
): string {
  const access = writable ? "read_write" : "read";
  return Array.from({ length: layout.count }, (_, index) =>
    `@group(0) @binding(${firstBinding + index}) var<storage, ${access}> ${name}_${index}: `
    + `array<${storageArray(storage)}>;`).join("\n");
}

/**
 * A WGSL function reading `name` at a whole-tensor element index.
 *
 * With one shard this is the array access itself; with several it is a chain
 * of comparisons, which a hot loop pays once per element read.
 */
export function shardLoader(layout: ShardLayout, name: string, storage: ActivationStorage): string {
  const upper = `${name.toUpperCase()}_SHARD`;
  if (layout.count === 1) {
    return `fn ${name}_load(index: u32) -> f32 { return ${storedElement(storage, `${name}_0`, "index")}; }`;
  }
  const branches = Array.from({ length: layout.count }, (_, index) => index === layout.count - 1
    ? `  return ${storedElement(storage, `${name}_${index}`, "local")};`
    : `  if (shard == ${index}u) { return ${storedElement(storage, `${name}_${index}`, "local")}; }`).join("\n");
  return `const ${upper}: u32 = ${layout.shardElements}u;
fn ${name}_load(index: u32) -> f32 {
  let shard = index / ${upper};
  let local = index - shard * ${upper};
${branches}
}`;
}

/**
 * A WGSL function writing `name`, by element for f32 and by packed word for
 * f16, where one word carries the two channels an invocation owns.
 */
export function shardStorer(layout: ShardLayout, name: string, storage: ActivationStorage): string {
  const packed = storage === "f16";
  const unit = packed ? Math.floor(layout.shardElements / 2) : layout.shardElements;
  const signature = packed
    ? `fn ${name}_store(word: u32, value: u32)` : `fn ${name}_store(index: u32, value: f32)`;
  const argument = packed ? "word" : "index";
  const upper = `${name.toUpperCase()}_STORE_SHARD`;
  if (layout.count === 1) return `${signature} { ${name}_0[${argument}] = value; }`;
  const branches = Array.from({ length: layout.count }, (_, index) => index === layout.count - 1
    ? `  ${name}_${index}[local] = value;`
    : `  if (shard == ${index}u) { ${name}_${index}[local] = value; return; }`).join("\n");
  return `const ${upper}: u32 = ${unit}u;
${signature} {
  let shard = ${argument} / ${upper};
  let local = ${argument} - shard * ${upper};
${branches}
}`;
}

/** Reads the value a packed store would overwrite, for a residual write. */
export function shardWordLoader(layout: ShardLayout, name: string): string {
  const upper = `${name.toUpperCase()}_STORE_SHARD`;
  if (layout.count === 1) return `fn ${name}_load_word(word: u32) -> u32 { return ${name}_0[word]; }`;
  const branches = Array.from({ length: layout.count }, (_, index) => index === layout.count - 1
    ? `  return ${name}_${index}[local];`
    : `  if (shard == ${index}u) { return ${name}_${index}[local]; }`).join("\n");
  return `fn ${name}_load_word(word: u32) -> u32 {
  let shard = word / ${upper};
  let local = word - shard * ${upper};
${branches}
}`;
}
