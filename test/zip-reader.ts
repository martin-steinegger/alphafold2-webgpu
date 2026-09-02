import { inflateRawSync } from "node:zlib";

/**
 * Reads a ZIP archive the way an unzip implementation does, central directory
 * first, so a test sees what a user's extractor would see rather than what the
 * writer intended.
 */
export function readZipArchive(bytes: Uint8Array): Map<string, Uint8Array> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let end = bytes.length - 22;
  while (end >= 0 && view.getUint32(end, true) !== 0x06054b50) end -= 1;
  if (end < 0) throw new Error("no end-of-central-directory record");
  const count = view.getUint16(end + 10, true);
  let offset = view.getUint32(end + 16, true);
  const files = new Map<string, Uint8Array>();
  for (let index = 0; index < count; index += 1) {
    if (view.getUint32(offset, true) !== 0x02014b50) throw new Error("bad central directory record");
    const method = view.getUint16(offset + 10, true);
    const compressedSize = view.getUint32(offset + 20, true);
    const size = view.getUint32(offset + 24, true);
    const nameLength = view.getUint16(offset + 28, true);
    const extraLength = view.getUint16(offset + 30, true);
    const commentLength = view.getUint16(offset + 32, true);
    const local = view.getUint32(offset + 42, true);
    const name = new TextDecoder().decode(bytes.subarray(offset + 46, offset + 46 + nameLength));
    if (view.getUint32(local, true) !== 0x04034b50) throw new Error(`bad local header for ${name}`);
    const start = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true);
    const payload = bytes.subarray(start, start + compressedSize);
    const content = method === 8 ? new Uint8Array(inflateRawSync(payload)) : payload;
    if (content.length !== size) throw new Error(`${name} unpacked to ${content.length} of ${size} bytes`);
    files.set(name, content);
    offset += 46 + nameLength + extraLength + commentLength;
  }
  return files;
}

export const zipText = (files: Map<string, Uint8Array>, name: string): string => {
  const content = files.get(name);
  if (content === undefined) throw new Error(`${name} is not in the archive`);
  return new TextDecoder().decode(content);
};
