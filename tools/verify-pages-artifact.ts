import { readdir, stat } from "node:fs/promises";
import { resolve } from "node:path";

const directory = resolve(process.argv[2] ?? "dist/web");
const maximumBytes = 900 * 1024 ** 2;

async function directoryBytes(path: string): Promise<number> {
  let total = 0;
  for (const entry of await readdir(path, { withFileTypes: true })) {
    const child = resolve(path, entry.name);
    if (entry.isDirectory()) total += await directoryBytes(child);
    else if (entry.isFile()) total += (await stat(child)).size;
    else throw new Error(`Pages artifact contains unsupported entry ${child}`);
    if (!Number.isSafeInteger(total) || total > maximumBytes) {
      throw new Error(`Pages artifact exceeds the ${(maximumBytes / 1024 ** 2).toFixed(0)} MiB safety limit`);
    }
  }
  return total;
}

const bytes = await directoryBytes(directory);
console.log(`Validated Pages artifact size: ${(bytes / 1024 ** 2).toFixed(1)} MiB`);
