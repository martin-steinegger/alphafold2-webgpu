/**
 * What a native host can find out about its GPU that WebGPU will not tell it.
 *
 * `adapter.info` carries the vendor, the architecture and the subgroup range,
 * but no memory heaps: Dawn only reports those under a feature this build does
 * not expose, and `maxBufferSize` is the API's theoretical maximum rather than
 * the card's. A browser has to plan without it. A native host does not.
 */
import { execFileSync } from "node:child_process";

/**
 * Total memory of the smallest visible GPU, or undefined when nothing answers.
 *
 * The smallest, because WebGPU does not say which adapter Dawn picked and this
 * cannot ask: `create(["adapter=<name>"])` matches a substring of the device
 * name and takes the first match, which on a host of identical cards is always
 * the first one. Sizing to the smallest is the answer that is right whichever
 * it chose.
 */
export function detectDeviceMemoryBytes(): number | undefined {
  try {
    const out = execFileSync("nvidia-smi",
      ["--query-gpu=memory.total", "--format=csv,noheader,nounits"],
      { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], timeout: 5_000 });
    const sizes = out.split("\n").map((line) => Number(line.trim()))
      .filter((value) => Number.isFinite(value) && value > 0);
    if (sizes.length === 0) return undefined;
    return Math.min(...sizes) * 1024 * 1024;
  } catch { return undefined; }
}

/**
 * Bytes a prediction may plan against, from the memory the host reports.
 *
 * Two thirds, so the driver, the window system and anything else sharing the
 * card are not squeezed out. A host that reports nothing gets undefined and
 * the caller keeps the browser's own conservative budget.
 */
export function nativeMemoryBudgetBytes(): number | undefined {
  const total = detectDeviceMemoryBytes();
  return total === undefined ? undefined : Math.floor(total * 2 / 3);
}
