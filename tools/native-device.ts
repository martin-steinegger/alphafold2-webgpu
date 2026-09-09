/**
 * What a native host can find out about its GPU that WebGPU will not tell it.
 *
 * `adapter.info` carries the vendor, the architecture and the subgroup range,
 * but no memory heaps: Dawn only reports those under a feature this build does
 * not expose, and `maxBufferSize` is the API's theoretical maximum rather than
 * the card's. A browser has to plan without it. A native host does not.
 */
import { execFileSync } from "node:child_process";
import { setCalibrationStore } from "../src/runtime/calibration-store.js";
import { mkdirSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { readdirSync, readFileSync } from "node:fs";
import { totalmem } from "node:os";

/**
 * Points the Vulkan loader at one GPU, by index, before an instance exists.
 *
 * WebGPU cannot choose a device: `requestAdapter` takes a power preference and
 * nothing else, and dawn.node's `adapter=<name>` matches a substring of the
 * device name and takes the first hit — which on a host of identical cards is
 * always the first one. Every adapter here reports the same vendor,
 * architecture, device and description, so there is nothing to match on.
 *
 * Mesa's device-select layer can, and it is already installed wherever Mesa is:
 * it is a GLOBAL implicit layer, so it sits above every driver including
 * Nvidia's own, and `DRI_PRIME` names a card by its PCI address rather than by
 * a name that repeats. The loader reads the variable when the instance is
 * made, so setting it here works as long as it happens before `create`.
 *
 * Returns the tag it set, or undefined when the host cannot say where the card
 * is, in which case the caller gets whichever device the loader prefers.
 */
export function selectGpu(index = cudaVisibleDeviceIndex()): string | undefined {
  if (index === undefined) return undefined;
  if (!Number.isSafeInteger(index) || index < 0) {
    throw new RangeError(`a GPU index must be a non-negative integer, not ${index}`);
  }
  const bus = queryGpu(["--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", String(index)]);
  if (bus === undefined) return undefined;
  // nvidia-smi writes 00000000:95:00.0; the layer wants pci-0000_95_00_0.
  const match = /^([0-9a-f]+):([0-9a-f]+):([0-9a-f]+)\.([0-9a-f]+)$/.exec(bus.trim().toLowerCase());
  if (match === null) return undefined;
  const domain = Number.parseInt(match[1]!, 16).toString(16).padStart(4, "0");
  const tag = `pci-${domain}_${match[2]}_${match[3]}_${match[4]}`;
  process.env.DRI_PRIME = tag;
  selected = index;
  return tag;
}

/** The index `selectGpu` chose, so the memory probe can ask about that card. */
let selected: number | undefined;

/**
 * The card `CUDA_VISIBLE_DEVICES` names, which is what people reach for.
 *
 * Nothing here uses CUDA, but that variable is how a GPU gets chosen on a
 * shared host and it costs nothing to honour. The first entry wins, since this
 * runs on one card. A plain number is an index as `nvidia-smi` counts them; a
 * `GPU-...` UUID is resolved to one, and is the form to prefer, because CUDA
 * numbers devices fastest-first unless `CUDA_DEVICE_ORDER=PCI_BUS_ID` says
 * otherwise while `nvidia-smi` always counts by bus.
 */
function cudaVisibleDeviceIndex(): number | undefined {
  const value = process.env.CUDA_VISIBLE_DEVICES?.split(",")[0]?.trim();
  if (value === undefined || value === "") return undefined;
  if (/^\d+$/.test(value)) return Number(value);
  const listed = queryGpu(["--query-gpu=index,uuid", "--format=csv,noheader"]);
  if (listed === undefined) return undefined;
  for (const line of listed.split("\n")) {
    const [index, uuid] = line.split(",").map((part) => part.trim());
    if (uuid !== undefined && uuid === value) return Number(index);
  }
  return undefined;
}

function queryGpu(args: readonly string[]): string | undefined {
  try {
    return execFileSync("nvidia-smi", [...args],
      { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], timeout: 5_000 });
  } catch { return undefined; }
}

/**
 * Total memory of the selected GPU, or of the smallest visible one.
 *
 * Once `selectGpu` has chosen, this asks about that card. Without a choice it
 * takes the smallest visible one, because nothing says which the loader
 * preferred and the smallest is the answer that is right whichever it was.
 */
export function detectDeviceMemoryBytes(): number | undefined {
    const out = queryGpu(selected === undefined
      ? ["--query-gpu=memory.total", "--format=csv,noheader,nounits"]
      : ["--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", String(selected)]);
    if (out === undefined) return amdDeviceMemoryBytes();
    const sizes = out.split("\n").map((line) => Number(line.trim()))
      .filter((value) => Number.isFinite(value) && value > 0);
    if (sizes.length === 0) return amdDeviceMemoryBytes();
    return Math.min(...sizes) * 1024 * 1024;
}

/**
 * The same figure for an AMD card, which has no `nvidia-smi` to ask.
 *
 * The kernel reports it per card in sysfs, so nothing has to be installed and
 * `rocm-smi` need not be present. The smallest is taken for the reason above:
 * without a choice, nothing says which card the loader preferred.
 *
 * An integrated part reports a heap it does not have — the Strix Halo here
 * says 96 GiB on a host holding 32 GiB, because that heap is carved out of
 * system memory on demand — so the host's own memory caps it. That keeps the
 * budget honest on an APU and changes nothing on a discrete card, whose board
 * memory is the smaller of the two.
 */
function amdDeviceMemoryBytes(): number | undefined {
  const heap = amdVramHeapBytes();
  return heap === undefined ? undefined : Math.min(heap, totalmem());
}

function amdVramHeapBytes(): number | undefined {
  let cards: readonly string[];
  // Read rather than globbed: every entry on the way to the file is a symlink
  // into the PCI tree, which a glob will not walk through.
  try { cards = readdirSync("/sys/class/drm"); } catch { return undefined; }
  let smallest: number | undefined;
  for (const card of cards) {
    if (!/^card\d+$/.test(card)) continue;
    try {
      const bytes = Number(
        readFileSync(`/sys/class/drm/${card}/device/mem_info_vram_total`, "utf8").trim(),
      );
      if (!Number.isFinite(bytes) || bytes <= 0) continue;
      if (smallest === undefined || bytes < smallest) smallest = bytes;
    } catch { /* a card that will not say is one this cannot plan against */ }
  }
  return smallest;
}

/**
 * Whether the card's memory is the host's memory.
 *
 * An integrated part reports a heap larger than the machine holds, because
 * that heap is carved out of system memory on demand. Nothing else does, so
 * the comparison identifies one without asking the driver what it is.
 */
function sharesHostMemory(): boolean {
  const heap = amdVramHeapBytes();
  return heap !== undefined && heap > totalmem();
}

/**
 * Bytes a prediction may plan against, from the memory the host reports.
 *
 * Two thirds, so the driver, the window system and anything else sharing the
 * card are not squeezed out. A host that reports nothing gets undefined and
 * the caller keeps the browser's own conservative budget.
 */
export function nativeMemoryBudgetBytes(): number | undefined {
  // A card that shares the host's memory is not one to spend it on. The wider
  // scratch windows buy their speed by keeping more of a pass resident, which
  // pays where board memory is separate and fast, and does not where it is the
  // same DRAM the passes already stream through: measured on a Strix Halo at
  // 1.60 s a recycle on the browser's own budget against 2.07 s at four times
  // it and 1.94 s at sixteen. So the host reports what it has and declines to
  // plan against it.
  if (sharesHostMemory()) return undefined;
  const total = detectDeviceMemoryBytes();
  return total === undefined ? undefined : Math.floor(total * 2 / 3);
}

/**
 * Keeps the same document in a file beside the manifest, node having no
 * `localStorage` without `--localstorage-file`. `AFWEBGPU_CALIBRATION`
 * overrides the path.
 *
 * Temporary name then rename, so a reader never sees a half-written file. NOT
 * serialised: two writers racing can lose one of the answers, which costs a
 * recalibration and avoids a lock file to clean up after a crash.
 */
export function useFileCalibrationStore(manifestPath?: string): string | undefined {
  const named = process.env.AFWEBGPU_CALIBRATION;
  const manifest = manifestPath ?? process.env.AFWEBGPU_MANIFEST;
  const path = named ?? (manifest === undefined
    ? undefined : join(dirname(manifest), "calibration.json"));
  if (path === undefined) return undefined;
  setCalibrationStore({
    read: () => {
      try { return readFileSync(path, "utf8"); } catch { return undefined; }
    },
    write: (document) => {
      try {
        mkdirSync(dirname(path), { recursive: true });
        const temporary = `${path}.${process.pid}.tmp`;
        writeFileSync(temporary, document);
        renameSync(temporary, path);
      } catch { /* read-only install, or no room. It measures next time. */ }
    },
  });
  return path;
}
