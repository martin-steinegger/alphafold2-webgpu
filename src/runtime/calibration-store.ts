/**
 * Calibration remembered between visits.
 *
 * Choosing the projection kernel costs 662-764 ms of every cold start, of a
 * requestAlphaFoldDevice that takes 807-864, because it compiles the
 * candidates and times them. Measure once, keep the answer.
 *
 * One JSON document of key to answer, the same in a browser and on a host: a
 * store reads and writes the whole thing, so IndexedDB holds one record and
 * node holds one file with identical contents.
 *
 * A store has to be installed, by useBrowserCalibrationStore or by the node
 * file store. Without one nothing is kept and every start measures, which is
 * what a browser did for as long as this reached for localStorage: the page
 * folds in a Web Worker and localStorage belongs to Window.
 */

/** Bumped when a stored answer could no longer be trusted. */
const SCHEMA = 3;
const DOCUMENT = "afwebgpu.calibration";

export interface CalibrationStore {
  read(): string | undefined;
  write(document: string): void;
}

const DATABASE = "afwebgpu";
const STORE = "calibration";

function openDatabase(): Promise<IDBDatabase | undefined> {
  return new Promise((resolve) => {
    const factory = (globalThis as { indexedDB?: IDBFactory }).indexedDB;
    if (factory === undefined) { resolve(undefined); return; }
    let request: IDBOpenDBRequest;
    try { request = factory.open(DATABASE, 1); } catch { resolve(undefined); return; }
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE)) request.result.createObjectStore(STORE);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => resolve(undefined);
    request.onblocked = () => resolve(undefined);
  });
}

/**
 * Installs the browser store, having loaded what it holds. Await it before the
 * device is built, since building the device is what calibrates.
 *
 * IndexedDB rather than localStorage because a worker has one and not the
 * other. It is asynchronous and a store is not, so the document is read once
 * into memory here and writes go back in the background: one that never lands
 * costs the next visit a measurement and nothing else.
 *
 * False where there is no IndexedDB, or a browser refuses it, leaving whatever
 * store was installed before and measuring as it did.
 */
export async function useBrowserCalibrationStore(): Promise<boolean> {
  const database = await openDatabase();
  if (database === undefined) return false;
  let current = await new Promise<string | undefined>((resolve) => {
    try {
      const request = database.transaction(STORE, "readonly").objectStore(STORE).get(DOCUMENT);
      request.onsuccess = () => resolve(typeof request.result === "string" ? request.result : undefined);
      request.onerror = () => resolve(undefined);
    } catch { resolve(undefined); }
  });
  setCalibrationStore({
    read: () => current,
    write: (value) => {
      current = value;
      try {
        database.transaction(STORE, "readwrite").objectStore(STORE).put(value, DOCUMENT);
      } catch { /* evicted, or the database closed under us */ }
    },
  });
  return true;
}

let installed: CalibrationStore | undefined;

/** Installs a store; undefined turns remembering off. */
export function setCalibrationStore(store: CalibrationStore | undefined): void {
  installed = store;
}

function load(): Record<string, unknown> {
  try {
    const raw = installed?.read();
    if (raw === undefined) return {};
    const parsed: unknown = JSON.parse(raw);
    return typeof parsed === "object" && parsed !== null
      ? parsed as Record<string, unknown> : {};
  } catch { return {}; }
}

/**
 * What identifies a device well enough to reuse an answer measured on it.
 *
 * description carries the driver version, so an update asks a new question.
 * scope names the model where the answer depends on it: a pair track 128
 * channels wide can want a different kernel from one 384 wide. Required, so
 * that a caller says which it is rather than defaulting into a shared key.
 */
export function calibrationKey(
  adapter: GPUAdapter | undefined, device: GPUDevice | undefined,
  configs: readonly { M: number; N: number; K: number; componentType: string }[],
  scope: string,
): string {
  const info = (adapter as { info?: Record<string, string> } | undefined)?.info
    ?? (device as { adapterInfo?: Record<string, string> } | undefined)?.adapterInfo;
  const identity = ["vendor", "architecture", "device", "description"]
    .map((field) => info?.[field] ?? "").join("/");
  const features = [...(device?.features ?? [])].sort().join(",");
  const shapes = configs.map((c) => `${c.M}x${c.N}x${c.K}:${c.componentType}`).sort().join(",");
  return `v${SCHEMA}|${identity}|${features}|${shapes}|${scope}`;
}

/** A remembered answer, validated rather than trusted. */
export function readCalibration<T>(
  key: string, valid: (value: unknown) => value is T,
): T | undefined {
  const held = load()[key];
  return held !== undefined && valid(held) ? held : undefined;
}

/** Remembers an answer. A store that refuses is not worth failing a fold over. */
export function writeCalibration(key: string, value: unknown): void {
  try {
    installed?.write(JSON.stringify({ ...load(), [key]: value }, null, 2));
  } catch { /* quota, private mode, or no store: the next start measures. */ }
}
