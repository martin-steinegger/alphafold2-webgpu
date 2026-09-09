/**
 * Calibration remembered between visits.
 *
 * Choosing the projection kernel costs 662-764 ms of every cold start, of a
 * requestAlphaFoldDevice that takes 807-864, because it compiles the
 * candidates and times them. Measure once, keep the answer.
 *
 * One JSON document of key to answer, the same in a browser and on a host: a
 * store reads and writes the whole thing, so localStorage holds one entry
 * and node holds one file with identical contents.
 */

/** Bumped when a stored answer could no longer be trusted. */
const SCHEMA = 3;
const DOCUMENT = "afwebgpu.calibration";

export interface CalibrationStore {
  read(): string | undefined;
  write(document: string): void;
}

let installed: CalibrationStore | undefined;
let chosen = false;

/** Installs a store; undefined restores localStorage. */
export function setCalibrationStore(store: CalibrationStore | undefined): void {
  installed = store;
  chosen = store !== undefined;
}

function backing(): CalibrationStore | undefined {
  if (chosen) return installed;
  // Reading localStorage throws where site data is blocked.
  try {
    const web = (globalThis as { localStorage?: Storage }).localStorage;
    return web === undefined ? undefined : {
      read: () => web.getItem(DOCUMENT) ?? undefined,
      write: (document) => web.setItem(DOCUMENT, document),
    };
  } catch { return undefined; }
}

function load(): Record<string, unknown> {
  try {
    const raw = backing()?.read();
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
    backing()?.write(JSON.stringify({ ...load(), [key]: value }, null, 2));
  } catch { /* quota, private mode, or no store: the next start measures. */ }
}
