import { describe, expect, it, afterEach } from "vitest";
import {
  calibrationKey, readCalibration, setCalibrationStore, writeCalibration,
} from "../src/runtime/calibration-store.js";

function memoryStore(): { document: string | undefined } & {
  read(): string | undefined; write(value: string): void;
} {
  const held = { document: undefined as string | undefined };
  return {
    get document() { return held.document; },
    set document(value) { held.document = value; },
    read: () => held.document,
    write: (value) => { held.document = value; },
  };
}

const adapter = (description: string): GPUAdapter => ({
  info: { vendor: "nvidia", architecture: "blackwell", device: "rtx", description },
} as unknown as GPUAdapter);

const device = (features: string[]): GPUDevice =>
  ({ features: new Set(features) } as unknown as GPUDevice);

const SHAPE = { M: 16, N: 16, K: 16, componentType: "f16" };

afterEach(() => { setCalibrationStore(undefined); });

describe("the calibration store", () => {
  it("gives an answer back", () => {
    setCalibrationStore(memoryStore());
    const key = calibrationKey(adapter("driver 1"), device(["shader-f16"]), [SHAPE], "");
    writeCalibration(key, { precision: "matrix", inner: 8 });
    expect(readCalibration(key, (v): v is { precision: string } => true))
      .toEqual({ precision: "matrix", inner: 8 });
  });

  it("keys on the driver, the features and the matrix shapes, so a changed device asks again", () => {
    const base = calibrationKey(adapter("driver 1"), device(["shader-f16"]), [SHAPE], "");
    expect(calibrationKey(adapter("driver 2"), device(["shader-f16"]), [SHAPE], "")).not.toBe(base);
    expect(calibrationKey(adapter("driver 1"), device(["shader-f16", "subgroups"]), [SHAPE], ""))
      .not.toBe(base);
    expect(calibrationKey(adapter("driver 1"), device(["shader-f16"]), [], "")).not.toBe(base);
    // ...and is stable for the same one, or nothing would ever be reused.
    expect(calibrationKey(adapter("driver 1"), device(["shader-f16"]), [SHAPE], "")).toBe(base);
  });

  it("keeps two models apart when a caller says they differ", () => {
    const shared = calibrationKey(adapter("d"), device(["shader-f16"]), [SHAPE], "");
    const af3 = calibrationKey(adapter("d"), device(["shader-f16"]), [SHAPE], "af3-128");
    const wide = calibrationKey(adapter("d"), device(["shader-f16"]), [SHAPE], "openfold-384");
    expect(new Set([shared, af3, wide]).size).toBe(3);
  });

  it("refuses a stored value that does not pass its guard", () => {
    const store = memoryStore();
    setCalibrationStore(store);
    const key = "k";
    store.document = JSON.stringify({ [key]: { precision: "nonsense" } });
    const valid = (v: unknown): v is { precision: "matrix" } =>
      (v as { precision?: string })?.precision === "matrix";
    expect(readCalibration(key, valid)).toBeUndefined();
  });

  it("survives a store that throws, which is a browser with site data blocked", () => {
    setCalibrationStore({
      read: () => { throw new Error("blocked"); },
      write: () => { throw new Error("blocked"); },
    });
    expect(() => writeCalibration("k", { a: 1 })).not.toThrow();
    expect(readCalibration("k", (v): v is object => true)).toBeUndefined();
  });

  it("ignores a value that is not JSON at all", () => {
    const store = memoryStore();
    setCalibrationStore(store);
    store.document = "{not json";
    expect(readCalibration("k", (v): v is object => true)).toBeUndefined();
  });
});
