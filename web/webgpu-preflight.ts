/**
 * WebGPU preflight for the AFWebGPU front end.
 *
 * A shallow `requestAdapter()` probe collapses several distinct failures into
 * one message. Each of them has a different remedy, so this module separates
 * them: the API being absent (browser or build without WebGPU), an insecure
 * origin, an adapter the driver refuses to hand out, a software rasterizer
 * standing in for the GPU, a device request the driver rejects, and an adapter
 * whose limits are below what AlphaFold inference binds. Diagnosis is reported
 * with the browser/platform combination responsible so the user is told what to
 * change rather than that "no adapter was found".
 */

const MIB = 1024 * 1024;

/** WebGPU limits every AlphaFold pipeline in this repository binds. */
export const AFWEBGPU_REQUIRED_LIMITS = {
  // The base tier AlphaFoldDeviceRequirements starts from; larger inputs raise it further.
  maxBufferSize: 256 * MIB,
  maxStorageBufferBindingSize: 128 * MIB,
  // The input embedder and IPA logits shaders bind eight storage buffers plus one uniform.
  maxStorageBuffersPerShaderStage: 8,
  // Widest workgroup in the repository is 32x8.
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupStorageSize: 16 * 1024,
  // IPA logits dispatch uses a 32768-wide grid.
  maxComputeWorkgroupsPerDimension: 32768,
} as const;

export type RequiredLimitName = keyof typeof AFWEBGPU_REQUIRED_LIMITS;

/** Structural subset of `GPUSupportedLimits` this check reads. */
export type PreflightLimits = { readonly [Name in RequiredLimitName]?: number };

export interface PreflightAdapterLike {
  readonly limits: PreflightLimits;
  readonly features?: { has(name: string): boolean } | undefined;
  readonly info?: {
    readonly vendor?: string; readonly architecture?: string;
    readonly device?: string; readonly description?: string;
    readonly isFallbackAdapter?: boolean;
  } | undefined;
  readonly isFallbackAdapter?: boolean | undefined;
  requestDevice(descriptor?: object): Promise<{ destroy(): void }>;
}

export interface PreflightGpuLike {
  requestAdapter(options?: {
    powerPreference?: string; forceFallbackAdapter?: boolean;
  }): Promise<PreflightAdapterLike | null>;
}

export type BrowserEngine = "chromium" | "gecko" | "webkit" | "unknown";
export type BrowserPlatform = "windows" | "macos" | "linux" | "android" | "ios" | "chromeos" | "unknown";

export interface BrowserIdentity {
  readonly engine: BrowserEngine;
  readonly name: string;
  readonly version: number | undefined;
  readonly platform: BrowserPlatform;
}

/**
 * `ready` runs at full speed; `warning` runs but slowly or unverified;
 * everything else must block the prediction.
 */
export type PreflightStatus = "ready" | "warning" | "unsupported" | "blocked" | "insufficient";

export interface PreflightShortfall {
  readonly limit: RequiredLimitName;
  readonly required: number;
  readonly available: number | undefined;
}

export interface PreflightAdapterSummary {
  readonly name: string;
  readonly vendor: string;
  readonly architecture: string;
  readonly fallback: boolean;
  readonly software: boolean;
}

export interface WebGpuPreflight {
  readonly status: PreflightStatus;
  /** True when a prediction may start. */
  readonly usable: boolean;
  readonly headline: string;
  readonly detail: string;
  readonly remedies: readonly string[];
  readonly adapter: PreflightAdapterSummary | undefined;
  readonly shortfalls: readonly PreflightShortfall[];
  readonly browser: BrowserIdentity;
}

export interface PreflightEnvironment {
  readonly gpu: PreflightGpuLike | undefined;
  readonly secureContext: boolean;
  readonly userAgent: string;
  readonly platformHint?: string | undefined;
  /** Requests and destroys a throwaway device to prove the driver accepts one. Defaults to true. */
  readonly probeDevice?: boolean | undefined;
  /** Guards against drivers whose adapter or device request never settles. Defaults to 15000. */
  readonly timeoutMilliseconds?: number | undefined;
}

const SOFTWARE_RENDERER = /llvmpipe|lavapipe|swiftshader|softpipe|software|warp|basic render|microsoft basic/i;

function version(userAgent: string, pattern: RegExp): number | undefined {
  const match = pattern.exec(userAgent);
  if (match?.[1] === undefined) return undefined;
  const value = Number.parseInt(match[1], 10);
  return Number.isFinite(value) ? value : undefined;
}

function detectPlatform(userAgent: string, platformHint: string | undefined): BrowserPlatform {
  const hint = `${platformHint ?? ""}`.toLowerCase();
  if (hint.includes("android")) return "android";
  if (hint.includes("win")) return "windows";
  if (hint.includes("mac")) return "macos";
  if (hint.includes("linux") || hint.includes("x11")) return "linux";
  if (/iphone|ipad|ipod|ios/.test(hint)) return "ios";
  if (/android/i.test(userAgent)) return "android";
  if (/cros/i.test(userAgent)) return "chromeos";
  if (/iphone|ipad|ipod/i.test(userAgent)) return "ios";
  if (/windows|win32|win64/i.test(userAgent)) return "windows";
  if (/mac os x|macintosh/i.test(userAgent)) return "macos";
  if (/linux|x11|freebsd/i.test(userAgent)) return "linux";
  return "unknown";
}

/** Names the browser engine and version so remedies can be specific. */
export function identifyBrowser(userAgent: string, platformHint?: string): BrowserIdentity {
  const platform = detectPlatform(userAgent, platformHint);
  if (/firefox\/|fxios\//i.test(userAgent)) {
    return { engine: "gecko", name: "Firefox",
      version: version(userAgent, /(?:firefox|fxios)\/(\d+)/i), platform };
  }
  if (/edg\//i.test(userAgent)) {
    return { engine: "chromium", name: "Edge", version: version(userAgent, /edg\/(\d+)/i), platform };
  }
  if (/opr\/|opera/i.test(userAgent)) {
    return { engine: "chromium", name: "Opera", version: version(userAgent, /opr\/(\d+)/i), platform };
  }
  if (/chrome\/|chromium\/|crios\//i.test(userAgent)) {
    return { engine: "chromium", name: /chromium\//i.test(userAgent) ? "Chromium" : "Chrome",
      version: version(userAgent, /(?:chrome|chromium|crios)\/(\d+)/i), platform };
  }
  if (/safari\//i.test(userAgent)) {
    return { engine: "webkit", name: "Safari", version: version(userAgent, /version\/(\d+)/i), platform };
  }
  return { engine: "unknown", name: "This browser", version: undefined, platform };
}

const CHROMIUM_ADVICE = "Chrome, Chromium, or Edge 121 or newer ships WebGPU on Windows, macOS, Linux, and ChromeOS.";

/** Explains an absent `navigator.gpu` for the browser that produced it. */
function missingApiDiagnosis(browser: BrowserIdentity): { detail: string; remedies: readonly string[] } {
  if (browser.engine === "gecko") {
    const linuxLike = browser.platform === "linux" || browser.platform === "unknown";
    return {
      detail: `${browser.name}${browser.version === undefined ? "" : ` ${browser.version}`} does not expose `
        + `navigator.gpu on ${platformLabel(browser.platform)}. Firefox enables WebGPU by default only on Windows `
        + "(Firefox 141 and newer); on Linux and macOS it is still behind a preference.",
      remedies: [
        "Open about:config, set dom.webgpu.enabled to true, and restart Firefox.",
        ...(linuxLike
          ? ["Firefox's Linux WebGPU backend needs Vulkan: confirm about:support reports Compositing "
            + "\"WebRender\" (not \"WebRender (Software)\") and that a Vulkan driver such as the NVIDIA "
            + "Vulkan ICD is installed (vulkaninfo should list your GPU).",
          "Distribution and snap/flatpak Firefox builds often ship WebGPU disabled outright; a Mozilla build "
            + "or Firefox Nightly is the reliable way to test it."]
          : []),
        CHROMIUM_ADVICE,
      ],
    };
  }
  if (browser.engine === "webkit") {
    return {
      detail: `${browser.name}${browser.version === undefined ? "" : ` ${browser.version}`} does not expose `
        + "navigator.gpu. Safari ships WebGPU from Safari 26; Safari 17 and 18 keep it behind a feature flag.",
      remedies: [
        "Update to Safari 26 or newer.",
        "On Safari 17/18: enable Settings ▸ Advanced ▸ Show features for web developers, then "
          + "Develop ▸ Feature Flags ▸ WebGPU.",
        CHROMIUM_ADVICE,
      ],
    };
  }
  if (browser.engine === "chromium") {
    return {
      detail: `${browser.name}${browser.version === undefined ? "" : ` ${browser.version}`} does not expose `
        + "navigator.gpu. WebGPU shipped in Chromium 113 on Windows/macOS/ChromeOS and in Chromium 121 on Linux, "
        + "and it stays disabled when the GPU process falls back to software rendering.",
      remedies: [
        "Update the browser to version 121 or newer.",
        "Open chrome://gpu and read the WebGPU row; it names the blocklist entry or driver problem responsible.",
        "Hardware acceleration must be on: chrome://settings/system ▸ \"Use graphics acceleration when available\".",
      ],
    };
  }
  return {
    detail: "This browser does not expose navigator.gpu, so WebGPU is unavailable.",
    remedies: [CHROMIUM_ADVICE, "Safari 26 and Firefox 141 on Windows also ship WebGPU."],
  };
}

/** Explains an adapter request that returned null or threw. */
function adapterFailureRemedies(browser: BrowserIdentity, fallbackAvailable: boolean): readonly string[] {
  const shared = fallbackAvailable
    ? ["Only a software fallback adapter is available, so the browser sees WebGPU but not your GPU. "
      + "That is a driver or blocklist problem, not a missing browser feature."]
    : [];
  if (browser.engine === "chromium") {
    return [...shared,
      "Open chrome://gpu; the \"WebGPU\" and \"Vulkan\" rows state why the GPU was rejected.",
      ...(browser.platform === "linux"
        ? ["Chromium on Linux drives WebGPU through Vulkan: install your GPU's Vulkan driver "
          + "(NVIDIA: libnvidia-gl / nvidia-driver-libs; Mesa: mesa-vulkan-drivers) and check `vulkaninfo`.",
        "If the driver is blocklisted, launching with --enable-features=Vulkan "
          + "--enable-unsafe-webgpu confirms whether the blocklist is the cause."]
        : ["Update the GPU driver, then restart the browser."]),
      "Headless, remote-desktop, and virtual-machine sessions often expose no hardware adapter at all."];
  }
  if (browser.engine === "gecko") {
    return [...shared,
      "In about:support, Compositing must read \"WebRender\"; \"WebRender (Software)\" means Firefox never "
        + "reaches the GPU and WebGPU cannot start.",
      "Install a working Vulkan driver for the GPU and restart Firefox.",
      CHROMIUM_ADVICE];
  }
  return [...shared, "Update the GPU driver and restart the browser.", CHROMIUM_ADVICE];
}

function platformLabel(platform: BrowserPlatform): string {
  const labels: Readonly<Record<BrowserPlatform, string>> = {
    windows: "Windows", macos: "macOS", linux: "Linux", android: "Android",
    ios: "iOS", chromeos: "ChromeOS", unknown: "this platform",
  };
  return labels[platform];
}

export function formatLimit(name: RequiredLimitName, value: number | undefined): string {
  if (value === undefined) return "not reported";
  return name.endsWith("Size") && value >= MIB ? `${(value / MIB).toFixed(0)} MiB` : value.toLocaleString();
}

function summarizeAdapter(adapter: PreflightAdapterLike): PreflightAdapterSummary {
  const info = adapter.info;
  const vendor = info?.vendor ?? "";
  const architecture = info?.architecture ?? "";
  // Chromium reports only the vendor unless the page is cross-origin isolated.
  const named = info?.description || info?.device || [vendor, architecture].filter((part) => part !== "").join(" ");
  const name = named === "" ? "unnamed WebGPU adapter" : named === vendor ? `${vendor} GPU` : named;
  const fallback = adapter.isFallbackAdapter === true || info?.isFallbackAdapter === true;
  const identity = `${vendor} ${architecture} ${info?.device ?? ""} ${info?.description ?? ""}`;
  return { name, vendor, architecture, fallback, software: fallback || SOFTWARE_RENDERER.test(identity) };
}

function missingLimits(limits: PreflightLimits): readonly PreflightShortfall[] {
  const names = Object.keys(AFWEBGPU_REQUIRED_LIMITS) as RequiredLimitName[];
  return names.flatMap((limit) => {
    const required = AFWEBGPU_REQUIRED_LIMITS[limit];
    const available = limits[limit];
    // An unreported limit is not evidence of a shortfall; only a reported, smaller one is.
    return available !== undefined && available < required ? [{ limit, required, available }] : [];
  });
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message || error.name : String(error);
}

async function withTimeout<T>(work: Promise<T>, milliseconds: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      work,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} did not respond within ${milliseconds / 1000} s`)),
          milliseconds);
      }),
    ]);
  } finally { if (timer !== undefined) clearTimeout(timer); }
}

/** Reads the preflight inputs out of the current page. */
export function browserPreflightEnvironment(): PreflightEnvironment {
  const agent = navigator as Navigator & { readonly userAgentData?: { readonly platform?: string } };
  return {
    gpu: (navigator as Navigator & { readonly gpu?: PreflightGpuLike }).gpu,
    secureContext: typeof isSecureContext === "boolean" ? isSecureContext : true,
    userAgent: navigator.userAgent,
    platformHint: agent.userAgentData?.platform ?? navigator.platform,
  };
}

/**
 * Classifies WebGPU availability without starting a prediction.
 *
 * The device probe is what separates "the adapter exists" from "the driver will
 * actually give this page a device", which is the failure Linux users hit most.
 */
export async function runWebGpuPreflight(environment: PreflightEnvironment): Promise<WebGpuPreflight> {
  const browser = identifyBrowser(environment.userAgent, environment.platformHint);
  const base = { adapter: undefined, shortfalls: [], browser } as const;
  const timeout = environment.timeoutMilliseconds ?? 15_000;

  if (environment.gpu === undefined) {
    if (!environment.secureContext) {
      return { ...base, status: "unsupported", usable: false,
        headline: "WebGPU needs a secure origin",
        detail: "This page is not a secure context, and browsers only expose navigator.gpu over HTTPS or on "
          + "localhost.",
        remedies: ["Load the site over https://, or serve it from http://localhost for local development."] };
    }
    const diagnosis = missingApiDiagnosis(browser);
    return { ...base, status: "unsupported", usable: false,
      headline: `WebGPU is unavailable in ${browser.name} on ${platformLabel(browser.platform)}`,
      detail: diagnosis.detail, remedies: diagnosis.remedies };
  }

  const gpu = environment.gpu;
  let adapter: PreflightAdapterLike | null = null;
  let requestError: unknown;
  try {
    adapter = await withTimeout(gpu.requestAdapter({ powerPreference: "high-performance" }), timeout,
      "requestAdapter()");
    // A high-performance preference can be refused on its own; retry unconstrained before giving up.
    adapter ??= await withTimeout(gpu.requestAdapter(), timeout, "requestAdapter()");
  } catch (error) { requestError = error; }

  if (adapter === null || adapter === undefined) {
    let fallbackAvailable = false;
    try {
      fallbackAvailable = await withTimeout(gpu.requestAdapter({ forceFallbackAdapter: true }), timeout,
        "requestAdapter()") !== null;
    } catch { fallbackAvailable = false; }
    return { ...base, status: "blocked", usable: false,
      headline: "No WebGPU adapter was granted",
      detail: `${browser.name} exposes WebGPU, but ${requestError === undefined
        ? "requesting an adapter returned none"
        : `requesting an adapter failed: ${errorMessage(requestError)}`}. The GPU is either blocklisted, driven by `
        + "a driver the browser rejects, or unreachable from this session.",
      remedies: adapterFailureRemedies(browser, fallbackAvailable) };
  }

  const summary = summarizeAdapter(adapter);
  const shortfalls = missingLimits(adapter.limits);
  if (shortfalls.length > 0) {
    return { ...base, adapter: summary, shortfalls, status: "insufficient", usable: false,
      headline: `${summary.name} is below AlphaFold's WebGPU limits`,
      detail: `This adapter reports ${shortfalls.map((shortfall) =>
        `${shortfall.limit} ${formatLimit(shortfall.limit, shortfall.available)} `
        + `(needs ${formatLimit(shortfall.limit, shortfall.required)})`).join(", ")}.`,
      remedies: [
        "Update the GPU driver; reported limits come straight from it.",
        "Mobile and integrated GPUs in compatibility mode expose the smallest tier and cannot run this model.",
      ] };
  }

  if (environment.probeDevice !== false) {
    try {
      const device = await withTimeout(adapter.requestDevice(), timeout, "requestDevice()");
      device.destroy();
    } catch (error) {
      return { ...base, adapter: summary, status: "blocked", usable: false,
        headline: `${summary.name} refused a WebGPU device`,
        detail: `The adapter was found, but requesting a device failed: ${errorMessage(error)}.`,
        remedies: adapterFailureRemedies(browser, false) };
    }
  }

  if (summary.software) {
    return { ...base, adapter: summary, status: "warning", usable: true,
      headline: `WebGPU is running on a software renderer (${summary.name})`,
      detail: "The browser fell back to CPU rasterization instead of your GPU. AlphaFold inference will run, but "
        + "orders of magnitude slower, and larger inputs will exhaust memory.",
      remedies: adapterFailureRemedies(browser, true) };
  }

  return { ...base, adapter: summary, status: "ready", usable: true,
    headline: `WebGPU ready · ${summary.name}`,
    detail: `${formatLimit("maxStorageBufferBindingSize", adapter.limits.maxStorageBufferBindingSize)} storage `
      + `binding, ${formatLimit("maxBufferSize", adapter.limits.maxBufferSize)} maximum buffer.`,
    remedies: [] };
}

/** Single-line message for a failed preflight, used when a prediction is blocked. */
export function preflightErrorMessage(preflight: WebGpuPreflight): string {
  return [preflight.headline, preflight.detail, ...preflight.remedies.slice(0, 1)].join(" ");
}
