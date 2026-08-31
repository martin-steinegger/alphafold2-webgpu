import { describe, expect, it } from "vitest";
import {
  AFWEBGPU_REQUIRED_LIMITS, identifyBrowser, preflightErrorMessage, runWebGpuPreflight,
  type PreflightAdapterLike, type PreflightEnvironment, type PreflightLimits,
} from "../web/webgpu-preflight.js";

const CHROME_LINUX = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
  + "Chrome/131.0.0.0 Safari/537.36";
const FIREFOX_LINUX = "Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0";
const SAFARI_MAC = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) "
  + "Version/18.1 Safari/605.1.15";

const ADEQUATE_LIMITS: PreflightLimits = { ...AFWEBGPU_REQUIRED_LIMITS };

function fakeAdapter(overrides: Partial<PreflightAdapterLike> = {}): PreflightAdapterLike {
  return {
    limits: ADEQUATE_LIMITS,
    info: { vendor: "nvidia", architecture: "turing", device: "", description: "NVIDIA GeForce RTX 2080" },
    requestDevice: () => Promise.resolve({ destroy: () => undefined }),
    ...overrides,
  };
}

function environment(overrides: Partial<PreflightEnvironment> = {}): PreflightEnvironment {
  return {
    gpu: { requestAdapter: () => Promise.resolve(fakeAdapter()) },
    secureContext: true, userAgent: CHROME_LINUX, timeoutMilliseconds: 1000, ...overrides,
  };
}

describe("identifyBrowser", () => {
  it("separates the engines the remedies branch on", () => {
    expect(identifyBrowser(CHROME_LINUX)).toMatchObject({ engine: "chromium", name: "Chrome", version: 131, platform: "linux" });
    expect(identifyBrowser(FIREFOX_LINUX)).toMatchObject({ engine: "gecko", name: "Firefox", version: 133, platform: "linux" });
    expect(identifyBrowser(SAFARI_MAC)).toMatchObject({ engine: "webkit", name: "Safari", version: 18, platform: "macos" });
  });

  it("does not report Chromium-based browsers as Chrome", () => {
    const edge = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      + "Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0";
    expect(identifyBrowser(edge)).toMatchObject({ engine: "chromium", name: "Edge", version: 131, platform: "windows" });
  });

  it("prefers the client-hint platform over the user-agent string", () => {
    expect(identifyBrowser(CHROME_LINUX, "Android").platform).toBe("android");
    expect(identifyBrowser("Mozilla/5.0 (Linux; Android 14) Chrome/131.0.0.0").platform).toBe("android");
  });
});

describe("runWebGpuPreflight", () => {
  it("reports a working adapter as ready", async () => {
    const preflight = await runWebGpuPreflight(environment());
    expect(preflight.status).toBe("ready");
    expect(preflight.usable).toBe(true);
    expect(preflight.headline).toContain("RTX 2080");
    expect(preflight.remedies).toEqual([]);
  });

  it("names Firefox on Linux when navigator.gpu is missing", async () => {
    const preflight = await runWebGpuPreflight(environment({ gpu: undefined, userAgent: FIREFOX_LINUX }));
    expect(preflight.status).toBe("unsupported");
    expect(preflight.usable).toBe(false);
    expect(preflight.headline).toBe("WebGPU is unavailable in Firefox on Linux");
    expect(preflight.remedies.join(" ")).toContain("dom.webgpu.enabled");
    expect(preflight.remedies.join(" ")).toContain("Vulkan");
  });

  it("blames the insecure origin before the browser", async () => {
    const preflight = await runWebGpuPreflight(environment({ gpu: undefined, secureContext: false }));
    expect(preflight.headline).toBe("WebGPU needs a secure origin");
    expect(preflight.remedies.join(" ")).toContain("https://");
  });

  it("retries without a power preference before declaring the adapter missing", async () => {
    const options: unknown[] = [];
    const preflight = await runWebGpuPreflight(environment({
      gpu: { requestAdapter: (value) => { options.push(value); return Promise.resolve(value === undefined ? fakeAdapter() : null); } },
    }));
    expect(options).toEqual([{ powerPreference: "high-performance" }, undefined]);
    expect(preflight.status).toBe("ready");
  });

  it("distinguishes a blocklisted GPU from a missing WebGPU implementation", async () => {
    const preflight = await runWebGpuPreflight(environment({
      gpu: {
        requestAdapter: (value) => Promise.resolve(value?.forceFallbackAdapter === true
          ? fakeAdapter({ isFallbackAdapter: true, info: { vendor: "google", description: "SwiftShader Device" } })
          : null),
      },
      userAgent: CHROME_LINUX,
    }));
    expect(preflight.status).toBe("blocked");
    expect(preflight.detail).toContain("blocklisted");
    expect(preflight.remedies.join(" ")).toContain("software fallback adapter");
    expect(preflight.remedies.join(" ")).toContain("chrome://gpu");
  });

  it("surfaces a thrown adapter request", async () => {
    const preflight = await runWebGpuPreflight(environment({
      gpu: { requestAdapter: () => Promise.reject(new Error("GPU process crashed")) },
    }));
    expect(preflight.status).toBe("blocked");
    expect(preflight.detail).toContain("GPU process crashed");
  });

  it("warns instead of failing when only a software renderer answers", async () => {
    const preflight = await runWebGpuPreflight(environment({
      gpu: { requestAdapter: () => Promise.resolve(fakeAdapter({ info: { vendor: "mesa", description: "llvmpipe (LLVM 17)" } })) },
    }));
    expect(preflight.status).toBe("warning");
    expect(preflight.usable).toBe(true);
    expect(preflight.adapter?.software).toBe(true);
  });

  it("rejects an adapter whose limits are below the model's bindings", async () => {
    const preflight = await runWebGpuPreflight(environment({
      gpu: {
        requestAdapter: () => Promise.resolve(fakeAdapter({
          limits: { ...ADEQUATE_LIMITS, maxStorageBufferBindingSize: 64 * 1024 * 1024, maxStorageBuffersPerShaderStage: 4 },
        })),
      },
    }));
    expect(preflight.status).toBe("insufficient");
    expect(preflight.usable).toBe(false);
    expect(preflight.shortfalls.map((shortfall) => shortfall.limit))
      .toEqual(["maxStorageBufferBindingSize", "maxStorageBuffersPerShaderStage"]);
    expect(preflight.detail).toContain("64 MiB");
    expect(preflight.detail).toContain("needs 128 MiB");
  });

  it("treats an unreported limit as unknown rather than as a shortfall", async () => {
    const preflight = await runWebGpuPreflight(environment({
      gpu: { requestAdapter: () => Promise.resolve(fakeAdapter({ limits: {} })) },
    }));
    expect(preflight.status).toBe("ready");
  });

  it("fails when the driver refuses a device even though the adapter exists", async () => {
    const preflight = await runWebGpuPreflight(environment({
      gpu: {
        requestAdapter: () => Promise.resolve(fakeAdapter({
          requestDevice: () => Promise.reject(new Error("Device creation failed: VK_ERROR_INITIALIZATION_FAILED")),
        })),
      },
    }));
    expect(preflight.status).toBe("blocked");
    expect(preflight.detail).toContain("VK_ERROR_INITIALIZATION_FAILED");
  });

  it("destroys the probe device so the page keeps no extra GPU handle", async () => {
    let destroyed = 0;
    await runWebGpuPreflight(environment({
      gpu: { requestAdapter: () => Promise.resolve(fakeAdapter({ requestDevice: () => Promise.resolve({ destroy: () => { destroyed += 1; } }) })) },
    }));
    expect(destroyed).toBe(1);
  });

  it("skips the device probe when asked", async () => {
    let requested = 0;
    const preflight = await runWebGpuPreflight(environment({
      probeDevice: false,
      gpu: { requestAdapter: () => Promise.resolve(fakeAdapter({ requestDevice: () => { requested += 1; return Promise.reject(new Error("unused")); } })) },
    }));
    expect(requested).toBe(0);
    expect(preflight.status).toBe("ready");
  });

  it("does not hang on an adapter request that never settles", async () => {
    const preflight = await runWebGpuPreflight(environment({
      gpu: { requestAdapter: () => new Promise(() => undefined) }, timeoutMilliseconds: 20,
    }));
    expect(preflight.status).toBe("blocked");
    expect(preflight.detail).toContain("did not respond within");
  });

  it("builds a one-line message for the run log", async () => {
    const preflight = await runWebGpuPreflight(environment({ gpu: undefined, userAgent: FIREFOX_LINUX }));
    const message = preflightErrorMessage(preflight);
    expect(message.startsWith(preflight.headline)).toBe(true);
    expect(message).toContain(preflight.remedies[0]!);
    expect(message).not.toContain("\n");
  });
});
