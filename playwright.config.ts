import { defineConfig } from "@playwright/test";

const gpuArgs = process.platform === "darwin"
  ? ["--enable-gpu", "--enable-unsafe-webgpu"]
  : [
    "--enable-gpu", "--enable-unsafe-webgpu", "--use-vulkan=native",
    "--use-angle=vulkan", "--enable-features=Vulkan",
  ];

export default defineConfig({
  testDir: "./test/browser",
  timeout: 30_000,
  webServer: {
    command: "npm run dev",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: true,
    timeout: 30_000,
  },
  use: {
    baseURL: "http://127.0.0.1:4173",
    headless: true,
    launchOptions: {
      args: gpuArgs,
    },
  },
});
