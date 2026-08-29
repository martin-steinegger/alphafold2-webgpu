import { defineConfig } from "@playwright/test";

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
      args: ["--enable-unsafe-webgpu", "--use-angle=vulkan", "--enable-features=Vulkan"],
    },
  },
});

