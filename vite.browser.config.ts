import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  root: resolve(import.meta.dirname, "web"),
  server: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true,
    fs: { allow: [resolve(import.meta.dirname)] },
  },
  build: {
    outDir: resolve(import.meta.dirname, "dist/web"),
    emptyOutDir: true,
  },
});
