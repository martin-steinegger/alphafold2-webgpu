import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { resolve, sep } from "node:path";
import { defineConfig } from "vite";

const qualificationRootValue = process.env.AFWEBGPU_QUALIFICATION_ASSET_ROOT?.trim();
const qualificationRoot = qualificationRootValue === undefined || qualificationRootValue === ""
  ? undefined : resolve(qualificationRootValue);

export default defineConfig({
  base: "./",
  root: resolve(import.meta.dirname, "web"),
  server: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true,
    fs: { allow: [resolve(import.meta.dirname)] },
  },
  plugins: qualificationRoot === undefined ? [] : [{
    name: "qualification-assets",
    configureServer(server) {
      server.middlewares.use("/qualification-assets", (request, response, next) => {
        const relative = decodeURIComponent((request.url ?? "/").split("?", 1)[0]!).replace(/^\/+/, "");
        const file = resolve(qualificationRoot, relative);
        if (file !== qualificationRoot && !file.startsWith(`${qualificationRoot}${sep}`)) {
          response.statusCode = 403; response.end("Forbidden"); return;
        }
        void stat(file).then((metadata) => {
          if (!metadata.isFile()) { next(); return; }
          response.setHeader("Content-Length", metadata.size);
          if (file.endsWith(".json")) response.setHeader("Content-Type", "application/json");
          else response.setHeader("Content-Type", "application/octet-stream");
          createReadStream(file).pipe(response);
        }, next);
      });
    },
  }],
  build: {
    outDir: resolve(import.meta.dirname, "dist/web"),
    emptyOutDir: true,
  },
});
