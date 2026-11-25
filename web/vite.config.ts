import { defineConfig } from "vitest/config";

// base "./" keeps every URL relative, so the built site works on GitHub Pages,
// any static host, or opened from a subfolder.
export default defineConfig({
  base: "./",
  build: { outDir: "dist", assetsInlineLimit: 0, chunkSizeWarningLimit: 2000 },
  test: { testTimeout: 60000 },
});
