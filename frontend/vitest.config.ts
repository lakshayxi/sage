import { defineConfig } from 'vitest/config'

// Kept separate from vite.config.ts on purpose: vitest bundles its own
// internal copy of Vite whose plugin types aren't quite identical to the
// project's own `vite` package, so merging `test:` into vite.config.ts (and
// importing defineConfig from 'vitest/config' there instead of 'vite') made
// `tsc -b` fail on vite.config.ts's `plugins: [react(), tailwindcss()]`
// with a type-incompatibility error -- harmless in practice, but it broke
// `npm run build`, which runs `tsc -b` first. This file isn't part of
// tsconfig.node.json's `include`, so it's never typechecked by `tsc -b` at
// all, and `vitest run` doesn't need it to be -- it transforms TypeScript
// itself.
export default defineConfig({
  test: {
    // Node environment (not jsdom) is enough for the current test suite
    // (frontend/src/api/chat.test.ts), which exercises fetch/ReadableStream
    // parsing logic, not DOM rendering.
    environment: 'node',
  },
})
