import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Dev-only proxy so the frontend can always call the API via relative paths
// (/chat, /conversations, /documents) -- matching how the built frontend is
// served from the same origin as the API in production (api/main.py mounts
// frontend/dist at "/"). The backend's permissive CORS would work too, but
// this avoids two different fetch-base-url code paths for dev vs prod.
//
// Test config lives in vitest.config.ts, not here: vitest bundles its own
// (slightly different) copy of Vite's plugin types internally, and
// importing `defineConfig` from `vitest/config` in this file to add a
// `test` block made `tsc -b` (part of `npm run build`) fail with a type
// conflict between the two -- see vitest.config.ts's docstring.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/chat': 'http://localhost:8000',
      '/conversations': 'http://localhost:8000',
      '/documents': 'http://localhost:8000',
    },
  },
})
