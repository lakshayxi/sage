import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Dev-only proxy so the frontend can always call the API via relative paths
// (/chat, /conversations, /documents) -- matching how the built frontend is
// served from the same origin as the API in production (api/main.py mounts
// frontend/dist at "/"). The backend's permissive CORS would work too, but
// this avoids two different fetch-base-url code paths for dev vs prod.
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
