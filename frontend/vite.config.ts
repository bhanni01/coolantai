import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Dev-only: forward API paths to the local backend so the app can use
    // same-origin relative URLs (matching production, where FastAPI serves
    // the built frontend directly). SSE streams through this proxy fine.
    proxy: {
      '/run': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
