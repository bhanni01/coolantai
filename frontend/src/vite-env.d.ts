/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 'true' drives the UI from the offline mock instead of the live backend. */
  readonly VITE_USE_MOCK?: string
  /** Base URL of the FastAPI backend. Defaults to '' (same origin; the Vite
   *  dev proxy forwards API paths to localhost:8000 during development). */
  readonly VITE_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
