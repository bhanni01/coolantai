/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 'true' drives the UI from the offline mock instead of the live backend. */
  readonly VITE_USE_MOCK?: string
  /** Base URL of the FastAPI backend (default http://localhost:8000). */
  readonly VITE_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
