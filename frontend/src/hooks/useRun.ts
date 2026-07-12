import type { RunController } from '../lib/types'
import { useLiveRun } from './useLiveRun'
import { useMockRun } from './useMockRun'

// Offline fallback for the founder demo: set VITE_USE_MOCK=true to drive the UI
// from the scripted mock instead of the live backend. Resolved once at module
// load, so exactly one hook is ever called (Rules of Hooks safe).
export const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true'

export const useRun: () => RunController = USE_MOCK ? useMockRun : useLiveRun
