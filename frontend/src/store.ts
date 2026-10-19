import { create } from 'zustand'
import { api, type Session } from './api'

interface AppState {
  sessions: Session[]
  activeCount: number
  refreshSessions: () => Promise<void>
}

export const useAppStore = create<AppState>((set) => ({
  sessions: [],
  activeCount: 0,
  refreshSessions: async () => {
    try {
      const [sessions, stats] = await Promise.all([api.getSessions(), api.getActive()])
      set({ sessions, activeCount: stats.active })
    } catch {
      // 服务未启动时静默，保留旧数据
    }
  },
}))
