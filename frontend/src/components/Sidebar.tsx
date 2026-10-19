import { useEffect } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import type { Session } from '../api'
import { useAppStore } from '../store'

const STATUS_TEXT: Record<Session['status'], string> = {
  running: '研究中',
  waiting_review: '待审阅',
  completed: '已完成',
  failed: '失败',
}

const STATUS_CLASS: Record<Session['status'], string> = {
  running: 'bg-blue-100 text-blue-700',
  waiting_review: 'bg-amber-100 text-amber-700',
  completed: 'bg-green-100 text-green-700',
  failed: 'bg-red-100 text-red-700',
}

export default function Sidebar() {
  const navigate = useNavigate()
  const { sessions, activeCount, refreshSessions } = useAppStore()

  useEffect(() => {
    refreshSessions()
    const timer = setInterval(refreshSessions, 5000)
    return () => clearInterval(timer)
  }, [refreshSessions])

  return (
    <aside className="flex h-full w-72 shrink-0 flex-col border-r border-gray-200 bg-gray-50">
      {/* 左上角：进行中的研究任务数 */}
      <div className="flex items-center justify-between px-4 pt-4">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-700">进行中的研究</span>
          <span
            className={`inline-flex h-6 min-w-6 items-center justify-center rounded-full px-2 text-xs font-semibold ${
              activeCount > 0 ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-500'
            }`}
          >
            {activeCount}
          </span>
        </div>
        <button
          onClick={() => navigate('/')}
          className="rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700"
        >
          + 新研究
        </button>
      </div>

      {/* 中部：历史研究对话记录 */}
      <div className="mt-4 flex-1 overflow-y-auto px-2">
        <div className="px-2 pb-2 text-xs font-medium tracking-wide text-gray-400">研究对话记录</div>
        {sessions.length === 0 && (
          <div className="px-2 py-6 text-center text-xs text-gray-400">暂无记录，点击「新研究」开始</div>
        )}
        {sessions.map((s) => (
          <NavLink
            key={s.id}
            to={`/chat/${s.id}`}
            className={({ isActive }) =>
              `mb-1 block rounded-lg px-3 py-2 transition-colors ${
                isActive ? 'bg-blue-50 ring-1 ring-blue-200' : 'hover:bg-gray-100'
              }`
            }
          >
            <div className="flex items-center justify-between gap-2">
              <span className="truncate text-sm text-gray-800">{s.title}</span>
              <span className={`shrink-0 rounded-full px-1.5 py-0.5 text-[10px] ${STATUS_CLASS[s.status]}`}>
                {STATUS_TEXT[s.status]}
              </span>
            </div>
            <div className="mt-0.5 text-[11px] text-gray-400">{s.updated_at}</div>
          </NavLink>
        ))}
      </div>

      {/* 左下角：研究报告记录入口 */}
      <div className="border-t border-gray-200 p-3">
        <NavLink
          to="/reports"
          className={({ isActive }) =>
            `flex items-center justify-between rounded-lg px-3 py-2 text-sm transition-colors ${
              isActive ? 'bg-blue-50 text-blue-700' : 'text-gray-700 hover:bg-gray-100'
            }`
          }
        >
          <span>研究报告记录</span>
          <span className="text-xs text-gray-400">查看全部 →</span>
        </NavLink>
      </div>
    </aside>
  )
}
