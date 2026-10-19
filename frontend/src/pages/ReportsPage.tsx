import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, type ReportMeta } from '../api'

export default function ReportsPage() {
  const [reports, setReports] = useState<ReportMeta[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.getReports()
      .then(setReports)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-3xl px-6 py-8">
        <h1 className="mb-1 text-xl font-bold text-gray-900">研究报告记录</h1>
        <p className="mb-6 text-sm text-gray-500">所有已生成完成的研究报告。</p>

        {loading && <div className="text-sm text-gray-400">加载中…</div>}
        {!loading && reports.length === 0 && (
          <div className="rounded-xl border border-dashed border-gray-300 p-10 text-center text-sm text-gray-400">
            还没有已生成的研究报告
          </div>
        )}
        <div className="space-y-3">
          {reports.map((r) => (
            <Link
              key={r.id}
              to={`/reports/${r.id}`}
              className="block rounded-xl border border-gray-200 p-4 transition-colors hover:border-blue-300 hover:bg-blue-50/50"
            >
              <div className="text-sm font-medium text-gray-900">{r.title}</div>
              <div className="mt-1 text-xs text-gray-400">生成于 {r.generated_at}</div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  )
}
