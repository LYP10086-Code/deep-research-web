import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import Markdown from 'react-markdown'
import { api, type Report } from '../api'

export default function ReportDetailPage() {
  const { reportId } = useParams()
  const [report, setReport] = useState<Report | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!reportId) return
    api.getReport(reportId)
      .then(setReport)
      .catch((e) => setError(String(e)))
  }, [reportId])

  if (error) return <div className="p-8 text-sm text-red-600">{error}</div>
  if (!report) return <div className="p-8 text-sm text-gray-400">加载中…</div>

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-3xl px-6 py-8">
        <div className="mb-4 flex items-center justify-between">
          <Link to="/reports" className="text-xs text-gray-500 hover:text-blue-600">
            ← 返回报告列表
          </Link>
          <Link
            to={`/chat/${report.session_id}`}
            className="text-xs text-gray-500 hover:text-blue-600"
          >
            查看来源会话 →
          </Link>
        </div>

        <h1 className="text-2xl font-bold text-gray-900">{report.title}</h1>
        <div className="mt-1 text-xs text-gray-400">生成于 {report.generated_at}</div>

        {report.brief && (
          <details className="mt-4 rounded-xl border border-gray-200 bg-gray-50 p-4" open>
            <summary className="cursor-pointer text-xs font-semibold text-gray-600">
              研究简报（协商定稿）
            </summary>
            <pre className="mt-2 whitespace-pre-wrap text-xs leading-relaxed text-gray-700">
              {report.brief}
            </pre>
          </details>
        )}

        <article className="report-md mt-6 text-sm text-gray-800">
          <Markdown>{report.content}</Markdown>
        </article>
      </div>
    </div>
  )
}
