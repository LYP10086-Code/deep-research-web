import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  api,
  NODE_LABELS,
  subscribeEvents,
  type ChatMessage,
  type Session,
} from '../api'

const STATUS_TEXT: Record<Session['status'], string> = {
  running: '研究进行中…',
  waiting_review: '等待你的审阅',
  completed: '研究已完成',
  failed: '研究失败',
}

export default function ChatPage() {
  const { sessionId } = useParams()
  const navigate = useNavigate()

  if (!sessionId) {
    return <NewChatForm onCreated={(id) => navigate(`/chat/${id}`)} />
  }
  return <ChatSession sessionId={sessionId} />
}

/** 新建研究会话 */
function NewChatForm({ onCreated }: { onCreated: (id: string) => void }) {
  const [question, setQuestion] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  const submit = async () => {
    if (!question.trim() || submitting) return
    setSubmitting(true)
    setError('')
    try {
      const s = await api.createSession(question.trim())
      onCreated(s.id)
    } catch (e) {
      setError(String(e))
      setSubmitting(false)
    }
  }

  return (
    <div className="flex h-full items-center justify-center">
      <div className="w-full max-w-xl px-6">
        <h1 className="mb-2 text-2xl font-bold text-gray-900">开始一项新研究</h1>
        <p className="mb-6 text-sm text-gray-500">
          描述你想研究的方向，系统会先与你协商研究简报，确认后再派出多智能体并行研究并生成报告。
        </p>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) submit()
          }}
          placeholder="例如：调研 2026 年固态电池产业化进展…（Ctrl+Enter 提交）"
          rows={5}
          className="w-full resize-none rounded-xl border border-gray-300 p-3 text-sm focus:border-blue-500 focus:outline-none"
        />
        {error && <div className="mt-2 text-xs text-red-600">{error}</div>}
        <button
          onClick={submit}
          disabled={!question.trim() || submitting}
          className="mt-4 rounded-xl bg-blue-600 px-5 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {submitting ? '创建中…' : '开始研究'}
        </button>
      </div>
    </div>
  )
}

/** 已有会话的聊天窗口 */
function ChatSession({ sessionId }: { sessionId: string }) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [status, setStatus] = useState<Session['status']>('running')
  const [currentNode, setCurrentNode] = useState<string | null>(null)
  const [reportId, setReportId] = useState<string | null>(null)
  const [feedback, setFeedback] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  // 加载历史 + 订阅 SSE
  useEffect(() => {
    let closed = false
    api.getMessages(sessionId).then(setMessages).catch(() => {})
    api.getSession(sessionId).then((s) => setStatus(s.status)).catch(() => {})

    const close = subscribeEvents(sessionId, {
      onStatus: (st) => {
        if (closed) return
        setStatus(st)
      },
      onNodeUpdate: (node) => {
        if (closed) return
        setCurrentNode(node)
      },
      onMessage: (msg) => {
        if (closed) return
        setMessages((prev) => [...prev, { ...msg, created_at: '' }])
      },
      onInterrupt: () => {
        if (closed) return
        setStatus('waiting_review')
        setCurrentNode(null)
      },
      onReportDone: (data) => {
        if (closed) return
        setReportId(data.report_id)
        setStatus('completed')
        setCurrentNode(null)
      },
      onError: () => {
        if (closed) return
        setStatus('failed')
        setCurrentNode(null)
      },
    })
    return () => {
      closed = true
      close()
    }
  }, [sessionId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, status])

  const sendFeedback = async () => {
    const text = feedback.trim()
    if (!text || status !== 'waiting_review') return
    setFeedback('')
    setStatus('running')
    try {
      await api.resume(sessionId, text)
    } catch (e) {
      setStatus('failed')
      alert(String(e))
    }
  }

  return (
    <div className="flex h-full flex-col">
      {/* 顶部状态栏 */}
      <div className="flex items-center justify-between border-b border-gray-200 px-6 py-3">
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-gray-900">{STATUS_TEXT[status]}</span>
          {status === 'running' && currentNode && (
            <span className="flex items-center gap-1.5 text-xs text-blue-600">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-blue-600" />
              {NODE_LABELS[currentNode] ?? currentNode}
            </span>
          )}
        </div>
        {reportId && (
          <Link
            to={`/reports/${reportId}`}
            className="rounded-lg bg-green-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-green-700"
          >
            查看研究报告 →
          </Link>
        )}
      </div>

      {/* 消息流 */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="mx-auto max-w-3xl space-y-4">
          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                  m.role === 'user'
                    ? 'rounded-br-sm bg-blue-600 text-white'
                    : 'rounded-bl-sm bg-gray-100 text-gray-900'
                }`}
              >
                {m.content}
              </div>
            </div>
          ))}

          {/* 人工审核卡片：按会话状态显示，刷新/重开页面后依然出现 */}
          {status === 'waiting_review' && (
            <div className="rounded-2xl border border-amber-300 bg-amber-50 p-4">
              <div className="mb-2 text-xs font-semibold tracking-wide text-amber-700">人工审核</div>
              <div className="text-sm text-gray-800">
                请查看上方的 AI 回复：回复 OK 继续；或直接输入你的补充说明 / 修改意见。
              </div>
              <div className="mt-3 flex gap-2">
                <input
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && sendFeedback()}
                  placeholder="输入 OK 确认，或直接输入修改意见…"
                  className="flex-1 rounded-lg border border-amber-300 bg-white px-3 py-2 text-sm focus:border-amber-500 focus:outline-none"
                />
                <button
                  onClick={sendFeedback}
                  disabled={!feedback.trim()}
                  className="rounded-lg bg-amber-600 px-4 py-2 text-sm font-medium text-white hover:bg-amber-700 disabled:opacity-50"
                >
                  发送
                </button>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </div>
    </div>
  )
}
